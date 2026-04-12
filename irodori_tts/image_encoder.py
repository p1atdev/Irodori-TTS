from __future__ import annotations

from collections.abc import Callable
from os import PathLike

import torch
import torch.nn as nn
from PIL import Image as PILImage
from timm import create_model
from timm import data as timm_data

from .config import CharacterProjectorConfig


def load_character_image(
    path: str | PathLike[str],
    *,
    background: tuple[int, int, int] = (255, 255, 255),
) -> PILImage.Image:
    """Open an image and return it as RGB, compositing transparent pixels onto a solid background."""
    img = PILImage.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        canvas = PILImage.new("RGB", img.size, background)
        canvas.paste(img, mask=img.split()[-1])
        return canvas
    return img.convert("RGB")

_PROJECTOR_LINEAR_INIT_STD = 0.02


def _init_projector_linear(module: nn.Linear) -> None:
    nn.init.normal_(module.weight, mean=0.0, std=_PROJECTOR_LINEAR_INIT_STD)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class _RMSNorm(nn.Module):
    """Lightweight RMSNorm to avoid circular imports with model.py."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms * self.weight).to(x_dtype)


class _MLPBlock(nn.Module):
    """A single MLP block: Linear -> SiLU -> Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_projector_linear(self.fc1)
        _init_projector_linear(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MLPProjector(nn.Module):
    """Projector made of stacked MLP blocks.

    Notes:
        ``num_layers`` counts MLP blocks, not Linear layers.
        Each block is ``Linear -> SiLU -> Linear``.
        So ``num_layers=1`` means a standard single MLP block.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        blocks: list[nn.Module] = []
        block_in_dim = in_dim
        for layer_idx in range(num_layers):
            block_out_dim = out_dim if layer_idx == num_layers - 1 else hidden_dim
            blocks.append(_MLPBlock(block_in_dim, hidden_dim, block_out_dim))
            block_in_dim = block_out_dim
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


def build_projector(
    config: CharacterProjectorConfig,
    backbone_dim: int,
    output_dim: int,
) -> nn.Module:
    """Factory for building a projector module from config."""
    hidden_dim = config.hidden_dim if config.hidden_dim is not None else backbone_dim

    if config.type == "mlp":
        return MLPProjector(
            in_dim=backbone_dim,
            hidden_dim=hidden_dim,
            out_dim=output_dim,
            num_layers=config.num_layers,
        )
    raise ValueError(f"Unknown projector type: {config.type!r}")


def build_character_transform(timm_model_id: str, image_size: int) -> Callable:
    """
    Build a torchvision transform for preprocessing character reference images.
    Uses the timm model's own recommended preprocessing config.
    The returned transform is a torchvision Compose and is safe to pickle for
    DataLoader multiprocessing workers.
    """
    m = create_model(timm_model_id, pretrained=False)
    data_cfg = timm_data.resolve_model_data_config(m)
    data_cfg["input_size"] = (3, image_size, image_size)
    transform = timm_data.create_transform(**data_cfg, is_training=False)
    del m
    return transform


class CharacterImageEncoder(nn.Module):
    """
    Character reference image encoder using a timm backbone.

    Encodes a batch of preprocessed images into a sequence of patch-level
    feature vectors projected to ``output_dim``.

    Args:
        timm_model_id: timm model identifier, e.g.
            ``"hf_hub:SmilingWolf/wd-eva02-large-tagger-v3"``.
        output_dim: Projected output dimension fed into JointAttention.
        use_all_patches: If True, use all spatial patch tokens.
            If False, use only the CLS token (index 0).
        image_size: Image resize target (H = W).
        pretrained: Whether to load pretrained backbone weights.
        projector_config: Configuration for the projector module.
    """

    def __init__(
        self,
        timm_model_id: str,
        output_dim: int,
        use_all_patches: bool = True,
        image_size: int = 448,
        pretrained: bool = True,
        projector_config: CharacterProjectorConfig | None = None,
    ):
        super().__init__()

        if projector_config is None:
            projector_config = CharacterProjectorConfig()

        # Load backbone without classification head, retaining all spatial tokens.
        backbone = create_model(
            timm_model_id,
            pretrained=pretrained,
            global_pool="",
        )
        # num_classes=0 にするとヘッドが削除されてから load_state_dict されてエラーになってしまうため、
        # 読み込んでからヘッドを消す
        reset_classifier = getattr(backbone, "reset_classifier", None)
        if callable(reset_classifier):
            reset_classifier(0)  # remove classifier head later
        self.backbone = backbone
        self.use_all_patches = use_all_patches
        self._image_size = image_size

        # Detect backbone output feature dimension via a probe forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            out = self.backbone(dummy)
            if out.ndim == 2:
                backbone_dim = int(out.shape[-1])
                self._out_format = "pooled"
            elif out.ndim == 3:
                backbone_dim = int(out.shape[-1])
                self._out_format = "seq"
            elif out.ndim == 4:
                backbone_dim = int(out.shape[1])
                self._out_format = "spatial"
            else:
                raise RuntimeError(
                    f"Unexpected backbone output ndim={out.ndim} for {timm_model_id!r}"
                )

        self.proj = build_projector(projector_config, backbone_dim, output_dim)
        # Post-projection norm for stable training (mirrors caption_norm pattern).
        self.norm = _RMSNorm(output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Preprocessed image batch of shape ``(B, 3, H, W)``.

        Returns:
            Feature tensor of shape ``(B, N_tokens, output_dim)``.
        """
        features = self.backbone(images)

        if self._out_format == "pooled":
            # (B, D) → (B, 1, D)
            features = features.unsqueeze(1)
        elif self._out_format == "spatial":
            # (B, C, H, W) → (B, H*W, C)
            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
            if not self.use_all_patches:
                features = features[:, :1, :]
        else:
            # seq: (B, N, D)
            if not self.use_all_patches:
                features = features[:, :1, :]

        projected = self.proj(features)  # (B, N, output_dim)
        return self.norm(projected)
