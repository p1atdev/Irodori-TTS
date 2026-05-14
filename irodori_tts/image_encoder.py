from collections.abc import Callable
from os import PathLike
from typing import NamedTuple

import torch
import torch.nn as nn
from PIL import Image as PILImage
from timm import create_model
from timm import data as timm_data

from .projector import ProjectorConfig, build_projector, resolve_projector_config


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


class CharacterImageEncoderOutput(NamedTuple):
    generation_state: torch.Tensor
    duration_state: torch.Tensor


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
        prepend_global_summary_token: Whether to prepend a global summary
            token computed as the mean of projected character tokens.
        split_duration_state: Whether to double the projector output dimension
            and split it into generation and duration states.
    """

    def __init__(
        self,
        timm_model_id: str,
        output_dim: int,
        use_all_patches: bool = True,
        image_size: int = 448,
        pretrained: bool = True,
        projector_config: ProjectorConfig | dict | None = None,
        hidden_state_index: int | None = None,
        prepend_global_summary_token: bool = False,
        split_duration_state: bool = False,
    ):
        super().__init__()

        if projector_config is None or isinstance(projector_config, dict):
            projector_config = resolve_projector_config(projector_config)

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
        self._hidden_state_index = hidden_state_index
        self.prepend_global_summary_token = bool(prepend_global_summary_token)
        self.split_duration_state = bool(split_duration_state)

        if hidden_state_index is not None and not hasattr(backbone, "forward_intermediates"):
            raise ValueError(
                f"Backbone {timm_model_id!r} does not support forward_intermediates; "
                "hidden_state_index cannot be used."
            )

        # Detect backbone output feature dimension via a probe forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            out = self._backbone_forward(dummy)
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

        projector_output_dim = output_dim * 2 if self.split_duration_state else output_dim
        self.proj = build_projector(projector_config, backbone_dim, projector_output_dim)
        # Post-projection norm for stable training (mirrors caption_norm pattern).
        self.norm = _RMSNorm(projector_output_dim)

    def _backbone_forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run the backbone, optionally returning an intermediate hidden state."""
        if self._hidden_state_index is None:
            return self.backbone(images)
        intermediates = self.backbone.forward_intermediates(
            images,
            intermediates_only=True,
        )  # type: ignore[attr-defined]
        hidden_state = intermediates[self._hidden_state_index]
        # maybe [batch_size, num_features, N, N]
        batch_size, dim, h, w = hidden_state.size()

        hidden_state = hidden_state.permute(0, 2, 3, 1).reshape(batch_size, h * w, dim)

        return hidden_state

    def forward(self, images: torch.Tensor) -> torch.Tensor | CharacterImageEncoderOutput:
        """
        Args:
            images: Preprocessed image batch of shape ``(B, 3, H, W)``.

        Returns:
            Feature tensor of shape ``(B, N_tokens, output_dim)``.
        """
        features = self._backbone_forward(images)

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

        projected = self.norm(self.proj(features))  # (B, N, output_dim)
        if self.prepend_global_summary_token:
            summary_token = projected.mean(dim=1, keepdim=True)
            projected = torch.cat([summary_token, projected], dim=1)
        if self.split_duration_state:
            generation_state, duration_state = projected.chunk(2, dim=-1)
            return CharacterImageEncoderOutput(
                generation_state=generation_state,
                duration_state=duration_state,
            )
        return projected
