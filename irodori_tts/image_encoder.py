import re
from collections.abc import Callable, Mapping
from os import PathLike
from typing import NamedTuple

import torch
import torch.nn as nn
from PIL import Image as PILImage
from timm import create_model
from timm import data as timm_data

from .projector import ProjectorConfig, build_projector, resolve_projector_config

_CCIP_PREFIX = "ccip:"
_CCIP_REPO_ID = "deepghs/ccip"
_CCIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CCIP_STD = (0.26862954, 0.26130258, 0.27577711)
_CCIP_CAFormer_PREFIX = "module._orig_mod.feature.backbone.caformer."
_CCIP_STAGE_BLOCK_PATTERN = re.compile(r"^stages\.(\d+)\.(\d+)\.")


class _CCIPModelSpec(NamedTuple):
    name: str
    timm_model_id: str
    checkpoint_filename: str


_CCIP_MODEL_SPECS = {
    "ccip-caformer_b36-24": _CCIPModelSpec(
        name="ccip-caformer_b36-24",
        timm_model_id="caformer_b36",
        checkpoint_filename="ccip-caformer_b36-24.ckpt",
    ),
    "ccip-caformer-24-randaug-pruned": _CCIPModelSpec(
        name="ccip-caformer-24-randaug-pruned",
        timm_model_id="caformer_s36",
        checkpoint_filename="ccip-caformer-24-randaug-pruned.ckpt",
    ),
}


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


def _validate_dropout_rate(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _build_timm_dropout_kwargs(
    *,
    drop_rate: float,
    attn_drop_rate: float,
    drop_path_rate: float,
) -> dict[str, float]:
    rates = {
        "drop_rate": _validate_dropout_rate("drop_rate", drop_rate),
        "attn_drop_rate": _validate_dropout_rate("attn_drop_rate", attn_drop_rate),
        "drop_path_rate": _validate_dropout_rate("drop_path_rate", drop_path_rate),
    }
    return {name: value for name, value in rates.items() if value > 0.0}


def _resolve_ccip_model_spec(model_id: str) -> _CCIPModelSpec | None:
    if not model_id.startswith(_CCIP_PREFIX):
        return None

    name = model_id[len(_CCIP_PREFIX) :].strip()
    if name.endswith(".ckpt"):
        name = name[: -len(".ckpt")]
    try:
        return _CCIP_MODEL_SPECS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_CCIP_MODEL_SPECS))
        raise ValueError(
            f"Unknown CCIP character encoder {model_id!r}. Supported models: {known}"
        ) from exc


def _ccip_stage_replacement(match: re.Match[str]) -> str:
    return f"stages.{match.group(1)}.blocks.{match.group(2)}."


def _convert_ccip_caformer_state_dict(
    checkpoint_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}

    for key, value in checkpoint_state.items():
        if not key.startswith(_CCIP_CAFormer_PREFIX):
            continue

        new_key = key[len(_CCIP_CAFormer_PREFIX) :]
        if new_key.startswith("downsample_layers.0.conv."):
            new_key = new_key.replace("downsample_layers.0.conv.", "stem.conv.", 1)
        elif new_key.startswith("downsample_layers.0.post_norm."):
            new_key = new_key.replace("downsample_layers.0.post_norm.", "stem.norm.", 1)
        else:
            for stage_index in (1, 2, 3):
                if new_key.startswith(f"downsample_layers.{stage_index}.pre_norm."):
                    new_key = new_key.replace(
                        f"downsample_layers.{stage_index}.pre_norm.",
                        f"stages.{stage_index}.downsample.norm.",
                        1,
                    )
                if new_key.startswith(f"downsample_layers.{stage_index}.conv."):
                    new_key = new_key.replace(
                        f"downsample_layers.{stage_index}.conv.",
                        f"stages.{stage_index}.downsample.conv.",
                        1,
                    )

        new_key = _CCIP_STAGE_BLOCK_PATTERN.sub(_ccip_stage_replacement, new_key)
        if new_key.startswith("norm."):
            new_key = new_key.replace("norm.", "head.norm.", 1)
        elif new_key.startswith("head.fc1."):
            new_key = new_key.replace("head.fc1.", "head.fc.fc1.", 1)
        elif new_key.startswith("head.norm."):
            new_key = new_key.replace("head.norm.", "head.fc.norm.", 1)
        elif new_key.startswith("head.fc2."):
            new_key = new_key.replace("head.fc2.", "head.fc.fc2.", 1)

        target_value = target_state.get(new_key)
        if (
            target_value is not None
            and value.ndim == 2
            and target_value.ndim == 4
            and target_value.shape[-2:] == (1, 1)
        ):
            value = value[:, :, None, None]

        converted[new_key] = value

    missing = [key for key in target_state if key not in converted]
    unexpected = [key for key in converted if key not in target_state]
    mismatched = [
        (key, tuple(value.shape), tuple(target_state[key].shape))
        for key, value in converted.items()
        if key in target_state and tuple(value.shape) != tuple(target_state[key].shape)
    ]
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:8]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:8]}")
        if mismatched:
            details.append(f"mismatched={mismatched[:8]}")
        raise RuntimeError("Failed to convert CCIP CAFormer checkpoint: " + "; ".join(details))

    return converted


def _load_ccip_caformer_state_dict(
    backbone: nn.Module,
    checkpoint_state: Mapping[str, torch.Tensor],
) -> None:
    converted = _convert_ccip_caformer_state_dict(checkpoint_state, backbone.state_dict())
    backbone.load_state_dict(converted, strict=True)


def _create_ccip_backbone(
    spec: _CCIPModelSpec,
    *,
    pretrained: bool,
    timm_dropout_kwargs: dict[str, float],
) -> nn.Module:
    backbone = create_model(
        spec.timm_model_id,
        pretrained=False,
        num_classes=2,
        global_pool="",
        **timm_dropout_kwargs,
    )
    if pretrained:
        from huggingface_hub import hf_hub_download

        checkpoint_path = hf_hub_download(_CCIP_REPO_ID, spec.checkpoint_filename)
        checkpoint_state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint_state, Mapping):
            raise RuntimeError(
                f"Expected CCIP checkpoint to contain a state_dict mapping, got {type(checkpoint_state)!r}"
            )
        _load_ccip_caformer_state_dict(backbone, checkpoint_state)
    return backbone


def _create_character_backbone(
    timm_model_id: str,
    *,
    pretrained: bool,
    timm_dropout_kwargs: dict[str, float],
) -> nn.Module:
    ccip_spec = _resolve_ccip_model_spec(timm_model_id)
    if ccip_spec is not None:
        return _create_ccip_backbone(
            ccip_spec,
            pretrained=pretrained,
            timm_dropout_kwargs=timm_dropout_kwargs,
        )

    return create_model(
        timm_model_id,
        pretrained=pretrained,
        global_pool="",
        **timm_dropout_kwargs,
    )


def build_character_transform(timm_model_id: str, image_size: int) -> Callable:
    """
    Build a torchvision transform for preprocessing character reference images.
    Uses the timm model's own recommended preprocessing config.
    The returned transform is a torchvision Compose and is safe to pickle for
    DataLoader multiprocessing workers.
    """
    if _resolve_ccip_model_spec(timm_model_id) is not None:
        return timm_data.create_transform(
            input_size=(3, image_size, image_size),
            is_training=False,
            interpolation="bilinear",
            mean=_CCIP_MEAN,
            std=_CCIP_STD,
            crop_pct=1.0,
        )

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
        drop_rate: Dropout probability passed to timm backbones that support it.
        attn_drop_rate: Attention dropout probability passed to timm backbones that support it.
        drop_path_rate: Stochastic depth probability passed to timm backbones that support it.
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
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        prepend_global_summary_token: bool = False,
        split_duration_state: bool = False,
    ):
        super().__init__()

        if projector_config is None or isinstance(projector_config, dict):
            projector_config = resolve_projector_config(projector_config)

        timm_dropout_kwargs = _build_timm_dropout_kwargs(
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Load backbone without classification head, retaining all spatial tokens.
        try:
            backbone = _create_character_backbone(
                timm_model_id,
                pretrained=pretrained,
                timm_dropout_kwargs=timm_dropout_kwargs,
            )
        except TypeError as exc:
            if timm_dropout_kwargs:
                names = ", ".join(sorted(timm_dropout_kwargs))
                raise TypeError(
                    f"Backbone {timm_model_id!r} failed to initialize with timm dropout kwargs: {names}"
                ) from exc
            raise
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
