from dataclasses import asdict
from typing import Any

import torch.nn as nn

from .mlp import MLPProjector, MLPProjectorConfig
from .resampler import ResamplerProjector, ResamplerProjectorConfig

ProjectorConfig = MLPProjectorConfig | ResamplerProjectorConfig

_CONFIG_TYPES: dict[str, type] = {
    "mlp": MLPProjectorConfig,
    "resampler": ResamplerProjectorConfig,
}


def resolve_projector_config(payload: dict[str, Any] | None) -> ProjectorConfig:
    """Build a concrete projector config dataclass from a raw mapping."""
    if payload is None:
        return MLPProjectorConfig()
    if not isinstance(payload, dict):
        raise ValueError(f"projector config must be a mapping, got {type(payload)!r}")

    data = dict(payload)
    proj_type = data.pop("type")
    if proj_type not in _CONFIG_TYPES:
        raise ValueError(f"Unknown projector type {proj_type!r}. Known: {sorted(_CONFIG_TYPES)}")

    return _CONFIG_TYPES[proj_type](type=proj_type, **data)


def projector_config_to_dict(config: ProjectorConfig) -> dict[str, Any]:
    return asdict(config)


def build_projector(
    config: ProjectorConfig,
    backbone_dim: int,
    output_dim: int,
) -> nn.Module:
    """Factory for building a projector module from a resolved config."""
    if isinstance(config, MLPProjectorConfig):
        hidden_dim = config.hidden_dim if config.hidden_dim is not None else backbone_dim
        return MLPProjector(
            in_dim=backbone_dim,
            hidden_dim=hidden_dim,
            out_dim=output_dim,
            num_layers=config.num_layers,
        )
    if isinstance(config, ResamplerProjectorConfig):
        return ResamplerProjector(
            in_dim=backbone_dim,
            out_dim=output_dim,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_query_tokens=config.num_query_tokens,
            depth=config.depth,
            gradient_checkpointing=config.gradient_checkpointing,
            qk_norm=config.qk_norm,
            is_gated=config.is_gated,
        )
    raise ValueError(f"Unknown projector config type: {type(config).__name__}")


__all__ = [
    "MLPProjector",
    "MLPProjectorConfig",
    "ResamplerProjector",
    "ResamplerProjectorConfig",
    "ProjectorConfig",
    "build_projector",
    "resolve_projector_config",
    "projector_config_to_dict",
]
