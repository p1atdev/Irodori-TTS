# ruff: noqa: E402
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from irodori_tts.projector import ResamplerProjector, build_projector, resolve_projector_config


def test_resampler_projector_config_passes_dropout_to_blocks() -> None:
    config = resolve_projector_config(
        {
            "type": "resampler",
            "num_heads": 2,
            "depth": 3,
            "num_query_tokens": 4,
            "dropout": 0.25,
        }
    )

    projector = build_projector(config, backbone_dim=6, output_dim=8)

    assert isinstance(projector, ResamplerProjector)
    assert [block.dropout.p for block in projector.blocks] == [0.25, 0.25, 0.25]


def test_resampler_projector_dropout_drops_residual_updates_in_train_mode() -> None:
    torch.manual_seed(0)
    projector = ResamplerProjector(
        in_dim=4,
        out_dim=8,
        num_heads=2,
        num_query_tokens=3,
        depth=2,
        dropout=1.0,
    )
    projector.train()

    first = projector(torch.randn(2, 5, 4))
    second = projector(torch.randn(2, 5, 4))

    assert torch.allclose(first, second)
