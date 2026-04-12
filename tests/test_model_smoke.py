from __future__ import annotations

# ruff: noqa: E402
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import irodori_tts.image_encoder as image_encoder
from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.rf import sample_euler_rf_cfg


class DummyImageBackbone(nn.Module):
    """Tiny backbone used to avoid network/model downloads in tests."""

    def __init__(self, feature_dim: int = 12, num_tokens: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_tokens = num_tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        return torch.zeros(
            batch_size,
            self.num_tokens,
            self.feature_dim,
            device=images.device,
            dtype=images.dtype,
        )


def make_base_config() -> ModelConfig:
    return ModelConfig(
        latent_dim=4,
        latent_patch_size=1,
        model_dim=16,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        text_vocab_size=32,
        text_tokenizer_repo="dummy/text",
        text_dim=8,
        text_layers=1,
        text_heads=2,
        text_mlp_ratio=2.0,
        speaker_dim=8,
        speaker_layers=1,
        speaker_heads=2,
        speaker_mlp_ratio=2.0,
        speaker_patch_size=1,
        timestep_embed_dim=8,
        adaln_rank=4,
        dropout=0.0,
    )


def make_character_config() -> ModelConfig:
    return ModelConfig(
        latent_dim=4,
        latent_patch_size=1,
        model_dim=16,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        text_vocab_size=32,
        text_tokenizer_repo="dummy/text",
        text_dim=8,
        text_layers=1,
        text_heads=2,
        text_mlp_ratio=2.0,
        use_character_condition=True,
        character_encoder_model="dummy/character-backbone",
        character_dim=8,
        character_use_all_patches=True,
        character_image_size=8,
        character_projector={
            "type": "mlp",
            "hidden_dim": 12,
            "num_layers": 1,
        },
        speaker_dim=8,
        speaker_layers=1,
        speaker_heads=2,
        speaker_mlp_ratio=2.0,
        speaker_patch_size=1,
        timestep_embed_dim=8,
        adaln_rank=4,
        dropout=0.0,
    )


def patch_character_backbone(monkeypatch) -> None:
    monkeypatch.setattr(
        image_encoder.timm,
        "create_model",
        lambda *args, **kwargs: DummyImageBackbone(),
    )


def test_base_model_forward_runs_without_error() -> None:
    cfg = make_base_config()
    model = TextToLatentRFDiT(cfg)

    batch_size = 2
    seq_len = 5
    text_len = 6
    ref_len = 4

    x_t = torch.randn(batch_size, seq_len, cfg.patched_latent_dim)
    t = torch.rand(batch_size)
    text_input_ids = torch.randint(0, cfg.text_vocab_size, (batch_size, text_len))
    text_mask = torch.ones(batch_size, text_len, dtype=torch.bool)
    speaker_latent = torch.randn(batch_size, ref_len, cfg.patched_latent_dim)
    speaker_mask = torch.ones(batch_size, ref_len, dtype=torch.bool)
    latent_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    out = model(
        x_t=x_t,
        t=t,
        text_input_ids=text_input_ids,
        text_mask=text_mask,
        speaker_latent=speaker_latent,
        speaker_mask=speaker_mask,
        latent_mask=latent_mask,
    )

    assert out.shape == x_t.shape
    assert out.dtype == x_t.dtype
    assert torch.isfinite(out).all()


def test_character_model_forward_runs_without_error(monkeypatch) -> None:
    patch_character_backbone(monkeypatch)
    cfg = make_character_config()
    model = TextToLatentRFDiT(cfg)

    batch_size = 2
    seq_len = 5
    text_len = 6

    x_t = torch.randn(batch_size, seq_len, cfg.patched_latent_dim)
    t = torch.rand(batch_size)
    text_input_ids = torch.randint(0, cfg.text_vocab_size, (batch_size, text_len))
    text_mask = torch.ones(batch_size, text_len, dtype=torch.bool)
    character_images = torch.randn(
        batch_size,
        3,
        cfg.character_image_size,
        cfg.character_image_size,
    )
    latent_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    out = model(
        x_t=x_t,
        t=t,
        text_input_ids=text_input_ids,
        text_mask=text_mask,
        speaker_latent=None,
        speaker_mask=None,
        character_images=character_images,
        latent_mask=latent_mask,
    )

    assert out.shape == x_t.shape
    assert out.dtype == x_t.dtype
    assert torch.isfinite(out).all()


def test_character_model_sampling_runs_without_error(monkeypatch) -> None:
    patch_character_backbone(monkeypatch)
    cfg = make_character_config()
    model = TextToLatentRFDiT(cfg)

    batch_size = 1
    text_len = 6
    sequence_length = 4

    text_input_ids = torch.randint(0, cfg.text_vocab_size, (batch_size, text_len))
    text_mask = torch.ones(batch_size, text_len, dtype=torch.bool)
    character_images = torch.randn(
        batch_size,
        3,
        cfg.character_image_size,
        cfg.character_image_size,
    )

    out = sample_euler_rf_cfg(
        model=model,
        text_input_ids=text_input_ids,
        text_mask=text_mask,
        ref_latent=None,
        ref_mask=None,
        sequence_length=sequence_length,
        character_images=character_images,
        num_steps=3,
        cfg_scale_text=1.0,
        cfg_scale_character=1.0,
        cfg_guidance_mode="independent",
        seed=0,
    )

    assert out.shape == (batch_size, sequence_length, cfg.patched_latent_dim)
    assert out.dtype == next(model.parameters()).dtype
    assert torch.isfinite(out).all()
