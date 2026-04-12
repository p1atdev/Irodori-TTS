from __future__ import annotations

# ruff: noqa: E402
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import irodori_tts.image_encoder as image_encoder
import train
from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT


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

    def reset_classifier(self, num_classes: int) -> None:
        del num_classes


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
        image_encoder,
        "create_model",
        lambda *args, **kwargs: DummyImageBackbone(),
    )


def test_apply_base_initialization_allows_character_upgrade(monkeypatch, tmp_path: Path) -> None:
    patch_character_backbone(monkeypatch)

    base_cfg = make_base_config()
    base_model = TextToLatentRFDiT(base_cfg)
    with torch.no_grad():
        base_model.in_proj.weight.fill_(0.1234)
        base_model.in_proj.bias.fill_(0.5678)

    checkpoint_path = tmp_path / "base_checkpoint.pt"
    torch.save(
        {
            "model": base_model.state_dict(),
            "model_config": asdict(base_cfg),
        },
        checkpoint_path,
    )

    char_cfg = make_character_config()
    char_model = TextToLatentRFDiT(char_cfg)
    character_proj_before = char_model.character_encoder.proj.blocks[0].fc1.weight.detach().clone()
    character_attn_before = char_model.blocks[0].attention.wk_character.weight.detach().clone()

    train._apply_base_initialization(
        char_model,
        model_cfg=char_cfg,
        base_init={"mode": "checkpoint", "checkpoint_path": str(checkpoint_path)},
        distributed=False,
        is_main_process=False,
    )

    assert torch.allclose(char_model.in_proj.weight, base_model.in_proj.weight)
    assert torch.allclose(char_model.in_proj.bias, base_model.in_proj.bias)
    assert torch.allclose(
        char_model.character_encoder.proj.blocks[0].fc1.weight,
        character_proj_before,
    )
    assert torch.allclose(
        char_model.blocks[0].attention.wk_character.weight,
        character_attn_before,
    )
