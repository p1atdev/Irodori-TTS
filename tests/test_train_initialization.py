from __future__ import annotations

# ruff: noqa: E402
import sys
import warnings
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
    # base_cfg has speaker condition enabled (no caption/character), so the
    # warm-start priority (caption > speaker > text) picks speaker weights.
    for i, block in enumerate(char_model.blocks):
        expected_wk = base_model.blocks[i].attention.wk_speaker.weight
        expected_wv = base_model.blocks[i].attention.wv_speaker.weight
        assert torch.equal(block.attention.wk_character.weight, expected_wk)
        assert torch.equal(block.attention.wv_character.weight, expected_wv)


def test_apply_base_initialization_skips_character_copy_on_shape_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    patch_character_backbone(monkeypatch)

    base_cfg = make_base_config()
    base_model = TextToLatentRFDiT(base_cfg)

    checkpoint_path = tmp_path / "base_checkpoint.pt"
    torch.save(
        {
            "model": base_model.state_dict(),
            "model_config": asdict(base_cfg),
        },
        checkpoint_path,
    )

    char_cfg = make_character_config()
    # Make character_dim differ from text_dim so the copy path should skip.
    char_cfg.character_dim = base_cfg.text_dim * 2
    char_model = TextToLatentRFDiT(char_cfg)
    character_attn_before = [
        block.attention.wk_character.weight.detach().clone() for block in char_model.blocks
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train._apply_base_initialization(
            char_model,
            model_cfg=char_cfg,
            base_init={"mode": "checkpoint", "checkpoint_path": str(checkpoint_path)},
            distributed=False,
            is_main_process=True,
        )

    assert any("shape mismatch" in str(w.message) for w in caught)
    for block, before in zip(char_model.blocks, character_attn_before, strict=True):
        assert torch.equal(block.attention.wk_character.weight, before)


def _fake_source_state(num_blocks: int, prefix: str, dim: int, in_dim: int) -> dict:
    state: dict[str, torch.Tensor] = {}
    for i in range(num_blocks):
        state[f"blocks.{i}.attention.wk_{prefix}.weight"] = torch.randn(dim, in_dim)
        state[f"blocks.{i}.attention.wv_{prefix}.weight"] = torch.randn(dim, in_dim)
    return state


def test_initialize_character_attention_prefers_caption(monkeypatch) -> None:
    patch_character_backbone(monkeypatch)

    char_cfg = make_character_config()
    char_model = TextToLatentRFDiT(char_cfg)
    dim = char_model.blocks[0].attention.wk_character.weight.shape[0]
    in_dim = char_model.blocks[0].attention.wk_character.weight.shape[1]

    state = _fake_source_state(len(char_model.blocks), "caption", dim, in_dim)
    # Also provide text weights; caption should win over them.
    state.update(_fake_source_state(len(char_model.blocks), "text", dim, in_dim))

    copied, skipped, sources = train.initialize_character_attention_from_checkpoint(
        char_model, state
    )

    assert copied == len(char_model.blocks)
    assert skipped == 0
    assert sources == {"caption": len(char_model.blocks), "speaker": 0, "text": 0}
    for i, block in enumerate(char_model.blocks):
        assert torch.equal(
            block.attention.wk_character.weight,
            state[f"blocks.{i}.attention.wk_caption.weight"],
        )
        assert torch.equal(
            block.attention.wv_character.weight,
            state[f"blocks.{i}.attention.wv_caption.weight"],
        )


def test_initialize_character_attention_prefers_speaker_over_text(monkeypatch) -> None:
    patch_character_backbone(monkeypatch)

    char_cfg = make_character_config()
    char_model = TextToLatentRFDiT(char_cfg)
    dim = char_model.blocks[0].attention.wk_character.weight.shape[0]
    in_dim = char_model.blocks[0].attention.wk_character.weight.shape[1]

    state = _fake_source_state(len(char_model.blocks), "speaker", dim, in_dim)
    state.update(_fake_source_state(len(char_model.blocks), "text", dim, in_dim))

    copied, skipped, sources = train.initialize_character_attention_from_checkpoint(
        char_model, state
    )

    assert copied == len(char_model.blocks)
    assert skipped == 0
    assert sources == {"caption": 0, "speaker": len(char_model.blocks), "text": 0}
    for i, block in enumerate(char_model.blocks):
        assert torch.equal(
            block.attention.wk_character.weight,
            state[f"blocks.{i}.attention.wk_speaker.weight"],
        )
        assert torch.equal(
            block.attention.wv_character.weight,
            state[f"blocks.{i}.attention.wv_speaker.weight"],
        )
