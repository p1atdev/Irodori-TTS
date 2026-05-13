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
from irodori_tts.projector import MLPProjectorConfig
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


def patch_character_backbone(monkeypatch, *, feature_dim: int = 12, num_tokens: int = 4) -> None:
    monkeypatch.setattr(
        image_encoder,
        "create_model",
        lambda *args, **kwargs: DummyImageBackbone(
            feature_dim=feature_dim,
            num_tokens=num_tokens,
        ),
    )


def test_character_projector_uses_explicit_initialization(monkeypatch) -> None:
    patch_character_backbone(monkeypatch, feature_dim=64, num_tokens=4)

    encoder = image_encoder.CharacterImageEncoder(
        timm_model_id="dummy/character-backbone",
        output_dim=32,
        use_all_patches=True,
        image_size=8,
        pretrained=False,
        projector_config=MLPProjectorConfig(
            hidden_dim=128,
            num_layers=2,
        ),
    )

    projector_linears = [m for m in encoder.proj.modules() if isinstance(m, nn.Linear)]
    assert projector_linears

    for linear in projector_linears:
        weight_std = float(linear.weight.std().item())
        assert 0.015 <= weight_std <= 0.025
        if linear.bias is not None:
            assert torch.count_nonzero(linear.bias).item() == 0

    assert torch.allclose(encoder.norm.weight, torch.ones_like(encoder.norm.weight))


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


def test_inference_rope_cache_does_not_poison_training_backward() -> None:
    cfg = make_base_config()
    model = TextToLatentRFDiT(cfg)

    with torch.inference_mode():
        freqs = model._rope_freqs(8, torch.device("cpu"))

    assert freqs.shape[0] == 8
    assert model._freqs_cis_cache.numel() == 0

    batch_size = 2
    seq_len = 4
    text_len = 6
    ref_len = 3

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
    out.square().mean().backward()

    assert not bool(getattr(model._freqs_cis_cache, "is_inference", lambda: False)())


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


def test_character_duration_predictor_uses_null_speaker_without_reference(monkeypatch) -> None:
    patch_character_backbone(monkeypatch)
    cfg = make_character_config()
    cfg.use_duration_predictor = True
    cfg.duration_hidden_dim = 16
    cfg.duration_layers = 1
    cfg.duration_dropout = 0.0
    cfg.duration_architecture = "token_sum_adarn_zero_no_aux"
    cfg.duration_speaker_fusion = "adarn_zero"
    model = TextToLatentRFDiT(cfg)

    assert not cfg.use_speaker_condition
    assert model.duration_predictor is not None
    assert model.duration_predictor.speaker_dim == cfg.speaker_dim

    batch_size = 2
    text_len = 6
    text_state = torch.randn(batch_size, text_len, cfg.text_dim)
    text_mask = torch.ones(batch_size, text_len, dtype=torch.bool)
    duration_features = torch.zeros(batch_size, cfg.duration_aux_dim)

    out = model.predict_duration_log_frames(
        text_state=text_state,
        text_mask=text_mask,
        speaker_state=None,
        speaker_mask=None,
        duration_features=duration_features,
        has_speaker=None,
    )

    assert out.shape == (batch_size,)
    assert torch.isfinite(out).all()


def test_character_duration_predictor_receives_character_state(monkeypatch) -> None:
    patch_character_backbone(monkeypatch)
    cfg = make_character_config()
    cfg.use_duration_predictor = True
    cfg.duration_hidden_dim = 16
    cfg.duration_layers = 1
    cfg.duration_dropout = 0.0
    cfg.duration_architecture = "token_sum_adarn_zero_no_aux"
    cfg.duration_speaker_fusion = "adarn_zero"
    model = TextToLatentRFDiT(cfg)

    assert model.duration_predictor is not None
    original_forward = model.duration_predictor.forward
    captured: dict[str, torch.Tensor | None] = {}

    def capture_forward(**kwargs):
        captured["speaker_state"] = kwargs["speaker_state"]
        captured["speaker_mask"] = kwargs["speaker_mask"]
        captured["has_speaker"] = kwargs["has_speaker"]
        return original_forward(**kwargs)

    monkeypatch.setattr(model.duration_predictor, "forward", capture_forward)

    batch_size = 2
    text_len = 6
    character_tokens = 3
    text_state = torch.randn(batch_size, text_len, cfg.text_dim)
    text_mask = torch.ones(batch_size, text_len, dtype=torch.bool)
    character_state = torch.randn(batch_size, character_tokens, cfg.speaker_dim)
    character_mask = torch.tensor(
        [
            [True, True, False],
            [False, False, False],
        ],
        dtype=torch.bool,
    )
    duration_features = torch.zeros(batch_size, cfg.duration_aux_dim)

    out = model.predict_duration_log_frames(
        text_state=text_state,
        text_mask=text_mask,
        speaker_state=None,
        speaker_mask=None,
        character_state=character_state,
        character_mask=character_mask,
        duration_features=duration_features,
        has_speaker=None,
    )

    expected_state, expected_mask = model._prepend_masked_mean_token(
        character_state,
        character_mask,
    )
    assert out.shape == (batch_size,)
    assert torch.isfinite(out).all()
    assert torch.allclose(captured["speaker_state"], expected_state)
    assert torch.equal(captured["speaker_mask"], expected_mask)
    assert torch.equal(captured["has_speaker"], torch.tensor([True, False]))


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
