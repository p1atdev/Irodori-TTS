# ruff: noqa: E402
import sys
from collections import OrderedDict
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


def test_character_encoder_passes_timm_dropout_config(monkeypatch) -> None:
    captured_kwargs = {}

    def fake_create_model(*args, **kwargs):
        del args
        captured_kwargs.update(kwargs)
        return DummyImageBackbone()

    monkeypatch.setattr(image_encoder, "create_model", fake_create_model)
    cfg = make_character_config()
    cfg.character_encoder_drop_rate = 0.1
    cfg.character_encoder_attn_drop_rate = 0.2
    cfg.character_encoder_drop_path_rate = 0.03

    TextToLatentRFDiT(cfg)

    assert captured_kwargs["drop_rate"] == 0.1
    assert captured_kwargs["attn_drop_rate"] == 0.2
    assert captured_kwargs["drop_path_rate"] == 0.03


def test_speaker_inversion_forward_bypasses_reference_encoder() -> None:
    cfg = make_base_config()
    model = TextToLatentRFDiT(cfg)
    model.enable_speaker_inversion(
        num_tokens=3,
        init_std=0.01,
        uncond_mode="noise",
        uncond_std=0.02,
    )
    with torch.no_grad():
        model.out_proj.weight.normal_(std=0.02)

    x_t = torch.randn(2, 4, cfg.patched_latent_dim)
    t = torch.full((2,), 0.5)
    text_ids = torch.ones((2, 5), dtype=torch.long)
    text_mask = torch.ones((2, 5), dtype=torch.bool)

    out = model(
        x_t=x_t,
        t=t,
        text_input_ids=text_ids,
        text_mask=text_mask,
        speaker_latent=None,
        speaker_mask=None,
        speaker_condition_dropout=torch.tensor([False, True]),
    )
    assert out.shape == x_t.shape

    out.square().mean().backward()
    assert model.speaker_inversion.embedding.grad is not None
    assert model.speaker_encoder.in_proj.weight.grad is None


def test_ccip_transform_uses_ccip_preprocess(monkeypatch) -> None:
    captured_kwargs = {}
    sentinel = object()

    def fake_create_transform(**kwargs):
        captured_kwargs.update(kwargs)
        return sentinel

    def unexpected_create_model(*args, **kwargs):
        del args, kwargs
        raise AssertionError("CCIP transform should not instantiate a timm model")

    monkeypatch.setattr(image_encoder.timm_data, "create_transform", fake_create_transform)
    monkeypatch.setattr(image_encoder, "create_model", unexpected_create_model)

    transform = image_encoder.build_character_transform("ccip:ccip-caformer_b36-24", 384)

    assert transform is sentinel
    assert captured_kwargs["input_size"] == (3, 384, 384)
    assert captured_kwargs["interpolation"] == "bilinear"
    assert captured_kwargs["crop_pct"] == 1.0
    assert captured_kwargs["mean"] == image_encoder._CCIP_MEAN
    assert captured_kwargs["std"] == image_encoder._CCIP_STD


def test_convert_ccip_caformer_state_dict_maps_timm_keys() -> None:
    prefix = image_encoder._CCIP_CAFormer_PREFIX
    target_state = OrderedDict(
        {
            "stem.conv.weight": torch.empty(2, 3, 7, 7),
            "stem.norm.weight": torch.empty(2),
            "stages.1.downsample.norm.weight": torch.empty(2),
            "stages.1.downsample.conv.weight": torch.empty(4, 2, 3, 3),
            "stages.0.blocks.0.token_mixer.pwconv1.weight": torch.empty(4, 2, 1, 1),
            "head.norm.weight": torch.empty(8),
            "head.fc.fc1.weight": torch.empty(32, 8),
            "head.fc.norm.weight": torch.empty(32),
            "head.fc.fc2.weight": torch.empty(2, 32),
        }
    )
    checkpoint_state = OrderedDict(
        {
            f"{prefix}downsample_layers.0.conv.weight": torch.randn(2, 3, 7, 7),
            f"{prefix}downsample_layers.0.post_norm.weight": torch.randn(2),
            f"{prefix}downsample_layers.1.pre_norm.weight": torch.randn(2),
            f"{prefix}downsample_layers.1.conv.weight": torch.randn(4, 2, 3, 3),
            f"{prefix}stages.0.0.token_mixer.pwconv1.weight": torch.randn(4, 2),
            f"{prefix}norm.weight": torch.randn(8),
            f"{prefix}head.fc1.weight": torch.randn(32, 8),
            f"{prefix}head.norm.weight": torch.randn(32),
            f"{prefix}head.fc2.weight": torch.randn(2, 32),
            "module._orig_mod.feature.backbone.attnpool.q_proj.weight": torch.randn(8, 8),
        }
    )

    converted = image_encoder._convert_ccip_caformer_state_dict(
        checkpoint_state,
        target_state,
    )

    assert set(converted) == set(target_state)
    assert converted["stages.0.blocks.0.token_mixer.pwconv1.weight"].shape == (4, 2, 1, 1)
    assert torch.equal(
        converted["stages.0.blocks.0.token_mixer.pwconv1.weight"].squeeze(-1).squeeze(-1),
        checkpoint_state[f"{prefix}stages.0.0.token_mixer.pwconv1.weight"],
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


def test_character_encoder_can_prepend_global_summary_token(monkeypatch) -> None:
    patch_character_backbone(monkeypatch, feature_dim=12, num_tokens=4)

    encoder = image_encoder.CharacterImageEncoder(
        timm_model_id="dummy/character-backbone",
        output_dim=8,
        use_all_patches=True,
        image_size=8,
        pretrained=False,
        projector_config=MLPProjectorConfig(
            hidden_dim=16,
            num_layers=1,
        ),
        prepend_global_summary_token=True,
    )

    out = encoder(torch.randn(2, 3, 8, 8))

    assert out.shape == (2, 5, 8)
    assert torch.allclose(out[:, 0], out[:, 1:].mean(dim=1))


def test_character_encoder_can_split_duration_state(monkeypatch) -> None:
    patch_character_backbone(monkeypatch, feature_dim=12, num_tokens=4)

    encoder = image_encoder.CharacterImageEncoder(
        timm_model_id="dummy/character-backbone",
        output_dim=8,
        use_all_patches=True,
        image_size=8,
        pretrained=False,
        projector_config=MLPProjectorConfig(
            hidden_dim=16,
            num_layers=1,
        ),
        prepend_global_summary_token=True,
        split_duration_state=True,
    )

    out = encoder(torch.randn(2, 3, 8, 8))

    assert isinstance(out, image_encoder.CharacterImageEncoderOutput)
    assert encoder.norm.weight.shape == (16,)
    assert out.generation_state.shape == (2, 5, 8)
    assert out.duration_state.shape == (2, 5, 8)
    assert torch.allclose(out.generation_state[:, 0], out.generation_state[:, 1:].mean(dim=1))
    assert torch.allclose(out.duration_state[:, 0], out.duration_state[:, 1:].mean(dim=1))


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


def test_character_duration_predictor_uses_split_duration_state(monkeypatch) -> None:
    patch_character_backbone(monkeypatch)
    cfg = make_character_config()
    cfg.character_projector_split_duration_state = True
    cfg.use_duration_predictor = True
    cfg.duration_hidden_dim = 16
    cfg.duration_layers = 1
    cfg.duration_dropout = 0.0
    cfg.duration_architecture = "token_sum_adarn_zero_no_aux"
    cfg.duration_speaker_fusion = "adarn_zero"
    model = TextToLatentRFDiT(cfg)

    class FakeSplitCharacterEncoder(nn.Module):
        def forward(self, images: torch.Tensor) -> image_encoder.CharacterImageEncoderOutput:
            batch_size = images.shape[0]
            shape = (batch_size, 3, cfg.character_dim_resolved)
            generation_state = torch.ones(shape, device=images.device, dtype=images.dtype)
            duration_state = torch.full(shape, 2.0, device=images.device, dtype=images.dtype)
            return image_encoder.CharacterImageEncoderOutput(
                generation_state=generation_state,
                duration_state=duration_state,
            )

    model.character_encoder = FakeSplitCharacterEncoder()

    assert model.duration_predictor is not None
    captured: dict[str, torch.Tensor | None] = {}

    def capture_forward(**kwargs):
        captured["speaker_state"] = kwargs["speaker_state"]
        captured["speaker_mask"] = kwargs["speaker_mask"]
        captured["has_speaker"] = kwargs["has_speaker"]
        batch_size = kwargs["text_state"].shape[0]
        return torch.zeros(batch_size, device=kwargs["text_state"].device)

    monkeypatch.setattr(model.duration_predictor, "forward", capture_forward)

    batch_size = 2
    text_len = 6
    text_input_ids = torch.randint(0, cfg.text_vocab_size, (batch_size, text_len))
    text_mask = torch.ones(batch_size, text_len, dtype=torch.bool)
    character_images = torch.randn(
        batch_size,
        3,
        cfg.character_image_size,
        cfg.character_image_size,
    )
    duration_features = torch.zeros(batch_size, cfg.duration_aux_dim)

    out = model(
        x_t=None,
        t=None,
        text_input_ids=text_input_ids,
        text_mask=text_mask,
        speaker_latent=None,
        speaker_mask=None,
        character_images=character_images,
        duration_features=duration_features,
        duration_only=True,
    )

    assert out.shape == (batch_size,)
    assert captured["speaker_state"] is not None
    assert captured["speaker_mask"] is not None
    assert captured["speaker_state"].shape == (batch_size, 4, cfg.speaker_dim)
    assert torch.allclose(
        captured["speaker_state"], torch.full_like(captured["speaker_state"], 2.0)
    )
    assert captured["speaker_mask"].all()
    assert torch.equal(captured["has_speaker"], torch.ones(batch_size, dtype=torch.bool))


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
