# ruff: noqa: E402
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import irodori_tts.generation_core as generation_core
from irodori_tts.config import ModelConfig


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def batch_encode(self, texts, max_length: int):
        texts = list(texts)
        self.calls.append((texts, int(max_length)))
        batch_size = len(texts)
        ids = torch.zeros((batch_size, max_length), dtype=torch.long)
        mask = torch.ones((batch_size, max_length), dtype=torch.bool)
        return ids, mask


class TinyModel(torch.nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=dtype))


class DummyCodec:
    def __init__(self, *, encoded_latent: torch.Tensor | None = None) -> None:
        self.sample_rate = 10
        self.device = torch.device("cpu")
        self.model = type("DummyCodecModel", (), {"hop_length": 4})()
        self.encoded_latent = (
            torch.zeros((1, 4, 2), dtype=torch.float32)
            if encoded_latent is None
            else encoded_latent.clone().float()
        )
        self.encode_calls: list[dict[str, object]] = []
        self.decode_inputs: list[torch.Tensor] = []

    def encode_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        normalize_db: float | None,
        ensure_max: bool,
    ) -> torch.Tensor:
        self.encode_calls.append(
            {
                "waveform_shape": tuple(waveform.shape),
                "sample_rate": int(sample_rate),
                "normalize_db": normalize_db,
                "ensure_max": bool(ensure_max),
            }
        )
        return self.encoded_latent.clone()

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(latent.detach().clone())
        batch_size = latent.shape[0]
        audio_len = latent.shape[1] * 10
        return torch.arange(audio_len, dtype=torch.float32).view(1, 1, -1).repeat(batch_size, 1, 1)


def make_model_config(*, use_caption_condition: bool = False) -> ModelConfig:
    return ModelConfig(
        latent_dim=2,
        latent_patch_size=2,
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
        use_caption_condition=use_caption_condition,
        speaker_dim=8,
        speaker_layers=1,
        speaker_heads=2,
        speaker_mlp_ratio=2.0,
        speaker_patch_size=1,
        timestep_embed_dim=8,
        adaln_rank=4,
        dropout=0.0,
    )


def test_normalize_generation_text_matches_runtime_rules() -> None:
    assert generation_core.normalize_generation_text("　（テスト？）\t") == "テスト?"


def test_resolve_lengths_follow_runtime_rules() -> None:
    assert generation_core.resolve_condition_lengths(
        default_text_max_len=123,
        default_caption_max_len=234,
        max_text_len=None,
        max_caption_len=None,
    ) == (123, 234)
    assert generation_core.resolve_sequence_lengths(
        seconds=0.45,
        sample_rate=1000,
        hop_length=100,
        latent_patch_size=4,
    ) == (450, 5, 2)


def test_resolve_cfg_scales_respects_override_and_disabled_branches() -> None:
    text, caption, speaker, character, messages = generation_core.resolve_cfg_scales(
        cfg_guidance_mode="independent",
        cfg_scale_text=1.0,
        cfg_scale_caption=2.0,
        cfg_scale_speaker=3.0,
        cfg_scale_character=4.0,
        cfg_scale=6.0,
        use_caption_condition=True,
        use_speaker_condition=False,
        use_character_condition=False,
    )

    assert (text, caption, speaker, character) == (6.0, 6.0, 0.0, 0.0)
    assert messages == [
        "info: speaker conditioning is disabled for this checkpoint; ignoring cfg_scale_speaker."
    ]


def test_find_flattening_point_detects_flat_tail() -> None:
    latent = torch.tensor(
        [
            [1.0, 1.0],
            [0.5, 0.5],
            [0.25, 0.25],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )

    assert (
        generation_core.find_flattening_point(
            latent,
            window_size=2,
            std_threshold=0.01,
            mean_threshold=0.05,
        )
        == 3
    )


def test_load_audio_falls_back_to_soundfile(monkeypatch) -> None:
    def boom(_path: str):
        raise RuntimeError("torchaudio failed")

    monkeypatch.setattr(generation_core.torchaudio, "load", boom)
    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        types.SimpleNamespace(
            read=lambda _path, dtype="float32": (
                np.array([0.25, -0.5], dtype=np.float32),
                24000,
            )
        ),
    )

    wav, sr = generation_core.load_audio("dummy.wav")

    assert sr == 24000
    assert wav.shape == (1, 2)
    assert torch.allclose(wav, torch.tensor([[0.25, -0.5]], dtype=torch.float32))


def test_prepare_reference_latent_returns_zero_mask_for_no_ref() -> None:
    codec = DummyCodec()
    model_cfg = make_model_config(use_caption_condition=False)
    messages: list[str] = []

    ref_latent, ref_mask = generation_core.prepare_reference_latent(
        request=generation_core.SamplingRequest(text="hello", no_ref=True),
        codec=codec,
        model_cfg=model_cfg,
        batch_size=2,
        model_device=torch.device("cpu"),
        model_dtype=torch.float32,
        messages=messages,
    )

    assert ref_latent is not None
    assert ref_mask is not None
    assert ref_latent.shape == (2, 1, 4)
    assert torch.count_nonzero(ref_latent).item() == 0
    assert torch.equal(ref_mask, torch.zeros((2, 1), dtype=torch.bool))
    assert messages == []


def test_prepare_reference_latent_trims_reference_audio_and_latent(monkeypatch) -> None:
    codec = DummyCodec(
        encoded_latent=torch.tensor(
            [
                [
                    [0.0, 1.0],
                    [2.0, 3.0],
                    [4.0, 5.0],
                    [6.0, 7.0],
                    [8.0, 9.0],
                ]
            ]
        )
    )
    model_cfg = make_model_config(use_caption_condition=False)
    messages: list[str] = []

    monkeypatch.setattr(
        generation_core,
        "load_audio",
        lambda _path: (torch.zeros((1, 12), dtype=torch.float32), 10),
    )

    ref_latent, ref_mask = generation_core.prepare_reference_latent(
        request=generation_core.SamplingRequest(
            text="hello",
            ref_wav="ref.wav",
            max_ref_seconds=0.5,
            ref_normalize_db=None,
            ref_ensure_max=True,
        ),
        codec=codec,
        model_cfg=model_cfg,
        batch_size=2,
        model_device=torch.device("cpu"),
        model_dtype=torch.float32,
        messages=messages,
    )

    assert codec.encode_calls == [
        {
            "waveform_shape": (1, 1, 5),
            "sample_rate": 10,
            "normalize_db": None,
            "ensure_max": True,
        }
    ]
    assert ref_latent is not None
    assert ref_mask is not None
    assert ref_latent.shape == (2, 1, 4)
    assert torch.equal(ref_mask, torch.ones((2, 1), dtype=torch.bool))
    assert any("reference audio exceeds max_ref_seconds" in message for message in messages)
    assert any("Trimming reference latent" in message for message in messages)


def test_prepare_speaker_embedding_condition_loads_embedding(tmp_path: Path) -> None:
    model_cfg = make_model_config(use_caption_condition=False)
    path = tmp_path / "speaker_embedding.pt"
    torch.save(
        {
            "speaker_embedding": torch.ones(3, model_cfg.speaker_dim),
            "speaker_uncond_embedding": torch.zeros(3, model_cfg.speaker_dim),
            "speaker_uncond_mode": "noise",
        },
        path,
    )
    messages: list[str] = []

    state, mask, uncond_state, uncond_mask, uncond_mode = (
        generation_core.prepare_speaker_embedding_condition(
            request=generation_core.SamplingRequest(
                text="hello",
                speaker_embedding=str(path),
            ),
            model_cfg=model_cfg,
            batch_size=2,
            model_device=torch.device("cpu"),
            model_dtype=torch.float32,
            messages=messages,
        )
    )

    assert state is not None
    assert mask is not None
    assert uncond_state is not None
    assert uncond_mask is not None
    assert state.shape == (2, 3, model_cfg.speaker_dim)
    assert torch.equal(mask, torch.ones((2, 3), dtype=torch.bool))
    assert torch.count_nonzero(uncond_state).item() == 0
    assert torch.equal(uncond_mask, torch.ones((2, 3), dtype=torch.bool))
    assert uncond_mode == "noise"
    assert any("using speaker inversion embedding" in message for message in messages)


def test_generate_from_components_uses_shared_runtime_rules(monkeypatch) -> None:
    model_cfg = make_model_config(use_caption_condition=True)
    model = TinyModel()
    tokenizer = RecordingTokenizer()
    caption_tokenizer = RecordingTokenizer()
    codec = DummyCodec()
    sampled: dict[str, object] = {}

    def fake_sample_euler_rf_cfg(**kwargs):
        sampled.update(kwargs)
        return torch.zeros(
            (1, kwargs["sequence_length"], model_cfg.patched_latent_dim),
            dtype=torch.float32,
        )

    monkeypatch.setattr(generation_core, "sample_euler_rf_cfg", fake_sample_euler_rf_cfg)

    result = generation_core.generate_from_components(
        model=model,
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        caption_tokenizer=caption_tokenizer,
        codec=codec,
        request=generation_core.SamplingRequest(
            text="　（テスト？）\t",
            caption="  cap  ",
            no_ref=True,
            seconds=0.9,
            num_steps=2,
            cfg_scale_speaker=0.0,
            seed=123,
            trim_tail=False,
        ),
        default_text_max_len=7,
        default_caption_max_len=7,
        character_image_transform=None,
        model_device=torch.device("cpu"),
        codec_device=torch.device("cpu"),
    )

    assert tokenizer.calls == [(["テスト?"], 7)]
    assert caption_tokenizer.calls == [(["cap"], 7)]
    assert sampled["sequence_length"] == 2
    assert len(codec.decode_inputs) == 1
    assert tuple(codec.decode_inputs[0].shape) == (1, 3, 2)
    assert result.audio.shape == (1, 9)
    assert result.used_seed == 123


def test_resolve_cfg_scales_rejects_joint_mode_mismatch() -> None:
    with pytest.raises(ValueError, match="cfg_guidance_mode='joint'"):
        generation_core.resolve_cfg_scales(
            cfg_guidance_mode="joint",
            cfg_scale_text=1.0,
            cfg_scale_caption=2.0,
            cfg_scale_speaker=1.0,
            cfg_scale_character=1.0,
            cfg_scale=None,
            use_caption_condition=True,
            use_speaker_condition=True,
            use_character_condition=False,
        )
