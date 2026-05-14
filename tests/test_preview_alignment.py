# ruff: noqa: E402
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train
from irodori_tts.config import ModelConfig, TrainConfig
from irodori_tts.generation_core import SamplingResult


class DummyModel:
    def __init__(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True


class DummyCodec:
    def __init__(self) -> None:
        self.sample_rate = 10
        self.device = torch.device("cpu")
        self.model = type("DummyCodecModel", (), {"hop_length": 4})()


class DummyWandbRun:
    def __init__(self) -> None:
        self.records: list[tuple[dict, int]] = []

    def log(self, payload: dict, step: int) -> None:
        self.records.append((payload, int(step)))


def test_resolve_preview_condition_lengths_uses_caption_fallback() -> None:
    text_max_len, caption_max_len = train.resolve_preview_condition_lengths(
        TrainConfig(max_text_len=123, max_caption_len=None)
    )

    assert text_max_len == 123
    assert caption_max_len == 123


def test_resolve_preview_sequence_lengths_uses_ceil_rules() -> None:
    target_samples, latent_steps, patched_steps = train.resolve_preview_sequence_lengths(
        seconds=0.45,
        sample_rate=1000,
        hop_length=100,
        latent_patch_size=4,
    )

    assert target_samples == 450
    assert latent_steps == 5
    assert patched_steps == 2


def test_preview_sample_to_sampling_request_uses_normalized_text_for_length_estimate() -> None:
    request, normalized_text = train.preview_sample_to_sampling_request(
        train.PreviewSampleConfig(
            text="　（テスト？）\t",
            caption="  cap  ",
            seconds=None,
            num_steps=2,
            seed=123,
        )
    )

    assert normalized_text == "テスト?"
    assert request.text == "　（テスト？）\t"
    assert request.caption == "  cap  "
    assert request.seconds == 2.0
    assert request.duration_scale == 1.0
    assert request.seed == 123
    assert request.trim_tail is True


def test_preview_sample_to_sampling_request_can_use_duration_predictor() -> None:
    request, normalized_text = train.preview_sample_to_sampling_request(
        train.PreviewSampleConfig(
            text="　（テスト？）\t",
            seconds=20.0,
            use_duration_predictor=True,
            duration_scale=1.25,
        )
    )

    assert normalized_text == "テスト?"
    assert request.seconds is None
    assert request.duration_scale == 1.25


def test_preview_sample_to_sampling_request_allows_trim_tail_false() -> None:
    request, _ = train.preview_sample_to_sampling_request(
        train.PreviewSampleConfig(text="テスト", trim_tail=False)
    )

    assert request.trim_tail is False


def test_run_preview_delegates_to_shared_core_and_logs_audio(monkeypatch) -> None:
    model_cfg = ModelConfig(
        latent_dim=2,
        latent_patch_size=2,
        use_caption_condition=True,
    )
    train_cfg = TrainConfig(max_text_len=7, max_caption_len=None, fixed_target_latent_steps=11)
    model = DummyModel()
    codec = DummyCodec()
    wandb_run = DummyWandbRun()
    captured: dict[str, object] = {}

    def fake_generate_from_components(**kwargs):
        captured.update(kwargs)
        audio = torch.arange(9, dtype=torch.float32).view(1, -1)
        return SamplingResult(
            audio=audio,
            audios=[audio],
            sample_rate=codec.sample_rate,
            stage_timings=[("stub", 0.01)],
            total_to_decode=0.01,
            used_seed=123,
            messages=[],
        )

    monkeypatch.setattr(train, "generate_from_components", fake_generate_from_components)
    monkeypatch.setattr(
        train.wandb,
        "Audio",
        lambda audio, sample_rate, caption: {
            "num_samples": int(len(audio)),
            "sample_rate": int(sample_rate),
            "caption": str(caption),
        },
    )

    train.run_preview(
        model=model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        samples=[
            train.PreviewSampleConfig(
                text="　（テスト？）\t",
                caption="  cap  ",
                seconds=None,
                num_steps=2,
                seed=123,
            )
        ],
        codec=codec,
        device=torch.device("cpu"),
        use_bf16=False,
        step=12,
        tokenizer=object(),
        caption_tokenizer=object(),
        character_image_transform=None,
        wandb_run=wandb_run,
    )

    request = captured["request"]
    assert isinstance(request, train.SamplingRequest)
    assert request.text == "　（テスト？）\t"
    assert request.caption == "  cap  "
    assert request.seconds == 2.0
    assert request.trim_tail is True
    assert request.seed == 123
    assert captured["default_text_max_len"] == 7
    assert captured["default_caption_max_len"] == 7
    assert captured["fixed_target_latent_steps"] == 11
    assert captured["codec_device"] == torch.device("cpu")
    assert len(wandb_run.records) == 1

    payload, step = wandb_run.records[0]
    assert step == 12
    assert payload["preview/audio_0"] == {
        "num_samples": 9,
        "sample_rate": 10,
        "caption": "テスト?",
    }
    assert model.training is True
