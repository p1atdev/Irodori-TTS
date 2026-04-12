from __future__ import annotations

# ruff: noqa: E402
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train
from irodori_tts.config import ModelConfig, TrainConfig


class DummyModel:
    def __init__(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True


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


class DummyCodec:
    def __init__(self) -> None:
        self.sample_rate = 10
        self.model = type("DummyCodecModel", (), {"hop_length": 4})()
        self.decode_inputs: list[torch.Tensor] = []

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(latent.detach().clone())
        batch_size = latent.shape[0]
        audio_len = latent.shape[1] * 10
        return torch.arange(audio_len, dtype=torch.float32).view(1, 1, -1).repeat(batch_size, 1, 1)


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


def test_run_preview_uses_runtime_text_rules_and_crops_output(monkeypatch) -> None:
    model_cfg = ModelConfig(
        latent_dim=2,
        latent_patch_size=2,
        use_caption_condition=True,
    )
    train_cfg = TrainConfig(max_text_len=7, max_caption_len=None)
    model = DummyModel()
    tokenizer = RecordingTokenizer()
    caption_tokenizer = RecordingTokenizer()
    codec = DummyCodec()
    wandb_run = DummyWandbRun()

    monkeypatch.setattr(
        train,
        "sample_euler_rf_cfg",
        lambda **kwargs: torch.zeros(
            (1, kwargs["sequence_length"], model_cfg.patched_latent_dim),
            dtype=torch.float32,
        ),
    )
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
                seconds=0.9,
                num_steps=2,
            )
        ],
        codec=codec,
        device=torch.device("cpu"),
        use_bf16=False,
        step=12,
        tokenizer=tokenizer,
        caption_tokenizer=caption_tokenizer,
        character_image_transform=None,
        wandb_run=wandb_run,
    )

    assert tokenizer.calls == [(["テスト?"], 7)]
    assert caption_tokenizer.calls == [(["cap"], 7)]
    assert len(codec.decode_inputs) == 1
    assert tuple(codec.decode_inputs[0].shape) == (1, 3, 2)
    assert len(wandb_run.records) == 1

    payload, step = wandb_run.records[0]
    assert step == 12
    assert payload["preview/audio_0"] == {
        "num_samples": 9,
        "sample_rate": 10,
        "caption": "テスト?",
    }
    assert model.training is True
