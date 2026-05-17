# ruff: noqa: E402
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio_app
import train
from irodori_tts.config import ModelConfig, TrainConfig
from irodori_tts.model import TextToLatentRFDiT


def make_config() -> ModelConfig:
    return ModelConfig(
        latent_dim=4,
        latent_patch_size=1,
        model_dim=16,
        num_layers=1,
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


class FakeCodec:
    sample_rate = 10

    def __init__(self) -> None:
        self.model = type("FakeCodecModel", (), {"hop_length": 2})()

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        audio = latent.float().sum(dim=-1).repeat_interleave(self.model.hop_length, dim=1)
        return audio.unsqueeze(1)


class FakeSpeakerSimilarityEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []

    def cosine_similarity(
        self,
        *,
        generated_audio: torch.Tensor,
        target_audio: torch.Tensor,
        sample_rate: int,
    ) -> float:
        self.calls.append((tuple(generated_audio.shape), tuple(target_audio.shape), sample_rate))
        return 0.75


def test_run_validation_accumulates_limited_speaker_similarity(monkeypatch) -> None:
    cfg = make_config()
    model = TextToLatentRFDiT(cfg)
    model.enable_speaker_inversion(num_tokens=2, init_std=0.01)
    calls: list[dict] = []

    def fake_sample_euler_rf_cfg(**kwargs):
        calls.append(kwargs)
        return torch.zeros(
            (
                kwargs["text_input_ids"].shape[0],
                kwargs["sequence_length"],
                cfg.patched_latent_dim,
            ),
            dtype=torch.float32,
        )

    monkeypatch.setattr(train, "sample_euler_rf_cfg", fake_sample_euler_rf_cfg)
    batch = {
        "text_ids": torch.ones((2, 5), dtype=torch.long),
        "text_mask": torch.ones((2, 5), dtype=torch.bool),
        "duration_features": torch.zeros((2, cfg.duration_aux_dim), dtype=torch.float32),
        "num_frames": torch.tensor([4, 4], dtype=torch.long),
        "latent": torch.ones((2, 4, cfg.latent_dim), dtype=torch.float32),
        "latent_patched": torch.ones((2, 4, cfg.patched_latent_dim), dtype=torch.float32),
        "latent_mask_patched": torch.ones((2, 4), dtype=torch.bool),
        "latent_mask_valid_patched": torch.ones((2, 4), dtype=torch.bool),
    }
    evaluator = FakeSpeakerSimilarityEvaluator()

    metrics = train.run_validation(
        model=model,
        sampling_model=model,
        loader=[batch],
        train_cfg=TrainConfig(
            speaker_inversion_enabled=True,
            speaker_similarity_valid_samples=1,
            speaker_similarity_num_steps=3,
        ),
        device=torch.device("cpu"),
        use_bf16=False,
        distributed=False,
        speaker_similarity_codec=FakeCodec(),
        speaker_similarity_evaluator=evaluator,
    )

    assert metrics["speaker_similarity"] == 0.75
    assert metrics["speaker_similarity_samples"] == 1.0
    assert len(evaluator.calls) == 1
    assert len(calls) == 1
    assert calls[0]["num_steps"] == 3
    assert calls[0]["sequence_length"] == 4


def test_gradio_speaker_embedding_path_resolution_is_exclusive() -> None:
    assert gradio_app._resolve_speaker_embedding(None, "") is None
    assert gradio_app._resolve_speaker_embedding(None, " speaker.pt ") == "speaker.pt"
    assert gradio_app._resolve_speaker_embedding({"path": "uploaded.pt"}, "") == "uploaded.pt"
    with pytest.raises(ValueError, match="either speaker embedding upload"):
        gradio_app._resolve_speaker_embedding("uploaded.pt", "speaker.pt")
