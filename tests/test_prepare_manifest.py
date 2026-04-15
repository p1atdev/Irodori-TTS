# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import prepare_manifest as pm


class DummyDataset:
    def __init__(self, rows: list[dict]):
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[idx]

    def cast_column(self, name: str, feature=None):
        return self


class DummyCodec:
    def encode_waveform(self, wav: torch.Tensor, sample_rate: int):
        del wav, sample_rate
        return [torch.zeros(4, 32)]


def make_args(tmp_path: Path, *, image_column: str | None = "image_file") -> argparse.Namespace:
    return argparse.Namespace(
        dataset="dummy_dataset",
        config=None,
        split="train",
        data_files=None,
        audio_column="audio",
        text_column="text",
        text_normalize=False,
        caption_column=None,
        image_column=image_column,
        speaker_column=None,
        speaker_columns=[],
        speaker_id_prefix=None,
        speaker_id_namespace="dummy_dataset",
        output_manifest=str(tmp_path / "manifest.jsonl"),
        latent_dir=str(tmp_path / "latents"),
        codec_repo="dummy/codec",
        codec_deterministic_encode=True,
        codec_deterministic_decode=True,
        normalize_db=None,
        device="cpu",
        num_gpus=None,
        shard_strategy="auto",
        merge_output=False,
        keep_shards=False,
        streaming=False,
        target_sample_rate=None,
        min_sample_rate=0,
        max_seconds=None,
        max_samples=None,
        skip_samples=0,
        prefetch=0,
        prefetch_workers=1,
        flush_every=0,
        progress=False,
        progress_all=False,
        log_every=0,
        seed=0,
        trust_remote_code=False,
        cache_dir=None,
    )


def make_sample(*, image_value: str | None) -> dict:
    return {
        "audio": {"array": [0.0, 0.1, -0.1, 0.0], "sampling_rate": 24000},
        "text": "テストです",
        "image_file": image_value,
    }


def test_prepare_example_rejects_relative_image_path(tmp_path: Path) -> None:
    args = make_args(tmp_path)

    item = pm._prepare_example(0, make_sample(image_value="images/ref.png"), args)

    assert item.status == "skip"
    assert item.skip_reason == "invalid_image_path"
    assert item.error is not None
    assert "absolute path" in item.error


def test_prepare_example_rejects_missing_image_file(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    missing_path = tmp_path / "missing.png"

    item = pm._prepare_example(0, make_sample(image_value=str(missing_path.resolve())), args)

    assert item.status == "skip"
    assert item.skip_reason == "invalid_image_path"
    assert item.error is not None
    assert "does not exist" in item.error


def test_prepare_example_allows_empty_image_path(tmp_path: Path) -> None:
    args = make_args(tmp_path)

    item = pm._prepare_example(0, make_sample(image_value=""), args)

    assert item.status == "ok"
    assert item.image_path is None


def test_run_worker_writes_image_path_to_manifest(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "character.png"
    image_path.write_bytes(b"not-an-image-but-it-exists")

    dataset = DummyDataset([make_sample(image_value=str(image_path.resolve()))])
    monkeypatch.setattr(pm, "load_dataset", lambda **kwargs: dataset)
    monkeypatch.setattr(pm.DACVAECodec, "load", lambda **kwargs: DummyCodec())

    args = make_args(tmp_path)
    pm._run_worker(args, rank=0, world_size=1, local_rank=0)

    manifest_path = tmp_path / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 1
    assert rows[0]["image_path"] == str(image_path.resolve())
    assert any((tmp_path / "latents").glob("*.pt"))
