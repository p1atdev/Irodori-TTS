# ruff: noqa: E402
import io
import json
import logging
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from irodori_tts.fastapi_server import (
    build_runtime_key_from_args,
    configure_application_logging,
    create_synthesis_response,
)
from irodori_tts.generation_core import SamplingResult


def _make_result(*, num_candidates: int = 1) -> SamplingResult:
    audios = [torch.zeros(1, 160, dtype=torch.float32) + float(i) for i in range(num_candidates)]
    return SamplingResult(
        audio=audios[0],
        audios=audios,
        sample_rate=16000,
        stage_timings=[("decode", 0.01)],
        total_to_decode=0.02,
        used_seed=123,
        messages=["info: test"],
    )


def test_configure_application_logging_adds_root_handler_and_sets_levels() -> None:
    root_logger = logging.getLogger()
    irodori_logger = logging.getLogger("irodori_tts")
    old_handlers = list(root_logger.handlers)
    old_root_level = root_logger.level
    old_irodori_level = irodori_logger.level

    try:
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)
        irodori_logger.setLevel(logging.WARNING)

        configure_application_logging("info")

        assert root_logger.handlers
        assert root_logger.level == logging.INFO
        assert irodori_logger.level == logging.INFO
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(old_handlers)
        root_logger.setLevel(old_root_level)
        irodori_logger.setLevel(old_irodori_level)


def test_build_runtime_key_ignores_legacy_watermark_arg(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"")
    args = SimpleNamespace(
        checkpoint=str(checkpoint),
        hf_checkpoint=None,
        model_device="cpu",
        codec_repo="test/codec",
        model_precision="fp32",
        codec_device="cpu",
        codec_precision="fp32",
        enable_watermark=True,
        compile_model=False,
        compile_dynamic=False,
    )

    key = build_runtime_key_from_args(args)

    assert key.checkpoint == str(checkpoint)
    assert key.model_device == "cpu"
    assert key.codec_repo == "test/codec"
    assert not hasattr(key, "enable_watermark")


def test_create_synthesis_response_single_candidate_returns_complete_wav_bytes(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "single"
    work_dir.mkdir()

    response = create_synthesis_response(
        result=_make_result(num_candidates=1),
        work_dir=work_dir,
        base_name="sample",
    )

    assert response.media_type == "audio/wav"
    assert response.headers["Content-Disposition"] == 'attachment; filename="sample.wav"'
    assert response.headers["X-Irodori-Seed"] == "123"
    assert response.body.startswith(b"RIFF")
    assert not work_dir.exists()


def test_create_synthesis_response_multiple_candidates_returns_zip_bytes(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "multi"
    work_dir.mkdir()

    response = create_synthesis_response(
        result=_make_result(num_candidates=2),
        work_dir=work_dir,
        base_name="sample",
    )

    assert response.media_type == "application/zip"
    assert response.headers["Content-Disposition"] == 'attachment; filename="sample.zip"'
    assert response.body.startswith(b"PK")

    with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
        assert sorted(archive.namelist()) == [
            "metadata.json",
            "sample_001.wav",
            "sample_002.wav",
        ]
        metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
        assert metadata["used_seed"] == 123
        assert metadata["num_candidates"] == 2

    assert not work_dir.exists()
