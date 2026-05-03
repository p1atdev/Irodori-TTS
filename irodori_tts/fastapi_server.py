from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import FastAPI
from fastapi.responses import Response
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from scalar_fastapi import AgentScalarConfig, get_scalar_api_reference

from .inference_runtime import (
    RuntimeKey,
    SamplingResult,
    clear_cached_runtime,
    default_runtime_device,
    get_cached_runtime,
    save_wav,
)

FIXED_SECONDS = 30.0
MAX_API_CANDIDATES = 32
DEFAULT_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
OPENAPI_TAGS = [
    {
        "name": "system",
        "description": "Server status, loaded checkpoint, and runtime configuration.",
    },
    {
        "name": "synthesis",
        "description": "Speech synthesis endpoints for text and optional conditioning inputs.",
    },
]
DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
SYNTHESIS_RESPONSES = {
    200: {
        "description": (
            "Generated audio. Returns `audio/wav` for a single candidate, or "
            "`application/zip` with `metadata.json` and WAV files when multiple candidates are requested."
        ),
        "content": {
            "audio/wav": {"schema": {"type": "string", "format": "binary"}},
            "application/zip": {"schema": {"type": "string", "format": "binary"}},
        },
    }
}


class HealthResponse(BaseModel):
    status: str
    mode: str
    checkpoint: str
    model_device: str
    model_precision: str
    codec_device: str
    codec_precision: str
    use_caption_condition: bool
    use_character_condition: bool
    use_speaker_condition: bool
    default_text_max_len: int
    default_caption_max_len: int


def build_server_parser(*, description: str, default_port: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint",
        default=None,
        help="Local checkpoint path (.pt or .safetensors).",
    )
    checkpoint_group.add_argument(
        "--hf-checkpoint",
        default=None,
        help="Hugging Face repo id to download model.safetensors from.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        default="info",
    )
    parser.add_argument(
        "--model-device",
        default=default_runtime_device(),
        help="Model inference device (e.g. cuda, mps, cpu).",
    )
    parser.add_argument(
        "--model-precision",
        choices=["fp32", "bf16"],
        default="bf16",
        help="Model precision for weights/compute.",
    )
    parser.add_argument(
        "--codec-device",
        default=default_runtime_device(),
        help="Codec device for decode (e.g. cuda, mps, cpu).",
    )
    parser.add_argument(
        "--codec-precision",
        choices=["fp32", "bf16"],
        default="bf16",
        help="Codec precision for weights/compute.",
    )
    parser.add_argument("--codec-repo", default=DEFAULT_CODEC_REPO)
    parser.add_argument(
        "--enable-watermark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable DACVAE watermark branch during decode (default: disabled).",
    )
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.compile for core inference methods (default: disabled).",
    )
    parser.add_argument(
        "--compile-dynamic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use dynamic=True for torch.compile (default: disabled).",
    )
    return parser


def resolve_server_checkpoint_path(*, checkpoint: str | None, hf_checkpoint: str | None) -> str:
    if checkpoint is not None:
        checkpoint_path = Path(str(checkpoint)).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return str(checkpoint_path)

    if hf_checkpoint is None:
        raise ValueError("Either checkpoint or hf_checkpoint must be set.")

    repo_id = str(hf_checkpoint).strip()
    if repo_id == "":
        raise ValueError("hf_checkpoint must be non-empty.")

    resolved = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    return str(resolved)


def build_runtime_key_from_args(args: argparse.Namespace) -> RuntimeKey:
    checkpoint_path = resolve_server_checkpoint_path(
        checkpoint=args.checkpoint,
        hf_checkpoint=args.hf_checkpoint,
    )
    return RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=str(args.model_device),
        codec_repo=str(args.codec_repo),
        model_precision=str(args.model_precision),
        codec_device=str(args.codec_device),
        codec_precision=str(args.codec_precision),
        enable_watermark=bool(args.enable_watermark),
        compile_model=bool(args.compile_model),
        compile_dynamic=bool(args.compile_dynamic),
    )


def create_api_app(*, title: str, description: str, lifespan) -> FastAPI:
    app = FastAPI(
        title=title,
        description=description,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_tags=OPENAPI_TAGS,
    )
    install_scalar(app=app, title=title)
    return app


def configure_application_logging(log_level: str) -> None:
    level_name = str(log_level).strip().upper()
    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        root_logger.addHandler(handler)
    root_logger.setLevel(level)
    logging.getLogger("irodori_tts").setLevel(level)


def runtime_lifespan(
    runtime_key: RuntimeKey,
    *,
    require_caption_condition: bool = False,
    require_character_condition: bool = False,
    require_speaker_condition: bool = False,
    forbid_caption_condition: bool = False,
    forbid_character_condition: bool = False,
):
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        runtime, _ = get_cached_runtime(runtime_key)
        _validate_runtime_mode(
            runtime,
            require_caption_condition=require_caption_condition,
            require_character_condition=require_character_condition,
            require_speaker_condition=require_speaker_condition,
            forbid_caption_condition=forbid_caption_condition,
            forbid_character_condition=forbid_character_condition,
        )
        app.state.runtime = runtime
        app.state.runtime_key = runtime_key
        try:
            yield
        finally:
            clear_cached_runtime()

    return _lifespan


def runtime_summary(runtime) -> dict[str, object]:
    return {
        "checkpoint": runtime.key.checkpoint,
        "model_device": runtime.key.model_device,
        "model_precision": runtime.key.model_precision,
        "codec_device": runtime.key.codec_device,
        "codec_precision": runtime.key.codec_precision,
        "use_caption_condition": bool(runtime.model_cfg.use_caption_condition),
        "use_character_condition": bool(runtime.model_cfg.use_character_condition),
        "use_speaker_condition": bool(runtime.model_cfg.use_speaker_condition),
        "default_text_max_len": int(runtime.default_text_max_len),
        "default_caption_max_len": int(runtime.default_caption_max_len),
    }


def build_health_response(*, mode: str, runtime) -> HealthResponse:
    return HealthResponse(status="ok", mode=mode, **runtime_summary(runtime))


def install_scalar(app: FastAPI, *, title: str) -> None:
    @app.get("/docs", include_in_schema=False)
    @app.get("/scalar", include_in_schema=False)
    async def scalar_docs():
        return get_scalar_api_reference(
            title=f"{title} - Scalar",
            openapi_url=app.openapi_url,
            scalar_proxy_url="https://proxy.scalar.com",
            default_open_all_tags=True,
            hide_download_button=False,
            hide_test_request_button=False,
            show_sidebar=True,
            agent=AgentScalarConfig(disabled=True),
            telemetry=False,
        )


def create_temp_work_dir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def create_synthesis_response(
    *,
    result: SamplingResult,
    work_dir: Path,
    base_name: str,
) -> Response:
    metadata = {
        "used_seed": int(result.used_seed),
        "sample_rate": int(result.sample_rate),
        "num_candidates": int(len(result.audios)),
        "messages": list(result.messages),
        "stage_timings": [
            {"name": name, "seconds": float(seconds)} for name, seconds in result.stage_timings
        ],
        "total_to_decode": float(result.total_to_decode),
    }

    if len(result.audios) == 1:
        wav_path = save_wav(
            work_dir / f"{base_name}.wav",
            result.audio.float(),
            result.sample_rate,
        )
        content = wav_path.read_bytes()
        media_type = "audio/wav"
        filename = f"{base_name}.wav"
    else:
        wav_paths: list[Path] = []
        for index, audio in enumerate(result.audios, start=1):
            wav_paths.append(
                save_wav(
                    work_dir / f"{base_name}_{index:03d}.wav",
                    audio.float(),
                    result.sample_rate,
                )
            )
        metadata_path = work_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        archive_path = work_dir / f"{base_name}.zip"
        with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
            archive.write(metadata_path, arcname=metadata_path.name)
            for wav_path in wav_paths:
                archive.write(wav_path, arcname=wav_path.name)
        content = archive_path.read_bytes()
        media_type = "application/zip"
        filename = f"{base_name}.zip"

    response = Response(content=content, media_type=media_type)
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    shutil.rmtree(work_dir, ignore_errors=True)

    response.headers["X-Irodori-Seed"] = str(result.used_seed)
    response.headers["X-Irodori-Sample-Rate"] = str(result.sample_rate)
    response.headers["X-Irodori-Candidates"] = str(len(result.audios))
    response.headers["X-Irodori-Total-To-Decode"] = f"{result.total_to_decode:.6f}"
    return response


def _validate_runtime_mode(
    runtime,
    *,
    require_caption_condition: bool,
    require_character_condition: bool,
    require_speaker_condition: bool,
    forbid_caption_condition: bool,
    forbid_character_condition: bool,
) -> None:
    if require_caption_condition and not runtime.model_cfg.use_caption_condition:
        raise ValueError(
            "Loaded checkpoint does not enable caption conditioning. "
            "Use a VoiceDesign checkpoint for this server."
        )
    if require_character_condition and not runtime.model_cfg.use_character_condition:
        raise ValueError(
            "Loaded checkpoint does not enable character conditioning. "
            "Use a Character Reference checkpoint for this server."
        )
    if require_speaker_condition and not runtime.model_cfg.use_speaker_condition:
        raise ValueError(
            "Loaded checkpoint does not enable speaker conditioning. "
            "Use a standard checkpoint for this server."
        )
    if forbid_caption_condition and runtime.model_cfg.use_caption_condition:
        raise ValueError(
            "Loaded checkpoint enables caption conditioning. "
            "Use server_voice_design.py for VoiceDesign checkpoints."
        )
    if forbid_character_condition and runtime.model_cfg.use_character_condition:
        raise ValueError(
            "Loaded checkpoint enables character conditioning. "
            "Use server_character.py for Character Reference checkpoints."
        )
