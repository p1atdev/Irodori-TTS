from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated, Literal

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from irodori_tts.fastapi_server import (
    FIXED_SECONDS,
    MAX_API_CANDIDATES,
    SYNTHESIS_RESPONSES,
    HealthResponse,
    build_health_response,
    build_runtime_key_from_args,
    build_server_parser,
    configure_application_logging,
    create_api_app,
    create_synthesis_response,
    create_temp_work_dir,
    runtime_lifespan,
)
from irodori_tts.inference_runtime import SamplingRequest

logger = logging.getLogger("irodori_tts.server")
logger.setLevel(logging.INFO)


class BaseRequest(BaseModel):
    text: str = Field(..., min_length=1)
    seconds: float = Field(default=FIXED_SECONDS, gt=0.0)
    num_steps: int = Field(default=40, ge=1)
    num_candidates: int = Field(default=1, ge=1, le=MAX_API_CANDIDATES)
    decode_mode: Literal["sequential", "batch"] = "sequential"
    seed: int | None = None
    cfg_guidance_mode: Literal["independent", "joint", "alternating"] = "independent"
    cfg_scale_text: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_scale: float | None = None
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    context_kv_cache: bool = True
    truncation_factor: float | None = None
    rescale_k: float | None = None
    rescale_sigma: float | None = None
    speaker_kv_scale: float | None = None
    speaker_kv_min_t: float = 0.9
    speaker_kv_max_layers: int | None = Field(default=None, gt=0)
    trim_tail: bool = True


def build_base_request(
    text: Annotated[str, Form(..., min_length=1)],
    seconds: Annotated[float, Form(gt=0.0)] = FIXED_SECONDS,
    num_steps: Annotated[int, Form(ge=1)] = 40,
    num_candidates: Annotated[int, Form(ge=1, le=MAX_API_CANDIDATES)] = 1,
    decode_mode: Annotated[Literal["sequential", "batch"], Form()] = "sequential",
    seed: Annotated[int | None, Form()] = None,
    cfg_guidance_mode: Annotated[Literal["independent", "joint", "alternating"], Form()] = (
        "independent"
    ),
    cfg_scale_text: Annotated[float, Form()] = 3.0,
    cfg_scale_speaker: Annotated[float, Form()] = 5.0,
    cfg_scale: Annotated[float | None, Form()] = None,
    cfg_min_t: Annotated[float, Form()] = 0.5,
    cfg_max_t: Annotated[float, Form()] = 1.0,
    context_kv_cache: Annotated[bool, Form()] = True,
    truncation_factor: Annotated[float | None, Form()] = None,
    rescale_k: Annotated[float | None, Form()] = None,
    rescale_sigma: Annotated[float | None, Form()] = None,
    speaker_kv_scale: Annotated[float | None, Form()] = None,
    speaker_kv_min_t: Annotated[float, Form()] = 0.9,
    speaker_kv_max_layers: Annotated[int | None, Form(gt=0)] = None,
    trim_tail: Annotated[bool, Form()] = True,
) -> BaseRequest:
    return BaseRequest(
        text=text,
        seconds=seconds,
        num_steps=num_steps,
        num_candidates=num_candidates,
        decode_mode=decode_mode,
        seed=seed,
        cfg_guidance_mode=cfg_guidance_mode,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_speaker=cfg_scale_speaker,
        cfg_scale=cfg_scale,
        cfg_min_t=cfg_min_t,
        cfg_max_t=cfg_max_t,
        context_kv_cache=context_kv_cache,
        truncation_factor=truncation_factor,
        rescale_k=rescale_k,
        rescale_sigma=rescale_sigma,
        speaker_kv_scale=speaker_kv_scale,
        speaker_kv_min_t=speaker_kv_min_t,
        speaker_kv_max_layers=speaker_kv_max_layers,
        trim_tail=trim_tail,
    )


def create_app(args) -> FastAPI:
    configure_application_logging(args.log_level)
    runtime_key = build_runtime_key_from_args(args)
    app = create_api_app(
        title="Irodori-TTS Standard Server",
        description=(
            "standard text-to-speech server with optional reference audio. "
            "The model is loaded once at server startup."
        ),
        lifespan=runtime_lifespan(
            runtime_key,
            require_speaker_condition=True,
            forbid_caption_condition=True,
            forbid_character_condition=True,
        ),
    )

    @app.get("/health", tags=["system"], summary="Get server and loaded runtime status")
    async def health() -> HealthResponse:
        return build_health_response(mode="standard", runtime=app.state.runtime)

    @app.post(
        "/synthesize",
        tags=["synthesis"],
        summary="Synthesize speech from text and optional reference audio",
        response_class=Response,
        responses=SYNTHESIS_RESPONSES,
    )
    async def synthesize(
        payload: Annotated[BaseRequest, Depends(build_base_request)],
        ref_audio: Annotated[UploadFile | None, File()] = None,
    ):
        if payload.text.strip() == "":
            raise HTTPException(status_code=400, detail="text is required.")

        work_dir = create_temp_work_dir("irodori_tts_standard_")
        try:
            ref_audio_path: str | None = None
            if ref_audio is not None:
                suffix = Path(ref_audio.filename or "reference.wav").suffix
                upload_path = work_dir / f"reference_audio{suffix or '.wav'}"
                upload_path.write_bytes(await ref_audio.read())
                ref_audio_path = str(upload_path)

            logger.info(
                "standard request: candidates=%s steps=%s seed=%s ref_audio=%s",
                payload.num_candidates,
                payload.num_steps,
                "random" if payload.seed is None else payload.seed,
                "on" if ref_audio_path is not None else "off",
            )
            result = app.state.runtime.synthesize(
                SamplingRequest(
                    text=payload.text,
                    caption=None,
                    character_image=None,
                    ref_wav=ref_audio_path,
                    ref_latent=None,
                    no_ref=ref_audio_path is None,
                    ref_normalize_db=-16.0,
                    ref_ensure_max=True,
                    num_candidates=payload.num_candidates,
                    decode_mode=payload.decode_mode,
                    seconds=payload.seconds,
                    max_ref_seconds=30.0,
                    max_text_len=None,
                    max_caption_len=None,
                    num_steps=payload.num_steps,
                    seed=payload.seed,
                    cfg_guidance_mode=payload.cfg_guidance_mode,
                    cfg_scale_text=payload.cfg_scale_text,
                    cfg_scale_caption=0.0,
                    cfg_scale_character=0.0,
                    cfg_scale_speaker=payload.cfg_scale_speaker,
                    cfg_scale=payload.cfg_scale,
                    cfg_min_t=payload.cfg_min_t,
                    cfg_max_t=payload.cfg_max_t,
                    truncation_factor=payload.truncation_factor,
                    rescale_k=payload.rescale_k,
                    rescale_sigma=payload.rescale_sigma,
                    context_kv_cache=payload.context_kv_cache,
                    speaker_kv_scale=payload.speaker_kv_scale,
                    speaker_kv_min_t=None
                    if payload.speaker_kv_scale is None
                    else payload.speaker_kv_min_t,
                    speaker_kv_max_layers=payload.speaker_kv_max_layers,
                    trim_tail=payload.trim_tail,
                ),
                log_fn=logger.info,
            )
            return create_synthesis_response(
                result=result,
                work_dir=work_dir,
                base_name="irodori_tts_standard",
            )
        except ValueError as exc:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise
        finally:
            if ref_audio is not None:
                await ref_audio.close()

    return app


def main() -> None:
    parser = build_server_parser(
        description="FastAPI server for standard Irodori-TTS checkpoints.",
        default_port=8000,
    )
    args = parser.parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
