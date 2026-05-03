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

logger = logging.getLogger("irodori_tts.server_character")
logger.setLevel(logging.INFO)


class CharacterRequest(BaseModel):
    text: str = Field(..., min_length=1)
    seconds: float = Field(default=FIXED_SECONDS, gt=0.0)
    num_steps: int = Field(default=40, ge=1)
    num_candidates: int = Field(default=1, ge=1, le=MAX_API_CANDIDATES)
    decode_mode: Literal["sequential", "batch"] = "sequential"
    seed: int | None = None
    cfg_guidance_mode: Literal["independent", "joint", "alternating"] = "independent"
    cfg_scale_text: float = 2.0
    cfg_scale_character: float = 3.0
    cfg_scale: float | None = None
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    context_kv_cache: bool = True
    max_text_len: int | None = Field()
    truncation_factor: float | None = None
    rescale_k: float | None = None
    rescale_sigma: float | None = None
    trim_tail: bool = True


def build_character_request(
    text: Annotated[str, Form(..., min_length=1)],
    seconds: Annotated[float, Form(gt=0.0)] = FIXED_SECONDS,
    num_steps: Annotated[int, Form(ge=1)] = 40,
    num_candidates: Annotated[int, Form(ge=1, le=MAX_API_CANDIDATES)] = 1,
    decode_mode: Annotated[Literal["sequential", "batch"], Form()] = "sequential",
    seed: Annotated[int | None, Form()] = None,
    cfg_guidance_mode: Annotated[Literal["independent", "joint", "alternating"], Form()] = (
        "independent"
    ),
    cfg_scale_text: Annotated[float, Form()] = 2.0,
    cfg_scale_character: Annotated[float, Form()] = 3.0,
    cfg_scale: Annotated[float | None, Form()] = None,
    cfg_min_t: Annotated[float, Form()] = 0.5,
    cfg_max_t: Annotated[float, Form()] = 1.0,
    context_kv_cache: Annotated[bool, Form()] = True,
    max_text_len: Annotated[int | None, Form()] = None,
    truncation_factor: Annotated[float | None, Form()] = None,
    rescale_k: Annotated[float | None, Form()] = None,
    rescale_sigma: Annotated[float | None, Form()] = None,
    trim_tail: Annotated[bool, Form()] = True,
) -> CharacterRequest:
    return CharacterRequest(
        text=text,
        seconds=seconds,
        num_steps=num_steps,
        num_candidates=num_candidates,
        decode_mode=decode_mode,
        seed=seed,
        cfg_guidance_mode=cfg_guidance_mode,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_character=cfg_scale_character,
        cfg_scale=cfg_scale,
        cfg_min_t=cfg_min_t,
        cfg_max_t=cfg_max_t,
        context_kv_cache=context_kv_cache,
        max_text_len=max_text_len,
        truncation_factor=truncation_factor,
        rescale_k=rescale_k,
        rescale_sigma=rescale_sigma,
        trim_tail=trim_tail,
    )


def create_app(args) -> FastAPI:
    configure_application_logging(args.log_level)
    runtime_key = build_runtime_key_from_args(args)
    app = create_api_app(
        title="Irodori-TTS Character Reference Server",
        description=(
            "image + text conditioning for Character Reference checkpoints. "
            "The model is loaded once at server startup."
        ),
        lifespan=runtime_lifespan(runtime_key, require_character_condition=True),
    )

    @app.get("/health", tags=["system"], summary="Get server and loaded runtime status")
    async def health() -> HealthResponse:
        return build_health_response(mode="character", runtime=app.state.runtime)

    @app.post(
        "/synthesize",
        tags=["synthesis"],
        summary="Synthesize speech from text and optional character reference image",
        response_class=Response,
        responses=SYNTHESIS_RESPONSES,
    )
    async def synthesize(
        payload: Annotated[CharacterRequest, Depends(build_character_request)],
        character_image: Annotated[UploadFile | None, File()] = None,
    ):
        if payload.text.strip() == "":
            raise HTTPException(status_code=400, detail="text is required.")

        work_dir = create_temp_work_dir("irodori_tts_character_")
        try:
            character_image_path: str | None = None
            if character_image is not None:
                suffix = Path(character_image.filename or "character_reference.png").suffix
                upload_path = work_dir / f"character_reference{suffix or '.png'}"
                upload_path.write_bytes(await character_image.read())
                character_image_path = str(upload_path)

            logger.info(
                (
                    "character request: seconds=%.3f decode_mode=%s trim_tail=%s "
                    "candidates=%s steps=%s seed=%s image=%s"
                    "text=%s"
                ),
                payload.seconds,
                payload.decode_mode,
                payload.trim_tail,
                payload.num_candidates,
                payload.num_steps,
                "random" if payload.seed is None else payload.seed,
                "on" if character_image_path is not None else "off",
                payload.text,
            )
            result = app.state.runtime.synthesize(
                SamplingRequest(
                    text=payload.text,
                    caption=None,
                    character_image=character_image_path,
                    ref_wav=None,
                    ref_latent=None,
                    no_ref=True,
                    ref_normalize_db=-16.0,
                    ref_ensure_max=True,
                    num_candidates=payload.num_candidates,
                    decode_mode=payload.decode_mode,
                    seconds=payload.seconds,
                    max_ref_seconds=30.0,
                    max_text_len=payload.max_text_len,
                    num_steps=payload.num_steps,
                    seed=payload.seed,
                    cfg_guidance_mode=payload.cfg_guidance_mode,
                    cfg_scale_text=payload.cfg_scale_text,
                    cfg_scale_caption=0.0,
                    cfg_scale_character=payload.cfg_scale_character,
                    cfg_scale_speaker=0.0,
                    cfg_scale=payload.cfg_scale,
                    cfg_min_t=payload.cfg_min_t,
                    cfg_max_t=payload.cfg_max_t,
                    truncation_factor=payload.truncation_factor,
                    rescale_k=payload.rescale_k,
                    rescale_sigma=payload.rescale_sigma,
                    context_kv_cache=payload.context_kv_cache,
                    speaker_kv_scale=None,
                    speaker_kv_min_t=None,
                    speaker_kv_max_layers=None,
                    trim_tail=payload.trim_tail,
                ),
                log_fn=logger.info,
            )
            logger.info(
                "character result: seed=%s samples=%s sample_rate=%s audio_seconds=%.3f",
                result.used_seed,
                int(result.audio.shape[-1]),
                result.sample_rate,
                float(result.audio.shape[-1]) / float(result.sample_rate),
            )
            return create_synthesis_response(
                result=result,
                work_dir=work_dir,
                base_name="irodori_tts_character",
            )
        except ValueError as exc:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise
        finally:
            if character_image is not None:
                await character_image.close()

    return app


def main() -> None:
    parser = build_server_parser(
        description="FastAPI server for Irodori-TTS character-reference checkpoints.",
        default_port=8002,
    )
    args = parser.parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
