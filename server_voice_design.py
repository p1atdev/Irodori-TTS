from __future__ import annotations

import logging
import shutil
from typing import Annotated, Literal

import uvicorn
from fastapi import Depends, FastAPI, Form, HTTPException
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

logger = logging.getLogger("irodori_tts.server_voice_design")
logger.setLevel(logging.INFO)


class VoiceDesignRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Speech text to synthesize.",
        examples=["こんにちは。今日は落ち着いた雰囲気で案内します。"],
    )
    caption: str | None = Field(
        default=None,
        description="Optional voice-design caption. If omitted or blank, text-only conditioning is used.",
        examples=["落ち着いた、近い距離感の女性話者"],
    )
    seconds: float = Field(
        default=FIXED_SECONDS, gt=0.0, description="Target output duration in seconds."
    )
    num_steps: int = Field(default=40, ge=1, description="Number of diffusion sampling steps.")
    num_candidates: int = Field(
        default=1,
        ge=1,
        le=MAX_API_CANDIDATES,
        description="Number of candidates to generate. Returns ZIP when greater than 1.",
    )
    decode_mode: Literal["sequential", "batch"] = Field(
        default="sequential",
        description="Codec decode mode. `batch` is faster but uses more memory.",
    )
    seed: int | None = Field(
        default=None,
        description="Sampling seed. If omitted, a random seed is generated.",
    )
    cfg_guidance_mode: Literal["independent", "joint", "alternating"] = Field(
        default="independent",
        description="Classifier-free guidance strategy.",
    )
    cfg_scale_text: float = Field(default=2.0, description="CFG scale for text conditioning.")
    cfg_scale_caption: float = Field(default=4.0, description="CFG scale for caption conditioning.")
    cfg_scale: float | None = Field(
        default=None,
        description="Optional shared CFG override for all enabled conditions.",
    )
    cfg_min_t: float = Field(default=0.5, description="Lower timestep bound where CFG is applied.")
    cfg_max_t: float = Field(default=1.0, description="Upper timestep bound where CFG is applied.")
    context_kv_cache: bool = Field(
        default=True,
        description="Enable cached context K/V projections for faster sampling.",
    )
    max_text_len: int | None = Field(
        default=None,
        description="Optional maximum token length for text conditioning.",
    )
    max_caption_len: int | None = Field(
        default=None,
        description="Optional maximum token length for caption conditioning.",
    )
    truncation_factor: float | None = Field(
        default=None,
        description="Optional scale for initial Gaussian noise before sampling.",
    )
    rescale_k: float | None = Field(
        default=None,
        description="Optional temporal score rescaling parameter k.",
    )
    rescale_sigma: float | None = Field(
        default=None,
        description="Optional temporal score rescaling parameter sigma.",
    )
    trim_tail: bool = Field(
        default=True,
        description="Trim trailing near-zero latent region after decoding.",
    )


def build_voice_design_request(
    text: Annotated[str, Form(..., min_length=1)],
    caption: Annotated[str | None, Form()] = None,
    seconds: Annotated[float, Form(gt=0.0)] = FIXED_SECONDS,
    num_steps: Annotated[int, Form(ge=1)] = 40,
    num_candidates: Annotated[int, Form(ge=1, le=MAX_API_CANDIDATES)] = 1,
    decode_mode: Annotated[Literal["sequential", "batch"], Form()] = "sequential",
    seed: Annotated[int | None, Form()] = None,
    cfg_guidance_mode: Annotated[
        Literal["independent", "joint", "alternating"], Form()
    ] = "independent",
    cfg_scale_text: Annotated[float, Form()] = 2.0,
    cfg_scale_caption: Annotated[float, Form()] = 4.0,
    cfg_scale: Annotated[float | None, Form()] = None,
    cfg_min_t: Annotated[float, Form()] = 0.5,
    cfg_max_t: Annotated[float, Form()] = 1.0,
    context_kv_cache: Annotated[bool, Form()] = True,
    max_text_len: Annotated[int | None, Form(gt=0)] = None,
    max_caption_len: Annotated[int | None, Form(gt=0)] = None,
    truncation_factor: Annotated[float | None, Form()] = None,
    rescale_k: Annotated[float | None, Form()] = None,
    rescale_sigma: Annotated[float | None, Form()] = None,
    trim_tail: Annotated[bool, Form()] = True,
) -> VoiceDesignRequest:
    return VoiceDesignRequest(
        text=text,
        caption=caption,
        seconds=seconds,
        num_steps=num_steps,
        num_candidates=num_candidates,
        decode_mode=decode_mode,
        seed=seed,
        cfg_guidance_mode=cfg_guidance_mode,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_caption=cfg_scale_caption,
        cfg_scale=cfg_scale,
        cfg_min_t=cfg_min_t,
        cfg_max_t=cfg_max_t,
        context_kv_cache=context_kv_cache,
        max_text_len=max_text_len,
        max_caption_len=max_caption_len,
        truncation_factor=truncation_factor,
        rescale_k=rescale_k,
        rescale_sigma=rescale_sigma,
        trim_tail=trim_tail,
    )


def create_app(args) -> FastAPI:
    configure_application_logging(args.log_level)
    runtime_key = build_runtime_key_from_args(args)
    app = create_api_app(
        title="Irodori-TTS Voice Design Server",
        description=(
            "caption + text conditioning for VoiceDesign checkpoints. "
            "The model is loaded once at server startup."
        ),
        lifespan=runtime_lifespan(runtime_key, require_caption_condition=True),
    )

    @app.get("/health", tags=["system"], summary="Get server and loaded runtime status")
    async def health() -> HealthResponse:
        return build_health_response(mode="voice_design", runtime=app.state.runtime)

    @app.post(
        "/synthesize",
        tags=["synthesis"],
        summary="Synthesize speech from text and voice-design caption",
        response_class=Response,
        responses=SYNTHESIS_RESPONSES,
    )
    async def synthesize(
        payload: Annotated[VoiceDesignRequest, Depends(build_voice_design_request)],
    ):
        if payload.text.strip() == "":
            raise HTTPException(status_code=400, detail="text is required.")

        work_dir = create_temp_work_dir("irodori_tts_voice_design_")
        try:
            logger.info(
                (
                    "voice_design request: seconds=%.3f decode_mode=%s trim_tail=%s "
                    "candidates=%s steps=%s seed=%s caption=%s"
                ),
                payload.seconds,
                payload.decode_mode,
                payload.trim_tail,
                payload.num_candidates,
                payload.num_steps,
                "random" if payload.seed is None else payload.seed,
                "on" if payload.caption and payload.caption.strip() else "off",
            )
            result = app.state.runtime.synthesize(
                SamplingRequest(
                    text=payload.text,
                    caption=payload.caption,
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
                    max_caption_len=payload.max_caption_len,
                    num_steps=payload.num_steps,
                    seed=payload.seed,
                    cfg_guidance_mode=payload.cfg_guidance_mode,
                    cfg_scale_text=payload.cfg_scale_text,
                    cfg_scale_caption=payload.cfg_scale_caption,
                    cfg_scale_character=0.0,
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
                "voice_design result: seed=%s samples=%s sample_rate=%s audio_seconds=%.3f",
                result.used_seed,
                int(result.audio.shape[-1]),
                result.sample_rate,
                float(result.audio.shape[-1]) / float(result.sample_rate),
            )
            return create_synthesis_response(
                result=result,
                work_dir=work_dir,
                base_name="irodori_tts_voice_design",
            )
        except ValueError as exc:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise

    return app


def main() -> None:
    parser = build_server_parser(
        description="FastAPI server for Irodori-TTS VoiceDesign checkpoints.",
        default_port=8001,
    )
    args = parser.parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
