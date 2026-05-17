import math
import secrets
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio

from .codec import DACVAECodec, patchify_latent, unpatchify_latent
from .config import ModelConfig
from .duration import build_duration_features
from .rf import sample_euler_rf_cfg
from .speaker_inversion import load_speaker_inversion_payload, speaker_inversion_batch_tensors
from .text_normalization import normalize_text
from .tokenizer import PretrainedTextTokenizer

MeasureStart = Callable[..., float]
MeasureEnd = Callable[..., float]


def _default_measure_start(*_devices: torch.device) -> float:
    return time.perf_counter()


def _default_measure_end(t0: float, *_devices: torch.device) -> float:
    return time.perf_counter() - t0


@dataclass
class SamplingRequest:
    text: str
    caption: str | None = None
    ref_wav: str | None = None
    ref_latent: str | None = None
    speaker_embedding: str | None = None
    no_ref: bool = False
    ref_normalize_db: float | None = -16.0
    ref_ensure_max: bool = True
    num_candidates: int = 1
    decode_mode: str = "sequential"
    seconds: float | None = None
    duration_scale: float = 1.0
    min_seconds: float = 0.5
    max_seconds: float = 30.0
    max_ref_seconds: float | None = 30.0
    max_text_len: int | None = None
    max_caption_len: int | None = None
    character_image: str | None = None
    num_steps: int = 40
    cfg_scale_text: float = 3.0
    cfg_scale_caption: float = 3.0
    cfg_scale_character: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_guidance_mode: str = "independent"
    cfg_scale: float | None = None
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    truncation_factor: float | None = None
    rescale_k: float | None = None
    rescale_sigma: float | None = None
    context_kv_cache: bool = True
    speaker_kv_scale: float | None = None
    speaker_kv_min_t: float | None = None
    speaker_kv_max_layers: int | None = None
    seed: int | None = None
    t_schedule_mode: str = "linear"
    sway_coeff: float = -1.0
    trim_tail: bool = True
    tail_window_size: int = 20
    tail_std_threshold: float = 0.05
    tail_mean_threshold: float = 0.1


@dataclass
class SamplingResult:
    audio: torch.Tensor
    audios: list[torch.Tensor]
    sample_rate: int
    stage_timings: list[tuple[str, float]]
    total_to_decode: float
    used_seed: int
    messages: list[str]


def normalize_generation_text(text: str) -> str:
    return normalize_text(str(text)).strip()


def resolve_condition_lengths(
    *,
    default_text_max_len: int,
    default_caption_max_len: int,
    max_text_len: int | None,
    max_caption_len: int | None,
) -> tuple[int, int]:
    text_len = int(default_text_max_len if max_text_len is None else int(max_text_len))
    if text_len <= 0:
        raise ValueError(f"max_text_len must be > 0, got {text_len}")
    caption_len = int(default_caption_max_len if max_caption_len is None else int(max_caption_len))
    if caption_len <= 0:
        raise ValueError(f"max_caption_len must be > 0, got {caption_len}")
    return text_len, caption_len


def resolve_sequence_lengths(
    *,
    seconds: float,
    sample_rate: int,
    hop_length: int,
    latent_patch_size: int,
) -> tuple[int, int, int]:
    target_samples = int(float(seconds) * sample_rate)
    latent_steps = max(1, math.ceil(target_samples / int(hop_length)))
    patched_steps = max(1, math.ceil(latent_steps / int(latent_patch_size)))
    return target_samples, latent_steps, patched_steps


def resolve_cfg_scales(
    *,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_caption: float,
    cfg_scale_speaker: float,
    cfg_scale_character: float = 3.0,
    cfg_scale: float | None,
    use_caption_condition: bool = True,
    use_speaker_condition: bool = True,
    use_character_condition: bool = False,
) -> tuple[float, float, float, float, list[str]]:
    """Normalize/validate CFG scales for guidance mode."""
    messages: list[str] = []
    text_val = float(cfg_scale_text)
    caption_val = float(cfg_scale_caption)
    speaker_val = float(cfg_scale_speaker)
    character_val = float(cfg_scale_character)

    if cfg_scale is not None:
        text_val = float(cfg_scale)
        caption_val = float(cfg_scale)
        speaker_val = float(cfg_scale)
        character_val = float(cfg_scale)
    if not use_speaker_condition:
        if speaker_val > 0.0:
            messages.append(
                "info: speaker conditioning is disabled for this checkpoint; ignoring cfg_scale_speaker."
            )
        speaker_val = 0.0
    if not use_character_condition:
        character_val = 0.0

    mode = str(cfg_guidance_mode).strip().lower()
    enabled_vals = [value for value in (text_val, speaker_val) if value > 0.0]
    if use_caption_condition and caption_val > 0.0:
        enabled_vals.append(caption_val)
    if use_character_condition and character_val > 0.0:
        enabled_vals.append(character_val)
    if mode == "joint" and enabled_vals and (max(enabled_vals) - min(enabled_vals) > 1e-6):
        raise ValueError(
            "cfg_guidance_mode='joint' requires equal enabled cfg_scale_text/cfg_scale_caption/cfg_scale_speaker/cfg_scale_character, "
            "or set cfg_scale."
        )

    return text_val, caption_val, speaker_val, character_val, messages


def find_flattening_point(
    latent: torch.Tensor,
    target_value: float = 0.0,
    window_size: int = 20,
    std_threshold: float = 0.05,
    mean_threshold: float = 0.1,
) -> int:
    if latent.ndim != 2:
        raise ValueError(f"Expected latent shape (T, D), got {tuple(latent.shape)}")
    total_steps = int(latent.shape[0])
    if total_steps <= 0 or window_size <= 0:
        return total_steps

    pad = torch.zeros(
        (window_size, latent.shape[1]),
        device=latent.device,
        dtype=latent.dtype,
    )
    padded = torch.cat([latent, pad], dim=0)
    for i in range(padded.shape[0] - window_size):
        window = padded[i : i + window_size]
        window_std = window.std(unbiased=False)
        window_mean = window.mean()
        if window_std < std_threshold and torch.abs(window_mean - target_value) < mean_threshold:
            return int(i)
    return total_steps


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    try:
        return torchaudio.load(str(path))
    except RuntimeError:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32")
        wav = torch.from_numpy(data)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.T
        return wav, sr


def _coerce_latent_shape(latent: torch.Tensor, latent_dim: int) -> torch.Tensor:
    if latent.ndim == 3 and latent.shape[0] == 1:
        latent = latent[0]
    if latent.ndim != 2:
        raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")
    if latent.shape[1] == latent_dim:
        return latent
    if latent.shape[0] == latent_dim:
        return latent.transpose(0, 1).contiguous()
    raise ValueError(
        f"Could not infer latent layout for shape={tuple(latent.shape)} and latent_dim={latent_dim}"
    )


def _prepare_text_and_caption_tensors(
    *,
    tokenizer: PretrainedTextTokenizer,
    caption_tokenizer: PretrainedTextTokenizer | None,
    request: SamplingRequest,
    model_cfg: ModelConfig,
    model_device: torch.device,
    default_text_max_len: int,
    default_caption_max_len: int,
) -> tuple[str, str, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    normalized_text = normalize_generation_text(request.text)
    if normalized_text == "":
        raise ValueError("text became empty after normalization.")

    text_max_len, caption_max_len = resolve_condition_lengths(
        default_text_max_len=default_text_max_len,
        default_caption_max_len=default_caption_max_len,
        max_text_len=request.max_text_len,
        max_caption_len=request.max_caption_len,
    )

    text_ids, text_mask = tokenizer.batch_encode(
        [normalized_text] * int(request.num_candidates),
        max_length=text_max_len,
    )
    text_ids = text_ids.to(model_device)
    text_mask = text_mask.to(model_device)

    caption_text = "" if request.caption is None else str(request.caption).strip()
    caption_ids = None
    caption_mask = None
    if model_cfg.use_caption_condition:
        if caption_tokenizer is None:
            raise RuntimeError(
                "Caption conditioning is enabled but caption tokenizer is not loaded."
            )
        caption_ids, caption_mask = caption_tokenizer.batch_encode(
            [caption_text] * int(request.num_candidates),
            max_length=caption_max_len,
        )
        if caption_text == "":
            caption_mask.zero_()
        caption_ids = caption_ids.to(model_device)
        caption_mask = caption_mask.to(model_device)

    return normalized_text, caption_text, text_ids, text_mask, caption_ids, caption_mask


def _prepare_character_images(
    *,
    request: SamplingRequest,
    model_cfg: ModelConfig,
    character_image_transform: Callable | None,
    model_device: torch.device,
) -> torch.Tensor | None:
    if not model_cfg.use_character_condition:
        return None

    if request.character_image is not None and str(request.character_image).strip():
        from .image_encoder import load_character_image

        if character_image_transform is None:
            raise RuntimeError(
                "Character conditioning is enabled but character image transform is not loaded."
            )
        image = load_character_image(Path(request.character_image))
        image_tensor = character_image_transform(image).unsqueeze(0).to(model_device)
        return image_tensor.expand(int(request.num_candidates), -1, -1, -1)

    return torch.zeros(
        int(request.num_candidates),
        3,
        model_cfg.character_image_size,
        model_cfg.character_image_size,
        device=model_device,
    )


def prepare_reference_latent(
    *,
    request: SamplingRequest,
    codec: DACVAECodec,
    model_cfg: ModelConfig,
    batch_size: int,
    model_device: torch.device,
    model_dtype: torch.dtype,
    messages: list[str],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not model_cfg.use_speaker_condition:
        if request.ref_wav is not None or request.ref_latent is not None:
            messages.append(
                "info: speaker conditioning is disabled for this checkpoint; ignoring reference input."
            )
        return None, None

    if request.no_ref:
        ref_len = max(1, int(model_cfg.speaker_patch_size))
        ref_latent_patched = torch.zeros(
            (
                batch_size,
                ref_len,
                model_cfg.latent_dim * model_cfg.latent_patch_size,
            ),
            device=model_device,
            dtype=model_dtype,
        )
        ref_mask = torch.zeros((batch_size, ref_len), dtype=torch.bool, device=model_device)
        return ref_latent_patched, ref_mask

    if request.ref_wav is None and request.ref_latent is None:
        raise ValueError("Specify either ref_wav/ref_latent, or set no_ref=True.")

    max_ref_latent_steps = None
    if request.max_ref_seconds is not None and request.max_ref_seconds > 0:
        max_ref_latent_steps = max(
            1,
            math.ceil(
                float(request.max_ref_seconds)
                * float(codec.sample_rate)
                / float(int(codec.model.hop_length))
            ),
        )

    if request.ref_latent is not None:
        latent_raw = torch.load(request.ref_latent, map_location="cpu", weights_only=True)
        ref_latent = _coerce_latent_shape(latent_raw, latent_dim=model_cfg.latent_dim).unsqueeze(0)
        ref_latent = ref_latent.to(dtype=model_dtype)
    else:
        wav, sr = load_audio(request.ref_wav)
        if request.max_ref_seconds is not None and request.max_ref_seconds > 0:
            max_ref_samples = max(1, int(float(request.max_ref_seconds) * float(sr)))
            if wav.shape[1] > max_ref_samples:
                messages.append(
                    f"warning: reference audio exceeds max_ref_seconds ({request.max_ref_seconds}s). "
                    f"Trimming from {float(wav.shape[1]) / float(sr):.2f}s to {float(max_ref_samples) / float(sr):.2f}s."
                )
                wav = wav[:, :max_ref_samples]
        if request.ref_normalize_db is not None:
            messages.append(
                f"info: reference loudness normalize enabled (target_db={float(request.ref_normalize_db):.2f}, includes peak safety scaling)."
            )
        elif request.ref_ensure_max:
            messages.append("info: reference peak safety scaling enabled (ensure_max=True).")
        ref_latent = codec.encode_waveform(
            wav.unsqueeze(0),
            sample_rate=int(sr),
            normalize_db=request.ref_normalize_db,
            ensure_max=bool(request.ref_ensure_max),
        ).cpu()

    if max_ref_latent_steps is not None and ref_latent.shape[1] > max_ref_latent_steps:
        messages.append(
            f"warning: reference latent steps ({ref_latent.shape[1]}) exceed max_ref_seconds bound ({max_ref_latent_steps} steps). "
            "Trimming reference latent."
        )
        ref_latent = ref_latent[:, :max_ref_latent_steps]

    ref_latent_patched = patchify_latent(ref_latent, model_cfg.latent_patch_size).to(model_device)
    if ref_latent_patched.shape[1] == 0:
        raise ValueError(
            "Reference latent length became zero after patchify. Use longer reference audio."
        )
    if batch_size > 1:
        ref_latent_patched = ref_latent_patched.repeat(batch_size, 1, 1)
    ref_mask = torch.ones(
        (batch_size, ref_latent_patched.shape[1]), dtype=torch.bool, device=model_device
    )
    return ref_latent_patched, ref_mask


def prepare_speaker_embedding_condition(
    *,
    request: SamplingRequest,
    model_cfg: ModelConfig,
    batch_size: int,
    model_device: torch.device,
    model_dtype: torch.dtype,
    messages: list[str],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, str]:
    if request.speaker_embedding is None:
        return None, None, None, None, "mask"
    if not model_cfg.use_speaker_condition:
        messages.append(
            "info: speaker conditioning is disabled for this checkpoint; ignoring speaker embedding."
        )
        return None, None, None, None, "mask"
    if request.ref_wav is not None or request.ref_latent is not None or request.no_ref:
        raise ValueError(
            "speaker_embedding cannot be combined with ref_wav/ref_latent/no_ref. "
            "Use exactly one speaker conditioning source."
        )

    payload = load_speaker_inversion_payload(request.speaker_embedding, model_cfg=model_cfg)
    state, mask, uncond_state, uncond_mask, uncond_mode = speaker_inversion_batch_tensors(
        payload,
        batch_size=batch_size,
        device=model_device,
        dtype=model_dtype,
    )
    messages.append(
        "info: using speaker inversion embedding "
        f"{payload['source_path']} tokens={state.shape[1]} uncond_mode={uncond_mode}."
    )
    return state, mask, uncond_state, uncond_mask, uncond_mode


def _decode_generated_audios(
    *,
    latent: torch.Tensor,
    codec: DACVAECodec,
    target_samples: int,
    decode_mode: str,
    trim_tail: bool,
    tail_window_size: int,
    tail_std_threshold: float,
    tail_mean_threshold: float,
) -> list[torch.Tensor]:
    trimmed_audios: list[torch.Tensor] = []
    if decode_mode == "batch":
        audio_batch = codec.decode_latent(latent).cpu()
        for index in range(latent.shape[0]):
            audio_item = audio_batch[index]
            max_samples = target_samples
            if trim_tail:
                flattening_point = find_flattening_point(
                    latent[index],
                    window_size=max(1, int(tail_window_size)),
                    std_threshold=float(tail_std_threshold),
                    mean_threshold=float(tail_mean_threshold),
                )
                flattening_samples = int(flattening_point * int(codec.model.hop_length))
                if flattening_samples > 0:
                    max_samples = min(max_samples, flattening_samples)
            trimmed_audios.append(audio_item[:, :max_samples])
        return trimmed_audios

    for index in range(latent.shape[0]):
        audio_item = codec.decode_latent(latent[index : index + 1]).cpu()[0]
        max_samples = target_samples
        if trim_tail:
            flattening_point = find_flattening_point(
                latent[index],
                window_size=max(1, int(tail_window_size)),
                std_threshold=float(tail_std_threshold),
                mean_threshold=float(tail_mean_threshold),
            )
            flattening_samples = int(flattening_point * int(codec.model.hop_length))
            if flattening_samples > 0:
                max_samples = min(max_samples, flattening_samples)
        trimmed_audios.append(audio_item[:, :max_samples])
    return trimmed_audios


def generate_from_components(
    *,
    model,
    model_cfg: ModelConfig,
    tokenizer: PretrainedTextTokenizer,
    caption_tokenizer: PretrainedTextTokenizer | None,
    codec: DACVAECodec,
    request: SamplingRequest,
    default_text_max_len: int,
    default_caption_max_len: int,
    character_image_transform: Callable | None,
    model_device: torch.device,
    codec_device: torch.device | None = None,
    use_bf16_autocast: bool = False,
    fixed_target_latent_steps: int | None = None,
    log_fn: Callable[[str], None] | None = None,
    measure_start: MeasureStart | None = None,
    measure_end: MeasureEnd | None = None,
) -> SamplingResult:
    measure_start = _default_measure_start if measure_start is None else measure_start
    measure_end = _default_measure_end if measure_end is None else measure_end
    resolved_codec_device = model_device if codec_device is None else torch.device(codec_device)

    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    messages: list[str] = []
    stage_timings: list[tuple[str, float]] = []

    manual_seconds = None if request.seconds is None else float(request.seconds)
    if manual_seconds is not None and manual_seconds <= 0:
        raise ValueError(f"seconds must be > 0 when provided, got {request.seconds}")
    duration_scale = float(request.duration_scale)
    if duration_scale <= 0:
        raise ValueError(f"duration_scale must be > 0, got {duration_scale}")
    min_seconds = float(request.min_seconds)
    max_seconds = float(request.max_seconds)
    if min_seconds <= 0:
        raise ValueError(f"min_seconds must be > 0, got {min_seconds}")
    if max_seconds < min_seconds:
        raise ValueError(
            f"max_seconds must be >= min_seconds, got min={min_seconds} max={max_seconds}"
        )
    batch_size = int(request.num_candidates)
    if batch_size <= 0:
        raise ValueError(f"num_candidates must be > 0, got {batch_size}")
    decode_mode = str(request.decode_mode).strip().lower()
    if decode_mode not in {"sequential", "batch"}:
        raise ValueError(
            f"Unsupported decode_mode={request.decode_mode!r}. Expected one of: sequential, batch."
        )

    truncation_factor = (
        None if request.truncation_factor is None else float(request.truncation_factor)
    )
    rescale_k = None if request.rescale_k is None else float(request.rescale_k)
    rescale_sigma = None if request.rescale_sigma is None else float(request.rescale_sigma)
    if truncation_factor is not None and truncation_factor <= 0:
        raise ValueError(f"truncation_factor must be > 0, got {truncation_factor}")
    if (rescale_k is None) != (rescale_sigma is None):
        raise ValueError("rescale_k and rescale_sigma must be set together.")
    if rescale_k is not None and rescale_k <= 0:
        raise ValueError(f"rescale_k must be > 0, got {rescale_k}")
    if rescale_sigma is not None and rescale_sigma <= 0:
        raise ValueError(f"rescale_sigma must be > 0, got {rescale_sigma}")

    speaker_kv_scale = None if request.speaker_kv_scale is None else float(request.speaker_kv_scale)
    speaker_kv_min_t = None
    speaker_kv_max_layers = (
        None if request.speaker_kv_max_layers is None else int(request.speaker_kv_max_layers)
    )
    if speaker_kv_scale is not None:
        if not model_cfg.use_speaker_condition:
            messages.append(
                "info: speaker conditioning is disabled for this checkpoint; ignoring speaker_kv_scale."
            )
            speaker_kv_scale = None
        else:
            if speaker_kv_scale <= 0:
                raise ValueError(f"speaker_kv_scale must be > 0, got {speaker_kv_scale}")
            speaker_kv_min_t = (
                0.9 if request.speaker_kv_min_t is None else float(request.speaker_kv_min_t)
            )
            if not (0.0 <= speaker_kv_min_t <= 1.0):
                raise ValueError(f"speaker_kv_min_t must be in [0, 1], got {speaker_kv_min_t}")
            if speaker_kv_max_layers is not None and speaker_kv_max_layers < 0:
                raise ValueError(
                    f"speaker_kv_max_layers must be >= 0 when specified, got {speaker_kv_max_layers}"
                )

    cfg_mode = str(request.cfg_guidance_mode).strip().lower()
    if cfg_mode not in {"independent", "joint", "alternating"}:
        raise ValueError(
            f"Unsupported cfg_guidance_mode={request.cfg_guidance_mode!r}. Expected one of: independent, joint, alternating."
        )

    if request.seed is None:
        used_seed = int(secrets.randbits(63))
        msg = f"info: seed not specified; using random seed {used_seed}."
        messages.append(msg)
        _log(msg)
    else:
        used_seed = int(request.seed)
        _log(f"[runtime] using seed: {used_seed}")

    total_t0 = measure_start(model_device, resolved_codec_device)
    with torch.inference_mode():
        t0 = measure_start(model_device)
        normalized_text, caption_text, text_ids, text_mask, caption_ids, caption_mask = (
            _prepare_text_and_caption_tensors(
                tokenizer=tokenizer,
                caption_tokenizer=caption_tokenizer,
                request=request,
                model_cfg=model_cfg,
                model_device=model_device,
                default_text_max_len=default_text_max_len,
                default_caption_max_len=default_caption_max_len,
            )
        )
        stage_sec = measure_end(t0, model_device)
        stage_timings.append(("tokenize_text", stage_sec))
        _log(f"[runtime] tokenize_text: {stage_sec * 1000.0:.1f} ms")

        (
            cfg_scale_text,
            cfg_scale_caption,
            cfg_scale_speaker,
            cfg_scale_character,
            scale_messages,
        ) = resolve_cfg_scales(
            cfg_guidance_mode=cfg_mode,
            cfg_scale_text=request.cfg_scale_text,
            cfg_scale_caption=request.cfg_scale_caption,
            cfg_scale_speaker=request.cfg_scale_speaker,
            cfg_scale_character=request.cfg_scale_character,
            cfg_scale=request.cfg_scale,
            use_caption_condition=bool(model_cfg.use_caption_condition and caption_text != ""),
            use_speaker_condition=model_cfg.use_speaker_condition,
            use_character_condition=model_cfg.use_character_condition,
        )
        messages.extend(scale_messages)
        for msg in scale_messages:
            _log(msg)

        character_images = _prepare_character_images(
            request=request,
            model_cfg=model_cfg,
            character_image_transform=character_image_transform,
            model_device=model_device,
        )
        if character_images is not None:
            character_images = character_images.to(dtype=next(model.parameters()).dtype)

        t0 = measure_start(model_device, resolved_codec_device)
        msg_count_before_ref = len(messages)
        speaker_state_override = None
        speaker_mask_override = None
        speaker_uncond_state = None
        speaker_uncond_mask = None
        speaker_uncond_mode = "mask"
        (
            speaker_state_override,
            speaker_mask_override,
            speaker_uncond_state,
            speaker_uncond_mask,
            speaker_uncond_mode,
        ) = prepare_speaker_embedding_condition(
            request=request,
            model_cfg=model_cfg,
            batch_size=batch_size,
            model_device=model_device,
            model_dtype=next(model.parameters()).dtype,
            messages=messages,
        )
        has_internal_speaker_inversion = getattr(model, "speaker_inversion", None) is not None
        if speaker_state_override is not None or has_internal_speaker_inversion:
            ref_latent = None
            ref_mask = None
        else:
            ref_latent, ref_mask = prepare_reference_latent(
                request=request,
                codec=codec,
                model_cfg=model_cfg,
                batch_size=batch_size,
                model_device=model_device,
                model_dtype=next(model.parameters()).dtype,
                messages=messages,
            )
        stage_sec = measure_end(t0, model_device, resolved_codec_device)
        stage_timings.append(("prepare_reference", stage_sec))
        for msg in messages[msg_count_before_ref:]:
            _log(msg)
        _log(f"[runtime] prepare_reference: {stage_sec * 1000.0:.1f} ms")

        hop_length = int(codec.model.hop_length)
        if manual_seconds is not None:
            clamped_seconds = min(max_seconds, max(min_seconds, manual_seconds))
            if clamped_seconds != manual_seconds:
                duration_msg = (
                    f"warning: manual duration {manual_seconds:.3f}s was clamped to "
                    f"{clamped_seconds:.3f}s."
                )
                messages.append(duration_msg)
                _log(duration_msg)
            target_samples, latent_steps, patched_steps = resolve_sequence_lengths(
                seconds=clamped_seconds,
                sample_rate=codec.sample_rate,
                hop_length=hop_length,
                latent_patch_size=model_cfg.latent_patch_size,
            )
            duration_msg = f"info: using manual duration {clamped_seconds:.3f}s."
            messages.append(duration_msg)
            _log(duration_msg)
        elif model_cfg.use_duration_predictor:
            t0 = measure_start(model_device)
            has_duration_reference = torch.zeros(
                (batch_size,), dtype=torch.bool, device=model_device
            )
            if model_cfg.use_speaker_condition and ref_mask is not None:
                has_duration_reference = ref_mask.any(dim=1)
            elif model_cfg.use_speaker_condition and speaker_mask_override is not None:
                has_duration_reference = speaker_mask_override.any(dim=1)
            elif model_cfg.use_speaker_condition and has_internal_speaker_inversion:
                has_duration_reference = torch.ones(
                    (batch_size,),
                    dtype=torch.bool,
                    device=model_device,
                )
            elif model_cfg.use_character_condition:
                has_character_reference = (
                    request.character_image is not None and str(request.character_image).strip()
                )
                has_duration_reference = torch.full(
                    (batch_size,),
                    bool(has_character_reference),
                    dtype=torch.bool,
                    device=model_device,
                )
            duration_features = build_duration_features(
                [normalized_text] * batch_size,
                token_counts=text_mask.sum(dim=1),
                max_text_len=int(text_mask.shape[1]),
                has_speaker=has_duration_reference,
            ).to(model_device)
            (
                duration_text_state,
                duration_text_mask,
                duration_speaker_state,
                duration_speaker_mask,
                _duration_caption_state,
                _duration_caption_mask,
                duration_character_state,
                duration_character_mask,
                duration_character_duration_state,
                _duration_character_noisy_state,
            ) = model.encode_conditions(
                text_input_ids=text_ids,
                text_mask=text_mask,
                speaker_latent=ref_latent,
                speaker_mask=ref_mask,
                caption_input_ids=caption_ids,
                caption_mask=caption_mask,
                speaker_state_override=speaker_state_override,
                speaker_mask_override=speaker_mask_override,
                speaker_uncond_state=speaker_uncond_state,
                speaker_uncond_mask=speaker_uncond_mask,
                speaker_uncond_mode=speaker_uncond_mode,
                character_images=character_images,
            )
            pred_log_frames = model.predict_duration_log_frames(
                text_state=duration_text_state,
                text_mask=duration_text_mask,
                speaker_state=duration_speaker_state,
                speaker_mask=duration_speaker_mask,
                character_state=duration_character_duration_state,
                character_mask=duration_character_mask,
                duration_features=duration_features,
                has_speaker=has_duration_reference,
            )
            pred_frames = torch.expm1(pred_log_frames).float().mean().item()
            scaled_frames = pred_frames * duration_scale
            min_frames = max(1, math.ceil(min_seconds * codec.sample_rate / hop_length))
            max_frames = max(1, math.floor(max_seconds * codec.sample_rate / hop_length))
            latent_steps = int(round(scaled_frames))
            latent_steps = max(min_frames, min(max_frames, latent_steps))
            target_samples = int(latent_steps * hop_length)
            patched_steps = max(1, math.ceil(latent_steps / int(model_cfg.latent_patch_size)))
            stage_sec = measure_end(t0, model_device)
            stage_timings.append(("predict_duration", stage_sec))
            msg = (
                f"info: predicted duration frames={pred_frames:.1f}, "
                f"scale={duration_scale:.3f}, using_frames={latent_steps} "
                f"({target_samples / float(codec.sample_rate):.3f}s)."
            )
            messages.append(msg)
            _log(msg)
            _log(f"[runtime] predict_duration: {stage_sec * 1000.0:.1f} ms")
        else:
            fallback_seconds = 30.0
            target_samples, latent_steps, patched_steps = resolve_sequence_lengths(
                seconds=fallback_seconds,
                sample_rate=codec.sample_rate,
                hop_length=hop_length,
                latent_patch_size=model_cfg.latent_patch_size,
            )
            msg = "info: checkpoint has no duration predictor; falling back to 30.000s."
            messages.append(msg)
            _log(msg)

        if (
            fixed_target_latent_steps is not None
            and int(fixed_target_latent_steps) > 0
            and latent_steps > int(fixed_target_latent_steps)
        ):
            msg = (
                f"warning: requested latent length ({latent_steps}) exceeds fixed_target_latent_steps ({fixed_target_latent_steps}) "
                "used in training. Long-tail stability may degrade."
            )
            messages.append(msg)
            _log(msg)

        t0 = measure_start(model_device)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16_autocast and model_device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            z_patched = sample_euler_rf_cfg(
                model=model,
                text_input_ids=text_ids,
                text_mask=text_mask,
                ref_latent=ref_latent,
                ref_mask=ref_mask,
                sequence_length=patched_steps,
                caption_input_ids=caption_ids,
                caption_mask=caption_mask,
                speaker_state_override=speaker_state_override,
                speaker_mask_override=speaker_mask_override,
                speaker_uncond_state=speaker_uncond_state,
                speaker_uncond_mask=speaker_uncond_mask,
                speaker_uncond_mode=speaker_uncond_mode,
                character_images=character_images,
                num_steps=int(request.num_steps),
                cfg_scale_text=cfg_scale_text,
                cfg_scale_caption=cfg_scale_caption,
                cfg_scale_speaker=cfg_scale_speaker,
                cfg_scale_character=cfg_scale_character,
                cfg_guidance_mode=cfg_mode,
                cfg_min_t=float(request.cfg_min_t),
                cfg_max_t=float(request.cfg_max_t),
                seed=used_seed,
                cfg_scale=None,
                truncation_factor=truncation_factor,
                rescale_k=rescale_k,
                rescale_sigma=rescale_sigma,
                use_context_kv_cache=bool(request.context_kv_cache),
                speaker_kv_scale=speaker_kv_scale,
                speaker_kv_max_layers=speaker_kv_max_layers,
                speaker_kv_min_t=speaker_kv_min_t,
                t_schedule_mode=str(request.t_schedule_mode),
                sway_coeff=float(request.sway_coeff),
            )
        stage_sec = measure_end(t0, model_device)
        stage_timings.append(("sample_rf", stage_sec))
        _log(f"[runtime] sample_rf: {stage_sec * 1000.0:.1f} ms")

        t0 = measure_start(model_device)
        z = unpatchify_latent(
            z_patched,
            patch_size=model_cfg.latent_patch_size,
            latent_dim=model_cfg.latent_dim,
        )
        stage_sec = measure_end(t0, model_device)
        stage_timings.append(("unpatchify_latent", stage_sec))
        _log(f"[runtime] unpatchify_latent: {stage_sec * 1000.0:.1f} ms")
        z = z[:, :latent_steps]

        t0 = measure_start(model_device, resolved_codec_device)
        trimmed_audios = _decode_generated_audios(
            latent=z,
            codec=codec,
            target_samples=target_samples,
            decode_mode=decode_mode,
            trim_tail=bool(request.trim_tail),
            tail_window_size=int(request.tail_window_size),
            tail_std_threshold=float(request.tail_std_threshold),
            tail_mean_threshold=float(request.tail_mean_threshold),
        )
        stage_sec = measure_end(t0, model_device, resolved_codec_device)
        stage_timings.append(("decode_latent", stage_sec))
        _log(f"[runtime] decode_latent ({decode_mode}): {stage_sec * 1000.0:.1f} ms")

    total_to_decode = measure_end(total_t0, model_device, resolved_codec_device)
    _log(f"[runtime] total_to_decode: {total_to_decode:.3f} s")

    return SamplingResult(
        audio=trimmed_audios[0],
        audios=trimmed_audios,
        sample_rate=int(codec.sample_rate),
        stage_timings=stage_timings,
        total_to_decode=total_to_decode,
        used_seed=used_seed,
        messages=messages,
    )
