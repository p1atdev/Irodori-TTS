from __future__ import annotations

import gc
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file

from . import generation_core
from .codec import DACVAECodec
from .config import ModelConfig
from .lora import checkpoint_state_uses_lora
from .model import TextToLatentRFDiT
from .tokenizer import PretrainedTextTokenizer


def _is_mps_available() -> bool:
    backends = getattr(torch, "backends", None)
    if backends is None or not hasattr(backends, "mps"):
        return False
    return bool(torch.backends.mps.is_available())


def resolve_runtime_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return resolved
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch.cuda.is_available() is False.")
        return resolved
    if resolved.type == "mps":
        if resolved.index is not None:
            raise ValueError("MPS device index is not supported. Use 'mps'.")
        if not _is_mps_available():
            raise ValueError("MPS device requested but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    raise ValueError(f"Unsupported inference device={resolved!s}. Expected one of: cpu, cuda, mps.")


def list_available_runtime_devices() -> list[str]:
    devices: list[str] = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if _is_mps_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def default_runtime_device() -> str:
    return list_available_runtime_devices()[0]


def list_available_runtime_precisions(device: str | torch.device) -> list[str]:
    resolved = resolve_runtime_device(device)
    if resolved.type == "cuda":
        return ["fp32", "bf16"]
    return ["fp32"]


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "synchronize"):
            mps.synchronize()


def _sync_devices(*devices: torch.device) -> None:
    seen: set[tuple[str, int | None]] = set()
    for device in devices:
        key = (device.type, device.index)
        if key in seen:
            continue
        _sync_device(device)
        seen.add(key)


def _measure_start(device: torch.device, *extra_devices: torch.device) -> float:
    _sync_devices(device, *extra_devices)
    return time.perf_counter()


def _measure_end(device: torch.device, t0: float, *extra_devices: torch.device) -> float:
    _sync_devices(device, *extra_devices)
    return time.perf_counter() - t0


@dataclass(frozen=True)
class RuntimeKey:
    checkpoint: str
    model_device: str
    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    model_precision: str = "fp32"
    codec_device: str = "cpu"
    codec_precision: str = "fp32"
    codec_deterministic_encode: bool = True
    codec_deterministic_decode: bool = True
    enable_watermark: bool = False
    compile_model: bool = False
    compile_dynamic: bool = False


def _maybe_compile_inference_model(
    model: TextToLatentRFDiT,
    *,
    enabled: bool,
    dynamic: bool,
) -> TextToLatentRFDiT:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("compile_model=True requires torch.compile (PyTorch 2+).")
    compile_kwargs = {"dynamic": bool(dynamic)}
    model.encode_conditions = torch.compile(model.encode_conditions, **compile_kwargs)
    model.build_context_kv_cache = torch.compile(model.build_context_kv_cache, **compile_kwargs)
    model.forward_with_encoded_conditions = torch.compile(
        model.forward_with_encoded_conditions,
        **compile_kwargs,
    )
    return model


def resolve_runtime_dtype(*, precision: str, device: torch.device) -> torch.dtype:
    mode = str(precision).strip().lower()
    if mode == "fp32":
        return torch.float32
    if mode == "bf16":
        if device.type != "cuda":
            raise ValueError("precision='bf16' currently requires CUDA device.")
        return torch.bfloat16
    raise ValueError(f"Unsupported precision={precision!r}. Expected one of: fp32, bf16.")


SamplingRequest = generation_core.SamplingRequest
SamplingResult = generation_core.SamplingResult
find_flattening_point = generation_core.find_flattening_point
resolve_cfg_scales = generation_core.resolve_cfg_scales


def _load_torch_checkpoint_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return payload


_CONFIG_META_KEY = "config_json"
_INFERENCE_CONFIG_KEYS = {"max_text_len", "max_caption_len", "fixed_target_latent_steps"}


def _load_checkpoint_from_pt(path: Path) -> tuple[dict[str, torch.Tensor], dict, dict | None]:
    ckpt = _load_torch_checkpoint_payload(path)
    model_state = ckpt.get("model")
    model_cfg = ckpt.get("model_config")
    train_cfg = ckpt.get("train_config")

    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint missing model weights dictionary: {path}")
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Checkpoint missing model_config dictionary: {path}")
    if train_cfg is not None and not isinstance(train_cfg, dict):
        raise ValueError(f"Checkpoint train_config must be a dictionary when present: {path}")

    if checkpoint_state_uses_lora(model_state):
        raise ValueError(
            f"LoRA checkpoints must be loaded from adapter directories or merged safetensors: {path}"
        )
    return model_state, model_cfg, _extract_inference_train_config(train_cfg)


def _parse_json_mapping(
    raw: str | None,
    *,
    field: str,
    path: Path,
    required: bool = False,
) -> dict | None:
    if raw is None:
        if required:
            raise ValueError(f"Missing required metadata field '{field}' in checkpoint: {path}")
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{field}' metadata for checkpoint: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata field '{field}' must decode to an object: {path}")
    return payload


def _extract_inference_train_config(raw: dict | None) -> dict | None:
    if raw is None:
        return None

    inference_cfg: dict[str, int] = {}
    for key in _INFERENCE_CONFIG_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, int):
            raise ValueError(f"Inference config key '{key}' must be int, got {type(value)!r}.")
        inference_cfg[key] = int(value)

    return inference_cfg or None


def _split_flat_checkpoint_config(path: Path, flat_config: dict) -> tuple[dict, dict | None]:
    model_cfg: dict[str, object] = {}
    inference_cfg: dict[str, int] = {}
    for key, value in flat_config.items():
        if key in _INFERENCE_CONFIG_KEYS:
            if not isinstance(value, int):
                raise ValueError(
                    f"Inference config key '{key}' must be int in checkpoint metadata: {path}"
                )
            inference_cfg[key] = int(value)
            continue
        model_cfg[key] = value
    return model_cfg, (inference_cfg or None)


def _load_checkpoint_from_safetensors(
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict, dict | None]:
    model_state = load_safetensors_file(str(path), device="cpu")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError(f"Safetensors checkpoint has no model weights: {path}")

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}

    flat_config = _parse_json_mapping(
        metadata.get(_CONFIG_META_KEY),
        field=_CONFIG_META_KEY,
        path=path,
        required=True,
    )
    model_cfg, inference_cfg = _split_flat_checkpoint_config(path=path, flat_config=flat_config)
    return model_state, model_cfg, inference_cfg


def _load_checkpoint_for_inference(path: Path) -> tuple[dict[str, torch.Tensor], dict, dict | None]:
    if path.suffix.lower() == ".safetensors":
        return _load_checkpoint_from_safetensors(path)
    return _load_checkpoint_from_pt(path)


class InferenceRuntime:
    def __init__(
        self,
        *,
        key: RuntimeKey,
        model_cfg: ModelConfig,
        train_cfg: dict | None,
        model: TextToLatentRFDiT,
        tokenizer: PretrainedTextTokenizer,
        caption_tokenizer: PretrainedTextTokenizer | None,
        codec: DACVAECodec,
        default_text_max_len: int,
        default_caption_max_len: int,
        character_image_transform: Callable | None = None,
    ) -> None:
        self.key = key
        self.model_device = resolve_runtime_device(key.model_device)
        self.codec_device = resolve_runtime_device(key.codec_device)
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.model = model
        self.tokenizer = tokenizer
        self.caption_tokenizer = caption_tokenizer
        self.codec = codec
        self.default_text_max_len = default_text_max_len
        self.default_caption_max_len = default_caption_max_len
        self.character_image_transform = character_image_transform
        self._infer_lock = threading.Lock()

    @classmethod
    def from_key(cls, key: RuntimeKey) -> InferenceRuntime:
        model_device = resolve_runtime_device(key.model_device)
        codec_device = resolve_runtime_device(key.codec_device)
        model_dtype = resolve_runtime_dtype(
            precision=key.model_precision,
            device=model_device,
        )
        codec_dtype = resolve_runtime_dtype(
            precision=key.codec_precision,
            device=codec_device,
        )

        model_state, model_cfg_dict, train_cfg = _load_checkpoint_for_inference(
            Path(key.checkpoint)
        )
        model_cfg = ModelConfig(**model_cfg_dict)

        model = TextToLatentRFDiT(model_cfg).to(model_device)
        model.load_state_dict(model_state)
        model = model.to(dtype=model_dtype)
        model.eval()
        model = _maybe_compile_inference_model(
            model,
            enabled=bool(key.compile_model),
            dynamic=bool(key.compile_dynamic),
        )

        tokenizer = PretrainedTextTokenizer.from_pretrained(
            repo_id=model_cfg.text_tokenizer_repo,
            add_bos=bool(model_cfg.text_add_bos),
            local_files_only=False,
        )
        if tokenizer.vocab_size != model_cfg.text_vocab_size:
            raise ValueError(
                f"text_vocab_size mismatch: checkpoint text_vocab_size={model_cfg.text_vocab_size} but tokenizer "
                f"({model_cfg.text_tokenizer_repo}) vocab_size={tokenizer.vocab_size}."
            )
        caption_tokenizer = None
        if model_cfg.use_caption_condition:
            caption_tokenizer = PretrainedTextTokenizer.from_pretrained(
                repo_id=model_cfg.caption_tokenizer_repo_resolved,
                add_bos=model_cfg.caption_add_bos_resolved,
                local_files_only=False,
            )
            if caption_tokenizer.vocab_size != model_cfg.caption_vocab_size_resolved:
                raise ValueError(
                    f"caption_vocab_size mismatch: checkpoint caption_vocab_size={model_cfg.caption_vocab_size_resolved} but tokenizer ({model_cfg.caption_tokenizer_repo_resolved}) "
                    f"vocab_size={caption_tokenizer.vocab_size}."
                )

        default_text_max_len = 256
        default_caption_max_len = default_text_max_len
        if isinstance(train_cfg, dict):
            ckpt_text_max_len = train_cfg.get("max_text_len")
            if isinstance(ckpt_text_max_len, int) and ckpt_text_max_len > 0:
                default_text_max_len = int(ckpt_text_max_len)
            ckpt_caption_max_len = train_cfg.get("max_caption_len")
            if isinstance(ckpt_caption_max_len, int) and ckpt_caption_max_len > 0:
                default_caption_max_len = int(ckpt_caption_max_len)
            else:
                default_caption_max_len = default_text_max_len

        codec = DACVAECodec.load(
            repo_id=key.codec_repo,
            device=str(codec_device),
            dtype=codec_dtype,
            deterministic_encode=bool(key.codec_deterministic_encode),
            deterministic_decode=bool(key.codec_deterministic_decode),
            enable_watermark=bool(key.enable_watermark),
        )
        if model_cfg.latent_dim != codec.latent_dim:
            raise ValueError(
                f"Latent dimension mismatch: checkpoint latent_dim={model_cfg.latent_dim} but codec latent_dim={codec.latent_dim}. "
                "Use a compatible codec/checkpoint pair."
            )

        character_image_transform = None
        if model_cfg.use_character_condition and model_cfg.character_encoder_model is not None:
            from .image_encoder import build_character_transform

            character_image_transform = build_character_transform(
                model_cfg.character_encoder_model,
                model_cfg.character_image_size,
            )

        return cls(
            key=key,
            model_cfg=model_cfg,
            train_cfg=train_cfg if isinstance(train_cfg, dict) else None,
            model=model,
            tokenizer=tokenizer,
            caption_tokenizer=caption_tokenizer,
            codec=codec,
            default_text_max_len=default_text_max_len,
            default_caption_max_len=default_caption_max_len,
            character_image_transform=character_image_transform,
        )

    def synthesize(
        self,
        req: SamplingRequest,
        *,
        log_fn: Callable[[str], None] | None = None,
    ) -> SamplingResult:
        def _log(msg: str) -> None:
            if log_fn is not None:
                log_fn(msg)

        _log(
            (
                "[runtime] start synthesize "
                "model_device={} model_precision={} codec_device={} codec_precision={} "
                "watermark={} mode={} seconds={} steps={} seed={} candidates={} decode_mode={}"
            ).format(
                self.key.model_device,
                self.key.model_precision,
                self.key.codec_device,
                self.key.codec_precision,
                self.codec.enable_watermark,
                req.cfg_guidance_mode,
                req.seconds,
                req.num_steps,
                "random" if req.seed is None else int(req.seed),
                req.num_candidates,
                req.decode_mode,
            )
        )
        with self._infer_lock, torch.inference_mode():
            fixed_target_latent_steps = None
            if isinstance(self.train_cfg, dict):
                candidate_steps = self.train_cfg.get("fixed_target_latent_steps")
                if isinstance(candidate_steps, int):
                    fixed_target_latent_steps = int(candidate_steps)
            result = generation_core.generate_from_components(
                model=self.model,
                model_cfg=self.model_cfg,
                tokenizer=self.tokenizer,
                caption_tokenizer=self.caption_tokenizer,
                codec=self.codec,
                request=req,
                default_text_max_len=self.default_text_max_len,
                default_caption_max_len=self.default_caption_max_len,
                character_image_transform=self.character_image_transform,
                model_device=self.model_device,
                codec_device=self.codec_device,
                use_bf16_autocast=False,
                fixed_target_latent_steps=fixed_target_latent_steps,
                log_fn=log_fn,
                measure_start=lambda *devices: _measure_start(*devices),
                measure_end=lambda t0, *devices: _measure_end(devices[0], t0, *devices[1:]),
            )

        _log("[runtime] done synthesize")
        return result

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        del self.codec
        gc.collect()
        for device in (self.model_device, self.codec_device):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                mps = getattr(torch, "mps", None)
                if mps is not None and hasattr(mps, "empty_cache"):
                    mps.empty_cache()


_RUNTIME_CACHE_LOCK = threading.Lock()
_RUNTIME_CACHE_KEY: RuntimeKey | None = None
_RUNTIME_CACHE_VALUE: InferenceRuntime | None = None


def get_cached_runtime(key: RuntimeKey) -> tuple[InferenceRuntime, bool]:
    global _RUNTIME_CACHE_KEY, _RUNTIME_CACHE_VALUE
    with _RUNTIME_CACHE_LOCK:
        if _RUNTIME_CACHE_VALUE is not None and _RUNTIME_CACHE_KEY == key:
            return _RUNTIME_CACHE_VALUE, False

        old_runtime = _RUNTIME_CACHE_VALUE
        runtime = InferenceRuntime.from_key(key)
        _RUNTIME_CACHE_KEY = key
        _RUNTIME_CACHE_VALUE = runtime

    if old_runtime is not None:
        old_runtime.unload()

    return runtime, True


def clear_cached_runtime() -> None:
    global _RUNTIME_CACHE_KEY, _RUNTIME_CACHE_VALUE
    with _RUNTIME_CACHE_LOCK:
        runtime = _RUNTIME_CACHE_VALUE
        _RUNTIME_CACHE_KEY = None
        _RUNTIME_CACHE_VALUE = None

    if runtime is not None:
        runtime.unload()


def _load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    return generation_core.load_audio(path)


def save_wav(path: str | Path, audio: torch.Tensor, sample_rate: int) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torchaudio.save(str(out_path), audio, sample_rate)
    except RuntimeError:
        import soundfile as sf

        sf.write(str(out_path), audio.squeeze(0).numpy(), sample_rate)
    return out_path
