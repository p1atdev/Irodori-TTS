from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import ModelConfig, TrainConfig

SPEAKER_INVERSION_FORMAT = "irodori_speaker_inversion"
SPEAKER_INVERSION_FORMAT_VERSION = 1
SPEAKER_INVERSION_UNCOND_MODES = {"mask", "noise"}
SPEAKER_EMBEDDING_KEY = "speaker_embedding"
SPEAKER_UNCOND_EMBEDDING_KEY = "speaker_uncond_embedding"


def normalize_speaker_embedding_tensor(
    tensor: torch.Tensor,
    *,
    speaker_dim: int,
    field_name: str = SPEAKER_EMBEDDING_KEY,
) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"{field_name} must have shape (tokens, dim), got {tuple(tensor.shape)}")
    if int(tensor.shape[0]) <= 0:
        raise ValueError(f"{field_name} must contain at least one token.")
    if int(tensor.shape[1]) != int(speaker_dim):
        raise ValueError(
            f"{field_name} dim mismatch: expected {int(speaker_dim)}, got {int(tensor.shape[1])}"
        )
    return tensor.detach().float().contiguous()


class SpeakerInversionEmbedding(nn.Module):
    """Learned speaker/style tokens that bypass the reference latent speaker encoder."""

    def __init__(
        self,
        *,
        num_tokens: int,
        speaker_dim: int,
        init_std: float,
        uncond_mode: str = "mask",
        uncond_std: float = 1.0,
        init_embedding: torch.Tensor | None = None,
        uncond_embedding: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        num_tokens = int(num_tokens)
        speaker_dim = int(speaker_dim)
        init_std = float(init_std)
        uncond_std = float(uncond_std)
        uncond_mode = str(uncond_mode).strip().lower()
        if num_tokens <= 0:
            raise ValueError(f"speaker inversion tokens must be > 0, got {num_tokens}")
        if speaker_dim <= 0:
            raise ValueError(f"speaker_dim must be > 0, got {speaker_dim}")
        if init_std < 0:
            raise ValueError(f"speaker inversion init_std must be >= 0, got {init_std}")
        if uncond_std < 0:
            raise ValueError(f"speaker inversion uncond_std must be >= 0, got {uncond_std}")
        if uncond_mode not in SPEAKER_INVERSION_UNCOND_MODES:
            raise ValueError(
                "speaker inversion uncond mode must be one of "
                f"{sorted(SPEAKER_INVERSION_UNCOND_MODES)}, got {uncond_mode!r}"
            )

        if init_embedding is None:
            embedding = torch.randn(num_tokens, speaker_dim, dtype=torch.float32) * init_std
        else:
            embedding = normalize_speaker_embedding_tensor(
                init_embedding,
                speaker_dim=speaker_dim,
                field_name=SPEAKER_EMBEDDING_KEY,
            )
            if int(embedding.shape[0]) != num_tokens:
                raise ValueError(
                    "speaker inversion init embedding token mismatch: "
                    f"expected {num_tokens}, got {int(embedding.shape[0])}"
                )
        self.embedding = nn.Parameter(embedding)
        self.uncond_mode = uncond_mode

        if uncond_embedding is None:
            uncond = torch.randn(num_tokens, speaker_dim, dtype=torch.float32) * uncond_std
        else:
            uncond = normalize_speaker_embedding_tensor(
                uncond_embedding,
                speaker_dim=speaker_dim,
                field_name=SPEAKER_UNCOND_EMBEDDING_KEY,
            )
            if int(uncond.shape[0]) != num_tokens:
                raise ValueError(
                    "speaker inversion uncond embedding token mismatch: "
                    f"expected {num_tokens}, got {int(uncond.shape[0])}"
                )
        self.register_buffer("uncond_embedding", uncond, persistent=True)

    @property
    def num_tokens(self) -> int:
        return int(self.embedding.shape[0])

    @property
    def speaker_dim(self) -> int:
        return int(self.embedding.shape[1])

    def forward(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.embedding.to(device=device, dtype=dtype)[None, :, :].expand(
            int(batch_size),
            -1,
            -1,
        )
        mask = torch.ones((int(batch_size), self.num_tokens), dtype=torch.bool, device=device)
        return state, mask

    def unconditional(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.uncond_mode == "noise":
            state = self.uncond_embedding.to(device=device, dtype=dtype)[None, :, :].expand(
                int(batch_size),
                -1,
                -1,
            )
            mask = torch.ones((int(batch_size), self.num_tokens), dtype=torch.bool, device=device)
            return state, mask

        state = torch.zeros(
            (int(batch_size), self.num_tokens, self.speaker_dim),
            dtype=dtype,
            device=device,
        )
        mask = torch.zeros((int(batch_size), self.num_tokens), dtype=torch.bool, device=device)
        return state, mask


def _extract_embedding_payload(raw: Any, *, model_cfg: ModelConfig) -> dict[str, Any]:
    if isinstance(raw, torch.Tensor):
        return {SPEAKER_EMBEDDING_KEY: raw}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Speaker inversion file must contain a tensor or dict, got {type(raw)!r}."
        )

    if SPEAKER_EMBEDDING_KEY in raw:
        return raw
    # Accept a raw state_dict saved from SpeakerInversionEmbedding for convenience.
    if "embedding" in raw:
        payload: dict[str, Any] = {SPEAKER_EMBEDDING_KEY: raw["embedding"]}
        if "uncond_embedding" in raw:
            payload[SPEAKER_UNCOND_EMBEDDING_KEY] = raw["uncond_embedding"]
        return payload

    raise ValueError(
        f"Speaker inversion file is missing '{SPEAKER_EMBEDDING_KEY}' for "
        f"speaker_dim={model_cfg.speaker_dim}."
    )


def load_speaker_inversion_payload(
    path: str | Path,
    *,
    model_cfg: ModelConfig,
) -> dict[str, Any]:
    source = Path(path).expanduser()
    raw = torch.load(source, map_location="cpu", weights_only=True)
    payload = _extract_embedding_payload(raw, model_cfg=model_cfg)
    embedding = normalize_speaker_embedding_tensor(
        payload[SPEAKER_EMBEDDING_KEY],
        speaker_dim=model_cfg.speaker_dim,
        field_name=SPEAKER_EMBEDDING_KEY,
    )

    out: dict[str, Any] = {
        SPEAKER_EMBEDDING_KEY: embedding,
        "source_path": str(source),
    }
    uncond_mode = str(payload.get("speaker_uncond_mode", "mask")).strip().lower()
    if uncond_mode not in SPEAKER_INVERSION_UNCOND_MODES:
        raise ValueError(
            f"speaker_uncond_mode must be one of {sorted(SPEAKER_INVERSION_UNCOND_MODES)}, "
            f"got {uncond_mode!r}"
        )
    out["speaker_uncond_mode"] = uncond_mode

    uncond = payload.get(SPEAKER_UNCOND_EMBEDDING_KEY)
    if uncond is not None:
        out[SPEAKER_UNCOND_EMBEDDING_KEY] = normalize_speaker_embedding_tensor(
            uncond,
            speaker_dim=model_cfg.speaker_dim,
            field_name=SPEAKER_UNCOND_EMBEDDING_KEY,
        )
    return out


def speaker_inversion_batch_tensors(
    payload: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, str]:
    embedding = payload[SPEAKER_EMBEDDING_KEY].to(device=device, dtype=dtype)
    state = embedding[None, :, :].expand(int(batch_size), -1, -1)
    mask = torch.ones((int(batch_size), embedding.shape[0]), dtype=torch.bool, device=device)
    uncond_mode = str(payload.get("speaker_uncond_mode", "mask")).strip().lower()

    uncond_state = None
    uncond_mask = None
    if uncond_mode == "noise":
        uncond_embedding = payload.get(SPEAKER_UNCOND_EMBEDDING_KEY)
        if uncond_embedding is None:
            uncond_embedding = torch.zeros_like(embedding)
        else:
            uncond_embedding = uncond_embedding.to(device=device, dtype=dtype)
        uncond_state = uncond_embedding[None, :, :].expand(int(batch_size), -1, -1)
        uncond_mask = torch.ones_like(mask)
    return state, mask, uncond_state, uncond_mask, uncond_mode


def speaker_inversion_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    module = getattr(model, "speaker_inversion", None)
    if not isinstance(module, SpeakerInversionEmbedding):
        raise ValueError("Model does not have an enabled SpeakerInversionEmbedding module.")
    return {
        SPEAKER_EMBEDDING_KEY: module.embedding.detach().cpu().float().clone(),
        SPEAKER_UNCOND_EMBEDDING_KEY: module.uncond_embedding.detach().cpu().float().clone(),
    }


def save_speaker_inversion_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    step: int,
    base_init: dict | None = None,
    extra_state: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = speaker_inversion_state_dict(model)
    module = model.speaker_inversion
    payload: dict[str, Any] = {
        "format": SPEAKER_INVERSION_FORMAT,
        "format_version": SPEAKER_INVERSION_FORMAT_VERSION,
        "step": int(step),
        "speaker_uncond_mode": module.uncond_mode,
        "speaker_tokens": int(state[SPEAKER_EMBEDDING_KEY].shape[0]),
        "speaker_dim": int(state[SPEAKER_EMBEDDING_KEY].shape[1]),
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "base_init": base_init,
        **state,
    }
    if extra_state:
        payload.update(extra_state)
    torch.save(payload, path)
