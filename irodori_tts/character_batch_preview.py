from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from .inference_runtime import RuntimeKey, SamplingRequest, get_cached_runtime, save_wav

DEFAULT_OUTPUT_DIR = Path("gradio_outputs_character_batch")
DEFAULT_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
DEFAULT_SEED = 0
GENERATION_PRESET_VERSION = "character-batch-preview-v1"
SUPPORTED_IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
}


@dataclass(frozen=True)
class CharacterBatchGenerationSettings:
    seed: int = DEFAULT_SEED
    num_steps: int = 40
    duration_scale: float = 1.0
    cfg_guidance_mode: str = "independent"
    cfg_scale_text: float = 3.0
    cfg_scale_character: float = 3.0
    t_schedule_mode: str = "linear"
    sway_coeff: float = -1.0
    context_kv_cache: bool = True


@dataclass(frozen=True)
class ModelIdentity:
    key: str
    label: str
    raw_checkpoint: str
    resolved_checkpoint: str
    fingerprint: str


@dataclass(frozen=True)
class CharacterImageItem:
    key: str
    path: Path
    name: str
    sha256: str


@dataclass(frozen=True)
class DialogueLine:
    key: str
    index: int
    text: str
    sha256: str


@dataclass(frozen=True)
class GenerationTask:
    cache_key: str
    audio_path: Path
    model: ModelIdentity
    image: CharacterImageItem
    line: DialogueLine
    settings: CharacterBatchGenerationSettings


@dataclass
class PreviewEntry:
    cache_key: str
    audio_path: str
    model_key: str
    model_label: str
    raw_checkpoint: str
    resolved_checkpoint: str
    checkpoint_fingerprint: str
    image_key: str
    image_path: str
    image_name: str
    image_sha256: str
    line_key: str
    text_index: int
    text: str
    text_sha256: str
    seed: int
    settings: dict[str, Any]
    sample_rate: int
    used_seed: int
    created_at: str
    messages: list[str] = field(default_factory=list)
    stage_timings: list[list[Any]] = field(default_factory=list)
    total_to_decode: float = 0.0
    preset_version: str = GENERATION_PRESET_VERSION

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PreviewEntry:
        return cls(
            cache_key=str(raw["cache_key"]),
            audio_path=str(raw["audio_path"]),
            model_key=str(raw["model_key"]),
            model_label=str(raw["model_label"]),
            raw_checkpoint=str(raw.get("raw_checkpoint", raw["model_label"])),
            resolved_checkpoint=str(raw.get("resolved_checkpoint", "")),
            checkpoint_fingerprint=str(raw.get("checkpoint_fingerprint", raw["model_key"])),
            image_key=str(raw["image_key"]),
            image_path=str(raw["image_path"]),
            image_name=str(raw.get("image_name", Path(str(raw["image_path"])).name)),
            image_sha256=str(raw.get("image_sha256", "")),
            line_key=str(raw.get("line_key", dialogue_line_key(raw["text_index"], raw["text"]))),
            text_index=int(raw["text_index"]),
            text=str(raw["text"]),
            text_sha256=str(raw.get("text_sha256", _sha256_text(str(raw["text"])))),
            seed=int(raw.get("seed", DEFAULT_SEED)),
            settings=dict(raw.get("settings", {})),
            sample_rate=int(raw.get("sample_rate", 0)),
            used_seed=int(raw.get("used_seed", raw.get("seed", DEFAULT_SEED))),
            created_at=str(raw.get("created_at", "")),
            messages=list(raw.get("messages", [])),
            stage_timings=[list(item) for item in raw.get("stage_timings", [])],
            total_to_decode=float(raw.get("total_to_decode", 0.0)),
            preset_version=str(raw.get("preset_version", GENERATION_PRESET_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CharacterBatchPreviewStore:
    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.index_path = self.output_dir / "index.json"
        self.entries: list[PreviewEntry] = []

    def load(self) -> CharacterBatchPreviewStore:
        self.entries = []
        if not self.index_path.is_file():
            return self
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        raw_entries = payload.get("entries", []) if isinstance(payload, dict) else []
        for raw in raw_entries:
            if not isinstance(raw, dict):
                continue
            try:
                self.entries.append(PreviewEntry.from_dict(raw))
            except (KeyError, TypeError, ValueError):
                continue
        return self

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": _utc_now_iso(),
            "entries": [entry.to_dict() for entry in self.entries],
        }
        tmp_path = self.index_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.index_path)

    def existing_entry(self, cache_key: str) -> PreviewEntry | None:
        for entry in self.entries:
            if entry.cache_key == cache_key and Path(entry.audio_path).is_file():
                return entry
        return None

    def available_entries(self) -> list[PreviewEntry]:
        return [entry for entry in self.entries if Path(entry.audio_path).is_file()]

    def upsert(self, entry: PreviewEntry) -> None:
        for idx, current in enumerate(self.entries):
            if current.cache_key == entry.cache_key:
                self.entries[idx] = entry
                return
        self.entries.append(entry)


def _sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(raw)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(value: str, *, max_len: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    slug = slug.strip("._-")
    return (slug or "item")[:max_len]


def dialogue_line_key(index: int, text: str) -> str:
    return _stable_hash({"index": int(index), "text": str(text)})[:16]


def read_dialogue_lines(path: str | Path) -> list[DialogueLine]:
    text_path = Path(path).expanduser()
    if not text_path.is_file():
        raise FileNotFoundError(f"Dialogue text file not found: {text_path}")

    lines: list[DialogueLine] = []
    for idx, raw_line in enumerate(text_path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw_line.strip()
        if text == "":
            continue
        text_sha256 = _sha256_text(text)
        lines.append(
            DialogueLine(
                key=dialogue_line_key(idx, text),
                index=idx,
                text=text,
                sha256=text_sha256,
            )
        )
    if not lines:
        raise ValueError(f"Dialogue text file has no non-empty lines: {text_path}")
    return lines


def collect_character_images(image_dir: str | Path) -> list[CharacterImageItem]:
    root = Path(image_dir).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"Character image directory not found: {root}")

    image_paths = sorted(
        [
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.casefold(),
    )
    if not image_paths:
        extensions = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(f"No supported images found in {root}. Supported: {extensions}")

    items: list[CharacterImageItem] = []
    for path in image_paths:
        resolved = path.resolve()
        image_sha256 = _sha256_bytes(resolved)
        image_key = _stable_hash({"path": str(resolved), "sha256": image_sha256})[:16]
        items.append(
            CharacterImageItem(
                key=image_key,
                path=resolved,
                name=resolved.name,
                sha256=image_sha256,
            )
        )
    return items


def resolve_checkpoint_path(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise ValueError("checkpoint is required.")

    path = Path(checkpoint).expanduser()
    if path.suffix.lower() in {".pt", ".safetensors"}:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return str(path.resolve())

    resolved = hf_hub_download(repo_id=checkpoint, filename="model.safetensors")
    return str(Path(resolved).resolve())


def identify_model(raw_checkpoint: str, resolved_checkpoint: str) -> ModelIdentity:
    resolved_path = Path(resolved_checkpoint).expanduser()
    label = str(raw_checkpoint).strip()
    payload: dict[str, Any] = {
        "raw_checkpoint": label,
        "resolved_checkpoint": str(resolved_path),
    }
    if resolved_path.exists():
        stat = resolved_path.stat()
        payload.update(
            {
                "resolved_checkpoint": str(resolved_path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    fingerprint = _stable_hash(payload)
    return ModelIdentity(
        key=fingerprint[:16],
        label=label,
        raw_checkpoint=label,
        resolved_checkpoint=str(resolved_path),
        fingerprint=fingerprint,
    )


def build_runtime_key(
    *,
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    codec_repo: str = DEFAULT_CODEC_REPO,
) -> RuntimeKey:
    return RuntimeKey(
        checkpoint=checkpoint,
        model_device=str(model_device),
        codec_repo=str(codec_repo),
        model_precision=str(model_precision),
        codec_device=str(codec_device),
        codec_precision=str(codec_precision),
        compile_model=False,
        compile_dynamic=False,
    )


def build_cache_key(
    *,
    model: ModelIdentity,
    image: CharacterImageItem,
    line: DialogueLine,
    settings: CharacterBatchGenerationSettings,
) -> str:
    return _stable_hash(
        {
            "preset_version": GENERATION_PRESET_VERSION,
            "model_fingerprint": model.fingerprint,
            "image_key": image.key,
            "image_sha256": image.sha256,
            "line_key": line.key,
            "text_sha256": line.sha256,
            "settings": asdict(settings),
        }
    )


def build_audio_path(
    *,
    output_dir: str | Path,
    model: ModelIdentity,
    image: CharacterImageItem,
    line: DialogueLine,
    settings: CharacterBatchGenerationSettings,
    cache_key: str,
) -> Path:
    model_dir = f"{slugify(model.label)}-{model.key}"
    image_dir = f"{slugify(image.path.stem)}-{image.key}"
    filename = (
        f"line{line.index:04d}_{line.sha256[:10]}_seed{int(settings.seed)}_{cache_key[:10]}.wav"
    )
    return Path(output_dir).expanduser() / "audio" / model_dir / image_dir / filename


def build_generation_tasks(
    *,
    model: ModelIdentity,
    images: list[CharacterImageItem],
    lines: list[DialogueLine],
    settings: CharacterBatchGenerationSettings,
    store: CharacterBatchPreviewStore,
) -> tuple[list[GenerationTask], list[PreviewEntry]]:
    pending: list[GenerationTask] = []
    cached: list[PreviewEntry] = []
    for image in images:
        for line in lines:
            cache_key = build_cache_key(
                model=model,
                image=image,
                line=line,
                settings=settings,
            )
            existing = store.existing_entry(cache_key)
            if existing is not None:
                cached.append(existing)
                continue
            pending.append(
                GenerationTask(
                    cache_key=cache_key,
                    audio_path=build_audio_path(
                        output_dir=store.output_dir,
                        model=model,
                        image=image,
                        line=line,
                        settings=settings,
                        cache_key=cache_key,
                    ),
                    model=model,
                    image=image,
                    line=line,
                    settings=settings,
                )
            )
    return pending, cached


def task_to_request(task: GenerationTask) -> SamplingRequest:
    settings = task.settings
    return SamplingRequest(
        text=task.line.text,
        caption=None,
        ref_wav=None,
        ref_latent=None,
        no_ref=True,
        ref_normalize_db=-16.0,
        ref_ensure_max=True,
        num_candidates=1,
        decode_mode="sequential",
        seconds=None,
        duration_scale=float(settings.duration_scale),
        max_ref_seconds=30.0,
        max_text_len=None,
        max_caption_len=None,
        character_image=str(task.image.path),
        num_steps=int(settings.num_steps),
        cfg_scale_text=float(settings.cfg_scale_text),
        cfg_scale_caption=0.0,
        cfg_scale_speaker=0.0,
        cfg_scale_character=float(settings.cfg_scale_character),
        cfg_guidance_mode=str(settings.cfg_guidance_mode),
        cfg_scale=None,
        cfg_min_t=0.5,
        cfg_max_t=1.0,
        truncation_factor=None,
        rescale_k=None,
        rescale_sigma=None,
        context_kv_cache=bool(settings.context_kv_cache),
        speaker_kv_scale=None,
        speaker_kv_min_t=None,
        speaker_kv_max_layers=None,
        seed=int(settings.seed),
        t_schedule_mode=str(settings.t_schedule_mode),
        sway_coeff=float(settings.sway_coeff),
        trim_tail=True,
    )


def preview_entry_from_result(
    *,
    task: GenerationTask,
    sample_rate: int,
    used_seed: int,
    messages: list[str],
    stage_timings: list[tuple[str, float]],
    total_to_decode: float,
) -> PreviewEntry:
    return PreviewEntry(
        cache_key=task.cache_key,
        audio_path=str(task.audio_path.resolve()),
        model_key=task.model.key,
        model_label=task.model.label,
        raw_checkpoint=task.model.raw_checkpoint,
        resolved_checkpoint=task.model.resolved_checkpoint,
        checkpoint_fingerprint=task.model.fingerprint,
        image_key=task.image.key,
        image_path=str(task.image.path),
        image_name=task.image.name,
        image_sha256=task.image.sha256,
        line_key=task.line.key,
        text_index=int(task.line.index),
        text=task.line.text,
        text_sha256=task.line.sha256,
        seed=int(task.settings.seed),
        settings=asdict(task.settings),
        sample_rate=int(sample_rate),
        used_seed=int(used_seed),
        created_at=_utc_now_iso(),
        messages=list(messages),
        stage_timings=[[name, float(sec)] for name, sec in stage_timings],
        total_to_decode=float(total_to_decode),
    )


def synthesize_character_batch(
    *,
    checkpoint: str,
    image_dir: str | Path,
    text_file: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    codec_repo: str = DEFAULT_CODEC_REPO,
    settings: CharacterBatchGenerationSettings | None = None,
    log_fn: Callable[[str], None] | None = None,
    progress_fn: Callable[[int, int, GenerationTask], None] | None = None,
) -> tuple[list[PreviewEntry], list[PreviewEntry], list[str]]:
    settings = CharacterBatchGenerationSettings() if settings is None else settings
    store = CharacterBatchPreviewStore(output_dir).load()

    resolved_checkpoint = resolve_checkpoint_path(checkpoint)
    model = identify_model(checkpoint, resolved_checkpoint)
    images = collect_character_images(image_dir)
    lines = read_dialogue_lines(text_file)
    pending, cached = build_generation_tasks(
        model=model,
        images=images,
        lines=lines,
        settings=settings,
        store=store,
    )
    if not pending:
        return (
            [],
            cached,
            [f"cache hit: {len(cached)} audio files already exist for this model/image/text set."],
        )

    runtime_key = build_runtime_key(
        checkpoint=resolved_checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        codec_repo=codec_repo,
    )
    runtime, reloaded = get_cached_runtime(runtime_key)
    if not runtime.model_cfg.use_character_condition:
        raise ValueError(
            "Loaded checkpoint does not enable character conditioning. "
            "Use a character-reference checkpoint for this preview tool."
        )

    messages = [
        f"runtime: {'reloaded' if reloaded else 'reused'}",
        f"model: {model.label}",
        f"images: {len(images)}",
        f"dialogue lines: {len(lines)}",
        f"pending: {len(pending)}",
        f"cached: {len(cached)}",
    ]
    generated: list[PreviewEntry] = []
    total = len(pending)
    for idx, task in enumerate(pending, start=1):
        if progress_fn is not None:
            progress_fn(idx, total, task)
        if log_fn is not None:
            log_fn(
                f"[character-batch] {idx}/{total} image={task.image.name} line={task.line.index} seed={task.settings.seed}"
            )

        result = runtime.synthesize(task_to_request(task), log_fn=log_fn)
        task.audio_path.parent.mkdir(parents=True, exist_ok=True)
        save_wav(task.audio_path, result.audios[0].float(), result.sample_rate)
        entry = preview_entry_from_result(
            task=task,
            sample_rate=result.sample_rate,
            used_seed=result.used_seed,
            messages=result.messages,
            stage_timings=result.stage_timings,
            total_to_decode=result.total_to_decode,
        )
        store.upsert(entry)
        store.save()
        generated.append(entry)

    messages.append(f"generated: {len(generated)}")
    messages.append(f"output_dir: {store.output_dir}")
    return generated, cached, messages


def load_preview_entries(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> list[PreviewEntry]:
    return CharacterBatchPreviewStore(output_dir).load().available_entries()


def library_rows(entries: list[PreviewEntry], *, limit: int = 500) -> list[list[Any]]:
    sorted_entries = sorted(
        entries,
        key=lambda entry: (
            entry.image_name.casefold(),
            int(entry.text_index),
            entry.model_label.casefold(),
            entry.created_at,
        ),
    )
    rows: list[list[Any]] = []
    for entry in sorted_entries[-limit:]:
        rows.append(
            [
                entry.created_at,
                entry.model_label,
                entry.image_name,
                entry.text_index,
                entry.text,
                entry.used_seed,
                entry.audio_path,
            ]
        )
    return rows
