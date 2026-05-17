#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gradio as gr

from irodori_tts.character_batch_preview import (
    DEFAULT_OUTPUT_DIR,
    CharacterBatchGenerationSettings,
    PreviewEntry,
    load_preview_entries,
    synthesize_character_batch,
)
from irodori_tts.inference_runtime import (
    clear_cached_runtime,
    default_runtime_device,
    list_available_runtime_devices,
    list_available_runtime_precisions,
)

LIBRARY_HEADERS = ["Created", "Model", "Image", "Line", "Text", "Seed", "Audio"]
DETAIL_HEADERS = ["Model", "Created", "Audio", "Duration Decode", "Messages"]


def _default_checkpoint() -> str:
    candidates: list[Path] = []
    for root in [Path("."), Path("outputs")]:
        if not root.exists():
            continue
        candidates.extend(root.glob("checkpoint_*.pt"))
        candidates.extend(root.glob("checkpoint_*.safetensors"))
        candidates.extend(root.glob("*/checkpoint_*.pt"))
        candidates.extend(root.glob("*/checkpoint_*.safetensors"))
    candidates = sorted(set(candidates))
    preferred = [path for path in candidates if "character" in str(path).lower()]
    if preferred:
        return str(preferred[-1])
    if candidates:
        return str(candidates[-1])
    return ""


def _default_model_device() -> str:
    return default_runtime_device()


def _default_codec_device() -> str:
    return default_runtime_device()


def _precision_choices_for_device(device: str) -> list[str]:
    return list_available_runtime_precisions(device)


def _on_model_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[-1])


def _on_codec_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[-1])


def _output_dir(raw: str | None) -> str:
    value = str(raw or "").strip()
    return value or str(DEFAULT_OUTPUT_DIR)


def _parse_seed(raw: str | int | float | None) -> int:
    if raw is None:
        return 0
    value = str(raw).strip()
    if value == "":
        return 0
    return int(value)


def _short_text(text: str, limit: int = 72) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def _model_choices(entries: list[PreviewEntry]) -> list[tuple[str, str]]:
    models: dict[str, str] = {}
    for entry in entries:
        models.setdefault(entry.model_key, f"{entry.model_label} [{entry.model_key[:8]}]")
    return sorted([(label, key) for key, label in models.items()], key=lambda item: item[0])


def _image_choices(entries: list[PreviewEntry]) -> list[tuple[str, str]]:
    images: dict[str, str] = {}
    for entry in entries:
        images.setdefault(entry.image_key, entry.image_name)
    return sorted([(label, key) for key, label in images.items()], key=lambda item: item[0])


def _line_choices(entries: list[PreviewEntry], image_key: str | None) -> list[tuple[str, str]]:
    if not image_key:
        return []
    lines: dict[str, tuple[int, str]] = {}
    for entry in entries:
        if entry.image_key == image_key:
            lines.setdefault(entry.line_key, (entry.text_index, entry.text))
    return [
        (f"{idx}: {_short_text(text)}", line_key)
        for line_key, (idx, text) in sorted(lines.items(), key=lambda item: item[1][0])
    ]


def _valid_values(choices: list[tuple[str, str]], selected: list[str] | None) -> list[str]:
    allowed = {value for _label, value in choices}
    return [value for value in selected or [] if value in allowed]


def _first_value(choices: list[tuple[str, str]]) -> str | None:
    return choices[0][1] if choices else None


def _selected_or_first(choices: list[tuple[str, str]], selected: str | None) -> str | None:
    allowed = {value for _label, value in choices}
    if selected in allowed:
        return selected
    return _first_value(choices)


def _pair_entries(
    entries: list[PreviewEntry],
    *,
    image_key: str | None,
    line_key: str | None,
    model_keys: list[str] | None,
) -> list[PreviewEntry]:
    allowed_models = set(model_keys or [])
    if not allowed_models:
        return []
    pair = [
        entry
        for entry in entries
        if entry.image_key == image_key
        and entry.line_key == line_key
        and entry.model_key in allowed_models
        and Path(entry.audio_path).is_file()
    ]
    return sorted(pair, key=lambda entry: (entry.model_label.casefold(), entry.created_at))


def _audio_choice_label(entry: PreviewEntry) -> str:
    return f"{entry.model_label} [{entry.model_key[:8]}] seed={entry.used_seed}"


def _audio_choices(entries: list[PreviewEntry]) -> list[tuple[str, str]]:
    return [(_audio_choice_label(entry), entry.cache_key) for entry in entries]


def _selected_audio_entry(
    entries: list[PreviewEntry],
    selected_audio_key: str | None,
) -> PreviewEntry | None:
    if not entries:
        return None
    for entry in entries:
        if entry.cache_key == selected_audio_key:
            return entry
    return entries[0]


def _preview_values(
    entries: list[PreviewEntry],
    *,
    image_key: str | None,
    line_key: str | None,
    model_keys: list[str] | None,
    selected_audio_key: str | None = None,
) -> tuple[object, str, object, object, list[list[Any]]]:
    pair = _pair_entries(
        entries,
        image_key=image_key,
        line_key=line_key,
        model_keys=model_keys,
    )
    if not pair:
        return (
            gr.update(value=None, visible=False),
            "",
            gr.update(choices=[], value=None),
            gr.update(value=None, visible=False),
            [],
        )

    selected_entry = _selected_audio_entry(pair, selected_audio_key)
    selected_audio_value = selected_entry.cache_key if selected_entry is not None else None
    selected_audio_path = selected_entry.audio_path if selected_entry is not None else None

    detail_rows = [
        [
            _audio_choice_label(entry),
            entry.created_at,
            entry.audio_path,
            f"{entry.total_to_decode:.3f}s",
            "\n".join(entry.messages),
        ]
        for entry in pair
    ]
    return (
        gr.update(value=pair[0].image_path, visible=True),
        pair[0].text,
        gr.update(choices=_audio_choices(pair), value=selected_audio_value),
        gr.update(
            value=selected_audio_path,
            label=_audio_choice_label(selected_entry) if selected_entry is not None else "Audio",
            visible=selected_entry is not None,
        ),
        detail_rows,
    )


def _load_library_metadata(
    output_dir: str | None,
    selected_models: list[str] | None,
) -> tuple[object, ...]:
    entries = load_preview_entries(_output_dir(output_dir), verify_files=False)
    model_choices = _model_choices(entries)
    model_values = _valid_values(model_choices, selected_models)
    image_choices = _image_choices(entries)
    status = f"loaded metadata for {len(entries)} cached audio files from {_output_dir(output_dir)}"
    return (
        status,
        [],
        gr.update(choices=model_choices, value=model_values),
        gr.update(choices=image_choices, value=None),
        gr.update(choices=[], value=None),
        gr.update(value=None, visible=False),
        "",
        gr.update(choices=[], value=None),
        gr.update(value=None, visible=False),
        [],
    )


def _select_image(
    output_dir: str | None,
    selected_image: str | None,
    selected_models: list[str] | None,
) -> tuple[object, ...]:
    entries = load_preview_entries(_output_dir(output_dir), verify_files=False)
    line_choices = _line_choices(entries, selected_image)
    line_value = _first_value(line_choices)
    image_preview, text_preview, audio_model_update, audio_update, detail_rows = _preview_values(
        entries,
        image_key=selected_image,
        line_key=line_value,
        model_keys=selected_models,
        selected_audio_key=None,
    )
    return (
        gr.update(choices=line_choices, value=line_value),
        image_preview,
        text_preview,
        audio_model_update,
        audio_update,
        detail_rows,
    )


def _preview_pair(
    output_dir: str | None,
    selected_image: str | None,
    selected_line: str | None,
    selected_models: list[str] | None,
    selected_audio_key: str | None,
) -> tuple[object, ...]:
    entries = load_preview_entries(_output_dir(output_dir), verify_files=False)
    image_preview, text_preview, audio_model_update, audio_update, detail_rows = _preview_values(
        entries,
        image_key=selected_image,
        line_key=selected_line,
        model_keys=selected_models,
        selected_audio_key=selected_audio_key,
    )
    return (image_preview, text_preview, audio_model_update, audio_update, detail_rows)


def _load_selected_audio(
    output_dir: str | None,
    selected_image: str | None,
    selected_line: str | None,
    selected_models: list[str] | None,
    selected_audio_key: str | None,
) -> object:
    entries = load_preview_entries(_output_dir(output_dir), verify_files=False)
    pair = _pair_entries(
        entries,
        image_key=selected_image,
        line_key=selected_line,
        model_keys=selected_models,
    )
    selected_entry = _selected_audio_entry(pair, selected_audio_key)
    if selected_entry is None:
        return gr.update(value=None, visible=False)
    return gr.update(
        value=selected_entry.audio_path,
        label=_audio_choice_label(selected_entry),
        visible=True,
    )


def _generate_batch(
    checkpoint: str,
    image_dir: str,
    text_file: str,
    output_dir: str,
    seed_raw: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    progress: gr.Progress = gr.Progress(),
) -> str:
    seed = _parse_seed(seed_raw)

    def log_fn(message: str) -> None:
        print(message, flush=True)

    def progress_fn(idx: int, total: int, task) -> None:
        progress(
            (idx - 1, total),
            desc=f"{task.image.name} / line {task.line.index}",
        )

    generated, cached, messages = synthesize_character_batch(
        checkpoint=checkpoint,
        image_dir=image_dir,
        text_file=text_file,
        output_dir=_output_dir(output_dir),
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        settings=CharacterBatchGenerationSettings(seed=seed),
        log_fn=log_fn,
        progress_fn=progress_fn,
    )
    progress(1.0, desc="done")

    status_lines = [
        *messages,
        f"generated_now: {len(generated)}",
        f"cache_hits: {len(cached)}",
        "library_selectors: press Refresh Library when you want to update selectors.",
    ]
    return "\n".join(status_lines)


def _clear_runtime_cache() -> str:
    clear_cached_runtime()
    return "cleared loaded model from memory"


def build_ui() -> gr.Blocks:
    default_checkpoint = _default_checkpoint()
    default_model_device = _default_model_device()
    default_codec_device = _default_codec_device()
    device_choices = list_available_runtime_devices()
    model_precision_choices = _precision_choices_for_device(default_model_device)
    codec_precision_choices = _precision_choices_for_device(default_codec_device)

    with gr.Blocks(title="Irodori-TTS Character Batch Preview") as demo:
        gr.Markdown("# Character Batch Preview")

        with gr.Row():
            checkpoint = gr.Textbox(
                label="Checkpoint (.pt/.safetensors or HF repo id)",
                value=default_checkpoint,
                scale=4,
            )
            image_dir = gr.Textbox(label="Image Directory", value="data/batch_samples", scale=2)
            text_file = gr.Textbox(label="Dialogue TXT", value="data/batch_dialogues.txt", scale=2)

        with gr.Row():
            output_dir = gr.Textbox(
                label="Preview Output Directory",
                value=str(DEFAULT_OUTPUT_DIR),
                scale=3,
            )
            seed_raw = gr.Textbox(label="Fixed Seed", value="0", scale=1)
            model_device = gr.Dropdown(
                label="Model Device",
                choices=device_choices,
                value=default_model_device,
                scale=1,
            )
            model_precision = gr.Dropdown(
                label="Model Precision",
                choices=model_precision_choices,
                value=model_precision_choices[-1],
                scale=1,
            )
            codec_device = gr.Dropdown(
                label="Codec Device",
                choices=device_choices,
                value=default_codec_device,
                scale=1,
            )
            codec_precision = gr.Dropdown(
                label="Codec Precision",
                choices=codec_precision_choices,
                value=codec_precision_choices[-1],
                scale=1,
            )

        with gr.Row():
            generate_btn = gr.Button("Generate Missing", variant="primary")
            refresh_btn = gr.Button("Refresh Library")
            clear_cache_btn = gr.Button("Unload Model")
        status = gr.Textbox(label="Status", lines=7, interactive=False)

        with gr.Row():
            model_filter = gr.Dropdown(
                label="Compare Models",
                choices=[],
                value=[],
                multiselect=True,
                scale=2,
            )
            image_select = gr.Dropdown(label="Image", choices=[], value=None, scale=1)
            line_select = gr.Dropdown(label="Dialogue", choices=[], value=None, scale=2)

        with gr.Row():
            image_preview = gr.Image(
                label="Reference Image",
                type="filepath",
                interactive=False,
                visible=False,
                height=280,
                scale=1,
            )
            text_preview = gr.Textbox(label="Dialogue Text", lines=4, interactive=False, scale=2)

        with gr.Row():
            audio_model_select = gr.Dropdown(
                label="Audio Model",
                choices=[],
                value=None,
                scale=2,
            )
            audio_preview = gr.Audio(
                label="Audio",
                type="filepath",
                interactive=False,
                visible=False,
                scale=3,
            )

        detail_table = gr.Dataframe(
            headers=DETAIL_HEADERS,
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
            label="Selected Pair",
        )
        library_table = gr.Dataframe(
            headers=LIBRARY_HEADERS,
            datatype=["str", "str", "str", "number", "str", "number", "str"],
            interactive=False,
            wrap=True,
            label="Library",
        )

        refresh_outputs = [
            status,
            library_table,
            model_filter,
            image_select,
            line_select,
            image_preview,
            text_preview,
            audio_model_select,
            audio_preview,
            detail_table,
        ]
        preview_outputs = [
            image_preview,
            text_preview,
            audio_model_select,
            audio_preview,
            detail_table,
        ]

        generate_btn.click(
            _generate_batch,
            inputs=[
                checkpoint,
                image_dir,
                text_file,
                output_dir,
                seed_raw,
                model_device,
                model_precision,
                codec_device,
                codec_precision,
            ],
            outputs=[status],
        )
        refresh_btn.click(
            _load_library_metadata,
            inputs=[output_dir, model_filter],
            outputs=refresh_outputs,
        )
        image_select.change(
            _select_image,
            inputs=[output_dir, image_select, model_filter],
            outputs=[line_select, *preview_outputs],
        )
        line_select.change(
            _preview_pair,
            inputs=[output_dir, image_select, line_select, model_filter, audio_model_select],
            outputs=preview_outputs,
        )
        model_filter.change(
            _preview_pair,
            inputs=[output_dir, image_select, line_select, model_filter, audio_model_select],
            outputs=preview_outputs,
        )
        audio_model_select.change(
            _load_selected_audio,
            inputs=[
                output_dir,
                image_select,
                line_select,
                model_filter,
                audio_model_select,
            ],
            outputs=[audio_preview],
        )
        model_device.change(
            _on_model_device_change,
            inputs=[model_device],
            outputs=[model_precision],
        )
        codec_device.change(
            _on_codec_device_change,
            inputs=[codec_device],
            outputs=[codec_precision],
        )
        clear_cache_btn.click(_clear_runtime_cache, outputs=[status])

        demo.load(
            _load_library_metadata,
            inputs=[output_dir, model_filter],
            outputs=refresh_outputs,
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gradio batch preview app for character-reference Irodori-TTS checkpoints."
    )
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7863)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=bool(args.share),
        debug=bool(args.debug),
    )


if __name__ == "__main__":
    main()
