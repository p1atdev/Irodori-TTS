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
    library_rows,
    load_preview_entries,
    synthesize_character_batch,
)
from irodori_tts.inference_runtime import (
    clear_cached_runtime,
    default_runtime_device,
    list_available_runtime_devices,
    list_available_runtime_precisions,
)

MAX_COMPARE_MODELS = 12
LIBRARY_HEADERS = ["Created", "Model", "Image", "Line", "Text", "Seed", "Audio"]
DETAIL_HEADERS = ["Model", "Created", "Audio", "Duration Decode", "Messages"]


def _default_checkpoint() -> str:
    candidates = sorted(
        [
            *Path(".").glob("**/checkpoint_*.pt"),
            *Path(".").glob("**/checkpoint_*.safetensors"),
        ]
    )
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
    pair = [
        entry
        for entry in entries
        if entry.image_key == image_key
        and entry.line_key == line_key
        and (not allowed_models or entry.model_key in allowed_models)
    ]
    return sorted(pair, key=lambda entry: (entry.model_label.casefold(), entry.created_at))


def _empty_audio_updates() -> list[object]:
    return [gr.update(value=None, visible=False) for _ in range(MAX_COMPARE_MODELS)]


def _preview_values(
    entries: list[PreviewEntry],
    *,
    image_key: str | None,
    line_key: str | None,
    model_keys: list[str] | None,
) -> tuple[object, str, list[object], list[list[Any]]]:
    pair = _pair_entries(
        entries,
        image_key=image_key,
        line_key=line_key,
        model_keys=model_keys,
    )
    if not pair:
        return gr.update(value=None, visible=False), "", _empty_audio_updates(), []

    audio_updates: list[object] = []
    for idx in range(MAX_COMPARE_MODELS):
        if idx < len(pair):
            entry = pair[idx]
            audio_updates.append(
                gr.update(
                    value=entry.audio_path,
                    label=f"{entry.model_label} [{entry.model_key[:8]}]",
                    visible=True,
                )
            )
        else:
            audio_updates.append(gr.update(value=None, visible=False))

    detail_rows = [
        [
            f"{entry.model_label} [{entry.model_key[:8]}]",
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
        audio_updates,
        detail_rows,
    )


def _library_state(
    output_dir: str | None,
    *,
    selected_models: list[str] | None = None,
    selected_image: str | None = None,
    selected_line: str | None = None,
) -> tuple[
    list[list[Any]],
    object,
    object,
    object,
    object,
    str,
    list[object],
    list[list[Any]],
    int,
]:
    entries = load_preview_entries(_output_dir(output_dir))
    model_choices = _model_choices(entries)
    model_values = _valid_values(model_choices, selected_models)
    image_choices = _image_choices(entries)
    image_value = _selected_or_first(image_choices, selected_image)
    line_choices = _line_choices(entries, image_value)
    line_value = _selected_or_first(line_choices, selected_line)
    image_preview, text_preview, audio_updates, detail_rows = _preview_values(
        entries,
        image_key=image_value,
        line_key=line_value,
        model_keys=model_values,
    )
    return (
        library_rows(entries),
        gr.update(choices=model_choices, value=model_values),
        gr.update(choices=image_choices, value=image_value),
        gr.update(choices=line_choices, value=line_value),
        image_preview,
        text_preview,
        audio_updates,
        detail_rows,
        len(entries),
    )


def _refresh_library(
    output_dir: str | None,
    selected_models: list[str] | None,
    selected_image: str | None,
    selected_line: str | None,
) -> tuple[object, ...]:
    (
        rows,
        model_update,
        image_update,
        line_update,
        image_preview,
        text_preview,
        audio_updates,
        detail_rows,
        entry_count,
    ) = _library_state(
        output_dir,
        selected_models=selected_models,
        selected_image=selected_image,
        selected_line=selected_line,
    )
    status = f"loaded {entry_count} cached audio files from {_output_dir(output_dir)}"
    return (
        status,
        rows,
        model_update,
        image_update,
        line_update,
        image_preview,
        text_preview,
        *audio_updates,
        detail_rows,
    )


def _select_image(
    output_dir: str | None,
    selected_image: str | None,
    selected_models: list[str] | None,
) -> tuple[object, ...]:
    entries = load_preview_entries(_output_dir(output_dir))
    line_choices = _line_choices(entries, selected_image)
    line_value = _first_value(line_choices)
    image_preview, text_preview, audio_updates, detail_rows = _preview_values(
        entries,
        image_key=selected_image,
        line_key=line_value,
        model_keys=selected_models,
    )
    return (
        gr.update(choices=line_choices, value=line_value),
        image_preview,
        text_preview,
        *audio_updates,
        detail_rows,
    )


def _preview_pair(
    output_dir: str | None,
    selected_image: str | None,
    selected_line: str | None,
    selected_models: list[str] | None,
) -> tuple[object, ...]:
    entries = load_preview_entries(_output_dir(output_dir))
    image_preview, text_preview, audio_updates, detail_rows = _preview_values(
        entries,
        image_key=selected_image,
        line_key=selected_line,
        model_keys=selected_models,
    )
    return (image_preview, text_preview, *audio_updates, detail_rows)


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
    selected_models: list[str] | None,
    selected_image: str | None,
    selected_line: str | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[object, ...]:
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

    preview_seed = generated[0] if generated else (cached[0] if cached else None)
    image_value = selected_image
    line_value = selected_line
    if preview_seed is not None:
        image_value = preview_seed.image_key
        line_value = preview_seed.line_key
    model_values = list(selected_models or [])
    if preview_seed is not None and model_values and preview_seed.model_key not in model_values:
        model_values.append(preview_seed.model_key)

    (
        rows,
        model_update,
        image_update,
        line_update,
        image_preview,
        text_preview,
        audio_updates,
        detail_rows,
        entry_count,
    ) = _library_state(
        output_dir,
        selected_models=model_values,
        selected_image=image_value,
        selected_line=line_value,
    )

    status_lines = [
        *messages,
        f"generated_now: {len(generated)}",
        f"cache_hits: {len(cached)}",
        f"library_total: {entry_count}",
    ]
    return (
        "\n".join(status_lines),
        rows,
        model_update,
        image_update,
        line_update,
        image_preview,
        text_preview,
        *audio_updates,
        detail_rows,
    )


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

        compare_audios: list[gr.Audio] = []
        with gr.Column():
            for row_idx in range(3):
                with gr.Row():
                    for col_idx in range(4):
                        idx = row_idx * 4 + col_idx
                        compare_audios.append(
                            gr.Audio(
                                label=f"Model {idx + 1}",
                                type="filepath",
                                interactive=False,
                                visible=False,
                                min_width=160,
                            )
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
            *compare_audios,
            detail_table,
        ]
        preview_outputs = [image_preview, text_preview, *compare_audios, detail_table]

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
                model_filter,
                image_select,
                line_select,
            ],
            outputs=refresh_outputs,
        )
        refresh_btn.click(
            _refresh_library,
            inputs=[output_dir, model_filter, image_select, line_select],
            outputs=refresh_outputs,
        )
        image_select.change(
            _select_image,
            inputs=[output_dir, image_select, model_filter],
            outputs=[line_select, *preview_outputs],
        )
        line_select.change(
            _preview_pair,
            inputs=[output_dir, image_select, line_select, model_filter],
            outputs=preview_outputs,
        )
        model_filter.change(
            _preview_pair,
            inputs=[output_dir, image_select, line_select, model_filter],
            outputs=preview_outputs,
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
            _refresh_library,
            inputs=[output_dir, model_filter, image_select, line_select],
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
