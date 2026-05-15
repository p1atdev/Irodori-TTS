# ruff: noqa: E402
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from irodori_tts import character_batch_preview as cbp


def _write_fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "b.webp").write_bytes(b"image-b")
    (image_dir / "a.png").write_bytes(b"image-a")
    (image_dir / "ignore.txt").write_text("not an image", encoding="utf-8")

    text_file = tmp_path / "lines.txt"
    text_file.write_text("  こんにちは  \n\n次のセリフです\n", encoding="utf-8")

    checkpoint = tmp_path / "checkpoint_character.safetensors"
    checkpoint.write_bytes(b"model")
    return image_dir, text_file, checkpoint


def test_collect_inputs_and_build_generation_tasks_use_image_text_cross_product(
    tmp_path: Path,
) -> None:
    image_dir, text_file, checkpoint = _write_fixture_inputs(tmp_path)

    images = cbp.collect_character_images(image_dir)
    lines = cbp.read_dialogue_lines(text_file)
    model = cbp.identify_model(str(checkpoint), str(checkpoint))
    store = cbp.CharacterBatchPreviewStore(tmp_path / "preview").load()

    pending, cached = cbp.build_generation_tasks(
        model=model,
        images=images,
        lines=lines,
        settings=cbp.CharacterBatchGenerationSettings(seed=123),
        store=store,
    )

    assert [image.name for image in images] == ["a.png", "b.webp"]
    assert [line.text for line in lines] == ["こんにちは", "次のセリフです"]
    assert len(pending) == 4
    assert cached == []
    assert {task.settings.seed for task in pending} == {123}
    assert all(task.audio_path.suffix == ".wav" for task in pending)


def test_preview_store_reuses_existing_audio_for_same_cache_key(tmp_path: Path) -> None:
    image_dir, text_file, checkpoint = _write_fixture_inputs(tmp_path)

    images = cbp.collect_character_images(image_dir)
    lines = cbp.read_dialogue_lines(text_file)
    model = cbp.identify_model(str(checkpoint), str(checkpoint))
    settings = cbp.CharacterBatchGenerationSettings(seed=0)
    store = cbp.CharacterBatchPreviewStore(tmp_path / "preview").load()

    pending, _cached = cbp.build_generation_tasks(
        model=model,
        images=images,
        lines=lines,
        settings=settings,
        store=store,
    )
    first = pending[0]
    first.audio_path.parent.mkdir(parents=True, exist_ok=True)
    first.audio_path.write_bytes(b"wav")
    entry = cbp.preview_entry_from_result(
        task=first,
        sample_rate=16000,
        used_seed=0,
        messages=["ok"],
        stage_timings=[("sample", 0.1)],
        total_to_decode=0.2,
    )
    store.upsert(entry)
    store.save()

    reloaded = cbp.CharacterBatchPreviewStore(tmp_path / "preview").load()
    pending, cached = cbp.build_generation_tasks(
        model=model,
        images=images,
        lines=lines,
        settings=settings,
        store=reloaded,
    )

    assert len(cached) == 1
    assert cached[0].cache_key == first.cache_key
    assert len(pending) == 3


def test_synthesize_character_batch_generates_missing_and_skips_cached(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_dir, text_file, checkpoint = _write_fixture_inputs(tmp_path)
    output_dir = tmp_path / "preview"
    requests = []

    class FakeRuntime:
        model_cfg = SimpleNamespace(use_character_condition=True)

        def synthesize(self, req, *, log_fn=None):
            requests.append(req)
            return SimpleNamespace(
                audios=[torch.zeros(1, 8)],
                sample_rate=16000,
                used_seed=req.seed,
                messages=["generated"],
                stage_timings=[("sample", 0.01)],
                total_to_decode=0.02,
            )

    def fake_save_wav(path, audio, sample_rate):
        del audio, sample_rate
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"wav")
        return out_path

    monkeypatch.setattr(cbp, "get_cached_runtime", lambda _key: (FakeRuntime(), True))
    monkeypatch.setattr(cbp, "save_wav", fake_save_wav)

    generated, cached, _messages = cbp.synthesize_character_batch(
        checkpoint=str(checkpoint),
        image_dir=image_dir,
        text_file=text_file,
        output_dir=output_dir,
        model_device="cpu",
        model_precision="fp32",
        codec_device="cpu",
        codec_precision="fp32",
        settings=cbp.CharacterBatchGenerationSettings(seed=777),
    )

    assert len(generated) == 4
    assert cached == []
    assert len(requests) == 4
    assert {request.seed for request in requests} == {777}
    assert all(request.character_image for request in requests)

    generated_again, cached_again, _messages = cbp.synthesize_character_batch(
        checkpoint=str(checkpoint),
        image_dir=image_dir,
        text_file=text_file,
        output_dir=output_dir,
        model_device="cpu",
        model_precision="fp32",
        codec_device="cpu",
        codec_precision="fp32",
        settings=cbp.CharacterBatchGenerationSettings(seed=777),
    )

    assert generated_again == []
    assert len(cached_again) == 4
    assert len(requests) == 4
