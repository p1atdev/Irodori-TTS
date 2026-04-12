from __future__ import annotations

# ruff: noqa: E402
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train
from irodori_tts.config import RegexModule, TrainConfig


def test_trainable_modules_resolves_regex_entries() -> None:
    cfg = TrainConfig(
        trainable_modules=[
            "character_encoder.proj",
            {"regex": r"^blocks\.\d+\.attention\.w[kv]_character\.weight$"},
        ]
    )

    assert cfg.trainable_modules_resolved == [
        "character_encoder.proj",
        RegexModule(regex=r"^blocks\.\d+\.attention\.w[kv]_character\.weight$"),
    ]


def test_trainable_modules_rejects_unknown_mapping_keys() -> None:
    cfg = TrainConfig(trainable_modules=[{"prefix": "character_encoder.proj"}])

    with pytest.raises(ValueError, match="only support the key 'regex'"):
        _ = cfg.trainable_modules_resolved


def test_regex_trainable_module_specs_match_parameter_names() -> None:
    specs = train._compile_trainable_module_specs(
        [
            "character_encoder.proj",
            RegexModule(regex=r"^blocks\.\d+\.attention\.w[kv]_character\.weight$"),
        ]
    )

    assert train._matches_trainable_module_spec(
        "character_encoder.proj.blocks.0.fc1.weight",
        specs[0],
    )
    assert train._matches_trainable_module_spec(
        "blocks.7.attention.wk_character.weight",
        specs[1],
    )
    assert train._matches_trainable_module_spec(
        "blocks.7.attention.wv_character.weight",
        specs[1],
    )
    assert not train._matches_trainable_module_spec(
        "blocks.7.attention.wk_text.weight",
        specs[1],
    )
