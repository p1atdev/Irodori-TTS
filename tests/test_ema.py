# ruff: noqa: E402
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train
from irodori_tts.config import ModelConfig, TrainConfig


def test_model_ema_updates_and_temporarily_applies_weights() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    ema = train.ModelEMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)

    assert torch.equal(model.weight, torch.full_like(model.weight, 3.0))
    assert torch.equal(ema.shadow["weight"], torch.full_like(model.weight, 2.0))

    with ema.apply_to(model):
        assert torch.equal(model.weight, torch.full_like(model.weight, 2.0))

    assert torch.equal(model.weight, torch.full_like(model.weight, 3.0))
    saved_state = ema.state_dict_for_save(model)
    assert torch.equal(saved_state["weight"], torch.full_like(model.weight, 2.0))


def test_save_checkpoint_pair_writes_regular_and_ema_weights(tmp_path: Path) -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_cfg = TrainConfig(ema_enabled=True)
    ema = train.ModelEMA(model, decay=0.5)

    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)

    path = tmp_path / "checkpoint_0000001.pt"
    train.save_checkpoint_pair(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        step=1,
        model_cfg=ModelConfig(),
        train_cfg=train_cfg,
        base_init=None,
        ema=ema,
    )

    ema_path = tmp_path / "checkpoint_0000001_ema.pt"
    assert path.is_file()
    assert ema_path.is_file()

    regular_payload = train._load_checkpoint_payload(path, map_location="cpu")
    ema_payload = train._load_checkpoint_payload(ema_path, map_location="cpu")

    assert torch.equal(regular_payload["model"]["weight"], torch.full((1, 1), 3.0))
    assert torch.equal(ema_payload["model"]["weight"], torch.full((1, 1), 2.0))
    assert ema_payload["checkpoint_variant"] == "ema"
    assert ema_payload["ema_decay"] == 0.5


def test_validation_model_context_uses_ema_weights() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = train.ModelEMA(model, decay=0.5)

    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)

    with train._validation_model_context(ema, model):
        assert torch.equal(model.weight, torch.full_like(model.weight, 2.0))

    assert torch.equal(model.weight, torch.full_like(model.weight, 3.0))


def test_periodic_checkpoint_retention_removes_ema_pair(tmp_path: Path) -> None:
    keep_path = tmp_path / "checkpoint_0000002.pt"
    stale_path = tmp_path / "checkpoint_0000001.pt"
    keep_path.write_bytes(b"regular")
    (tmp_path / "checkpoint_0000002_ema.pt").write_bytes(b"ema")
    stale_path.write_bytes(b"regular")
    (tmp_path / "checkpoint_0000001_ema.pt").write_bytes(b"ema")

    train.enforce_periodic_checkpoint_limit(tmp_path, keep_count=1)

    assert keep_path.exists()
    assert (tmp_path / "checkpoint_0000002_ema.pt").exists()
    assert not stale_path.exists()
    assert not (tmp_path / "checkpoint_0000001_ema.pt").exists()
