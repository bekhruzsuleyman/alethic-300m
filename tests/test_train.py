"""Training: exact resume, overfitting one batch, and the --max-cost guard through the CLI."""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F
import yaml

from alethic import cli
from alethic.model.transformer import ModelConfig, Transformer
from alethic.tokenizer.freeze import write_frozen
from alethic.tokenizer.specials import SpecialsRegistry
from alethic.train import checkpoint as ckpt
from alethic.train.loop import RunSpec, TrainError, train
from alethic.train.optim import build_optimizer, lr_at
from conftest import tiny_model_cfg, tiny_train_cfg, write_train_shards


def _spec(tmp_path, run: str, attention: str = "sgca", **kw) -> RunSpec:
    """A RunSpec over shared random shards in tmp_path."""
    shards_dir = tmp_path / "shards"
    if not (shards_dir / "manifest.json").exists():
        write_train_shards(shards_dir, n_tokens=4000, vocab=128)
    shards = sorted((shards_dir / "train" / "python").glob("*.bin"))
    specials = tmp_path / "specials.json"
    SpecialsRegistry(specials)
    return RunSpec(model_cfg=tiny_model_cfg(attention), train_cfg=kw.pop("train_cfg", tiny_train_cfg()),
                   shards=shards, run_dir=tmp_path / "runs" / run, specials_path=specials,
                   tokenizer_sha256="0" * 64, device="cpu", **kw)


def _final_params(run_dir) -> dict:
    state, _ = ckpt.load_checkpoint(ckpt.latest_checkpoint(run_dir / "checkpoints"))
    return state["model"]


@pytest.mark.parametrize("attention", ["sgca", "gqa"])
def test_resume_exact(tmp_path, attention):
    """20 straight steps == 10 steps, resume, 10 more (within 1e-6)."""
    straight = train(_spec(tmp_path, "straight", attention))
    first = train(_spec(tmp_path, "split", attention, stop_step=10))
    assert first["step"] == 10 and first["reason"] == "stop-step"
    second = train(_spec(tmp_path, "split", attention, resume="auto"))
    assert straight["step"] == second["step"] == 20 and second["reason"] == "done"

    losses = first["losses"] + second["losses"]
    assert max(abs(a - b) for a, b in zip(straight["losses"], losses)) <= 1e-6
    a, b = _final_params(tmp_path / "runs" / "straight"), _final_params(tmp_path / "runs" / "split")
    assert a.keys() == b.keys()
    assert max((a[k] - b[k]).abs().max().item() for k in a) <= 1e-6
    # Retention: last two (15, 20) plus the every-10 one (10).
    kept = [s for s, _ in ckpt.list_checkpoints(tmp_path / "runs" / "straight" / "checkpoints")]
    assert kept == [10, 15, 20]
    with pytest.raises(TrainError, match="already has checkpoints"):
        train(_spec(tmp_path, "straight", attention))


@pytest.mark.parametrize("attention", ["sgca", "gqa"])
def test_overfit(attention):
    """A tiny model reaches loss below 0.1 on one repeated batch within 200 steps."""
    torch.manual_seed(0)
    model = Transformer(ModelConfig.from_dict(tiny_model_cfg(attention)))
    opt = build_optimizer(model, dict(tiny_train_cfg()["optim"], peak_lr=1e-2), "cpu")
    batch = torch.randint(0, 128, (4, 33))
    loss_value = float("inf")
    for step in range(200):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, 1e-2, 1e-3, 10, 200)
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 128), batch[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_value = loss.item()
        if loss_value < 0.1:
            break
    assert loss_value < 0.1, f"loss {loss_value:.3f} after {step + 1} steps"


def _data_dir(tmp_path, monkeypatch) -> tuple:
    """An ALETHIC_DATA_DIR with a frozen (stand-in) tokenizer, shards and YAML configs."""
    root = tmp_path / "data"
    monkeypatch.setenv("ALETHIC_DATA_DIR", str(root))
    tok = root / "tokenizer" / "alethic-tokenizer-v1"
    tok.mkdir(parents=True)
    for name in ("alethic-tokenizer-v1.model", "tokenizer.yaml", "report.json"):
        (tok / name).write_bytes(name.encode())
    frozen = write_frozen(tok, tok / "alethic-tokenizer-v1.model", tok / "tokenizer.yaml", tok / "report.json")
    write_train_shards(root / "shards", n_tokens=4000, vocab=128, tokenizer_sha256=frozen["model_sha256"])
    model_yaml, train_yaml = tmp_path / "model.yaml", tmp_path / "train.yaml"
    model_yaml.write_text(yaml.safe_dump(tiny_model_cfg("sgca")), encoding="utf-8")
    cfg = tiny_train_cfg(steps=50, cost={"price_per_hour": 3600.0, "peak_flops": 1.0e12})  # $1 per second
    train_yaml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return root, ["--model", str(model_yaml), "--train", str(train_yaml), "--device", "cpu"]


def test_cost_guard(tmp_path, monkeypatch):
    """--max-cost stops training and writes a checkpoint; --resume auto continues from it."""
    root, args = _data_dir(tmp_path, monkeypatch)
    assert cli.main(["train", *args, "--max-cost", "0.000001"]) == 0
    run = root / "runs" / "tiny_sgca"
    found = ckpt.list_checkpoints(run / "checkpoints")
    assert [s for s, _ in found] == [1], "should stop after the first step that crossed the budget"
    log = [json.loads(line) for line in (run / "train_log.jsonl").read_text().splitlines()]
    assert log[-1]["step"] == 1 and log[-1]["dollars_spent"] >= 0.000001
    assert {"loss", "tokens_per_sec", "mfu", "hours_left", "dollars_left"} <= log[-1].keys()
    manifest = json.loads((found[0][1] / "manifest.json").read_text())
    assert len(manifest["specials_sha256"]) == 64 and (found[0][1] / "specials.json").is_file()

    assert cli.main(["train", *args]) == 1                       # refuses to overwrite the run
    assert cli.main(["train", *args, "--resume", "auto", "--stop-step", "3"]) == 0
    assert ckpt.latest_checkpoint(run / "checkpoints").name == "step_0000003"


def test_train_refuses_wrong_tokenizer(tmp_path, monkeypatch):
    """A tokenizer whose file no longer matches FROZEN.json stops the run with a message."""
    root, args = _data_dir(tmp_path, monkeypatch)
    (root / "tokenizer" / "alethic-tokenizer-v1" / "alethic-tokenizer-v1.model").write_bytes(b"changed")
    assert cli.main(["train", *args]) == 1
