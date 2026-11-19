"""Freeze the tokenizer (FROZEN.json) and check the freeze before every later stage."""
from __future__ import annotations

import json
from pathlib import Path

from alethic.config import sha256_file, write_json_atomic

FROZEN_NAME = "FROZEN.json"


class FrozenError(RuntimeError):
    """The tokenizer is not frozen, or its model file no longer matches the frozen hash."""


def write_frozen(tokenizer_dir: str | Path, model_path: str | Path, config_path: str | Path,
                 report_path: str | Path) -> dict:
    """Write FROZEN.json with the SHA-256 of the model file, the config and the evaluation report."""
    record = {
        "model_file": Path(model_path).name,
        "model_sha256": sha256_file(model_path),
        "config_sha256": sha256_file(config_path),
        "report_sha256": sha256_file(report_path),
    }
    write_json_atomic(Path(tokenizer_dir) / FROZEN_NAME, record)
    return record


def verify_frozen(tokenizer_dir: str | Path) -> tuple[Path, str]:
    """Return (model path, model SHA-256) after checking the file against FROZEN.json.

    Stops with a message saying what to do if the tokenizer is not frozen or was changed.
    """
    tdir = Path(tokenizer_dir)
    frozen_path = tdir / FROZEN_NAME
    if not frozen_path.is_file():
        raise FrozenError(
            f"{frozen_path} not found. Run `python -m alethic tokenizer-build` (milestone M1) "
            "and freeze the tokenizer before this stage."
        )
    record = json.loads(frozen_path.read_text(encoding="utf-8"))
    model_path = tdir / record["model_file"]
    if not model_path.is_file():
        raise FrozenError(f"{model_path} is missing; restore it from the frozen tokenizer.")
    actual = sha256_file(model_path)
    if actual != record["model_sha256"]:
        raise FrozenError(
            f"{model_path} has SHA-256 {actual}, but FROZEN.json says {record['model_sha256']}. "
            "The tokenizer changed after it was frozen; restore the frozen file. Do not re-freeze "
            "unless you also rebuild every shard."
        )
    return model_path, actual
