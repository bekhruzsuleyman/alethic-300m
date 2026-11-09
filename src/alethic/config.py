"""Load YAML settings, refuse placeholders, and resolve paths under ALETHIC_DATA_DIR."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

# Any string value containing one of these is treated as "not filled in yet".
PLACEHOLDER_MARKERS = ("REPLACE_", "YOUR_")


class ConfigError(RuntimeError):
    """A setting is missing, still a placeholder, or otherwise unusable."""


def find_placeholders(value: Any, where: str = "") -> list[str]:
    """Return 'path = value' for every string in a nested dict/list that looks like a placeholder."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found += find_placeholders(item, f"{where}.{key}" if where else str(key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            found += find_placeholders(item, f"{where}[{i}]")
    elif isinstance(value, str) and any(m in value for m in PLACEHOLDER_MARKERS):
        found.append(f"{where} = {value!r}")
    return found


def check_no_placeholders(cfg: dict, source: str) -> None:
    """Stop with a message listing every placeholder value in cfg."""
    bad = find_placeholders(cfg)
    if bad:
        lines = "\n  ".join(bad)
        raise ConfigError(
            f"{source} still has placeholder values:\n  {lines}\n"
            "Fill in real values (see the comments in that file), then run again."
        )


def load_config(path: str | Path) -> dict:
    """Read one YAML file into a dict and refuse it if any value is a placeholder."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")
    with p.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"{p} must contain a YAML mapping at the top level.")
    check_no_placeholders(cfg, str(p))
    return cfg


def data_dir() -> Path:
    """Return $ALETHIC_DATA_DIR, the root for every large file; stop if it is not set."""
    value = os.environ.get("ALETHIC_DATA_DIR", "").strip()
    if not value:
        raise ConfigError(
            "ALETHIC_DATA_DIR is not set. Set it to a folder on a large disk, e.g.\n"
            "  Linux:   export ALETHIC_DATA_DIR=/mnt/data/alethic\n"
            "  Windows: $env:ALETHIC_DATA_DIR = 'D:\\alethic-data'"
        )
    return Path(value)


def resolve_path(value: str | Path, root: Path | None = None) -> Path:
    """Absolute paths stay as they are; relative ones are placed under root (default ALETHIC_DATA_DIR)."""
    p = Path(value)
    if p.is_absolute():
        return p
    return (root if root is not None else data_dir()) / p


def config_hash(cfg: dict) -> str:
    """SHA-256 of a config dict in a stable JSON form, used to detect changed settings."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path: str | Path, obj: Any) -> None:
    """Write JSON to path.tmp, flush it to disk, then rename it over path."""
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
