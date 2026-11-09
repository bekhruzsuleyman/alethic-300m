"""Atomic checkpoints: runs/<name>/checkpoints/step_0000500/{state.pt, specials.json, manifest.json}.

A checkpoint is written into a .tmp folder and renamed only when complete, so a crash
never leaves a half checkpoint. Kept: the last two, plus every 10,000-step one.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import torch

from alethic.config import sha256_file, write_json_atomic

_STEP_DIR = re.compile(r"^step_(\d{7})$")


class CheckpointError(RuntimeError):
    """A checkpoint is missing files or fails its hash check."""


def step_dir(ckpt_root: Path, step: int) -> Path:
    """Folder name for a step, e.g. step_0000500."""
    return Path(ckpt_root) / f"step_{step:07d}"


def list_checkpoints(ckpt_root: Path) -> list[tuple[int, Path]]:
    """Finished checkpoints as (step, folder), oldest first."""
    root = Path(ckpt_root)
    if not root.is_dir():
        return []
    found = []
    for p in root.iterdir():
        m = _STEP_DIR.match(p.name)
        if m and p.is_dir() and (p / "manifest.json").is_file():
            found.append((int(m.group(1)), p))
    return sorted(found)


def latest_checkpoint(ckpt_root: Path) -> Path | None:
    """The newest finished checkpoint folder, or None."""
    found = list_checkpoints(ckpt_root)
    return found[-1][1] if found else None


def save_checkpoint(ckpt_root: Path, step: int, state: dict, specials_path: Path,
                    info: dict, keep_last: int, keep_every: int) -> Path:
    """Write state.pt, a copy of specials.json and manifest.json atomically; then prune old ones."""
    root = Path(ckpt_root)
    final = step_dir(root, step)
    if final.exists():
        return final                       # already saved at this step
    tmp = root / f"{final.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with open(tmp / "state.pt", "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    shutil.copyfile(specials_path, tmp / "specials.json")
    manifest = dict(info, step=step,
                    state_sha256=sha256_file(tmp / "state.pt"),
                    specials_sha256=sha256_file(tmp / "specials.json"))
    write_json_atomic(tmp / "manifest.json", manifest)
    os.replace(tmp, final)
    prune(root, keep_last, keep_every)
    return final


def load_checkpoint(folder: Path) -> tuple[dict, dict]:
    """Return (state, manifest) after checking the state file's SHA-256."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    actual = sha256_file(folder / "state.pt")
    if actual != manifest["state_sha256"]:
        raise CheckpointError(f"{folder}/state.pt is corrupt (SHA-256 mismatch); resume from an older one")
    state = torch.load(folder / "state.pt", map_location="cpu", weights_only=True)
    return state, manifest


def prune(ckpt_root: Path, keep_last: int, keep_every: int) -> None:
    """Delete checkpoints except the newest keep_last and those at multiples of keep_every."""
    found = list_checkpoints(ckpt_root)
    newest = {s for s, _ in found[-keep_last:]} if keep_last > 0 else set()
    for step, folder in found:
        if step in newest or (keep_every > 0 and step % keep_every == 0):
            continue
        shutil.rmtree(folder)


def remove_stray_tmp(ckpt_root: Path) -> None:
    """Delete half-written *.tmp checkpoint folders left by a crash."""
    root = Path(ckpt_root)
    if root.is_dir():
        for p in root.glob("*.tmp"):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
