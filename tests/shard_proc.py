"""Helper for test_shard_atomic: builds a few deterministic units of shards.

Run as a script so the test can kill it:  python shard_proc.py OUT [--slow] [--crash-at N]
--slow sleeps briefly per document; --crash-at N writes half of the N-th shard file and exits.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from alethic.data import shards
from alethic.data.encode import is_validation

N_UNITS = 6
SHARD_TOKENS = 4096


def unit_docs(unit: int, slow: bool):
    """Deterministic documents for one unit: (ids ending with EOS 2, utf8 bytes, is_val)."""
    rng = np.random.default_rng(unit)
    for d in range(150):
        n = int(rng.integers(20, 400))
        ids = np.append(rng.integers(3, 50000, size=n), 2)
        if slow:
            time.sleep(0.004)
        yield ids, 4 * n, is_validation("python", f"repo-{unit}-{d % 40}")


def build(out: Path, slow: bool = False) -> None:
    """Resume-safe build: drop stray .tmp files, skip finished units, run the rest."""
    shards.remove_stray_tmp(out)
    state = shards.StateLog(out / "state.jsonl")
    for u in range(N_UNITS):
        unit = {"id": f"python/file{u}:0-150", "source": "python", "name": f"file{u}-r000"}
        shards.run_unit(unit, unit_docs(u, slow), out, state, shard_tokens=SHARD_TOKENS)


def _install_crash(at: int) -> None:
    """Make the at-th shard write stop halfway and kill the process, like a power cut."""
    real = shards._write_bytes
    calls = {"n": 0}

    def crashing(f, data):
        calls["n"] += 1
        if calls["n"] == at:
            f.write(data[: len(data) // 2])
            f.flush()
            os._exit(3)
        real(f, data)

    shards._write_bytes = crashing


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--crash-at", type=int, default=0)
    a = p.parse_args()
    if a.crash_at:
        _install_crash(a.crash_at)
    build(Path(a.out), slow=a.slow)
    sys.exit(0)
