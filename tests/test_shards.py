"""Crash-safe shards and the deterministic train / validation split."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from alethic.data import shards
from alethic.data.encode import is_validation

TESTS = Path(__file__).resolve().parent


def _bins(root: Path) -> dict[str, str]:
    """SHA-256 of every .bin under root, keyed by relative path."""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.bin"))}


def _run_proc(out: Path, *flags: str) -> subprocess.Popen:
    """Start shard_proc.py in a child process."""
    return subprocess.Popen([sys.executable, str(TESTS / "shard_proc.py"), str(out), *flags])


def _check_no_partial(out: Path, reference: dict[str, str]) -> None:
    """Every .bin that exists after a crash is complete: byte-identical to the clean build."""
    for rel, digest in _bins(out).items():
        assert reference[rel] == digest, f"partial or wrong shard {rel}"


def test_shard_atomic(tmp_path):
    """Killing a write leaves no partial shard; a rerun gives identical hashes."""
    import shard_proc

    ref = tmp_path / "reference"
    shard_proc.build(ref)
    reference = _bins(ref)
    assert len(reference) > 20
    state = shards.StateLog(ref / "state.jsonl")
    assert len(state.records) == shard_proc.N_UNITS

    # 1) Crash injected in the middle of writing a shard (half the bytes on disk).
    crashed = tmp_path / "crashed"
    proc = _run_proc(crashed, "--crash-at", "9")
    assert proc.wait(timeout=120) == 3
    assert list(crashed.rglob("*.tmp")), "the crash should leave a half-written .tmp"
    _check_no_partial(crashed, reference)
    shard_proc.build(crashed)
    assert not list(crashed.rglob("*.tmp"))
    assert _bins(crashed) == reference

    # 2) The process is killed from outside at several moments.
    unfinished = []
    for delay in (0.3, 0.9, 1.7):
        killed = tmp_path / f"killed_{delay}"
        proc = _run_proc(killed, "--slow")
        deadline = time.time() + 60
        while not (killed / "state.jsonl").exists() and time.time() < deadline:
            time.sleep(0.02)
        time.sleep(delay)
        proc.kill()
        proc.wait(timeout=60)
        _check_no_partial(killed, reference)
        done_before = len(shards.StateLog(killed / "state.jsonl").records)
        shard_proc.build(killed)
        assert _bins(killed) == reference, f"rerun after kill at +{delay}s differs"
        records = shards.StateLog(killed / "state.jsonl").records
        assert len(records) == shard_proc.N_UNITS
        unfinished.append(shard_proc.N_UNITS - done_before)
    assert all(u > 0 for u in unfinished), f"every kill must land mid-build; units left: {unfinished}"


def test_state_torn_line(tmp_path):
    """A half-written last line in state.jsonl is cut off and that unit is redone."""
    log = tmp_path / "state.jsonl"
    log.write_bytes(b'{"unit": "a", "tokens": 1}\n{"unit": "b", "tok')
    state = shards.StateLog(log)
    assert state.done("a") and not state.done("b")
    assert log.read_bytes() == b'{"unit": "a", "tokens": 1}\n'


def test_shard_cut_and_sidecar(tmp_path):
    """Shards are cut at document boundaries; sidecars hold counts, bytes and SHA-256."""
    w = shards.ShardWriter(tmp_path, "math", "f", "val", shard_tokens=10, store_doc_bytes=True)
    for n in (4, 4, 4, 12, 3):
        w.add(shards.to_tokens(np.full(n, n)), nbytes=n * 3)
    recs = w.close()
    assert [r["tokens"] for r in recs] == [8, 4, 12, 3]
    assert [r["doc_tokens"] for r in recs] == [[4, 4], [4], [12], [3]]
    side = json.loads((tmp_path / "math" / "f-0000.json").read_text(encoding="utf-8"))
    data = (tmp_path / "math" / "f-0000.bin").read_bytes()
    assert side["sha256"] == hashlib.sha256(data).hexdigest() and side["utf8_bytes"] == 24
    assert np.frombuffer(data, dtype="<u2").tolist() == [4] * 8
    with pytest.raises(ValueError):
        shards.to_tokens([70000])


def test_split_function():
    """is_validation is deterministic, ~0.1% of groups, and keyed by group only."""
    keys = [f"github.com/user{i}/repo{i % 997}" for i in range(200_000)]
    first = [is_validation("python", k) for k in keys]
    assert first == [is_validation("python", k) for k in keys]                 # deterministic
    assert [is_validation("code_other", k) for k in keys[:5000]] == first[:5000]  # same repo, same side
    unique = sorted(set(keys))
    share = sum(is_validation("python", k) for k in unique) / len(unique)
    assert 0.0005 < share < 0.0015
    # Train and validation never share a group key, however many documents a group has.
    docs = [(k, j) for k in unique[:3000] for j in range(3)]
    train_keys = {k for k, _ in docs if not is_validation("fineweb_edu", k)}
    val_keys = {k for k, _ in docs if is_validation("fineweb_edu", k)}
    assert train_keys.isdisjoint(val_keys) and train_keys | val_keys == set(unique[:3000])
    with pytest.raises(ValueError):
        is_validation("unknown", "x")
    with pytest.raises(ValueError):
        is_validation("python", "")
