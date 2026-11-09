"""Crash-safe uint16 shards, their .json sidecars, and the state.jsonl unit log.

Rules (plan, Data pipeline 6-7): write to .tmp, flush, rename, and only then append the
finished unit to state.jsonl. On restart, finished units are skipped and stray .tmp files
deleted. A unit that was cut off is simply redone; it rewrites the same names with the
same bytes, so a killed-and-rerun build gives identical shard hashes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from alethic.config import write_json_atomic

SHARD_TOKENS = 67_108_864        # 128 MiB of uint16
TOKEN_DTYPE = np.dtype("<u2")    # little-endian uint16
_CHUNK = 8 << 20                 # write 8 MiB at a time


def _write_bytes(f, data: bytes) -> None:
    """Write data to an open file in 8 MiB chunks."""
    view = memoryview(data)
    for start in range(0, len(view), _CHUNK):
        f.write(view[start:start + _CHUNK])


def write_shard(path: Path, tokens: np.ndarray, meta: dict) -> dict:
    """Atomically write tokens to path (.bin) and meta + size + SHA-256 to path's .json sidecar."""
    data = np.ascontiguousarray(tokens, dtype=TOKEN_DTYPE).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        _write_bytes(f, data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    record = dict(meta, tokens=len(data) // 2, sha256=hashlib.sha256(data).hexdigest())
    write_json_atomic(path.with_suffix(".json"), record)
    return record


def to_tokens(ids: Iterable[int] | np.ndarray) -> np.ndarray:
    """Convert ids to uint16, refusing any id that does not fit."""
    arr = np.asarray(ids, dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() >= 65536):
        raise ValueError("token id outside 0..65535 cannot be stored as uint16")
    return arr.astype(TOKEN_DTYPE)


class ShardWriter:
    """Collects documents for one (unit, split) and cuts ~128 MiB shards at document boundaries."""

    def __init__(self, root: Path, source: str, name: str, split: str,
                 shard_tokens: int = SHARD_TOKENS, store_doc_bytes: bool = False) -> None:
        """Shards go to root/<source>/<name>-<part>.bin; validation also keeps per-doc bytes."""
        self.root, self.source, self.name, self.split = Path(root), source, name, split
        self.shard_tokens = shard_tokens
        self.store_doc_bytes = store_doc_bytes
        self.records: list[dict] = []
        self._docs: list[np.ndarray] = []
        self._doc_bytes: list[int] = []
        self._ntok = 0

    def add(self, ids: np.ndarray, nbytes: int) -> None:
        """Add one document (ids already end with EOS) and its UTF-8 byte count."""
        if self._docs and self._ntok + len(ids) > self.shard_tokens:
            self._flush()
        self._docs.append(ids)
        self._doc_bytes.append(int(nbytes))
        self._ntok += len(ids)

    def close(self) -> list[dict]:
        """Write the last shard and return one record per shard written."""
        if self._docs:
            self._flush()
        return self.records

    def _flush(self) -> None:
        """Write the buffered documents as the next part."""
        part = len(self.records)
        rel = Path(self.source) / f"{self.name}-{part:04d}.bin"
        meta = {"source": self.source, "split": self.split, "path": rel.as_posix(),
                "documents": len(self._docs), "utf8_bytes": sum(self._doc_bytes)}
        if self.store_doc_bytes:
            meta["doc_bytes"] = list(self._doc_bytes)
            meta["doc_tokens"] = [len(d) for d in self._docs]
        self.records.append(write_shard(self.root / rel, np.concatenate(self._docs), meta))
        self._docs, self._doc_bytes, self._ntok = [], [], 0


class StateLog:
    """Append-only state.jsonl: one line per finished unit."""

    def __init__(self, path: Path) -> None:
        """Read finished units; a torn last line (crash during append) is cut off."""
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        good_end = 0
        for line in raw.splitlines(keepends=True):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                break
            if not line.endswith(b"\n"):
                break
            self.records[rec["unit"]] = rec
            good_end += len(line)
        if good_end != len(raw):
            with open(self.path, "r+b") as f:
                f.truncate(good_end)

    def done(self, unit_id: str) -> bool:
        """True if this unit already finished in an earlier run."""
        return unit_id in self.records

    def append(self, record: dict) -> None:
        """Append one finished unit and force it to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "ab") as f:
            f.write((json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        self.records[record["unit"]] = record


def remove_stray_tmp(root: Path) -> int:
    """Delete every leftover *.tmp file under root; return how many were removed."""
    count = 0
    for p in Path(root).rglob("*.tmp"):
        p.unlink()
        count += 1
    return count


def run_unit(unit: dict, docs: Iterable[tuple[np.ndarray, int, bool]], out_root: Path,
             state: StateLog, shard_tokens: int = SHARD_TOKENS) -> dict | None:
    """Write one work unit's documents to train/ and val/ shards, then log it in state.

    unit: {"id", "source", "name"}. docs yields (ids ending with EOS, utf8 bytes, is_val).
    Returns the state record, or None if the unit was already finished.
    """
    if state.done(unit["id"]):
        return None
    writers = {
        False: ShardWriter(out_root / "train", unit["source"], unit["name"], "train", shard_tokens),
        True: ShardWriter(out_root / "val", unit["source"], unit["name"], "val", shard_tokens,
                          store_doc_bytes=True),
    }
    for ids, nbytes, is_val in docs:
        writers[bool(is_val)].add(to_tokens(ids), nbytes)
    shards = writers[False].close() + writers[True].close()
    for rec in shards:   # keep state.jsonl small: per-doc lists stay in the sidecars
        rec.pop("doc_bytes", None)
        rec.pop("doc_tokens", None)
    record = {"unit": unit["id"], "source": unit["source"], "shards": shards,
              "tokens": sum(r["tokens"] for r in shards)}
    state.append(record)
    return record
