"""Exact deduplication by 8-byte blake2b text hashes (milestone M2).

Workers hash each document and write the hashes to disk; one global drop list keeps the
first copy in (source, file, row) order.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from alethic.config import write_json_atomic

HASH_DTYPE = np.dtype("<u8")


def text_hash(text: str) -> bytes:
    """8-byte blake2b hash of a document's UTF-8 text."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


def hash_texts(texts: Iterable[str]) -> np.ndarray:
    """Hash a sequence of texts into a 1-D uint64 array (8 bytes per document)."""
    raw = b"".join(text_hash(t) for t in texts)
    return np.frombuffer(raw, dtype=HASH_DTYPE).copy()


def write_unit_hashes(hashes_dir: Path, unit_id: str, hashes: np.ndarray, utf8_bytes: int) -> dict:
    """Atomically write one unit's 8-byte document hashes (.bin) and its summary (.json)."""
    safe = unit_id.replace("/", "__").replace(":", "_")
    hashes_dir.mkdir(parents=True, exist_ok=True)
    bin_path = hashes_dir / f"{safe}.bin"
    json_path = hashes_dir / f"{safe}.json"
    tmp = bin_path.with_name(bin_path.name + ".tmp")
    data = np.ascontiguousarray(hashes, dtype=HASH_DTYPE).tobytes()
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, bin_path)
    meta = {"unit": unit_id, "docs": int(len(hashes)), "utf8_bytes": int(utf8_bytes), "file": bin_path.name}
    write_json_atomic(json_path, meta)
    return meta


def read_unit_hash_meta(hashes_dir: Path, unit_id: str) -> dict | None:
    """Return the saved hash metadata for unit_id if its .bin and .json both exist."""
    safe = unit_id.replace("/", "__").replace(":", "_")
    bin_path = hashes_dir / f"{safe}.bin"
    json_path = hashes_dir / f"{safe}.json"
    if not bin_path.is_file() or not json_path.is_file():
        return None
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    if bin_path.stat().st_size != 8 * int(meta["docs"]):
        return None
    return meta


def build_drop_list(unit_ids: list[str], hashes_dir: Path) -> tuple[dict[str, set[int]], dict]:
    """Build the global duplicate drop list in the given (source, file, row) unit order.

    Returns ({unit_id: set of 0-based doc indices to drop}, summary_dict).
    """
    arrays: list[np.ndarray] = []
    lengths: list[int] = []
    for uid in unit_ids:
        safe = uid.replace("/", "__").replace(":", "_")
        arr = np.frombuffer((hashes_dir / f"{safe}.bin").read_bytes(), dtype=HASH_DTYPE)
        arrays.append(arr)
        lengths.append(len(arr))
    if not arrays or sum(lengths) == 0:
        return {uid: set() for uid in unit_ids}, {"total_docs": 0, "unique_docs": 0, "dropped_duplicates": 0}
    all_hashes = np.concatenate(arrays)
    _, first_indices = np.unique(all_hashes, return_index=True)
    keep = np.zeros(len(all_hashes), dtype=bool)
    keep[first_indices] = True
    drops: dict[str, set[int]] = {}
    offset = 0
    for uid, n in zip(unit_ids, lengths):
        unit_keep = keep[offset:offset + n]
        drops[uid] = set(np.flatnonzero(~unit_keep).tolist())
        offset += n
    summary = {
        "total_docs": int(len(all_hashes)),
        "unique_docs": int(len(first_indices)),
        "dropped_duplicates": int(len(all_hashes) - len(first_indices)),
        "per_unit_dropped": {uid: len(d) for uid, d in drops.items() if d},
    }
    write_json_atomic(hashes_dir / "drop_list.json", summary)
    return drops, summary
