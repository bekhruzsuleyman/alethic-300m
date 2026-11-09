"""Parallel encoding of work units into shards, and the train / validation split.

M0 provides is_validation(). The worker pool that reads parquet row ranges, encodes them
with the frozen codec and feeds data.shards.run_unit is milestone M2.
"""
from __future__ import annotations

import hashlib

KNOWN_SOURCES = frozenset(
    {"python", "code_other", "fineweb_edu", "math", "uzbek_latn", "uzbek_cyrl"}
)
VAL_FRACTION = 0.001          # 0.1% of groups go to validation
_BUCKETS = 1_000_000


def is_validation(source: str, key: str, fraction: float = VAL_FRACTION) -> bool:
    """True if this document's group belongs to the validation split.

    The key is the repo name for code, the `id` for FineWeb-Edu and the URL for math and
    Uzbek. Only the key is hashed (not the source), so a repo that appears in both code
    sources lands on the same side in both. Pure function: same input, same answer.
    """
    if source not in KNOWN_SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(KNOWN_SOURCES)}")
    if not isinstance(key, str) or not key:
        raise ValueError(f"empty group key for source {source!r}; every document needs one")
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8, person=b"alethic-split").digest()
    return int.from_bytes(digest, "little") % _BUCKETS < int(round(fraction * _BUCKETS))


def encode_units(*_args, **_kwargs) -> None:
    """Run every work unit through a pool of workers, one per vCPU (milestone M2)."""
    raise NotImplementedError("Parallel encoding is part of milestone M2.")
