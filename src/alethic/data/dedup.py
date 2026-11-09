"""Exact deduplication by 8-byte blake2b text hashes (milestone M2).

Workers hash each document and write the hashes to disk; one global drop list keeps the
first copy in (source, file, row) order.
"""
from __future__ import annotations

import hashlib


def text_hash(text: str) -> bytes:
    """8-byte blake2b hash of a document's UTF-8 text."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


def build_drop_list(*_args, **_kwargs) -> None:
    """Build the global duplicate drop list (milestone M2)."""
    raise NotImplementedError("Deduplication is part of milestone M2.")
