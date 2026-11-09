"""Tokenizer evaluation (milestone M1).

Will report, per source on >= 2,000 held-out documents: round trip (must be 100.00%),
bytes per token, tokens per word, byte-fallback share, Uzbek tokens per word per script,
the newline + 4/8/12-space indentation rule, and the 30 most frequent learned pieces.
"""
from __future__ import annotations


def evaluate_tokenizer(*_args, **_kwargs) -> dict:
    """Run every evaluation rule and return the report (milestone M1)."""
    raise NotImplementedError("Tokenizer evaluation is part of milestone M1.")
