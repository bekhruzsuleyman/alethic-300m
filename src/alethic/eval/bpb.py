"""Bits per byte on the validation shards, per source and overall.

BPB = (sum of loss in nats over all tokens) / (ln 2 * total UTF-8 bytes of the documents).
Comparable across tokenizers, unlike loss per token. The evaluation driver over validation
shards (fixed 2M-token slice every 1,000 steps, all shards at the end) comes with M2/M4.
"""
from __future__ import annotations

import math


def bits_per_byte(loss_nats_sum: float, utf8_bytes: int) -> float:
    """Convert summed token loss (nats) over documents into bits per UTF-8 byte."""
    if utf8_bytes <= 0:
        raise ValueError("utf8_bytes must be positive")
    return loss_nats_sum / (math.log(2) * utf8_bytes)


def evaluate_bpb(*_args, **_kwargs) -> dict:
    """Bits per byte per source on validation shards (milestone M2/M4)."""
    raise NotImplementedError("Validation BPB needs validation shards (milestone M2) and lands before M4.")
