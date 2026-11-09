"""Benchmark decontamination (milestone M2).

Drops any document sharing a 13-gram of whitespace-separated words with the HumanEval or
MBPP test problems and solutions (pinned revisions), and logs drop counts per source.
"""
from __future__ import annotations


def build_benchmark_ngrams(*_args, **_kwargs) -> None:
    """Collect the 13-grams of the benchmark problems and solutions (milestone M2)."""
    raise NotImplementedError("Decontamination is part of milestone M2.")
