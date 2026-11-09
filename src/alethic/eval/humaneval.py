"""Optional HumanEval / MBPP pass@1 (greedy, one sample), reported with decontamination counts.

Runs after M7 as a sanity check; scores at 300M parameters are expected to be low.
"""
from __future__ import annotations


def pass_at_1(*_args, **_kwargs) -> dict:
    """Greedy pass@1 on HumanEval and MBPP (after M7)."""
    raise NotImplementedError("HumanEval / MBPP evaluation is optional and comes after M7.")
