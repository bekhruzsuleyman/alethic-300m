"""The "vibe words" on the bottom row while Alethic works.

A word is not on a timer: it moves on when real work happens (a tool call, or every TOKENS_PER_WORD generated tokens).
The words are about what the model really does: weigh, sift, measure, settle on an answer.
"""
from __future__ import annotations

import random

TOKENS_PER_WORD = 80
WORDS = ["Weighing", "Sifting", "Distilling", "Resolving", "Aligning", "Sampling", "Tracing", "Binding", "Collapsing", "Estimating",
         "Decoding", "Refining", "Balancing", "Folding", "Mapping", "Measuring", "Settling", "Narrowing", "Converging", "Reckoning",
         "Attending", "Projecting", "Averaging", "Gating", "Modulating", "Oscillating", "Tuning", "Triangulating"]
RUNNING = ["Dispatching", "Probing", "Fetching", "Verifying", "Opening", "Querying", "Applying", "Testing", "Executing", "Checking"]


def _order(pool: list, seed: int, cycle: int) -> list:
    """The order the words appear in during one pass through the list; a pass never starts with the word the last one ended on."""
    order = random.Random(seed * 1000 + cycle).sample(pool, len(pool))
    if cycle > 0 and order[0] == random.Random(seed * 1000 + cycle - 1).sample(pool, len(pool))[-1]:
        order[0], order[1] = order[1], order[0]
    return order


class Vibes:
    """Picks the word to show. step moves on with tool calls and with token counts, never with the clock."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.step = 0
        self._tokens = 0

    def word(self, phase: str) -> str:
        pool = RUNNING if phase == "running" else WORDS
        return _order(pool, self.seed, self.step // len(pool))[self.step % len(pool)]

    def tokens(self, total: int) -> None:
        """Tokens generated so far this turn. Every TOKENS_PER_WORD of them move to the next word."""
        if total > self._tokens:
            self.step += total // TOKENS_PER_WORD - self._tokens // TOKENS_PER_WORD
            self._tokens = total

    def call(self) -> None:
        self.step += 1

    def new_turn(self) -> None:
        self._tokens = 0
