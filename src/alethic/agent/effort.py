"""Effort levels: how many tool calls one message may use and how long each model step may be."""
from __future__ import annotations

LEVELS = {"low": (3, 160), "medium": (10, 400), "high": (20, 800)}  # (max tool calls, max new tokens per step)


def apply(agent, level: str) -> None:
    agent.max_steps, agent.max_new = LEVELS[level]
