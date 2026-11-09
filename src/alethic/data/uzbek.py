"""Deterministic apostrophe cleaning for the two Uzbek sources (data cleaning, not tokenizer).

The app that uses the model must run the same function on Uzbek user text.
- After o, O, g or G, an apostrophe-like character becomes U+02BB (as in oʻ, gʻ).
- Between any other two letters it becomes U+02BC (as in maʼno).
- Anywhere else it is left alone.
"""
from __future__ import annotations

# U+0027 ', U+2018 ‘, U+2019 ’, U+0060 `, U+00B4 ´, U+02BC ʼ
APOSTROPHES = frozenset("'‘’`´ʼ")
TURNED_COMMA = "ʻ"   # ʻ  used after o / g
APOSTROPHE = "ʼ"     # ʼ  used between other letters
OG = frozenset("oOgG")


def normalize_uzbek(text: str) -> tuple[str, int]:
    """Return (cleaned text, number of characters changed). Decisions use the original text."""
    if not any(ch in APOSTROPHES for ch in text):
        return text, 0
    out = list(text)
    changed = 0
    for i, ch in enumerate(text):
        if ch not in APOSTROPHES or i == 0:
            continue
        prev = text[i - 1]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if prev in OG:
            new = TURNED_COMMA
        elif prev.isalpha() and nxt.isalpha():
            new = APOSTROPHE
        else:
            continue
        if new != ch:
            out[i] = new
            changed += 1
    return "".join(out), changed
