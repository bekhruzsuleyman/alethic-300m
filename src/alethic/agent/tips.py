"""Short tips shown at the bottom while the model works. They move on with the work (tool calls and tokens), not with a clock."""
from __future__ import annotations

import random

EVERY = 2   # a tip stays for this many vibe steps (a step = one tool call or 80 tokens)
TIPS = [
    "/details hides or shows every tool block. /mouse on lets you click headers.",
    "/open shows the last file beside the chat. ←→ resize it, q closes it.",
    "/mouse on lets you click tool headers, but then the wheel stops scrolling.",
    "/folder switches the work folder and starts a new conversation.",
    "Ctrl+C stops a reply. The unfinished answer is dropped.",
    "/effort low makes answers faster; /effort high allows more tool calls.",
    "/undo removes your last message and the answer to it.",
    "/retry runs your last message again. The new answer can differ.",
    "/compact forgets the oldest turns when the context is nearly full.",
    "/context shows how much of the context window is used and by whom.",
    "Type / to open the command menu. Tab completes, Esc closes it.",
    "/temp 0 makes the model always pick the most likely token (it may loop).",
    "/save writes the raw conversation into the work folder.",
    "/auto turns permission prompts off or on (off = runs without asking).",
    "/files lists what is in the work folder; /cat shows a file.",
]


def tip_at(step: int, seed: int = 0, pool=None) -> str:
    """The tip for a step count. Each pass shows every tip once, in a different order per session."""
    pool = pool or TIPS
    n = step // EVERY
    cycle, i = divmod(n, len(pool))
    order = random.Random(seed * 7919 + cycle).sample(pool, len(pool))
    if cycle > 0 and order[0] == random.Random(seed * 7919 + cycle - 1).sample(pool, len(pool))[-1]:
        order[0], order[1] = order[1], order[0]
    return order[i]
