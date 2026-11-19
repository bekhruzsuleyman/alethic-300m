"""Where things are on the screen: a pure model used to turn a mouse click (row) into the tool block it hit.

The screen is rebuilt from the log (clear + replay), so content starts on row 1 and the input frame sits right below it.
If content plus frame is taller than the window, the terminal has scrolled by the difference.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[78]")


@dataclass
class BlockMark:
    id: object
    header: str                       # plain text of the clickable header line
    body: list = field(default_factory=list)   # plain text of the lines below it


def _wrap(text: str, cols: int) -> list:
    cols = max(1, cols)
    return [text[i:i + cols] for i in range(0, len(text), cols)] or [""]


def physical_lines(segments: list, cols: int) -> list:
    """[(plain text of one screen row, block id or None)]. A block's id is on its header's first row only."""
    out, cur = [], ""
    for seg in segments:
        if isinstance(seg, BlockMark):
            if cur:
                out += [(t, None) for t in _wrap(cur, cols)]
                cur = ""
            rows = _wrap(seg.header, cols)
            out += [(t, seg.id if i == 0 else None) for i, t in enumerate(rows)]
            for ln in seg.body:
                out += [(t, None) for t in _wrap(ln, cols)]
            continue
        cur += ANSI.sub("", seg)
        while "\n" in cur:
            line, cur = cur.split("\n", 1)
            out += [(t, None) for t in _wrap(line, cols)]
    if cur:
        out += [(t, None) for t in _wrap(cur, cols)]
    return out


def wrap_ansi_lines(text: str, cols: int) -> list:
    """Split ANSI-styled text into physical screen rows of at most `cols` visible characters."""
    if not text:
        return []
    cols = max(1, cols)
    logical = text[:-1].split("\n") if text.endswith("\n") else text.split("\n")
    rows = []
    for ln in logical:
        if len(ANSI.sub("", ln)) <= cols:
            rows.append(ln)
            continue
        cur, vis, active = [], 0, ""
        pos = 0
        for m in ANSI.finditer(ln):
            chunk = ln[pos:m.start()]
            for ch in chunk:
                if vis == cols:
                    if active:
                        cur.append("\x1b[0m")
                    rows.append("".join(cur))
                    cur = [active] if active else []
                    vis = 0
                cur.append(ch)
                vis += 1
            esc = m.group(0)
            cur.append(esc)
            if esc in ("\x1b[0m", "\x1b[m"):
                active = ""
            elif esc.endswith("m"):
                active = esc
            pos = m.end()
        for ch in ln[pos:]:
            if vis == cols:
                if active:
                    cur.append("\x1b[0m")
                rows.append("".join(cur))
                cur = [active] if active else []
                vis = 0
            cur.append(ch)
            vis += 1
        rows.append("".join(cur))
    return rows


def hit(lines: list, row: int, rows: int, frame_h: int, scroll: int = 0):
    """The block id whose header is on screen `row` (1-based), or None."""
    top_rows = max(1, rows - frame_h)
    scrolled = max(0, len(lines) - top_rows - scroll)
    i = row - 1 + scrolled
    if row < 1 or i >= len(lines) or row > top_rows:
        return None
    return lines[i][1]
