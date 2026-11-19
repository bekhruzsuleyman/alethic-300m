"""The file viewer (/open, /expand): the whole document on the left with colors and line numbers,
the end of the chat on the right. Everything is a pure function of its inputs except run(), which draws on the screen."""
from __future__ import annotations

from alethic.agent import brand as b
from alethic.agent import highlight as hl
from alethic.agent.blocks import fit

MIN_SPLIT = 70          # narrower windows show the document alone
CLOSE_KEYS = ("q", "esc", "ctrl-c", "ctrl-d")


MIN_CHAT, MIN_DOC, STEP = 20, 30, 4


def widths(cols: int, doc=None):
    """(chat width, document width); the divider takes one more column. Chat width 0 = document only.
    `doc` is the width the user chose with the arrow keys; None = a default share."""
    if cols < MIN_SPLIT:
        return 0, cols
    want = int(cols * 0.58) if doc is None else doc
    d = max(MIN_DOC, min(cols - 1 - MIN_CHAT, want))
    return cols - 1 - d, d


def resize(doc, key: str, cols: int):
    """New document width after a resize key: left or [ makes the document wider, right or ] narrower."""
    chat, cur = widths(cols, doc)
    if key in ("left", "["):
        return widths(cols, cur + STEP)[1]
    if key in ("right", "]"):
        return widths(cols, cur - STEP)[1]
    return doc


def scroll(top: int, key: str, page: int, total: int) -> int:
    last = max(0, total - page)
    if key == "down":
        top += 1
    elif key == "up":
        top -= 1
    elif key == "pgdn":
        top += page - 1
    elif key == "pgup":
        top -= page - 1
    elif key == "home":
        top = 0
    elif key == "end":
        top = last
    return max(0, min(last, top))


def _lines(text: str) -> list:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def document_rows(name: str, text: str, top: int, height: int, width: int) -> list:
    lines = _lines(text)
    nw = max(3, len(str(len(lines))))
    h = hl.Highlighter(hl.lang_for_path(name))
    tokens = [h.line(ln.replace("\t", "    ")) or [("", "")] for ln in lines[:top + height]]   # state flows from line 1
    rows = []
    for i in range(top, top + height):
        if i >= len(lines):
            rows.append([(" " * width, "")])
            continue
        row = [(str(i + 1).rjust(nw), b.MUTED), ("  ", "")] + tokens[i]
        row = fit(row, width)
        rows.append(row + [(" " * (width - sum(len(t) for t, _ in row)), "")])
    return rows


def chat_rows(text: str, height: int, width: int) -> list:
    wrapped = []
    for ln in text.rstrip("\n").split("\n") if text.strip() else []:
        ln = ln.replace("\t", "    ").rstrip()
        wrapped += [ln[i:i + width] for i in range(0, len(ln), width)] or [""]
    tail = wrapped[-height:]
    rows = [[(" " * width, "")] for _ in range(height - len(tail))]
    rows += [[(ln.ljust(width), b.MUTED)] for ln in tail]
    return rows


def _bar(left: str, right: str, width: int, style: str) -> list:
    room = max(0, width - len(right) - 1)
    left = left if len(left) <= room else left[:max(0, room - 1)] + "…"
    return [(left + " " * (width - len(left) - len(right)) + right, style)]


def screen(name: str, text: str, chat: str, top: int, cols: int, lines: int, unicode: bool = True, doc=None) -> list:
    """All rows of the viewer: a header, the body (chat | document), and a status bar at the bottom.
    The document is on the right; `doc` is its width (None = default)."""
    body = max(1, lines - 2)
    total = len(_lines(text))
    top = scroll(top, "", body, total)
    left, right = widths(cols, doc)
    rows_doc = document_rows(name, text, top, body, right)
    if left:
        v = "│" if unicode else "|"
        rows = [_bar(" chat", "", left, b.MUTED) + [(v, b.RULE)] + _bar(" " + name, "", right, b.AMBER)]
        rows += [c + [(v, b.RULE)] + d for c, d in zip(chat_rows(chat, body, left), rows_doc)]
    else:
        rows = [_bar(" " + name, "", cols, b.AMBER)] + rows_doc
    last = min(total, top + body)
    keys = "↑↓ PgUp PgDn scroll · ←→ resize · q close" if unicode else "Up/Down PgUp PgDn scroll, Left/Right resize, q close"
    rows.append(_bar(" lines %d-%d of %d" % (min(top + 1, total), last, total), keys + " ", cols, b.MUTED))
    return rows


def plain_text(name: str, text: str) -> str:
    """The file with line numbers and no screen control: for terminals that cannot do the full viewer."""
    lines = _lines(text)
    w = len(str(len(lines)))
    return "\n".join("%s  %s" % (str(i + 1).rjust(w), ln) for i, ln in enumerate(lines))


def run(name: str, text: str, chat: str, read, write, size, unicode: bool = True, color: bool = True, truecolor: bool = True) -> None:
    """Show the viewer until q/Esc. `read` returns one named key, `write` prints, `size` gives (columns, rows)."""
    def paint(row) -> str:
        if not color:
            return "".join(t for t, _ in row)
        return "".join("\x1b[%sm%s\x1b[0m" % (b.style_code(st, truecolor), t) if st else t for t, st in row)

    top, doc = 0, None
    write("\x1b[?1049h\x1b[?25l")
    try:
        while True:
            cols, rows = size()
            cols, rows = max(20, cols - 1), max(4, rows)
            top = scroll(top, "", max(1, rows - 2), len(_lines(text)))
            write("\x1b[H" + "".join(paint(r) + "\x1b[K" + ("\r\n" if i < rows - 1 else "")
                                     for i, r in enumerate(screen(name, text, chat, top, cols, rows, unicode, doc))))
            key = read()
            if key in CLOSE_KEYS:
                break
            top = scroll(top, key, max(1, rows - 2), len(_lines(text)))
            doc = resize(doc, key, cols)
    finally:
        write("\x1b[?25h\x1b[?1049l")
