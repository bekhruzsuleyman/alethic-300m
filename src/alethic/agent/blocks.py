"""Boxed blocks for the screen: one per tool call (terminal, file, search) and one per code fence in an answer.

Everything returns rows of [(text, style)] so it can be tested without a terminal; the UI does the painting.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from alethic.agent import brand as b
from alethic.agent import highlight as hl
from alethic.agent import websearch

SEP = "sep"          # a row marker: a divider line inside a box
CODE_ROWS = 14       # how many lines of code a block shows before "… +N more lines"
OUT_ROWS = 10        # same, for command output and search results
NUMBERED = re.compile(r"^(\d+) {3}(.*)$")
FENCE_LINE = re.compile(r"^\s*```([\w+#.-]*)\s*$")

@dataclass
class View:
    header: list   # [(text, style)] the "▸ Running command in terminal" line
    rows: list     # box rows; empty while the call has no result yet


def _width(row) -> int:
    return sum(len(t) for t, _ in row)


def fit(row, width: int, more: str = "…") -> list:
    """Cut a row to `width` columns, ending in an ellipsis when something was cut."""
    if _width(row) <= width:
        return list(row)
    out, used = [], 0
    room = max(0, width - len(more))
    for text, style in row:
        if used + len(text) <= room:
            out.append((text, style))
            used += len(text)
        else:
            if room - used > 0:
                out.append((text[:room - used], style))
            break
    out.append((more, out[-1][1] if out else ""))
    return out


def _pad(row: list, width: int) -> list:
    """A row of exactly `width` columns: two spaces in front, the content (cut with an ellipsis if too long), spaces behind."""
    width = max(12, width)
    row = fit(row, width - 4, "…")
    return [("  ", "")] + row + [(" " * (width - 2 - _width(row)), "")]


def box_top(title: str, width: int, unicode: bool = True) -> list:
    """The first row of a block: its name (a file, a language, "terminal") in the muted color."""
    return _pad([(title, b.MUTED)], width)


def box_row(row, width: int, unicode: bool = True) -> list:
    """One content row of a block `width` columns wide; the SEP marker gives an empty row."""
    return _pad([] if row == SEP else row, width)


def box_bottom(width: int, unicode: bool = True) -> list:
    return _pad([], width)


def box(title: str, rows: list, width: int, unicode: bool = True) -> list:
    """A block: title row, content rows, an empty row at the end. The screen paints all of it on a slightly lighter background."""
    return [box_top(title, width, unicode)] + [box_row(r, width, unicode) for r in rows] + [box_bottom(width, unicode)]


def _text_rows(text: str, style: str, limit: int) -> list:
    lines = text.split("\n") if text else ["(no output)"]
    rows = [[(ln.replace("\t", "    "), style)] for ln in lines[:limit]]
    if len(lines) > limit:
        rows.append([("… +%d more lines" % (len(lines) - limit), b.MUTED)])
    return rows


def _code_rows(text: str, lang: str, limit: int, hint: str = "") -> list:
    h = hl.Highlighter(lang)
    lines = text.rstrip("\n").split("\n") if text.strip() else []
    rows = [h.line(ln.replace("\t", "    ")) or [("", "")] for ln in lines[:limit]]
    if len(lines) > limit:
        rows.append([("… +%d more lines%s" % (len(lines) - limit, hint), b.MUTED)])
    return rows


def _numbered_rows(text: str, lang: str, limit: int, hint: str = "") -> list:
    lines = text.split("\n")
    parsed = [NUMBERED.match(ln) for ln in lines]
    if not all(parsed):
        return _text_rows(text, "", limit)
    h = hl.Highlighter(lang)
    w = max(len(m.group(1)) for m in parsed)
    rows = []
    for m in parsed[:limit]:
        rows.append([(m.group(1).rjust(w), b.MUTED), ("  ", "")] + (h.line(m.group(2).replace("\t", "    ")) or [("", "")]))
    if len(lines) > limit:
        rows.append([("… +%d more lines%s" % (len(lines) - limit, hint), b.MUTED)])
    return rows


def _split_file(args: str):
    name, _, rng = args.partition(",")
    return name.strip().strip("\"'"), rng.strip()


def _header(text: str) -> list:
    return [("▸ ", b.AMBER), (text, "bold")]


def _result_rows(result: str, ok: bool, limit: int) -> list:
    return [[("", "")]] + _text_rows(result, b.MUTED if ok else "red", limit)


def _search_rows(result: str, ok: bool) -> list:
    rows = []
    for ln in result.split("\n")[:OUT_ROWS * 2]:
        m = re.match(r"^(\d+\. )(.*?)( \(.*\))?$", ln)
        if m and not ln.startswith(" "):
            rows.append([(m.group(1), b.MUTED), (m.group(2), "bold"), (m.group(3) or "", b.MUTED)])
        elif ln.startswith(" "):
            rows.append([(ln.strip(), "")])
        else:
            rows.append([(ln, b.MUTED if ok else "red")])
    return rows


def tool_view(call, result, ok: bool, width: int, unicode: bool = True) -> View:
    """The header line and the padded block (title, rows, empty row) for one tool call. `result` None = still running."""
    head, title, rows = tool_parts(call, result, ok)
    return View(_header(head), box(title, rows, width, unicode) if rows else [])


def tool_parts(call, result, ok: bool):
    """(header text, block title, content rows) for one tool call, rows not padded: for screens that size blocks themselves."""
    kind, args, body = call.kind, call.args, call.body
    rows, title = [], kind
    if kind == "shell":
        head, title = "Running command in terminal", "terminal"
        if result is not None:
            h = hl.Highlighter("bash")
            rows = [[("$ ", b.MUTED)] + (h.line(ln) or [("", "")]) for ln in (args.split("\n") if args else [""])]
            rows += _result_rows(result, ok, OUT_ROWS)
    elif kind == "read":
        name, rng = _split_file(args)
        head, title = "Reading %s%s" % (name, " lines " + rng if rng else ""), name
        if result is not None:
            rows = _numbered_rows(result, hl.lang_for_path(name), CODE_ROWS, " · /open %s to see all" % name) if ok else _text_rows(result, "red", OUT_ROWS)
    elif kind in ("write", "edit"):
        name, rng = _split_file(args)
        verb = "Writing" if kind == "write" else "Editing"
        head, title = "%s %s%s" % (verb, name, " lines " + rng if rng and kind == "edit" else ""), name
        if result is not None:
            rows = _code_rows(body or "", hl.lang_for_path(name), CODE_ROWS, " · /open %s to see all" % name)
            rows += _result_rows(result, ok, 3) if result or not ok else []
    elif kind == "search":
        content, source = websearch.split_args(args)
        head, title = 'Searching %s for "%s"' % (source, content), source
        if result is not None:
            rows = _search_rows(result, ok)
    else:
        head = "Calling %s(%s)" % (kind, args) if kind != "?" else "Calling %s" % args
        if result is not None:
            rows = _text_rows(result, b.MUTED if ok else "red", OUT_ROWS)
    return head, title, rows


class FenceParser:
    """Splits streamed answer text into text and ``` code blocks.

    feed()/finish() return events: ("text", str), ("open", lang), ("code", one line), ("close",).
    A fence only counts at the start of a line; text is passed on at once unless it might still turn into a fence.
    """

    def __init__(self):
        self.in_code = False
        self.buf = ""

    def feed(self, text: str) -> list:
        ev = []
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            ev += self._line(line, True)
        if self.buf and not self.in_code and not self._might_fence(self.buf):
            ev.append(("text", self.buf))
            self.buf = ""
        return ev

    @staticmethod
    def _might_fence(partial: str) -> bool:
        p = partial.lstrip(" ")
        return p.startswith("```") or (p != "" and set(p) == {"`"})

    def _line(self, line: str, newline: bool) -> list:
        m = FENCE_LINE.match(line)
        if self.in_code:
            if m and not m.group(1):
                self.in_code = False
                return [("close",)]
            return [("code", line)]
        if m:
            self.in_code = True
            return [("open", m.group(1))]
        return [("text", line + ("\n" if newline else ""))]

    def finish(self) -> list:
        ev = []
        if self.buf:
            ev += self._line(self.buf, False) if not self.in_code else [("code", self.buf)]
            self.buf = ""
        if self.in_code:
            self.in_code = False
            ev.append(("close",))
        return ev
