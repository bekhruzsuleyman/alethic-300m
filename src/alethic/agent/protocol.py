"""The text protocol the SFT data teaches: <call> kind(args) </call>, then a <result> block written by the harness.

shell, read and search calls are complete at </call>. write and edit are followed by the file text, which runs
until the next <call> or </assistant>; the harness cuts the generated text there.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

OPEN, CLOSE, END = "<call>", "</call>", "</assistant>"
WITH_BODY = ("write", "edit")
HEADER = re.compile(r"^\s*(\w+)\((.*)\)\s*$", re.S)


@dataclass
class Call:
    kind: str
    args: str
    body: str | None = None


@dataclass
class Scan:
    call: Call | None = None  # a call that is ready to run
    cut: int = 0              # how much of the text belongs to that call
    done: bool = False        # the model closed its turn


def scan(text: str) -> Scan:
    """Look at the text generated so far and say whether a tool call is ready or the turn is over."""
    start = text.find(OPEN)
    end = text.find(END)
    if start < 0 or (0 <= end < start):
        return Scan(done=end >= 0)
    close = text.find(CLOSE, start)
    if close < 0:
        return Scan()
    m = HEADER.match(text[start + len(OPEN):close])
    after = close + len(CLOSE)
    if text[after:after + 1] == "\n":
        after += 1
    if not m:
        return Scan(call=Call("?", text[start + len(OPEN):close].strip()), cut=after)
    kind, args = m.group(1), m.group(2).strip()
    if kind not in WITH_BODY:
        return Scan(call=Call(kind, args), cut=after)
    stops = [i for i in (text.find(OPEN, after), text.find(END, after)) if i >= 0]
    if not stops:
        return Scan()
    stop = min(stops)
    return Scan(call=Call(kind, args, text[after:stop]), cut=stop)
