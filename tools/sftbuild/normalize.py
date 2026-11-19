"""Parse every example format (long/short header, text fences or plain tags) into one record."""
import re
from dataclasses import dataclass

HEADER = re.compile(r"^### ex (\d+) \|\s*(.*)$")


@dataclass
class Example:
    id: str
    type: str
    lang: str
    source: str
    user: str
    assistant: str


def _fields(rest: str) -> dict:
    parts = [p.strip() for p in rest.split("|")]
    if all(":" in p for p in parts):  # long form: key: value
        return {k.strip(): v.strip() for k, v in (p.split(":", 1) for p in parts)}
    return dict(zip(["type", "lang", "source"], parts))  # short form


def _fenced(body: str) -> list:
    """Outermost fenced blocks, closed by the same number of backticks that opened them."""
    out, lines, i = [], body.split("\n"), 0
    while i < len(lines):
        m = re.match(r"^(`{3,})text\s*$", lines[i])
        if m:
            close, j = m.group(1), i + 1
            while j < len(lines) and lines[j].rstrip() != close:
                j += 1
            out.append("\n".join(lines[i + 1:j]))
            i = j
        i += 1
    return out


def parse_examples(text: str) -> list:
    blocks, cur = [], None
    for line in text.split("\n"):
        m = HEADER.match(line)
        if m:
            cur = [m, []]
            blocks.append(cur)
        elif cur is not None:
            cur[1].append(line)
    out = []
    for m, body_lines in blocks:
        body = "\n".join(body_lines).strip("\n")
        f = _fields(m.group(2))
        wrapped = re.fullmatch(r"(`{3,})text\n(<user>\n.*</assistant>)\n\1", body, re.S)
        if wrapped:
            body = wrapped.group(2)
        tag = re.search(r"<user>\n(.*?)\n</user>\n<assistant>\n(.*)\n</assistant>\s*$", body, re.S)
        if tag:
            user, asst = tag.group(1), tag.group(2)
        else:
            fb = _fenced(body)
            user, asst = fb[0], fb[1]
        out.append(Example(m.group(1), f.get("type", ""), f.get("lang", ""), f.get("source", ""), user, asst))
    return out
