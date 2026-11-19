"""Turn the model's raw tagged text into what the screen shows: plain text, dim plan and map, no calls or file bodies."""
from __future__ import annotations

import re

THINK_OPEN = ("<think>", "<thinking>", "<thought>", "<reasoning>")
THINK_CLOSE = ("</think>", "</thinking>", "</thought>", "</reasoning>")
TAGS = ("<plan>", "</plan>", "<map>", "</map>", "<call>", "</call>", "</assistant>", "<assistant>", "<result>", "</result>") + THINK_OPEN + THINK_CLOSE
SHOWN = ("text", "plan", "map", "thinking")
# protocol-looking tags the model may write anywhere: never shown, the words between them are kept
EXTRA = ("<user>", "</user>", "<system>", "</system>", "<tool>", "</tool>")
EXTRA_RE = re.compile(r"</?(?:user|system|tool|assistant|call|result|plan|map|think|thinking|thought|reasoning)>|<\|[^|>\n]*\|>")
SPECIAL_PARTIAL = re.compile(r"^<\|[^|>\n]*\|?$")


def _partial_tag(text: str) -> int:
    """Length of a trailing piece of text that could still become a tag ('<pl'), else 0."""
    i = text.rfind("<")
    if i < 0:
        return 0
    tail = text[i:]
    if SPECIAL_PARTIAL.match(tail):
        return len(tail)
    return len(tail) if any(t.startswith(tail) and t != tail for t in TAGS + EXTRA) else 0


class Renderer:
    """feed() returns [(style, text)] with style 'text', 'plan', 'map' or 'thinking'. Calls and file bodies are never shown here."""

    def __init__(self):
        self.buf, self.mode, self.header = "", "text", ""

    def feed(self, chunk: str) -> list:
        self.buf += chunk
        hold = _partial_tag(self.buf)
        work, self.buf = self.buf[:len(self.buf) - hold], self.buf[len(self.buf) - hold:]
        return self._consume(work)

    def finish(self) -> list:
        work, self.buf = self.buf, ""
        hold = _partial_tag(work)
        if hold >= 2:  # a tag that never finished is noise, not text
            work = work[:len(work) - hold]
        return self._consume(work)

    def _consume(self, text: str) -> list:
        out = []
        while text:
            hit = [(text.find(t), t) for t in TAGS if t in text]
            if not hit:
                self._emit(out, text)
                break
            pos, tag = min(hit)
            self._emit(out, text[:pos])
            text = text[pos + len(tag):]
            if tag == "<call>":
                self.mode, self.header = "call", ""
            elif tag == "</call>":
                kind = self.header.strip().split("(", 1)[0]
                self.mode = "body" if kind in ("write", "edit") else "text"
            elif tag == "<plan>":
                self.mode = "plan"
            elif tag == "<map>":
                self.mode = "map"
            elif tag == "<result>":
                self.mode = "result"
            elif tag in THINK_OPEN:
                self.mode = "thinking"
            else:  # closing tags and assistant tags
                self.mode = "text"
        return out

    def _emit(self, out, text):
        if not text:
            return
        if self.mode == "call":
            self.header += text
        elif self.mode in SHOWN:
            text = EXTRA_RE.sub("", text)
            if text:
                out.append((self.mode, text))
