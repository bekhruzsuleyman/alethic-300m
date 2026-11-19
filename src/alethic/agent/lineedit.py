"""The input line as a small state machine: typing, cursor keys, history and the "/" menu. It draws nothing."""
from __future__ import annotations

from dataclasses import dataclass

from alethic.agent.menu import suggest, takes_argument


@dataclass
class Result:
    kind: str        # "submit", "interrupt" or "eof"
    text: str = ""


class LineEditor:
    def __init__(self, history=None):
        self.history = list(history or [])
        self.buffer, self.cursor = "", 0
        self.selected, self.closed = 0, False
        self._hist_at, self._draft = len(self.history), ""

    # menu ---------------------------------------------------------------
    def suggestions(self) -> list:
        return [] if self.closed else suggest(self.buffer)

    def _set(self, text: str, cursor: int | None = None) -> None:
        self.buffer = text
        self.cursor = len(text) if cursor is None else cursor
        self.selected, self.closed = 0, False

    def _changed(self) -> None:
        self.selected, self.closed = 0, False
        self._hist_at = len(self.history)

    # drawing helper -----------------------------------------------------
    def window(self, width: int):
        """The part of the line that fits in `width` columns and where the cursor sits inside it."""
        width = max(4, width)
        start = self.cursor - width + 1 if self.cursor >= width else 0
        return self.buffer[start:start + width], self.cursor - start

    # keys ---------------------------------------------------------------
    def feed(self, key: str):
        """Handle one key. Returns a Result when the line is finished, otherwise None."""
        if len(key) == 1:
            if ord(key) >= 32:
                self.buffer = self.buffer[:self.cursor] + key + self.buffer[self.cursor:]
                self.cursor += 1
                self._changed()
            return None
        n = len(self.suggestions())
        if key == "enter":
            return self._enter()
        if key == "ctrl-c":
            return Result("interrupt")
        if key == "ctrl-d":
            if not self.buffer:
                return Result("eof")
            return self.feed("delete")
        if key == "backspace" and self.cursor:
            self.buffer = self.buffer[:self.cursor - 1] + self.buffer[self.cursor:]
            self.cursor -= 1
            self._changed()
        elif key == "delete" and self.cursor < len(self.buffer):
            self.buffer = self.buffer[:self.cursor] + self.buffer[self.cursor + 1:]
            self._changed()
        elif key == "left":
            self.cursor = max(0, self.cursor - 1)
        elif key == "right":
            self.cursor = min(len(self.buffer), self.cursor + 1)
        elif key == "home":
            self.cursor = 0
        elif key == "end":
            self.cursor = len(self.buffer)
        elif key == "ctrl-u":
            self.buffer, self.cursor = self.buffer[self.cursor:], 0
            self._changed()
        elif key == "ctrl-w":
            head = self.buffer[:self.cursor].rstrip()
            head = head[:head.rfind(" ") + 1]
            self.buffer, self.cursor = head + self.buffer[self.cursor:], len(head)
            self._changed()
        elif key == "tab" and n:
            self._complete(self.suggestions()[self.selected].text)
        elif key == "esc":
            self.closed = True
        elif key in ("up", "down"):
            if n:
                self.selected = (self.selected + (1 if key == "down" else -1)) % n
            else:
                self._walk_history(-1 if key == "up" else 1)
        return None

    def _complete(self, text: str) -> None:
        self._set(text + " " if takes_argument(text) else text)

    def _enter(self):
        sugg = self.suggestions()
        if sugg:
            pick = sugg[self.selected].text
            if pick != self.buffer:
                self._complete(pick)
                if takes_argument(pick):
                    return None          # it needs text after it: fill it in and wait
        return Result("submit", self.buffer)

    def _walk_history(self, step: int) -> None:
        if not self.history:
            return
        if self._hist_at == len(self.history):
            self._draft = self.buffer
        at = min(max(self._hist_at + step, 0), len(self.history))
        if at == self._hist_at:
            return
        self._hist_at = at
        keep = at
        self._set(self._draft if at == len(self.history) else self.history[at])
        self._hist_at = keep
