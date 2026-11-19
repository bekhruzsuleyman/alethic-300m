"""One conversation: generate, run any tool call the model wrote, put the real result in the history, repeat."""
from __future__ import annotations

from dataclasses import dataclass

from alethic.agent.protocol import END, scan


@dataclass
class TurnResult:
    steps: int = 0          # tool calls run this turn
    truncated: bool = False  # stopped by the step limit
    answer: str = ""        # text of the last model step
    reason: str = ""        # why the turn was cut short


MAX_FAILS = 3


def _stop(text: str) -> bool:
    s = scan(text)
    return s.call is not None or s.done


class Agent:
    def __init__(self, generate, workspace, max_steps=10, max_new=400):
        self.generate, self.ws, self.max_steps, self.max_new = generate, workspace, max_steps, max_new
        self.history = ""
        self.last_message = ""
        self._marks = []  # history length before each turn, for /undo and /compact

    def reset(self):
        self.history, self._marks = "", []

    @property
    def turns(self) -> int:
        return len(self._marks)

    def compact(self, keep: int = 2) -> int:
        """Forget the oldest turns, keeping the last `keep`, so the conversation fits the context again. Returns how many were dropped."""
        if len(self._marks) <= keep:
            return 0
        cut, dropped = self._marks[-keep], len(self._marks) - keep
        self.history = self.history[cut:]
        self._marks = [m - cut for m in self._marks[-keep:]]
        return dropped

    def undo(self):
        if self._marks:
            self.history = self.history[:self._marks.pop()]

    def ask(self, message: str, on_event=lambda *e: None) -> TurnResult:
        """Run one user message to the end of the model's answer. Events: ('token', text), ('call', Call),
        ('result', Call, Outcome)."""
        self.last_message = message
        self._marks.append(len(self.history))
        self.history += "<user>\n%s\n</user>\n<assistant>\n" % message
        res, fails = TurnResult(), 0
        try:
            while True:
                text = self.generate(self.history, _stop, self.max_new, lambda t: on_event("token", t))
                s = scan(text)
                res.answer = text
                if s.call is None:
                    self.history += text if s.done else text.rstrip("\n") + "\n" + END
                    self.history = self.history.rstrip("\n") + "\n"
                    return res
                self.history += text[:s.cut]
                if not self.history.endswith("\n"):
                    self.history += "\n"
                on_event("call", s.call)
                out = self.ws.execute(s.call)
                if out.result is not None:
                    self.history += "<result>\n%s\n</result>\n" % out.result
                on_event("result", s.call, out)
                res.steps += 1
                fails = 0 if out.ok else fails + 1
                if fails >= MAX_FAILS or res.steps >= self.max_steps:
                    res.truncated = True
                    res.reason = "%d failed tool calls in a row" % fails if fails >= MAX_FAILS else "step limit"
                    self.history += END + "\n"
                    return res
        except BaseException:
            self.history = self.history[:self._marks.pop()]  # an interrupted turn leaves no half conversation
            raise
