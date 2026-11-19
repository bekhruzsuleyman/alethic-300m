"""The agent as a program a screen can talk to: one JSON object per line in and out. The Ink (React) screen in ui/ uses it.

Out: hello, text, plan, code_open, code, code_close, call, tool, status, info, clear, doc, confirm, details, done, quit.
In:  submit {text}, confirm {answer}.
Text and code are already split and colored here (same Renderer, FenceParser and Highlighter as the classic screen),
so the screen only has to draw them.
"""
from __future__ import annotations

import json
import queue
import threading
import time

from alethic.agent import blocks, menu, tips, vibe
from alethic.agent import highlight as hl
from alethic.agent.commands import run_command
from alethic.agent.render import Renderer


# the tips that fit the Ink screen: no mouse clicks or side viewer there
INK_TIPS = [t for t in tips.TIPS if not any(w in t for w in ("Click", "beside", "/mouse"))] + [
    "Ctrl+O folds or opens every tool block, the new ones too.",
    "/open prints a whole file with line numbers.",
    "The mouse wheel scrolls the chat. /clear starts clean.",
]


def summary_of(text: str) -> str:
    lines = [ln for ln in (text or "").split("\n") if ln.strip()]
    first = (lines[0].strip() if lines else "done")[:60]
    return first + (" (+%d lines)" % (len(lines) - 1) if len(lines) > 1 else "")


class JsonBridge:
    """The Session's `ui`: everything it would print becomes an event."""

    def __init__(self, out):
        self._out, self._lock = out, threading.Lock()
        self.details, self.mouse, self.last_file = True, True, ""
        self.answers = queue.Queue()

    def send(self, **event) -> None:
        with self._lock:
            self._out.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._out.flush()

    def line(self, text: str = "") -> None:
        self.send(t="info", text=text, kind="info")

    info = line

    def warn(self, text: str) -> None:
        self.send(t="info", text=text, kind="warn")

    error = warn

    def view_file(self, name: str, text: str) -> None:
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        h = hl.Highlighter(hl.lang_for_path(name))
        self.send(t="doc", name=name, rows=[h.line(ln.replace("\t", "    ")) or [["", ""]] for ln in lines])

    def set_details(self, value: bool) -> None:
        self.details = value
        self.send(t="details", value=value)

    def confirm(self, call) -> str:
        self.send(t="confirm", cmd=hl.Highlighter("bash").line(call.args))
        return self.answers.get()

    def pick_folder(self, default: str, first: str = "") -> str:
        return first or ""


class Turn:
    """One answer streamed out as events."""

    def __init__(self, bridge: JsonBridge):
        self.b = bridge
        self.r, self.fence, self.hl = Renderer(), blocks.FenceParser(), None
        self.calls = self.tokens = 0
        self.t0 = self.last = time.monotonic()
        self.gen, self.phase = 0.0, "generating"
        self.vibes = vibe.Vibes(int(self.t0) % 9973)
        self.seed = int(self.t0) % 9973
        self.pending = None

    def spans(self, spans) -> None:
        for style, text in spans:
            if style in ("plan", "map", "thinking"):
                self._events(self.fence.finish())
                self.b.send(t="plan", kind="plan" if style == "thinking" else style, text=text)
            else:
                self._events(self.fence.feed(text))

    def _events(self, events) -> None:
        for ev in events:
            if ev[0] == "text":
                self.b.send(t="text", text=ev[1])
            elif ev[0] == "open":
                self.hl = hl.Highlighter(ev[1])
                self.b.send(t="code_open", lang=ev[1] or "code")
            elif ev[0] == "code" and self.hl:
                self.b.send(t="code", runs=self.hl.line(ev[1].replace("\t", "    ")) or [["", ""]])
            elif ev[0] == "close":
                self.b.send(t="code_close")

    def event(self, kind, *a) -> None:
        if kind == "token":
            self.tokens += 1
            self.phase = "generating"
            self.vibes.tokens(self.tokens)
            self.spans(self.r.feed(a[0]))
        elif kind == "call":
            self.spans(self.r.finish())
            self._events(self.fence.finish())
            self.r = Renderer()
            call = a[0]
            self.phase = "running"
            self.calls += 1
            self.vibes.call()
            head, _, _ = blocks.tool_parts(call, None, True)
            if call.kind in ("read", "write", "edit"):
                self.b.last_file = blocks._split_file(call.args)[0]
            self.pending = call
            self.b.send(t="call", head=head)
        elif kind == "result":
            call, out = a
            self.phase = "generating"
            text = out.result if out.result is not None else ("" if out.ok else "error")
            head, title, rows = blocks.tool_parts(call, text, out.ok)
            self.b.send(t="tool", head=head, title=title, rows=rows, ok=bool(out.ok), summary=summary_of(text))

    def status(self) -> None:
        now = time.monotonic()
        if self.phase == "generating":
            self.gen += now - self.last
        self.last = now
        self.b.send(t="status", calls=self.calls, tokens=self.tokens, seconds=now - self.t0,
                    rate=self.tokens / self.gen if self.gen > 0.05 else 0.0,
                    word=self.vibes.word(self.phase),
                    tip=tips.tip_at(self.calls * tips.EVERY + self.tokens // 200, self.seed, INK_TIPS))


def run_turn(session, bridge: JsonBridge, message: str) -> None:
    turn = Turn(bridge)
    stop = threading.Event()

    def ticker():
        while not stop.wait(0.2):
            turn.status()
    t = threading.Thread(target=ticker, daemon=True)
    t.start()
    note = ""
    try:
        res = session.agent.ask(message, turn.event)
        turn.spans(turn.r.finish())
        turn._events(turn.fence.finish())
        if res.truncated:
            note = "stopped after %d tool calls (%s)" % (res.steps, res.reason)
    except Exception as exc:
        note = "error: %s" % exc
    finally:
        stop.set()
        t.join()
    turn.status()
    bridge.send(t="done", note=note, calls=turn.calls, tokens=turn.tokens, seconds=time.monotonic() - turn.t0,
                rate=turn.tokens / turn.gen if turn.gen > 0.05 else 0.0)


def handle(session, bridge: JsonBridge, text: str) -> bool:
    """Run one line from the user. False = quit."""
    cmd = run_command(text, session) if text.startswith("/") else None
    if text.startswith("/") and cmd is not None:
        if cmd.message:
            bridge.send(t="info", text=cmd.message, kind="info")
        if cmd.clear:
            bridge.send(t="clear")
        if cmd.quit:
            return False
        bridge.send(t="done", note="", calls=0, tokens=0, seconds=0, rate=0, command=True,
                    effort=session.effort, auto=session.ws.auto, folder=session.ws.root)
        if cmd.rerun:
            run_turn(session, bridge, cmd.rerun)
        return True
    run_turn(session, bridge, text)
    return True


def serve(session, inp, out) -> None:
    bridge = session.ui
    bridge._out = out
    bridge.send(t="hello", title=session.title, folder=session.ws.root, effort=session.effort, auto=session.ws.auto,
                commands=[dict(name=c.name, usage=c.usage, description=c.description) for c in menu.COMMANDS])
    jobs, alive = queue.Queue(), True

    def worker():
        while True:
            text = jobs.get()
            if text is None:
                return
            if not handle(session, bridge, text):
                bridge.send(t="quit")
                return
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    for raw in inp:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except ValueError:
            continue
        if msg.get("t") == "submit":
            jobs.put(msg.get("text", ""))
        elif msg.get("t") == "confirm":
            bridge.answers.put(msg.get("answer", "n"))
    jobs.put(None)
    th.join(timeout=60)
