"""The terminal screen: header, boxed input line with the "/" menu, streamed answer, tool calls and a live stats row."""
from __future__ import annotations

import os
import random
import shutil
import sys
import threading
import time
from dataclasses import dataclass

import re

from alethic.agent import blocks, brand, screenmap, tips, vibe, viewer
from alethic.agent.protocol import Call
from alethic.agent.keys import KeyReader
from alethic.agent.lineedit import LineEditor

CODES = brand.NAMED          # style names the screen understands; "#RRGGBB" colors work too
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[78]")
TRANSCRIPT_CAP = 200_000


def has_truecolor(env=None) -> bool:
    env = os.environ if env is None else env
    return (env.get("COLORTERM", "").lower() in ("truecolor", "24bit") or bool(env.get("WT_SESSION"))
            or env.get("TERM_PROGRAM", "") in ("vscode", "iTerm.app", "WezTerm", "ghostty"))


def reduced_motion(env=None) -> bool:
    return (os.environ if env is None else env).get("ALETHIC_REDUCED_MOTION", "") not in ("", "0")


def enable_ansi() -> bool:
    """Windows consoles need virtual-terminal mode switched on before they understand colors and cursor moves.
    Returns False when that did not work (very old consoles): the screen then stays plain."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if not k.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        return bool(k.SetConsoleMode(h, mode.value | 0x0004))
    except Exception:
        return False


def use_utf8_output() -> None:
    """Make Windows print Unicode: switch the console code page and the stdout encoding to UTF-8."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def is_fancy(env=None, os_name=None) -> bool:
    """Windows Terminal, VS Code, macOS and Linux terminals have fonts with the special symbols; the old console may not."""
    env = os.environ if env is None else env
    os_name = os_name or os.name
    if env.get("TERM") == "dumb":
        return False
    if os_name != "nt":
        return True
    return bool(env.get("WT_SESSION") or env.get("TERM_PROGRAM") or env.get("ConEmuANSI") == "ON")


def glyphs_for(unicode: bool, fancy: bool) -> dict:
    if unicode and fancy:
        return dict(dot="●", hook="⎿", bar="│", arrow="›", rule="─", spin="◐◓◑◒", star="✻", more="…", tool="▸")
    if unicode:
        return dict(dot="●", hook="└", bar="│", arrow=">", rule="─", spin="-\\|/", star="*", more="…", tool=">")
    return dict(dot="*", hook="|_", bar="|", arrow=">", rule="-", spin="-\\|/", star="*", more="...", tool=">")


def status_text(calls: int, tokens: int, seconds: float, rate: float) -> str:
    return "%d tool call%s · %d token%s · %.1fs · %.1f tok/s" % (calls, "" if calls == 1 else "s", tokens,
                                                              "" if tokens == 1 else "s", seconds, rate)


def _terminal_size():
    size = shutil.get_terminal_size((80, 24))
    return size.columns, size.lines


class Block:
    """One tool call on the screen: its header (clickable) and its output, shown open or collapsed."""

    def __init__(self, call, open_: bool):
        self.call, self.open, self.result = call, open_, None   # result: (text, ok) once the tool has answered


@dataclass
class Frame:
    lines: list
    cursor_row: int
    cursor_col: int


class UI:
    def __init__(self, out=None, color=None, unicode=None, isatty=None, fancy=None, live=None, animate=True,
                 size=None, keys=None, vt=True, truecolor=None):
        self.out = out or sys.stdout
        self.isatty = self.out.isatty() if isatty is None else isatty
        self.vt = vt
        self.color = (self.isatty and vt and "NO_COLOR" not in os.environ) if color is None else color
        enc = (getattr(self.out, "encoding", "") or "").lower()
        self.unicode = enc.startswith("utf") if unicode is None else unicode
        self.fancy = is_fancy() if fancy is None else fancy
        self.truecolor = has_truecolor() if truecolor is None else truecolor
        self.g = glyphs_for(self.unicode, self.fancy)
        self.dot, self.hook, self.arrow, self.bar = self.g["dot"], self.g["hook"], self.g["arrow"], self.g["bar"]
        self.live = (self.isatty and vt and os.environ.get("TERM") != "dumb" and not os.environ.get("ALETHIC_NO_LIVE")) if live is None else live
        self.animate = animate
        self._size = size or _terminal_size
        self.keys = keys if keys is not None else (KeyReader() if self.isatty and vt else None)
        self.history: list = []
        self._lock = threading.RLock()
        self._col0, self._last_style = True, "text"
        # one turn (your message and everything the model does for it)
        self._turn_on = self._pinned = False
        self._pin_rows = 0
        self._ticker = None
        self._paused = False
        self._t0 = self._phase_t0 = time.time()
        self._tokens = self._calls = 0
        self._gen, self._phase = 0.0, "generating"
        self._clock = time.time
        self._seed = random.randrange(1000)   # a different word order each session
        self.vibes = vibe.Vibes(self._seed)
        self._sleep = time.sleep
        self._tbuf = ""                       # plain copy of everything printed: the viewer shows its tail
        self.last_file = ""
        self._pending = None                  # the tool call whose result has not arrived yet
        self._fence = blocks.FenceParser()
        self._in_box = False
        self.details = True                   # False: new tool calls start collapsed (one line)
        self.mouse = False                    # off: the wheel scrolls the terminal. /mouse on lets a click on a tool header toggle its block
        self.log: list = []                   # everything printed: text and Block objects, so the screen can be redrawn
        self.blocks: list = []
        self._track = True

    # basics -------------------------------------------------------------
    def paint(self, text: str, *styles) -> str:
        styles = [x for x in styles if x]
        if not self.color or not styles:
            return text
        codes = [brand.style_code(brand.AMBER if x == "brand" else x, self.truecolor) for x in styles]
        return "\x1b[%sm%s\x1b[0m" % (";".join(codes), text)

    def paint_row(self, row, bg: str = "") -> str:
        """A row of (text, style) runs; with `bg` every run (blanks too) sits on that background color."""
        if not (bg and self.color):
            return "".join(self.paint(t, st) if st else t for t, st in row)
        back = brand.bg_code(bg, self.truecolor)
        out = []
        for t, st in row:
            fg = brand.style_code(brand.AMBER if st == "brand" else st, self.truecolor) if st else ""
            out.append("\x1b[%s%sm%s\x1b[0m" % (fg + ";" if fg else "", back, t))
        return "".join(out)

    def _write(self, text: str) -> None:
        with self._lock:
            self.out.write(text)
            self.out.flush()
            if text:
                self._col0 = text.endswith("\n")
                if self._track:
                    self._tbuf = (self._tbuf + ANSI.sub("", text))[-TRANSCRIPT_CAP - 8000:]

    def transcript(self) -> str:
        """Everything printed so far, without colors (about the last 200 KB)."""
        return self._tbuf[-TRANSCRIPT_CAP:]

    def _emit(self, text: str) -> None:
        """Print text that belongs to the conversation: it is also kept in the log so the screen can be redrawn."""
        if text:
            if self.log and isinstance(self.log[-1], str):
                self.log[-1] += text
            else:
                self.log.append(text)
            if len(self.log) > 4000:
                del self.log[:1000]
                self.blocks = [b for b in self.blocks if b in self.log]
        self._write(text)

    def line(self, text: str = "") -> None:
        self._emit(text + "\n")

    # screens ------------------------------------------------------------
    def _fit(self, text: str, width: int, keep_end: bool = False) -> str:
        """Shorten `text` to `width` columns with an ellipsis; a path keeps its end because that is the useful part."""
        if len(text) <= width:
            return text
        more = self.g["more"]
        room = max(0, width - len(more))
        return more + text[len(text) - room:] if keep_end else text[:room] + more

    def _intro_ok(self) -> bool:
        return bool(self.isatty and self.vt and self.animate and not reduced_motion())

    def banner(self, title: str, folder: str, auto: bool, effort: str = "") -> None:
        w = max(20, self._size()[0] - 1)
        mode = "runs commands without asking" if auto else "asks before running commands"
        self.line()
        if w >= 56:  # room for the two discs on the left
            if self._intro_ok():
                gaps = brand.intro_gaps()
                for i, gap in enumerate(gaps[:-1]):
                    if i:
                        self._write("\x1b[5A")
                    for row in brand.mark_rows(self.unicode and self.fancy, gap):
                        self._write("\r\x1b[2K  " + self.paint_row(row) + "\n")
                    self._sleep(0.36 / (len(gaps) - 1))
                self._write("\x1b[5A")
            room = w - brand.MARK_WIDTH - 4
            texts = [self.paint(self._fit(self.g["star"] + " " + title, room), "brand", "bold"),
                     self.paint(self._fit("%s%s" % ("effort %s · " % effort if effort else "", mode), room), "dim"),
                     self.paint("folder  " + self._fit(folder, room - 8, keep_end=True), "dim"),
                     self.paint(self._fit("/ for commands · Ctrl+C stops a reply · Ctrl+D quits", room), "dim"), ""]
            for row, text in zip(brand.mark_rows(self.unicode and self.fancy), texts):
                self.line("\r\x1b[2K  " + self.paint_row(row) + " " * (brand.MARK_WIDTH - sum(len(t) for t, _ in row) + 2) + text
                          if self.isatty and self.vt else "  " + self.paint_row(row) + "  " + text)
            self.line()
            return
        self.line("  " + self.paint(self._fit(self.g["star"] + " " + title, w - 2), "brand", "bold"))
        self.line(self.paint(self._fit("    %s%s" % ("effort %s · " % effort if effort else "", mode), w), "dim"))
        self.line(self.paint("    folder  " + self._fit(folder, w - 12, keep_end=True), "dim"))
        self.line(self.paint(self._fit("    / for commands · Ctrl+C stops a reply · Ctrl+D quits", w), "dim"))
        self.line()

    def clear(self) -> None:
        if self.isatty:
            self._write("\x1b[2J\x1b[H")

    # the input box ------------------------------------------------------
    def frame(self, editor: LineEditor, status: str, width: int, rows: int) -> Frame:
        """The lines of the input box: a rule, your line, a rule, then the menu or a hint. Pure: it draws nothing."""
        w = max(12, width - 1)
        rule = self.paint(self.g["rule"] * w, "dim")
        text, col = editor.window(w - 2)
        lines = [rule, self.paint(self.arrow + " ", "brand", "bold") + text, rule]
        sugg = editor.suggestions()
        if sugg:
            shown = max(1, min(8, rows - 6))
            sel = editor.selected
            start = 0 if sel < shown else min(sel - shown + 1, len(sugg) - shown)
            name_w = max(len(s.usage) for s in sugg)
            for i in range(start, min(len(sugg), start + shown)):
                s = sugg[i]
                head = "  %s " % (self.arrow if i == sel else " ")
                room = w - len(head) - name_w - 3
                more = self.g["more"]
                desc = s.description if len(s.description) <= room else (s.description[:room - len(more)] + more if room > len(more) else "")
                if i == sel:
                    lines.append(self.paint(head + s.usage.ljust(name_w) + "  " + desc, "brand", "bold"))
                else:
                    lines.append(head + s.usage.ljust(name_w) + "  " + self.paint(desc, "dim"))
            hint = "  Tab complete · Enter run · Esc close"
            count = "%d/%d" % (sel + 1, len(sugg))
            lines.append(self.paint((hint + count.rjust(w - len(hint)))[:w], "dim"))
        else:
            left = "  / commands · Ctrl+C stop · Ctrl+D quit"
            if len(left) + len(status) + 2 > w:
                left = "  / commands"
            if len(left) + len(status) + 2 > w:
                status = ""
            lines.append(self.paint(left + status.rjust(w - len(left)), "dim"))
        return Frame(lines, 1, len(self.arrow) + 1 + col)

    def _show(self, f: Frame, previous: Frame | None) -> None:
        body = "\n".join(f.lines)
        back = (len(f.lines) - 1) - f.cursor_row
        start = "\r\x1b[J" if previous is None else "\x1b[%dA\r\x1b[J" % previous.cursor_row
        self._write(start + body + ("\x1b[%dA" % back if back else "") + "\r\x1b[%dG" % (f.cursor_col + 1))
        self._col0 = False

    def _wipe(self, f: Frame) -> None:
        self._write(("\x1b[%dA" % f.cursor_row if f.cursor_row else "") + "\r\x1b[J")
        self._col0 = True

    def prompt(self, status: str = ""):
        """Read one line. Returns None on end of input; raises KeyboardInterrupt on Ctrl+C."""
        if not (self.keys and self.vt and self.isatty and self.keys.usable()):
            self._write(self.paint(self.arrow + " ", "brand", "bold"))
            try:
                text = input().strip()
            except EOFError:
                return None
            self._remember(text)
            return text
        click = self._clickable()
        if click:
            self.repaint()      # a fresh screen: content starts on row 1, so a click row can be mapped to a block exactly
        elif not self._col0:
            self._write("\n")
        editor, shown = LineEditor(self.history), None
        result = None
        if self.isatty and self.vt:
            self._write("\x1b[?25h")   # always show the text cursor at the prompt
        if hasattr(self.keys, "mouse"):
            self.keys.mouse = click
        try:
            with self.keys.session():
                if click:
                    self._write("\x1b[?1000h\x1b[?1006h")
                while result is None:
                    cols, rows = self._size()
                    f = self.frame(editor, status, cols, rows)
                    self._show(f, shown)
                    shown = f
                    key = self.keys.read()
                    if key.startswith("click:"):
                        if click and not editor.suggestions():
                            _, _, r = key.split(":")
                            hit = screenmap.hit(self.screen_lines(cols), int(r), rows, len(f.lines))
                            if hit is not None:
                                self.toggle(hit)
                                shown = None
                        continue
                    result = editor.feed(key)
        finally:
            if click:
                self._write("\x1b[?1000l\x1b[?1006l")
            if shown is not None:
                self._wipe(shown)
        if result.kind == "interrupt":
            raise KeyboardInterrupt
        if result.kind == "eof":
            return None
        text = result.text.strip()
        if text:
            self.line(self.paint(self.arrow + " ", "brand", "bold") + self.paint(text, "bold"))
        self._remember(text)
        return text

    def _remember(self, text: str) -> None:
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)

    # the answer ---------------------------------------------------------
    def _box_width(self) -> int:
        return max(24, min(100, self._size()[0] - 4))

    def _box_line(self, row) -> None:
        self.line("  " + self.paint_row(row, brand.BLOCK_BG))

    def stream(self, spans) -> None:
        for style, text in spans:
            if style in ("plan", "map", "thinking"):
                self._close_code()
                if style != self._last_style:
                    if not self._col0:
                        self._emit("\n")
                    self._emit(self.paint(self.g["tool"] + " " + ("plan" if style == "thinking" else style) + " ", "#A39E8A", "italic"))
                    text = text.lstrip(" ")
                self._last_style = style
                self._emit(self.paint(text, "#A39E8A", "italic"))
                continue
            if self._last_style in ("plan", "map", "thinking") and not self._col0:
                self._emit("\n")
            self._last_style = style
            self._events(self._fence.feed(text))

    def _events(self, events) -> None:
        w, uni = self._box_width(), self.unicode
        for ev in events:
            if ev[0] == "text":
                self._emit(ev[1])
            elif ev[0] == "open":
                if not self._col0:
                    self._emit("\n")
                self._hl, self._in_box = blocks.hl.Highlighter(ev[1]), True
                self._box_line(blocks.box_top(ev[1] or "code", w, uni))
            elif ev[0] == "code":
                self._box_line(blocks.box_row(self._hl.line(ev[1].replace("\t", "    ")) or [("", "")], w, uni))
            elif ev[0] == "close":
                self._box_line(blocks.box_bottom(w, uni))
                self._in_box = False

    def _close_code(self) -> None:
        self._events(self._fence.finish())

    def end_stream(self) -> None:
        self._close_code()
        if not self._col0:
            self._emit("\n")
        self._last_style = "text"

    def _header_text(self, b: Block) -> str:
        view = blocks.tool_view(b.call, None, True, self._box_width(), self.unicode)
        return "  " + self.paint_row([(self.g["tool"] + " ", brand.AMBER)] + view.header[1:]) + "\n"

    def _body_text(self, b: Block) -> str:
        """The block under its header: the boxed output when open, one summary line when collapsed."""
        if b.result is None:
            return ""
        text, ok = b.result
        if not b.open:
            lines = [ln for ln in text.split("\n") if ln.strip()] if text else []
            first = (lines[0].strip() if lines else "done")[:60]
            more = " (+%d lines hidden)" % (len(lines) - 1) if len(lines) > 1 else ""
            return self.paint("  %s %s%s · click the header to show" % (self.hook, first, more), "red" if not ok else "#A39E8A") + "\n"
        view = blocks.tool_view(b.call, text, ok, self._box_width(), self.unicode)
        return "".join("  " + self.paint_row(row, brand.BLOCK_BG) + "\n" for row in view.rows)

    def call(self, call) -> None:
        self.end_stream()
        self.set_phase("running")
        self.vibes.call()
        self._calls += 1
        if call.kind in ("read", "write", "edit"):
            self.last_file = blocks._split_file(call.args)[0]
        b = Block(call, self.details)
        self._pending = b
        self.blocks.append(b)
        self.log.append(b)
        self._write(self._header_text(b))

    def result(self, text: str, limit: int = 8, ok: bool = True, call=None) -> None:
        self.set_phase("generating")
        b = self._pending
        if b is None or (call is not None and b.call is not call):
            b = Block(call or Call("?", "result", None), self.details)
            self.blocks.append(b)
            self.log.append(b)
            self._write(self._header_text(b))
        self._pending = None
        b.result = (text, ok)
        self._write(self._body_text(b))

    # redraw and clicks --------------------------------------------------
    def set_details(self, value: bool) -> None:
        """Show or hide every tool block, and make new ones start that way."""
        self.details = value
        for b in self.blocks:
            b.open = value
        if self.isatty and self.vt and not self._turn_on:
            self.repaint()

    def toggle(self, b: Block) -> None:
        b.open = not b.open
        self.repaint()

    def repaint(self) -> None:
        """Clear the screen and print the whole conversation again from the log (the transcript is not touched)."""
        self._track = False
        try:
            self._write("\x1b[2J\x1b[H")
            for seg in self.log:
                self._write(seg if isinstance(seg, str) else self._header_text(seg) + self._body_text(seg))
        finally:
            self._track = True
        self._col0 = True

    def screen_lines(self, cols: int) -> list:
        segs = [seg if isinstance(seg, str) else screenmap.BlockMark(
            seg, ANSI.sub("", self._header_text(seg)).rstrip("\n"), ANSI.sub("", self._body_text(seg)).rstrip("\n").split("\n") if seg.result else [])
            for seg in self.log]
        return screenmap.physical_lines(segs, cols)

    def _clickable(self) -> bool:
        return bool(self.mouse and self.blocks and self.isatty and self.vt and self.keys and self.keys.usable())

    def view_file(self, name: str, text: str) -> None:
        """Show a whole file next to the end of the chat; press q to come back. Plain numbered text where that cannot work."""
        if not (self.keys and self.vt and self.isatty and self.keys.usable()):
            self.line(viewer.plain_text(name, text))
            return
        with self.keys.session():
            viewer.run(name, text, self.transcript()[-20000:], self.keys.read, self._write, self._size,
                       unicode=self.unicode, color=self.color, truecolor=self.truecolor)
        self._col0 = True

    def info(self, text: str) -> None:
        self.line(text)

    def warn(self, text: str) -> None:
        self.line(self.paint(text, "yellow"))

    def error(self, text: str) -> None:
        self.line(self.paint(text, "red"))

    def confirm(self, call) -> str:
        """Ask before a command that is not on the safe list: returns 'y', 'n' or 'a' (always)."""
        self.end_stream()
        self._paused = True
        try:
            self._write(self.paint("? run `%s`  " % call.args, "yellow") + self.paint("[y]es / [n]o / [a]lways: ", "dim"))
            try:
                ans = input().strip().lower()
            except EOFError:
                ans = "n"
        finally:
            self._paused = False
        return "a" if ans.startswith("a") else "y" if ans.startswith("y") else "n"

    # the live stats row -------------------------------------------------
    def set_phase(self, phase: str) -> None:
        if phase == self._phase:
            return
        with self._lock:
            now = time.time()
            if self._phase == "generating":
                self._gen += now - self._phase_t0
            self._phase, self._phase_t0 = phase, now

    def tick(self, tokens: int) -> None:
        self._tokens = tokens
        self.vibes.tokens(tokens)

    def stats(self):
        now = time.time()
        gen = self._gen + (now - self._phase_t0 if self._phase == "generating" else 0.0)
        return dict(calls=self._calls, tokens=self._tokens, seconds=now - self._t0,
                    rate=self._tokens / gen if gen > 0.05 else 0.0)

    def begin_turn(self) -> None:
        with self._lock:
            self._t0 = self._phase_t0 = time.time()
            self._tokens = self._calls = 0
            self.vibes.new_turn()
            self._gen, self._phase, self._turn_on = 0.0, "generating", True
            if self.live:
                self._pin()
        if self._pinned and self.animate and not self._ticker:
            stop = threading.Event()
            t = threading.Thread(target=lambda: [self.refresh() for _ in iter(lambda: not stop.wait(0.1), False)], daemon=True)
            self._ticker = (t, stop)
            t.start()

    PIN = 5   # rows kept at the bottom during an answer: rule, locked input line, rule, tip, status row

    def _pin(self) -> None:
        cols, rows = self._size()
        if rows < 10 or cols < 24:
            return
        self._pin_rows, self._pinned = rows, True
        self._write("\n" * self.PIN + "\x1b[%dA\x1b7\x1b[1;%dr\x1b8" % (self.PIN, rows - self.PIN))

    def _unpin(self) -> None:
        if self._pinned:
            rows = self._pin_rows
            clear = "".join("\x1b[%d;1H\x1b[2K" % r for r in range(rows - self.PIN + 1, rows + 1))
            self._write("\x1b7%s\x1b[r\x1b8" % clear)
            self._pinned = False

    def _locked_rows(self, width: int) -> list:
        """The input box while the model answers: visible, but it cannot send until the answer is done."""
        rule = self.paint(self.g["rule"] * width, "dim")
        text = ("  " + "locked while I answer · Ctrl+C stops")[:max(0, width - 2)]
        return [rule, self.paint(self.arrow + " ", "dim") + self.paint(text.strip(), "dim", "italic"), rule]

    def refresh(self) -> None:
        """Redraw the locked input box and the stats row at the bottom of the screen (about ten times a second)."""
        with self._lock:
            if not (self._turn_on and self._pinned) or self._paused:
                return
            cols, rows = self._size()
            move = ""
            if rows != self._pin_rows:
                old = self._pin_rows
                move = "".join("\x1b[%d;1H\x1b[2K" % r for r in range(old - self.PIN + 1, old + 1)) + "\x1b[1;%dr" % (rows - self.PIN)
                self._pin_rows = rows
            w = max(1, cols - 1)
            box = self._locked_rows(w) + [self.paint(("  tip: " + tips.tip_at(self.vibes.step, self._seed))[:w], "dim")]
            draw = "".join("\x1b[%d;1H\x1b[2K%s" % (rows - self.PIN + 1 + i, line) for i, line in enumerate(box))
            self.out.write("\x1b7%s%s\x1b[%d;1H\x1b[2K%s\x1b8" % (move, draw, rows, self._status_row(w)))
            self.out.flush()

    def _status_row(self, width: int) -> str:
        """A cycling spinner, a vibe word, then the numbers; cut to `width` columns."""
        spin = brand.spinner(0.0 if reduced_motion() else self._clock(), self.unicode and self.fancy)
        word, more = self.vibes.word(self._phase), self.g["more"]
        head = "%s %s%s" % (spin, word, more)
        if len(head) >= width:
            return self.paint(head[:width], "brand")
        tail = (" · " + status_text(**self.stats()))[:width - len(head)]
        return self.paint(spin + " ", "brand") + self.paint(word + more, "brand") + self.paint(tail, "dim")

    def end_turn(self) -> str:
        """Free the bottom row and print the final numbers. Safe to call twice."""
        if not self._turn_on:
            return ""
        if self._ticker:
            self._ticker[1].set()
            self._ticker[0].join()
            self._ticker = None
        self.end_stream()
        self._unpin()
        self._turn_on = False
        line = status_text(**self.stats())
        self.line(self.paint("  " + line, "dim"))
        return line

    def reset_terminal(self) -> None:
        """Always leave the terminal normal, even after a crash: frees the bottom row."""
        if self._ticker:
            self._ticker[1].set()
        self._unpin()
