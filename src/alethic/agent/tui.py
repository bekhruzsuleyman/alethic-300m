"""The Textual front-end: a scrolling chat you can click, a locked input line, a file pane on the right."""
from __future__ import annotations

import threading
import time

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from alethic.agent import blocks, brand, folderpick, menu, tips, vibe
from alethic.agent.commands import run_command
from alethic.agent.render import Renderer
from alethic.agent.tui_widgets import (AnswerText, CodeBlock, ConfirmScreen, DocPane, FolderScreen, PlanText, PromptInput,
                                       Splitter, ToolBlock, rows_to_text)
from alethic.agent.ui import status_text

MIN_CHAT, MIN_DOC = 24, 30

CSS = """
Screen { background: #0D0D0B; color: #F5E9C8; }
#main { height: 1fr; }
#chat { width: 1fr; padding: 0 1; background: #0D0D0B; scrollbar-size-vertical: 1; }
#split { width: 1; color: #A39E8A; background: #0D0D0B; display: none; }
#split:hover { color: #FFB020; }
#doc { width: 60; display: none; background: #14130F; }
#doc-head { height: 1; background: #1C1B17; }
#doc-title { width: 1fr; }
#doc-close { width: 5; color: #FFB020; background: #1C1B17; }
#doc-close:hover { background: #FFB020; color: #0D0D0B; }
#doc-scroll { height: 1fr; scrollbar-size-vertical: 1; scrollbar-size-horizontal: 1; }
#doc-text { width: auto; padding: 0 1; }
.user { margin-top: 1; color: #FFB020; text-style: bold; }
.answer { margin-top: 0; }
.plan { color: #A39E8A; }
.info { color: #A39E8A; }
.warn { color: #FFB020; }
.codeblock { background: #1C1B17; margin: 1 0; padding: 0 1; }
.toolblock { height: auto; margin-top: 1; }
.tb-head { height: 1; }
.tb-head:hover { background: #1C1B17; }
.tb-body { background: #1C1B17; padding: 0 1; margin-left: 2; }
#menu { display: none; height: auto; max-height: 8; background: #1C1B17; border: none; padding: 0 1; }
#tip { height: 1; color: #A39E8A; padding: 0 1; display: none; }
#status { height: 1; color: #FFB020; padding: 0 1; display: none; }
#prompt { border: tall #A39E8A; background: #14130F; }
#prompt:focus { border: tall #FFB020; }
#prompt:disabled { opacity: 55%; }
#foot { height: 1; color: #A39E8A; padding: 0 1; }
ConfirmScreen, FolderScreen { align: center middle; }
#confirm, #folder { width: 80%; max-width: 100; height: auto; background: #1C1B17; border: tall #FFB020; padding: 1 2; }
#confirm-cmd { margin: 1 0; }
#folder-list { height: auto; max-height: 10; margin: 1 0; background: #1C1B17; }
"""


def _in_main(app) -> bool:
    return threading.get_ident() == getattr(app, "_main_thread", None)


class TuiBridge:
    """What the Session sees as its `ui`: the commands and permission questions talk to this, and it talks to the screen."""

    def __init__(self):
        self.app = None
        self.details = True
        self.mouse = True
        self.last_file = ""

    def attach(self, app) -> None:
        self.app = app

    def _do(self, fn, *args):
        if self.app is None:
            return None
        if _in_main(self.app):
            return fn(*args)
        return self.app.call_from_thread(fn, *args)

    # messages
    def line(self, text: str = "") -> None:
        self._do(self.app.say_info, text)

    info = line

    def warn(self, text: str) -> None:
        self._do(self.app.say_info, text, "warn")

    error = warn

    def view_file(self, name: str, text: str) -> None:
        self._do(self.app.open_doc, name, text)

    def set_details(self, value: bool) -> None:
        self.details = value
        self._do(self.app.apply_details, value)

    def confirm(self, call) -> str:
        done, box = threading.Event(), []
        self._do(lambda: self.app.push_screen(ConfirmScreen(call), lambda a: (box.append(a), done.set())))
        done.wait()
        return box[0] if box and box[0] else "n"

    def pick_folder(self, default: str, first: str = "") -> str:
        if first:
            return first
        done, box = threading.Event(), []
        recent = folderpick.load_recent(folderpick.storage_path())
        self._do(lambda: self.app.push_screen(FolderScreen(recent, default), lambda a: (box.append(a), done.set())))
        done.wait()
        chosen = box[0] if box else None
        if chosen:
            folderpick.remember(folderpick.storage_path(), chosen)
        return chosen or ""


class AgentApp(App):
    CSS = CSS
    BINDINGS = [("pageup", "page('up')", "scroll up"), ("pagedown", "page('down')", "scroll down"),
                ("ctrl+left", "widen(4)", "wider file"), ("ctrl+right", "widen(-4)", "narrower file"),
                ("ctrl+c", "leave", "leave")]
    ENABLE_COMMAND_PALETTE = False

    class Token(Message):
        def __init__(self, text: str):
            super().__init__()
            self.text = text

    class Called(Message):
        def __init__(self, call):
            super().__init__()
            self.call = call

    class Resulted(Message):
        def __init__(self, call, out):
            super().__init__()
            self.call, self.out = call, out

    class Finished(Message):
        def __init__(self, message: str = "", command=None):
            super().__init__()
            self.message, self.command = message, command

    def __init__(self, session, bridge: TuiBridge):
        super().__init__()
        self.session, self.bridge = session, bridge
        bridge.attach(self)
        self.busy = False
        self.history, self.hpos = [], 0
        self.vibes, self.seed = vibe.Vibes(int(time.time()) % 9973), int(time.time()) % 9973
        self._reset_stream()
        self._t0 = self._last = 0.0
        self._tokens = self._calls = 0
        self._gen = 0.0
        self._phase = "generating"
        self._pending = None
        self.suggestions = []

    # layout ---------------------------------------------------------------
    def compose(self) -> ComposeResult:
        self.chat = VerticalScroll(id="chat")
        self.split = Splitter()
        self.doc = DocPane()
        yield Horizontal(self.chat, self.split, self.doc, id="main")
        yield OptionList(id="menu")
        yield Static("", id="tip", markup=False)
        yield Static("", id="status", markup=False)
        yield PromptInput(placeholder="message the model, or type / for commands", id="prompt")
        yield Static("", id="foot", markup=False)

    def on_mount(self) -> None:
        self._main_thread = threading.get_ident()
        self.chat.anchor()
        self.show_banner()
        self.update_foot()
        self.set_interval(0.1, self.tick)
        self.query_one("#prompt").focus()

    def show_banner(self) -> None:
        s = self.session
        rows = brand.mark_rows(True, 0)
        art = rows_to_text(rows)
        info = Text()
        info.append(s.title + "\n", style="bold " + brand.AMBER)
        info.append("folder  %s\n" % s.ws.root, style=brand.CREAM)
        info.append("%s · %s\n" % (s.effort, "runs commands without asking" if s.ws.auto else "asks before running commands"),
                    style=brand.MUTED)
        info.append("type / for commands · click a ▸ header to hide or show a block", style=brand.MUTED)
        self._add(Horizontal(Static(art, markup=False, classes="mark"), Static(info, markup=False), classes="banner"))

    def _add(self, widget):
        self.chat.mount(widget)
        return widget

    def update_foot(self) -> None:
        s = self.session
        self.query_one("#foot").update("%s · %s" % (s.effort, "auto" if s.ws.auto else "asks first"))

    def say_info(self, text: str, kind: str = "info") -> None:
        for ln in (text or " ").split("\n"):
            self._add(Static(Text(ln or " "), markup=False, classes=kind))

    # messages from the worker ----------------------------------------------
    def _reset_stream(self) -> None:
        self.renderer, self.fence = Renderer(), blocks.FenceParser()
        self.answer = self.plan = self.code = None
        self.last_style = "text"

    def stream(self, spans) -> None:
        for style, text in spans:
            if style in ("plan", "map", "thinking"):
                self._close_code()
                if style != self.last_style or self.plan is None:
                    self.plan = self._add(PlanText("plan" if style == "thinking" else style))
                    text = text.lstrip(" ")
                self.last_style = style
                self.plan.add(text)
                continue
            if self.last_style in ("plan", "map", "thinking"):
                self.plan = None
            self.last_style = style
            self._events(self.fence.feed(text))

    def _events(self, events) -> None:
        for ev in events:
            if ev[0] == "text":
                if self.answer is None:
                    self.answer = self._add(AnswerText())
                self.answer.add(ev[1])
            elif ev[0] == "open":
                self.answer = None
                self.code = self._add(CodeBlock(ev[1]))
            elif ev[0] == "code" and self.code is not None:
                self.code.add_line(ev[1])
            elif ev[0] == "close":
                self.code = self.answer = None

    def _close_code(self) -> None:
        self._events(self.fence.finish())
        self.answer = None

    @on(Token)
    def _token(self, m: "AgentApp.Token") -> None:
        self._tokens += 1
        self._phase = "generating"
        self.vibes.tokens(self._tokens)
        self.stream(self.renderer.feed(m.text))

    @on(Called)
    def _called(self, m: "AgentApp.Called") -> None:
        self.stream(self.renderer.finish())
        self._close_code()
        self.renderer = Renderer()
        self.answer = self.plan = None
        self._phase = "running"
        self._calls += 1
        self.vibes.call()
        kind_file = blocks._split_file(m.call.args)[0] if m.call.kind in ("read", "write", "edit") else ""
        if kind_file:
            self.bridge.last_file = kind_file
        self._pending = self._add(ToolBlock(m.call, self.bridge.details))

    @on(Resulted)
    def _resulted(self, m: "AgentApp.Resulted") -> None:
        self._phase = "generating"
        out = m.out
        text = out.result if out.result is not None else ("" if out.ok else "error")
        block = self._pending
        if block is None or block.call is not m.call:
            block = self._add(ToolBlock(m.call, self.bridge.details))
        self._pending = None
        block.set_result(text, out.ok)

    @on(Finished)
    def _finished(self, m: "AgentApp.Finished") -> None:
        self.stream(self.renderer.finish())
        self._close_code()
        if m.message:
            self.say_info(m.message, "warn")
        if self._turn_on:
            self._add(Static(Text(status_text(self._calls, self._tokens, time.monotonic() - self._t0,
                                              self._tokens / self._gen if self._gen > 0.05 else 0.0), style=brand.MUTED),
                             markup=False, classes="info"))
        self._turn_on = False
        cmd = m.command
        if cmd is not None:
            if cmd.message:
                self.say_info(cmd.message)
            if cmd.clear:
                self.clear_chat()
            if cmd.quit:
                self.exit()
                return
        self.set_busy(False)
        self.update_foot()
        if cmd is not None and cmd.rerun:
            self.start_turn(cmd.rerun)

    _turn_on = False

    # running ----------------------------------------------------------------
    def set_busy(self, value: bool) -> None:
        self.busy = value
        box = self.query_one("#prompt")
        box.disabled = value
        self.query_one("#status").display = value
        self.query_one("#tip").display = value
        if not value:
            box.focus()

    def start_turn(self, message: str) -> None:
        self._reset_stream()
        self._pending = None
        self._t0 = self._last = time.monotonic()
        self._tokens = self._calls = 0
        self._gen = 0.0
        self._phase = "generating"
        self._turn_on = True
        self.vibes.new_turn()
        self.set_busy(True)
        self.tick()
        s = self.session

        def work():
            def on_event(kind, *a):
                if kind == "token":
                    self.post_message(self.Token(a[0]))
                elif kind == "call":
                    self.post_message(self.Called(a[0]))
                elif kind == "result":
                    self.post_message(self.Resulted(a[0], a[1]))
            note = ""
            try:
                res = s.agent.ask(message, on_event)
                if res.truncated:
                    note = "stopped after %d tool calls (%s)" % (res.steps, res.reason)
            except Exception as exc:  # the model or a tool broke: show it, keep the chat alive
                note = "error: %s" % exc
            self.post_message(self.Finished(note))

        self.run_worker(work, thread=True, exclusive=False)

    def start_command(self, line: str) -> None:
        self.set_busy(True)
        self._turn_on = False
        s = self.session

        def work():
            try:
                cmd = run_command(line, s)
            except Exception as exc:
                self.post_message(self.Finished("error: %s" % exc))
                return
            self.post_message(self.Finished("", cmd))

        self.run_worker(work, thread=True, exclusive=False)

    def clear_chat(self) -> None:
        self.chat.remove_children()
        self._reset_stream()
        self._pending = None
        self.close_doc()

    def tick(self) -> None:
        if not self.busy or not self._turn_on:
            return
        now = time.monotonic()
        if self._phase == "generating":
            self._gen += now - self._last
        self._last = now
        t = now - self._t0
        stats = status_text(self._calls, self._tokens, t, self._tokens / self._gen if self._gen > 0.05 else 0.0)
        word = self.vibes.word(self._phase)
        self.query_one("#status").update(Text("%s %s…  %s" % (brand.spinner(t, True), word, stats), no_wrap=True,
                                              overflow="ellipsis"))
        self.query_one("#tip").update(Text("tip: " + tips.tip_at(self._calls * tips.EVERY + self._tokens // 200, self.seed),
                                           no_wrap=True, overflow="ellipsis"))

    # input ------------------------------------------------------------------
    @on(PromptInput.Submitted, "#prompt")
    def _submit(self, event: PromptInput.Submitted) -> None:
        value = event.value.strip()
        box = event.input
        if self.busy or not value:
            return
        if self.suggestions and value.startswith("/") and " " not in value:
            top = self.suggestions[self._menu_index()]
            if top.text != value and top.text.startswith(value) and not any(c.name == value[1:] for c in menu.COMMANDS):
                self._complete(box)
                return
        box.value = ""
        self.hide_menu()
        self.history.append(value)
        self.hpos = len(self.history)
        if value.startswith("/"):
            self.start_command(value)
        else:
            self._add(Static(Text("› " + value), markup=False, classes="user"))
            self.start_turn(value)

    @on(PromptInput.Changed, "#prompt")
    def _changed(self, event: PromptInput.Changed) -> None:
        self.suggestions = menu.suggest(event.value) if event.value.startswith("/") else []
        m = self.query_one("#menu", OptionList)
        if not self.suggestions or self.busy:
            m.display = False
            return
        m.clear_options()
        m.add_options([Option("%-18s %s" % (s.usage, s.description)) for s in self.suggestions])
        m.highlighted = 0
        m.display = True

    def hide_menu(self) -> None:
        self.suggestions = []
        self.query_one("#menu").display = False

    def _menu_index(self) -> int:
        h = self.query_one("#menu", OptionList).highlighted
        return min(h or 0, len(self.suggestions) - 1)

    def _complete(self, box) -> None:
        top = self.suggestions[self._menu_index()]
        box.value = top.text + (" " if menu.takes_argument(top.text) else "")
        box.cursor_position = len(box.value)

    def prompt_key(self, key: str) -> bool:
        box = self.query_one("#prompt")
        m = self.query_one("#menu", OptionList)
        if self.suggestions and m.display:
            if key == "tab":
                self._complete(box)
            elif key == "down":
                m.action_cursor_down()
            else:
                m.action_cursor_up()
            return True
        if key == "tab":
            return True
        if not self.history or self.busy:
            return True
        self.hpos = max(0, self.hpos - 1) if key == "up" else min(len(self.history), self.hpos + 1)
        box.value = self.history[self.hpos] if self.hpos < len(self.history) else ""
        box.cursor_position = len(box.value)
        return True

    # file pane and scrolling ---------------------------------------------------
    def open_doc(self, name: str, text: str) -> None:
        self.doc.show(name, text)
        self.doc.display = self.split.display = True
        if not getattr(self, "_doc_set", False):
            self._doc_set = True
            self.doc.styles.width = max(MIN_DOC, self.size.width * 55 // 100)

    def close_doc(self) -> None:
        self.doc.display = self.split.display = False

    async def set_doc_width(self, width: int) -> None:
        self.doc.styles.width = max(MIN_DOC, min(width, self.size.width - MIN_CHAT - 1))

    def action_widen(self, step: str) -> None:
        if self.doc.display:
            self.doc.styles.width = max(MIN_DOC, min(self.doc.region.width + int(step), self.size.width - MIN_CHAT - 1))

    @on(Splitter.Dragged)
    def _dragged(self, m: Splitter.Dragged) -> None:
        self.doc.styles.width = max(MIN_DOC, min(self.size.width - m.screen_x, self.size.width - MIN_CHAT - 1))

    def apply_details(self, value: bool) -> None:
        for b in self.query(ToolBlock):
            b.set_open(value)

    def action_page(self, direction: str) -> None:
        (self.chat.scroll_page_up if direction == "up" else self.chat.scroll_page_down)(animate=False)

    def action_leave(self) -> None:
        self.exit()
