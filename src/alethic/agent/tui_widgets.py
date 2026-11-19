"""The pieces of the Textual screen: tool blocks you can click, code blocks, the file pane with a draggable edge, dialogs."""
from __future__ import annotations

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from alethic.agent import blocks, brand
from alethic.agent import highlight as hl


def rich_style(style: str):
    """Our style names ("bold", "red", "brand", "#RRGGBB") as a Rich style string."""
    if not style:
        return None
    return brand.AMBER if style == "brand" else style


def runs_to_text(runs, wrap: bool = False) -> Text:
    t = Text(no_wrap=not wrap, overflow="fold" if wrap else "ellipsis")
    for text, style in runs:
        t.append(text, style=rich_style(style))
    return t


def rows_to_text(rows, wrap: bool = False) -> Text:
    out = Text(no_wrap=not wrap, overflow="fold" if wrap else "ellipsis")
    for i, row in enumerate(rows):
        if i:
            out.append("\n")
        out.append_text(runs_to_text(row, wrap))
    return out


def summary_of(text: str, ok: bool) -> str:
    lines = [ln for ln in (text or "").split("\n") if ln.strip()]
    first = (lines[0].strip() if lines else "done")[:60]
    return first + (" (+%d lines)" % (len(lines) - 1) if len(lines) > 1 else "")


class BlockHeader(Static):
    """The clickable line `▸ Running command in terminal`."""

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.parent.toggle()


class ToolBlock(Vertical):
    """One tool call: a header you can click, and under it the command/code/output on a lighter background."""

    def __init__(self, call, open_: bool = True, unicode: bool = True):
        super().__init__(classes="toolblock")
        self.call, self.unicode = call, unicode
        self.open_, self.result = open_, None
        self.head, self.title, _ = blocks.tool_parts(call, None, True)
        self.header = BlockHeader(self._header_text(), classes="tb-head")
        self.body = Static("", classes="tb-body", markup=False)

    def compose(self) -> ComposeResult:
        yield self.header
        yield self.body

    def on_mount(self) -> None:
        self._refresh()

    def _header_text(self) -> Text:
        t = Text(no_wrap=True, overflow="ellipsis")
        t.append(("▾ " if self.open_ else "▸ ") if self.unicode else ("v " if self.open_ else "> "), style=brand.AMBER)
        t.append(self.head, style="bold")
        if not self.open_ and self.result is not None:
            t.append("  · " + summary_of(*self.result), style="red" if not self.result[1] else brand.MUTED)
        return t

    def set_result(self, text: str, ok: bool) -> None:
        self.result = (text, ok)
        _, title, rows = blocks.tool_parts(self.call, text, ok)
        self.body.update(rows_to_text([[(title, brand.MUTED)]] + rows) if rows else Text(""))
        self._refresh()

    def toggle(self) -> None:
        self.open_ = not self.open_
        self._refresh()

    def set_open(self, value: bool) -> None:
        self.open_ = value
        self._refresh()

    def _refresh(self) -> None:
        self.header.update(self._header_text())
        self.body.display = bool(self.open_ and self.result is not None)


class AnswerText(Static):
    """The model's answer, growing as tokens arrive."""

    def __init__(self):
        super().__init__("", classes="answer", markup=False)
        self.text = Text()

    def add(self, text: str) -> None:
        self.text.append(text)
        self.update(self.text)


class PlanText(Static):
    def __init__(self, label: str):
        super().__init__("", classes="plan", markup=False)
        self.text = Text()
        self.text.append("▸ %s " % label, style="italic " + brand.MUTED)

    def add(self, text: str) -> None:
        self.text.append(text, style="italic " + brand.MUTED)
        self.update(self.text)


class CodeBlock(Static):
    """A fenced code block in an answer: language title, then colored code on the lighter background."""

    def __init__(self, lang: str):
        super().__init__("", classes="codeblock", markup=False)
        self.hl = hl.Highlighter(lang)
        self.rows = [[(lang or "code", brand.MUTED)]]
        self.update(rows_to_text(self.rows))

    def add_line(self, line: str) -> None:
        self.rows.append(self.hl.line(line.replace("\t", "    ")) or [("", "")])
        self.update(rows_to_text(self.rows))


def numbered_rows(name: str, text: str) -> list:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    width = max(3, len(str(len(lines))))
    h = hl.Highlighter(hl.lang_for_path(name))
    return [[(str(i + 1).rjust(width), brand.MUTED), ("  ", "")] + (h.line(ln.replace("\t", "    ")) or [("", "")])
            for i, ln in enumerate(lines)]


class DocClose(Static):
    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.app.close_doc()


class DocPane(Vertical):
    """The file viewer on the right of the chat: the whole document, scrollable with the wheel, keys and scrollbars."""

    def __init__(self):
        super().__init__(id="doc")
        self.title = Static("", id="doc-title", markup=False)
        self.close = DocClose(" ✕ ", id="doc-close")
        self.content = Static("", id="doc-text", markup=False)
        self.scroller = ScrollableContainer(self.content, id="doc-scroll")

    def compose(self) -> ComposeResult:
        yield Horizontal(self.title, self.close, id="doc-head")
        yield self.scroller

    def show(self, name: str, text: str) -> None:
        n = len(text.split("\n")) - (1 if text.endswith("\n") else 0)
        self.title.update(Text(" %s · %d lines" % (name, n), style="bold " + brand.AMBER, no_wrap=True, overflow="ellipsis"))
        self.content.update(rows_to_text(numbered_rows(name, text)))
        self.scroller.scroll_home(animate=False)


class Splitter(Static):
    """A thin handle between the chat and the file pane: drag it left or right to resize."""

    class Dragged(Message):
        def __init__(self, screen_x: int) -> None:
            super().__init__()
            self.screen_x = screen_x

    def __init__(self):
        super().__init__("┃", id="split")
        self.dragging = False

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self.dragging = True
        self.capture_mouse()
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self.dragging:
            self.post_message(self.Dragged(event.screen_x))
            event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        self.dragging = False
        self.release_mouse()
        event.stop()


class PromptInput(Input):
    """The input line. Up/Down walk the history (or the command menu), Tab completes a command."""

    def on_key(self, event: events.Key) -> None:
        if event.key in ("up", "down", "tab"):
            if self.app.prompt_key(event.key):
                event.stop()
                event.prevent_default()


class ConfirmScreen(ModalScreen):
    """Asks before running a command: y = yes, n = no, a = always (stop asking this session)."""

    BINDINGS = [("y", "answer('y')", "yes"), ("n", "answer('n')", "no"), ("a", "answer('a')", "always"),
                ("escape", "answer('n')", "no")]

    def __init__(self, call):
        super().__init__()
        self.call = call

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm"):
            yield Static(Text("Run this command?", style="bold " + brand.AMBER))
            yield Static(runs_to_text([("$ ", brand.MUTED)] + hl.Highlighter("bash").line(self.call.args), wrap=True), id="confirm-cmd")
            yield Static(Text("[y]es   [n]o   [a]lways (do not ask again)", style=brand.MUTED))

    def action_answer(self, key: str) -> None:
        self.dismiss(key)


class FolderScreen(ModalScreen):
    """Pick the work folder: a recent one, the suggestion, or type a path. Dismisses with the path or None."""

    BINDINGS = [("escape", "cancel", "cancel")]

    def __init__(self, recent: list, default: str):
        super().__init__()
        self.options = [(p, p) for p in recent]
        if default and default not in recent:
            self.options.append((default, default + "  (new folder)"))

    def compose(self) -> ComposeResult:
        with Vertical(id="folder"):
            yield Static(Text("Where should I work? I can only read and write inside this folder.", style="bold " + brand.AMBER))
            yield OptionList(*[Option(label, id=str(i)) for i, (_, label) in enumerate(self.options)], id="folder-list")
            yield Input(placeholder="or type a path and press Enter (Esc cancels)", id="folder-input")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self.options[int(event.option.id)][0])

    def on_mount(self) -> None:
        self.query_one("#folder-input").focus()

    def on_key(self, event: events.Key) -> None:
        if event.key in ("up", "down") and isinstance(self.focused, Input):
            self.query_one("#folder-list").focus()
            event.stop()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip().strip("\"'")
        if value:
            self.dismiss(value)
        elif self.options:
            self.dismiss(self.options[0][0])

    def action_cancel(self) -> None:
        self.dismiss(None)
