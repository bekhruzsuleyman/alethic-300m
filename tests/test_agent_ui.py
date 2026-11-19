import io

from alethic.agent import vibe
from alethic.agent.lineedit import LineEditor
from alethic.agent.protocol import Call
from alethic.agent.ui import UI, glyphs_for, status_text


def make(**kw):
    out = io.StringIO()
    args = dict(out=out, color=False, unicode=False, isatty=False, fancy=False)
    args.update(kw)
    return UI(**args), out


class FakeKeys:
    """Stands in for the keyboard: hands out the given keys one by one."""

    def __init__(self, *keys):
        self.keys = list(keys)

    def usable(self):
        return True

    def session(self):
        import contextlib
        return contextlib.nullcontext()

    def read(self):
        return self.keys.pop(0)


def typed(text):
    return list(text)


# ---- the old basics, still true ------------------------------------------
def test_banner_names_the_model_the_folder_and_the_effort():
    ui, out = make()
    ui.banner("Alethic SFT-2", "work", auto=False, effort="high")
    text = out.getvalue()
    assert "Alethic SFT-2" in text and "work" in text and "high" in text and "/" in text


def test_call_prints_the_action_header_and_the_result_fills_the_box():
    ui, out = make(unicode=True, fancy=True)
    c = Call("shell", "python3 a.py", None)
    ui.call(c)
    assert "▸ Running command in terminal" in out.getvalue() and "terminal  " not in out.getvalue()
    ui.result("hello", ok=True)
    text = out.getvalue()
    assert "$ python3 a.py" in text and "hello" in text and "terminal" in text and not set("╭╮╰╯│├┤") & set(text)


def test_write_shows_the_file_and_its_code_in_a_box():
    ui, out = make(unicode=True, fancy=True)
    ui.call(Call("write", "a.py", "x = 1\ny = 2\n"))
    ui.result("wrote a.py (2 lines)")
    text = out.getvalue()
    assert "▸ Writing a.py" in text and "x = 1" in text and "wrote a.py" in text


def test_long_results_are_shortened():
    ui, out = make(unicode=True)
    ui.call(Call("shell", "ls", None))
    ui.result("\n".join(str(i) for i in range(30)))
    text = out.getvalue()
    lines = [l.strip() for l in text.split("\n")]
    assert "0" in lines and "9" in lines and "29" not in lines and "… +20 more lines" in lines


def test_ascii_terminals_get_an_ascii_box_and_arrow():
    ui, out = make(unicode=False)
    ui.call(Call("shell", "ls", None))
    ui.result("a")
    text = out.getvalue()
    assert text.isascii() and "> Running command in terminal" in text and "terminal" in text


def test_stream_prints_spans_and_plan_is_marked():
    ui, out = make()
    ui.stream([("plan", " one file "), ("text", "Done.")])
    assert "plan" in out.getvalue() and "Done." in out.getvalue()


def test_no_color_codes_when_color_is_off():
    ui, out = make()
    ui.stream([("text", "hi")])
    assert "\x1b[" not in out.getvalue()


def test_color_codes_when_color_is_on():
    ui, out = make(color=True)
    ui.error("boom")
    assert "\x1b[" in out.getvalue() and "boom" in out.getvalue()


def test_clear_sends_the_clear_screen_code_only_on_a_terminal():
    ui, out = make()
    ui.clear()
    assert out.getvalue() == ""
    ui, out = make(isatty=True)
    ui.clear()
    assert "\x1b[2J" in out.getvalue()


# ---- which symbols a terminal gets ---------------------------------------
def test_modern_terminals_get_the_fancy_symbols():
    g = glyphs_for(unicode=True, fancy=True)
    assert g["hook"] == "⎿" and g["spin"] != "-\\|/"


def test_plain_windows_console_gets_safe_symbols():
    g = glyphs_for(unicode=True, fancy=False)
    assert g["hook"] == "└" and g["spin"] == "-\\|/"


def test_ascii_only_output_gets_ascii_symbols():
    g = glyphs_for(unicode=False, fancy=False)
    assert all(ord(c) < 128 for v in g.values() for c in v)


# ---- the live stats line --------------------------------------------------
def test_status_text_has_calls_tokens_seconds_and_rate():
    assert status_text(calls=0, tokens=49, seconds=11.7, rate=4.2) == "0 tool calls · 49 tokens · 11.7s · 4.2 tok/s"
    assert status_text(calls=1, tokens=1, seconds=0.5, rate=2.0).startswith("1 tool call ·")


def live_ui(rows=24):
    return make(isatty=True, live=True, animate=False, size=lambda: (80, rows))


def test_begin_turn_reserves_the_bottom_row_and_end_turn_gives_it_back():
    ui, out = live_ui(24)
    ui.begin_turn()
    assert "\x1b[1;19r" in out.getvalue()
    ui.end_turn()
    assert "\x1b[r" in out.getvalue()
    assert out.getvalue().index("\x1b[1;19r") < out.getvalue().index("\x1b[r")


def test_refresh_writes_the_numbers_on_the_bottom_row_and_restores_the_cursor():
    ui, out = live_ui(24)
    ui.begin_turn()
    ui.tick(49)
    ui.refresh()
    text = out.getvalue()
    assert "\x1b[24;1H" in text and "49 tokens" in text and "tok/s" in text and any(w in text for w in vibe.WORDS)
    assert text.rindex("\x1b7") < text.rindex("\x1b8")


def test_the_status_line_counts_tool_calls_and_changes_phase():
    ui, out = live_ui()
    ui.begin_turn()
    ui.call(Call("shell", "ls", None))
    ui.refresh()
    last = out.getvalue().split("\x1b7")[-1]
    assert "1 tool call " in last and any(w in last for w in vibe.RUNNING)
    ui.result("a.py")
    ui.refresh()
    assert any(w in out.getvalue().split("\x1b7")[-1] for w in vibe.WORDS)


def test_end_turn_prints_a_final_stats_line_and_returns_it():
    ui, out = live_ui()
    ui.begin_turn()
    ui.tick(30)
    line = ui.end_turn()
    assert "30 tokens" in line and "30 tokens" in out.getvalue().split("\x1b[r")[-1]


def test_without_live_support_nothing_is_pinned_but_the_final_line_still_prints():
    ui, out = make(isatty=True, live=False)
    ui.begin_turn()
    ui.tick(5)
    ui.refresh()
    ui.end_turn()
    assert "\x1b[1;" not in out.getvalue() and "5 tokens" in out.getvalue()


def test_end_turn_twice_is_harmless_and_reset_terminal_always_frees_the_region():
    ui, out = live_ui()
    ui.begin_turn()
    ui.end_turn()
    n = out.getvalue().count("\x1b[r")
    ui.end_turn()
    assert out.getvalue().count("\x1b[r") == n
    ui.begin_turn()
    ui.reset_terminal()
    assert out.getvalue().count("\x1b[r") == n + 1


def test_a_resized_window_moves_the_pinned_row():
    sizes = [(80, 24)]
    ui, out = make(isatty=True, live=True, animate=False, size=lambda: sizes[0])
    ui.begin_turn()
    sizes[0] = (80, 30)
    ui.refresh()
    assert "\x1b[1;25r" in out.getvalue() and "\x1b[30;1H" in out.getvalue()


# ---- the input box and the "/" menu --------------------------------------
def test_frame_is_two_rules_around_the_prompt_with_a_hint_underneath():
    ui, _ = make()
    f = ui.frame(LineEditor(), status="medium · asks first", width=40, rows=24)
    assert len(f.lines) == 4
    assert set(f.lines[0]) <= {"-"} and set(f.lines[2]) <= {"-"}
    assert f.lines[1].startswith("> ") and "medium" in f.lines[3] and "/" in f.lines[3]
    assert (f.cursor_row, f.cursor_col) == (1, 2)


def test_frame_shows_the_typed_text_and_puts_the_cursor_after_it():
    ui, _ = make()
    ed = LineEditor()
    for ch in "hi":
        ed.feed(ch)
    f = ui.frame(ed, "", 40, 24)
    assert f.lines[1] == "> hi" and f.cursor_col == 4


def test_frame_lists_matching_commands_and_marks_the_selected_one():
    ui, _ = make()
    ed = LineEditor()
    ed.feed("/"); ed.feed("down")
    f = ui.frame(ed, "", 80, 24)
    menu = f.lines[3:-1]
    assert len(menu) == min(8, len(ed.suggestions())) and menu[1].lstrip().startswith(">") and not menu[0].lstrip().startswith(">")
    assert "/new" in menu[0] and "/clear" in menu[1]
    assert "Tab" in f.lines[-1] and "Enter" in f.lines[-1] and "2/%d" % len(ed.suggestions()) in f.lines[-1]


def test_frame_menu_never_grows_past_the_screen_and_follows_the_selection():
    ui, _ = make()
    ed = LineEditor()
    ed.feed("/")
    for _ in range(12):
        ed.feed("down")
    f = ui.frame(ed, "", 80, rows=14)
    assert len(f.lines) <= 14 - 2
    assert any(line.lstrip().startswith(">") for line in f.lines)


def test_frame_has_no_lines_wider_than_the_terminal():
    ui, _ = make()
    ed = LineEditor()
    ed.feed("/")
    f = ui.frame(ed, "medium · asks first", 30, 24)
    assert all(len(line) <= 29 for line in f.lines)


def test_prompt_with_keys_returns_the_typed_line_and_leaves_a_clean_record():
    ui, out = make(isatty=True, keys=FakeKeys(*typed("hi"), "enter"), size=lambda: (60, 24))
    assert ui.prompt() == "hi"
    text = out.getvalue()
    assert text.endswith("> hi\n") and "\x1b[J" in text


def test_prompt_menu_flow_picks_a_command_with_the_keyboard():
    ui, _ = make(isatty=True, keys=FakeKeys("/", "e", "enter", "down", "enter"), size=lambda: (60, 24))
    assert ui.prompt() == "/effort medium"


def test_prompt_remembers_earlier_lines_for_the_up_arrow():
    ui, _ = make(isatty=True, keys=FakeKeys("a", "enter", "up", "enter"), size=lambda: (60, 24))
    assert ui.prompt() == "a" and ui.prompt() == "a"


def test_prompt_ctrl_c_raises_and_ctrl_d_ends_the_input():
    import pytest
    ui, _ = make(isatty=True, keys=FakeKeys("ctrl-c"), size=lambda: (60, 24))
    with pytest.raises(KeyboardInterrupt):
        ui.prompt()
    ui, _ = make(isatty=True, keys=FakeKeys("ctrl-d"), size=lambda: (60, 24))
    assert ui.prompt() is None


def test_prompt_without_a_keyboard_falls_back_to_plain_input(monkeypatch):
    ui, out = make()
    monkeypatch.setattr("builtins.input", lambda *a: "  hello ")
    assert ui.prompt() == "hello"


# ---- which terminals count as modern -------------------------------------
def test_is_fancy_by_terminal():
    from alethic.agent.ui import is_fancy
    assert is_fancy({}, "posix") and is_fancy({"TERM": "xterm"}, "posix")
    assert not is_fancy({"TERM": "dumb"}, "posix")
    assert is_fancy({"WT_SESSION": "abc"}, "nt") and is_fancy({"TERM_PROGRAM": "vscode"}, "nt")
    assert not is_fancy({}, "nt")


def test_a_console_without_ansi_gets_no_colors_no_live_row_and_plain_input():
    ui, out = make(isatty=True, vt=False, color=None, live=None, fancy=False)
    assert not ui.color and not ui.live and ui.keys is None


def test_banner_lines_always_fit_the_window_even_with_a_long_folder():
    ui, out = make(size=lambda: (40, 24))
    ui.banner("Alethic agent · SFT-2 · 300M", "C:\\Users\\Someone\\Documents\\Projects\\alethic\\work", False, "medium")
    assert all(len(line) <= 39 for line in out.getvalue().split("\n"))
    assert "work" in out.getvalue()          # the end of a path is the useful part, so that is what is kept


# ---- the spinner and the vibe words -------------------------------------------------------------------------------
from alethic.agent import brand  # noqa: E402


def row_text(ui, out):
    return out.getvalue().split("\x1b7")[-1]


def test_the_bottom_row_shows_a_spinner_a_vibe_word_and_the_numbers():
    ui, out = live_ui()
    ui._clock = lambda: 100.0
    ui.begin_turn()
    ui.tick(17)
    ui.refresh()
    text = row_text(ui, out).replace("\u2026", "...")
    assert brand.spinner(100.0, False) in text and ui.vibes.word("generating") + "..." in text and "17 tokens" in text


def test_the_spinner_cycles_with_the_clock_and_the_word_does_not():
    ui, out = live_ui()
    frames, words = set(), set()
    for t in (100.0, 100.13, 100.26, 100.39, 130.0):
        ui._clock = lambda t=t: t
        ui.begin_turn()
        ui.refresh()
        frames.add(brand.spinner(t, False))
        words.add(ui.vibes.word("generating"))
        ui.end_turn()
    assert len(frames) >= 3 and len(words) == 1


def test_the_word_moves_on_with_tokens_and_tool_calls_only():
    ui, out = live_ui()
    ui.begin_turn()
    w0 = ui.vibes.word("generating")
    ui.tick(79)
    assert ui.vibes.word("generating") == w0
    ui.tick(80)
    w1 = ui.vibes.word("generating")
    assert w1 != w0
    ui.call(Call("shell", "ls", None))
    assert ui.vibes.word("generating") != w1
    ui.result("x")
    ui.end_turn()


def test_the_row_never_runs_past_the_window_at_any_width():
    import re
    esc = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b[78]")
    for cols in (24, 30, 40, 60, 80, 120):
        out = io.StringIO()
        ui = UI(out=out, color=True, unicode=True, isatty=True, live=True, animate=False, size=lambda c=cols: (c, 24), keys=None)
        ui.begin_turn()
        ui.tick(12345)
        ui.refresh()
        visible = esc.sub("", row_text(ui, out).split("\x1b[24;1H")[-1].split("\x1b[2K")[-1])
        assert len(visible) <= cols - 1, (cols, len(visible), visible)


def test_truecolor_terminals_get_exact_colors_and_others_get_256():
    ui, out = make(color=True, truecolor=True)
    ui.line(ui.paint("x", brand.AMBER))
    assert "\x1b[38;2;255;176;32m" in out.getvalue()
    ui2, out2 = make(color=True, truecolor=False)
    ui2.line(ui2.paint("x", brand.AMBER))
    assert "\x1b[38;5;" in out2.getvalue() and "38;2;" not in out2.getvalue()


# ---- the logo and the start-up animation ---------------------------------------------------------------------------
def test_the_banner_shows_the_two_discs_beside_the_title_only_when_there_is_room():
    ui, out = make(size=lambda: (80, 24), unicode=True, fancy=True)
    ui.banner("Alethic agent", "work", auto=False, effort="medium")
    text = out.getvalue()
    assert "█" in text and "a█" not in text and "Alethic agent" in text
    ui2, out2 = make(size=lambda: (30, 24), unicode=True, fancy=True)
    ui2.banner("Alethic agent", "work", auto=False, effort="medium")
    assert "█" not in out2.getvalue() and "Alethic agent" in out2.getvalue()


def test_the_start_up_animation_slides_the_discs_together_and_is_quick():
    slept = []
    ui, out = make(size=lambda: (80, 24), unicode=True, fancy=True, isatty=True, animate=True)
    ui._sleep = slept.append
    ui.banner("Alethic agent", "work", auto=False)
    assert len(slept) == len(brand.intro_gaps()) - 1 and 0.2 < sum(slept) < 0.5
    assert out.getvalue().count("\x1b[5A") >= len(slept)      # each frame redraws over the last one


def test_no_animation_when_not_a_terminal_or_reduced_motion(monkeypatch):
    slept = []
    ui, out = make(size=lambda: (80, 24), unicode=True, fancy=True, isatty=False, animate=True)
    ui._sleep = slept.append
    ui.banner("Alethic agent", "work", auto=False)
    assert slept == []
    monkeypatch.setenv("ALETHIC_REDUCED_MOTION", "1")
    ui2, out2 = make(size=lambda: (80, 24), unicode=True, fancy=True, isatty=True, animate=True)
    ui2._sleep = slept.append
    ui2.banner("Alethic agent", "work", auto=False)
    assert slept == [] and "█" in out2.getvalue()


# ---- code in answers ---------------------------------------------------------------------------------------------
def test_code_fences_in_an_answer_become_a_titled_box_with_colors():
    ui, out = make(color=True, truecolor=True, unicode=True, size=lambda: (80, 24))
    ui.stream([("text", "Try:\n```python\nx = 1\nprint(x)\n```\nDone.")])
    ui.end_stream()
    text = out.getvalue()
    plain = __import__("re").sub(r"\x1b\[[0-9;]*m", "", text)
    assert "  python" in plain and "  x = 1" in plain and "Try:" in plain and "Done." in plain and "╭" not in plain
    assert "```" not in plain and "38;2;" in text


def test_an_answer_cut_off_inside_a_fence_still_closes_its_box():
    ui, out = make(unicode=True, size=lambda: (80, 24))
    ui.stream([("text", "```rust\nlet x = 1;")])
    ui.end_stream()
    assert "rust" in out.getvalue() and "let x = 1;" in out.getvalue() and "```" not in out.getvalue()


def test_plan_and_map_lines_start_with_the_arrow():
    ui, out = make(unicode=True, fancy=True)
    ui.stream([("plan", "read the file first"), ("text", "Done.")])
    text = out.getvalue()
    assert "▸ plan read the file first\nDone." in text


# ---- the transcript the viewer reads ---------------------------------------------------------------------------------
def test_the_transcript_keeps_plain_text_and_the_last_file_touched():
    ui, out = make(color=True, unicode=True)
    ui.call(Call("write", "src/a.py", "x = 1\n"))
    ui.result("wrote")
    assert ui.last_file == "src/a.py"
    assert "\x1b" not in ui.transcript() and "x = 1" in ui.transcript()
    ui.call(Call("read", "b.py, 1-10", None))
    assert ui.last_file == "b.py"


def test_the_transcript_is_capped():
    ui, out = make()
    for _ in range(3000):
        ui.line("y" * 100)
    assert len(ui.transcript()) <= 210_000


def test_view_file_prints_numbered_text_when_there_is_no_real_terminal():
    ui, out = make()
    ui.view_file("a.py", "a = 1\nb = 2\n")
    assert "1  a = 1" in out.getvalue() and "2  b = 2" in out.getvalue()


def test_view_file_opens_the_viewer_on_a_terminal_and_q_closes_it():
    ui, out = make(isatty=True, unicode=True, fancy=True, size=lambda: (100, 12), keys=FakeKeys("down", "q"))
    ui.line("hello from the chat")
    ui.view_file("a.py", "x = 1\n" * 30)
    text = out.getvalue()
    assert "\x1b[?1049h" in text and "\x1b[?1049l" in text and "hello from the chat" in text.split("\x1b[?1049h")[1]


def test_plan_line_has_one_space_after_its_label():
    ui, out = make(unicode=True, fancy=True)
    ui.stream([("plan", " write the file")])
    assert "▸ plan write the file" in out.getvalue()


def test_blocks_sit_on_a_lighter_background_and_plain_mode_has_none():
    ui, out = make(color=True, truecolor=True, unicode=True, fancy=True)
    ui.call(Call("shell", "ls", None))
    ui.result("a")
    assert "48;2;28;27;23" in out.getvalue()
    ui2, out2 = make(unicode=True, fancy=True)
    ui2.call(Call("shell", "ls", None))
    ui2.result("a")
    assert "\x1b[" not in out2.getvalue()


def test_details_off_hides_the_blocks_and_keeps_one_line_per_call():
    ui, out = make(unicode=True, fancy=True)
    ui.details = False
    ui.call(Call("shell", "ls", None))
    ui.result("one\ntwo\nthree")
    text = out.getvalue()
    assert "▸ Running command in terminal" in text and "one (+2 lines hidden)" in text and "two" not in text.replace("two lines", "")
    assert "click the header" in text


def test_the_input_box_stays_visible_but_locked_during_an_answer():
    ui, out = live_ui(24)
    ui.begin_turn()
    ui.refresh()
    text = out.getvalue()
    assert "locked" in text and ui.arrow in text and "\x1b[20;1H" in text and "\x1b[24;1H" in text


def test_a_tip_shows_above_the_status_row_and_changes_with_work_only():
    from alethic.agent import tips
    ui, out = live_ui(24)
    ui._clock = lambda: 100.0
    ui.begin_turn()
    ui.refresh()
    first = tips.tip_at(0, ui._seed)
    assert ("tip: " + first)[:70] in out.getvalue()   # the row is cut to the screen width
    ui._clock = lambda: 500.0               # much later, no work done: same tip
    out.truncate(0)
    out.seek(0)
    ui.refresh()
    assert "tip: " + first in out.getvalue()
    for _ in range(tips.EVERY):
        ui.call(Call("shell", "ls", None))
        ui.result("x")
    out.truncate(0)
    out.seek(0)
    ui.refresh()
    assert "tip: " + first not in out.getvalue()


# ---- click to hide or show a tool block -------------------------------------------------------------------------
def plain_of(s):
    import re
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[78]", "", s)


def with_block(ui, out, result="line1\nline2\nline3"):
    ui.line("hello")
    ui.call(Call("shell", "ls", None))
    ui.result(result)
    out.truncate(0)
    out.seek(0)


def test_toggling_a_block_redraws_the_screen_with_that_block_collapsed_and_back():
    ui, out = make(isatty=True, unicode=True, fancy=True, size=lambda: (80, 24))
    with_block(ui, out)
    (b,) = ui.blocks
    assert b.open
    ui.toggle(b)
    text = out.getvalue()
    assert "\x1b[2J" in text and "▸ Running command in terminal" in text and "line1 (+2 lines hidden)" in text and "hello" in text
    assert "$ ls" not in plain_of(text)
    out.truncate(0)
    out.seek(0)
    ui.toggle(b)
    assert "$ ls" in plain_of(out.getvalue()) and "hidden" not in out.getvalue()


def test_set_details_changes_every_block_and_the_default_for_new_ones():
    ui, out = make(isatty=True, unicode=True, fancy=True, size=lambda: (80, 24))
    with_block(ui, out)
    ui.set_details(False)
    assert [b.open for b in ui.blocks] == [False] and ui.details is False
    ui.call(Call("shell", "pwd", None))
    ui.result("x")
    assert [b.open for b in ui.blocks] == [False, False]
    ui.set_details(True)
    assert [b.open for b in ui.blocks] == [True, True]


def test_replay_keeps_everything_that_was_printed_in_order():
    ui, out = make(isatty=True, unicode=True, fancy=True, size=lambda: (80, 24))
    ui.line("first")
    ui.call(Call("shell", "ls", None))
    ui.result("a")
    ui.line("last")
    out.truncate(0)
    out.seek(0)
    ui.repaint()
    text = plain_of(out.getvalue())
    assert text.index("first") < text.index("Running command") < text.index("$ ls") < text.index("last")


def click_row_of_header(ui, cols, rows, frame_h):
    from alethic.agent import screenmap as sm
    lines = ui.screen_lines(cols)
    scrolled = max(0, len(lines) + frame_h - rows)
    return next(i + 1 - scrolled for i, (_, m) in enumerate(lines) if m is not None)


def test_a_click_on_a_header_at_the_prompt_toggles_it_and_clicks_elsewhere_do_nothing():
    keys = FakeKeys("click:5:1", "click:5:999", "x", "enter")
    ui, out = make(isatty=True, unicode=True, fancy=True, size=lambda: (80, 24), keys=keys)
    ui.mouse = True
    ui.line("hello")
    ui.call(Call("shell", "ls", None))
    ui.result("line1\nline2")
    row = click_row_of_header(ui, 80, 24, 4)
    keys.keys[0] = "click:5:%d" % row
    out.truncate(0)
    out.seek(0)
    assert ui.prompt() == "x"
    text = out.getvalue()
    assert "\x1b[?1000h\x1b[?1006h" in text and "\x1b[?1000l\x1b[?1006l" in text
    assert "line1 (+1 lines hidden)" in text and ui.blocks[0].open is False


def test_the_prompt_asks_for_mouse_reports_only_when_there_is_something_to_click_and_it_is_on():
    ui, out = make(isatty=True, unicode=True, fancy=True, size=lambda: (80, 24), keys=FakeKeys("enter"))
    ui.prompt()
    assert "\x1b[?1000h" not in out.getvalue()
    ui2, out2 = make(isatty=True, unicode=True, fancy=True, size=lambda: (80, 24), keys=FakeKeys("enter"))
    ui2.call(Call("shell", "ls", None))
    ui2.result("a")
    ui2.mouse = False
    out2.truncate(0)
    out2.seek(0)
    ui2.prompt()
    assert "\x1b[?1000h" not in out2.getvalue()


def test_a_menu_open_while_clicking_does_not_toggle_by_mistake():
    keys = FakeKeys("/", "click:5:1", "esc", "enter")
    ui, out = make(isatty=True, unicode=True, fancy=True, size=lambda: (80, 24), keys=keys)
    ui.call(Call("shell", "ls", None))
    ui.result("a\nb")
    row = click_row_of_header(ui, 80, 24, 4)
    keys.keys[1] = "click:5:%d" % row
    ui.prompt()
    assert ui.blocks[0].open is True


def test_mouse_reporting_is_off_by_default_so_the_wheel_scrolls_the_terminal():
    keys = FakeKeys("x", "enter")
    ui, out = make(isatty=True, unicode=True, fancy=True, size=lambda: (80, 24), keys=keys)
    ui.line("hello")
    ui.call(Call("shell", "ls", None))
    ui.result("line1")
    out.truncate(0)
    out.seek(0)
    assert ui.prompt() == "x"
    assert "\x1b[?1000h" not in out.getvalue() and "\x1b[2J" not in out.getvalue()


def test_the_prompt_always_turns_the_text_cursor_on():
    keys = FakeKeys("x", "enter")
    ui, out = make(isatty=True, unicode=True, fancy=True, size=lambda: (80, 24), keys=keys)
    assert ui.prompt() == "x"
    assert "\x1b[?25h" in out.getvalue()
