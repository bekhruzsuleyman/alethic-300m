import io

from alethic.agent.cli import Session, repl
from alethic.agent.ui import UI


class FakeKeys:
    def __init__(self, *keys):
        self.keys = list(keys)

    def usable(self):
        return True

    def session(self):
        import contextlib
        return contextlib.nullcontext()

    def read(self):
        return self.keys.pop(0)


class Script:
    def __init__(self, *texts):
        self.texts = list(texts)
        self.temperature, self.last_tokens = 0.0, 3

    def generate(self, history, stop, max_new, on_text):
        t = self.texts.pop(0)
        on_text(t)
        return t


def run(tmp_path, monkeypatch, inputs, *texts, auto=True):
    out = io.StringIO()
    ui = UI(out=out, color=False, unicode=False, isatty=False)
    eng = Script(*texts)
    s = Session(eng.generate, eng, str(tmp_path), ui, count_tokens=len, auto=auto, mode="posix")
    it = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda *a: next(it))
    repl(s, ui, title="test")
    return out.getvalue()


def test_whole_session_writes_a_real_file_runs_it_and_shows_it(tmp_path, monkeypatch):
    out = run(tmp_path, monkeypatch, ["make a.py", "/quit"],
              "<plan> one file </plan>\n<call> write(a.py) </call>\nprint(2+3)\n<call>",
              "<call> shell(python3 a.py) </call>",
              "It printed 5.\n</assistant>")
    assert (tmp_path / "a.py").read_text() == "print(2+3)\n"
    assert "Writing a.py" in out and "$ python3 a.py" in out and "5" in out and "It printed 5." in out
    assert "<call>" not in out and "print(2+3)" in out  # the file text shows only in the write preview


def test_slash_command_does_not_reach_the_model(tmp_path, monkeypatch):
    out = run(tmp_path, monkeypatch, ["/files", "/quit"])
    assert "empty" in out


def test_declining_a_command_tells_the_model(tmp_path, monkeypatch):
    (tmp_path / "d.txt").write_text("keep\n")
    out = run(tmp_path, monkeypatch, ["go", "n", "/quit"], "<call> shell(rm d.txt) </call>", "ok\n</assistant>", auto=False)
    assert (tmp_path / "d.txt").exists() and "declined" in out


def test_always_answer_turns_auto_mode_on(tmp_path, monkeypatch):
    (tmp_path / "d.txt").write_text("x\n")
    run(tmp_path, monkeypatch, ["go", "a", "/quit"], "<call> shell(rm d.txt) </call>", "ok\n</assistant>", auto=False)
    assert not (tmp_path / "d.txt").exists()


def test_end_of_input_leaves_cleanly(tmp_path, monkeypatch):
    out = io.StringIO()
    ui = UI(out=out, color=False, unicode=False, isatty=False)
    eng = Script()
    s = Session(eng.generate, eng, str(tmp_path), ui, count_tokens=len, mode="posix")

    def eof(*a):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    repl(s, ui, title="test")


def test_retry_runs_the_last_message_again(tmp_path, monkeypatch):
    out = run(tmp_path, monkeypatch, ["first", "/retry", "/quit"], "one\n</assistant>", "two\n</assistant>")
    assert "one" in out and "two" in out


def test_effort_flag_sets_the_limits(tmp_path):
    ui = UI(out=io.StringIO(), color=False, unicode=False, isatty=False)
    eng = Script()
    s = Session(eng.generate, eng, str(tmp_path), ui, count_tokens=len, mode="posix", effort="low")
    assert (s.agent.max_steps, s.agent.max_new, s.effort) == (3, 160, "low")


def test_every_answer_ends_with_the_stats_line(tmp_path, monkeypatch):
    out = run(tmp_path, monkeypatch, ["go", "/quit"], "<call> shell(ls) </call>", "done\n</assistant>")
    assert "1 tool call" in out and "tok/s" in out


def test_the_input_box_shows_effort_and_permissions(tmp_path):
    out = io.StringIO()
    ui = UI(out=out, color=False, unicode=False, isatty=True, live=False, size=lambda: (80, 24), keys=FakeKeys("ctrl-d"))
    eng = Script()
    s = Session(eng.generate, eng, str(tmp_path), ui, count_tokens=len, mode="posix", effort="high")
    repl(s, ui, title="test")
    assert "high · asks first" in out.getvalue()


def test_an_interrupted_turn_frees_the_bottom_row(tmp_path):
    out = io.StringIO()
    ui = UI(out=out, color=False, unicode=False, isatty=True, live=True, animate=False, size=lambda: (80, 24))

    def boom(history, stop, max_new, on_text):
        raise KeyboardInterrupt
    eng = Script()
    s = Session(boom, eng, str(tmp_path), ui, count_tokens=len, mode="posix")
    from alethic.agent.cli import _turn
    _turn(s, ui, "hi")
    text = out.getvalue()
    assert "\x1b[r" in text and "interrupted" in text and not ui._pinned


def test_default_temperature_is_not_greedy():
    from alethic.agent.cli import DEFAULT_TEMP, build_parser
    a = build_parser().parse_args(["--weights", "w", "--tokenizer", "t"])
    assert a.temp == DEFAULT_TEMP == 0.3


def test_switching_the_folder_starts_a_new_conversation_in_the_new_place_and_keeps_the_permissions(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    out = io.StringIO()
    ui = UI(out=out, color=False, unicode=False, isatty=False)
    s = Session(Script("hi\n</assistant>").generate, Script(), str(a), ui, count_tokens=len, auto=True, mode="posix")
    s.agent.ask("hello")
    assert s.agent.history
    monkeypatch.setattr("alethic.agent.folderpick.pick", lambda *a_, **k: str(b))
    msg = s.choose_folder("")
    assert s.ws.root == str(b) and s.ws.auto and s.agent.history == "" and str(b) in msg


def test_cancelling_the_folder_picker_changes_nothing(tmp_path, monkeypatch):
    out = io.StringIO()
    ui = UI(out=out, color=False, unicode=False, isatty=False)
    s = Session(Script().generate, Script(), str(tmp_path), ui, count_tokens=len, mode="posix")
    monkeypatch.setattr("alethic.agent.folderpick.pick", lambda *a_, **k: None)
    assert "unchanged" in s.choose_folder("") and s.ws.root == str(tmp_path)


def test_no_folder_given_asks_and_non_interactive_runs_fall_back_to_the_suggested_one(tmp_path, monkeypatch):
    from alethic.agent.cli import build_parser, resolve_folder
    a = build_parser().parse_args(["--weights", "w", "--tokenizer", "t"])
    assert a.folder is None
    said = []
    assert resolve_folder(a, said.append, interactive=False, suggest=str(tmp_path / "w")) == str(tmp_path / "w")
    monkeypatch.setattr("alethic.agent.folderpick.pick", lambda *a_, **k: str(tmp_path))
    assert resolve_folder(a, said.append, interactive=True, suggest="x") == str(tmp_path)
    a2 = build_parser().parse_args(["--weights", "w", "--tokenizer", "t", "--folder", "mine"])
    assert resolve_folder(a2, said.append, interactive=True, suggest="x") == "mine"
    monkeypatch.setattr("alethic.agent.folderpick.pick", lambda *a_, **k: None)
    assert resolve_folder(a, said.append, interactive=True, suggest="x") is None


def test_the_classic_terminal_is_the_default_and_textual_is_opt_in():
    from alethic.agent.cli import build_parser, wants_textual
    base = ["--weights", "w", "--tokenizer", "t"]
    assert wants_textual(build_parser().parse_args(base), tty=True) is False
    assert wants_textual(build_parser().parse_args(base + ["--textual"]), tty=True) is True
    assert wants_textual(build_parser().parse_args(base + ["--textual"]), tty=False) is False
