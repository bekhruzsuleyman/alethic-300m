from types import SimpleNamespace

from alethic.agent.commands import run_command
from alethic.agent.loop import Agent
from alethic.agent.tools import Workspace


def ctx(tmp_path):
    ws = Workspace(str(tmp_path), mode="posix")
    agent = Agent(lambda h, s, m, t: "ok\n</assistant>", ws)
    return SimpleNamespace(agent=agent, ws=ws, engine=SimpleNamespace(temperature=0.0), count_tokens=lambda t: len(t),
                           max_seq_len=1000, title="Alethic test", effort="medium")


def test_non_commands_are_not_handled(tmp_path):
    assert run_command("write a function", ctx(tmp_path)) is None


def test_help_lists_commands(tmp_path):
    r = run_command("/help", ctx(tmp_path))
    assert "/new" in r.message and "/auto" in r.message and "/quit" in r.message and not r.quit


def test_new_clears_the_conversation(tmp_path):
    c = ctx(tmp_path)
    c.agent.ask("hi")
    run_command("/new", c)
    assert c.agent.history == ""


def test_auto_toggles(tmp_path):
    c = ctx(tmp_path)
    assert "on" in run_command("/auto", c).message and c.ws.auto
    assert "off" in run_command("/auto", c).message and not c.ws.auto


def test_files_and_cat(tmp_path):
    c = ctx(tmp_path)
    (tmp_path / "a.py").write_text("print(1)\n")
    assert "a.py" in run_command("/files", c).message
    assert "print(1)" in run_command("/cat a.py", c).message
    assert "error" in run_command("/cat ../x", c).message


def test_temp_sets_the_sampling_temperature(tmp_path):
    c = ctx(tmp_path)
    run_command("/temp 0.7", c)
    assert c.engine.temperature == 0.7
    assert "usage" in run_command("/temp abc", c).message


def test_save_writes_a_transcript(tmp_path):
    c = ctx(tmp_path)
    c.agent.ask("hi")
    run_command("/save t.txt", c)
    assert "<user>" in (tmp_path / "t.txt").read_text()


def test_quit_and_unknown(tmp_path):
    assert run_command("/quit", ctx(tmp_path)).quit
    assert "unknown" in run_command("/zzz", ctx(tmp_path)).message


def test_clear_resets_and_asks_for_a_cleared_screen(tmp_path):
    c = ctx(tmp_path)
    c.agent.ask("hi")
    r = run_command("/clear", c)
    assert c.agent.history == "" and r.clear


def test_effort_shows_and_sets_levels(tmp_path):
    c = ctx(tmp_path)
    assert "medium" in run_command("/effort", c).message
    assert "low" in run_command("/effort low", c).message
    assert c.effort == "low" and (c.agent.max_steps, c.agent.max_new) == (3, 160)
    run_command("/effort high", c)
    assert (c.agent.max_steps, c.agent.max_new) == (20, 800)
    assert "usage" in run_command("/effort turbo", c).message


def test_status_shows_the_main_settings(tmp_path):
    c = ctx(tmp_path)
    c.agent.ask("hi")
    m = run_command("/status", c).message
    for word in ("Alethic test", "medium", "ask", "0.0", "tokens", "1 turn"):
        assert word in m, word


def test_context_shows_a_bar_and_a_breakdown(tmp_path):
    c = ctx(tmp_path)
    c.agent.history = "<user>\nhello\n</user>\n<assistant>\n<call> shell(ls) </call>\n<result>\na.py\n</result>\ndone\n</assistant>\n"
    m = run_command("/context", c).message
    assert "/ 1000" in m and "%" in m and "you" in m and "tool results" in m and "model" in m


def test_compact_drops_old_turns(tmp_path):
    c = ctx(tmp_path)
    for t in ("a", "b", "c"):
        c.agent.ask(t)
    r = run_command("/compact", c)
    assert "dropped 1" in r.message and "<user>\na" not in c.agent.history and "<user>\nc" in c.agent.history


def test_retry_asks_the_cli_to_rerun_the_last_message(tmp_path):
    c = ctx(tmp_path)
    c.agent.ask("write a.py")
    r = run_command("/retry", c)
    assert r.rerun == "write a.py" and c.agent.history == ""
    assert "nothing" in run_command("/retry", ctx(tmp_path)).message


def test_tools_lists_the_tools(tmp_path):
    m = run_command("/tools", ctx(tmp_path)).message
    for t in ("shell", "read", "write", "edit", "search"):
        assert t in m


def viewer_ctx(tmp_path):
    c = ctx(tmp_path)
    shown = []
    c.ui = SimpleNamespace(last_file="", view_file=lambda name, text: shown.append((name, text)))
    return c, shown


def test_open_shows_the_whole_file_in_the_viewer(tmp_path):
    c, shown = viewer_ctx(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n" * 300)
    r = run_command("/open a.py", c)
    assert shown == [("a.py", "x = 1\n" * 300)] and r.message == ""


def test_open_and_expand_default_to_the_last_file_touched(tmp_path):
    c, shown = viewer_ctx(tmp_path)
    (tmp_path / "b.rs").write_text("fn main() {}\n")
    c.ui.last_file = "b.rs"
    run_command("/expand", c)
    run_command("/open", c)
    assert [n for n, _ in shown] == ["b.rs", "b.rs"]


def test_open_without_a_file_or_with_a_bad_path_explains(tmp_path):
    c, shown = viewer_ctx(tmp_path)
    assert "no file yet" in run_command("/open", c).message
    assert "error" in run_command("/open ../secret.txt", c).message
    assert "error" in run_command("/open missing.py", c).message
    assert shown == []


def test_folder_switches_through_the_session_and_says_where(tmp_path):
    c = ctx(tmp_path)
    c.choose_folder = lambda arg: "work folder: %s (new conversation)" % (arg or "picked")
    assert run_command("/folder", c).message == "work folder: picked (new conversation)"
    assert "x" in run_command("/folder x", c).message


def test_folder_where_it_cannot_be_changed_explains(tmp_path):
    assert "not available" in run_command("/folder", ctx(tmp_path)).message


def test_details_toggles_and_accepts_on_and_off(tmp_path):
    c, _ = viewer_ctx(tmp_path)
    c.ui.details = True
    assert "hidden" in run_command("/details", c).message and c.ui.details is False
    assert "shown" in run_command("/details on", c).message and c.ui.details is True
    run_command("/details off", c)
    assert c.ui.details is False


def test_mouse_toggles_and_explains_text_selection(tmp_path):
    c, _ = viewer_ctx(tmp_path)
    c.ui.mouse = True
    assert "off" in run_command("/mouse", c).message and c.ui.mouse is False
    assert "select text" in run_command("/mouse off", c).message
    assert "on" in run_command("/mouse on", c).message and c.ui.mouse is True
