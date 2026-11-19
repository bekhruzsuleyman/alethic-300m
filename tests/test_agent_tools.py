import os

import pytest

from alethic.agent.protocol import Call
from alethic.agent.tools import Workspace, python_command, rewrite_python


def ws(tmp_path, **kw):
    kw.setdefault("mode", "posix")
    return Workspace(str(tmp_path), **kw)


def run(w, kind, args, body=None):
    return w.execute(Call(kind, args, body)).result


def test_write_creates_nested_file_and_gives_no_result(tmp_path):
    w = ws(tmp_path)
    assert run(w, "write", "src/a.py", "print(1)\n") is None
    assert (tmp_path / "src" / "a.py").read_text() == "print(1)\n"


@pytest.mark.parametrize("bad", ["../x.py", "/etc/x.py", "a/../../x.py"])
def test_write_outside_the_workspace_is_refused(tmp_path, bad):
    out = run(ws(tmp_path), "write", bad, "x\n")
    assert out.startswith("error:") and not (tmp_path.parent / "x.py").exists()


def test_work_prefix_maps_to_the_workspace(tmp_path):
    w = ws(tmp_path)
    run(w, "write", "/work/a.py", "x\n")
    assert (tmp_path / "a.py").exists()


def test_read_numbers_lines_like_training(tmp_path):
    w = ws(tmp_path)
    run(w, "write", "a.py", "l1\nl2\nl3\n")
    assert run(w, "read", "a.py, 1-2") == "1   l1\n2   l2"
    assert run(w, "read", "a.py, 2-99") == "2   l2\n3   l3"


def test_read_errors_become_result_text(tmp_path):
    w = ws(tmp_path)
    assert run(w, "read", "nope.py, 1-3").startswith("error:")
    run(w, "write", "a.py", "x\n")
    assert run(w, "read", "a.py, 5-9").startswith("error:")
    assert run(w, "read", "a.py").startswith("error:")


def test_edit_replaces_the_named_lines(tmp_path):
    w = ws(tmp_path)
    run(w, "write", "a.py", "a\nb\nc\nd\n")
    assert run(w, "edit", "a.py, 2-2", "B\n") is None
    assert (tmp_path / "a.py").read_text() == "a\nB\nc\nd\n"
    run(w, "edit", "a.py, 3-4", "X\nY\nZ\n")
    assert (tmp_path / "a.py").read_text() == "a\nB\nX\nY\nZ\n"


def test_edit_out_of_range_is_an_error_and_changes_nothing(tmp_path):
    w = ws(tmp_path)
    run(w, "write", "a.py", "a\nb\n")
    assert run(w, "edit", "a.py, 5-5", "x\n").startswith("error:")
    assert (tmp_path / "a.py").read_text() == "a\nb\n"


def test_shell_runs_and_hides_the_real_path(tmp_path):
    w = ws(tmp_path)
    assert run(w, "shell", "echo hi") == "hi"
    assert run(w, "shell", "pwd") == "/work"


def test_python3_works_even_without_a_python3_on_the_path(tmp_path):
    assert run(ws(tmp_path), "shell", 'python3 -c "print(2+3)"') == "5"


def test_blocked_commands_return_a_reason(tmp_path):
    for cmd in ("curl http://x", "sudo ls", "ls /", "cat ../x", "rm -rf d"):
        assert run(ws(tmp_path), "shell", cmd).startswith("blocked:"), cmd


def test_timeout_and_output_cap(tmp_path):
    assert run(ws(tmp_path, timeout=1), "shell", "sleep 3") == "(timed out after 1s)"
    out = run(ws(tmp_path, cap=50), "shell", 'python3 -c "print(\'x\'*500)"')
    assert len(out) < 80 and out.endswith("[truncated]")


def test_search_with_nothing_found_says_no_results(tmp_path):
    w = Workspace(str(tmp_path), search_fetch=lambda url: '{"batchcomplete": ""}')
    assert run(w, "search", "anything") == "no results"


def test_unknown_tool(tmp_path):
    assert run(ws(tmp_path), "fly", "x") == "error: unknown tool fly"


def test_unsafe_shell_commands_need_approval_safe_ones_do_not(tmp_path):
    asked = []
    w = ws(tmp_path, approve=lambda call: asked.append(call.args) or False)
    (tmp_path / "d.txt").write_text("keep\n")
    assert run(w, "shell", "rm d.txt") == "(declined by the user)"
    assert (tmp_path / "d.txt").exists()
    assert run(w, "shell", "echo ok") == "ok" and asked == ["rm d.txt"]


def test_auto_mode_never_asks(tmp_path):
    w = ws(tmp_path, approve=lambda call: False)
    w.auto = True
    (tmp_path / "d.txt").write_text("x\n")
    run(w, "shell", "rm d.txt")
    assert not (tmp_path / "d.txt").exists()


def test_windows_mode_emulates_basic_commands(tmp_path):
    w = ws(tmp_path, mode="windows")
    run(w, "write", "b.txt", "B\n")
    run(w, "write", "a.txt", "A\n")
    assert run(w, "shell", "ls") == "a.txt\nb.txt"
    assert run(w, "shell", "cat a.txt") == "A"
    assert run(w, "shell", "pwd") == "/work"
    w.auto = True
    run(w, "shell", "mkdir d")
    assert os.path.isdir(tmp_path / "d")


def test_rewrite_python_only_at_command_start():
    assert rewrite_python("python3 a.py && python b.py", "PY") == "PY a.py && PY b.py"
    assert rewrite_python("echo python3", "PY") == "echo python3"
    assert rewrite_python("py -3 a.py", "PY") == "PY a.py" and rewrite_python("py a.py", "PY") == "PY a.py"


def test_python_command_prefers_this_python_when_not_frozen():
    assert python_command(False, "/usr/bin/python3", lambda n: None) == '"/usr/bin/python3"'


def test_frozen_app_looks_for_py_then_python_then_python3():
    found = {"py": "C:/Windows/py.exe", "python": "C:/P/python.exe"}
    assert python_command(True, "app.exe", found.get) == "py -3"
    del found["py"]
    assert python_command(True, "app.exe", found.get) == '"C:/P/python.exe"'


def test_the_windows_store_stub_is_not_python():
    found = {"python": "C:/Users/x/AppData/Local/Microsoft/WindowsApps/python.exe", "python3": "C:/P/python3.exe"}
    assert python_command(True, "app.exe", found.get) == '"C:/P/python3.exe"'


def test_no_python_at_all_gives_none_and_the_model_hears_about_it(tmp_path):
    assert python_command(True, "app.exe", lambda n: None) is None
    out = run(ws(tmp_path, python=None), "shell", 'python3 -c "print(1)"')
    assert out.startswith("error: no Python") and "python.org" in out
    assert run(ws(tmp_path, python=None), "shell", "echo hi") == "hi"


def test_write_ignores_a_line_range_the_model_adds_after_the_path(tmp_path):
    w = ws(tmp_path)
    run(w, "write", "main.py, 1-16", "x = 1\n")
    assert (tmp_path / "main.py").read_text() == "x = 1\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["main.py"]


def test_edit_past_the_end_explains_what_to_do_instead(tmp_path):
    w = ws(tmp_path)
    (tmp_path / "a.py").write_text("1\n2\n3\n")
    msg = run(w, "edit", "a.py, 121-125", "x\n")
    assert "3 lines" in msg and "write(" in msg and "1-3" in msg


def test_error_results_show_the_work_folder_as_slash_work_not_the_real_path(tmp_path):
    ws = Workspace(str(tmp_path), mode="posix", python="python3")
    out = ws.execute(Call("read", "nope.py, 1-5"))
    assert not out.ok and str(tmp_path) not in out.result and "/work/nope.py" in out.result


# --- search(content, source) ---------------------------------------------------------------------------------------
import json as _json

from alethic.agent import websearch
from alethic.agent.protocol import Call as _Call


def _wiki(*rows):
    return _json.dumps({"query": {"pages": {str(i): {"title": t, "index": i, "extract": x} for i, (t, x) in enumerate(rows, 1)}}})


def test_search_shows_numbered_results_from_the_source(tmp_path):
    seen = []
    ws = Workspace(str(tmp_path), search_fetch=lambda url: seen.append(url) or _wiki(("Paris", "Paris is the capital of France."), ("Lyon", "Lyon is a city.")))
    out = ws.execute(_Call("search", "capital of France, wikipedia"))
    assert out.result == "1. Paris (en.wikipedia.org/wiki/Paris)\n   Paris is the capital of France.\n2. Lyon (en.wikipedia.org/wiki/Lyon)\n   Lyon is a city."
    assert "gsrsearch=capital+of+France" in seen[0]


def test_search_without_a_source_word_uses_wikipedia_and_nothing_found_says_no_results(tmp_path):
    ws = Workspace(str(tmp_path), search_fetch=lambda url: _json.dumps({"batchcomplete": ""}))
    assert ws.execute(_Call("search", "zorvel-7")).result == "no results"
    assert websearch.split_args("a, b, wikipedia") == ("a, b", "wikipedia")
    assert websearch.split_args("who is Ada, Lovelace") == ("who is Ada, Lovelace", "wikipedia")


def test_search_unknown_source_and_network_failure_are_errors_the_model_can_read(tmp_path):
    def down(url):
        raise OSError("no internet")
    ws = Workspace(str(tmp_path), search_fetch=down)
    assert "not connected" in ws.execute(_Call("search", "pandas, brave")).result
    r = ws.execute(_Call("search", "Paris, wikipedia")).result
    assert r.startswith("error: could not reach wikipedia")
