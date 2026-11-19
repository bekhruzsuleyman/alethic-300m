import pytest

from tools.repair.replay import ReplayError, replay
from tools.sftbuild.plain import build_plain


def run(body, files=None, tmp="x"):
    return replay(body, files or {}, tmp)


def test_calc_result_is_computed_not_trusted(tmp_path):
    out = replay("<call> calc(4873*29) </call>\n<result> 999 </result>\n141,317.", {}, str(tmp_path))
    assert out == "<call> calc(4873*29) </call>\n<result> 141317 </result>\n141,317."


def test_calc_float_noise_is_rounded(tmp_path):
    out = replay("<call> calc(7*2.50 + 4*1.80) </call>\n<result> 1 </result>", {}, str(tmp_path))
    assert "<result> 24.7 </result>" in out


def test_calc_rejects_anything_but_arithmetic(tmp_path):
    with pytest.raises(ReplayError):
        replay("<call> calc(__import__('os').getcwd()) </call>", {}, str(tmp_path))


def test_list_is_sorted_with_slash_on_folders(tmp_path):
    files = {"main.py": "", "README.md": "", "assets/a.png": ""}
    out = replay("<call> list(.) </call>\n<result> old </result>", files, str(tmp_path))
    assert out == "<call> list(.) </call>\n<result> README.md\nassets/\nmain.py </result>"


def test_list_of_an_empty_folder(tmp_path):
    out = replay("<call> list(.) </call>\n<result> x </result>", {}, str(tmp_path))
    assert "<result> (empty) </result>" in out


def test_read_is_numbered_from_the_real_file(tmp_path):
    out = replay("<call> read(c.py, 2-3) </call>\n<result> bogus </result>", {"c.py": "a = 1\nb = 2\nc = 3\n"}, str(tmp_path))
    assert out == "<call> read(c.py, 2-3) </call>\n<result> 2   b = 2\n3   c = 3 </result>"


def test_read_past_the_end_stops_at_the_last_line(tmp_path):
    out = replay("<call> read(c.py, 1-10) </call>\n<result> bogus </result>", {"c.py": "a = 1\nb = 2\n"}, str(tmp_path))
    assert out == "<call> read(c.py, 1-10) </call>\n<result> 1   a = 1\n2   b = 2 </result>"


def test_read_starting_past_the_end_fails_loudly(tmp_path):
    with pytest.raises(ReplayError):
        replay("<call> read(c.py, 5-9) </call>", {"c.py": "a = 1\n"}, str(tmp_path))


def test_failing_shell_gets_a_real_result_and_a_generated_error(tmp_path):
    body = "<call> shell(python main.py) </call>\n<result> old </result>\n<error> old </error>\nIt fails."
    out = replay(body, {"main.py": "a = 1\nprint(b)\n"}, str(tmp_path))
    assert "NameError: name 'b' is not defined" in out
    assert "/work/main.py" in out and str(tmp_path) not in out
    assert "<error> main.py, line 2: NameError: name 'b' is not defined </error>" in out
    assert "old" not in out and out.endswith("It fails.")


def test_passing_shell_has_no_error_line(tmp_path):
    out = replay("<call> shell(python -c \"print('hi')\") </call>\n<result> hi </result>", {}, str(tmp_path))
    assert out == "<call> shell(python -c \"print('hi')\") </call>\n<result> hi </result>"


def test_silent_shell_result_says_no_output(tmp_path):
    out = replay("<call> shell(true) </call>\n<result> whatever </result>", {}, str(tmp_path))
    assert "<result> (no output) </result>" in out


def test_write_then_edit_then_run_follow_real_state(tmp_path):
    body = ("<call> write(g.py) </call>\nprint(\"hello\")\n<call> shell(python g.py) </call>\n<result> x </result>\n"
            "<call> edit(g.py, 1-1) </call>\nprint(\"hi\")\n<call> shell(python g.py) </call>\n<result> x </result>\nDone.")
    out = replay(body, {}, str(tmp_path))
    assert out.count("<result> hello </result>") == 1 and out.count("<result> hi </result>") == 1
    assert "print(\"hello\")\n<call> shell" in out and out.endswith("Done.")


def test_edit_can_replace_one_line_by_several(tmp_path):
    body = "<call> edit(m.py, 2-2) </call>\nx = 1\nprint(x)\n<call> shell(python m.py) </call>\n<result> z </result>"
    out = replay(body, {"m.py": "import os\nprint(x)\n"}, str(tmp_path))
    assert "<result> 1 </result>" in out


def test_other_lines_pass_through_unchanged(tmp_path):
    body = "<thinking> check </thinking>\n<call> calc(1+1) </call>\n<result> 2 </result>\n<done> ok\nTwo."
    assert replay(body, {}, str(tmp_path)) == body


def test_plain_builder_does_not_flag_replayed_examples():
    text = ("### ex 0900 | type: E | lang: en | source: replay\n<user>\nq\n</user>\n<assistant>\n"
            "<call> calc(1+1) </call>\n<result> 2 </result>\nTwo.\n</assistant>\n")
    s = build_plain(text)
    assert all(x["flags"] == [] for x in s)
    assert build_plain(text.replace("source: replay", "source: script"))[0]["flags"] == ["hand_result"]
