import pytest

from tools.sftbuild.session import SessionError, build_session


def big_module(n_funcs=40):
    """250+ line module: n_funcs functions of 6 lines each, f0..fN."""
    out = []
    for i in range(n_funcs):
        out += ["def f%d(x):" % i, '    """doc %d"""' % i, "    y = x + %d" % i, "    z = y * 2", "    return z", ""]
    return "\n".join(out)


def window(n, goal="G", todo="T", done="NONE"):
    return "### WINDOW %d\nGOAL: %s\nTODO: %s\nDONE_SO_FAR: %s\n" % (n, goal, todo, done)


def write_step(path, content, attempt="ok"):
    return "STEP: write | PATH: %s | ATTEMPT: %s\nCONTENT:\n<<<\n%s\n>>>\n" % (path, attempt, content)


def stage(n, kind, body):
    return "### STAGE %d | %s | en\n%s\n" % (n, kind, body)


def flat(sample):
    return [(s["t"], s["loss"]) for s in sample["segments"]]


def all_samples(text, tmp_path, **kw):
    return build_session(text, str(tmp_path / "w"), **kw)


def test_goal_block_only_after_window_one(tmp_path):
    text = (window(1) + stage(1, "chat", "USER: hi\nASSISTANT: hello")
            + window(2, goal="G2", todo="T2", done="D2") + stage(2, "chat", "USER: again\nASSISTANT: ok"))
    s1, s2 = all_samples(text, tmp_path)
    assert not any(t.startswith("<goal>") for t, _ in flat(s1))
    first = flat(s2)[0]
    assert first == ("<goal>\nG2\n</goal>\n<todo>\nT2\n</todo>\n<done>\nD2\n</done>\n", 0)


def test_assistant_goal_todo_line_becomes_tags_with_loss(tmp_path):
    text = window(1) + stage(1, "chat", "USER: sum?\nASSISTANT: GOAL: build x. TODO: a ; b")
    (s,) = all_samples(text, tmp_path)
    assert ("<goal>\nbuild x.\n</goal>\n<todo>\na ; b\n</todo>\n", 1) in flat(s)


def test_edit_range_is_real_line_numbers(tmp_path):
    body = ("USER: fix\nPLAN: write then edit\n" + write_step("m.py", big_module(3))
            + "STEP: edit | PATH: m.py | BUG_LINE:     y = x + 1 | NEW_LINE:     y = x - 1\nFINAL: done")
    (s,) = all_samples(window(1) + stage(1, "work", body), tmp_path)
    assert ("<call> edit(m.py, 9-9) </call>\n    y = x - 1\n", 1) in flat(s)
    assert "y = x - 1" in (tmp_path / "w" / "m.py").read_text()


def test_edit_text_found_twice_raises(tmp_path):
    body = ("USER: fix\nPLAN: p\n" + write_step("m.py", "v = 1\nv = 1\n")
            + "STEP: edit | PATH: m.py | BUG_LINE: v = 1 | NEW_LINE: v = 2\nFINAL: d")
    with pytest.raises(SessionError, match="twice|ambiguous"):
        all_samples(window(1) + stage(1, "work", body), tmp_path)


def test_traceback_placeholder_replaced_by_real_output(tmp_path):
    text = window(1) + stage(1, "work", "USER: broken:\n{TRACEBACK}\nfix it\nFINAL: ok")
    (s,) = all_samples(text, tmp_path, traceback=({}, 'python -c "raise IndexError(\'boom\')"'))
    user = flat(s)[0][0]
    assert "IndexError: boom" in user and "{TRACEBACK}" not in user and "fix it" in user


def test_read_chunks_followed_by_map_and_stub(tmp_path):
    body = ("USER: read\nPLAN: p\n" + write_step("m.py", big_module(40))
            + "STEP: read | PATH: m.py (lines 1-100)\nSTEP: read | PATH: m.py (lines 101-200)\nFINAL: ok")
    samples = all_samples(window(1) + stage(1, "work", body), tmp_path)
    assert len(samples) == 3  # map after chunk 1, map after chunk 2, tail after the final answer
    last_before_tail = [t for t, _ in flat(samples[1])]
    assert "<result>\n(dropped; see map)\n</result>" in last_before_tail  # chunk 1 stubbed
    maps = [t for t, _ in flat(samples[1]) if t.startswith("<map>")]
    assert maps[0].startswith("<map> f0 1-5") and maps[1].count("->") > maps[0].count("->")


def test_partial_read_without_scan_has_no_map(tmp_path):
    body = ("USER: read\nPLAN: p\n" + write_step("m.py", big_module(40))
            + "STEP: read | PATH: m.py (lines 60-100)\nFINAL: ok")
    (s,) = all_samples(window(1) + stage(1, "work", body), tmp_path)
    assert not any(t.startswith("<map>") for t, _ in flat(s))


def test_syntax_error_file_gets_no_map_and_a_flag(tmp_path):
    body = ("USER: read\nPLAN: p\n" + write_step("m.py", "def (:\n    pass")
            + "STEP: read | PATH: m.py (lines 1-2)\nFINAL: ok")
    samples = all_samples(window(1) + stage(1, "work", body), tmp_path)
    assert all("no_map" in s["flags"] for s in samples)
    assert not any(t.startswith("<map>") for s in samples for t, _ in flat(s))


def test_failed_write_attempt_masked(tmp_path):
    body = "USER: w\nPLAN: p\n" + write_step("m.py", "x = 1", attempt="failed") + "FINAL: ok"
    (s,) = all_samples(window(1) + stage(1, "work", body), tmp_path)
    call = [(t, l) for t, l in flat(s) if t.startswith("<call> write(m.py)")]
    assert call and call[0][1] == 0


def test_run_is_accepted_as_shell(tmp_path):
    body = "USER: r\nPLAN: p\nSTEP: run | CMD: echo hi\nFINAL: ok"
    (s,) = all_samples(window(1) + stage(1, "work", body), tmp_path)
    assert ("<call> shell(echo hi) </call>\n", 1) in flat(s)
    assert ("<result>\nhi\n</result>\n", 0) in flat(s)


def test_write_content_line_that_looks_like_a_marker(tmp_path):
    content = "intro\nFINAL: not an answer\nUSER: not a user\nSTEP: not a step\nend"
    body = "USER: w\nPLAN: p\n" + write_step("notes.txt", content) + "FINAL: ok"
    all_samples(window(1) + stage(1, "work", body), tmp_path)
    assert (tmp_path / "w" / "notes.txt").read_text() == content + "\n"


def test_refuses_to_wipe_an_unmarked_nonempty_workdir(tmp_path):
    keep = tmp_path / "w"
    keep.mkdir()
    (keep / "precious.txt").write_text("do not delete")
    with pytest.raises(SessionError, match="workdir"):
        all_samples(window(1) + stage(1, "chat", "USER: hi\nASSISTANT: hello"), tmp_path)
    assert (keep / "precious.txt").exists()


def test_second_run_on_own_workdir_is_allowed(tmp_path):
    text = window(1) + stage(1, "chat", "USER: hi\nASSISTANT: hello")
    all_samples(text, tmp_path)
    all_samples(text, tmp_path)
