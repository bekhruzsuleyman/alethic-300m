import pytest

from tools.sftbuild.terminal import SkeletonError, build_terminal


def sk(eid="0700", kind="inspect", steps=None, plan="Look.", user="Show a.txt.", final="Done."):
    if steps is None:
        steps = ["STEP: shell | cat a.txt | EXPECT: OK"]
    return ("### ex %s | X | en | deepseek\nKIND: %s\nFILES:\n--- a.txt\n<<<\nhello\n>>>\nUSER: %s\nPLAN: %s\n%s\nFINAL: %s\n"
            % (eid, kind, user, plan, "\n".join(steps), final))


def flat(sample):
    return [(s["t"], s["loss"]) for s in sample["segments"]]


def test_ok_flow_loss_mask(tmp_path):
    (smp,) = build_terminal(sk(), str(tmp_path))
    f = flat(smp)
    assert f[0] == ("<user>\nShow a.txt.\n</user>\n", 0)
    assert ("<plan> Look. </plan>\n", 1) in f
    assert ("<call> shell(cat a.txt) </call>\n", 1) in f
    assert ("<result>\nhello\n</result>\n", 0) in f
    assert f[-2] == ("Done.\n", 1)


def test_expect_mismatch_raises(tmp_path):
    with pytest.raises(SkeletonError):
        build_terminal(sk(steps=["STEP: shell | cat nope.txt | EXPECT: OK"]), str(tmp_path))


def test_self_inflicted_fail_is_masked(tmp_path):
    steps = ["STEP: shell | cat nope.txt | EXPECT: FAIL", "STEP: shell | cat a.txt | EXPECT: OK"]
    (smp,) = build_terminal(sk(kind="fix", steps=steps), str(tmp_path))
    f = dict(flat(smp))
    assert f["<call> shell(cat nope.txt) </call>\n"] == 0
    assert f["<plan> Look. </plan>\n"] == 0
    assert f["<call> shell(cat a.txt) </call>\n"] == 1


def test_user_caused_fail_not_masked(tmp_path):
    steps = ["STEP: shell | cat nope.txt | EXPECT: FAIL"]
    (smp,) = build_terminal(sk(eid="0621", kind="fix", steps=steps), str(tmp_path), user_caused={"0621"})
    assert dict(flat(smp))["<call> shell(cat nope.txt) </call>\n"] == 1


def test_policy_block_raises(tmp_path):
    with pytest.raises(SkeletonError):
        build_terminal(sk(steps=["STEP: shell | sudo ls | EXPECT: OK"]), str(tmp_path))


def test_plan_none_and_no_steps(tmp_path):
    (smp,) = build_terminal(sk(kind="refuse", plan="NONE", steps=[], final="No."), str(tmp_path))
    texts = [t for t, _ in flat(smp)]
    assert not any("<plan>" in t or "<call>" in t for t in texts)


def test_workdir_is_shown_as_fixed_root(tmp_path):
    (smp,) = build_terminal(sk(steps=["STEP: shell | pwd | EXPECT: OK"]), str(tmp_path))
    assert ("<result>\n/work\n</result>\n", 0) in flat(smp)


def test_file_content_line_that_looks_like_a_marker(tmp_path):
    text = ("### ex 0701 | X | en | deepseek\nKIND: inspect\nFILES:\n--- a.txt\n<<<\nUSER: from the file\nFINAL: nope\n>>>\n"
            "USER: Show a.txt.\nPLAN: Look.\nSTEP: shell | cat a.txt | EXPECT: OK\nFINAL: Done.\n")
    (smp,) = build_terminal(text, str(tmp_path / "w"))
    assert smp["segments"][0]["t"] == "<user>\nShow a.txt.\n</user>\n"
    assert ("<result>\nUSER: from the file\nFINAL: nope\n</result>\n", 0) in flat(smp)


def test_refuses_to_wipe_an_unmarked_nonempty_workdir(tmp_path):
    keep = tmp_path / "w"
    keep.mkdir()
    (keep / "precious.txt").write_text("do not delete")
    with pytest.raises(SkeletonError, match="workdir"):
        build_terminal(sk(), str(keep))
    assert (keep / "precious.txt").exists()


def sk2(steps, files="--- a.txt\n<<<\nalpha\nbeta\ngamma\n>>>", eid="0702", kind="search_fix"):
    return ("### ex %s | F | en | deepseek\nKIND: %s\nFILES:\n%s\nUSER: Fix a.txt.\nPLAN: Run it.\n%s\nFINAL: Done.\n"
            % (eid, kind, files, "\n".join(steps)))


def test_read_step_records_numbered_result(tmp_path):
    (smp,) = build_terminal(sk2(["STEP: read | PATH: a.txt (lines 1-2)"]), str(tmp_path))
    f = flat(smp)
    assert ("<call> read(a.txt, 1-2) </call>\n", 1) in f
    assert ("<result>\n1   alpha\n2   beta\n</result>\n", 0) in f


def test_read_range_past_end_returns_existing_lines_and_keeps_the_asked_range(tmp_path):
    (smp,) = build_terminal(sk2(["STEP: read | PATH: a.txt (lines 1-9)"]), str(tmp_path))
    f = flat(smp)
    assert ("<call> read(a.txt, 1-9) </call>\n", 1) in f
    assert ("<result>\n1   alpha\n2   beta\n3   gamma\n</result>\n", 0) in f


def test_read_starting_past_end_raises(tmp_path):
    with pytest.raises(SkeletonError, match="range"):
        build_terminal(sk2(["STEP: read | PATH: a.txt (lines 7-9)"]), str(tmp_path))


def test_edit_step_changes_the_file_and_records_the_new_line(tmp_path):
    steps = ["STEP: edit | PATH: a.txt | BUG_LINE: beta | NEW_LINE: BETA",
             "STEP: shell | cat a.txt | EXPECT: OK"]
    (smp,) = build_terminal(sk2(steps), str(tmp_path))
    f = flat(smp)
    assert ("<call> edit(a.txt, 2-2) </call>\nBETA\n", 1) in f
    assert ("<result>\nalpha\nBETA\ngamma\n</result>\n", 0) in f


def test_edit_of_a_missing_line_raises(tmp_path):
    with pytest.raises(SkeletonError, match="found"):
        build_terminal(sk2(["STEP: edit | PATH: a.txt | BUG_LINE: nope | NEW_LINE: x"]), str(tmp_path))


def test_steps_keep_their_order(tmp_path):
    steps = ["STEP: shell | cat a.txt | EXPECT: OK",
             "STEP: read | PATH: a.txt (lines 1-3)",
             "STEP: edit | PATH: a.txt | BUG_LINE: gamma | NEW_LINE: GAMMA",
             "STEP: shell | tail -n 1 a.txt | EXPECT: OK"]
    (smp,) = build_terminal(sk2(steps), str(tmp_path))
    calls = [t.split("\n")[0] for t, _ in flat(smp) if t.startswith("<call>")]
    assert calls == ["<call> shell(cat a.txt) </call>", "<call> read(a.txt, 1-3) </call>",
                     "<call> edit(a.txt, 3-3) </call>", "<call> shell(tail -n 1 a.txt) </call>"]


def test_edit_path_outside_workdir_raises(tmp_path):
    with pytest.raises(SkeletonError, match="outside"):
        build_terminal(sk2(["STEP: edit | PATH: ../x.txt | BUG_LINE: a | NEW_LINE: b"]), str(tmp_path / "w"))



def project(steps, eid="2001"):
    return ("### ex %s | P | en | deepseek\nKIND: project\nSTACK: python\nUSER: Write hi.\nPLAN: One file.\n%s\nFINAL: Done.\n"
            % (eid, "\n".join(steps)))


WRITE = "STEP: write | PATH: hi.py\n<<<\nprint(\"hi\")\n>>>"


def test_write_step_creates_the_file_and_records_the_content(tmp_path):
    (smp,) = build_terminal(project([WRITE, "STEP: shell | python3 hi.py | EXPECT: OK"]), str(tmp_path))
    f = flat(smp)
    assert ("<call> write(hi.py) </call>\nprint(\"hi\")\n", 1) in f
    assert ("<result>\nhi\n</result>\n", 0) in f


def test_write_content_that_looks_like_a_step_is_content(tmp_path):
    step = "STEP: write | PATH: n.txt\n<<<\nSTEP: shell | rm -rf x | EXPECT: OK\nFINAL: nope\n>>>"
    (smp,) = build_terminal(project([step, "STEP: shell | cat n.txt | EXPECT: OK"]), str(tmp_path))
    assert ("<result>\nSTEP: shell | rm -rf x | EXPECT: OK\nFINAL: nope\n</result>\n", 0) in flat(smp)


def test_write_then_edit_then_run_in_order(tmp_path):
    steps = ["STEP: write | PATH: hi.py\n<<<\nprint(\"hi\"\n>>>",
             "STEP: shell | python3 hi.py | EXPECT: FAIL",
             "STEP: edit | PATH: hi.py | BUG_LINE: print(\"hi\" | NEW_LINE: print(\"hi\")",
             "STEP: shell | python3 hi.py | EXPECT: OK"]
    (smp,) = build_terminal(project(steps), str(tmp_path))
    calls = [t.split("\n")[0] for t, _ in flat(smp) if t.startswith("<call>")]
    assert calls == ["<call> write(hi.py) </call>", "<call> shell(python3 hi.py) </call>",
                     "<call> edit(hi.py, 1-1) </call>", "<call> shell(python3 hi.py) </call>"]


def test_write_outside_workdir_raises(tmp_path):
    step = "STEP: write | PATH: ../evil.py\n<<<\nx\n>>>"
    with pytest.raises(SkeletonError, match="outside"):
        build_terminal(project([step]), str(tmp_path / "w"))


class FakeSearcher:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def search(self, source, query, unique=None):
        self.calls.append((source, query, unique))
        return self.reply


def sfix(reply_expect="loadTestsFromTestCase", kind="search_fix", unique=None):
    head = "### ex 0703 | F | en | deepseek\nKIND: %s\n" % kind
    if unique:
        head += "UNIQUE: %s\n" % unique
    return (head + "FILES:\n--- a.txt\n<<<\nhello\n>>>\nUSER: Fix it.\nPLAN: Look it up.\n"
            "STEP: search | QUERY: python unittest has no attribute makeSuite | SOURCE: docs.python.org | EXPECT_IN_RESULT: %s\n"
            "FINAL: Done.\n" % reply_expect)


def test_search_step_records_call_and_result(tmp_path):
    s = FakeSearcher("1. unittest\n   use loadTestsFromTestCase")
    (smp,) = build_terminal(sfix(), str(tmp_path), searcher=s)
    f = flat(smp)
    assert ("<call> search(python unittest has no attribute makeSuite) </call>\n", 1) in f
    assert ("<result>\n1. unittest\n   use loadTestsFromTestCase\n</result>\n", 0) in f
    assert s.calls == [("docs.python.org", "python unittest has no attribute makeSuite", None)]


def test_search_expectation_is_accent_and_case_insensitive(tmp_path):
    build_terminal(sfix("LOADTESTSFROMTESTCASE, unittést"), str(tmp_path),
                   searcher=FakeSearcher("use loadTestsFromTestCase in unittest"))


def test_search_missing_expected_word_raises(tmp_path):
    with pytest.raises(SkeletonError, match="EXPECT_IN_RESULT"):
        build_terminal(sfix("nothing like this"), str(tmp_path), searcher=FakeSearcher("use loadTestsFromTestCase"))


def test_search_without_a_searcher_raises(tmp_path):
    with pytest.raises(SkeletonError, match="searcher"):
        build_terminal(sfix(), str(tmp_path))


def test_search_none_expects_no_results_and_passes_unique(tmp_path):
    s = FakeSearcher("no results")
    build_terminal(sfix("NONE", kind="search_empty", unique="KRT-9183"), str(tmp_path), searcher=s)
    assert s.calls[0][2] == "KRT-9183"


def test_search_none_with_real_hits_raises(tmp_path):
    with pytest.raises(SkeletonError, match="NONE"):
        build_terminal(sfix("NONE", kind="search_empty", unique="KRT-9183"), str(tmp_path),
                       searcher=FakeSearcher("1. something"))


def pfix(kind="project_fix"):
    steps = ["STEP: write | PATH: hi.py\n<<<\nprint(\"hi\"\n>>>",
             "STEP: shell | python3 hi.py | EXPECT: FAIL",
             "STEP: edit | PATH: hi.py | BUG_LINE: print(\"hi\" | NEW_LINE: print(\"hi\")",
             "STEP: shell | python3 hi.py | EXPECT: OK"]
    return project(steps).replace("KIND: project", "KIND: " + kind)


def test_project_fix_masks_the_buggy_write_and_the_failing_run(tmp_path):
    (smp,) = build_terminal(pfix(), str(tmp_path))
    f = flat(smp)
    assert ("<call> write(hi.py) </call>\nprint(\"hi\"\n", 0) in f      # buggy code is not trained on
    assert ("<call> shell(python3 hi.py) </call>\n", 0) in f           # the failing run
    assert ("<call> edit(hi.py, 1-1) </call>\nprint(\"hi\")\n", 1) in f
    assert ("<plan> One file. </plan>\n", 1) in f                       # the plan itself is fine
    assert f[-2] == ("Done.\n", 1)


def test_plain_project_never_masks_anything(tmp_path):
    steps = [WRITE, "STEP: shell | python3 hi.py | EXPECT: OK"]
    (smp,) = build_terminal(project(steps), str(tmp_path))
    assert all(loss == 1 for t, loss in flat(smp) if t.startswith(("<call>", "<plan>", "Done")))
