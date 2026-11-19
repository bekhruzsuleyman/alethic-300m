from tools.sftbuild.seq import STUB, Seq


def test_stub_text():
    assert STUB == "<result>\n(dropped; see map)\n</result>"


def test_no_map_gives_one_sample():
    s = Seq()
    s.add("<user>\nq\n</user>\n", 0)
    s.add("answer\n", 1)
    out = s.samples("P1")
    assert len(out) == 1 and out[0]["id"] == "P1.0"
    assert [x["loss"] for x in out[0]["segments"]] == [0, 1]


def _two_reads():
    s = Seq()
    s.add("<user>\nq\n</user>\n", 0)
    s.add("<call> read(a.py, 1-100) </call>\n", 1)
    r1 = s.add("<result>\nBIG1\n</result>\n", 0, "read")
    m1 = s.add("<map> f 1-9 </map>\n", 1, "map")
    s.set_read_map(r1, m1)
    s.add("<call> read(a.py, 101-200) </call>\n", 1)
    r2 = s.add("<result>\nBIG2\n</result>\n", 0, "read")
    m2 = s.add("<map> f 1-9 -> g 12-20 </map>\n", 1, "map")
    s.set_read_map(r2, m2)
    return s


def test_map_event_splits_and_stubs_old_read():
    out = _two_reads().samples("W1")
    assert len(out) == 2
    texts1 = [x["t"] for x in out[0]["segments"]]
    texts2 = [x["t"] for x in out[1]["segments"]]
    assert "<result>\nBIG1\n</result>\n" in texts1  # sample 1 still sees its own read
    assert STUB in texts2 and not any("BIG1" in t for t in texts2)
    assert any("BIG2" in t for t in texts1) is False  # read 2 only exists in sample 2


def test_context_before_split_has_loss_zero():
    out = _two_reads().samples("W1")
    first_call_in_s2 = out[1]["segments"][1]
    assert first_call_in_s2["t"].startswith("<call> read(a.py, 1-100)")
    assert first_call_in_s2["loss"] == 0
    assert out[0]["segments"][1]["loss"] == 1  # trained once, in sample 1


def test_user_and_result_never_have_loss():
    for smp in _two_reads().samples("W1"):
        for seg in smp["segments"]:
            if seg["t"].startswith(("<user>", "<result>")):
                assert seg["loss"] == 0
