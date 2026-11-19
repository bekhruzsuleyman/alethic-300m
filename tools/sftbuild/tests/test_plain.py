from tools.sftbuild.plain import build_plain

EX = """### ex 0001 | A | en | deepseek
<user>
What is 2+2?
</user>
<assistant>
4.
</assistant>
"""

WITH_RESULT = """### ex 0002 | D | en | deepseek
<user>
Total of 3 and 4?
</user>
<assistant>
<call> calc(3+4) </call>
<result>
7
</result>
The total is 7.
</assistant>
"""


def flat(s):
    return [(g["t"], g["loss"]) for g in s["segments"]]


def test_plain_one_sample_per_example_assistant_loss_only():
    (s,) = build_plain(EX)
    assert s["id"] == "P0001.0"
    assert flat(s) == [("<user>\nWhat is 2+2?\n</user>\n", 0), ("<assistant>\n", 0),
                       ("4.\n", 1), ("</assistant>\n", 0)]


def test_hand_written_result_block_has_no_loss_and_is_flagged():
    (s,) = build_plain(WITH_RESULT)
    f = flat(s)
    assert ("<result>\n7\n</result>\n", 0) in f
    assert ("<call> calc(3+4) </call>\n", 1) in f
    assert s["flags"] == ["hand_result"]


def test_clean_example_has_no_flags():
    assert build_plain(EX)[0]["flags"] == []


INLINE = """### ex 0003 | D | en | deepseek
<user>
Perimeter 12 by 7?
</user>
<assistant>
<thinking> p = 2 * (12 + 7). </thinking>
<call> calc(2*(12+7)) </call>
<result> 38 </result>
<call> shell(python app.py) </call>
<result> Traceback (most recent call last):
  File "app.py", line 3
ImportError: x </result>
<error> app.py, line 3: ImportError: x </error>
The perimeter is 38.
</assistant>
"""


def test_inline_and_multiline_results_and_errors_have_no_loss():
    (s,) = build_plain(INLINE)
    f = flat(s)
    masked = [t for t, l in f if l == 0 and (t.startswith("<result>") or t.startswith("<error>"))]
    assert len(masked) == 3
    assert not any(l == 1 and ("<result>" in t or "<error>" in t) for t, l in f)
    assert any(l == 1 and "<call> calc(2*(12+7)) </call>" in x for x, l in f)
    assert ("The perimeter is 38.\n", 1) in f
    assert s["flags"] == ["hand_result"]


def test_second_turn_user_text_gets_no_loss():
    from tools.sftbuild.plain import build_plain
    text = ("### ex 1003 | R | en | deepseek\n<user>\nq1\n</user>\n<assistant>\nno1\n</assistant>\n"
            "<user>\nq2 pushback\n</user>\n<assistant>\nno2\n</assistant>\n")
    s = build_plain(text)[0]
    loss_text = "".join(g["t"] for g in s["segments"] if g["loss"])
    assert "no1" in loss_text and "no2" in loss_text
    assert "q2 pushback" not in loss_text and "<user>" not in loss_text and "</assistant>" not in loss_text
    assert "".join(g["t"] for g in s["segments"]).count("<assistant>") == 2
