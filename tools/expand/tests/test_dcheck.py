from tools.expand import dcheck

PROB = {"0001": "351"}
GOOD = """### trace 0001
<thinking>
1. Roots r, s of x^2 - 9x + 14 = 0 give r + s = 9 and r*s = 14.
2. r^2 + s^2 = (r + s)^2 - 2rs = 81 - 28 = 53.
<map> known: r+s=9, rs=14, r^2+s^2=53 | open: r^3+s^3 </map>
3. r^3 + s^3 = (r + s)(r^2 - rs + s^2) = 9 * (53 - 14) = 9 * 39 = 351.
check: roots are 2 and 7, so 2^3 + 7^3 = 8 + 343 = 351.
<map> known: r^3+s^3=351 | open: none </map>
</thinking>
ANSWER: 351"""


def errs(text, probs=PROB):
    return dcheck.check(probs, text)["0001"]


def test_good_trace_passes_including_chained_equals_and_powers():
    assert errs(GOOD) == []


def test_wrong_final_answer_and_wrong_arithmetic_are_caught():
    assert any("answer" in e for e in errs(GOOD.replace("ANSWER: 351", "ANSWER: 352")))
    assert any("arithmetic" in e for e in errs(GOOD.replace("81 - 28 = 53", "81 - 28 = 54")))


def test_only_format_arithmetic_and_answer_are_checked_no_style_rules():
    assert any("thinking" in e for e in errs(GOOD.replace("<thinking>\n", "")))
    assert any("ANSWER" in e for e in errs(GOOD + "\nThat is all."))
    assert errs(GOOD.replace("1. Roots", "1. Maybe the roots")) == []
    assert errs(GOOD.replace("check: roots are 2 and 7, so 2^3 + 7^3 = 8 + 343 = 351.\n", "")) == []
    assert errs(GOOD.replace("open: none", "open: more")) == []


def test_missing_trace_is_reported():
    assert dcheck.check(PROB, "nothing here")["0001"] == ["trace missing"]


def test_build_sample_uses_statement_as_user_turn_and_learns_the_whole_trace():
    s = dcheck.build_sample("0001", "Find r^3 + s^3.", GOOD)
    assert s["id"] == "D0001" and "deepseek_unreviewed" in s["flags"]
    assert s["segments"][0] == {"t": "<user>\nFind r^3 + s^3.\n</user>\n", "loss": 0}
    learned = [x["t"] for x in s["segments"] if x["loss"]]
    assert len(learned) == 1 and learned[0].startswith("<thinking>") and learned[0].rstrip().endswith("Answer: 351")


def test_function_notation_and_log_text_are_not_arithmetic_mistakes():
    for ok in ["f(2) = 40", "so log 128 = 7", "g(-2) = 16 - 28 - 2", "P(1+1) = 5*2*2 = 20", "sqrt 49 = 7", "2^7 = 128 and log2(128) = 7"]:
        assert dcheck.bad_chains(ok) == [], ok
    assert dcheck.bad_chains("so 7 * 8 = 54") == ["7 * 8 = 54"]


def test_clean_joins_traces_even_when_each_one_is_in_its_own_text_block():
    two = "```text\n" + GOOD.replace("0001", "0001") + "\n```\n\n```text\n" + GOOD.replace("0001", "0002") + "\n```\n\n0001 ok\n0002 ok"
    assert set(dcheck.parse(dcheck.clean(two))) == {"0001", "0002"}
