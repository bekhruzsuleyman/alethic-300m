import re
from tools.expand import cthink

GOOD = """### think 0001 | word problem | en
<question>
A shop sells pens at 3 coins each. How much do 14 pens cost?
</question>
<thinking>
1. Each pen costs 3 coins, and there are 14 pens.
2. Total = 14 * 3 = 42.
check: 42 / 3 = 14 pens.
</thinking>
<answer>
14 pens cost 42 coins.
</answer>
<verify>
print(14 * 3)
</verify>"""


def one(text):
    return cthink.check(text)["0001"]


def test_good_example_passes():
    assert one(GOOD) == []


def test_verify_code_must_print_a_value_found_in_the_answer():
    assert any("verify" in e for e in one(GOOD.replace("print(14 * 3)", "print(14 * 4)")))
    assert any("verify" in e for e in one(GOOD.replace("print(14 * 3)", "raise SystemExit(1)")))


def test_wrong_arithmetic_in_thinking_is_rejected_and_loose_style_is_not():
    assert any("arithmetic" in e for e in one(GOOD.replace("14 * 3 = 42", "14 * 3 = 43")))
    assert one(GOOD.replace("1. Each pen", "1. Maybe each pen")) == []


def test_missing_parts_are_reported_and_extra_tags_ignored():
    assert any("format" in e for e in one(GOOD.replace("<verify>\nprint(14 * 3)\n</verify>", "")))
    assert one(GOOD.replace("<question>", "<level>easy</level>\n<question>")) == []


def test_build_sample_puts_thinking_first_in_the_trained_answer_and_drops_verify():
    s = cthink.build_sample("0001", GOOD)
    assert s["id"] == "C0001" and s["segments"][0]["t"].startswith("<user>\nA shop sells pens")
    learned = "".join(x["t"] for x in s["segments"] if x["loss"])
    assert learned.startswith("<thinking>") and learned.rstrip().endswith("14 pens cost 42 coins.") and "print(" not in learned


def test_rows_prompt_and_run(tmp_path):
    from tools.expand import runner
    rows = cthink.make_rows(6, seed=2)
    assert rows == cthink.make_rows(6, seed=2) and {r["level"] for r in cthink.make_rows(100, seed=1)} == {"easy", "medium", "hard"}
    p = cthink.build_prompt("BASE", rows)
    assert "do NOT reply READY" in p and all("think %s |" % r["id"] in p for r in rows)

    def call(prompt):
        ids = re.findall(r"^think (\d{4}) \|", prompt.split("Rows:")[-1], re.M)
        return "```text\n" + "\n\n".join(GOOD.replace("0001", i) for i in ids) + "\n```", {"completion_tokens": 5}
    stats = cthink.run(cthink.plan_batches(rows, 3), call, tmp_path, runner.Budget(5, .3, .006, 1.2))
    assert stats["accepted"] == 6 and (tmp_path / "built_C.jsonl").read_text().count("\n") == 6
