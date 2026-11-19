from tools.expand import bcode

GOOD_PY = """### code 0001 | py | en
<task>
Write a function is_even(n) that returns True for even integers.
</task>
<plan>
Return whether n modulo 2 equals 0.
</plan>
```python
def is_even(n):
    return n % 2 == 0
```
<tests>
assert is_even(4) is True
assert is_even(7) is False
assert is_even(0) is True
</tests>"""


def one(text):
    return bcode.check(text)["0001"]


def test_good_python_example_passes_and_runs_the_tests():
    assert one(GOOD_PY) == []


def test_code_that_fails_its_own_tests_is_rejected():
    assert any("tests failed" in e for e in one(GOOD_PY.replace("% 2 == 0", "% 2 == 1")))


def test_infinite_loop_is_stopped_by_the_timeout():
    bad = GOOD_PY.replace("    return n % 2 == 0", "    while True:\n        pass")
    assert any("timeout" in e for e in one(bad))


def test_only_missing_parts_and_missing_asserts_are_structure_errors():
    assert any("plan" in e or "format" in e for e in one(GOOD_PY.replace("<plan>\nReturn whether n modulo 2 equals 0.\n</plan>\n", "")))
    no_assert = GOOD_PY.replace("assert is_even(4) is True\nassert is_even(7) is False\nassert is_even(0) is True", "is_even(4)")
    assert any("assert" in e for e in one(no_assert))
    assert one(GOOD_PY.replace("Return whether", "Let me return whether")) == []         # no wording rules


def test_javascript_example_runs_with_node():
    js = """### code 0001 | js | en
<task>
Write isEven(n) in JavaScript.
</task>
<plan>
Return whether n modulo 2 equals 0.
</plan>
```javascript
function isEven(n) { return n % 2 === 0; }
```
<tests>
console.assert(isEven(4) === true);
console.assert(isEven(7) === false);
console.assert(isEven(0) === true);
</tests>"""
    assert one(js) == []
    assert any("tests failed" in e for e in one(js.replace("n % 2 === 0", "n % 2 === 1")))


def test_build_sample_drops_tests_and_trains_plan_plus_code_only():
    s = bcode.build_sample("0001", GOOD_PY)
    assert s["id"] == "B0001" and s["segments"][0]["t"].startswith("<user>\nWrite a function is_even")
    learned = "".join(x["t"] for x in s["segments"] if x["loss"])
    assert learned.startswith("<plan>") and "def is_even" in learned and "assert" not in learned


def test_rows_are_deterministic_unique_and_follow_the_language_mix():
    a, b = bcode.make_rows(200, seed=3), bcode.make_rows(200, seed=3)
    assert a == b and len({(r["topic"], r["level"]) for r in a}) > 60
    assert 0.6 <= sum(r["lang"] == "py" for r in a) / 200 <= 0.85 and any(r["ulang"] == "uz" for r in a)


def test_prompt_lists_every_row_and_forbids_ready():
    rows = bcode.make_rows(5, seed=1)
    p = bcode.build_prompt("BASE", rows)
    assert p.startswith("BASE") and "do NOT reply READY" in p and all("code %s |" % r["id"] in p for r in rows)


def test_run_keeps_passing_examples_writes_files_and_resumes(tmp_path):
    from tools.expand import runner
    rows = bcode.make_rows(4, seed=2)

    def call(prompt):
        out = []
        for r in rows:
            ex = GOOD_PY.replace("0001", r["id"]).replace("| py | en", "| py | " + r["ulang"])
            out.append(ex if r["id"] != rows[0]["id"] else ex.replace("% 2 == 0", "% 2 == 1"))   # the first one has wrong code
        return "```text\n" + "\n\n".join(out) + "\n```", {"completion_tokens": 10}

    stats = bcode.run(bcode.plan_batches(rows, 4), call, tmp_path, runner.Budget(5, .3, .006, 1.2))
    assert stats["accepted"] == 3 and (tmp_path / "built_B.jsonl").read_text().count("\n") == 3
    assert bcode.run(bcode.plan_batches(rows, 4), lambda p: 1 / 0, tmp_path, runner.Budget(5, .3, .006, 1.2))["batches"] == 0


def test_examples_in_separate_text_blocks_are_all_read():
    other = GOOD_PY.replace("0001", "0002")
    reply = "```text\n" + GOOD_PY + "\n```\n\n```text\n" + other + "\n```\n\n0001 ok\n0002 ok"
    acc, rej = bcode.process_reply(reply)
    assert [i for i, _ in acc] == ["0001", "0002"] and not rej


def test_reprocess_rebuilds_outputs_from_saved_replies(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "B_1.txt").write_text("```text\n" + GOOD_PY + "\n```\n```text\n" + GOOD_PY.replace("0001", "0002") + "\n```", encoding="utf-8")
    stats = bcode.reprocess(raw, tmp_path / "out")
    assert stats["accepted"] == 2 and (tmp_path / "out" / "built_B.jsonl").read_text().count("\n") == 2


def test_extra_topic_and_level_lines_and_odd_spacing_do_not_matter():
    odd = GOOD_PY.replace("<task>", "<topic>math</topic>\n<level>easy</level>\n<task>")
    assert one(odd) == []
    assert bcode.build_sample("0001", odd)["id"] == "B0001"
    assert one(GOOD_PY.replace("</plan>\n```python", "</plan>\n\n```python")) == []
