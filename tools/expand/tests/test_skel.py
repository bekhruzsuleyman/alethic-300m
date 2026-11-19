import json

from tools.expand import skel

GOOD = """### ex 2001 | P | en | deepseek
KIND: project
STACK: python
USER: Write a python script that sums 1 to 10 and prints it.
PLAN: One file, sum with a loop, run.
STEP: write | PATH: main.py
<<<
total = 0
for i in range(1, 11):
    total += i
print("total", total)
>>>
STEP: shell | python3 main.py | EXPECT: OK
FINAL: Done. It prints total 55."""


def test_good_skeleton_builds_with_real_results_and_a_wrong_expectation_is_rejected():
    ok, rej = skel.process_reply("```text\n" + GOOD + "\n```")
    assert [i for i, _, _ in ok] == ["2001"] and not rej
    assert "total 55" in "".join(s["t"] for s in ok[0][2][0]["segments"])
    bad = GOOD.replace("EXPECT: OK", "EXPECT: FAIL")
    ok, rej = skel.process_reply("```text\n" + bad + "\n```")
    assert not ok and "expected FAIL" in rej["2001"][0]


def test_each_skeleton_in_its_own_block_is_read_and_bad_ones_do_not_stop_good_ones():
    second = GOOD.replace("2001", "2002")
    broken = GOOD.replace("2001", "2003").replace("print(", "pritn(")
    reply = "\n\n".join("```text\n%s\n```" % x for x in (GOOD, broken, second)) + "\n\n2001 ok"
    ok, rej = skel.process_reply(reply)
    assert [i for i, _, _ in ok] == ["2001", "2002"] and list(rej) == ["2003"]


def test_rows_are_deterministic_cover_every_stack_and_the_prompt_forbids_ready():
    rows = skel.make_rows(120, seed=5)
    assert rows == skel.make_rows(120, seed=5)
    assert {r["stack"] for r in rows} >= {"python", "rust", "go", "c", "javascript", "typescript", "html_css"}
    assert {r["kind"] for r in rows} == {"project", "project_fix"} and {r["lang"] for r in rows} == {"en", "uz", "mix"}
    p = skel.build_prompt("BASE", rows[:3])
    assert "do NOT reply READY" in p and all("%s %s" % (r["id"], r["lang"]) in p for r in rows[:3])


def test_run_writes_files_resumes_and_counts(tmp_path):
    from tools.expand import runner
    rows = skel.make_rows(4, start=2001, seed=1)

    def call(prompt):
        import re
        ids = re.findall(r"^(\d{4}) (?:en|uz|mix) ", prompt.split("Scenarios:")[-1], re.M)
        return "\n\n".join("```text\n%s\n```" % GOOD.replace("2001", i) for i in ids), {"completion_tokens": 3}
    stats = skel.run(skel.plan_batches(rows, 2), call, tmp_path, runner.Budget(5, .3, .006, 1.2))
    assert stats["accepted"] == 4 and (tmp_path / "built_P.jsonl").read_text().count("\n") >= 4
    assert skel.run(skel.plan_batches(rows, 2), lambda p: 1 / 0, tmp_path, runner.Budget(5, .3, .006, 1.2))["batches"] == 0
