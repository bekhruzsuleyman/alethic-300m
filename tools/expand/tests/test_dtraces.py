import json

from tools.expand import dcheck, dtraces, runner


def good_trace(qid, answer):
    return (f"### trace {qid}\n<thinking>\n1. Work the problem step by step.\n<map> known: setup done | open: result </map>\n"
            f"2. The result is {answer}.\ncheck: substituting gives the same value.\n<map> known: result={answer} | open: none </map>\n"
            f"</thinking>\nANSWER: {answer}")


def fake_call_factory(wrong_ids=()):
    def call(prompt):
        ids = [l.split()[2] for l in prompt.split("```text\n")[-1].splitlines() if l.startswith("### problem")]
        by_id = {p["id"]: p for p in dtraces.make_problems(40)}
        traces = [good_trace(i, "999999" if i in wrong_ids else by_id[i]["answer"]) for i in ids]
        return "```text\n" + "\n\n".join(traces) + "\n```\n" + "\n".join(i + " ok" for i in ids), {"completion_tokens": 100}
    return call


def test_make_problems_is_deterministic_unique_and_has_known_answers():
    a, b = dtraces.make_problems(30), dtraces.make_problems(30)
    assert a == b and len({p["statement"] for p in a}) == 30
    assert all(p["answer"] and p["id"].isdigit() and len(p["id"]) == 4 for p in a)
    assert {p["family"] for p in a} == set(__import__("tools.deepthink.mathgen", fromlist=["x"]).FAMILIES)


def test_prompt_contains_base_the_pack_and_forbids_the_ready_handshake():
    probs = dtraces.make_problems(3)
    prompt = dtraces.build_prompt("BASE TEXT", probs, 1)
    assert prompt.startswith("BASE TEXT") and "### problem %s | math | en" % probs[0]["id"] in prompt
    assert "do NOT reply READY" in prompt


def test_process_reply_keeps_verified_traces_and_rejects_wrong_answers():
    probs = dtraces.make_problems(4)
    reply, _ = fake_call_factory(wrong_ids=(probs[1]["id"],))(dtraces.build_prompt("B", probs, 1))
    acc, rej = dtraces.process_reply(probs, reply)
    assert [i for i, _ in acc] == [probs[0]["id"], probs[2]["id"], probs[3]["id"]]
    assert list(rej) == [probs[1]["id"]] and "answer differs" in rej[probs[1]["id"]][0]


def test_run_writes_outputs_resumes_and_stops_on_a_bad_rate(tmp_path):
    probs = dtraces.make_problems(10)
    batches = dtraces.plan_batches(probs, per_batch=5)
    budget = runner.Budget(5.0, 0.3, 0.006, 1.2)
    stats = dtraces.run(batches, fake_call_factory(), tmp_path, budget, workers=2)
    assert stats["accepted"] == 10 and stats["batches"] == 2
    lines = (tmp_path / "built_D.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10 and json.loads(lines[0])["segments"][2]["loss"] == 1
    calls = []
    dtraces.run(batches, lambda p: calls.append(p) or ("", {}), tmp_path, budget, workers=2)
    assert calls == []                                                          # raw replies exist: nothing is called again
    bad = tmp_path / "bad"
    stats = dtraces.run(batches, lambda p: ("nothing useful", {}), bad, budget, workers=1, min_batches_for_breaker=1)
    assert stats["stopped"].startswith("low acceptance")


def test_reprocess_reads_traces_from_separate_text_blocks_and_a_no_op_run_keeps_old_stats(tmp_path):
    probs = dtraces.make_problems(4)
    batches = dtraces.plan_batches(probs, per_batch=4)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "D_1.txt").write_text("\n\n".join("```text\n%s\n```" % good_trace(p["id"], p["answer"]) for p in probs), encoding="utf-8")
    stats = dtraces.reprocess(batches, raw, tmp_path / "out")
    assert stats["accepted"] == 4 and (tmp_path / "out" / "built_D.jsonl").read_text().count("\n") == 4
    (tmp_path / "stats.json").write_text('{"accepted": 4}')
    again = dtraces.run(batches, lambda p: 1 / 0, tmp_path, runner.Budget(5, .3, .006, 1.2))   # raw reply exists: nothing runs
    assert again["batches"] == 0 and '"accepted": 4' in (tmp_path / "stats.json").read_text()
