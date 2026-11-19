import json

from tools.expand import runner
from tools.expand.tests.test_pcheck import T_OK, T_ROWS, E_OK, E_ROWS


def test_plan_batches_cover_ids_without_gaps_and_pair_numbers():
    bs = runner.plan_batches("T", start=3001, total=40, per_batch=15, seed=1)
    assert [b["count"] for b in bs] == [15, 15, 10] and [b["start"] for b in bs] == [3001, 3016, 3031]
    es = runner.plan_batches("E", start=4301, total=30, per_batch=14, seed=1)
    assert [b["first_pair"] for b in es] == [1, 8, 15] and [b["count"] for b in es] == [14, 14, 2]


def test_process_reply_keeps_good_examples_and_reports_bad_ones():
    bad = T_OK.replace("RAM ishlayotgan", "Ajoyib! RAM ishlayotgan")
    reply = "```text\n" + bad + "\n```\n3001 ok\n3002 ok"
    acc, rejects = runner.process_reply("T", T_ROWS, reply)
    assert [i for i, _ in acc] == ["3002"] and "3001" in rejects


def test_process_reply_without_block_uses_whole_text_and_drops_pairs_together():
    acc, rejects = runner.process_reply("E", E_ROWS, E_OK)
    assert [i for i, _ in acc] == ["4301", "4302"] and not rejects
    acc, rejects = runner.process_reply("E", E_ROWS, E_OK.replace("Verdict changed: not yet.", "Verdict changed: weak."))
    assert acc == [] and set(rejects) == {"4301", "4302"}


def test_dedup_drops_near_identical_first_user_turn():
    seen = runner.Seen()
    a = ("1", "### ex 1 | T | en | deepseek\n<user>\nWhat is a hash table and how does it work?\n</user>\n<assistant>\nx\n</assistant>")
    b = ("2", "### ex 2 | T | en | deepseek\n<user>\nwhat is a hash table, and how does it work\n</user>\n<assistant>\ny\n</assistant>")
    assert seen.add(*a) is True and seen.add(*b) is False


def test_budget_counts_cache_hits_cheaper_and_stops():
    b = runner.Budget(cap=0.01, miss=1.0, hit=0.1, out=2.0)
    b.add({"prompt_tokens": 1000, "prompt_cache_hit_tokens": 900, "completion_tokens": 1000})
    assert abs(b.spent - (100 * 1.0 + 900 * 0.1 + 1000 * 2.0) / 1e6) < 1e-12
    assert b.over() is False
    b.add({"prompt_tokens": 0, "completion_tokens": 10_000})
    assert b.over() is True


def test_run_resumes_skips_done_batches_and_stops_when_over_budget(tmp_path):
    calls = []
    def fake(prompt):
        calls.append(prompt)
        return "```text\n" + T_OK + "\n```", {"prompt_tokens": 10, "completion_tokens": 10}
    batches = [{"kind": "T", "start": 3001, "count": 2, "batch": 1, "rows": T_ROWS, "first_pair": 1, "base": "BASE"},
               {"kind": "T", "start": 3003, "count": 2, "batch": 2, "rows": T_ROWS, "first_pair": 1, "base": "BASE"}]
    b = runner.Budget(cap=10, miss=0, hit=0, out=0)
    stats = runner.run(batches, fake, tmp_path, b, workers=1)
    assert stats["accepted"] == 2 and len(calls) == 2           # dedup keeps the 2nd batch from adding the same texts
    calls.clear()
    runner.run(batches, fake, tmp_path, runner.Budget(cap=10, miss=0, hit=0, out=0), workers=1)
    assert calls == []                                            # raw replies exist: nothing is called again
    out = (tmp_path / "accepted_T.txt").read_text()
    assert "### ex 3001" in out and (tmp_path / "built_T.jsonl").read_text().strip()
    over = runner.Budget(cap=0.0, miss=1.0, hit=1.0, out=1.0)
    over.spent = 1.0
    assert runner.run([dict(batches[0], batch=9)], fake, tmp_path / "x", over, workers=1)["stopped"] == "budget"


def test_process_reply_accepts_pair_halves_that_share_one_header_id_and_a_trailing_checklist():
    """DeepSeek often writes both halves of a pair under the same id (4301 twice) and ends with '4301 ok'."""
    import re
    second = E_OK.split("### ex 4302", 1)[1]
    shared = E_OK.split("### ex 4302", 1)[0] + "### ex 4301" + second          # the update half also says 4301
    reply = "```text\n" + shared + "\n```\n\n4301 ok"
    acc, rejects = runner.process_reply("E", E_ROWS, reply)
    assert not rejects and [i for i, _ in acc] == ["4301", "4302"]
    fenceless = shared + "\n\n4301 ok\n4303 ok"                               # no fence: the checklist must not break the last example
    acc, rejects = runner.process_reply("E", E_ROWS, fenceless)
    assert not rejects and len(acc) == 2
    assert not re.search(r"\d{4} ok", "".join(t for _, t in acc))


def test_reprocess_rebuilds_outputs_from_saved_raw_replies_without_api(tmp_path):
    second = E_OK.split("### ex 4302", 1)[1]
    shared = E_OK.split("### ex 4302", 1)[0] + "### ex 4301" + second
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "E_4301.txt").write_text("```text\n" + shared + "\n```\n4301 ok", encoding="utf-8")
    (raw / "T_3001.txt").write_text(T_OK, encoding="utf-8")
    batches = [{"kind": "E", "start": 4301, "rows": E_ROWS}, {"kind": "T", "start": 3001, "rows": T_ROWS},
               {"kind": "T", "start": 3003, "rows": T_ROWS}]                 # 3003 has no raw file: skipped
    wd = tmp_path / "out"
    stats = runner.reprocess(batches, raw, wd)
    assert stats["accepted"] == 4 and stats["asked"] == 4 and stats["batches"] == 2
    assert "4302" in (wd / "accepted_E.txt").read_text(encoding="utf-8")
    assert (wd / "built_T.jsonl").read_text(encoding="utf-8").count("deepseek_unreviewed") >= 2
    assert (wd / "rejects.jsonl").read_text(encoding="utf-8") == ""


def test_dedup_keeps_both_halves_of_a_pair_and_drops_a_repeated_pair_whole():
    seen = runner.Seen()
    acc, _ = runner.process_reply("E", E_ROWS, E_OK)
    assert [i for i, _ in runner.dedup("E", acc, seen)] == ["4301", "4302"]
    assert runner.dedup("E", acc, seen) == []                                     # the same pair again: both go


def test_style_note_is_added_only_when_a_style_is_set_and_the_checker_follows_it():
    rows = ["3001 en 2 F topic a"]
    assert "STYLE FOR THIS BATCH" not in runner.build_prompt("BASE", "T", rows, "b.md", 1)
    runner.STYLE["T"] = "detailed"
    try:
        assert "NO word limit" in runner.build_prompt("BASE", "T", rows, "b.md", 1)
        long_t = T_OK.replace("16 GB bo'lsa, ko'proq dastur ochiq turadi.", "16 GB bo'lsa, ko'proq dastur ochiq turadi. " + " ".join(["RAM keeps programs close to the processor so they react fast."] * 9))
        acc, rejects = runner.process_reply("T", T_ROWS, long_t)               # style comes from runner.STYLE
        assert "3001" in [i for i, _ in acc]
    finally:
        runner.STYLE.clear()
    acc, rejects = runner.process_reply("T", T_ROWS, long_t)
    assert "3001" in rejects                                                  # strict again: over 70 words
