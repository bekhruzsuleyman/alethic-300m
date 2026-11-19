import ast
import re
from collections import Counter

from tools.skilltraces.build import DEFAULT_COUNTS, KINDS, build_all
from tools.sftbuild.mapper import ast_map

DATA = build_all(per_kind=20, seed=3)


def text(e, loss=None):
    return "".join(s["t"] for s in e["segments"] if loss is None or s["loss"] == loss)


def test_kinds_langs_ids_and_determinism():
    assert {e["kind"] for e in DATA} == set(KINDS)
    assert {e["lang"] for e in DATA} == {"en", "uz"}
    assert len({e["id"] for e in DATA}) == len(DATA)
    assert build_all(per_kind=3, seed=3) == build_all(per_kind=3, seed=3)
    assert sum(DEFAULT_COUNTS.values()) > 1500


def test_structure_and_losses():
    for e in DATA:
        segs = e["segments"]
        assert segs[0]["t"].startswith("<user>") and segs[0]["loss"] == 0
        assert segs[-1]["t"] == "</assistant>\n" and segs[-1]["loss"] == 0
        for s in segs:
            if s["t"].startswith("<result>"):
                assert s["loss"] == 0
            if s["t"].startswith(("<call>", "<map>", "<plan>")):
                assert s["loss"] == 1


def test_every_map_entry_points_at_the_real_def_line_of_what_was_read():
    n = 0
    for e in DATA:
        if not e["kind"].startswith(("map_", "edit_")):
            continue
        seen = {}
        for s in e["segments"]:
            if s["t"].startswith("<result>"):
                for m in re.finditer(r"(?m)^(\d+)[ ]+(.*)$", s["t"]):
                    seen[int(m.group(1))] = m.group(2)
            if s["t"].startswith("<map>"):
                for name, a, z in re.findall(r"(\w+) (\d+)-(\d*)", s["t"]):
                    n += 1
                    assert seen[int(a)].startswith("def %s(" % name), (e["id"], name, a, seen.get(int(a)))
    assert n > 50


def test_edit_changes_the_right_line_and_the_final_sentence_matches_the_run():
    for e in DATA:
        if not e["kind"].startswith(("map_edit", "edit_")):
            continue
        t = text(e)
        calls = re.findall(r"<call> edit\((\S+), (\d+)-(\d+)\) </call>", t)
        assert calls, e["id"]
        res = [s["t"] for s in e["segments"] if s["t"].startswith("<result>")]
        assert not any("error" in r.lower() for r in res), (e["id"], res)
        final = [s for s in e["segments"] if s["loss"] and not s["t"].startswith("<")][-1]["t"]
        last_run = [r for r in res if "Traceback" not in r][-1]
        values = [l.split(" ", 1)[1] for l in last_run.splitlines()[1:-1] if " " in l]
        assert any(v in final for v in values), (e["id"], final, last_run)


def test_a_range_read_is_always_before_the_edit_that_uses_it():
    for e in DATA:
        if e["kind"].startswith("map_edit"):
            t = text(e)
            assert t.index("<map>") < t.index("<call> edit(")
            m = re.search(r"<call> edit\((\S+), (\d+)-(\d+)\)", t)
            assert re.search(r"<call> read\(%s, \d+-\d+\)" % re.escape(m.group(1)), t[:m.start()])


def final_of(e):
    return [s for s in e["segments"] if s["loss"] and not s["t"].startswith("<")][-1]["t"]


def test_search_hit_uses_two_arguments_and_the_answer_comes_from_the_top_result():
    hits = [e for e in DATA if e["kind"] == "search_hit"]
    assert hits
    for e in hits:
        t = text(e)
        assert re.search(r"<call> search\([^)]+, wikipedia\) </call>", t)
        res = re.search(r"<result>\n(.*?)\n</result>", t, re.S).group(1)
        assert res.startswith("1. ") and "en.wikipedia.org/wiki/" in res
        first_snippet = res.split("\n")[1].strip()
        words = set(re.findall(r"[A-Za-z]{5,}", first_snippet))
        assert words & set(re.findall(r"[A-Za-z]{5,}", final_of(e))) or e["lang"] == "uz", (final_of(e), first_snippet)


def test_the_user_names_the_tool_and_alethic_does_exactly_that():
    kinds = {k: [e for e in DATA if e["kind"] == k] for k in ("ask_search", "ask_run", "ask_call")}
    assert all(kinds.values())
    for e in kinds["ask_search"]:
        assert "<call> search(" in text(e)
    for e in kinds["ask_run"]:
        assert re.search(r"<call> shell\(python3? \S+\) </call>", text(e))
    for e in kinds["ask_call"]:
        assert re.search(r"<call> (read|shell)\(", text(e))
    asked_brave = [e for e in kinds["ask_search"] if "brave" in e["segments"][0]["t"]]
    for e in asked_brave:
        assert "not connected" in text(e) and text(e).count("<call> search(") == 2


def test_search_none_gets_the_real_no_results_and_the_answer_does_not_make_a_claim():
    none = [e for e in DATA if e["kind"] == "search_none"]  # real empty result
    assert none
    for e in none:
        t = text(e)
        assert "<result>\nno results\n</result>" in t
        final = [s for s in e["segments"] if s["loss"] and not s["t"].startswith("<")][-1]["t"]
        assert re.search(r"(?i)found nothing|nothing came back|no results|natija yo.q|natija chiqmadi|ayta olmayman|topilmadi|topa olmadim|can.t|cannot|not sure|bilmayman|tasdiqlay|ishonchli", final), final


def test_no_eval_prompt_and_answers_are_varied():
    from tools.evalset.prompts import eval_set
    big = build_all(per_kind=80, seed=4)
    bad = {p["prompt"] for p in eval_set()}
    for e in big:
        assert e["segments"][0]["t"].split("\n")[1] not in bad
    for kind in ("search_hit", "search_none", "ask_run"):
        finals = Counter(re.sub(r"\d+", "N", [s for s in e["segments"] if s["loss"] and not s["t"].startswith("<")][-1]["t"])
                         for e in big if e["kind"] == kind)
        assert len(finals) > 30, (kind, len(finals))
        assert max(finals.values()) / sum(finals.values()) < 0.05


def test_replay_accepts_the_generated_data_and_catches_a_changed_result():
    import copy
    from tools.skilltraces.replay import check
    import json, os
    path = os.path.join("docs", "sft", "record", "wiki_search_cache.json")
    cache = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    sample = [e for e in build_all(per_kind=4, seed=9) if cache or e["kind"] not in ("search_hit", "ask_search", "search_none")]
    for e in sample:
        assert check(e, cache) == [], (e["id"], check(e, cache))
    e = copy.deepcopy(next(e for e in sample if e["kind"] == "map_edit"))
    i = next(i for i, s in enumerate(e["segments"]) if s["t"].startswith("<map>"))
    e["segments"][i]["t"] = e["segments"][i]["t"].replace("-", "-9", 1)
    assert check(e, cache)
    e2 = copy.deepcopy(next(e for e in sample if e["kind"] == "ask_run"))
    j = next(i for i, s in enumerate(e2["segments"]) if s["t"].startswith("<result>"))
    e2["segments"][j]["t"] = "<result>\nwrong\n</result>\n"
    assert check(e2, cache)
