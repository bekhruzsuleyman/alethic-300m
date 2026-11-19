from tools.expand import compose


def smp(i, text="x", flags=None):
    return {"id": i, "segments": [{"t": "<user>\n%s %s\n</user>\n" % (i, text), "loss": 0}, {"t": "answer %s" % i, "loss": 1}],
            "flags": flags or []}


def pool(prefix, n, flags=None):
    return [smp("%s%d.0" % (prefix, i), flags=flags) for i in range(n)]


def f_examples(kind, n, per):
    return [smp("F%s%03d.%d" % (kind[0], i, k), flags=["kind:" + kind]) for i in range(n) for k in range(per)]


def sources():
    return {"hand": pool("h", 100), "identity": pool("i", 10), "T": pool("t", 900), "E": pool("e", 100), "B": pool("b", 800),
            "C": pool("c", 200), "D": pool("d", 5), "P": pool("p", 150),
            "F": f_examples("map", 40, 5) + f_examples("control", 200, 1) + f_examples("small", 100, 1),
            "G": f_examples("map", 30, 6) + f_examples("control", 40, 1)}


def test_total_is_close_to_the_target_and_fixed_sources_are_kept_whole():
    out = compose.compose(sources(), total=1500, seed=1)
    n = sum(len(v) for v in out.values())
    assert 1450 <= n <= 1560
    for name in ("hand", "identity", "C", "D", "P", "E"):
        assert len(out[name]) == len(sources()[name])


def test_f_budget_keeps_the_control_and_small_shares_and_whole_examples():
    out = compose.compose(sources(), total=1500, seed=1, f_budget=100, g_budget=50)
    kinds = [s["flags"][0] for s in out["F"]]
    assert 95 <= len(kinds) <= 110
    assert kinds.count("kind:control") >= 0.3 * len(kinds) and kinds.count("kind:small") >= 0.1 * len(kinds)
    ids = {}
    for s in out["F"]:
        ids.setdefault(s["id"].split(".")[0], []).append(s["id"])
    per_map = [len(v) for k, v in ids.items() if k.startswith("Fm")]
    assert all(c == 5 for c in per_map)                 # a map example is never cut in half


def test_deterministic_and_no_duplicate_texts():
    a = compose.compose(sources(), total=1500, seed=4)
    b = compose.compose(sources(), total=1500, seed=4)
    assert {k: [s["id"] for s in v] for k, v in a.items()} == {k: [s["id"] for s in v] for k, v in b.items()}
    src = sources()
    src["T"] = src["T"] + [dict(src["T"][0], id="dupe")]
    out = compose.compose(src, total=1500, seed=4)
    assert "dupe" not in {s["id"] for s in out["T"]}


def test_short_sources_are_used_in_full_and_report_says_so():
    src = sources()
    src["B"] = pool("b", 30)
    out = compose.compose(src, total=3000, seed=1)
    assert len(out["B"]) == 30
    rep = compose.report(out)
    assert rep["B"] == 30 and rep["total"] == sum(len(v) for v in out.values())
