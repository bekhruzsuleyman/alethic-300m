"""Mix the checked sources into one SFT set of about `total` samples (the 5k set of DATASET_PLAN).
Small, rare, hand-made and thinking/project sources are kept whole; F and G are cut to their budgets by whole
examples (a map example is never cut in half); the rest of the room is filled with T chats and B code."""
import hashlib
import random

FIXED = ("hand", "identity", "E", "C", "D", "P")
F_SHARE = {"map": 0.50, "control": 0.35, "small": 0.15}
G_SHARE = {"map": 0.80, "control": 0.20}
T_SHARE = 0.55


def _key(s):
    return hashlib.sha1("".join(x["t"] for x in s["segments"]).encode("utf-8")).hexdigest()


def _dedupe(samples, seen):
    out = []
    for s in samples:
        k = _key(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def _kind(s):
    return next((f[5:] for f in s.get("flags", []) if f.startswith("kind:")), "map")


def _by_examples(samples, shares, budget, rng):
    groups = {}
    for s in samples:
        groups.setdefault(s["id"].split(".")[0], []).append(s)
    out = []
    for kind, share in shares.items():
        ids = sorted(i for i, g in groups.items() if _kind(g[0]) == kind)
        rng.shuffle(ids)
        got, want = 0, round(budget * share)
        for i in ids:
            if got >= want:
                break
            out += groups[i]
            got += len(groups[i])
    return out


def compose(sources, total=5000, seed=7, f_budget=417, g_budget=250):
    rng = random.Random(seed)
    seen = set()
    out = {name: _dedupe(sources.get(name, []), seen) for name in FIXED}
    out["F"] = _by_examples(_dedupe(sources.get("F", []), seen), F_SHARE, f_budget, rng)
    out["G"] = _by_examples(_dedupe(sources.get("G", []), seen), G_SHARE, g_budget, rng)
    pools = {n: _dedupe(sources.get(n, []), seen) for n in ("T", "B")}
    for p in pools.values():
        rng.shuffle(p)
    room = max(0, total - sum(len(v) for v in out.values()))
    want_t = min(len(pools["T"]), round(room * T_SHARE))
    want_b = min(len(pools["B"]), room - want_t)
    want_t = min(len(pools["T"]), room - want_b)
    out["T"], out["B"] = pools["T"][:want_t], pools["B"][:want_b]
    return out


def report(out):
    rep = {name: len(v) for name, v in out.items()}
    rep["total"] = sum(rep.values())
    return rep
