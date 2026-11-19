"""D thinking traces: generate math problems with known answers, send packs to DeepSeek, keep only verified traces."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.deepthink import mathgen
from tools.deepthink.problem import verify as problem_verify
from tools.expand import dcheck

NOTE = ("\n\n## Run instructions\n\nThis is an API call, so do NOT reply READY and skip Phase A. Do Phase B now: reply with the "
        "traces for ALL problems below, in order, in ONE ```text block, then the `NNNN ok` lines.\n\n")


def make_problems(n, start=1):
    """n problems, ids zero-padded from `start`, families rotating; same call -> same problems."""
    out, seen = [], set()
    for k in range(n):
        family, seed = mathgen.FAMILIES[k % len(mathgen.FAMILIES)], 1000 + (start - 1 + k) // len(mathgen.FAMILIES)
        while True:
            p = mathgen.generate(family, seed)
            if p.statement not in seen:
                break
            seed += 7919
        seen.add(p.statement)
        out.append({"id": "%04d" % (start + k), "family": family, "seed": seed, "statement": p.statement,
                    "answer": str(p.spec["answer"])})
    return out


def _verifier(probs):
    by_id = {p["id"]: p for p in probs}

    def verify(qid, got):
        p = by_id[qid]
        if got.strip() == p["answer"].strip():
            return True
        return problem_verify(mathgen.generate(p["family"], p["seed"]), got, ".")[0]
    return verify


def build_prompt(base_text, probs, batch_no=1):
    pack = "\n\n".join("### problem %s | math | en\n<statement>\n%s\n</statement>" % (p["id"], p["statement"]) for p in probs)
    return base_text.rstrip() + NOTE + "```text\n" + pack + "\n```"


def process_reply(probs, reply):
    text = dcheck.clean(reply)
    res = dcheck.check({p["id"]: p["answer"] for p in probs}, text, _verifier(probs))
    parsed = dcheck.parse(text)
    acc = [(p["id"], "### trace %s\n%s" % (p["id"], parsed[p["id"]])) for p in probs if not res[p["id"]]]
    return acc, {i: e for i, e in res.items() if e}


def plan_batches(probs, per_batch=10):
    return [{"kind": "D", "start": int(probs[i]["id"]), "probs": probs[i:i + per_batch]} for i in range(0, len(probs), per_batch)]


def run(batches, call, workdir, budget, workers=4, min_batches_for_breaker=8, min_rate=0.25, base_text="BASE", log=print):
    wd = Path(workdir)
    (wd / "raw").mkdir(parents=True, exist_ok=True)
    lock, state = threading.Lock(), {"asked": 0, "accepted": 0, "batches": 0, "stopped": None}

    def one(b):
        raw = wd / "raw" / ("D_%d.txt" % b["start"])
        if state["stopped"] or raw.is_file():
            return
        if budget.over():
            state["stopped"] = "budget"
            return
        prompt, best = build_prompt(base_text, b["probs"], b["start"]), None
        for _ in range(2):
            reply, usage = call(prompt)
            budget.add(usage)
            acc, rej = process_reply(b["probs"], reply)
            if best is None or len(acc) > len(best[1]):
                best = (reply, acc, rej)
            if len(acc) >= 0.5 * len(b["probs"]) or budget.over():
                break
        reply, acc, rej = best
        by_id = {p["id"]: p for p in b["probs"]}
        with lock:
            raw.write_text(reply, encoding="utf-8")
            with open(wd / "accepted_D.txt", "a", encoding="utf-8") as f:
                f.write("".join(t + "\n\n" for _, t in acc))
            with open(wd / "built_D.jsonl", "a", encoding="utf-8") as f:
                for i, t in acc:
                    f.write(dcheck.dumps(dcheck.build_sample(i, by_id[i]["statement"], t)) + "\n")
            with open(wd / "rejects.jsonl", "a", encoding="utf-8") as f:
                for i, e in rej.items():
                    f.write(json.dumps({"id": i, "errors": e}) + "\n")
            state["asked"] += len(b["probs"])
            state["accepted"] += len(acc)
            state["batches"] += 1
            log("batch D %d: kept %d of %d, spent $%.3f" % (b["start"], len(acc), len(b["probs"]), budget.spent))
            if state["batches"] >= min_batches_for_breaker and state["accepted"] < min_rate * state["asked"]:
                state["stopped"] = "low acceptance (%d of %d)" % (state["accepted"], state["asked"])

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, batches))
    stats = {"accepted": state["accepted"], "asked": state["asked"], "batches": state["batches"],
             "stopped": state["stopped"], "spent_usd": round(budget.spent, 4)}
    if state["batches"] or not (wd / "stats.json").is_file():            # a run that skipped everything must not wipe old stats
        (wd / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


def reprocess(batches, raw_dir, wd):
    """Re-check every saved raw reply with the current checker. No API call, no cost."""
    wd = Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    for name in ("accepted_D.txt", "built_D.jsonl", "rejects.jsonl"):
        (wd / name).write_text("", encoding="utf-8")
    state = {"accepted": 0, "asked": 0, "batches": 0}
    for b in batches:
        f = Path(raw_dir) / ("D_%d.txt" % b["start"])
        if not f.is_file():
            continue
        acc, rej = process_reply(b["probs"], f.read_text(encoding="utf-8"))
        by_id = {p["id"]: p for p in b["probs"]}
        with open(wd / "accepted_D.txt", "a", encoding="utf-8") as o:
            o.write("".join(t + "\n\n" for _, t in acc))
        with open(wd / "built_D.jsonl", "a", encoding="utf-8") as o:
            for i, t in acc:
                o.write(dcheck.dumps(dcheck.build_sample(i, by_id[i]["statement"], t)) + "\n")
        with open(wd / "rejects.jsonl", "a", encoding="utf-8") as o:
            for i, e in rej.items():
                o.write(json.dumps({"id": i, "errors": e}) + "\n")
        state["asked"] += len(b["probs"])
        state["accepted"] += len(acc)
        state["batches"] += 1
    (wd / "stats.json").write_text(json.dumps(state), encoding="utf-8")
    return state
