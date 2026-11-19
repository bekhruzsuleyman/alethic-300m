"""Overnight expansion runner: prompt -> DeepSeek API -> check -> keep. No network code here except call_deepseek.

run() takes any `call(prompt) -> (reply_text, usage_dict)`, so it is tested with a fake API.
Files written under workdir: raw/<kind>_<start>.txt (every reply), accepted_<kind>.txt, built_<kind>.jsonl,
rejects.jsonl, stats.json. A batch whose raw file exists is not called again, so a rerun resumes.
"""
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.expand import pcheck, plan
from tools.sftbuild.plain import build_plain

API = "https://api.deepseek.com"
PER_BATCH = {"T": 15, "E": 14}
RANGES = {"T": (3001, 1300), "E": (4301, 1100)}
BASES = {"T": "deepseek_TE_chats_0801-0880.md", "E": "deepseek_E_pairs_0831-0880.md"}


def plan_batches(kind, start, total, per_batch, seed):
    out, done, n = [], 0, 0
    while done < total:
        count = min(per_batch, total - done)
        s = start + done
        first_pair = 1 + (s - start) // 2
        rows = plan.make_rows(kind, count, s, seed, first_pair)
        out.append({"kind": kind, "start": s, "count": count, "batch": n + 1, "rows": rows,
                    "first_pair": first_pair, "base": BASES[kind]})
        done, n = done + count, n + 1
    return out


def build_prompt(base_text, kind, rows, base_name, batch):
    return base_text.rstrip() + "\n\n---\n\n" + plan.addendum(kind, rows, base_name, batch, STYLE.get(kind))


def _split_examples(text):
    heads = list(pcheck.HEADER.finditer(text))
    out = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out.append((h.group(1), text[h.start():end].strip()))
    return out


_CHECKLIST = re.compile(r"^\d{4} ok[ \t]*\n?", re.M)       # DeepSeek sometimes ends with "4903 ok" lines


def fix_pair_ids(text):
    """DeepSeek often writes both halves of an E pair under the same id (4301, 4301); the second half is id+1."""
    prev = [None]

    def fix(m):
        i = m.group(1)
        if i == prev[0]:
            prev[0] = None
            return m.group(0).replace(i, "%04d" % (int(i) + 1), 1)
        prev[0] = i
        return m.group(0)
    return pcheck.HEADER.sub(fix, text)


STYLE = {}                                                      # {"T": "detailed", "E": "natural"}; empty = the old strict formats


def process_reply(kind, rows, reply, style=None):
    """-> ([(id, example_text)] that passed, {id: [errors]} for the rest)."""
    m = re.search(r"```text\n(.*?)\n```", reply, re.S)
    text = _CHECKLIST.sub("", m.group(1) if m else reply).rstrip()
    if kind == "E":
        text = fix_pair_ids(text)
    res = pcheck.check(kind, rows, text, style or STYLE.get(kind, "strict"))
    ex = dict(_split_examples(text))
    accepted = [(i, ex[i]) for i, errs in res.items() if not errs and i in ex]
    rejects = {i: errs for i, errs in res.items() if errs}
    return accepted, rejects


class Seen:
    """Drops an example whose first user turn is almost the same as one already kept."""

    def __init__(self, threshold=0.7):
        self.sets, self.th = [], threshold

    def add(self, _id, example):
        m = re.search(r"<user>\n(.*?)\n</user>", example, re.S)
        s = set(re.findall(r"\w+", (m.group(1) if m else example).lower()))
        for o in self.sets:
            if s and o and len(s & o) / len(s | o) >= self.th:
                return False
        self.sets.append(s)
        return True


def dedup(kind, acc, seen):
    """Keep what is not a near-duplicate. The two halves of an E pair start with the same user turn on purpose,
    so a pair is checked once (by its first half) and kept or dropped as a whole."""
    if kind != "E":
        return [(i, ex) for i, ex in acc if seen.add(i, ex)]
    kept = []
    for k in range(0, len(acc) - 1, 2):
        if seen.add(*acc[k]):
            kept += [acc[k], acc[k + 1]]
    return kept


class Budget:
    def __init__(self, cap, miss, hit, out):
        self.cap, self.miss, self.hit, self.out, self.spent = cap, miss, hit, out, 0.0
        self._lock = threading.Lock()

    def add(self, usage):
        p, c = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        h = min(usage.get("prompt_cache_hit_tokens", 0), p)
        with self._lock:
            self.spent += ((p - h) * self.miss + h * self.hit + c * self.out) / 1e6

    def over(self):
        return self.spent >= self.cap


def call_deepseek(prompt, key, model, max_tokens=16000, temperature=1.0, retries=4):
    import time
    import requests
    for attempt in range(retries):
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "stream": False, "max_tokens": max_tokens, "temperature": temperature,
                "thinking": {"type": "disabled"}}               # thinking tokens would eat max_tokens
        r = requests.post(API + "/chat/completions", timeout=900, headers={"Authorization": "Bearer " + key}, json=body)
        if r.status_code == 400:                                 # this model may not know the "thinking" field
            body.pop("thinking")
            r = requests.post(API + "/chat/completions", timeout=900, headers={"Authorization": "Bearer " + key}, json=body)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        j = r.json()
        ch = j["choices"][0]
        usage = dict(j.get("usage", {}))
        usage["_finish"] = ch.get("finish_reason")
        usage["_reasoning_chars"] = len(ch["message"].get("reasoning_content") or "")
        return ch["message"]["content"] or "", usage
    raise RuntimeError("DeepSeek API kept failing")


def list_models(key):
    import requests
    r = requests.get(API + "/models", headers={"Authorization": "Bearer " + key}, timeout=60)
    r.raise_for_status()
    return [m["id"] for m in r.json()["data"]]


def run(batches, call, workdir, budget, workers=6, min_batches_for_breaker=8, min_rate=0.25, log=print):
    """Process batches in parallel; returns stats. Stops early on budget or a bad acceptance rate."""
    wd = Path(workdir)
    (wd / "raw").mkdir(parents=True, exist_ok=True)
    lock, seen, state = threading.Lock(), Seen(), {"asked": 0, "accepted": 0, "batches": 0, "stopped": None}
    base_cache = {}
    for kind in {b["kind"] for b in batches}:               # load what earlier runs kept, for dedup
        f = wd / ("accepted_%s.txt" % kind)
        if f.is_file():
            for i, ex in _split_examples(f.read_text(encoding="utf-8")):
                seen.add(i, ex)

    def one(b):
        if state["stopped"]:
            return
        raw = wd / "raw" / ("%s_%d.txt" % (b["kind"], b["start"]))
        if raw.is_file():
            return
        if budget.over():
            state["stopped"] = "budget"
            return
        prompt = build_prompt(b["base_text"], b["kind"], b["rows"], b["base"], b["batch"]) if "base_text" in b else \
            build_prompt(base_cache.get(b["base"], b["base"]), b["kind"], b["rows"], b["base"], b["batch"])
        best = None
        for attempt in range(2):                               # one retry when most of a batch is rejected
            reply, usage = call(prompt)
            budget.add(usage)
            acc, rej = process_reply(b["kind"], b["rows"], reply)
            if best is None or len(acc) > len(best[1]):
                best = (reply, acc, rej, usage)
            if len(acc) >= 0.5 * len(b["rows"]) * (2 if b["kind"] == "E" else 1) or budget.over():
                break
        reply, acc, rej, usage = best
        with lock:
            kept = dedup(b["kind"], acc, seen)
            dropped_dup = [i for i, _ in acc if i not in {k for k, _ in kept}]
            raw.write_text(reply, encoding="utf-8")
            if kept:
                text = "\n\n".join(ex for _, ex in kept) + "\n\n"
                with open(wd / ("accepted_%s.txt" % b["kind"]), "a", encoding="utf-8") as f:
                    f.write(text)
                with open(wd / ("built_%s.jsonl" % b["kind"]), "a", encoding="utf-8") as f:
                    for s in build_plain(text):
                        s["flags"] = (s.get("flags") or []) + ["deepseek_unreviewed"]
                        f.write(json.dumps(s, ensure_ascii=True) + "\n")
            with open(wd / "rejects.jsonl", "a", encoding="utf-8") as f:
                for i, errs in rej.items():
                    f.write(json.dumps({"id": i, "errors": errs}) + "\n")
                for i in dropped_dup:
                    f.write(json.dumps({"id": i, "errors": ["near-duplicate of an earlier example"]}) + "\n")
            state["asked"] += len(b["rows"]) * (2 if b["kind"] == "E" else 1)
            state["accepted"] += len(kept)
            state["batches"] += 1
            log("batch %s %d: kept %d, rejected %d, spent $%.3f" % (b["kind"], b["start"], len(kept), len(rej) + len(dropped_dup), budget.spent))
            if not kept:
                from collections import Counter
                top = Counter(e for errs in rej.values() for e in errs).most_common(4)
                log("  why: %s | finish=%s reasoning_chars=%s completion_tokens=%s reply_chars=%d | starts: %r" % (
                    top, usage.get("_finish"), usage.get("_reasoning_chars"), usage.get("completion_tokens"), len(reply), reply[:160]))
            if state["batches"] >= min_batches_for_breaker and state["accepted"] < min_rate * state["asked"]:
                state["stopped"] = "low acceptance rate (%d of %d)" % (state["accepted"], state["asked"])

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, batches))
    stats = {"accepted": state["accepted"], "asked": state["asked"], "batches": state["batches"],
             "stopped": state["stopped"], "spent_usd": round(budget.spent, 4)}
    (wd / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


def reprocess(batches, raw_dir, wd, styles=None):
    """Re-check every saved raw reply with the current checker. No API call, no cost. Writes the same files as run()."""
    wd = Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    for name in ("accepted_T.txt", "accepted_E.txt", "built_T.jsonl", "built_E.jsonl", "rejects.jsonl"):
        (wd / name).write_text("", encoding="utf-8")
    seen, state = Seen(), {"accepted": 0, "asked": 0, "batches": 0}
    for b in batches:
        raw = Path(raw_dir) / ("%s_%d.txt" % (b["kind"], b["start"]))
        if not raw.is_file():
            continue
        acc, rej = process_reply(b["kind"], b["rows"], raw.read_text(encoding="utf-8"), style=(styles or {}).get(b["kind"], "strict"))
        kept = dedup(b["kind"], acc, seen)
        kept_ids = {k for k, _ in kept}
        if kept:
            text = "\n\n".join(ex for _, ex in kept) + "\n\n"
            with open(wd / ("accepted_%s.txt" % b["kind"]), "a", encoding="utf-8") as f:
                f.write(text)
            with open(wd / ("built_%s.jsonl" % b["kind"]), "a", encoding="utf-8") as f:
                for s in build_plain(text):
                    s["flags"] = (s.get("flags") or []) + ["deepseek_unreviewed"]
                    f.write(json.dumps(s, ensure_ascii=True) + "\n")
        with open(wd / "rejects.jsonl", "a", encoding="utf-8") as f:
            for i, errs in rej.items():
                f.write(json.dumps({"id": i, "errors": errs}) + "\n")
            for i, _ in acc:
                if i not in kept_ids:
                    f.write(json.dumps({"id": i, "errors": ["near-duplicate of an earlier example"]}) + "\n")
        state["asked"] += len(b["rows"]) * (2 if b["kind"] == "E" else 1)
        state["accepted"] += len(kept)
        state["batches"] += 1
    state["stopped"] = None
    (wd / "stats.json").write_text(json.dumps(state), encoding="utf-8")
    return state
