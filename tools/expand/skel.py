"""P projects (tool use in a real sandbox): DeepSeek writes skeletons, tools.sftbuild runs every step for real,
and only skeletons that build with the expected results become samples (real tool output, real errors)."""
import json
import random
import re
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.sftbuild.terminal import build_terminal

STACKS = ["python"] * 4 + ["javascript"] * 3 + ["rust", "go", "c", "c", "typescript", "html_css", "python+node", "rust+python", "go+python", "python+c", "node+python"]
TOPICS = ["word frequency counter", "temperature table", "prime sieve", "binary search", "roman numerals", "tokenizer for a fixed line", "matrix multiplication",
          "string reverse and palindrome check", "LRU cache with fixed operations", "CSV column sum", "JSON pretty printer", "shopping cart total", "stack class",
          "queue class", "fibonacci table", "gcd and lcm table", "bank account class with fixed deposits", "temperature statistics", "anagram grouping",
          "balanced brackets check", "run-length encoding", "caesar cipher", "bubble sort with fixed data", "merge sort", "linked list class", "binary tree depth",
          "bit counting", "base conversion table", "leap year table", "FizzBuzz variant", "unit converter table", "simple state machine", "text wrap to width 20",
          "histogram printed with stars", "pascal triangle", "collatz steps", "perfect numbers under 500", "string compression", "vowel counter", "date difference in days"]
NOTE = ("\n\n## Run instructions\n\nThis is an API call, so do NOT reply READY and skip the work plan. Write one skeleton for EVERY scenario row below, in order, "
        "in the exact format above. Put each skeleton in its own ```text block. Print no self-check lines.\n\n")


def make_rows(n, start=2001, seed=7):
    rng = random.Random(seed)
    rows = []
    for k in range(n):
        stack = rng.choice(STACKS)
        rows.append({"id": "%04d" % (start + k), "lang": rng.choice(["en", "en", "uz", "mix"]), "stack": stack,
                     "kind": "project_fix" if rng.random() < 0.34 else "project", "topic": rng.choice(TOPICS)})
    return rows


def build_prompt(base, rows):
    table = "\n".join("%s %s   %-13s  %-11s  %s" % (r["id"], r["lang"], r["stack"], r["kind"], r["topic"]) for r in rows)
    return base.rstrip() + NOTE + "Scenarios:\n```text\nid   lang stack          kind         topic\n" + table + "\n```"


def plan_batches(rows, per_batch=6):
    return [{"kind": "P", "start": int(rows[i]["id"]), "rows": rows[i:i + per_batch]} for i in range(0, len(rows), per_batch)]


def _blocks(reply):
    text = "\n\n".join(re.findall(r"```text\n(.*?)\n```", reply, re.S)) or reply
    return [b.strip() for b in re.split(r"\n(?=### ex )", text) if b.strip().startswith("### ex")]


def process_reply(reply):
    """-> (accepted [(id, skeleton, samples)], rejected {id: [reason]})"""
    ok, rej = [], {}
    for b in _blocks(reply):
        m = re.match(r"### ex (\d+)", b)
        eid = m.group(1)
        wd = tempfile.mkdtemp(prefix="skel_")
        try:
            samples = build_terminal(b + "\n", wd)
            if not samples:
                raise ValueError("no samples")
            ok.append((eid, b, samples))
        except Exception as exc:                                  # a skeleton that does not run as written is dropped
            rej[eid] = [str(exc)[:160]]
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    return ok, rej


def _write(wd, ok, rej):
    with open(wd / "accepted_P.txt", "a", encoding="utf-8") as f:
        f.write("".join(b + "\n\n" for _, b, _ in ok))
    with open(wd / "built_P.jsonl", "a", encoding="utf-8") as f:
        for _, _, samples in ok:
            for s in samples:
                s["flags"] = (s.get("flags") or []) + ["deepseek_unreviewed", "real_tool_results"]
                f.write(json.dumps(s, ensure_ascii=True) + "\n")
    with open(wd / "rejects.jsonl", "a", encoding="utf-8") as f:
        for i, e in rej.items():
            f.write(json.dumps({"id": i, "errors": e}) + "\n")


def run(batches, call, workdir, budget, workers=4, min_batches_for_breaker=8, min_rate=0.2, base_text="BASE", log=print):
    wd = Path(workdir)
    (wd / "raw").mkdir(parents=True, exist_ok=True)
    lock, state = threading.Lock(), {"asked": 0, "accepted": 0, "batches": 0, "stopped": None}

    def one(b):
        raw = wd / "raw" / ("P_%d.txt" % b["start"])
        if state["stopped"] or raw.is_file():
            return
        if budget.over():
            state["stopped"] = "budget"
            return
        reply, usage = call(build_prompt(base_text, b["rows"]))
        budget.add(usage)
        ok, rej = process_reply(reply)
        with lock:
            raw.write_text(reply, encoding="utf-8")
            _write(wd, ok, rej)
            state["asked"] += len(b["rows"])
            state["accepted"] += len(ok)
            state["batches"] += 1
            log("batch P %d: kept %d of %d, spent $%.3f" % (b["start"], len(ok), len(b["rows"]), budget.spent))
            if state["batches"] >= min_batches_for_breaker and state["accepted"] < min_rate * state["asked"]:
                state["stopped"] = "low acceptance (%d of %d)" % (state["accepted"], state["asked"])

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, batches))
    stats = {"accepted": state["accepted"], "asked": state["asked"], "batches": state["batches"],
             "stopped": state["stopped"], "spent_usd": round(budget.spent, 4)}
    if state["batches"] or not (wd / "stats.json").is_file():
        (wd / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats
