"""C thinking examples: a question, a thinking block, the final answer; a python <verify> snippet must print the answer value."""
import json
import random
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.expand import dcheck

HEAD = re.compile(r"^### think (\d{4}) \| .+? \| (en|uz)[ \t]*$", re.M)
PARTS = [re.compile(p, re.S) for p in (r"<question>\s*(.+?)\s*</question>", r"<thinking>\s*(.+?)\s*</thinking>",
                                       r"<answer>\s*(.+?)\s*</answer>", r"<verify>\s*(.+?)\s*</verify>")]
TOPICS = ["arithmetic word problem", "percentages and discounts", "probability with coins or dice", "algebra: solve for x", "geometry: area and perimeter",
          "logic puzzle with a unique answer", "unit conversion", "time and schedule", "ratios and proportions", "counting arrangements",
          "average and totals", "speed, distance and time", "money and change", "simple interest and growth", "number patterns and sequences",
          "divisibility and remainders", "planning a function with edge cases", "complexity of a loop (exact count of steps)",
          "ages and relations", "work and rate", "set counting (inclusion-exclusion)", "base conversion", "scheduling with constraints"]
LEVELS = ["easy"] * 35 + ["medium"] * 35 + ["hard"] * 30
NOTE = ("\n\n## Run instructions\n\nThis is an API call, so do NOT reply READY. Write one example for EVERY row below, in order, in ONE ```text "
        "block. Each example has exactly this shape:\n\n### think NNNN | topic | en\n<question>\nthe user's question in the language of the header (en or uz), "
        "self-contained, with all numbers\n</question>\n<thinking>\nplain numbered steps with the arithmetic written out using `=`, then a `check:` line "
        "(English, no filler; the depth follows the level: easy = 1 to 3 short steps, medium = a few steps, hard = a careful multi-step solution)\n</thinking>\n"
        "<answer>\nthe final answer as a short, clear sentence in the language of the question\n</answer>\n<verify>\nplain python (no input, no files) that "
        "computes the answer independently and prints ONLY the final value, written exactly as it appears in the answer sentence\n</verify>\n\n"
        "Every number must be exactly correct. The question must have one definite answer. Vary the wording and the scenario.\n\n")


def make_rows(n, start=1, seed=7):
    rng = random.Random(seed)
    return [{"id": "%04d" % (start + k), "topic": rng.choice(TOPICS), "level": rng.choice(LEVELS),
             "lang": "uz" if rng.random() < 0.25 else "en"} for k in range(n)]


def build_prompt(base, rows):
    table = "\n".join("think %s | %s | %s | level: %s" % (r["id"], r["topic"], r["lang"], r["level"]) for r in rows)
    return base.rstrip() + NOTE + "Rows:\n```text\n" + table + "\n```"


def plan_batches(rows, per_batch=10):
    return [{"kind": "C", "start": int(rows[i]["id"]), "rows": rows[i:i + per_batch]} for i in range(0, len(rows), per_batch)]


def parse(text):
    heads = list(HEAD.finditer(text))
    return {h.group(1): (h.group(2), text[h.end():(heads[i + 1].start() if i + 1 < len(heads) else len(text))].strip())
            for i, h in enumerate(heads)}


def _parts(body):
    found = [p.search(body) for p in PARTS]
    return tuple(m.group(1) for m in found) if all(found) else None


def _norm(s):
    return re.sub(r"[,\s]", "", s.lower())


def _check_one(body):
    parts = _parts(body)
    if not parts:
        return ["format: <question>, <thinking>, <answer>, <verify>"]
    _q, think, answer, verify = parts
    errs = ["wrong arithmetic: " + c for c in dcheck.bad_chains(think)]
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "v.py"
        f.write_text(verify + "\n", encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True, text=True, timeout=5)
        except subprocess.TimeoutExpired:
            return errs + ["verify code timed out"]
    out = p.stdout.strip()
    if p.returncode != 0 or not out:
        errs.append("verify code failed or printed nothing")
    elif _norm(out.splitlines()[-1]) not in _norm(answer):
        errs.append("verify result '%s' is not in the answer" % out.splitlines()[-1][:40])
    return errs


def check(text):
    return {i: _check_one(b) for i, (_l, b) in parse(text).items()}


def build_sample(qid, text):
    lang, body = parse(text)[qid]
    q, think, answer, _v = _parts(body)
    return {"id": "C" + qid,
            "segments": [{"t": "<user>\n%s\n</user>\n" % q, "loss": 0}, {"t": "<assistant>\n", "loss": 0},
                         {"t": "<thinking>\n%s\n</thinking>\n%s\n" % (think, answer), "loss": 1}, {"t": "</assistant>\n", "loss": 0}],
            "flags": ["deepseek_unreviewed"]}


def process_reply(reply):
    blocks = re.findall(r"```text\n(.*?)\n```(?=\s*(?:```text|\d{4} ok|$))", reply, re.S)
    text = "\n\n".join(blocks) if blocks else reply
    res, parsed = check(text), parse(text)
    acc = [(i, "### think %s | x | %s\n%s" % (i, parsed[i][0], parsed[i][1])) for i in parsed if not res[i]]
    return acc, {i: e for i, e in res.items() if e}


def _write(wd, acc, rej):
    with open(wd / "accepted_C.txt", "a", encoding="utf-8") as f:
        f.write("".join(t + "\n\n" for _, t in acc))
    with open(wd / "built_C.jsonl", "a", encoding="utf-8") as f:
        for i, t in acc:
            f.write(dcheck.dumps(build_sample(i, t)) + "\n")
    with open(wd / "rejects.jsonl", "a", encoding="utf-8") as f:
        for i, e in rej.items():
            f.write(json.dumps({"id": i, "errors": e}) + "\n")


def run(batches, call, workdir, budget, workers=4, min_batches_for_breaker=8, min_rate=0.25, base_text="BASE", log=print):
    wd = Path(workdir)
    (wd / "raw").mkdir(parents=True, exist_ok=True)
    lock, state = threading.Lock(), {"asked": 0, "accepted": 0, "batches": 0, "stopped": None}

    def one(b):
        raw = wd / "raw" / ("C_%d.txt" % b["start"])
        if state["stopped"] or raw.is_file():
            return
        if budget.over():
            state["stopped"] = "budget"
            return
        reply, usage = call(build_prompt(base_text, b["rows"]))
        budget.add(usage)
        acc, rej = process_reply(reply)
        with lock:
            raw.write_text(reply, encoding="utf-8")
            _write(wd, acc, rej)
            state["asked"] += len(b["rows"])
            state["accepted"] += len(acc)
            state["batches"] += 1
            log("batch C %d: kept %d of %d, spent $%.3f" % (b["start"], len(acc), len(b["rows"]), budget.spent))
            if state["batches"] >= min_batches_for_breaker and state["accepted"] < min_rate * state["asked"]:
                state["stopped"] = "low acceptance (%d of %d)" % (state["accepted"], state["asked"])

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, batches))
    stats = {"accepted": state["accepted"], "asked": state["asked"], "batches": state["batches"],
             "stopped": state["stopped"], "spent_usd": round(budget.spent, 4)}
    if state["batches"] or not (wd / "stats.json").is_file():
        (wd / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats
