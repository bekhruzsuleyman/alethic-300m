"""B code examples: task + plan line + code; the code is run against DeepSeek's own asserts and only passing code is kept."""
import json
import os
import re
import subprocess
import sys
import tempfile

HEAD = re.compile(r"^### code (\d{4}) \| (py|js) \| (en|uz)[ \t]*$", re.M)
PARTS = [re.compile(p, re.S) for p in (r"<task>\s*(.+?)\s*</task>", r"<plan>\s*(.+?)\s*</plan>",
                                       r"```(python|javascript)[ \t]*\n(.*?)\n```", r"<tests>\s*(.+?)\s*</tests>")]


def _parts(body):
    """(task, plan, fence, code, tests) found anywhere in the body, or None; extra tags such as <topic> are ignored."""
    task, plan, code, tests = (p.search(body) for p in PARTS)
    if not (task and plan and code and tests):
        return None
    return task.group(1), plan.group(1), code.group(1), code.group(2), tests.group(1)


LANG = {"py": ("python", [sys.executable, "-I"], ".py"), "js": ("javascript", ["node"], ".js")}


def parse(text):
    heads = list(HEAD.finditer(text))
    return {h.group(1): (h.group(2), h.group(3), text[h.end():(heads[i + 1].start() if i + 1 < len(heads) else len(text))].strip())
            for i, h in enumerate(heads)}


def _run(lang, code, tests, timeout=5):
    _name, cmd, ext = LANG[lang]
    prog = code + "\n" + tests + "\n"
    if lang == "js":                                         # console.assert only prints; make a failed one exit with an error
        prog = "const __a = console.assert; console.assert = (c, ...r) => { if (!c) { throw new Error('assert'); } };\n" + prog
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "t" + ext)
        open(f, "w", encoding="utf-8").write(prog)
        try:
            p = subprocess.run(cmd + [f], cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "timeout (code ran longer than %d s)" % timeout
    return None if p.returncode == 0 else "tests failed: " + (p.stderr.strip().splitlines() or ["?"])[-1][:100]


def _check_one(lang, ulang, body):
    parts = _parts(body)
    if not parts:
        return ["format: <task>, <plan>, one code fence, <tests>"]
    task, plan, fence, code, tests = parts
    errs = []
    if fence != LANG[lang][0]:
        errs.append("code fence does not match the language")
    if "assert" not in tests:
        errs.append("tests have no assert")
    if not errs:
        err = _run(lang, code, tests)
        if err:
            errs.append(err)
    return errs


def check(text):
    return {i: _check_one(l, u, b) for i, (l, u, b) in parse(text).items()}


def build_sample(qid, text):
    lang, _u, body = parse(text)[qid]
    task, plan, fence, code, _tests = _parts(body)
    return {"id": "B" + qid,
            "segments": [{"t": "<user>\n%s\n</user>\n" % task, "loss": 0}, {"t": "<assistant>\n", "loss": 0},
                         {"t": "<plan>\n%s\n</plan>\n```%s\n%s\n```\n" % (plan, fence, code), "loss": 1},
                         {"t": "</assistant>\n", "loss": 0}],
            "flags": ["deepseek_unreviewed"]}


def dumps(sample):
    return json.dumps(sample, ensure_ascii=True)


import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOPICS = ["strings", "lists and arrays", "dictionaries and maps", "sorting", "searching", "recursion", "math helpers",
          "dates and time", "parsing text", "stacks and queues", "matrix and grids", "prime numbers", "word counting",
          "number formatting", "validation", "bit tricks", "run-length encoding", "palindromes", "anagrams", "unit conversion",
          "simple classes", "iterators and generators", "binary numbers", "roman numerals", "caesar cipher", "intervals",
          "two pointers", "prefix sums", "gcd and lcm", "fizzbuzz variants", "file path handling", "csv line parsing",
          "balanced brackets", "linked list helpers", "memoization", "dynamic programming basics", "set operations",
          "string compression", "temperature and money", "min and max problems"]
LEVELS = ["easy"] * 35 + ["medium"] * 35 + ["hard"] * 30
NOTE = ("\n\n## Run instructions\n\nThis is an API call, so do NOT reply READY. Write one example for EVERY row below, in order, "
        "in ONE ```text block. Each example has exactly this shape (header, then these four parts):\n\n"
        "### code NNNN | py | en\n<task>\nthe user's request, one or two sentences, in the language of the header (en or uz)\n</task>\n"
        "<plan>\none line, what the code does\n</plan>\n```python\ncomplete, serious, production-quality code for the task\n```\n<tests>\n"
        "a few plain assert lines that call the code and pass (JavaScript: console.assert)\n</tests>\n\n"
        "The language code in the header (py or js) and the fence (python or javascript) must match the row. Code must run as written, "
        "no input(), no files, no network, no printing needed. Vary the task wording, function names and style; do not repeat "
        "tasks. The difficulty level sets how deep the task is: easy = one small function; medium = a few functions or a small class with edge cases; hard = a real algorithm, data structure or mini-module with several interacting parts and tricky edge cases. Write serious, complete code like a senior engineer, with real edge-case handling; never toy, trivial or padded code. Test the edge cases too. No greetings, no filler, no explanations outside the parts.\n\n")


def make_rows(n, start=1, seed=7):
    rng = random.Random(seed)
    out = []
    for k in range(n):
        out.append({"id": "%04d" % (start + k), "lang": "js" if rng.random() < 0.22 else "py",
                    "ulang": "uz" if rng.random() < 0.25 else "en", "topic": rng.choice(TOPICS), "level": rng.choice(LEVELS)})
    return out


def build_prompt(base, rows):
    table = "\n".join("code %s | %s | %s | topic: %s | level: %s" % (r["id"], r["lang"], r["ulang"], r["topic"], r["level"]) for r in rows)
    return base.rstrip() + NOTE + "Rows:\n```text\n" + table + "\n```"


def plan_batches(rows, per_batch=10):
    return [{"kind": "B", "start": int(rows[i]["id"]), "rows": rows[i:i + per_batch]} for i in range(0, len(rows), per_batch)]


def process_reply(reply):
    blocks = re.findall(r"```text\n(.*?)\n```(?=\s*(?:```text|\d{4} ok|$))", reply, re.S)   # one block, or one block per example
    text = "\n\n".join(blocks) if blocks else reply
    res, parsed = check(text), parse(text)
    heads = {i: h for i, h in zip(parsed, HEAD.findall(text))}
    acc = [(i, "### code %s | %s | %s\n%s" % (i, parsed[i][0], parsed[i][1], parsed[i][2])) for i in parsed if not res[i]]
    return acc, {i: e for i, e in res.items() if e}


def run(batches, call, workdir, budget, workers=4, min_batches_for_breaker=8, min_rate=0.25, base_text="BASE", log=print):
    wd = Path(workdir)
    (wd / "raw").mkdir(parents=True, exist_ok=True)
    lock, state = threading.Lock(), {"asked": 0, "accepted": 0, "batches": 0, "stopped": None}

    def one(b):
        raw = wd / "raw" / ("B_%d.txt" % b["start"])
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
            with open(wd / "accepted_B.txt", "a", encoding="utf-8") as f:
                f.write("".join(t + "\n\n" for _, t in acc))
            with open(wd / "built_B.jsonl", "a", encoding="utf-8") as f:
                for i, t in acc:
                    f.write(dumps(build_sample(i, t)) + "\n")
            with open(wd / "rejects.jsonl", "a", encoding="utf-8") as f:
                for i, e in rej.items():
                    f.write(json.dumps({"id": i, "errors": e}) + "\n")
            state["asked"] += len(b["rows"])
            state["accepted"] += len(acc)
            state["batches"] += 1
            log("batch B %d: kept %d of %d, spent $%.3f" % (b["start"], len(acc), len(b["rows"]), budget.spent))
            if state["batches"] >= min_batches_for_breaker and state["accepted"] < min_rate * state["asked"]:
                state["stopped"] = "low acceptance (%d of %d)" % (state["accepted"], state["asked"])

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, batches))
    stats = {"accepted": state["accepted"], "asked": state["asked"], "batches": state["batches"],
             "stopped": state["stopped"], "spent_usd": round(budget.spent, 4)}
    if state["batches"] or not (wd / "stats.json").is_file():
        (wd / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


def reprocess(raw_dir, wd):
    """Re-check every saved raw reply with the current checker. No API call, no cost."""
    wd = Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    for name in ("accepted_B.txt", "built_B.jsonl", "rejects.jsonl"):
        (wd / name).write_text("", encoding="utf-8")
    state = {"accepted": 0, "asked": 0, "batches": 0}
    for f in sorted(Path(raw_dir).glob("B_*.txt"), key=lambda p: int(p.stem.split("_")[1])):
        acc, rej = process_reply(f.read_text(encoding="utf-8"))
        with open(wd / "accepted_B.txt", "a", encoding="utf-8") as o:
            o.write("".join(t + "\n\n" for _, t in acc))
        with open(wd / "built_B.jsonl", "a", encoding="utf-8") as o:
            for i, t in acc:
                o.write(dumps(build_sample(i, t)) + "\n")
        with open(wd / "rejects.jsonl", "a", encoding="utf-8") as o:
            for i, e in rej.items():
                o.write(json.dumps({"id": i, "errors": e}) + "\n")
        state["asked"] += len(acc) + len(rej)
        state["accepted"] += len(acc)
        state["batches"] += 1
    (wd / "stats.json").write_text(json.dumps(state), encoding="utf-8")
    return state
