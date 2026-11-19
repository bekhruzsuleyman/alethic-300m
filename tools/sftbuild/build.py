"""python -m tools.sftbuild.build --mode {plain,terminal,session} --src FILE --out PREFIX"""
import argparse
import json
import sys

from tools.sftbuild import budget
from tools.sftbuild.plain import build_plain
from tools.sftbuild.session import build_session
from tools.sftbuild.terminal import build_terminal


def _loss_share(sample):
    total = sum(len(s["t"]) for s in sample["segments"])
    return round(sum(len(s["t"]) for s in sample["segments"] if s["loss"]) / total, 3) if total else 0.0


def _preview(samples):
    lines = []
    for smp in samples:
        lines.append("=" * 20 + " " + smp["id"])
        for seg in smp["segments"]:
            for ln in seg["t"].rstrip("\n").split("\n"):
                lines.append(ln if seg["loss"] else "| " + ln)
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sftbuild")
    ap.add_argument("--mode", required=True, choices=["plain", "terminal", "session"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", default="/tmp/sftbuild_work")
    ap.add_argument("--traceback-json", help='session only: {"files": {...}, "cmd": "..."} for {TRACEBACK}')
    ap.add_argument("--search-cache", help="terminal only: JSON file of recorded search results")
    ap.add_argument("--search-mode", default="live", choices=["live", "record", "replay"],
                    help="live: fetch; record: fetch and save to the cache; replay: cache only, no network")
    ap.add_argument("--skip-budget", action="store_true", help="do not count tokens (recorded in the report)")
    a = ap.parse_args(argv)

    text = open(a.src, encoding="utf-8").read()
    if a.mode == "plain":
        samples = build_plain(text)
    elif a.mode == "terminal":
        from tools.sftbuild.search import Searcher
        samples = build_terminal(text, a.workdir, searcher=Searcher(cache_path=a.search_cache, mode=a.search_mode))
    else:
        tb = None
        if a.traceback_json:
            spec = json.load(open(a.traceback_json))
            tb = (spec["files"], spec["cmd"])
        samples = build_session(text, a.workdir, traceback=tb)

    over = []
    if not a.skip_budget:
        try:
            over = [s["id"] for s in samples if budget.over_budget(s)]
        except budget.BudgetError as exc:
            print("error:", exc, file=sys.stderr)
            return 2
    with open(a.out + ".jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=True) + "\n")
    with open(a.out + "_preview.txt", "w", encoding="utf-8") as f:
        f.write(_preview(samples))
    report = {
        "samples": len(samples),
        "budget": "skipped" if a.skip_budget else "checked",
        "over_budget_ids": over,
        "flags": {s["id"]: s["flags"] for s in samples if s.get("flags")},
        "loss_share_by_sample": {s["id"]: _loss_share(s) for s in samples},
    }
    with open(a.out + "_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
