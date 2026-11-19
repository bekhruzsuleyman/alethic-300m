"""Stage B data: GOLD repeated, scripted error and program traces, identity, and a little stage A chat to keep the model's general skill.

GOLD copies keep their ids on purpose: the trainer splits train/validation by id, so every copy of an example lands on the
same side and the validation loss is never measured on text the model has trained on.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

NOT_GOLD = ("error_traces", "prog_traces", "term_traces", "math_traces", "skill_traces", "gemini_sessions", "identity_", "plain_batch1_", "plain_batch2_")   # the batch1/2 files are duplicates of the cleaned set


def _read(path):
    """One JSON record per line. Iterate the file, never str.splitlines(): that also cuts at U+2028, U+0085 and \\x1c inside text."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def take_first(src, dst, n: int) -> int:
    """Copy the first n whole lines of a JSONL file; returns how many were copied."""
    count = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            if count >= n:
                break
            if line.strip():
                fout.write(line if line.endswith("\n") else line + "\n")
                count += 1
    return count


def build_stage_b(built_dir, replay_path, gold_repeat: int = 3, replay_n: int = 2000, seed: int = 7) -> dict:
    built = Path(built_dir)
    gold = [s for f in sorted(built.glob("*.jsonl")) if not f.name.startswith(NOT_GOLD) for s in _read(f)]
    identity = _read(built / "identity_en_uz_built.jsonl") + _read(built / "identity_variants.jsonl")
    errors = _read(built / "error_traces.jsonl")
    pool = _read(replay_path)
    replay = random.Random(seed).sample(pool, min(replay_n, len(pool)))
    mix = {"gold": gold * gold_repeat, "identity": identity, "errors": errors, "replay": replay}
    if (built / "prog_traces.jsonl").is_file():             # scripted write-and-run, edit and one-liner traces: once, they are already varied
        mix["progs"] = _read(built / "prog_traces.jsonl")
    if (built / "term_traces.jsonl").is_file():             # plain-words file and folder commands (save, read, make folder, ...): once
        mix["terms"] = _read(built / "term_traces.jsonl")
    if (built / "gemini_sessions.jsonl").is_file():         # long real sessions written by Gemini, checked by replay.py: twice (few, rich)
        mix["sessions"] = _read(built / "gemini_sessions.jsonl") * 2
    if (built / "skill_traces.jsonl").is_file():            # edit/map/search and "search/run/call ..." commands: once
        mix["skills"] = _read(built / "skill_traces.jsonl")
    if (built / "math_traces.jsonl").is_file():             # number questions answered by asking python: once
        mix["maths"] = _read(built / "math_traces.jsonl")
    return mix


def write_mix(mix: dict, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, samples in mix.items():
        (out / ("%s.jsonl" % name)).write_text("".join(json.dumps(s, ensure_ascii=True) + "\n" for s in samples), encoding="utf-8")


def report(mix: dict) -> dict:
    total = sum(len(v) for v in mix.values())
    return {"total": total, "counts": {k: len(v) for k, v in mix.items()}, "share": {k: len(v) / total for k, v in mix.items()}}
