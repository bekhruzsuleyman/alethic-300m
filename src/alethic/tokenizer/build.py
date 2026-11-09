"""`tokenizer-build`: sample -> train -> eval -> freeze, each stage skipped if its output exists.

Layout under $ALETHIC_DATA_DIR:
  raw/<source>/...                       downloaded parquet (reused by the M2 data build)
  tokenizer/work/streams/                one finished file set per sample stream
  tokenizer/work/sample.txt              merged, shuffled training sample
  tokenizer/alethic-tokenizer-v1/        .model, .vocab, report.json, tokenizer.yaml, FROZEN.json
  tokenizer/specials.json                created at freeze
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from alethic.config import data_dir, resolve_path, write_json_atomic
from alethic.data.fetch import iter_source
from alethic.tokenizer.codec import Codec
from alethic.tokenizer.evaluate import evaluate_tokenizer
from alethic.tokenizer.freeze import FROZEN_NAME, FrozenError, write_frozen
from alethic.tokenizer.sample import build_sample, read_heldout
from alethic.tokenizer.specials import SpecialsRegistry
from alethic.tokenizer.train import train_tokenizer

STAGES = ("sample", "train", "eval", "freeze")


def _timed(work: Path, stage: str, seconds: float, extra: dict | None = None) -> None:
    """Record a stage's wall time (the measured speeds go into RUN_RECORD.md)."""
    path = work / "timings.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[stage] = dict(extra or {}, seconds=round(seconds, 1))
    write_json_atomic(path, data)
    print(f"[{stage}] {seconds / 60:.1f} min {extra or ''}")


def stage_sample(tok_cfg: dict, data_cfg: dict, work: Path, size_bytes: int | None,
                 only: list[str] | None) -> dict | None:
    """Download what is needed and build the sample streams (all, or `only`), then merge."""
    raw_root = resolve_path(data_cfg["paths"]["raw"])
    seed = int(tok_cfg["seed"])
    fetch_stats: dict[str, dict] = {}

    def open_stream(stream: str, source: str):
        src = data_cfg["sources"][source]
        paths = [stream.split("/", 1)[1]] if source == "code_other" else None
        fetch_stats[stream] = {}
        return iter_source(source, src, raw_root, seed, fetch_stats[stream], paths)

    s = tok_cfg["sample"]
    t0 = time.perf_counter()
    report = build_sample(open_stream, dict(s["weights"]), list(data_cfg["sources"]["code_other"]["paths"]),
                          int(size_bytes or s["size_bytes"]), int(tok_cfg["evaluation"]["heldout_docs_per_source"]),
                          int(s["max_line_bytes"]), seed, work, only)
    old = work / "fetch_stats.json"
    merged = json.loads(old.read_text(encoding="utf-8")) if old.exists() else {}
    merged.update({k: v for k, v in fetch_stats.items() if v})
    write_json_atomic(old, merged)
    _timed(work, "sample" if report else f"sample:{','.join(only or [])}", time.perf_counter() - t0)
    return report


def stage_train(tok_cfg: dict, work: Path, tdir: Path, threads: int | None) -> Path:
    """Train SentencePiece on work/sample.txt (skipped if the model exists)."""
    model = tdir / f"{tok_cfg['name']}.model"
    if model.exists():
        return model
    sample = work / "sample.txt"
    if not sample.exists():
        raise FileNotFoundError(f"{sample} missing; run the sample stage first")
    overrides = {"num_threads": threads} if threads else None
    model, seconds = train_tokenizer(tok_cfg, sample, tdir, overrides)
    _timed(work, "train", seconds, {"sample_bytes": sample.stat().st_size, "threads": threads})
    return model


def stage_eval(tok_cfg: dict, work: Path, tdir: Path) -> dict:
    """Evaluate on the held-out documents; writes tdir/report.json."""
    t0 = time.perf_counter()
    codec = Codec(tdir / f"{tok_cfg['name']}.model")
    report = evaluate_tokenizer(codec, read_heldout(work), tok_cfg["evaluation"])
    sample_report = work / "sample_report.json"
    if sample_report.exists():
        report["sample"] = json.loads(sample_report.read_text(encoding="utf-8"))
    write_json_atomic(tdir / "report.json", report)
    _timed(work, "eval", time.perf_counter() - t0)
    for name, s in report["sources"].items():
        print(f"  {name:12s} roundtrip {s['roundtrip_share']:.4%}  bytes/token {s['bytes_per_token']:.2f}  "
              f"tokens/word {s['tokens_per_word']:.2f}  byte-fallback {s['byte_fallback_share']:.4%}")
    print(f"  indentation {report['indentation']}")
    print(f"  gates {report['gates']}  flags {report['flags']}  freeze_allowed {report['freeze_allowed']}")
    return report


def stage_freeze(tdir: Path, config_path: Path, specials_path: Path) -> dict:
    """Freeze only if the report allows it; copy the config next to the model; create specials.json."""
    report = json.loads((tdir / "report.json").read_text(encoding="utf-8"))
    if not report["freeze_allowed"]:
        raise FrozenError(f"freeze blocked: gates {report['gates']}, flags {report['flags']}. "
                           "Fix the cause (or, for the Uzbek flag, get the owner's decision) first.")
    shutil.copyfile(config_path, tdir / "tokenizer.yaml")
    model = next(tdir.glob("*.model"))
    record = write_frozen(tdir, model, tdir / "tokenizer.yaml", tdir / "report.json")
    SpecialsRegistry(specials_path)
    print(f"frozen: {tdir / FROZEN_NAME} model_sha256 {record['model_sha256']}")
    return record


def run(tok_cfg: dict, data_cfg: dict, config_path: Path, specials_rel: str, stage: str,
        size_bytes: int | None = None, only: list[str] | None = None, threads: int | None = None) -> int:
    """Run one stage or all of them in order. Returns a process exit code."""
    root = data_dir()
    work = root / "tokenizer" / "work"
    tdir = resolve_path(data_cfg["paths"]["tokenizer"])
    work.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    todo = STAGES if stage == "all" else (stage,)
    if "sample" in todo and not (work / "sample.txt").exists():
        if stage_sample(tok_cfg, data_cfg, work, size_bytes, only) is None:
            return 0 if stage == "sample" else 3       # partial sample: stop before training
    if "train" in todo:
        stage_train(tok_cfg, work, tdir, threads)
    if "eval" in todo and not (tdir / "report.json").exists():
        stage_eval(tok_cfg, work, tdir)
    if "freeze" in todo and not (tdir / FROZEN_NAME).exists():
        stage_freeze(tdir, config_path, root / specials_rel)
    return 0
