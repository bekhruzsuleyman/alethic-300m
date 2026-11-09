"""Build the tokenizer training sample and the held-out evaluation documents (M1, plan step 1).

The sample is a shuffled, source-weighted text file with one source line per line (SentencePiece
reads one sentence per line). Weights: python 22 : other code 10 : FineWeb-Edu 17 : math 5 :
Uzbek Latin 4 : Uzbek Cyrillic 2, fixed seed. Validation groups (is_validation) never enter it.
Lines over 16,384 bytes and lines containing <sp00>..<sp63> text are dropped and counted.
The first held-out documents of every stream are kept apart for evaluation, never in the sample.

Each stream (a source, or one other-code language) is saved on its own under work/streams/, so a
crash or a partial run (e.g. before StarCoderData access) never redoes finished streams.
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Callable, Iterator

from alethic.config import write_json_atomic
from alethic.data.encode import is_validation

SLOT_TEXT = re.compile(r"<sp\d\d>")
Docs = Iterator[tuple[str, str]]          # (group key, text)


def plan_streams(weights: dict[str, float], code_langs: list[str]) -> dict[str, tuple[str, float]]:
    """{stream: (source, weight)}; other code is shared equally by its languages ("code_other/rust")."""
    streams = {}
    for source, w in weights.items():
        if source == "code_other":
            for lang in code_langs:
                streams[f"code_other/{lang}"] = (source, w / len(code_langs))
        else:
            streams[source] = (source, float(w))
    return streams


def take_stream(docs: Docs, source: str, quota: int, heldout_n: int, max_line: int) -> dict:
    """Read documents until `quota` sample bytes are collected after `heldout_n` held-out docs."""
    out = {"lines": [], "heldout": [], "bytes": 0, "docs": 0, "val_skipped": 0,
           "dropped_long": 0, "dropped_slot_text": 0}
    for key, text in docs:
        if is_validation(source, key or text[:200]):
            out["val_skipped"] += 1
            continue
        if len(out["heldout"]) < heldout_n:
            out["heldout"].append({"key": key, "text": text})
            continue
        out["docs"] += 1
        for line in text.split("\n"):
            if not line:
                continue
            raw = line.encode("utf-8")
            if len(raw) > max_line:
                out["dropped_long"] += 1
            elif SLOT_TEXT.search(line):
                out["dropped_slot_text"] += 1
            else:
                out["lines"].append(raw)
                out["bytes"] += len(raw) + 1
        if out["bytes"] >= quota:
            return out
    raise RuntimeError(f"{source}: data ran out after {out['bytes']:,} of {quota:,} bytes")


def _stream_paths(out_dir: Path, stream: str) -> tuple[Path, Path, Path]:
    """Lines file, held-out file and stats file of one stream."""
    safe = stream.replace("/", "__")
    d = out_dir / "streams"
    return d / f"{safe}.txt", d / f"{safe}.heldout.jsonl", d / f"{safe}.json"


def build_stream(stream: str, source: str, docs: Docs, quota: int, heldout_n: int,
                 max_line: int, out_dir: Path) -> dict:
    """Build one stream unless it is already finished; its stats file is written last (= done)."""
    lines_p, held_p, stats_p = _stream_paths(out_dir, stream)
    if stats_p.exists():
        return json.loads(stats_p.read_text(encoding="utf-8"))
    stats_p.parent.mkdir(parents=True, exist_ok=True)
    got = take_stream(docs, source, quota, heldout_n, max_line)
    _write_lines(lines_p, got.pop("lines"))
    heldout = got.pop("heldout")
    _write_jsonl(held_p, heldout)
    stats = dict(got, stream=stream, source=source, quota=quota, heldout=len(heldout))
    write_json_atomic(stats_p, stats)
    print(f"stream {stream}: {stats['bytes']:,} bytes, {stats['docs']:,} docs, "
          f"{stats['dropped_long']} long lines dropped, {stats['val_skipped']} validation docs skipped")
    return stats


def build_sample(open_stream: Callable[[str, str], Docs], weights: dict[str, float],
                 code_langs: list[str], size_bytes: int, heldout_per_source: int, max_line: int,
                 seed: int, out_dir: Path, only: list[str] | None = None) -> dict | None:
    """Build every stream (or only some), then merge them into out_dir/sample.txt.

    open_stream(stream, source) returns that stream's documents (it downloads lazily).
    Returns the merged report, or None when `only` leaves streams unbuilt.
    """
    streams = plan_streams(weights, code_langs)
    total_w = sum(w for _, w in streams.values())
    count = {}
    for source, _ in streams.values():
        count[source] = count.get(source, 0) + 1
    stats = {}
    for name, (source, w) in sorted(streams.items()):
        if only is not None and name not in only:
            continue
        quota = int(size_bytes * w / total_w)
        heldout_n = -(-heldout_per_source // count[source])            # ceil
        stats[name] = build_stream(name, source, open_stream(name, source), quota, heldout_n,
                                   max_line, out_dir)
    missing = [n for n in streams if not _stream_paths(out_dir, n)[2].exists()]
    if missing:
        print(f"streams not built yet: {missing}")
        return None
    return merge_sample(out_dir, sorted(streams), seed, size_bytes)


def merge_sample(out_dir: Path, streams: list[str], seed: int, size_bytes: int) -> dict:
    """Shuffle all stream lines together (fixed seed) into sample.txt; write sample_report.json."""
    lines: list[bytes] = []
    report = {"size_bytes_target": size_bytes, "seed": seed, "streams": {}}
    for name in streams:
        lines_p, _, stats_p = _stream_paths(out_dir, name)
        report["streams"][name] = json.loads(stats_p.read_text(encoding="utf-8"))
        lines += lines_p.read_bytes().split(b"\n")[:-1]
    random.Random(seed).shuffle(lines)
    _write_lines(out_dir / "sample.txt", lines)
    report["lines"] = len(lines)
    report["bytes"] = sum(len(l) + 1 for l in lines)
    report["dropped_long_total"] = sum(s["dropped_long"] for s in report["streams"].values())
    write_json_atomic(out_dir / "sample_report.json", report)
    return report


def read_heldout(out_dir: Path) -> dict[str, list[str]]:
    """Held-out texts per source (other-code languages merged into code_other)."""
    result: dict[str, list[str]] = {}
    for p in sorted((out_dir / "streams").glob("*.heldout.jsonl")):
        source = p.name.split(".")[0].split("__")[0]
        with open(p, encoding="utf-8") as f:
            result.setdefault(source, []).extend(json.loads(l)["text"] for l in f)
    return result


def _write_lines(path: Path, lines: list[bytes]) -> None:
    """Write one line per entry to path.tmp, flush, then rename (atomic)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        for i in range(0, len(lines), 100_000):
            f.write(b"".join(l + b"\n" for l in lines[i:i + 100_000]))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as JSON lines, atomically."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
