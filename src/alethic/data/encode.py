"""Parallel encoding of work units into shards, and the train / validation split (M2/M3).

Cleans Uzbek text before hashing, builds one global 8-byte blake2b duplicate drop list in
(source, file, row) order, drops documents sharing a 13-gram with HumanEval/MBPP, encodes
row-group units in parallel with EOS=2 into ~128 MiB uint16 shards, and writes manifest.json.
"""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow.parquet as pq

from alethic.config import ConfigError, resolve_path, write_json_atomic
from alethic.data import shards
from alethic.data.decontam import DecontamIndex, build_benchmark_ngrams
from alethic.data.dedup import build_drop_list, hash_texts, read_unit_hash_meta, write_unit_hashes
from alethic.data.fetch import check_source, download, list_parquet, read_row_group_range
from alethic.tokenizer.codec import Codec
from alethic.tokenizer.freeze import verify_frozen

KNOWN_SOURCES = frozenset({"python", "code_other", "fineweb_edu", "math", "uzbek_latn", "uzbek_cyrl"})
VAL_FRACTION = 0.001          # 0.1% of groups go to validation
_BUCKETS = 1_000_000
MEASURED_BPT = {"python": 3.60, "code_other": 3.19, "fineweb_edu": 5.20,
                "math": 3.94, "uzbek_latn": 3.85, "uzbek_cyrl": 5.58}
_W_CODEC: Codec | None = None
_W_DECONTAM: DecontamIndex | None = None
_W_EOS: int = 2
_W_VAL_FRAC: float = VAL_FRACTION


def is_validation(source: str, key: str, fraction: float = VAL_FRACTION) -> bool:
    """True if this document's group belongs to the validation split (hashed by key only)."""
    if source not in KNOWN_SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(KNOWN_SOURCES)}")
    if not isinstance(key, str) or not key:
        raise ValueError(f"empty group key for source {source!r}; every document needs one")
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8, person=b"alethic-split").digest()
    return int.from_bytes(digest, "little") % _BUCKETS < int(round(fraction * _BUCKETS))


def _stream_targets(data_cfg: dict, target_scale: float) -> list[tuple[str, str, list[str] | None, int]]:
    """[(stream, source, paths_override, target_tokens)] in deterministic sorted order."""
    out = []
    for source in sorted(data_cfg["sources"]):
        src = data_cfg["sources"][source]
        if src.get("repo") != "local":
            check_source(source, src)
        total = max(1, int(round(int(src["tokens"]) * target_scale)))
        if source == "code_other":
            langs = list(src["paths"])
            for i, lang in enumerate(langs):
                share = total // len(langs) + (1 if i < total % len(langs) else 0)
                out.append((f"code_other/{lang}", source, [lang], max(1, share)))
        else:
            out.append((source, source, None, total))
    return out


def _file_units(stream: str, source: str, src: dict, path_in_repo: str, local_path: Path,
                unit_cap_tokens: int, bpt: float) -> list[dict]:
    """Split one parquet file into row-group (or row-slice) units of at most unit_cap_tokens."""
    pf = pq.ParquetFile(local_path)
    n_rg = pf.num_row_groups
    if n_rg == 0:
        return []
    max_bytes = max(1, int(unit_cap_tokens * bpt))
    stem = path_in_repo.replace("/", "_").removesuffix(".parquet")
    base = {"source": source, "stream": stream, "path_in_repo": path_in_repo,
            "parquet_path": str(local_path), "text_column": src["text_column"], "key_column": src["key_column"]}
    units: list[dict] = []
    start = 0
    while start < n_rg:
        rg0 = pf.metadata.row_group(start)
        rg0_bytes = max(1, rg0.total_byte_size)
        if rg0_bytes > max_bytes * 1.5 and rg0.num_rows > 1:
            step = max(1, int(rg0.num_rows * max_bytes / rg0_bytes))
            for r0 in range(0, rg0.num_rows, step):
                r1 = min(rg0.num_rows, r0 + step)
                units.append({**base, "id": f"{source}/{path_in_repo}:{start}[{r0}:{r1}]",
                              "name": f"{stem}-r{start:04d}_{r0:06d}",
                              "rg_start": start, "rg_end": start + 1, "row_start": r0, "row_end": r1})
            start += 1
            continue
        acc, end = 0, start
        while end < n_rg:
            rg_bytes = max(1, pf.metadata.row_group(end).total_byte_size)
            if end > start and acc + rg_bytes > max_bytes:
                break
            acc += rg_bytes
            end += 1
        units.append({**base, "id": f"{source}/{path_in_repo}:{start}-{end}",
                      "name": f"{stem}-r{start:04d}_{end:04d}",
                      "rg_start": start, "rg_end": end, "row_start": None, "row_end": None})
        start = end
    return units


def _read_unit_docs(unit: dict) -> tuple[list[tuple[str, str]], int]:
    """Read one unit's (key, text) documents and Uzbek normalization edit count."""
    return read_row_group_range(Path(unit["parquet_path"]), unit["source"], unit["text_column"],
                                unit["key_column"], int(unit["rg_start"]), int(unit["rg_end"]),
                                unit.get("row_start"), unit.get("row_end"))


def _hash_worker(args: tuple[dict, str]) -> dict:
    """Hash one unit's documents and write <unit>.bin + <unit>.json under hashes_dir."""
    unit, hashes_dir = args[0], Path(args[1])
    cached = read_unit_hash_meta(hashes_dir, unit["id"])
    if cached is not None and "uzbek_chars_changed" in cached:
        return cached
    docs, uz_changed = _read_unit_docs(unit)
    texts = [t for _, t in docs]
    meta = write_unit_hashes(hashes_dir, unit["id"], hash_texts(texts), sum(len(t.encode("utf-8")) for t in texts))
    meta["uzbek_chars_changed"] = int(uz_changed)
    write_json_atomic(hashes_dir / f"{unit['id'].replace('/', '__').replace(':', '_')}.json", meta)
    return meta


def _init_encode_worker(model_path: str, ngrams: list[list[str]], ngram_n: int,
                        eos_id: int, val_fraction: float) -> None:
    """Load the frozen Codec and DecontamIndex once per worker process."""
    global _W_CODEC, _W_DECONTAM, _W_EOS, _W_VAL_FRAC
    _W_CODEC, _W_EOS, _W_VAL_FRAC = Codec(model_path), int(eos_id), float(val_fraction)
    _W_DECONTAM = DecontamIndex((tuple(g) for g in ngrams), n=ngram_n)


def _encode_worker(args: tuple[dict, set[int], str, int]) -> dict:
    """Filter duplicates + contaminated docs, batch-encode with EOS, and write unit shards."""
    unit, drop_set, out_root, shard_tokens = args[0], args[1], Path(args[2]), args[3]
    assert _W_CODEC is not None and _W_DECONTAM is not None
    docs, uz_changed = _read_unit_docs(unit)
    kept_keys, kept_texts = [], []
    dedup_dropped = decontam_dropped = 0
    for idx, (key, text) in enumerate(docs):
        if idx in drop_set:
            dedup_dropped += 1
        elif _W_DECONTAM.is_contaminated(text):
            decontam_dropped += 1
        else:
            kept_keys.append(key or text[:200])
            kept_texts.append(text)
    ids_list = _W_CODEC.encode_batch(kept_texts, num_threads=1) if kept_texts else []
    writers = {
        False: shards.ShardWriter(out_root / "train", unit["source"], unit["name"], "train", shard_tokens),
        True: shards.ShardWriter(out_root / "val", unit["source"], unit["name"], "val", shard_tokens,
                                  store_doc_bytes=True),
    }
    train_tokens = val_tokens = total_bytes = 0
    for key, text, ids in zip(kept_keys, kept_texts, ids_list):
        arr = np.append(np.asarray(ids, dtype=np.int64), _W_EOS)
        nbytes = len(text.encode("utf-8"))
        is_val = is_validation(unit["source"], key, _W_VAL_FRAC)
        writers[is_val].add(shards.to_tokens(arr), nbytes)
        total_bytes += nbytes
        train_tokens += 0 if is_val else len(arr)
        val_tokens += len(arr) if is_val else 0
    written = writers[False].close() + writers[True].close()
    for rec in written:
        rec.pop("doc_bytes", None)
        rec.pop("doc_tokens", None)
    return {"unit": unit["id"], "source": unit["source"], "stream": unit["stream"], "shards": written,
            "tokens": train_tokens + val_tokens, "train_tokens": train_tokens, "val_tokens": val_tokens,
            "docs": len(kept_texts), "utf8_bytes": total_bytes, "dedup_dropped": dedup_dropped,
            "decontam_dropped": decontam_dropped, "uzbek_chars_changed": uz_changed}


def plan_and_hash_units(data_cfg: dict, raw_root: Path, hashes_dir: Path, target_scale: float,
                        pool: mp.pool.Pool | None = None,
                        on_checkpoint: Callable[[str], None] | None = None) -> tuple[list[dict], dict[str, int]]:
    """Download needed files, split into row-group units, and run the hash pass."""
    overfetch = max(float(data_cfg.get("overfetch", 1.15)), 1.25)
    unit_max = int(data_cfg["unit_max_tokens"])
    all_units: list[dict] = []
    stream_targets: dict[str, int] = {}

    for stream, source, paths_ov, target_tok in _stream_targets(data_cfg, target_scale):
        stream_targets[stream] = target_tok
        src = data_cfg["sources"][source]
        bpt = MEASURED_BPT.get(source, float(src["bytes_per_token"]))
        need_bytes = int(target_tok * bpt * overfetch)
        unit_cap = min(unit_max, max(1, target_tok // 8))
        dest = raw_root / source
        got_utf8 = 0
        for path_in_repo, _size in list_parquet(src, paths_ov, local_dir=dest):
            local = dest / path_in_repo
            if not local.is_file():
                local = download(src, path_in_repo, dest)
            file_units = _file_units(stream, source, src, path_in_repo, local, unit_cap, bpt)
            tasks = [(u, str(hashes_dir)) for u in file_units]
            metas = list(pool.imap(_hash_worker, tasks)) if pool else [_hash_worker(t) for t in tasks]
            for u, m in zip(file_units, metas):
                u["utf8_bytes"] = int(m["utf8_bytes"])
                u["uzbek_chars_changed"] = int(m.get("uzbek_chars_changed", 0))
                all_units.append(u)
                got_utf8 += int(m["utf8_bytes"])
                if got_utf8 >= need_bytes:
                    break
            if got_utf8 >= need_bytes:
                break
        if on_checkpoint:
            on_checkpoint(f"hash:{stream}")
    return all_units, stream_targets


def encode_units(data_cfg: dict, target_scale: float | None = None, workers: int | None = None,
                 on_checkpoint: Callable[[str], None] | None = None) -> dict:
    """Full M2/M3 pipeline: verify frozen tokenizer, hash & dedup, decontaminate, encode shards."""
    scale = float(target_scale if target_scale is not None else data_cfg.get("target_scale", 1.0))
    if scale <= 0:
        raise ConfigError(f"target_scale must be positive, got {scale}")
    raw_root = resolve_path(data_cfg["paths"]["raw"])
    hashes_dir = resolve_path(data_cfg["paths"]["hashes"])
    shards_dir = resolve_path(data_cfg["paths"]["shards"])
    model_path, tokenizer_sha = verify_frozen(resolve_path(data_cfg["paths"]["tokenizer"]))

    shards_dir.mkdir(parents=True, exist_ok=True)
    shards.remove_stray_tmp(shards_dir)
    state = shards.StateLog(shards_dir / "state.jsonl")

    n_workers = int(workers or os.cpu_count() or 1)
    t0 = time.perf_counter()
    decontam = build_benchmark_ngrams(data_cfg["decontam"], raw_root)
    init_args = (str(model_path), decontam.to_list(), int(data_cfg["decontam"].get("ngram", 13)),
                 int(data_cfg.get("eos_id", 2)), float(data_cfg.get("val_fraction", VAL_FRACTION)))
    shard_tokens = int(data_cfg.get("shard_tokens", shards.SHARD_TOKENS))

    if n_workers == 1:
        _init_encode_worker(*init_args)
    cm = mp.get_context("spawn").Pool(n_workers, _init_encode_worker, init_args) if n_workers > 1 else nullcontext(None)
    with cm as pool:
        units, stream_targets = plan_and_hash_units(data_cfg, raw_root, hashes_dir, scale, pool, on_checkpoint)
        drops_by_unit, dedup_summary = build_drop_list([u["id"] for u in units], hashes_dir)
        if on_checkpoint:
            on_checkpoint("dedup")

        by_stream: dict[str, list[dict]] = {}
        for u in units:
            by_stream.setdefault(u["stream"], []).append(u)
        done_tok = {s: sum(int(state.records[u["id"]]["tokens"]) for u in us if state.done(u["id"]))
                    for s, us in by_stream.items()}
        for s in stream_targets:
            done_tok.setdefault(s, 0)

        encode_t0 = last_report = time.perf_counter()
        new_tokens = 0
        while True:
            batch: list[dict] = []
            for s, target_tok in sorted(stream_targets.items()):
                rem = target_tok - done_tok[s]
                if rem <= 0:
                    continue
                bpt = MEASURED_BPT.get(s.split("/", 1)[0], 3.8)
                est_acc = 0
                for u in by_stream.get(s, []):
                    if not state.done(u["id"]):
                        batch.append(u)
                        est_acc += max(1, int(u["utf8_bytes"] / bpt))
                        if est_acc >= rem:
                            break
            if not batch:
                break
            tasks = [(u, drops_by_unit[u["id"]], str(shards_dir), shard_tokens) for u in batch]
            results = pool.imap(_encode_worker, tasks) if pool else (_encode_worker(t) for t in tasks)
            for rec in results:
                s = rec["stream"]
                if done_tok[s] >= stream_targets[s]:
                    for sh in rec["shards"]:
                        p = shards_dir / sh["split"] / sh["path"]
                        p.unlink(missing_ok=True)
                        p.with_suffix(".json").unlink(missing_ok=True)
                    continue
                state.append(rec)
                done_tok[s] += int(rec["tokens"])
                new_tokens += int(rec["tokens"])
                now = time.perf_counter()
                if now - last_report >= 30.0:
                    tps = new_tokens / max(now - encode_t0, 1e-6)
                    rem_all = sum(max(0, t - done_tok[k]) for k, t in stream_targets.items())
                    print(f"[encode] {sum(done_tok.values()):,} tok | {tps / n_workers:,.0f} tok/s/worker "
                          f"({tps:,.0f} total) | ETA {(rem_all / max(tps, 1.0)) / 60.0:.1f} min", flush=True)
                    last_report = now
                    if on_checkpoint:
                        on_checkpoint("encode_progress")
            if on_checkpoint:
                on_checkpoint("encode_batch")

    return shards.write_shards_manifest(
        data_cfg, shards_dir, state, units, stream_targets, tokenizer_sha,
        dedup_summary, scale, n_workers, time.perf_counter() - t0, new_tokens, KNOWN_SOURCES,
    )
