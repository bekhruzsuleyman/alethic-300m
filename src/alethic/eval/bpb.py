"""Bits per byte on the validation shards, per source and overall (M2/M4).

BPB = (sum of loss in nats over all tokens) / (ln 2 * total UTF-8 bytes of the documents).
Comparable across tokenizers, unlike loss per token. Evaluated on a fixed 2M-token slice
per source every 1,000 steps and on all validation shards at the end of a run.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from alethic.data.shards import TOKEN_DTYPE


def bits_per_byte(loss_nats_sum: float, utf8_bytes: int) -> float:
    """Convert summed token loss (nats) over documents into bits per UTF-8 byte."""
    if utf8_bytes <= 0:
        raise ValueError("utf8_bytes must be positive")
    return loss_nats_sum / (math.log(2) * utf8_bytes)


def load_val_slices(shards_dir: Path,
                    max_tokens_per_source: int | None = 2_000_000) -> dict[str, tuple[np.ndarray, int]]:
    """Return {source: (uint16_tokens, utf8_bytes)} from validation shards in shards_dir."""
    manifest_path = Path(shards_dir) / "manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_source: dict[str, list[dict]] = {}
    for rec in manifest.get("shards", []):
        if rec.get("split") == "val":
            by_source.setdefault(rec["source"], []).append(rec)

    slices: dict[str, tuple[np.ndarray, int]] = {}
    for source, recs in sorted(by_source.items()):
        chunks: list[np.ndarray] = []
        total_tok = 0
        total_bytes = 0
        for rec in recs:
            bin_path = Path(shards_dir) / "val" / rec["path"]
            side_path = bin_path.with_suffix(".json")
            if not bin_path.is_file():
                continue
            arr = np.frombuffer(bin_path.read_bytes(), dtype=TOKEN_DTYPE)
            side = json.loads(side_path.read_text(encoding="utf-8")) if side_path.is_file() else rec
            doc_tokens = side.get("doc_tokens")
            doc_bytes = side.get("doc_bytes")
            if max_tokens_per_source is None or total_tok + len(arr) <= max_tokens_per_source or not doc_tokens:
                chunks.append(arr)
                total_tok += len(arr)
                total_bytes += int(side.get("utf8_bytes", rec.get("utf8_bytes", len(arr) * 4)))
            else:
                need = max_tokens_per_source - total_tok
                take_tok = 0
                take_bytes = 0
                for dt, db in zip(doc_tokens, doc_bytes):
                    take_tok += int(dt)
                    take_bytes += int(db)
                    if take_tok >= need:
                        break
                chunks.append(arr[:take_tok])
                total_tok += take_tok
                total_bytes += take_bytes
            if max_tokens_per_source is not None and total_tok >= max_tokens_per_source:
                break
        if chunks and total_tok >= 2 and total_bytes > 0:
            slices[source] = (np.concatenate(chunks), total_bytes)
    return slices


def score_tokens_nats(model, tokens: np.ndarray, seq_len: int, micro_batch: int,
                      device: str, use_bf16: bool = False) -> tuple[float, int]:
    """Sum cross-entropy loss (in nats) across `tokens` in windows of up to seq_len."""
    import torch
    import torch.nn.functional as F

    n = int(len(tokens))
    if n < 2:
        return 0.0, 0
    was_training = model.training
    model.eval()
    dev_type = device.split(":")[0]
    n_full = (n - 1) // seq_len
    total_nats = 0.0
    scored = 0
    try:
        with torch.no_grad():
            if n_full > 0:
                end_full = n_full * seq_len
                starts = range(0, end_full, seq_len)
                windows = np.stack([tokens[s:s + seq_len + 1].astype(np.int64) for s in starts])
                for i in range(0, len(windows), micro_batch):
                    chunk = torch.from_numpy(windows[i:i + micro_batch]).to(device)
                    with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_bf16):
                        logits = model(chunk[:, :-1])
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                           chunk[:, 1:].reshape(-1), reduction="sum")
                    total_nats += float(loss.item())
                    scored += int(chunk[:, 1:].numel())
            rem_start = n_full * seq_len
            if n - 1 > rem_start:
                tail = torch.from_numpy(tokens[rem_start:].astype(np.int64)[None, :]).to(device)
                with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_bf16):
                    logits = model(tail[:, :-1])
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                       tail[:, 1:].reshape(-1), reduction="sum")
                total_nats += float(loss.item())
                scored += int(tail[:, 1:].numel())
    finally:
        model.train(was_training)
    return total_nats, scored


def evaluate_bpb(model, shards_dir: Path, seq_len: int, micro_batch: int = 8,
                 device: str = "cpu", use_bf16: bool = False,
                 max_tokens_per_source: int | None = 2_000_000) -> dict:
    """Evaluate bits per byte per source and overall on validation shards in shards_dir."""
    slices = load_val_slices(shards_dir, max_tokens_per_source=max_tokens_per_source)
    sources_out: dict[str, dict] = {}
    sum_nats = 0.0
    sum_bytes = 0
    sum_tokens = 0
    for source, (tokens, utf8_bytes) in slices.items():
        nats, scored = score_tokens_nats(model, tokens, seq_len, micro_batch, device, use_bf16)
        if scored == 0 or utf8_bytes <= 0:
            continue
        bpb = bits_per_byte(nats, utf8_bytes)
        sources_out[source] = {
            "bpb": round(bpb, 6),
            "loss_nats": round(nats / scored, 6),
            "tokens": scored,
            "utf8_bytes": utf8_bytes,
        }
        sum_nats += nats
        sum_bytes += utf8_bytes
        sum_tokens += scored
    overall = bits_per_byte(sum_nats, sum_bytes) if sum_bytes > 0 else None
    return {
        "overall_bpb": round(overall, 6) if overall is not None else None,
        "total_tokens": sum_tokens,
        "total_utf8_bytes": sum_bytes,
        "sources": sources_out,
    }
