"""SFT data: JSONL samples -> token ids + labels, train/val split, padded batches.

A sample is {"id": "P0319.0", "segments": [{"t": text, "loss": 0 or 1}, ...]}.
Loss 1 means "the model must learn to write this text"; loss 0 is context only.
Labels use -100 for "no loss here", the value torch's cross_entropy ignores.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch

EOS_ID, PAD_ID = 2, 3
CLOSE_TAG = "</assistant>"


class NoLossError(ValueError):
    """A sample has no token with loss, so it would teach nothing (and could make the loss NaN)."""


def load_samples(paths) -> list[dict]:
    """Read every non-empty line of every JSONL file."""
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    return out


def encode_sample(sp, segments: list[dict], close_tag_loss: bool = True, eos: bool = True) -> tuple[list[int], list[int]]:
    """Tokenise the WHOLE text at once (so tokens match normal use), then mark each token.

    A token gets a label (its own id) only if it lies completely inside a loss=1 segment.
    The builder gave `</assistant>` loss 0; with close_tag_loss=True we train it anyway, because
    the model must learn to stop. If the sample ends with `</assistant>`, an EOS id is appended
    and trained too.
    """
    text = "".join(s["t"] for s in segments)
    spans, pos = [], 0                       # (start_byte, end_byte, loss) per segment
    for s in segments:
        n = len(s["t"].encode("utf-8"))
        loss = 1 if s["loss"] else 0
        if close_tag_loss and s["t"].lstrip().startswith(CLOSE_TAG):
            loss = 1
        spans.append((pos, pos + n, loss))
        pos += n
    proto = sp.encode(text, return_type="proto")
    ids, labels, k = [], [], 0
    for piece in proto.pieces:
        while k < len(spans) and spans[k][1] <= piece.begin:
            k += 1
        inside = k < len(spans) and spans[k][0] <= piece.begin and piece.end <= spans[k][1]
        ids.append(piece.id)
        labels.append(piece.id if inside and spans[k][2] else -100)
    if eos and segments and segments[-1]["t"].lstrip().startswith(CLOSE_TAG):
        ids.append(EOS_ID)
        labels.append(EOS_ID)
    if all(l == -100 for l in labels):
        raise NoLossError("no token with loss in this sample")
    return ids, labels


def encode_all(sp, samples: list[dict], max_len: int, close_tag_loss: bool = True,
               eos: bool = True) -> tuple[list[dict], dict]:
    """Encode many samples; drop (and report) samples longer than max_len or without loss."""
    enc, too_long, no_loss = [], [], []
    for s in samples:
        try:
            ids, labels = encode_sample(sp, s["segments"], close_tag_loss, eos)
        except NoLossError:
            no_loss.append(s["id"])
            continue
        if len(ids) > max_len:
            too_long.append(s["id"])
            continue
        enc.append({"id": s["id"], "ids": ids, "labels": labels})
    report = {"kept": len(enc), "dropped_too_long": too_long, "dropped_no_loss": no_loss,
              "trained_tokens": sum(sum(1 for l in e["labels"] if l != -100) for e in enc),
              "total_tokens": sum(len(e["ids"]) for e in enc)}
    return enc, report


def source_of(sample_id: str) -> str:
    """'P0319.2' -> 'P0319': parts of one example share a source and must stay on the same side."""
    return sample_id.split(".")[0]


def split_train_val(samples: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic split by source id (hash), so no example leaks from train into val."""
    train, val = [], []
    for s in samples:
        h = hashlib.sha256(f"{seed}:{source_of(s['id'])}".encode()).digest()
        frac = int.from_bytes(h[:8], "big") / 2**64
        (val if frac < val_fraction else train).append(s)
    return train, val


def batches(enc: list[dict], batch_size: int, pad_id: int, seed: int, epoch: int):
    """Shuffle (differently each epoch, same every run), then yield padded (ids, labels) tensors."""
    order = list(range(len(enc)))
    random.Random(seed * 100003 + epoch).shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [enc[j] for j in order[i:i + batch_size]]
        width = max(len(e["ids"]) for e in chunk)
        x = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        y = torch.full((len(chunk), width), -100, dtype=torch.long)
        for r, e in enumerate(chunk):
            n = len(e["ids"])
            x[r, :n] = torch.tensor(e["ids"])
            y[r, :n] = torch.tensor(e["labels"])
        yield x, y
