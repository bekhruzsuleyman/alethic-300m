"""Training data loader: memory-mapped shards, one global seeded permutation of windows.

The shards, in the given order, form one packed token stream (documents are already joined
by EOS; attention is not masked across documents). Window w is the seq_len + 1 tokens
starting at w * seq_len, so inputs are its first seq_len tokens and targets the last
seq_len. Step k uses permuted windows k*B .. k*B + B - 1. The loader state is the step number.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from alethic.data.shards import TOKEN_DTYPE


class LoaderError(RuntimeError):
    """The shards are missing, the wrong size, or too small for one window."""


class WindowLoader:
    """Deterministic batches of windows from a list of uint16 shard files."""

    def __init__(self, shard_paths: list[Path], seq_len: int, batch_seqs: int, seed: int) -> None:
        """Memory-map every shard and compute the number of whole windows."""
        if not shard_paths:
            raise LoaderError("no training shards given")
        self.paths = [Path(p) for p in shard_paths]
        self.maps = [np.memmap(p, dtype=TOKEN_DTYPE, mode="r") for p in self.paths]
        sizes = np.array([len(m) for m in self.maps], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(sizes)])   # stream offset of each shard
        self.total_tokens = int(self.starts[-1])
        self.seq_len, self.batch_seqs, self.seed = seq_len, batch_seqs, seed
        self.n_windows = (self.total_tokens - 1) // seq_len
        if self.n_windows < batch_seqs:
            raise LoaderError(f"{self.total_tokens} tokens give {self.n_windows} windows; need {batch_seqs}")
        self._perm_epoch = -1
        self._perm = np.empty(0, dtype=np.int64)

    @property
    def steps_per_epoch(self) -> int:
        """Whole steps in one pass over all windows."""
        return self.n_windows // self.batch_seqs

    def permutation(self, epoch: int) -> np.ndarray:
        """The seeded order of all windows for one epoch (cached for the current epoch)."""
        if epoch != self._perm_epoch:
            rng = np.random.default_rng([self.seed, epoch])
            self._perm = rng.permutation(self.n_windows)
            self._perm_epoch = epoch
        return self._perm

    def window_ids(self, step: int) -> np.ndarray:
        """Window numbers used by one step; past one epoch, the next epoch's order is used."""
        flat = np.arange(step * self.batch_seqs, (step + 1) * self.batch_seqs, dtype=np.int64)
        out = np.empty_like(flat)
        for i, j in enumerate(flat):
            epoch, pos = divmod(int(j), self.n_windows)
            out[i] = self.permutation(epoch)[pos]
        return out

    def read(self, start: int, length: int) -> np.ndarray:
        """Read `length` tokens from the packed stream starting at `start` (may cross shards)."""
        out = np.empty(length, dtype=np.int64)
        done = 0
        shard = int(np.searchsorted(self.starts, start, side="right")) - 1
        while done < length:
            offset = start + done - int(self.starts[shard])
            take = min(length - done, len(self.maps[shard]) - offset)
            out[done:done + take] = self.maps[shard][offset:offset + take]
            done += take
            shard += 1
        return out

    def window(self, w: int) -> np.ndarray:
        """Tokens of window w: seq_len + 1 of them, as int64."""
        return self.read(w * self.seq_len, self.seq_len + 1)

    def batch(self, step: int) -> np.ndarray:
        """The (batch_seqs, seq_len + 1) int64 array for one step."""
        return np.stack([self.window(int(w)) for w in self.window_ids(step)])


def seen_shard_names(old_manifest: Path) -> set[str]:
    """Shard names (manifest `path`) to leave out of training.

    Reads either an older manifest (names of its training shards) or an exclude file
    {"exclude": [names]} written by `seen_in_new_manifest` (phase 2: shards of a bigger build that overlap seen data)."""
    old = json.loads(Path(old_manifest).read_text(encoding="utf-8"))
    if "exclude" in old:
        return set(old["exclude"])
    return {rec["path"] for rec in old["shards"] if rec["split"] == "train"}


_SHARD_RE = re.compile(r"^(?P<source>[^/]+)/(?P<stem>.+)-r(?P<a>\d{4})_(?P<b>\d{4,6})-(?P<part>\d{4})\.bin$")


def _shard_row_groups(path: str) -> tuple[tuple[str, str], range] | None:
    """((source, file stem), parquet row groups the shard's unit covered), from the shard name.

    Unit names end -r<start>_<end> (4-digit end = row groups start..end-1) or -r<start>_<row> (6-digit = a
    slice of row group start). A slice counts as its whole row group (conservative)."""
    m = _SHARD_RE.match(path)
    if not m:
        return None
    a, b = int(m["a"]), m["b"]
    rgs = range(a, int(b)) if len(b) == 4 else range(a, a + 1)
    return (m["source"], m["stem"]), rgs


def seen_in_new_manifest(old_manifest: Path, new_manifest: Path) -> tuple[set[str], dict]:
    """Names of training shards in the new manifest whose parquet row groups were already trained on.

    The old (small) build took a prefix of every file's row groups, but its unit boundaries differ from the
    new build's, so shard names do not match; row groups do. Any new shard touching a seen row group is excluded
    (a straddling shard costs a little unseen data but never repeats seen data)."""
    old = json.loads(Path(old_manifest).read_text(encoding="utf-8"))
    new = json.loads(Path(new_manifest).read_text(encoding="utf-8"))
    seen: dict[tuple[str, str], set[int]] = {}
    for rec in old["shards"]:
        if rec["split"] != "train":
            continue
        parsed = _shard_row_groups(rec["path"])
        if parsed is None:
            raise LoaderError(f"cannot read the row groups from shard name {rec['path']}")
        seen.setdefault(parsed[0], set()).update(parsed[1])
    excluded: set[str] = set()
    ex_tokens = unseen_tokens = 0
    for rec in new["shards"]:
        if rec["split"] != "train":
            continue
        parsed = _shard_row_groups(rec["path"])
        if parsed is not None and parsed[0] in seen and seen[parsed[0]].intersection(parsed[1]):
            excluded.add(rec["path"])
            ex_tokens += int(rec["tokens"])
        else:
            unseen_tokens += int(rec["tokens"])
    old_tokens = sum(int(r["tokens"]) for r in old["shards"] if r["split"] == "train")
    return excluded, {"old_train_tokens": old_tokens, "excluded_tokens": ex_tokens, "unseen_train_tokens": unseen_tokens}


def train_shards_from_manifest(shards_dir: Path, exclude: set[str] | None = None) -> tuple[list[Path], dict]:
    """Return the training shard paths (manifest order) and the manifest, checking file sizes.

    `exclude` holds shard names to leave out (phase 2 skips the shards the model already trained on)."""
    manifest_path = Path(shards_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise LoaderError(f"{manifest_path} not found; build the data first (data-build, M2/M3).")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = []
    for rec in manifest["shards"]:
        if rec["split"] != "train" or (exclude and rec["path"] in exclude):
            continue
        p = Path(shards_dir) / "train" / rec["path"]
        if not p.is_file() or p.stat().st_size != 2 * rec["tokens"]:
            raise LoaderError(f"shard {p} is missing or not {rec['tokens']} tokens; rebuild or re-copy it")
        paths.append(p)
    return paths, manifest
