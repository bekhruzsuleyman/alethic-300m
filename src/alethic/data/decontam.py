"""Benchmark decontamination (milestone M2).

Drops any document sharing a 13-gram of whitespace-separated words with the HumanEval or
MBPP test problems and solutions (pinned revisions), and logs drop counts per source.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq

from alethic.config import ConfigError, write_json_atomic
from alethic.data.fetch import check_source, download, list_parquet


def extract_ngrams(text: str, n: int = 13) -> list[tuple[str, ...]]:
    """All n-grams of whitespace-separated words in text."""
    words = text.split()
    if len(words) < n:
        return []
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


class DecontamIndex:
    """Fast exact 13-gram matcher keyed by (first_word, last_word) of each benchmark n-gram."""

    def __init__(self, ngrams: Iterable[tuple[str, ...]], n: int = 13) -> None:
        self.n = int(n)
        self.by_ends: dict[tuple[str, str], set[tuple[str, ...]]] = {}
        self.total_ngrams = 0
        for g in ngrams:
            if len(g) == self.n:
                bucket = self.by_ends.setdefault((g[0], g[-1]), set())
                if g not in bucket:
                    bucket.add(g)
                    self.total_ngrams += 1

    @classmethod
    def from_texts(cls, texts: Iterable[str], n: int = 13) -> DecontamIndex:
        """Build an index from an iterable of benchmark texts."""
        all_ngrams: list[tuple[str, ...]] = []
        for t in texts:
            all_ngrams.extend(extract_ngrams(t, n))
        return cls(all_ngrams, n=n)

    def is_contaminated(self, text: str) -> bool:
        """True if `text` shares any n-gram of whitespace-separated words with the index."""
        if not self.by_ends:
            return False
        words = text.split()
        n = self.n
        if len(words) < n:
            return False
        nm1 = n - 1
        by_ends = self.by_ends
        for i in range(len(words) - nm1):
            cand = by_ends.get((words[i], words[i + nm1]))
            if cand is not None and tuple(words[i:i + n]) in cand:
                return True
        return False

    def to_list(self) -> list[list[str]]:
        """Serialisable list of all distinct n-grams in the index."""
        return [list(g) for bucket in self.by_ends.values() for g in sorted(bucket)]


def build_benchmark_ngrams(decontam_cfg: dict, raw_root: Path) -> DecontamIndex:
    """Download pinned HumanEval + MBPP parquet files (cached) and return their DecontamIndex."""
    n = int(decontam_cfg.get("ngram", 13))
    bench_root = Path(raw_root) / "_benchmarks"
    cache_path = bench_root / f"ngrams_{n}.json"
    if cache_path.is_file():
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        return DecontamIndex((tuple(g) for g in saved["ngrams"]), n=n)

    texts: list[str] = []
    for name, bcfg in sorted(decontam_cfg.get("benchmarks", {}).items()):
        if bcfg.get("repo") != "local":
            check_source(name, bcfg)
        dest = bench_root / name
        cols = list(bcfg["columns"])
        for path_in_repo, _size in list_parquet(bcfg, local_dir=dest):
            local_cand = dest / path_in_repo
            local = local_cand if local_cand.is_file() else download(bcfg, path_in_repo, dest)
            table = pq.read_table(local)
            missing = set(cols) - set(table.schema.names)
            if missing:
                raise ConfigError(f"benchmark {name} ({local}) missing columns {sorted(missing)}")
            rows = zip(*(table.column(c).to_pylist() for c in cols))
            for row in rows:
                parts = [str(x) for x in row if x]
                texts.extend(parts)
                if len(parts) > 1:
                    texts.append("\n".join(parts))

    index = DecontamIndex.from_texts(texts, n=n)
    bench_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(cache_path, {"ngram": n, "count": index.total_ngrams, "ngrams": index.to_list()})
    return index
