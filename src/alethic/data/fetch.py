"""Download parquet files with huggingface_hub, pinned to a commit, and read documents from them.

Files are downloaded one by one in sorted order to $ALETHIC_DATA_DIR/raw/<source>/ (never with
streaming), so M1 and M2 reuse the same downloads. HF_TOKEN is read from the environment by
huggingface_hub; this module never prints or logs it. Nothing here uploads anything.
"""
from __future__ import annotations

import random
import re
import time
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from alethic.config import ConfigError
from alethic.data.uzbek import normalize_uzbek

UZBEK_SOURCES = ("uzbek_latn", "uzbek_cyrl")
_SHA = re.compile(r"^[0-9a-f]{40}$")


def check_source(name: str, src: dict) -> None:
    """Stop unless the source is pinned to a full commit hash (and other code is approved)."""
    rev = src.get("revision")
    if not isinstance(rev, str) or not _SHA.match(rev):
        raise ConfigError(f"data.yaml sources.{name}.revision is {rev!r}; pin it to a 40-character "
                          "commit hash (HfApi().dataset_info(repo).sha) before downloading.")
    if name == "code_other" and src.get("approved") is not True:
        raise ConfigError("sources.code_other.approved is not true; the owner must approve the "
                          "other-code language list in data.yaml before anything is downloaded.")


def list_parquet(src: dict, paths: list[str] | None = None) -> list[tuple[str, int]]:
    """(path in repo, size in bytes) of every parquet file under the source's folders, sorted."""
    api = HfApi()
    found = []
    for folder in paths or src["paths"]:
        for f in api.list_repo_tree(src["repo"], repo_type="dataset", revision=src["revision"],
                                    path_in_repo=folder, recursive=True):
            if f.path.endswith(".parquet"):
                found.append((f.path, int(f.size)))
    if not found:
        raise ConfigError(f"no parquet files in {src['repo']} under {paths or src['paths']}")
    return sorted(found)


def download(src: dict, path_in_repo: str, dest: Path, tries: int = 4) -> Path:
    """Download one file into dest (kept if already complete); retry network errors with backoff."""
    for attempt in range(1, tries + 1):
        try:
            return Path(hf_hub_download(src["repo"], path_in_repo, repo_type="dataset",
                                        revision=src["revision"], local_dir=str(dest)))
        except Exception as err:     # network hiccups; gated-access errors repeat and then raise
            if attempt == tries:
                raise
            wait = 5 * 2 ** attempt
            print(f"download {path_in_repo} failed ({type(err).__name__}); retry in {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def read_documents(path: Path, source: str, text_col: str, key_col: str, seed: int,
                   stats: dict) -> Iterator[tuple[str, str]]:
    """Yield (group key, text) from one parquet file, row groups in a seeded random order.

    Uzbek sources are cleaned with normalize_uzbek(); stats["uzbek_chars_changed"] counts edits.
    """
    pf = pq.ParquetFile(path)
    missing = {text_col, key_col} - set(pf.schema_arrow.names)
    if missing:
        raise ConfigError(f"{path} has no column(s) {sorted(missing)}; columns are {pf.schema_arrow.names}")
    order = list(range(pf.num_row_groups))
    random.Random(f"{seed}:{path.name}").shuffle(order)
    for rg in order:
        table = pf.read_row_group(rg, columns=[key_col, text_col])
        for key, text in zip(table.column(key_col).to_pylist(), table.column(text_col).to_pylist()):
            if not text:
                continue
            if source in UZBEK_SOURCES:
                text, changed = normalize_uzbek(text)
                stats["uzbek_chars_changed"] = stats.get("uzbek_chars_changed", 0) + changed
            yield str(key or ""), text


def iter_source(name: str, src: dict, raw_root: Path, seed: int, stats: dict,
                paths: list[str] | None = None) -> Iterator[tuple[str, str]]:
    """Yield documents of a source, downloading its files lazily in sorted order.

    stats collects files and bytes downloaded; `paths` limits it to some folders (one language).
    """
    check_source(name, src)
    dest = raw_root / name
    for path_in_repo, size in list_parquet(src, paths):
        local = download(src, path_in_repo, dest)
        stats.setdefault("files", []).append(path_in_repo)
        stats["parquet_bytes"] = stats.get("parquet_bytes", 0) + size
        yield from read_documents(local, name, src["text_column"], src["key_column"], seed, stats)
