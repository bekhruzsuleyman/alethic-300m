"""Download parquet files with huggingface_hub, pinned to a commit (milestone M2).

Will: resolve and pin each dataset's commit hash into data.yaml; list real subset and
column names for the owner; download files one by one in sorted order to
$ALETHIC_DATA_DIR/raw/<source>/; stop a source at 1.15 x its estimated byte target.
Never uses streaming, never uploads, never prints HF_TOKEN.
"""
from __future__ import annotations


def fetch_sources(*_args, **_kwargs) -> None:
    """Download every configured source (milestone M2)."""
    raise NotImplementedError("Fetching is part of milestone M2.")
