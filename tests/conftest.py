"""Shared test fixtures: a tiny tokenizer trained with the real settings, tiny models and shards."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from alethic.config import load_config, write_json_atomic
from alethic.data.shards import to_tokens
from alethic.tokenizer.codec import Codec
from alethic.tokenizer.train import train_tokenizer

TESTS = Path(__file__).resolve().parent
CONFIGS = TESTS.parent / "configs"
sys.path.insert(0, str(TESTS))   # lets tests import corpus.py and shard_proc.py

TINY_VOCAB = 1024


@pytest.fixture(scope="session")
def tokenizer_cfg() -> dict:
    """configs/tokenizer.yaml as a dict."""
    return load_config(CONFIGS / "tokenizer.yaml")


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory, tokenizer_cfg) -> Path:
    """Train a 1,024-id tokenizer once per session with the real settings (only size/threads changed)."""
    from corpus import make_corpus
    work = tmp_path_factory.mktemp("tokenizer")
    corpus = make_corpus(work / "corpus.txt")
    model_path, _seconds = train_tokenizer(
        tokenizer_cfg, corpus, work / "model",
        overrides={"vocab_size": TINY_VOCAB, "num_threads": 4},
    )
    return model_path


@pytest.fixture
def tiny_codec(tiny_model_path, tmp_path) -> Codec:
    """A Codec on the tiny model with a fresh specials.json in tmp_path."""
    return Codec(tiny_model_path, specials_path=tmp_path / "specials.json")


def tiny_model_cfg(attention: str, vocab_size: int = 128, max_seq_len: int = 64) -> dict:
    """A 2-layer model config with the same structure as the real ones."""
    cfg = {"name": f"tiny_{attention}", "attention": attention, "n_layers": 2, "d_model": 32,
           "n_heads": 4, "n_kv_heads": 2, "head_dim": 8, "ffn_hidden": 64 if attention == "gqa" else 48,
           "vocab_size": vocab_size, "max_seq_len": max_seq_len, "rope_theta": 10000.0,
           "norm_eps": 1e-6, "init_std": 0.02}
    if attention == "sgca":
        cfg["alpha_init"] = 1.0
    return cfg


def tiny_train_cfg(steps: int = 20, **changes) -> dict:
    """A CPU training config: fp32, no compile, 4 x 16-token sequences per step."""
    cfg = {"seed": 7, "device": "cpu", "precision": "fp32", "compile": False,
           "seq_len": 16, "batch_seqs": 4, "micro_batch": 2, "total_tokens": steps * 64, "steps": steps,
           "optim": {"peak_lr": 3e-3, "min_lr": 3e-4, "warmup_steps": 3, "betas": [0.9, 0.95],
                     "eps": 1e-8, "weight_decay": 0.1, "grad_clip": 1.0},
           "checkpoint": {"every": 5, "keep_last": 2, "keep_every": 10},
           "log_every": 5, "eval_every": 1000,
           "cost": {"price_per_hour": 3.0, "peak_flops": 1.0e12},
           "paths": {"tokenizer": "tokenizer/alethic-tokenizer-v1", "shards": "shards",
                     "runs": "runs", "specials": "tokenizer/specials.json"}}
    cfg.update(changes)
    return cfg


def write_train_shards(shards_dir: Path, n_tokens: int, vocab: int, seed: int = 0,
                       tokenizer_sha256: str = "0" * 64, n_shards: int = 3) -> list[Path]:
    """Write random-token train shards plus a manifest.json like the data stage will."""
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab, size=n_tokens)
    cuts = np.linspace(0, n_tokens, n_shards + 1).astype(int)
    records, paths = [], []
    for i in range(n_shards):
        rel = f"python/test-{i:04d}.bin"
        p = shards_dir / "train" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(to_tokens(tokens[cuts[i]:cuts[i + 1]]).tobytes())
        records.append({"path": rel, "split": "train", "tokens": int(cuts[i + 1] - cuts[i])})
        paths.append(p)
    write_json_atomic(shards_dir / "manifest.json", {"tokenizer_sha256": tokenizer_sha256, "shards": records})
    return paths
