"""Train the SentencePiece Unigram tokenizer from a one-line-per-source-line sample file.

M0 builds the training call and its option checks. Building the 2 GB source-weighted
sample from the fetched data (plan, Tokenizer step 1) is milestone M1.
"""
from __future__ import annotations

import time
from pathlib import Path

import sentencepiece as spm

from alethic.tokenizer.specials import all_slot_pieces

# Settings the plan fixes; train_tokenizer refuses a config that changes any of them.
REQUIRED_OPTIONS = {
    "model_type": "unigram",
    "normalization_rule_name": "identity",
    "remove_extra_whitespaces": False,
    "add_dummy_prefix": False,
    "split_by_whitespace": False,
    "allow_whitespace_only_pieces": True,
    "byte_fallback": True,
    "character_coverage": 1.0,
    "shuffle_input_sentence": True,
    "unk_id": 0,
    "bos_id": 1,
    "eos_id": 2,
    "pad_id": 3,
}


class TokenizerConfigError(RuntimeError):
    """tokenizer.yaml contradicts a setting the plan fixes."""


def spm_options(cfg: dict, input_path: str | Path, model_prefix: str | Path,
                overrides: dict | None = None) -> dict:
    """Build the keyword arguments for SentencePieceTrainer.train from tokenizer.yaml.

    `overrides` may change only size/speed settings (vocab_size, num_threads, ...), which the
    tests use to train a tiny tokenizer with the real settings.
    """
    opts = dict(cfg["sentencepiece"])
    opts.update(overrides or {})
    for key, want in REQUIRED_OPTIONS.items():
        if opts.get(key) != want:
            raise TokenizerConfigError(f"sentencepiece.{key} must be {want!r}, got {opts.get(key)!r}")
    if "user_defined_symbols" in opts:
        raise TokenizerConfigError("set extra symbols through whitespace_symbols, not user_defined_symbols")
    if not opts["vocab_size"] < 65536:
        raise TokenizerConfigError("vocab_size must be below 65,536 so ids fit uint16")
    extra = list(cfg.get("whitespace_symbols") or [])
    if any(" " in s for s in extra):
        # Verified on sentencepiece 0.2.3: symbols with real spaces are accepted but never match,
        # because input spaces become U+2581 before matching. Write runs as U+2581 instead.
        raise TokenizerConfigError("write whitespace_symbols with U+2581 ('▁'), not spaces")
    # Slots first, so <sp00>..<sp63> get ids 4..67 right after the control ids.
    opts["user_defined_symbols"] = all_slot_pieces() + extra
    opts["input"] = str(input_path)
    opts["model_prefix"] = str(model_prefix)
    return opts


def train_tokenizer(cfg: dict, input_path: str | Path, out_dir: str | Path,
                    overrides: dict | None = None) -> tuple[Path, float]:
    """Train on input_path and write <name>.model / <name>.vocab into out_dir.

    Returns the model path and the training time in seconds.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / cfg["name"]
    opts = spm_options(cfg, input_path, prefix, overrides)
    spm.set_random_generator_seed(int(cfg["seed"]))
    start = time.perf_counter()
    spm.SentencePieceTrainer.train(**opts)
    return Path(f"{prefix}.model"), time.perf_counter() - start


def build_sample(*_args, **_kwargs) -> None:
    """Write the shuffled, source-weighted training sample (milestone M1)."""
    raise NotImplementedError("The tokenizer sample builder is part of milestone M1.")
