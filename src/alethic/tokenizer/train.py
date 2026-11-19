"""Train the SentencePiece Unigram tokenizer from a one-line-per-source-line sample file.

M0 builds the training call and its option checks. Building the 2 GB source-weighted
sample from the fetched data (plan, Tokenizer step 1) is milestone M1.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Callable

import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as spm_pb2

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


# Escape that keeps a literal U+2581 exact. SentencePiece uses U+2581 as its own space marker,
# so a real one would decode as " ". Before encoding, a real U+2581 becomes U+E000 "0" and a real
# U+E000 becomes U+E000 "1"; decoding reverses both. Like "\\" in strings, so it is lossless.
# Both rule sets are stored inside the .model file (verified on sentencepiece 0.2.3).
NORMALIZE_RULES = "2581\tE000 30\nE000\tE000 31\n"
DENORMALIZE_RULES = "E000 30\t2581\nE000 31\tE000\n"


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
    if cfg.get("u2581_escape", True):
        # Identity plus the U+2581 escape: the rule files replace the "identity" rule name.
        out = Path(model_prefix).parent
        (out / "normalize.tsv").write_text(NORMALIZE_RULES, encoding="utf-8")
        (out / "denormalize.tsv").write_text(DENORMALIZE_RULES, encoding="utf-8")
        del opts["normalization_rule_name"]
        opts["normalization_rule_tsv"] = str(out / "normalize.tsv")
        opts["denormalization_rule_tsv"] = str(out / "denormalize.tsv")
    return opts


def export_seeds(model_path: Path, seeds_path: Path, scale_chars: int = 10**9) -> int:
    """Export learned multi-character pieces from a checkpoint .model into a SentencePiece seed TSV."""
    mp = spm_pb2.ModelProto()
    mp.ParseFromString(model_path.read_bytes())
    normal = spm_pb2.ModelProto.SentencePiece.NORMAL
    char_prob = sum(math.exp(p.score) for p in mp.pieces if p.type == normal and len(p.piece) == 1)
    scale = max(scale_chars, 10**6) / max(char_prob, 1e-6)
    lines = [
        f"{p.piece}\t{max(1, int(round(math.exp(p.score) * scale)))}"
        for p in mp.pieces
        if p.type == normal and len(p.piece) > 1 and not any(c in p.piece for c in "\t\r\n\0")
    ]
    tmp = seeds_path.with_name(seeds_path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, seeds_path)
    return len(lines)


def _run_spm_atomic(opts: dict, prefix: Path, seed: int) -> None:
    """Train SentencePiece into prefix.tmp.{model,vocab}, fsync, then rename to prefix.{model,vocab}."""
    tmp_prefix = prefix.with_name(prefix.name + ".tmp")
    for ext in (".model", ".vocab"):
        Path(f"{tmp_prefix}{ext}").unlink(missing_ok=True)
    run_opts = dict(opts, model_prefix=str(tmp_prefix))
    spm.set_random_generator_seed(seed)
    spm.SentencePieceTrainer.train(**run_opts)
    for ext in (".vocab", ".model"):            # .model renamed last: its presence marks the step done
        src, dst = Path(f"{tmp_prefix}{ext}"), Path(f"{prefix}{ext}")
        with open(src, "rb+") as f:
            os.fsync(f.fileno())
        os.replace(src, dst)


def train_tokenizer(cfg: dict, input_path: str | Path, out_dir: str | Path,
                    overrides: dict | None = None, work_dir: str | Path | None = None,
                    on_checkpoint: Callable[[str], None] | None = None) -> tuple[Path, float]:
    """Train on input_path and write <name>.model / <name>.vocab into out_dir with EM checkpoints.

    Intermediate vocab sizes in `checkpoint_vocabs` are saved under work_dir/spm_checkpoints/
    and committed via `on_checkpoint`, so a killed run resumes from the last finished checkpoint.
    """
    ov = dict(overrides or {})
    raw_ckpts = ov.pop("checkpoint_vocabs", cfg.get("checkpoint_vocabs") if "vocab_size" not in ov else None)
    prefix = Path(out_dir) / cfg["name"]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    base_opts = spm_options(cfg, input_path, prefix, ov)
    target_vocab = int(base_opts["vocab_size"])
    ckpts = sorted([int(v) for v in (raw_ckpts or []) if int(v) > target_vocab], reverse=True)
    ckpt_dir = Path(work_dir or out_dir) / "spm_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    sample_bytes = Path(input_path).stat().st_size
    start = time.perf_counter()

    prev_model: Path | None = None
    for v in ckpts:
        ckpt_prefix = ckpt_dir / f"ckpt_{v}"
        ckpt_model = Path(f"{ckpt_prefix}.model")
        if ckpt_model.exists():
            print(f"[train] resuming from checkpoint {ckpt_model.name}")
            prev_model = ckpt_model
            continue
        step_opts = dict(base_opts, vocab_size=v, hard_vocab_limit=False)
        if prev_model is not None:
            seeds_tsv = ckpt_dir / f"seeds_from_{prev_model.stem}.tsv"
            n_seeds = export_seeds(prev_model, seeds_tsv, sample_bytes)
            print(f"[train] loaded {n_seeds:,} seed pieces from {prev_model.name} -> target {v:,}")
            step_opts["seed_sentencepieces_file"] = str(seeds_tsv)
        _run_spm_atomic(step_opts, ckpt_prefix, seed)
        prev_model = ckpt_model
        print(f"[train] saved checkpoint {ckpt_model.name} ({time.perf_counter() - start:.1f}s)")
        if on_checkpoint:
            on_checkpoint(f"ckpt_{v}")

    final_opts = dict(base_opts)
    if prev_model is not None:
        seeds_tsv = ckpt_dir / f"seeds_from_{prev_model.stem}.tsv"
        n_seeds = export_seeds(prev_model, seeds_tsv, sample_bytes)
        print(f"[train] loaded {n_seeds:,} seed pieces from {prev_model.name} -> final {target_vocab:,}")
        final_opts["seed_sentencepieces_file"] = str(seeds_tsv)
    _run_spm_atomic(final_opts, prefix, seed)
    if on_checkpoint:
        on_checkpoint("model")
    return Path(f"{prefix}.model"), time.perf_counter() - start
