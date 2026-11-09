"""Command line: python -m alethic <command>.

Commands: tokenizer-build, data-fetch, data-build, train, eval.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from alethic.config import ConfigError, load_config, resolve_path

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"


def cmd_tokenizer_build(args: argparse.Namespace) -> int:
    """Run the tokenizer pipeline (M1): sample, train, eval, freeze."""
    from alethic.tokenizer import build

    only = [s.strip() for s in args.streams.split(",")] if args.streams else None
    specials = load_config(args.train)["paths"]["specials"]
    return build.run(load_config(args.tokenizer), load_config(args.data), Path(args.tokenizer),
                     specials, args.stage, args.size_bytes, only, args.threads)


def cmd_data_fetch(args: argparse.Namespace) -> int:
    """Download pinned parquet files for each source up to target_scale (M2)."""
    from alethic.data.encode import MEASURED_BPT, _stream_targets
    from alethic.data.fetch import fetch_source_parquet

    data_cfg = load_config(args.data)
    raw_root = resolve_path(data_cfg["paths"]["raw"])
    scale = float(args.target_scale if args.target_scale is not None else data_cfg.get("target_scale", 1.0))
    overfetch = max(float(data_cfg.get("overfetch", 1.15)), 1.25)
    for stream, source, paths_ov, target_tok in _stream_targets(data_cfg, scale):
        src = data_cfg["sources"][source]
        bpt = MEASURED_BPT.get(source, float(src["bytes_per_token"]))
        files = fetch_source_parquet(source, src, raw_root, int(target_tok * bpt * overfetch), paths_ov)
        print(f"fetched {stream}: {len(files)} parquet file(s)")
    return 0


def cmd_data_build(args: argparse.Namespace,
                   on_checkpoint: Callable[[str], None] | None = None) -> int:
    """Dedup, decontaminate, encode, and shard the corpus (M2/M3)."""
    from alethic.data.encode import encode_units

    manifest = encode_units(load_config(args.data), target_scale=args.target_scale,
                            workers=args.workers, on_checkpoint=on_checkpoint)
    print(f"built shards: {manifest['total_tokens']:,} total tokens "
          f"(train {manifest['train_tokens']:,}, val {manifest['val_tokens']:,}), "
          f"uzbek share {manifest['uzbek_combined_share']:.4f}")
    return 0


def cmd_train(args: argparse.Namespace,
              on_checkpoint: Callable[[Path], None] | None = None) -> int:
    """Check the frozen tokenizer and the shard manifest, then train (or resume)."""
    from alethic.data.loader import train_shards_from_manifest
    from alethic.tokenizer.freeze import verify_frozen
    from alethic.train.loop import RunSpec, train

    model_cfg = load_config(args.model)
    train_cfg = load_config(args.train)
    if getattr(args, "steps", None):
        train_cfg["steps"] = int(args.steps)
    if getattr(args, "peak_lr", None):
        train_cfg["optim"]["peak_lr"] = float(args.peak_lr)
        train_cfg["optim"]["min_lr"] = float(args.peak_lr) / 10.0
    if getattr(args, "warmup_steps", None) is not None:
        train_cfg["optim"]["warmup_steps"] = int(args.warmup_steps)
    paths = train_cfg["paths"]
    _, tokenizer_sha = verify_frozen(resolve_path(paths["tokenizer"]))
    shards, manifest = train_shards_from_manifest(resolve_path(paths["shards"]))
    if manifest.get("tokenizer_sha256") != tokenizer_sha:
        raise ConfigError(
            "The shards were built with a different tokenizer (manifest tokenizer_sha256 "
            f"{manifest.get('tokenizer_sha256')} != frozen {tokenizer_sha}). Rebuild the shards."
        )
    run_name = args.run_name or model_cfg["name"]
    spec = RunSpec(
        model_cfg=model_cfg, train_cfg=train_cfg, shards=shards,
        run_dir=resolve_path(paths["runs"]) / run_name,
        specials_path=resolve_path(paths["specials"]),
        tokenizer_sha256=tokenizer_sha,
        device=args.device or train_cfg["device"],
        resume=None if args.resume == "none" else args.resume,
        max_cost=args.max_cost, stop_step=args.stop_step,
        on_checkpoint=on_checkpoint,
    )
    summary = train(spec)
    print(f"finished: step {summary['step']}/{summary['steps']}, reason {summary['reason']}, "
          f"spent ${summary['dollars_spent']:.2f}, checkpoint {summary['checkpoint']}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate validation bits per byte on a checkpoint (M2/M4)."""
    import torch
    from alethic.eval.bpb import evaluate_bpb
    from alethic.model.transformer import ModelConfig, Transformer
    from alethic.train import checkpoint as ckpt

    model_cfg = load_config(args.model)
    train_cfg = load_config(args.train)
    if getattr(args, "steps", None):
        train_cfg["steps"] = int(args.steps)
    if getattr(args, "peak_lr", None):
        train_cfg["optim"]["peak_lr"] = float(args.peak_lr)
        train_cfg["optim"]["min_lr"] = float(args.peak_lr) / 10.0
    if getattr(args, "warmup_steps", None) is not None:
        train_cfg["optim"]["warmup_steps"] = int(args.warmup_steps)
    paths = train_cfg["paths"]
    device = args.device or train_cfg["device"]
    if args.checkpoint:
        folder = Path(args.checkpoint)
    else:
        run_name = args.run_name or model_cfg["name"]
        latest = ckpt.latest_checkpoint(resolve_path(paths["runs"]) / run_name / "checkpoints")
        if latest is None:
            raise ConfigError(f"no checkpoints found for run {run_name!r}")
        folder = latest
    state, _ = ckpt.load_checkpoint(folder)
    model = Transformer(ModelConfig.from_dict(model_cfg)).to(device)
    model.load_state_dict(state["model"])
    res = evaluate_bpb(
        model, resolve_path(paths["shards"]), int(train_cfg["seq_len"]),
        int(train_cfg["micro_batch"]), device, train_cfg.get("precision") == "bf16",
        max_tokens_per_source=None if args.full else 2_000_000,
    )
    out = {"step": state["step"], "checkpoint": str(folder), **res}
    print(json.dumps(out))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Define every command and its options."""
    parser = argparse.ArgumentParser(prog="python -m alethic")
    sub = parser.add_subparsers(dest="command", required=True)
    tb = sub.add_parser("tokenizer-build", help="sample, train, evaluate, freeze (M1)")
    tb.add_argument("--stage", default="all", choices=["all", "sample", "train", "eval", "freeze"])
    tb.add_argument("--streams", default=None, help="comma list of sample streams to build now")
    tb.add_argument("--size-bytes", type=int, default=None, help="override sample.size_bytes")
    tb.add_argument("--threads", type=int, default=None, help="override sentencepiece num_threads")
    tb.add_argument("--tokenizer", default=str(CONFIGS / "tokenizer.yaml"))
    tb.add_argument("--data", default=str(CONFIGS / "data.yaml"))
    tb.add_argument("--train", default=str(CONFIGS / "train.yaml"))
    tb.set_defaults(func=cmd_tokenizer_build)

    df = sub.add_parser("data-fetch", help="pin and download sources (M2)")
    df.add_argument("--data", default=str(CONFIGS / "data.yaml"))
    df.add_argument("--target-scale", type=float, default=None, help="fraction of 6B target to fetch")
    df.set_defaults(func=cmd_data_fetch)

    db = sub.add_parser("data-build", help="dedup, decontaminate, encode, shard (M2)")
    db.add_argument("--data", default=str(CONFIGS / "data.yaml"))
    db.add_argument("--target-scale", type=float, default=None, help="fraction of 6B target to build")
    db.add_argument("--workers", type=int, default=None, help="worker processes (default: cpu_count)")
    db.set_defaults(func=cmd_data_build)

    ev = sub.add_parser("eval", help="validation bits per byte (M2/M4)")
    ev.add_argument("--model", default=str(CONFIGS / "model_sgca.yaml"))
    ev.add_argument("--train", default=str(CONFIGS / "train.yaml"))
    ev.add_argument("--run-name", default=None, help="run folder under paths.runs")
    ev.add_argument("--checkpoint", default=None, help="specific checkpoint folder (default: latest)")
    ev.add_argument("--device", default=None, help="override train.yaml device")
    ev.add_argument("--full", action="store_true", help="evaluate all validation shards instead of 2M/source")
    ev.set_defaults(func=cmd_eval)

    t = sub.add_parser("train", help="train or resume one model")
    t.add_argument("--model", default=str(CONFIGS / "model_sgca.yaml"), help="model config YAML")
    t.add_argument("--train", default=str(CONFIGS / "train.yaml"), help="training config YAML")
    t.add_argument("--run-name", default=None, help="folder under paths.runs (default: model name)")
    t.add_argument("--resume", default="none", help="'auto' (latest checkpoint), 'none', or a checkpoint folder")
    t.add_argument("--max-cost", type=float, default=None, help="dollars; save a checkpoint and stop when reached")
    t.add_argument("--stop-step", type=int, default=None, help="stop after this many total steps (schedule unchanged)")
    t.add_argument("--steps", type=int, default=None, help="override train.yaml steps (changes cosine schedule length)")
    t.add_argument("--peak-lr", type=float, default=None, help="override optim.peak_lr (and min_lr = peak_lr / 10)")
    t.add_argument("--warmup-steps", type=int, default=None, help="override optim.warmup_steps")
    t.add_argument("--device", default=None, help="override train.yaml device, e.g. cpu")
    t.set_defaults(func=cmd_train)
    return parser


def main(argv: list[str] | None = None,
         on_checkpoint: Callable | None = None) -> int:
    """Parse arguments and run the command; known errors print a message instead of a traceback."""
    from alethic.data.loader import LoaderError
    from alethic.tokenizer.freeze import FrozenError
    from alethic.tokenizer.specials import SpecialsError

    known: list[type[BaseException]] = [ConfigError, FrozenError, LoaderError, SpecialsError]
    try:   # the training errors need torch, which the data/tokenizer machines do not install
        from alethic.train.checkpoint import CheckpointError
        from alethic.train.loop import TrainError
        known += [CheckpointError, TrainError]
    except ImportError:
        pass

    args = build_parser().parse_args(argv)
    try:
        if on_checkpoint is not None and args.func in (cmd_data_build, cmd_train):
            return args.func(args, on_checkpoint=on_checkpoint)
        return args.func(args)
    except tuple(known) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
