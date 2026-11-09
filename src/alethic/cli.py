"""Command line: python -m alethic <command>.

Commands: tokenizer-build, data-fetch, data-build, train, eval.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alethic.config import ConfigError, load_config, resolve_path

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"


def _not_built(milestone: str):
    """Return a handler for a command that a later milestone implements."""
    def handler(_args: argparse.Namespace) -> int:
        print(f"This command is built in milestone {milestone}.", file=sys.stderr)
        return 2
    return handler


def cmd_tokenizer_build(args: argparse.Namespace) -> int:
    """Run the tokenizer pipeline (M1): sample, train, eval, freeze."""
    from alethic.tokenizer import build

    only = [s.strip() for s in args.streams.split(",")] if args.streams else None
    specials = load_config(args.train)["paths"]["specials"]
    return build.run(load_config(args.tokenizer), load_config(args.data), Path(args.tokenizer),
                     specials, args.stage, args.size_bytes, only, args.threads)


def cmd_train(args: argparse.Namespace) -> int:
    """Check the frozen tokenizer and the shard manifest, then train (or resume)."""
    from alethic.data.loader import train_shards_from_manifest
    from alethic.tokenizer.freeze import verify_frozen
    from alethic.train.loop import RunSpec, train

    model_cfg = load_config(args.model)
    train_cfg = load_config(args.train)
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
    )
    summary = train(spec)
    print(f"finished: step {summary['step']}/{summary['steps']}, reason {summary['reason']}, "
          f"spent ${summary['dollars_spent']:.2f}, checkpoint {summary['checkpoint']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Define every command and its options."""
    parser = argparse.ArgumentParser(prog="python -m alethic")
    sub = parser.add_subparsers(dest="command", required=True)
    tb = sub.add_parser("tokenizer-build", help="sample, train, evaluate, freeze (M1)")
    tb.add_argument("--stage", default="all", choices=["all", "sample", "train", "eval", "freeze"])
    tb.add_argument("--streams", default=None, help="comma list of sample streams to build now, "
                    "e.g. fineweb_edu,math,uzbek_latn,uzbek_cyrl")
    tb.add_argument("--size-bytes", type=int, default=None, help="override sample.size_bytes")
    tb.add_argument("--threads", type=int, default=None, help="override sentencepiece num_threads")
    tb.add_argument("--tokenizer", default=str(CONFIGS / "tokenizer.yaml"))
    tb.add_argument("--data", default=str(CONFIGS / "data.yaml"))
    tb.add_argument("--train", default=str(CONFIGS / "train.yaml"))
    tb.set_defaults(func=cmd_tokenizer_build)
    sub.add_parser("data-fetch", help="pin and download sources (M2)").set_defaults(func=_not_built("M2"))
    sub.add_parser("data-build", help="dedup, decontaminate, encode, shard (M2)").set_defaults(func=_not_built("M2"))
    sub.add_parser("eval", help="validation bits per byte (M2/M4)").set_defaults(func=_not_built("M2/M4"))

    t = sub.add_parser("train", help="train or resume one model")
    t.add_argument("--model", default=str(CONFIGS / "model_sgca.yaml"), help="model config YAML")
    t.add_argument("--train", default=str(CONFIGS / "train.yaml"), help="training config YAML")
    t.add_argument("--run-name", default=None, help="folder under paths.runs (default: model name)")
    t.add_argument("--resume", default="none", help="'auto' (latest checkpoint), 'none', or a checkpoint folder")
    t.add_argument("--max-cost", type=float, default=None, help="dollars; save a checkpoint and stop when reached")
    t.add_argument("--stop-step", type=int, default=None, help="stop after this many total steps (schedule unchanged)")
    t.add_argument("--device", default=None, help="override train.yaml device, e.g. cpu")
    t.set_defaults(func=cmd_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the command; known errors print a message instead of a traceback."""
    from alethic.data.loader import LoaderError
    from alethic.tokenizer.freeze import FrozenError
    from alethic.tokenizer.specials import SpecialsError
    from alethic.train.checkpoint import CheckpointError
    from alethic.train.loop import TrainError

    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, FrozenError, LoaderError, SpecialsError, CheckpointError, TrainError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
