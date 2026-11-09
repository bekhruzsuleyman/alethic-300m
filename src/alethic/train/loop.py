"""The training loop: deterministic data order, grad accumulation, checkpoints, cost guard, resume."""
from __future__ import annotations

import json
import math
import platform
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from alethic.config import config_hash
from alethic.data.loader import WindowLoader
from alethic.model.params import count_parameters, flops_per_token
from alethic.model.transformer import ModelConfig, Transformer
from alethic.tokenizer.specials import SpecialsRegistry, check_append_only, read_names
from alethic.train import checkpoint as ckpt
from alethic.train.cost import CostGuard, mfu
from alethic.train.optim import build_optimizer, lr_at


class TrainError(RuntimeError):
    """The run cannot start or resume safely."""


@dataclass
class RunSpec:
    """Everything one `train` invocation needs."""
    model_cfg: dict
    train_cfg: dict
    shards: list[Path]
    run_dir: Path
    specials_path: Path
    tokenizer_sha256: str
    device: str
    resume: str | None = None        # None, "auto", or a checkpoint folder
    max_cost: float | None = None    # dollars; save and stop once reached
    stop_step: int | None = None     # stop (with a checkpoint) after this many total steps


def total_steps(train_cfg: dict) -> int:
    """Steps for the run: `steps` if set, else ceil(total_tokens / tokens per step)."""
    if train_cfg.get("steps"):
        return int(train_cfg["steps"])
    per_step = int(train_cfg["seq_len"]) * int(train_cfg["batch_seqs"])
    return math.ceil(int(train_cfg["total_tokens"]) / per_step)


def versions() -> dict:
    """Software versions recorded with every run (plan rule 8)."""
    import sentencepiece
    out = {"python": platform.python_version(), "torch": torch.__version__,
           "numpy": np.__version__, "sentencepiece": sentencepiece.__version__}
    if torch.cuda.is_available():
        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


def rng_state() -> dict:
    """All random-number states, in a form torch.load(weights_only=True) accepts."""
    np_state = np.random.get_state()
    state = {"python": random.getstate(), "torch": torch.get_rng_state(),
             "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]]}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    """Restore the states saved by rng_state()."""
    py = state["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    torch.set_rng_state(state["torch"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class StopFlag:
    """Catches SIGINT / SIGTERM so the loop can save a checkpoint before exiting."""

    def __init__(self) -> None:
        """Install the handlers (main thread only) and remember the old ones."""
        self.reason: str | None = None
        self._old = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._old[sig] = signal.signal(sig, self._handle)
            except ValueError:   # not in the main thread
                pass

    def _handle(self, signum, _frame) -> None:
        """Record the signal; the loop checks it after each step."""
        self.reason = f"signal {signal.Signals(signum).name}"

    def restore(self) -> None:
        """Put the old handlers back."""
        for sig, old in self._old.items():
            signal.signal(sig, old)


def _identity(spec: RunSpec, steps: int) -> dict:
    """Settings that must not change between a checkpoint and its resume."""
    tc = spec.train_cfg
    shards = [(f"{p.parent.name}/{p.name}", p.stat().st_size) for p in spec.shards]
    return {"model_config_sha256": config_hash(spec.model_cfg),
            "data_order": {"seed": int(tc["seed"]), "seq_len": int(tc["seq_len"]),
                           "batch_seqs": int(tc["batch_seqs"]), "steps": steps},
            "shards_sha256": config_hash({"shards": shards}),
            "tokenizer_sha256": spec.tokenizer_sha256}


def _pick_resume(spec: RunSpec, ckpt_root: Path) -> Path | None:
    """The checkpoint to resume from, or None for a fresh start."""
    latest = ckpt.latest_checkpoint(ckpt_root)
    if spec.resume is None:
        if latest is not None:
            raise TrainError(f"{ckpt_root} already has checkpoints; pass --resume auto or use a new --run-name")
        return None
    if spec.resume == "auto":
        return latest
    return Path(spec.resume)


def train(spec: RunSpec) -> dict:
    """Run (or resume) training; returns a summary with the final step and the stop reason."""
    session_t0 = time.perf_counter()     # the rental bills from process start, so the guard does too
    tc = spec.train_cfg
    mcfg = ModelConfig.from_dict(spec.model_cfg)
    device, seed = spec.device, int(tc["seed"])
    seq_len, batch_seqs, micro = int(tc["seq_len"]), int(tc["batch_seqs"]), int(tc["micro_batch"])
    if batch_seqs % micro:
        raise TrainError(f"batch_seqs {batch_seqs} must be a multiple of micro_batch {micro}")
    steps = total_steps(tc)
    stop_at = min(steps, spec.stop_step) if spec.stop_step else steps
    run_dir, ckpt_root = Path(spec.run_dir), Path(spec.run_dir) / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt.remove_stray_tmp(ckpt_root)
    SpecialsRegistry(spec.specials_path)   # validates specials.json (creates it with the initial names)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = Transformer(mcfg).to(device)
    n_params = count_parameters(model)
    opt = build_optimizer(model, tc["optim"], device)
    loader = WindowLoader(spec.shards, seq_len, batch_seqs, seed)
    identity = _identity(spec, steps)
    step, tokens_seen, base_wall = 0, 0, 0.0

    folder = _pick_resume(spec, ckpt_root)
    if folder is not None:
        state, manifest = ckpt.load_checkpoint(folder)
        for key, value in identity.items():
            if manifest.get(key) != value:
                raise TrainError(f"cannot resume: {key} differs from checkpoint {folder}")
        check_append_only(read_names(folder / "specials.json"), read_names(spec.specials_path))
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        set_rng_state(state["rng"])
        step, tokens_seen, base_wall = state["step"], state["tokens_seen"], state["wall_seconds"]
        print(f"resumed from {folder} at step {step}")

    info = dict(identity, versions=versions(), n_params=n_params, model=mcfg.name)
    with open(run_dir / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"start_step": step, **info}) + "\n")
    print(json.dumps({"start_step": step, "steps": steps, **info}))

    fwd = torch.compile(model) if tc.get("compile") else model
    use_bf16 = tc.get("precision") == "bf16"
    guard = CostGuard(float(tc["cost"]["price_per_hour"]), spec.max_cost)
    fpt = flops_per_token(n_params, mcfg.n_layers, mcfg.d_model, seq_len)
    o = tc["optim"]
    tokens_per_step = seq_len * batch_seqs
    log_every = int(tc["log_every"])
    wall = base_wall

    def save(at_step: int) -> Path:
        """Write a checkpoint for the state after `at_step` steps."""
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": at_step,
                 "tokens_seen": tokens_seen, "wall_seconds": wall, "rng": rng_state()}
        c = tc["checkpoint"]
        return ckpt.save_checkpoint(ckpt_root, at_step, state, spec.specials_path, info,
                                    int(c["keep_last"]), int(c["keep_every"]))

    stop = StopFlag()
    reason, losses, saved = "done", [], None
    log_losses, log_t0, log_tokens = [], time.perf_counter(), 0
    try:
        while step < stop_at:
            lr = lr_at(step, float(o["peak_lr"]), float(o["min_lr"]), int(o["warmup_steps"]), steps)
            for group in opt.param_groups:
                group["lr"] = lr
            batch = torch.from_numpy(loader.batch(step)).to(device)
            step_loss = torch.zeros((), device=device)
            for chunk in batch.split(micro):
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=use_bf16):
                    logits = fwd(chunk[:, :-1])
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                       chunk[:, 1:].reshape(-1)) / (batch_seqs // micro)
                loss.backward()
                step_loss += loss.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(o["grad_clip"]))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            tokens_seen += tokens_per_step
            loss_value = float(step_loss)
            losses.append(loss_value)
            log_losses.append(loss_value)
            log_tokens += tokens_per_step
            wall = base_wall + time.perf_counter() - session_t0   # all resumes included
            if stop.reason:
                reason = stop.reason
            elif guard.over_budget(wall):
                reason = "max-cost"

            if step % log_every == 0 or step == stop_at or reason != "done":
                dt = max(time.perf_counter() - log_t0, 1e-9)
                tps = log_tokens / dt
                hours_left, dollars_left = guard.projection((steps - step) * tokens_per_step, tps)
                row = {"step": step, "loss": sum(log_losses) / len(log_losses), "lr": lr,
                       "tokens_seen": tokens_seen, "tokens_per_sec": tps,
                       "mfu": mfu(tps, fpt, float(tc["cost"]["peak_flops"])),
                       "hours_left": hours_left, "dollars_left": dollars_left,
                       "dollars_spent": guard.spent(wall)}
                with open(run_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row))
                log_losses, log_t0, log_tokens = [], time.perf_counter(), 0

            if reason != "done":
                saved = save(step)
                print(f"stopping at step {step}: {reason}; checkpoint {saved}")
                break
            if step % int(tc["checkpoint"]["every"]) == 0:
                saved = save(step)
        else:
            saved = save(step)
            if step < steps:
                reason = "stop-step"
    finally:
        stop.restore()
    return {"step": step, "steps": steps, "reason": reason, "checkpoint": saved,
            "tokens_seen": tokens_seen, "losses": losses, "n_params": n_params,
            "dollars_spent": guard.spent(wall)}
