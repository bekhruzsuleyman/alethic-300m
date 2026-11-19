"""Supervised fine-tuning (SFT) of the pretrained Alethic model on prompt -> answer samples.

Run:  python -m alethic.sft.train --init-from <checkpoint folder> --data a.jsonl b.jsonl --out runs/sft1
Idea: continue training the pretrained weights, but only learn from tokens the samples mark with loss=1
(the assistant's own text). Small learning rate, few epochs, keep the epoch with the best validation loss.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

from alethic.config import load_config, write_json_atomic
from alethic.model.transformer import ModelConfig, Transformer
from alethic.sft import data as sdata
from alethic.train import checkpoint as ckpt
from alethic.train.optim import build_optimizer


def loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Sum of cross-entropy over labelled positions, and how many there are.

    Position t predicts token t+1, so logits drop their last step and labels drop their first.
    """
    lg, tg = logits[:, :-1], labels[:, 1:]
    n = int((tg != -100).sum())
    total = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), tg.reshape(-1),
                            ignore_index=-100, reduction="sum")
    return total, n


def make_lr(peak: float, min_ratio: float, warmup: int, total: int):
    """Linear warmup to peak, then cosine down to peak * min_ratio."""
    floor = peak * min_ratio
    def lr(step: int) -> float:
        if step < warmup:
            return peak * (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))
    return lr


def load_base(model: torch.nn.Module, folder) -> dict:
    """Load only the weights of a pretraining checkpoint (not the Adam state: SFT starts a fresh optimizer)."""
    earlier = Path(folder) / "best" / "model.pt"            # the output of an earlier SFT run (stage A -> stage B)
    if earlier.is_file():
        state = torch.load(earlier, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        return {"step": None, "n_tensors": len(state["model"]), "from_sft": str(folder)}
    state, manifest = ckpt.load_checkpoint(Path(folder))     # checks the SHA-256 first
    model.load_state_dict(state["model"])
    return {"step": manifest.get("step"), "n_tensors": len(state["model"])}


def _micro_chunks(x: torch.Tensor, y: torch.Tensor, max_tokens: int):
    """Split a batch by rows so no piece has more than max_tokens tokens (keeps GPU memory bounded)."""
    rows = max(1, max_tokens // x.shape[1])
    for i in range(0, x.shape[0], rows):
        yield x[i:i + rows], y[i:i + rows]


@torch.no_grad()
def eval_loss(model, enc, batch_size, device, bf16, max_micro_tokens) -> float | None:
    """Mean loss per labelled token over a list of encoded samples (None if there are none)."""
    if not enc:
        return None
    model.eval()
    total, count = 0.0, 0
    for x, y in sdata.batches(enc, batch_size, sdata.PAD_ID, 0, 0):
        for cx, cy in _micro_chunks(x, y, max_micro_tokens):
            cx, cy = cx.to(device), cy.to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=bf16):
                lg = model(cx)
            s, n = loss_sum(lg, cy)
            total += float(s)
            count += n
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def greedy(model, sp, prompt: str, device: str, max_new: int, max_len: int) -> str:
    """Greedy answer for one prompt (no KV cache: simple and slow, fine for a handful of checks)."""
    model.eval()
    ids = sp.encode(f"<user>\n{prompt}\n</user>\n<assistant>\n")
    start = len(ids)
    for _ in range(max_new):
        if len(ids) >= max_len:
            break
        logits = model(torch.tensor([ids], device=device))[0, -1]
        nxt = int(logits.argmax())
        if nxt == sdata.EOS_ID:
            break
        ids.append(nxt)
        if sp.decode(ids[start:]).rstrip().endswith(sdata.CLOSE_TAG):
            break
    model.train()
    return sp.decode(ids[start:])


def run_sft(cfg: dict) -> dict:
    """Train, validate every epoch, keep the best epoch, write logs and a report; returns a summary."""
    device = cfg["device"]
    torch.manual_seed(cfg["seed"])
    out = Path(cfg["out"])
    (out / "best").mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=str(cfg["tokenizer"]))

    samples = sdata.load_samples([Path(p) for p in cfg["data"]])
    train_s, val_s = sdata.split_train_val(samples, cfg["val_fraction"], cfg["seed"])
    train_e, train_rep = sdata.encode_all(sp, train_s, cfg["max_len"])
    val_e, val_rep = sdata.encode_all(sp, val_s, cfg["max_len"])
    if not train_e:
        raise ValueError("no training samples left after filtering")

    mcfg = ModelConfig.from_dict(cfg["model"])
    model = Transformer(mcfg).to(device)
    base = load_base(model, cfg["init_from"]) if cfg.get("init_from") else {"step": None, "note": "random init"}
    opt = build_optimizer(model, {"weight_decay": cfg["weight_decay"], "peak_lr": cfg["peak_lr"],
                                  "betas": [0.9, 0.95], "eps": 1e-8}, device)
    steps_per_epoch = math.ceil(len(train_e) / cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["epochs"]
    lr_of = make_lr(cfg["peak_lr"], cfg["min_lr_ratio"], cfg["warmup_steps"], total_steps)
    bf16 = bool(cfg["bf16"])
    print(json.dumps({"train": len(train_e), "val": len(val_e), "steps_per_epoch": steps_per_epoch,
                      "trained_tokens": train_rep["trained_tokens"], "base": base}), flush=True)

    best, best_epoch, bad, step, t0 = float("inf"), -1, 0, 0, time.time()
    epochs_run = 0
    for epoch in range(cfg["epochs"]):
        epochs_run = epoch + 1
        run_loss, run_n = 0.0, 0
        for x, y in sdata.batches(train_e, cfg["batch_size"], sdata.PAD_ID, cfg["seed"], epoch):
            for g in opt.param_groups:
                g["lr"] = lr_of(step)
            n_total = int((y[:, 1:] != -100).sum())
            for cx, cy in _micro_chunks(x, y, cfg["max_micro_tokens"]):
                cx, cy = cx.to(device), cy.to(device)
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=bf16):
                    lg = model(cx)
                s, _ = loss_sum(lg, cy)
                (s / max(n_total, 1)).backward()          # token mean over the whole batch
                run_loss += float(s.detach())
            run_n += n_total
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        val = eval_loss(model, val_e, cfg["batch_size"], device, bf16, cfg["max_micro_tokens"])
        row = {"epoch": epoch + 1, "step": step, "train_loss": run_loss / max(run_n, 1), "val_loss": val,
               "lr": lr_of(step - 1), "seconds": round(time.time() - t0, 1)}
        with open(out / "train_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        score = val if val is not None else row["train_loss"]
        if score < best:
            best, best_epoch, bad = score, epoch + 1, 0
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_loss": val}, out / "best" / "model.pt")
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"early stop: no improvement for {bad} epochs", flush=True)
                break

    state = torch.load(out / "best" / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    samples_out = [{"prompt": p, "output": greedy(model, sp, p, device, cfg["max_new_tokens"], mcfg.max_seq_len)}
                   for p in cfg.get("sample_prompts") or []]
    report = {"best_val_loss": best, "best_epoch": best_epoch, "epochs_run": epochs_run, "steps": step,
              "base": base, "train": train_rep, "val": val_rep, "samples": samples_out,
              "config": {k: v for k, v in cfg.items() if k != "model"}}
    write_json_atomic(out / "report.json", report)
    return report


def parse_args(argv=None) -> dict:
    """Command-line settings -> the cfg dict used by run_sft."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-from", help="pretraining checkpoint folder (state.pt + manifest.json)")
    p.add_argument("--model", default="configs/model_gqa.yaml")
    p.add_argument("--tokenizer", required=True, help="the frozen SentencePiece .model file")
    p.add_argument("--data", nargs="+", required=True, help="JSONL sample files")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--peak-lr", type=float, default=3e-5)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--max-micro-tokens", type=int, default=16384)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--sample-prompt", action="append", default=[])
    a = p.parse_args(argv)
    return dict(model=load_config(a.model), init_from=a.init_from, tokenizer=a.tokenizer, data=a.data, out=a.out,
                epochs=a.epochs, batch_size=a.batch_size, peak_lr=a.peak_lr, min_lr_ratio=a.min_lr_ratio,
                warmup_steps=a.warmup_steps, weight_decay=a.weight_decay, val_fraction=a.val_fraction,
                patience=a.patience, seed=a.seed, max_len=a.max_len, max_micro_tokens=a.max_micro_tokens,
                max_new_tokens=a.max_new_tokens, device=a.device,
                bf16=a.device.startswith("cuda") and not a.no_bf16, sample_prompts=a.sample_prompt)


if __name__ == "__main__":
    report = run_sft(parse_args())
    print(json.dumps({k: report[k] for k in ("best_val_loss", "best_epoch", "epochs_run")}))
