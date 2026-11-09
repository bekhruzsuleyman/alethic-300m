"""AdamW with weight decay on matrices only, and the warmup + cosine learning-rate schedule."""
from __future__ import annotations

import math

import torch
from torch import nn


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Two groups: decayed weight matrices, and everything else (norms, biases, alpha, embedding)."""
    embedding = model.tok_emb.weight
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and p is not embedding:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, optim_cfg: dict, device: str) -> torch.optim.AdamW:
    """AdamW with the configured betas, eps and weight decay (fused kernels on CUDA)."""
    return torch.optim.AdamW(
        param_groups(model, float(optim_cfg["weight_decay"])),
        lr=float(optim_cfg["peak_lr"]),
        betas=tuple(float(b) for b in optim_cfg["betas"]),
        eps=float(optim_cfg["eps"]),
        fused=device.startswith("cuda"),
    )


def lr_at(step: int, peak: float, min_lr: float, warmup: int, total: int) -> float:
    """Learning rate for a 0-based step: linear warmup to peak, then cosine decay to min_lr."""
    if step < warmup:
        return peak * (step + 1) / warmup
    span = max(1, total - warmup)
    progress = min(1.0, (step - warmup) / span)
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + math.cos(math.pi * progress))
