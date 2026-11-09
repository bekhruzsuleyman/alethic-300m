"""Building blocks shared by both models: RMSNorm, rotary position embedding, SwiGLU."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square norm with a learned scale and no bias, computed in float32."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """dim: size of the last axis; eps: added inside the square root."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x / rms(x) * weight, returned in x's dtype."""
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).type_as(x)


def rope_tables(seq_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin tables of shape (seq_len, head_dim) for rotary embedding."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)  # (T, D/2)
    angles = torch.cat([angles, angles], dim=-1)                               # (T, D)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x of shape (B, H, T, D) by position: x*cos + rotate_half(x)*sin."""
    t = x.shape[-2]
    cos, sin = cos[:t].to(x.dtype), sin[:t].to(x.dtype)
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


class SwiGLU(nn.Module):
    """Feed-forward layer: down( silu(gate(x)) * up(x) ), three matrices, no biases."""

    def __init__(self, d_model: int, hidden: int) -> None:
        """d_model -> hidden -> d_model."""
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated feed-forward."""
        return self.down(F.silu(self.gate(x)) * self.up(x))
