"""Sin-Gated Control Attention (SGCA), as defined at bekhruz.org/research/sgca.

    H  = RMSNorm(X)
    A  = o_proj(GQA(H))
    C  = (W H + b) * (alpha * sin(A))
    X' = X + C

GQA stays the retrieval step; the residual update is a sine-modulated control term.
gate_proj holds W and b (d_model -> d_model, with bias) and reads the same H as attention.
alpha is a learnable vector of length d_model (one scale per channel), starting at 1.0.
"""
from __future__ import annotations

import torch
from torch import nn

from alethic.model.attention_gqa import GQAttention


class SGCAttention(GQAttention):
    """GQA plus gate_proj and alpha. Returns the control term C (the block adds it to X)."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 alpha_init: float = 1.0) -> None:
        """Create the GQA projections, gate_proj (with bias) and alpha."""
        super().__init__(d_model, n_heads, n_kv_heads, head_dim)
        self.alpha_init = alpha_init
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        self.alpha = nn.Parameter(torch.full((d_model,), float(alpha_init)))

    def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """C = gate_proj(H) * (alpha * sin(A)) with A = o_proj(GQA(H))."""
        a = super().forward(h, cos, sin)
        return self.gate_proj(h) * (self.alpha * torch.sin(a))
