"""Grouped-query attention: 16 query heads share 4 key/value heads, causal, RoPE, no biases."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from alethic.model.layers import apply_rope


class GQAttention(nn.Module):
    """Causal grouped-query attention ending in o_proj. Returns A = o_proj(GQA(h))."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
        """Create q, k, v and o projections (no biases)."""
        super().__init__()
        if n_heads % n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """h: normalized input (B, T, d_model). Returns the attention output after o_proj."""
        b, t, _ = h.shape
        q = self.q_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.o_proj(y.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim))
