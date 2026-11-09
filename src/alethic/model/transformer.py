"""The decoder-only transformer used by both models (they differ only in attention and FFN width)."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
from torch import nn

from alethic.model.attention_gqa import GQAttention
from alethic.model.attention_sgca import SGCAttention
from alethic.model.layers import RMSNorm, SwiGLU, rope_tables


@dataclass(frozen=True)
class ModelConfig:
    """Model shape, read from configs/model_*.yaml."""
    name: str
    attention: str            # "gqa" or "sgca"
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_hidden: int
    vocab_size: int
    max_seq_len: int
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    init_std: float = 0.02
    alpha_init: float = 1.0
    expected_params: int | None = None

    @classmethod
    def from_dict(cls, cfg: dict) -> "ModelConfig":
        """Build from a YAML dict; unknown keys are an error so typos are caught."""
        known = {f.name for f in fields(cls)}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        if cfg.get("attention") not in ("gqa", "sgca"):
            raise ValueError(f"attention must be 'gqa' or 'sgca', got {cfg.get('attention')!r}")
        return cls(**cfg)


class Block(nn.Module):
    """Pre-norm block: X = X + mixer(norm1(X)); X = X + ffn(norm2(X)).

    For GQA the mixer returns A; for SGCA it returns the control term C.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        """Create the two norms, the attention mixer and the SwiGLU FFN."""
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        if cfg.attention == "sgca":
            self.attn = SGCAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.alpha_init)
        else:
            self.attn = GQAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply the attention sub-layer, then the FFN sub-layer, each with a residual."""
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.ffn(self.norm2(x))


class Transformer(nn.Module):
    """Token embedding, n_layers blocks, final RMSNorm, output tied to the embedding."""

    def __init__(self, cfg: ModelConfig) -> None:
        """Create every layer and initialise the weights."""
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        cos, sin = rope_tables(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        """Normal(0, 0.02) for matrices and embeddings; output projections of attention and FFN
        use 0.02 / sqrt(2 * n_layers); norms and alpha start at their set values, biases at 0."""
        std = self.cfg.init_std
        out_std = std / math.sqrt(2 * self.cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name == "norm.weight":
                p.fill_(1.0)
            elif name.endswith(".alpha"):
                p.fill_(self.cfg.alpha_init)
            elif name.endswith(".bias"):
                p.zero_()
            elif name.endswith("attn.o_proj.weight") or name.endswith("ffn.down.weight"):
                p.normal_(0.0, out_std)
            else:
                p.normal_(0.0, std)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Token ids (B, T) -> logits (B, T, vocab_size)."""
        if idx.shape[1] > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {idx.shape[1]} > max_seq_len {self.cfg.max_seq_len}")
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        return self.norm(x) @ self.tok_emb.weight.t()
