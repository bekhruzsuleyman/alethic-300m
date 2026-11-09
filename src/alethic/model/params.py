"""Parameter counts (measured and by formula) and model FLOPs per token."""
from __future__ import annotations

from torch import nn

from alethic.model.transformer import ModelConfig


def count_parameters(model: nn.Module) -> int:
    """Number of distinct parameters (the tied embedding is counted once)."""
    return sum(p.numel() for p in model.parameters())


def parameter_breakdown(cfg: ModelConfig) -> dict[str, int]:
    """Parameter count by part, from the formula in the plan."""
    d, hd = cfg.d_model, cfg.head_dim
    attn = d * cfg.n_heads * hd + 2 * d * cfg.n_kv_heads * hd + cfg.n_heads * hd * d
    sgca = (d * d + d + d) if cfg.attention == "sgca" else 0    # gate W, gate b, alpha
    ffn = 3 * d * cfg.ffn_hidden
    norms = 2 * d
    per_layer = attn + sgca + ffn + norms
    return {
        "attention_per_layer": attn,
        "sgca_extra_per_layer": sgca,
        "ffn_per_layer": ffn,
        "norms_per_layer": norms,
        "per_layer": per_layer,
        "layers": cfg.n_layers * per_layer,
        "embedding": cfg.vocab_size * d,
        "final_norm": d,
        "total": cfg.n_layers * per_layer + cfg.vocab_size * d + d,
    }


def expected_parameters(cfg: ModelConfig) -> int:
    """Total parameters by formula."""
    return parameter_breakdown(cfg)["total"]


def flops_per_token(n_params: int, n_layers: int, d_model: int, seq_len: int) -> int:
    """Training FLOPs per token: 6N + 12 * L * d * s (plan, Cost guard)."""
    return 6 * n_params + 12 * n_layers * d_model * seq_len
