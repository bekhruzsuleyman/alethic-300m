"""Model tests: exact parameter counts, causality, and the SGCA gate against a by-hand formula."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from alethic.config import load_config
from alethic.model.params import count_parameters, expected_parameters, flops_per_token, parameter_breakdown
from alethic.model.transformer import ModelConfig, Transformer
from conftest import CONFIGS, tiny_model_cfg

TARGET = 302_908_416


@pytest.mark.parametrize("name", ["model_gqa.yaml", "model_sgca.yaml"])
def test_param_counts(name):
    """SGCA and GQA both have exactly 302,908,416 parameters (built on the meta device: no memory)."""
    cfg = ModelConfig.from_dict(load_config(CONFIGS / name))
    with torch.device("meta"):
        model = Transformer(cfg)
    assert count_parameters(model) == TARGET == expected_parameters(cfg) == cfg.expected_params
    b = parameter_breakdown(cfg)
    assert b["attention_per_layer"] == 2_621_440
    assert b["embedding"] == 51_200_000 and b["final_norm"] == 1_024
    if cfg.attention == "gqa":
        assert b["ffn_per_layer"] == 7_864_320 and b["per_layer"] == 10_487_808
    else:
        assert b["sgca_extra_per_layer"] == 1_050_624
        assert b["ffn_per_layer"] == 7_864_320 - 1_050_624
        sgca_layer = model.blocks[0].attn
        assert sgca_layer.gate_proj.bias is not None and sgca_layer.alpha.shape == (1024,)
    assert model.tok_emb.weight is not None and len(list(model.parameters())) == len(dict(model.named_parameters()))
    assert flops_per_token(TARGET, 24, 1024, 4096) == 3_025_410_048   # ~3.03e9 per token


@pytest.mark.parametrize("attention", ["gqa", "sgca"])
def test_causality(attention):
    """Changing token t never changes the logits before t."""
    torch.manual_seed(0)
    model = Transformer(ModelConfig.from_dict(tiny_model_cfg(attention))).eval()
    idx = torch.randint(0, 128, (2, 24))
    with torch.no_grad():
        base = model(idx)
        for t in (0, 7, 23):
            changed = idx.clone()
            changed[:, t] = (changed[:, t] + 1) % 128
            out = model(changed)
            assert torch.equal(out[:, :t], base[:, :t]), f"position < {t} changed"
            assert not torch.allclose(out[:, t], base[:, t]), "the changed position should change"


def _rms(x, w, eps):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


def _rope(x, theta):
    """Rotary embedding written out: pairs (i, i + D/2) rotated by position * theta^(-2i/D)."""
    _, _, t, d = x.shape
    i = torch.arange(d // 2, dtype=torch.float32)
    ang = torch.arange(t, dtype=torch.float32)[:, None] * theta ** (-2 * i / d)[None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([x1 * ang.cos() - x2 * ang.sin(), x2 * ang.cos() + x1 * ang.sin()], dim=-1)


def test_sgca_gate():
    """Shape is right, alpha starts at 1.0, gate_proj has a bias, and the block equals X + C (+ FFN)."""
    cfg = ModelConfig.from_dict(tiny_model_cfg("sgca"))
    torch.manual_seed(0)
    model = Transformer(cfg)
    block = model.blocks[0]
    attn = block.attn
    assert attn.alpha.shape == (cfg.d_model,) and torch.equal(attn.alpha, torch.ones(cfg.d_model))
    assert attn.gate_proj.bias is not None and torch.equal(attn.gate_proj.bias, torch.zeros(cfg.d_model))
    assert attn.gate_proj.weight.shape == (cfg.d_model, cfg.d_model)
    assert 0.01 < attn.gate_proj.weight.std().item() < 0.03     # std 0.02, not the scaled one

    with torch.no_grad():   # make alpha and b non-trivial so the check means something
        attn.alpha.copy_(torch.linspace(-2.0, 2.0, cfg.d_model))
        attn.gate_proj.bias.normal_(0.0, 0.5)
        attn.o_proj.weight.normal_(0.0, 0.5)
    x = torch.randn(2, 10, cfg.d_model)
    with torch.no_grad():
        out = block(x, model.rope_cos, model.rope_sin)

        b, t, d = x.shape
        nh, nkv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        h = _rms(x, block.norm1.weight, cfg.norm_eps)
        q = (h @ attn.q_proj.weight.T).view(b, t, nh, hd).transpose(1, 2)
        k = (h @ attn.k_proj.weight.T).view(b, t, nkv, hd).transpose(1, 2)
        v = (h @ attn.v_proj.weight.T).view(b, t, nkv, hd).transpose(1, 2)
        q, k = _rope(q, cfg.rope_theta), _rope(k, cfg.rope_theta)
        k, v = k.repeat_interleave(nh // nkv, dim=1), v.repeat_interleave(nh // nkv, dim=1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(hd)
        scores = scores.masked_fill(torch.triu(torch.ones(t, t, dtype=torch.bool), 1), float("-inf"))
        y = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, t, nh * hd)
        a = y @ attn.o_proj.weight.T
        c = (h @ attn.gate_proj.weight.T + attn.gate_proj.bias) * (attn.alpha * torch.sin(a))
        x1 = x + c
        h2 = _rms(x1, block.norm2.weight, cfg.norm_eps)
        ffn = (F.silu(h2 @ block.ffn.gate.weight.T) * (h2 @ block.ffn.up.weight.T)) @ block.ffn.down.weight.T
        expected = x1 + ffn

    assert out.shape == x.shape
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5), (out - expected).abs().max()
    assert model(torch.randint(0, 128, (2, 10))).shape == (2, 10, cfg.vocab_size)


def test_init_scales():
    """Output projections start with std 0.02 / sqrt(2 * n_layers); norms at 1."""
    cfg = ModelConfig.from_dict(dict(tiny_model_cfg("gqa"), d_model=256, head_dim=64, ffn_hidden=512))
    torch.manual_seed(0)
    model = Transformer(cfg)
    out_std = 0.02 / math.sqrt(2 * cfg.n_layers)
    assert abs(model.blocks[0].attn.o_proj.weight.std().item() - out_std) < 0.1 * out_std
    assert abs(model.blocks[0].ffn.down.weight.std().item() - out_std) < 0.1 * out_std
    assert abs(model.blocks[0].attn.q_proj.weight.std().item() - 0.02) < 0.002
    assert torch.equal(model.norm.weight, torch.ones(cfg.d_model))
