"""Inference with a key/value cache for the GQA model, so each new token costs one step instead of a full pass.

It reads the weights of a loaded Transformer and repeats its forward math; tests check it equals the full pass.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from alethic.model.layers import apply_rope


class KVDecoder:
    def __init__(self, model):
        if model.cfg.attention != "gqa":
            raise NotImplementedError("the cached decoder supports the gqa model only")
        self.model = model
        self.ids: list = []
        self.k: list = [None] * len(model.blocks)
        self.v: list = [None] * len(model.blocks)

    def _truncate(self, keep: int) -> None:
        self.ids = self.ids[:keep]
        for i in range(len(self.k)):
            if self.k[i] is not None:
                self.k[i], self.v[i] = self.k[i][:, :, :keep], self.v[i][:, :, :keep]

    @torch.inference_mode()
    def logits(self, ids: list) -> torch.Tensor:
        """Logits (vocab,) for the token after `ids`. Reuses the cache for the part shared with the last call."""
        m, common = self.model, 0
        while common < min(len(ids), len(self.ids)) and ids[common] == self.ids[common]:
            common += 1
        common = min(common, len(ids) - 1)  # always run at least the last token
        self._truncate(common)
        x = m.tok_emb(torch.tensor([ids[common:]], device=m.tok_emb.weight.device))
        t = x.shape[1]
        cos, sin = m.rope_cos[common:common + t], m.rope_sin[common:common + t]
        for i, blk in enumerate(m.blocks):
            a, h = blk.attn, blk.norm1(x)
            q = a.q_proj(h).view(1, t, a.n_heads, a.head_dim).transpose(1, 2)
            k = a.k_proj(h).view(1, t, a.n_kv_heads, a.head_dim).transpose(1, 2)
            v = a.v_proj(h).view(1, t, a.n_kv_heads, a.head_dim).transpose(1, 2)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            if self.k[i] is not None:
                k, v = torch.cat([self.k[i], k], dim=2), torch.cat([self.v[i], v], dim=2)
            self.k[i], self.v[i] = k, v
            if t == 1:
                y = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
            elif common == 0:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            else:
                rows = common + torch.arange(t, device=q.device)[:, None]
                mask = torch.arange(common + t, device=q.device)[None, :] <= rows
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
            x = x + a.o_proj(y.transpose(1, 2).reshape(1, t, a.n_heads * a.head_dim))
            x = x + blk.ffn(blk.norm2(x))
        self.ids = list(ids)
        return (m.norm(x[:, -1:]) @ m.tok_emb.weight.t())[0, 0]
