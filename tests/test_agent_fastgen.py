import pytest
import torch

from alethic.agent.fastgen import KVDecoder
from alethic.model.transformer import ModelConfig, Transformer

TINY = dict(name="tiny", attention="gqa", n_layers=3, d_model=32, n_heads=4, n_kv_heads=2, head_dim=8,
            ffn_hidden=64, vocab_size=100, max_seq_len=64)


def model():
    torch.manual_seed(0)
    return Transformer(ModelConfig.from_dict(TINY)).eval()


def full(m, ids):
    with torch.no_grad():
        return m(torch.tensor([ids]))[0, -1]


def test_cached_logits_match_the_full_forward_pass():
    m = model()
    dec = KVDecoder(m)
    ids = [5, 9, 3, 77, 12, 8, 1, 40]
    assert torch.allclose(dec.logits(ids[:5]), full(m, ids[:5]), atol=1e-4)
    for n in range(6, 9):  # one token at a time, reusing the cache
        assert torch.allclose(dec.logits(ids[:n]), full(m, ids[:n]), atol=1e-4)


def test_changed_history_reuses_only_the_shared_prefix():
    m = model()
    dec = KVDecoder(m)
    dec.logits([5, 9, 3, 77, 12, 8])
    other = [5, 9, 3, 50, 60, 70, 80]
    assert torch.allclose(dec.logits(other), full(m, other), atol=1e-4)


def test_same_ids_twice_still_gives_logits():
    m = model()
    dec = KVDecoder(m)
    ids = [5, 9, 3]
    a = dec.logits(ids)
    assert torch.allclose(dec.logits(ids), a, atol=1e-4)


def test_only_gqa_is_supported():
    cfg = dict(TINY, attention="sgca")
    with pytest.raises(NotImplementedError):
        KVDecoder(Transformer(ModelConfig.from_dict(cfg)).eval())
