import sys

import torch
import yaml

from alethic.agent.modelio import load_model, read_config, read_flat_yaml
from alethic.model.transformer import ModelConfig, Transformer


def test_flat_reader_matches_pyyaml_on_the_real_config():
    text = open("configs/model_gqa.yaml", encoding="utf-8").read()
    assert read_flat_yaml(text) == yaml.safe_load(text)


def test_flat_reader_types_comments_and_quotes():
    got = read_flat_yaml("# top\nname: gqa   # trailing\nn: 24\nlr: 1.0e-6\nx: 10000.0\nflag: true\nq: \"a b\"\n\nempty:\n")
    assert got == {"name": "gqa", "n": 24, "lr": 1e-6, "x": 10000.0, "flag": True, "q": "a b", "empty": None}


def test_read_config_works_when_pyyaml_is_missing(monkeypatch, tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: t\nn: 3\n")
    monkeypatch.setitem(sys.modules, "yaml", None)  # makes `import yaml` fail
    assert read_config(str(p)) == {"name": "t", "n": 3}


def test_load_model_gives_the_same_logits_as_the_saved_model(tmp_path):
    cfg = dict(name="tiny", attention="gqa", n_layers=2, d_model=32, n_heads=4, n_kv_heads=2, head_dim=8,
               ffn_hidden=64, vocab_size=100, max_seq_len=64)
    torch.manual_seed(0)
    m = Transformer(ModelConfig.from_dict(cfg)).eval()
    torch.save({"model": m.state_dict()}, tmp_path / "w.pt")
    (tmp_path / "c.yaml").write_text("\n".join("%s: %s" % kv for kv in cfg.items()) + "\n")
    loaded = load_model(str(tmp_path / "w.pt"), str(tmp_path / "c.yaml"))
    ids = torch.tensor([[1, 5, 9, 33]])
    with torch.no_grad():
        assert torch.allclose(loaded(ids), m(ids), atol=1e-6)
    assert not loaded.training
