import json
import math

import pytest
import sentencepiece as spm
import torch
import torch.nn.functional as F

from alethic.config import sha256_file
from alethic.model.transformer import ModelConfig, Transformer
from alethic.sft import train as sft

TINY = dict(name="tiny", attention="gqa", n_layers=2, d_model=32, n_heads=4, n_kv_heads=2, head_dim=8,
            ffn_hidden=64, vocab_size=300, max_seq_len=128)
SEGS = [{"t": "<user>\nhello world\n</user>\n", "loss": 0}, {"t": "<assistant>\n", "loss": 0},
        {"t": "yes the quick brown fox\n", "loss": 1}, {"t": "</assistant>\n", "loss": 0}]


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    (d / "c.txt").write_text("\n".join(
        "hello world %d the quick brown fox <user> </user> <assistant> </assistant> yes no" % i for i in range(300)))
    spm.SentencePieceTrainer.train(input=str(d / "c.txt"), model_prefix=str(d / "tiny"), vocab_size=300,
                                   model_type="unigram", byte_fallback=True, bos_id=1, eos_id=2, unk_id=0,
                                   pad_id=3, normalization_rule_name="identity", add_dummy_prefix=False)
    return str(d / "tiny.model")


def write_data(path, n=12):
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"id": "S%d.0" % i, "segments": SEGS}) + "\n")


def test_sft_loss_counts_only_labelled_positions():
    logits = torch.randn(1, 5, 10)
    labels = torch.tensor([[-100, -100, 4, -100, 7]])
    total, n = sft.loss_sum(logits, labels)
    # shift: logits[:, :-1] predicts labels[:, 1:] = [-100, 4, -100, 7] -> two labelled targets
    expected = F.cross_entropy(logits[0, :-1][[1, 3]], torch.tensor([4, 7]), reduction="sum")
    assert n == 2 and torch.allclose(total, expected)


def test_lr_schedule_warms_up_then_decays():
    f = sft.make_lr(peak=1e-4, min_ratio=0.1, warmup=4, total=20)
    assert f(0) < f(3) <= 1e-4 + 1e-12
    assert f(19) == pytest.approx(1e-5, rel=0.2)
    assert f(10) < f(4)


def test_load_base_checks_sha_and_loads_weights(tmp_path):
    a = Transformer(ModelConfig.from_dict(TINY))
    folder = tmp_path / "base"
    folder.mkdir()
    torch.save({"model": a.state_dict(), "optimizer": {}, "step": 7}, folder / "state.pt")
    (folder / "manifest.json").write_text(json.dumps({"state_sha256": sha256_file(folder / "state.pt"), "step": 7}))
    b = Transformer(ModelConfig.from_dict(TINY))
    info = sft.load_base(b, folder)
    assert info["step"] == 7
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    (folder / "state.pt").write_bytes(b"corrupt")
    with pytest.raises(Exception):
        sft.load_base(b, folder)


def test_run_sft_overfits_tiny_data_saves_best_and_samples(tmp_path, tok):
    data = tmp_path / "d.jsonl"
    write_data(data)
    out = tmp_path / "out"
    cfg = dict(model=TINY, tokenizer=str(tok), data=[str(data)], out=str(out), init_from=None, epochs=60,
               batch_size=4, peak_lr=3e-3, min_lr_ratio=0.1, warmup_steps=2, weight_decay=0.0, val_fraction=0.25,
               seed=1, device="cpu", bf16=False, patience=50, max_len=128, max_micro_tokens=4096,
               sample_prompts=["hello world"], max_new_tokens=12)
    res = sft.run_sft(cfg)
    rows = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert rows[-1]["train_loss"] < rows[0]["train_loss"] * 0.5
    assert math.isfinite(res["best_val_loss"])
    assert (out / "best" / "model.pt").is_file() and (out / "report.json").is_file()
    assert len(res["samples"]) == 1 and isinstance(res["samples"][0]["output"], str)


def test_early_stopping_stops_before_all_epochs(tmp_path, tok):
    data = tmp_path / "d.jsonl"
    write_data(data, 12)
    cfg = dict(model=TINY, tokenizer=str(tok), data=[str(data)], out=str(tmp_path / "o"), init_from=None,
               epochs=40, batch_size=4, peak_lr=5e-2, min_lr_ratio=1.0, warmup_steps=1, weight_decay=0.0,
               val_fraction=0.25, seed=1, device="cpu", bf16=False, patience=1, max_len=128,
               max_micro_tokens=4096, sample_prompts=[], max_new_tokens=4)
    res = sft.run_sft(cfg)
    assert res["epochs_run"] < 40
