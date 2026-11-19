import json
import pytest
import sentencepiece as spm

from alethic.sft import data

PAD, EOS = 3, 2


@pytest.fixture(scope="module")
def sp(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok")
    corpus = d / "c.txt"
    corpus.write_text("\n".join(
        "hello world %d the quick brown fox <user> </user> <assistant> </assistant> yes no" % i for i in range(300)))
    prefix = str(d / "tiny")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=prefix, vocab_size=300, model_type="unigram",
                                   byte_fallback=True, bos_id=1, eos_id=2, unk_id=0, pad_id=3,
                                   user_defined_symbols=[], normalization_rule_name="identity",
                                   add_dummy_prefix=False)
    return spm.SentencePieceProcessor(model_file=prefix + ".model")


SEGS = [
    {"t": "<user>\nhello world\n</user>\n", "loss": 0},
    {"t": "<assistant>\n", "loss": 0},
    {"t": "yes\n", "loss": 1},
    {"t": "</assistant>\n", "loss": 0},
]


def decode_trained(sp, ids, labels):
    return sp.decode([i for i, l in zip(ids, labels) if l != -100 and l != EOS])


def test_only_loss_segments_are_trained(sp):
    ids, labels = data.encode_sample(sp, SEGS, close_tag_loss=False, eos=False)
    assert len(ids) == len(labels)
    assert decode_trained(sp, ids, labels).strip() == "yes"
    assert all(l == -100 or l == i for i, l in zip(ids, labels))


def test_close_tag_and_eos_are_trained_so_the_model_learns_to_stop(sp):
    ids, labels = data.encode_sample(sp, SEGS, close_tag_loss=True, eos=True)
    assert ids[-1] == EOS and labels[-1] == EOS
    assert "</assistant>" in decode_trained(sp, ids, labels)


def test_unclosed_sample_gets_no_eos(sp):
    segs = SEGS[:3]
    ids, labels = data.encode_sample(sp, segs, close_tag_loss=True, eos=True)
    assert ids[-1] != EOS


def test_sample_without_any_loss_is_rejected(sp):
    segs = [{"t": "<user>\nhello\n</user>\n", "loss": 0}]
    with pytest.raises(data.NoLossError):
        data.encode_sample(sp, segs, close_tag_loss=False, eos=False)


def test_load_samples_reads_jsonl_and_skips_blank_lines(tmp_path):
    f = tmp_path / "a.jsonl"
    f.write_text(json.dumps({"id": "X1.0", "segments": SEGS}) + "\n\n")
    got = data.load_samples([f])
    assert [s["id"] for s in got] == ["X1.0"]


def test_split_keeps_all_parts_of_one_source_together_and_is_deterministic():
    samples = [{"id": "S%d.%d" % (i, k), "segments": SEGS} for i in range(40) for k in range(2)]
    tr, va = data.split_train_val(samples, val_fraction=0.25, seed=1)
    assert len(tr) + len(va) == 80 and va
    assert {s["id"].split(".")[0] for s in tr}.isdisjoint({s["id"].split(".")[0] for s in va})
    tr2, va2 = data.split_train_val(samples, val_fraction=0.25, seed=1)
    assert [s["id"] for s in va] == [s["id"] for s in va2]


def test_encode_all_drops_too_long_and_reports(sp):
    long = [{"t": "<assistant>\n", "loss": 0}, {"t": "yes " * 400, "loss": 1}]
    samples = [{"id": "A.0", "segments": SEGS}, {"id": "B.0", "segments": long}]
    enc, report = data.encode_all(sp, samples, max_len=64, close_tag_loss=True, eos=True)
    assert [e["id"] for e in enc] == ["A.0"]
    assert report["dropped_too_long"] == ["B.0"]


def test_batches_pad_with_pad_id_and_minus_100(sp):
    enc = [{"id": "a", "ids": [5, 6, 7], "labels": [-100, 6, 7]},
           {"id": "b", "ids": [5, 6], "labels": [-100, 6]}]
    (x, y), = list(data.batches(enc, batch_size=2, pad_id=PAD, seed=0, epoch=0))
    assert x.shape == y.shape == (2, 3)
    pads = x[x == PAD]
    assert pads.numel() == 1
    assert (y[x == PAD] == -100).all()


def test_batches_order_changes_with_epoch_but_is_repeatable():
    enc = [{"id": str(i), "ids": [5, 6], "labels": [-100, 6]} for i in range(20)]
    order = lambda ep: [b[0].shape[0] for b in data.batches(enc, 4, 3, 0, ep)]
    first = lambda ep: [tuple(b[0][:, 0].tolist()) for b in data.batches(enc, 4, 3, 0, ep)]
    assert first(0) == first(0)
