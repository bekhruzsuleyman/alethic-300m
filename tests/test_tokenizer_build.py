"""M1 pipeline pieces: source checks, parquet reading, the sample builder and the evaluation report."""
from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alethic.config import ConfigError
from alethic.data.encode import is_validation
from alethic.data.fetch import check_source, read_documents
from alethic.tokenizer.codec import Codec
from alethic.tokenizer.evaluate import evaluate_tokenizer, indentation_cases
from alethic.tokenizer.sample import build_sample, read_heldout
from alethic.tokenizer.train import train_tokenizer

PINNED = "a" * 40


def test_check_source():
    with pytest.raises(ConfigError, match="pin it"):
        check_source("math", {"revision": None})
    with pytest.raises(ConfigError, match="approve"):
        check_source("code_other", {"revision": PINNED, "approved": False})
    check_source("code_other", {"revision": PINNED, "approved": True})


def test_read_documents(tmp_path):
    p = tmp_path / "x.parquet"
    pq.write_table(pa.table({"url": ["u1", "u2", "u3"], "text": ["o'zbek", "", "ma'no"]}), p, row_group_size=1)
    stats = {}
    docs = list(read_documents(p, "uzbek_latn", "text", "url", 0, stats))
    assert sorted(docs) == [("u1", "o\u02bbzbek"), ("u3", "ma\u02bcno")]     # empty text skipped, cleaned
    assert stats["uzbek_chars_changed"] == 2
    with pytest.raises(ConfigError, match="no column"):
        list(read_documents(p, "math", "content", "url", 0, {}))


def _docs(source: str, n: int = 5000):
    """Synthetic documents; some lines are too long and some contain slot text."""
    for i in range(n):
        lines = [f"{source} doc {i} line {j} " + "x" * (j * 7) for j in range(6)]
        if i % 50 == 0:
            lines.append("y" * 20000)
        if i % 70 == 0:
            lines.append("bad <sp07> text")
        yield f"{source}-key-{i}", "\n".join(lines)


def test_sample_builder(tmp_path):
    weights = {"python": 22, "code_other": 10, "fineweb_edu": 17, "math": 5, "uzbek_latn": 4, "uzbek_cyrl": 2}
    opened = []

    def open_stream(stream, source):
        opened.append(stream)
        return _docs(stream)

    kw = dict(weights=weights, code_langs=["rust", "go"], size_bytes=600_000, heldout_per_source=20,
              max_line=16384, seed=3)
    partial = build_sample(open_stream, out_dir=tmp_path, only=["math", "uzbek_latn"], **kw)
    assert partial is None and sorted(opened) == ["math", "uzbek_latn"]
    report = build_sample(open_stream, out_dir=tmp_path, **kw)
    assert report is not None and "math" in opened[:2]
    streams = report["streams"]
    assert streams["code_other/rust"]["heldout"] == 10 and streams["python"]["heldout"] == 20
    total = sum(s["bytes"] for s in streams.values())
    for source, w in weights.items():           # mix within 2 points of the weights
        got = sum(s["bytes"] for s in streams.values() if s["source"] == source) / total
        assert abs(got - w / 60) < 0.02, (source, got)
    assert all(s["dropped_long"] > 0 and s["dropped_slot_text"] > 0 for s in streams.values())
    sample = (tmp_path / "sample.txt").read_bytes()
    assert b"<sp07>" not in sample and max(len(l) for l in sample.split(b"\n")) <= 16384
    heldout = read_heldout(tmp_path)
    assert sorted(heldout) == ["code_other", "fineweb_edu", "math", "python", "uzbek_cyrl", "uzbek_latn"]
    assert all(not is_validation("python", f"python-key-{i}") for i in range(20)
               if f"python doc {i} line" in heldout["python"][0][:20])
    held_line = heldout["python"][0].split("\n")[0].encode()
    assert held_line not in sample.split(b"\n")          # held-out documents never in the sample
    first = (tmp_path / "sample.txt").read_bytes()
    (tmp_path / "sample.txt").unlink()
    again = build_sample(open_stream, out_dir=tmp_path, **kw)    # streams reused, merge is deterministic
    assert (tmp_path / "sample.txt").read_bytes() == first and again["bytes"] == report["bytes"]


def test_evaluate_report(tiny_codec):
    code = ["def f(x):\n    if x:\n        for i in x:\n            print(i)\n    return x\n"] * 30
    report = evaluate_tokenizer(tiny_codec, {"python": code, "uzbek_latn": ["oʻzbek tili maktab"] * 30},
                                {"top_pieces": 5, "indentation_max_tokens": 2, "indentation_min_share": 0.99,
                                 "uzbek_max_tokens_per_word": 3.0, "heldout_docs_per_source": 30})
    ind = report["indentation"]
    assert ind["4"]["cases"] == 60 and ind["8"]["cases"] == 30 and ind["12"]["cases"] == 30
    assert all(v["share"] == 1.0 for v in ind.values())
    py = report["sources"]["python"]
    assert py["roundtrip_share"] == 1.0 and len(py["top_learned_pieces"]) == 5
    assert report["gates"] == {"roundtrip_100": True, "indentation": True, "heldout_docs_enough": True}
    json.dumps(report)                                   # the report must be JSON-serialisable
    bad = indentation_cases(tiny_codec, ["a\n            b"], max_tokens=1)
    assert bad["12"] == {"cases": 1, "ok": 0, "share": 0.0}


def test_train_tokenizer_checkpoints(tmp_path, tokenizer_cfg):
    """Intermediate EM checkpoints are saved, committed, and reused on resume after an interruption."""
    from corpus import make_corpus
    corpus = make_corpus(tmp_path / "corpus.txt")
    tags = []

    def stop_after_first(tag: str) -> None:
        tags.append(tag)
        if tag == "ckpt_2000":
            raise KeyboardInterrupt("simulated preemption after ckpt_2000")

    ov = {"vocab_size": 1024, "num_threads": 4, "checkpoint_vocabs": [2000, 1400]}
    with pytest.raises(KeyboardInterrupt):
        train_tokenizer(tokenizer_cfg, corpus, tmp_path / "out", ov, work_dir=tmp_path,
                        on_checkpoint=stop_after_first)
    ckpt_2000 = tmp_path / "spm_checkpoints" / "ckpt_2000.model"
    assert ckpt_2000.exists() and not (tmp_path / "out" / f"{tokenizer_cfg['name']}.model").exists()
    mtime_2000 = ckpt_2000.stat().st_mtime_ns

    resumed_tags = []
    model_path, _ = train_tokenizer(tokenizer_cfg, corpus, tmp_path / "out", ov, work_dir=tmp_path,
                                    on_checkpoint=resumed_tags.append)
    assert ckpt_2000.stat().st_mtime_ns == mtime_2000      # ckpt_2000 was reused, not retrained
    assert resumed_tags == ["ckpt_1400", "model"]
    codec = Codec(model_path, specials_path=tmp_path / "specials.json")
    assert codec.vocab_size == 1024
    assert codec.decode(codec.encode("def f(x):\n    return x\t▁\n")) == "def f(x):\n    return x\t▁\n"
