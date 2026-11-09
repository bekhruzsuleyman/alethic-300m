"""End-to-end M2/M3/M4 tests: dedup, decontam, parallel/serial encode_units, and validation BPB."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from alethic import cli
from alethic.data.decontam import DecontamIndex, build_benchmark_ngrams, extract_ngrams
from alethic.data.dedup import build_drop_list, hash_texts, read_unit_hash_meta, write_unit_hashes
from alethic.data.encode import encode_units
from alethic.eval.bpb import bits_per_byte, evaluate_bpb, load_val_slices
from alethic.model.transformer import ModelConfig, Transformer
from alethic.tokenizer.freeze import write_frozen
from conftest import tiny_model_cfg, tiny_train_cfg


def test_dedup_keeps_first_in_unit_order(tmp_path):
    """Global deduplication keeps the first copy in (source, file, row) unit order."""
    u1_hashes = hash_texts(["doc A", "doc B", "doc A"])
    u2_hashes = hash_texts(["doc C", "doc B", "doc D"])
    write_unit_hashes(tmp_path, "python/f1:0-1", u1_hashes, 15)
    write_unit_hashes(tmp_path, "math/f1:0-1", u2_hashes, 15)
    assert read_unit_hash_meta(tmp_path, "python/f1:0-1")["docs"] == 3

    drops, summary = build_drop_list(["python/f1:0-1", "math/f1:0-1"], tmp_path)
    assert drops["python/f1:0-1"] == {2}      # second "doc A" in unit 1 dropped
    assert drops["math/f1:0-1"] == {1}        # "doc B" already seen in unit 1 -> dropped in unit 2
    assert summary["total_docs"] == 6 and summary["unique_docs"] == 4 and summary["dropped_duplicates"] == 2


def test_decontam_13gram_index(tmp_path):
    """A document sharing any 13-word sequence with a benchmark is flagged; 12-word overlap is kept."""
    bench_doc = "def has_close_elements ( numbers : List [ float ] , threshold : float ) -> bool : return True"
    assert len(extract_ngrams(bench_doc, n=13)) == len(bench_doc.split()) - 12
    idx = DecontamIndex.from_texts([bench_doc], n=13)
    leaked = "prefix words here " + " ".join(bench_doc.split()[:13]) + " suffix words"
    clean_12 = "prefix words here " + " ".join(bench_doc.split()[:12]) + " different ending"
    assert idx.is_contaminated(leaked) is True
    assert idx.is_contaminated(clean_12) is False
    assert idx.is_contaminated("short text") is False

    # Also test build_benchmark_ngrams with a local benchmark parquet file.
    bench_dir = tmp_path / "raw" / "_benchmarks" / "humaneval" / "openai_humaneval"
    bench_dir.mkdir(parents=True)
    pq.write_table(pa.table({"prompt": [bench_doc], "canonical_solution": ["pass"]}),
                   bench_dir / "test-00000.parquet")
    cfg = {"ngram": 13, "benchmarks": {
        "humaneval": {"repo": "local", "paths": ["openai_humaneval"], "revision": "a" * 40,
                      "columns": ["prompt", "canonical_solution"]},
    }}
    built = build_benchmark_ngrams(cfg, tmp_path / "raw")
    assert built.is_contaminated(leaked) is True
    # Second call reads from cached ngrams_13.json.
    cached = build_benchmark_ngrams(cfg, tmp_path / "raw")
    assert cached.total_ngrams == built.total_ngrams


def _write_local_corpus(raw_root: Path) -> None:
    """Create tiny local parquet files for all 6 sources plus a benchmark."""
    leaked_13 = "def leaked_benchmark_func ( a , b , c , d , e , f ) : return a + b"
    bench_dir = raw_root / "_benchmarks" / "humaneval" / "openai_humaneval"
    bench_dir.mkdir(parents=True)
    pq.write_table(pa.table({"prompt": [leaked_13], "canonical_solution": ["return 42"]}),
                   bench_dir / "test.parquet")

    # Python: include a duplicate doc, a leaked doc, and normal docs (plus a validation key).
    py_dir = raw_root / "python"
    py_dir.mkdir(parents=True)
    py_keys = [f"repo_{i}" for i in range(40)]
    py_texts = [f"def func_{i}(x):\n    return x + {i}\n" for i in range(40)]
    py_texts[1] = py_texts[0]                          # duplicate of doc 0
    py_texts[2] = f"# header\n{leaked_13}\n"           # contaminated by benchmark 13-gram
    pq.write_table(pa.table({"repo_name": py_keys, "text": py_texts}), py_dir / "train.parquet", row_group_size=10)

    # code_other (rust):
    co_dir = raw_root / "code_other" / "rust"
    co_dir.mkdir(parents=True)
    pq.write_table(pa.table({"repo_name": [f"rs_{i}" for i in range(25)],
                             "content": [f"fn main_{i}() {{ let x = {i}; }}\n" for i in range(25)]}),
                   co_dir / "train.parquet", row_group_size=10)

    # fineweb_edu:
    fw_dir = raw_root / "fineweb_edu" / "data"
    fw_dir.mkdir(parents=True)
    pq.write_table(pa.table({"id": [f"fw_{i}" for i in range(25)],
                             "text": [f"Educational article number {i} about science and math." for i in range(25)]}),
                   fw_dir / "train.parquet", row_group_size=10)

    # math:
    ma_dir = raw_root / "math" / "data"
    ma_dir.mkdir(parents=True)
    pq.write_table(pa.table({"url": [f"http://math/{i}" for i in range(25)],
                             "text": [f"Theorem {i}: For all x > {i}, x + 1 > {i}." for i in range(25)]}),
                   ma_dir / "train.parquet", row_group_size=10)

    # uzbek_latn (with apostrophes that normalize_uzbek cleans):
    ul_dir = raw_root / "uzbek_latn" / "data"
    ul_dir.mkdir(parents=True)
    pq.write_table(pa.table({"url": [f"http://uzl/{i}" for i in range(25)],
                             "text": [f"O'zbekiston ma'naviyat maktabi {i}-dars." for i in range(25)]}),
                   ul_dir / "train.parquet", row_group_size=10)

    # uzbek_cyrl:
    uc_dir = raw_root / "uzbek_cyrl" / "data"
    uc_dir.mkdir(parents=True)
    pq.write_table(pa.table({"url": [f"http://uzc/{i}" for i in range(25)],
                             "text": [f"Ўзбекистон маънавият мактаби {i}-дарс." for i in range(25)]}),
                   uc_dir / "train.parquet", row_group_size=10)


def test_encode_units_and_eval_bpb(tmp_path, monkeypatch, tiny_codec):
    """encode_units hashes, deduplicates, decontaminates, writes train/val shards, and evaluates BPB."""
    root = tmp_path / "data"
    monkeypatch.setenv("ALETHIC_DATA_DIR", str(root))
    tok_dir = root / "tokenizer" / "alethic-tokenizer-v1"
    tok_dir.mkdir(parents=True)
    model_file = tok_dir / "alethic-tokenizer-v1.model"
    model_file.write_bytes(Path(tiny_codec.model_path).read_bytes())
    (tok_dir / "tokenizer.yaml").write_text("name: alethic-tokenizer-v1\n", encoding="utf-8")
    (tok_dir / "report.json").write_text("{}", encoding="utf-8")
    frozen = write_frozen(tok_dir, model_file, tok_dir / "tokenizer.yaml", tok_dir / "report.json")

    _write_local_corpus(root / "raw")
    rev = "a" * 40
    data_cfg = {
        "paths": {"raw": "raw", "hashes": "hashes", "shards": "shards",
                  "tokenizer": "tokenizer/alethic-tokenizer-v1"},
        "target_scale": 1.0,
        "overfetch": 1.5,
        "unit_max_tokens": 200,
        "shard_tokens": 500,
        "eos_id": 2,
        "val_fraction": 0.25,     # 25% validation in this tiny test so every source gets both train & val shards
        "sources": {
            "python": {"repo": "local", "paths": ["."], "revision": rev,
                       "text_column": "text", "key_column": "repo_name", "tokens": 300, "bytes_per_token": 3.5},
            "code_other": {"repo": "local", "paths": ["rust"], "revision": rev, "approved": True,
                           "text_column": "content", "key_column": "repo_name", "tokens": 200, "bytes_per_token": 3.2},
            "fineweb_edu": {"repo": "local", "paths": ["data"], "revision": rev,
                            "text_column": "text", "key_column": "id", "tokens": 250, "bytes_per_token": 5.0},
            "math": {"repo": "local", "paths": ["data"], "revision": rev,
                     "text_column": "text", "key_column": "url", "tokens": 150, "bytes_per_token": 4.0},
            "uzbek_latn": {"repo": "local", "paths": ["data"], "revision": rev, "license": "CC-BY-4.0",
                           "text_column": "text", "key_column": "url", "tokens": 150, "bytes_per_token": 4.0},
            "uzbek_cyrl": {"repo": "local", "paths": ["data"], "revision": rev, "license": "CC-BY-4.0",
                           "text_column": "text", "key_column": "url", "tokens": 100, "bytes_per_token": 5.5},
        },
        "decontam": {"ngram": 13, "benchmarks": {
            "humaneval": {"repo": "local", "paths": ["openai_humaneval"], "revision": rev,
                          "columns": ["prompt", "canonical_solution"]},
        }},
    }
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data_cfg), encoding="utf-8")

    # Run via CLI with 1 worker, then rerun to verify state.jsonl idempotence.
    assert cli.main(["data-build", "--data", str(data_yaml), "--workers", "1"]) == 0
    manifest1 = json.loads((root / "shards" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest1["tokenizer_sha256"] == frozen["model_sha256"]
    assert manifest1["dedup"]["dropped_duplicates"] >= 1
    assert manifest1["decontam_dropped"]["python"] == 1
    assert manifest1["uzbek_chars_changed"]["uzbek_latn"] >= 14
    assert manifest1["train_tokens"] > 0 and manifest1["val_tokens"] > 0
    assert set(manifest1["attributions"].keys()) == {"uzbek_latn", "uzbek_cyrl"}

    # Rerun over finished state.jsonl produces the exact same shard list and token counts.
    manifest2 = encode_units(data_cfg, workers=1)
    assert manifest2["shards"] == manifest1["shards"]
    assert manifest2["total_tokens"] == manifest1["total_tokens"]

    # Verify BPB calculation and evaluate_bpb on the generated validation shards.
    assert math.isclose(bits_per_byte(math.log(2) * 100, 50), 2.0)
    slices = load_val_slices(root / "shards", max_tokens_per_source=50)
    assert len(slices) > 0
    mcfg = tiny_model_cfg("sgca")
    mcfg["vocab_size"] = tiny_codec.vocab_size
    model = Transformer(ModelConfig.from_dict(mcfg))
    bpb_res = evaluate_bpb(model, root / "shards", seq_len=16, micro_batch=2, device="cpu")
    assert bpb_res["overall_bpb"] is not None and bpb_res["overall_bpb"] > 0
