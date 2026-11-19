import json
import os

from tools.release import pack_newborn as pk


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _rows(kind, n):
    return [{"id": "%s_%d.0" % (kind, i), "kind": kind, "lang": "en",
             "segments": [{"t": "hi", "loss": 0}, {"t": "yo", "loss": 1}]} for i in range(n)]


def test_shards_are_numbered_and_complete(tmp_path):
    src = tmp_path / "built"
    src.mkdir()
    _write(src / "math_traces.jsonl", _rows("math", 7))
    _write(src / "skill_traces.jsonl", _rows("skill", 6))
    out = tmp_path / "out"
    info = pk.pack(str(src), str(out), per_shard=5, seed=1, parquet=False)
    names = sorted(os.listdir(out))
    assert "alethic-newborn0001.jsonl" in names
    assert "alethic-newborn0003.jsonl" in names
    assert "alethic-newborn0004.jsonl" not in names
    assert info["examples"] == 13
    assert info["shards"] == 3
    got = []
    for n in names:
        if n.endswith(".jsonl"):
            got += [json.loads(l) for l in open(out / n, encoding="utf-8")]
    assert sorted(r["id"] for r in got) == sorted(r["id"] for r in _rows("math", 7) + _rows("skill", 6))


def test_shards_are_mixed_not_one_kind_each(tmp_path):
    src = tmp_path / "built"
    src.mkdir()
    _write(src / "math_traces.jsonl", _rows("math", 20))
    _write(src / "skill_traces.jsonl", _rows("skill", 20))
    out = tmp_path / "out"
    pk.pack(str(src), str(out), per_shard=10, seed=3, parquet=False)
    first = [json.loads(l)["kind"] for l in open(out / "alethic-newborn0001.jsonl", encoding="utf-8")]
    assert set(first) == {"math", "skill"}


def test_same_seed_same_shards(tmp_path):
    src = tmp_path / "built"
    src.mkdir()
    _write(src / "a_traces.jsonl", _rows("a", 30))
    _write(src / "b_traces.jsonl", _rows("b", 30))
    o1, o2 = tmp_path / "o1", tmp_path / "o2"
    pk.pack(str(src), str(o1), per_shard=25, seed=9, parquet=False)
    pk.pack(str(src), str(o2), per_shard=25, seed=9, parquet=False)
    assert open(o1 / "alethic-newborn0001.jsonl", encoding="utf-8").read() == open(o2 / "alethic-newborn0001.jsonl", encoding="utf-8").read()


def test_manifest_counts_kinds(tmp_path):
    src = tmp_path / "built"
    src.mkdir()
    _write(src / "math_traces.jsonl", _rows("math", 4))
    out = tmp_path / "out"
    pk.pack(str(src), str(out), per_shard=10, seed=1, parquet=False)
    m = json.load(open(out / "alethic-newborn.manifest.json", encoding="utf-8"))
    assert m["name"] == "alethic-newborn"
    assert m["kinds"] == {"math": 4}
    assert m["files"][0]["file"] == "alethic-newborn0001.jsonl"
    assert m["files"][0]["examples"] == 4


def test_the_legal_name_never_reaches_the_released_shards(tmp_path):
    built = tmp_path / "built"
    built.mkdir()
    rows = [{"id": "a", "kind": "who", "lang": "en", "segments": [
        {"t": "<assistant>\nI am Alethic, made by Bekhruz Suleyman.\n</assistant>\n", "loss": 1}]},
        {"id": "b", "kind": "who", "lang": "uz", "segments": [{"t": "Yaratuvchim: Bekhruz Suleyman.", "loss": 1}]}]
    with open(built / "x.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    out = tmp_path / "out"
    pk.pack(str(built), str(out), per_shard=10)
    text = open(out / "alethic-newborn0001.jsonl", encoding="utf-8").read().lower()
    assert "bexruz" not in text and "suleymanov" not in text
    assert "made by bekhruz suleyman." in text
