from tools.opendata.report import flatten_text, pick_file, summarize


def test_flatten_joins_every_string_in_nested_rows():
    row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello", "n": 3}], "id": 7}
    text = flatten_text(row)
    assert "hi" in text and "hello" in text and "assistant" in text and "7" not in text


def test_flatten_accepts_plain_strings_and_none():
    assert flatten_text("abc") == "abc" and flatten_text(None) == ""


def test_summarize_reports_how_many_fit_the_context():
    s = summarize([100, 500, 3000, 5000, 9000], limit=4096)
    assert s["n"] == 5 and s["median"] == 3000 and s["fit"] == 3 and abs(s["fit_pct"] - 60.0) < 1e-9
    assert s["fit_2048"] == 2


def test_summarize_of_nothing_does_not_crash():
    assert summarize([], limit=4096)["n"] == 0


FILES = {"README.md": 10, "data/test-0.parquet": 50, "data/train-0.parquet": 900, "data/train-1.parquet": 300, "x.jsonl": 5}


def test_pick_file_prefers_the_smallest_train_data_file():
    assert pick_file(FILES) == "data/train-1.parquet"


def test_pick_file_skips_files_that_are_too_big_and_non_data_files():
    assert pick_file(FILES, max_bytes=200) == "x.jsonl"
    assert pick_file({"README.md": 1, "a.png": 2}) is None


def test_pick_file_honours_a_preferred_pattern():
    files = {"json-mode.json": 10, "func-calling.json": 30}
    assert pick_file(files, prefer="func-calling") == "func-calling.json"


import json
from types import SimpleNamespace

from tools.opendata.report import read_rows, report_one


def test_read_rows_handles_jsonl_json_and_parquet(tmp_path):
    rows = [{"messages": [{"role": "user", "content": "hi %d" % i}]} for i in range(5)]
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    (tmp_path / "b.json").write_text(json.dumps(rows))
    assert len(read_rows(str(tmp_path / "a.jsonl"))) == 5 and len(read_rows(str(tmp_path / "b.json"), limit=3)) == 3
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist([{"text": "x%d" % i} for i in range(7)]), str(tmp_path / "c.parquet"))
    assert [r["text"] for r in read_rows(str(tmp_path / "c.parquet"), limit=4)] == ["x0", "x1", "x2", "x3"]


class FakeApi:
    def __init__(self, files, card=None):
        self.files, self.card = files, card

    def dataset_info(self, repo, files_metadata=True):
        return SimpleNamespace(card_data=self.card, gated=False,
                               siblings=[SimpleNamespace(rfilename=f, size=s) for f, s in self.files.items()])


def test_report_one_gives_license_fit_and_a_sample(tmp_path):
    f = tmp_path / "train.jsonl"
    f.write_text("\n".join(json.dumps({"t": "w " * n}) for n in (10, 20, 5000)) + "\n")
    api = FakeApi({"README.md": 1, "train.jsonl": 100}, card={"license": "apache-2.0"})
    lines = report_one(api, lambda **kw: str(f), lambda t: len(t.split()), "o/x", "demo", None)
    text = "\n".join(lines)
    assert "apache-2.0" in text and "fit in 4096 tokens: 2 of 3" in text and "first row starts" in text


def test_report_one_says_so_when_nothing_can_be_sampled():
    lines = report_one(FakeApi({"README.md": 1}), None, None, "o/x", "demo", None)
    assert any("no data file" in l for l in lines)


def test_report_one_works_when_the_card_has_no_license():
    api = FakeApi({"a.jsonl": 5}, card=None)
    import tempfile, os
    p = os.path.join(tempfile.mkdtemp(), "a.jsonl"); open(p, "w").write('{"t": "hello"}\n')
    assert "license: None" in "\n".join(report_one(api, lambda **kw: p, len, "o/x", "demo", None))
