import pytest

from tools.release import upload_dataset as up


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, **kw):
        self.calls.append(("create_repo", kw))

    def upload_file(self, **kw):
        self.calls.append(("upload_file", kw["path_in_repo"]))

    def update_repo_settings(self, **kw):
        self.calls.append(("settings", kw))


def _make(tmp_path, shards=2):
    d = tmp_path / "dataset"
    d.mkdir()
    for i in range(1, shards + 1):
        (d / ("alethic-newborn%04d.jsonl" % i)).write_text("{}\n")
        (d / ("alethic-newborn%04d.parquet" % i)).write_bytes(b"PAR1")
    (d / "alethic-newborn.manifest.json").write_text("{}")
    card = tmp_path / "card.md"
    card.write_text("---\nlicense: cc-by-4.0\n---\n# x\n")
    return str(d), str(card)


def test_files_are_card_manifest_and_every_shard(tmp_path):
    d, card = _make(tmp_path, shards=2)
    names = [r for _, r in up.files_to_upload(d, card)]
    assert names == ["README.md", "alethic-newborn.manifest.json",
                     "alethic-newborn0001.jsonl", "alethic-newborn0001.parquet",
                     "alethic-newborn0002.jsonl", "alethic-newborn0002.parquet"]


def test_missing_shards_stop_the_upload(tmp_path):
    d = tmp_path / "dataset"
    d.mkdir()
    card = tmp_path / "card.md"
    card.write_text("x")
    with pytest.raises(up.UploadError):
        up.files_to_upload(str(d), str(card))


def test_upload_creates_repo_sends_files_then_turns_gate_on(tmp_path):
    d, card = _make(tmp_path, shards=1)
    api = FakeApi()
    up.upload(api, "bekhruzsuleyman/alethic-newborn", d, card)
    kinds = [c[0] for c in api.calls]
    assert kinds[0] == "create_repo" and api.calls[0][1]["repo_type"] == "dataset" and api.calls[0][1]["exist_ok"]
    assert kinds[-1] == "settings" and api.calls[-1][1]["gated"] == "auto"
    assert kinds.count("upload_file") == 4


def test_card_has_gate_fields_and_matching_license(tmp_path):
    text = open(up.CARD_PATH, encoding="utf-8").read()
    head = text.split("---")[1]
    assert "extra_gated_prompt:" in head and "extra_gated_fields:" in head
    assert "license: cc-by-4.0" in head
