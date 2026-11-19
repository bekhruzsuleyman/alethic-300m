import pytest

from tools.release import push_weights as pw


class FakeApi:
    def __init__(self, files):
        self.files = files
        self.commits = []

    def list_repo_files(self, repo_id, repo_type="model"):
        return self.files

    def create_commit(self, repo_id, operations, commit_message, repo_type="model"):
        self.commits.append((repo_id, [(o.src_path_in_repo, o.path_in_repo) for o in operations], commit_message))


FILES = ["config.json", "model.pt", "runs/sft3c/best/model.pt", "runs/sft3c/best/meta.json",
         "runs/sft3c/last/model.pt", "runs/sft3b/best/model.pt"]


def test_copies_only_the_best_folder_to_the_new_subfolder():
    api = FakeApi(FILES)
    pw.push(api, "u/r", "runs/sft3c/best", "sft/sft3c")
    assert api.commits[0][1] == [("runs/sft3c/best/meta.json", "sft/sft3c/meta.json"),
                                 ("runs/sft3c/best/model.pt", "sft/sft3c/model.pt")]


def test_root_files_are_never_touched():
    api = FakeApi(FILES)
    pw.push(api, "u/r", "runs/sft3c/best", "sft/sft3c")
    assert all(not dst.startswith(("model.pt", "config.json")) for _, dst in api.commits[0][1])


def test_missing_source_stops_without_a_commit():
    api = FakeApi(["model.pt"])
    with pytest.raises(pw.PushError):
        pw.push(api, "u/r", "runs/sft3c/best", "sft/sft3c")
    assert api.commits == []


def test_refuses_a_destination_at_the_root():
    with pytest.raises(pw.PushError):
        pw.push(FakeApi(FILES), "u/r", "runs/sft3c/best", "")
