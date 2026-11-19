import os

from tools.release import export_agent as ea


def _put(root, rel, text=""):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)


def _fake_repo(tmp_path):
    r = str(tmp_path / "repo")
    _put(r, "src/alethic/__init__.py")
    _put(r, "src/alethic/agent/__init__.py")
    _put(r, "src/alethic/agent/cli.py", "from alethic.model.net import Net\nfrom alethic.agent import loop\n")
    _put(r, "src/alethic/agent/loop.py", "import alethic.util\n")
    _put(r, "src/alethic/model/__init__.py")
    _put(r, "src/alethic/model/net.py", "from alethic.model.layers import L\n")
    _put(r, "src/alethic/model/layers.py")
    _put(r, "src/alethic/util.py")
    _put(r, "src/alethic/train/__init__.py")
    _put(r, "src/alethic/train/loop.py", "x = 1\n")
    _put(r, "tests/conftest.py", "from alethic.data.shards import to_tokens\n")
    _put(r, "src/alethic/data/__init__.py")
    _put(r, "src/alethic/data/shards.py")
    _put(r, "tests/test_agent_cli.py")
    _put(r, "tests/test_train_loop.py")
    _put(r, "deploy/31_agent_sft3c.bat", "echo hi\n")
    _put(r, "deploy/27_train_sft3c_mom_account.bat", "echo train\n")
    _put(r, "pyproject.toml", "[project]\n")
    _put(r, "configs/model_gqa.yaml", "n: 1\n")
    _put(r, "configs/train.yaml", "lr: 1\n")
    return r


def test_closure_follows_alethic_imports_and_skips_the_rest(tmp_path):
    r = _fake_repo(tmp_path)
    got = ea.module_closure(r, ["alethic.agent"])
    assert "src/alethic/model/net.py" in got and "src/alethic/model/layers.py" in got
    assert "src/alethic/util.py" in got and "src/alethic/agent/loop.py" in got
    assert "src/alethic/train/loop.py" not in got


def test_export_copies_agent_tests_and_launchers_only(tmp_path):
    r = _fake_repo(tmp_path)
    out = str(tmp_path / "out")
    ea.export(r, out)
    have = {os.path.relpath(os.path.join(a, f), out).replace(os.sep, "/") for a, _, fs in os.walk(out) for f in fs}
    assert {"src/alethic/agent/cli.py", "tests/test_agent_cli.py",
            "deploy/31_agent_sft3c.bat", "configs/model_gqa.yaml", "README.md", "LICENSE", "pyproject.toml"} <= have
    assert "tests/conftest.py" not in have and "src/alethic/data/shards.py" not in have
    assert "tests/test_train_loop.py" not in have and "src/alethic/train/loop.py" not in have
    assert "deploy/27_train_sft3c_mom_account.bat" not in have and "configs/train.yaml" not in have


def test_the_scan_is_the_same_private_text_scan(tmp_path):
    r = _fake_repo(tmp_path)
    _put(r, "src/alethic/agent/cli.py", "# " + "dil" + "noza" + "\n")
    out = str(tmp_path / "out")
    ea.export(r, out)
    assert ea.scan(out)


def test_export_leaves_a_marker_the_publish_script_checks(tmp_path):
    r = _fake_repo(tmp_path)
    out = str(tmp_path / "out")
    ea.export(r, out)
    assert os.path.exists(os.path.join(out, ".alethic_export_ok"))
