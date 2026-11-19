import json

import pytest

from tools.sftbuild import budget
from tools.sftbuild.build import main


@pytest.fixture(autouse=True)
def reset():
    yield
    budget.set_tokenizer(None)


def ex(n, answer):
    return "### ex %04d | A | en | deepseek\n<user>\nq\n</user>\n<assistant>\n%s\n</assistant>\n" % (n, answer)


def src(tmp_path, *answers):
    p = tmp_path / "in.txt"
    p.write_text("\n".join(ex(i + 1, a) for i, a in enumerate(answers)))
    return str(p)


def test_cli_plain_writes_three_files(tmp_path):
    prefix = str(tmp_path / "out")
    assert main(["--mode", "plain", "--src", src(tmp_path, "4."), "--out", prefix, "--skip-budget"]) == 0
    for suffix in (".jsonl", "_preview.txt", "_report.json"):
        assert (tmp_path / ("out" + suffix)).exists()


def test_jsonl_roundtrip(tmp_path):
    prefix = str(tmp_path / "out")
    main(["--mode", "plain", "--src", src(tmp_path, "a", "b"), "--out", prefix, "--skip-budget"])
    rows = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["P0001.0", "P0002.0"]
    assert all("segments" in r for r in rows)


def test_preview_marks_no_loss_lines(tmp_path):
    prefix = str(tmp_path / "out")
    main(["--mode", "plain", "--src", src(tmp_path, "4."), "--out", prefix, "--skip-budget"])
    lines = (tmp_path / "out_preview.txt").read_text().splitlines()
    assert "| <user>" in lines and "4." in lines


def test_cli_exits_1_when_over_budget(tmp_path):
    budget.set_tokenizer(lambda s: len(s.split()))
    prefix = str(tmp_path / "out")
    code = main(["--mode", "plain", "--src", src(tmp_path, "ok", " ".join(["w"] * 5000)), "--out", prefix])
    assert code == 1
    rep = json.loads((tmp_path / "out_report.json").read_text())
    assert rep["over_budget_ids"] == ["P0002.0"] and rep["samples"] == 2 and rep["budget"] == "checked"


def test_report_has_loss_share_and_flags(tmp_path):
    prefix = str(tmp_path / "out")
    main(["--mode", "plain", "--src", src(tmp_path, "4."), "--out", prefix, "--skip-budget"])
    rep = json.loads((tmp_path / "out_report.json").read_text())
    assert 0 < rep["loss_share_by_sample"]["P0001.0"] < 1
    assert rep["flags"] == {} and rep["budget"] == "skipped"


def test_cli_exits_2_when_tokenizer_cannot_load(tmp_path, monkeypatch):
    monkeypatch.setenv("ALETHIC_TOKENIZER", "/nonexistent/tokenizer.json")
    assert main(["--mode", "plain", "--src", src(tmp_path, "4."), "--out", str(tmp_path / "out")]) == 2


def test_cli_terminal_search_replays_from_the_cache(tmp_path):
    skeleton = ("### ex 0703 | F | en | deepseek\nKIND: search_fix\nFILES:\n--- a.txt\n<<<\nhello\n>>>\nUSER: Fix it.\nPLAN: Look it up.\n"
                "STEP: search | QUERY: python unittest has no attribute makeSuite | SOURCE: docs.python.org | EXPECT_IN_RESULT: loadTestsFromTestCase\n"
                "FINAL: Done.\n")
    s = tmp_path / "s.txt"
    s.write_text(skeleton)
    cache = tmp_path / "c.json"
    cache.write_text(json.dumps({"docs.python.org|python unittest has no attribute makeSuite|": "1. unittest\n   use loadTestsFromTestCase"}))
    prefix = str(tmp_path / "out")
    rc = main(["--mode", "terminal", "--src", str(s), "--out", prefix, "--workdir", str(tmp_path / "w"),
               "--search-cache", str(cache), "--search-mode", "replay", "--skip-budget"])
    assert rc == 0
    assert "search(python unittest has no attribute makeSuite)" in (tmp_path / "out.jsonl").read_text()


def test_cli_terminal_search_replay_miss_is_an_error(tmp_path):
    skeleton = ("### ex 0703 | F | en | deepseek\nKIND: search_fix\nFILES:\n--- a.txt\n<<<\nhello\n>>>\nUSER: Fix it.\nPLAN: Look it up.\n"
                "STEP: search | QUERY: python a b c d | SOURCE: docs.python.org | EXPECT_IN_RESULT: x\nFINAL: Done.\n")
    s = tmp_path / "s.txt"
    s.write_text(skeleton)
    cache = tmp_path / "c.json"
    cache.write_text("{}")
    with pytest.raises(Exception, match="recorded"):
        main(["--mode", "terminal", "--src", str(s), "--out", str(tmp_path / "o"), "--workdir", str(tmp_path / "w"),
              "--search-cache", str(cache), "--search-mode", "replay", "--skip-budget"])
