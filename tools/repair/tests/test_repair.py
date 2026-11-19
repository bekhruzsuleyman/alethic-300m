import pytest

from tools.repair.repair import RepairError, repair, to_instant
from tools.sftbuild.normalize import Example, parse_examples


def ex(id_, type_="C", user="q", asst="a", source="deepseek"):
    return Example(id_, type_, "en", source, user, asst)


def fenced(id_, type_, user, asst, source="deepseek"):
    return "### ex %s | type: %s | lang: en | diff: easy | topic: t | source: %s\n```text\n%s\n```\n```text\n%s\n```\n" % (
        id_, type_, source, user, asst)


def test_to_instant_drops_the_thinking_block_and_relabels():
    e = to_instant(ex("0068", "C", asst="<thinking>\nstep\nstep\n</thinking>\nO(log n)."))
    assert e.type == "A" and e.assistant == "O(log n)."


def test_to_instant_refuses_an_example_without_thinking():
    with pytest.raises(RepairError):
        to_instant(ex("0068", asst="O(log n)."))


def test_removed_ids_are_gone_and_the_rest_is_converted_to_tag_format(tmp_path):
    text = fenced("0001", "A", "q1", "a1") + fenced("0002", "A", "q2", "a2")
    out, report = repair(text, {"remove": {"0002"}}, str(tmp_path))
    assert [e.id for e in parse_examples(out)] == ["0001"] and report["removed"] == ["0002"]
    assert "<user>\nq1\n</user>\n<assistant>\na1\n</assistant>" in out


def test_removing_an_id_that_is_not_there_fails_loudly(tmp_path):
    with pytest.raises(RepairError):
        repair(fenced("0001", "A", "q", "a"), {"remove": {"0009"}}, str(tmp_path))


def test_patch_must_match_exactly_once(tmp_path):
    text = fenced("0001", "A", "q", "one two one")
    with pytest.raises(RepairError):
        repair(text, {"patch": {"0001": [("assistant", "one", "x")]}}, str(tmp_path))
    out, _ = repair(text, {"patch": {"0001": [("assistant", "two", "2")]}}, str(tmp_path))
    assert "one 2 one" in out


def test_rewrite_replaces_the_whole_assistant_text(tmp_path):
    out, _ = repair(fenced("0001", "A", "q", "old"), {"rewrite": {"0001": ("q2", "new")}}, str(tmp_path))
    e = parse_examples(out)[0]
    assert (e.user, e.assistant) == ("q2", "new")


def test_replayed_examples_get_real_results_and_source_replay(tmp_path):
    text = fenced("0001", "E", "What is 2+3?", "<call> calc(2+3) </call>\n<result> 99 </result>\n5.", source="script")
    out, report = repair(text, {"fixtures": {"0001": {}}}, str(tmp_path))
    e = parse_examples(out)[0]
    assert e.source == "replay" and "<result> 5 </result>" in e.assistant
    assert report["replayed"] == ["0001"] and report["changed_results"] == ["0001"]


def test_unchanged_results_are_not_reported_as_changed(tmp_path):
    text = fenced("0001", "E", "q", "<call> calc(2+3) </call>\n<result> 5 </result>\n5.", source="script")
    _, report = repair(text, {"fixtures": {"0001": {}}}, str(tmp_path))
    assert report["replayed"] == ["0001"] and report["changed_results"] == []


def test_instant_ids_are_relabelled(tmp_path):
    text = fenced("0068", "C", "q", "<thinking>\nx\n</thinking>\nA.")
    out, report = repair(text, {"instant": {"0068"}}, str(tmp_path))
    assert parse_examples(out)[0].type == "A" and report["instant"] == ["0068"]
