import json
import re

import pytest

from tools.errtraces.build import KINDS, build_all, to_text

FAIL = r"error|Error|blocked|Traceback|No such file"


@pytest.fixture(scope="module")
def data():
    return build_all(per_kind=6, seed=7)


def test_every_kind_is_built_and_ids_are_unique(data):
    assert {e["kind"] for e in data} == set(KINDS)
    assert len({e["id"] for e in data}) == len(data) == 6 * len(KINDS)


def test_same_seed_gives_the_same_data_and_another_seed_differs():
    a, b, c = build_all(per_kind=3, seed=1), build_all(per_kind=3, seed=1), build_all(per_kind=3, seed=2)
    assert a == b and a != c


def test_structure_user_first_assistant_closed_and_no_real_folder(data):
    for e in data:
        segs = e["segments"]
        assert segs[0]["t"].startswith("<user>\n") and segs[0]["loss"] == 0
        assert segs[1]["t"] == "<assistant>\n" and segs[-1]["t"] == "</assistant>\n"
        text = to_text(e)
        assert "/tmp" not in text and "pytest-of" not in text and "\\Users" not in text


def test_every_example_contains_a_real_failure_that_is_not_trained_on(data):
    for e in data:
        segs = e["segments"]
        i = next(i for i, s in enumerate(segs) if s["t"].startswith("<result>") and re.search(FAIL, s["t"]))
        assert segs[i]["loss"] == 0                      # a result is never trained on
        call = max(j for j in range(i) if segs[j]["t"].startswith("<call>"))
        assert segs[call]["loss"] == 0                   # the mistake that caused it is not taught either


def test_the_recovery_after_the_failure_is_trained_on(data):
    for e in data:
        segs = e["segments"]
        i = next(i for i, s in enumerate(segs) if s["t"].startswith("<result>") and re.search(FAIL, s["t"]))
        later = [s for s in segs[i + 1:] if not s["t"].startswith("<result>") and s["t"] != "</assistant>\n"]
        assert later and all(s["loss"] == 1 for s in later)


def test_bug_fix_runs_really_end_clean_or_in_an_honest_stop(data):
    for e in data:
        text = to_text(e)
        if e["kind"] in ("bug_fix", "bug_fix_edit_range"):
            last_result = re.findall(r"<result>\n(.*?)\n</result>", text, re.S)[-1]
            assert "Traceback" not in last_result and "Error" not in last_result
        if e["kind"] in ("blocked", "give_up"):
            assert not re.search(r"<call>[^\n]*</call>\s*</assistant>", text)      # the turn ends with words, not a call
            assert not segs_end_with_call(e)


def segs_end_with_call(e):
    return e["segments"][-2]["t"].startswith("<call>")


def test_languages_are_mixed_and_tagged(data):
    assert {e["lang"] for e in data} >= {"en", "uz"}


def test_text_is_valid_json_lines(data):
    for e in data:
        json.loads(json.dumps(e))


def test_all_texts_are_unique():
    d = build_all(per_kind=40, seed=3)
    assert len({to_text(e) for e in d}) == len(d)


def test_uzbek_bug_fix_request_is_a_clean_sentence():
    for e in build_all(per_kind=10, seed=5):
        if e["kind"] == "bug_fix" and e["lang"] == "uz":
            user = e["segments"][0]["t"]
            assert user.endswith(" yozib ber.\n</user>\n") and user.startswith("<user>\n") and user[7].isupper()


def test_plans_match_what_the_example_does():
    for e in build_all(per_kind=30, seed=9):
        if e["kind"] in ("edit_range", "read_range", "read_no_range", "unknown_tool", "blocked", "missing_file"):
            plan = e["segments"][2]["t"]
            assert "yozaman" not in plan and "write one file" not in plan and "write the code" not in plan
