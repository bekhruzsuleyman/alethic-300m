import re

import pytest

from tools.evalset.prompts import eval_set
from tools.progtraces.build import KINDS, build_all, to_text


@pytest.fixture(scope="module")
def data():
    return build_all(per_kind=40, seed=5)


def last_result(e):
    rs = [s["t"] for s in e["segments"] if s["t"].startswith("<result>")]
    return rs[-1][len("<result>\n"):-len("\n</result>\n")]


def final_say(e):
    return e["segments"][-2]["t"]


def calls(e):
    return re.findall(r"<call> (\w+)\((.*?)\) </call>", to_text(e))


def test_every_kind_is_built_with_unique_ids_and_both_languages(data):
    assert {e["kind"] for e in data} == set(KINDS)
    assert len({e["id"] for e in data}) == len(data)
    assert {e["lang"] for e in data} == {"en", "uz"}


def test_same_seed_same_data_other_seed_differs():
    assert build_all(per_kind=4, seed=1) == build_all(per_kind=4, seed=1) != build_all(per_kind=4, seed=2)


def test_structure_and_nothing_is_a_failure_or_leaks_a_real_path(data):
    for e in data:
        segs = e["segments"]
        assert segs[0]["t"].startswith("<user>\n") and segs[0]["loss"] == 0 and segs[1]["t"] == "<assistant>\n" and segs[-1]["t"] == "</assistant>\n"
        text = to_text(e)
        assert "/tmp" not in text and "Traceback" not in text and "Error" not in text and "error:" not in text
        for s in segs:
            if s["t"].startswith("<result>") or s["t"].startswith("<user>"):
                assert s["loss"] == 0
            else:
                assert s["loss"] == 1 or s["t"] == "<assistant>\n" or s["t"] == "</assistant>\n"


def test_the_final_sentence_only_says_what_the_real_output_shows(data):
    """Honesty: every printed line (or the first and last of a long output) appears in the final words."""
    for e in data:
        if e["kind"] not in ("write_run", "quick_calc", "edit_pos"):
            continue
        out = last_result(e).split("\n")
        say = final_say(e)
        need = out if len(out) <= 8 else [out[0], out[-1]]
        for line in need:
            assert line in say, (e["id"], line, say)


def test_the_program_depends_on_the_request_so_it_cannot_be_copied(data):
    """The numbers in the request appear in the written code or the command."""
    checked = 0
    for e in data:
        if e["kind"] != "write_run":
            continue
        user = e["segments"][0]["t"]
        nums = set(re.findall(r"\b\d+\b", user))
        code = "".join(s["t"] for s in e["segments"] if "<call> write(" in s["t"])
        for n in nums:
            assert n in code, (user, code)
            checked += 1
    assert checked > 20


def test_requests_and_programs_are_diverse(data):
    wr = [e for e in data if e["kind"] == "write_run"]
    users = {e["segments"][0]["t"] for e in wr}
    codes = {"".join(s["t"] for s in e["segments"] if "<call> write(" in s["t"]) for e in wr}
    firsts = {re.sub(r"[\w.']+", "w", e["segments"][0]["t"])[:30] for e in wr}
    assert len(users) >= len(wr) * 0.95 and len(codes) >= len(wr) * 0.9 and len(firsts) >= 4


def test_eval_prompts_never_appear_in_training(data):
    text = "".join(to_text(e) for e in data)
    for p in eval_set():
        assert "<user>\n%s\n</user>" % p["prompt"] not in text


def test_edit_targets_the_line_the_read_showed_and_lines_are_spread_out():
    eds = [e for e in build_all(per_kind=150, seed=9) if e["kind"] == "edit_pos"]
    assert len(eds) >= 100
    starts = []
    for e in eds:
        text = to_text(e)
        user = e["segments"][0]["t"]
        name = re.search(r"\b([A-Z]{3,12})\b", user).group(1)
        shown = re.search(r"^(\d+)\s+%s = " % name, last_read(e), re.M)
        k = int(shown.group(1))
        m = re.search(r"<call> edit\((.+?), (\d+)-(\d+)\) </call>", text)
        assert int(m.group(2)) == int(m.group(3)) == k, (user, k, m.groups())
        starts.append(k)
    assert len(set(starts)) >= 20 and starts.count(1) <= len(starts) * 0.1 and max(starts) >= 25


def last_read(e):
    for s in e["segments"]:
        if s["t"].startswith("<result>") and re.search(r"^\d+\s", s["t"][9:], re.M):
            return s["t"]
    raise AssertionError("no read result")


def test_one_liners_ask_python_for_the_answer_and_quote_it(data):
    q = [e for e in data if e["kind"] == "quick_calc"]
    assert q and all(calls(e)[0][0] == "shell" and "python" in calls(e)[0][1] for e in q)


def test_a_folder_listing_names_exactly_the_files_that_exist(data):
    for e in (x for x in data if x["kind"] == "list_files"):
        listed = last_result(e).split("\n")
        for name in listed:
            assert name in final_say(e)


def test_uzbek_calc_requests_have_no_stray_case_endings():
    for e in build_all(per_kind=60, seed=3):
        if e["kind"] == "quick_calc" and e["lang"] == "uz":
            user = e["segments"][0]["t"]
            assert " ni ni " not in user and "darajasi ni" not in user and "foizi ni" not in user and "ildizi ni" not in user and "qoldiq ni" not in user


def test_read_range_is_a_constant_the_model_can_know_not_the_file_length():
    from tools.progtraces.build import build_all as ba
    es = [e for e in ba(per_kind=0, seed=3, counts={"edit_pos": 60})]
    ranges = {re.search(r"read\(.*?, (\d+-\d+)\)", to_text(e)).group(1) for e in es}
    assert ranges == {"1-100"}, ranges
