import math
import re

import pytest

from tools.evalset.prompts import eval_set
from tools.mathtraces.build import KINDS, build_all, to_text


@pytest.fixture(scope="module")
def data():
    return build_all(per_kind=40, seed=11)


def calls(e):
    return re.findall(r"<call> (\w+)\((.*?)\) </call>", to_text(e))


def result(e):
    rs = [s["t"] for s in e["segments"] if s["t"].startswith("<result>")]
    return rs[-1][len("<result>\n"):-len("\n</result>\n")]


def user(e):
    return e["segments"][0]["t"][len("<user>\n"):-len("\n</user>\n")]


def final_say(e):
    return e["segments"][-2]["t"]


def test_every_kind_in_both_languages_unique_ids_and_deterministic(data):
    assert {e["kind"] for e in data} == set(KINDS)
    assert len({e["id"] for e in data}) == len(data)
    for k in KINDS:
        assert {e["lang"] for e in data if e["kind"] == k} == {"en", "uz"}, k
    assert build_all(per_kind=3, seed=1) == build_all(per_kind=3, seed=1) != build_all(per_kind=3, seed=2)


def test_every_question_is_answered_by_asking_python_and_the_user_never_has_to_say_so(data):
    plain = 0
    for e in data:
        cs = calls(e)
        assert len(cs) == 1 and cs[0][0] == "shell" and re.match(r"python3? -c \"(print\(.*\)|x = .*; print\(x\))\"$", cs[0][1]), cs
        plain += "python" not in user(e).lower()
    assert plain >= 0.95 * len(data)


def test_the_answer_python_printed_is_right_and_is_what_the_final_sentence_says(data):
    for e in data:
        raw = re.search(r'-c "(.*)"$', calls(e)[0][1]).group(1)
        code = re.match(r"print\((.*)\)$", raw).group(1) if raw.startswith("print(") else re.match(r"x = (.*); print\(x\)$", raw).group(1)
        assert str(eval(code, {"math": math, "round": round})) == result(e), (code, result(e))
        assert result(e) in final_say(e), (result(e), final_say(e))


def test_structure_and_losses(data):
    for e in data:
        segs = e["segments"]
        assert segs[0]["loss"] == 0 and segs[1]["t"] == "<assistant>\n" and segs[-1]["t"] == "</assistant>\n"
        for s in segs:
            if s["t"].startswith(("<result>", "<user>")):
                assert s["loss"] == 0
            elif s["t"].startswith(("<call>", "<plan>")):
                assert s["loss"] == 1
        assert "Traceback" not in to_text(e) and "/tmp" not in to_text(e)


def test_eval_prompts_never_appear_in_training(data):
    text = "".join(to_text(e) for e in data)
    for p in eval_set():
        assert "<user>\n%s\n</user>" % p["prompt"] not in text


def test_requests_are_diverse_and_cover_the_word_problem_shapes(data):
    users = [user(e) for e in data]
    assert len({re.sub(r"\d+(\.\d+)?", "N", u) for u in users}) > 60
    joined = " ".join(users).lower()
    for word in ("average speed", "%", "discount", "area", "km", "minutes", "total", "split"):
        assert word in joined, word


def test_numbers_in_the_question_change_so_nothing_can_be_memorised(data):
    nums = {tuple(re.findall(r"\d+", user(e))) for e in data}
    assert len(nums) > 0.85 * len(data)
