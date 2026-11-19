from tools.evalset.prompts import eval_set
from tools.identity.build import build_identity


def text(e):
    return "".join(s["t"] for s in e["segments"])


def test_about_two_hundred_unique_examples_with_unique_ids():
    d = build_identity(seed=1)
    assert 190 <= len(d) <= 260 and len({text(e) for e in d}) == len(d) and len({e["id"] for e in d}) == len(d)


def test_no_eval_prompt_is_trained_on():
    train = "".join(text(e) for e in build_identity(seed=1))
    for e in eval_set():
        assert "<user>\n%s\n</user>" % e["prompt"] not in train


def test_every_answer_names_alethic_and_loss_is_only_on_the_assistant():
    for e in build_identity(seed=2):
        t = text(e)
        assert "Alethic" in "".join(s["t"] for s in e["segments"] if s["loss"] == 1)
        for s in e["segments"]:
            assert s["loss"] == (1 if not s["t"].startswith("<") else 0)


def test_is_it_another_model_is_always_answered_with_no():
    for e in build_identity(seed=3):
        user = e["segments"][0]["t"]
        if e["kind"] == "not_model":
            ans = e["segments"][2]["t"]
            assert ans.startswith(("No", "Yo'q")), ans


def test_both_languages_and_the_facts_never_drift():
    d = build_identity(seed=4)
    assert {e["lang"] for e in d} == {"en", "uz"}
    for e in d:
        t = text(e)
        assert "310M" not in t and "1B" not in t and "OpenAI" not in t.replace("not OpenAI", "")
