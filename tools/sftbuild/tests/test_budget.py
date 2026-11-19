import pytest

from tools.sftbuild import budget


@pytest.fixture(autouse=True)
def reset():
    yield
    budget.set_tokenizer(None)


def sample(*texts):
    return {"id": "s", "segments": [{"t": t, "loss": 0} for t in texts]}


def test_budget_counts_all_segments():
    budget.set_tokenizer(lambda s: len(s.split()))
    assert budget.sample_tokens(sample("a b\n", "c d e")) == 5


def test_over_budget_true_for_5000_token_sample():
    budget.set_tokenizer(lambda s: len(s.split()))
    big = sample(" ".join(["w"] * 5000))
    assert budget.over_budget(big) is True
    assert budget.over_budget(sample("a b c")) is False
    assert budget.over_budget(big, limit=6000) is False


def test_missing_tokenizer_raises_not_guesses(monkeypatch):
    monkeypatch.setenv("ALETHIC_TOKENIZER", "/nonexistent/path/tokenizer.json")
    with pytest.raises(budget.BudgetError):
        budget.count_tokens("hello")


def _tiny_model(tmp_path):
    import sentencepiece as spm
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n".join("hello world %d the quick brown fox <user> </assistant>" % i for i in range(200)))
    prefix = str(tmp_path / "tiny")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=prefix, vocab_size=300, model_type="unigram",
                                   byte_fallback=True, add_dummy_prefix=False, normalization_rule_name="identity")
    return prefix + ".model", spm.SentencePieceProcessor(model_file=prefix + ".model")


def test_sentencepiece_model_file_is_counted_with_that_model(tmp_path, monkeypatch):
    model, sp = _tiny_model(tmp_path)
    monkeypatch.setenv("ALETHIC_TOKENIZER", model)
    text = "<user>\nhello world\n</user>\n"
    assert budget.count_tokens(text) == len(sp.encode(text))


def test_local_model_is_found_without_the_environment_variable(tmp_path, monkeypatch):
    model, sp = _tiny_model(tmp_path)
    monkeypatch.delenv("ALETHIC_TOKENIZER", raising=False)
    monkeypatch.setattr(budget, "LOCAL_MODEL", model)
    assert budget.count_tokens("hello world") == len(sp.encode("hello world"))
