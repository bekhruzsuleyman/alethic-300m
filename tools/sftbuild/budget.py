"""Token budget per sample, counted with the real tokenizer (never a characters-per-token guess)."""
import os

DEFAULT_TOKENIZER = "bekhruzsuleyman/alethic-300m"  # or a local .model / tokenizer.json via ALETHIC_TOKENIZER
# the frozen SentencePiece model inside the repo, used when ALETHIC_TOKENIZER is not set and the file exists
LOCAL_MODEL = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tokenizer", "alethic-tokenizer-v1",
                           "alethic-tokenizer-v1.model")
_counter = None


class BudgetError(Exception):
    pass


def set_tokenizer(fn):
    """Inject a counting function (tests); None resets to lazy loading."""
    global _counter
    _counter = fn


def _load():
    where = os.environ.get("ALETHIC_TOKENIZER") or (LOCAL_MODEL if os.path.exists(LOCAL_MODEL) else DEFAULT_TOKENIZER)
    try:
        if where.endswith(".model"):
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=where)
            return lambda s: len(sp.encode(s))
        if where.endswith(".json"):
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(where)
            return lambda s: len(tok.encode(s).ids)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(where)
        return lambda s: len(tok(s)["input_ids"])
    except Exception as exc:
        raise BudgetError("cannot load tokenizer %r: %s" % (where, exc)) from exc


def count_tokens(text: str) -> int:
    global _counter
    if _counter is None:
        _counter = _load()
    return _counter(text)


def sample_tokens(sample: dict) -> int:
    return count_tokens("".join(seg["t"] for seg in sample["segments"]))


def over_budget(sample: dict, limit: int = 4096) -> bool:
    return sample_tokens(sample) > limit
