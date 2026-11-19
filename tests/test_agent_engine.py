import torch

from alethic.agent.engine import Engine


class FakeSP:
    def encode(self, s):
        return [ord(c) for c in s]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class Script:
    """Decoder that 'predicts' a fixed continuation, one id per call."""

    def __init__(self, text, prompt_len, vocab=256):
        self.ids, self.p, self.v = [ord(c) for c in text], prompt_len, vocab

    def logits(self, ids):
        k = len(ids) - self.p
        out = torch.zeros(self.v)
        out[self.ids[k] if k < len(self.ids) else 2] = 10.0
        return out


def engine(text, prompt):
    return Engine(Script(text, len(prompt)), FakeSP(), max_seq_len=10_000)


def test_generates_until_eos_and_streams_the_same_text():
    e, seen = engine("hello", "q:"), []
    out = e.generate("q:", stop=lambda t: False, max_new=50, on_text=seen.append)
    assert out == "hello" and "".join(seen) == "hello"


def test_stop_function_ends_generation_early():
    out = engine("abc</call>zzz", "q:").generate("q:", stop=lambda t: "</call>" in t, max_new=50, on_text=lambda s: None)
    assert out.endswith("</call>") and "zzz" not in out


def test_max_new_limits_length():
    out = engine("x" * 100, "q:").generate("q:", stop=lambda t: False, max_new=7, on_text=lambda s: None)
    assert out == "x" * 7


def test_a_repeating_loop_is_cut_off():
    out = engine("ab" * 400, "q:").generate("q:", stop=lambda t: False, max_new=500, on_text=lambda s: None)
    assert len(out) < 200


def test_context_limit_stops_generation():
    e = Engine(Script("x" * 100, 2), FakeSP(), max_seq_len=10)
    assert len(e.generate("q:", stop=lambda t: False, max_new=50, on_text=lambda s: None)) == 8


def test_top_k_one_sampling_equals_greedy():
    g = engine("hello", "q:")
    s = engine("hello", "q:")
    s.temperature, s.top_k = 0.8, 1
    run = lambda e: e.generate("q:", stop=lambda t: False, max_new=20, on_text=lambda x: None)
    assert run(g) == run(s) == "hello"
