"""Tokenizer tests on a tiny model trained with the real tokenizer.yaml settings."""
from __future__ import annotations

import random

import pytest

from alethic.tokenizer.specials import NUM_SLOTS, slot_piece

HARD_CASES = [
    "def f(x):\n    return x\n",
    "def f(x):\r\n\treturn x  \r\n",            # CRLF, tab, trailing spaces
    "    leading spaces\n\n\n   \t \n",
    "if a:\n        b = 1\n            c = 2\n",
    "   ",                                         # whitespace only
    "\t\t\t",
    "\r",
    "x = 'oʻzbek gʻalaba maʼno'  # Uzbek",
    "Ўзбекистон — давлат.",
    "emoji: 😀🎉👍🏽 👨‍👩‍👧‍👦 🇺🇿",
    "e\u0301 combining, NFKC-sensitive: ﬁ ① ｶ ㎏ Ⅳ",
    "CJK 漢字 かな 한국어",
    "math $\\int_0^1 f(x)\\,dx$ and ∀x∈ℝ",
    "zero\u200bwidth\u200dand\ufeffbom",
    "nul\x00and\x07bell\x1b[0m",
    "nbsp\u00a0ideographic\u3000space",
    "<sp00> <sp63> <thinking> </s> <s> <unk> <pad>",
    "",
    "x" * 5000,
    " " * 100,
]


def _random_text(rng: random.Random) -> str:
    """Random text mixing ASCII, whitespace, several scripts, emoji and arbitrary code points."""
    pools = [
        lambda: rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJ0123456789(){}[]:;.,=+-*/'\"#_"),
        lambda: rng.choice([" ", "  ", "    ", "\t", "\n", "\r\n", "\r", "\x0b", "\x0c"]),
        lambda: chr(rng.randint(0x00A0, 0x024F)),    # Latin-1 and Latin Extended
        lambda: chr(rng.randint(0x0400, 0x04FF)),    # Cyrillic
        lambda: chr(rng.randint(0x4E00, 0x9FFF)),    # CJK
        lambda: chr(rng.randint(0x1F300, 0x1FAFF)),  # emoji
        lambda: chr(rng.randint(0x0300, 0x036F)),    # combining marks
        lambda: chr(rng.choice([rng.randint(0x0, 0xD7FF), rng.randint(0xE000, 0x10FFFF)])),  # anything
    ]
    return "".join(rng.choice(pools)() for _ in range(rng.randint(0, 120)))


def test_tokenizer_roundtrip(tiny_codec):
    """decode(encode(x)) == x for 100% of random text and real-looking code."""
    rng = random.Random(1234)
    samples = HARD_CASES + [_random_text(rng) for _ in range(2000)]
    failures = [s for s in samples if tiny_codec.decode(tiny_codec.encode(s)) != s]
    assert not failures, f"{len(failures)}/{len(samples)} failed, first: {failures[0]!r}"
    batch = tiny_codec.encode_batch(samples, num_threads=2)
    assert batch == [tiny_codec.encode(s) for s in samples]


@pytest.mark.xfail(strict=True, reason=(
    "Known SentencePiece limitation, reported to the owner in M0: the input character U+2581 '▁' "
    "is SentencePiece's own space marker, so it decodes as ' '. escape_whitespaces=false would fix "
    "it but the Unigram trainer in sentencepiece 0.2.3 rejects that. Owner decision needed before M1."))
def test_u2581_roundtrip(tiny_codec):
    """A literal U+2581 in the input does not survive encode/decode (documents the limitation)."""
    text = "pieces like '▁the' in tokenizer code"
    assert tiny_codec.decode(tiny_codec.encode(text)) == text


def test_newline_bytes(tiny_codec):
    """A newline is the byte token <0x0A>; tabs, CRLF and trailing spaces round-trip."""
    nl = tiny_codec.sp.piece_to_id("<0x0A>")
    assert tiny_codec.is_byte(nl)
    assert tiny_codec.encode("\n") == [nl]
    assert nl in tiny_codec.encode("a = 1\nb = 2")
    assert tiny_codec.encode("\n\n") == [nl, nl]
    for text in ["a\tb", "\t\tx", "line\r\nnext\r\n", "trailing   \nspaces \t \n", "\r\n\r\n"]:
        assert tiny_codec.decode(tiny_codec.encode(text)) == text


def test_indentation(tiny_codec):
    """Newline plus 4, 8 or 12 spaces encodes to at most 2 tokens."""
    for n in (4, 8, 12):
        ids = tiny_codec.encode("\n" + " " * n)
        pieces = [tiny_codec.piece(i) for i in ids]
        assert len(ids) <= 2, f"newline + {n} spaces -> {pieces}"


def test_vocab_ids(tiny_codec, tokenizer_cfg):
    """Every id fits uint16; EOS is id 2; each <spNN> is exactly one id (4 + NN)."""
    assert tokenizer_cfg["sentencepiece"]["vocab_size"] == 50000 < 65536
    assert tiny_codec.vocab_size < 65536
    assert tiny_codec.eos_id == 2 and tiny_codec.piece(2) == "</s>"
    assert [tiny_codec.piece(i) for i in range(4)] == ["<unk>", "<s>", "</s>", "<pad>"]
    for slot in range(NUM_SLOTS):
        assert tiny_codec.encode(slot_piece(slot)) == [4 + slot]
    sample = tiny_codec.encode("def f(x):\n    return x  # 😀 oʻzbek")
    assert all(0 <= i < tiny_codec.vocab_size for i in sample)
