"""Uzbek apostrophe cleaning."""
from __future__ import annotations

from alethic.data.uzbek import APOSTROPHES, normalize_uzbek

OKINA = "ʻ"   # ʻ
APOS = "ʼ"    # ʼ


def test_uzbek_apostrophe():
    assert len(APOSTROPHES) == 6
    for v in APOSTROPHES:
        # After o, O, g, G -> U+02BB (also word-final, as in togʻ).
        assert normalize_uzbek(f"o{v}zbek")[0] == f"o{OKINA}zbek"
        assert normalize_uzbek(f"O{v}zbekiston")[0] == f"O{OKINA}zbekiston"
        assert normalize_uzbek(f"g{v}alaba")[0] == f"g{OKINA}alaba"
        assert normalize_uzbek(f"G{v}ARB")[0] == f"G{OKINA}ARB"
        assert normalize_uzbek(f"tog{v} bog{v}")[0] == f"tog{OKINA} bog{OKINA}"
        # Between other letters -> U+02BC (Latin and Cyrillic).
        assert normalize_uzbek(f"ma{v}no ta{v}lim")[0] == f"ma{APOS}no ta{APOS}lim"
        assert normalize_uzbek(f"мас{v}ул")[0] == f"мас{APOS}ул"
        # Not after o/g and not between two letters -> unchanged.
        for text in (f"{v}salom{v}", f"a{v} b", f"1{v}2", f"x {v}y", f"{v}"):
            assert normalize_uzbek(text) == (text, 0)
    # Counts: only characters that actually changed.
    assert normalize_uzbek("o'zbek ma'no g‘alaba") == (f"o{OKINA}zbek ma{APOS}no g{OKINA}alaba", 3)
    assert normalize_uzbek(f"ma{APOS}no o{OKINA}zbek") == (f"ma{APOS}no o{OKINA}zbek", 0)
    # Text without apostrophes is unchanged.
    for text in ("salom dunyo", "Ўзбекистон", "", "def f(x):\n    return x"):
        assert normalize_uzbek(text) == (text, 0)
    # Deterministic.
    s = "O'zbekiston g'alaba ma`no tog´ bog’"
    assert normalize_uzbek(s) == normalize_uzbek(s)
