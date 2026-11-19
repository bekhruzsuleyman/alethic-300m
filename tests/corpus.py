"""A small deterministic training corpus for the tiny test tokenizer (code, prose, Uzbek, math).

One source line per line, like the real tokenizer sample. Python code uses 4/8/12-space
indentation often, so the real SentencePiece settings can learn whitespace-run pieces.
"""
from __future__ import annotations

import random
from pathlib import Path

SYLLABLES = ["ka", "lo", "mi", "ta", "ne", "ro", "si", "du", "pe", "va", "zo", "gu", "bi", "fa", "xo"]
ENGLISH = ("the of and to in is that for it as with was on be by this are from at or an have not "
           "data model value result function return number list string learning system school").split()
UZBEK = ("oʻzbek tili gʻalaba maktab kitob oʻqituvchi bilan uchun yaxshi katta shahar togʻ bogʻ "
         "maʼno taʼlim davlat xalq yangi ish yil kun oʻrta gʻarb").split()
CYRILLIC = "ўзбек тили мактаб китоб билан учун яхши катта шаҳар давлат халқ янги иш йил кун".split()
MATH = [r"\frac{a}{b}", r"x^2 + y^2 = z^2", r"\sum_{i=1}^{n} i", r"\int_0^1 f(x)\,dx", r"\alpha + \beta",
        r"\sqrt{2}", r"e^{i\pi} + 1 = 0", r"\lim_{n \to \infty}"]


def _ident(rng: random.Random) -> str:
    """A random identifier like 'kalomi_ta'."""
    word = "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(1, 3)))
    return word if rng.random() < 0.7 else word + "_" + rng.choice(SYLLABLES)


def _python_block(rng: random.Random) -> list[str]:
    """One function or class with nested indentation."""
    name, arg, var = _ident(rng), _ident(rng), _ident(rng)
    lines = [f"def {name}({arg}, {var}=None):",
             f'    """Return the {rng.choice(ENGLISH)} of {arg}."""',
             f"    if {arg} is None:",
             f"        return {rng.randint(0, 99)}",
             f"    for {var} in range(len({arg})):",
             f"        if {var} % {rng.randint(2, 9)} == 0:",
             f"            {name}_total += {arg}[{var}]",
             f"            print('{rng.choice(ENGLISH)}', {var})",
             "        else:",
             f"            {var} = {var} + {rng.randint(1, 9)}",
             f"    return {name}_total", ""]
    if rng.random() < 0.3:
        lines += [f"class {name.title()}:", f"    def __init__(self, {arg}):",
                  f"        self.{arg} = {arg}", f"        if {arg}:",
                  f"            self.{var} = [{arg}]", ""]
    if rng.random() < 0.1:   # some tab-indented code
        lines += [f"def {name}_tab():", f"\treturn {rng.randint(0, 9)}", ""]
    return lines


def make_corpus(path: Path, seed: int = 0, blocks: int = 1500) -> Path:
    """Write the corpus to path and return it."""
    rng = random.Random(seed)
    lines: list[str] = []
    for _ in range(blocks):
        lines += _python_block(rng)
        lines.append(" ".join(rng.choice(ENGLISH) for _ in range(rng.randint(5, 20))).capitalize() + ".")
        lines.append(" ".join(rng.choice(UZBEK) for _ in range(rng.randint(4, 12))).capitalize() + ".")
        lines.append(" ".join(rng.choice(CYRILLIC) for _ in range(rng.randint(4, 12))) + ".")
        lines.append("Let $" + " ".join(rng.choice(MATH) for _ in range(rng.randint(1, 4))) + "$.")
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
