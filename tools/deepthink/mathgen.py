"""Seeded math problem generators with a safe sympy answer checker.

Seeds >= HOLDOUT_START are held out and must never be used for training.
Statements use plain ASCII math.
"""
import hashlib
import itertools
import math
import random
import re
from fractions import Fraction

import sympy
from sympy.parsing.sympy_parser import (convert_xor, parse_expr,
                                        standard_transformations)

from tools.deepthink.problem import Problem, register

FAMILIES = ["algebra", "number_theory", "counting", "probability",
            "calculus", "linear_algebra"]
HOLDOUT_START = 900000

_ALLOWED_NAMES = {"x", "y", "n", "pi", "E", "e", "sqrt", "sin", "cos", "exp",
                  "log", "ln", "Rational", "oo"}
_LOCALS = {"x": sympy.Symbol("x"), "y": sympy.Symbol("y"), "n": sympy.Symbol("n"),
           "pi": sympy.pi, "E": sympy.E, "e": sympy.E, "sqrt": sympy.sqrt,
           "sin": sympy.sin, "cos": sympy.cos, "exp": sympy.exp,
           "log": sympy.log, "ln": sympy.log, "Rational": sympy.Rational}
_CHARS = re.compile(r"^[0-9A-Za-z_+\-*/^().,\s]*$")
_TRANSFORMS = standard_transformations + (convert_xor,)


def is_holdout(problem):
    return int(problem.id.rsplit("-", 1)[1]) >= HOLDOUT_START


def train_seeds(stop):
    if stop > HOLDOUT_START:
        raise ValueError("train seeds must stay below %d" % HOLDOUT_START)
    return range(stop)


def _parse(text):
    text = text.strip()
    if not text or len(text) > 200 or not _CHARS.match(text):
        raise ValueError("bad characters")
    if "__" in text:
        raise ValueError("bad name")
    for name in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", text):
        if name not in _ALLOWED_NAMES:
            raise ValueError("name not allowed: " + name)
    glob = {"Integer": sympy.Integer, "Float": sympy.Float,
            "Rational": sympy.Rational, "Symbol": sympy.Symbol}
    return parse_expr(text, local_dict=dict(_LOCALS), global_dict=glob,
                      transformations=_TRANSFORMS, evaluate=True)


def _sympy_verify(problem, answer, workdir):
    try:
        truth = _parse(str(problem.spec["answer"]))
        got = _parse(answer)
        diff = sympy.simplify(truth - got)
    except Exception as exc:  # any parse or math failure means "not correct"
        return False, "unparseable answer: %s" % type(exc).__name__
    ok = diff == 0
    return ok, "" if ok else "answer differs"


register("sympy", _sympy_verify)


def _mk(family, seed, statement, answer, steps):
    return Problem(id="math-%s-%d" % (family, seed), domain="math",
                   statement=statement, verifier="sympy",
                   spec={"answer": str(answer), "steps": steps},
                   source="mathgen", license="own")


def _frac(fr):
    fr = Fraction(fr)
    return str(fr.numerator) if fr.denominator == 1 else "%d/%d" % (fr.numerator, fr.denominator)


def _algebra(rng, seed):
    if seed % 2 == 0:
        p, q = rng.randint(3, 19), rng.randint(2, 40)
        ans = p ** 3 - 3 * p * q
        st = ("Let r and s be the roots of x^2 - %dx + %d = 0. "
              "Find r^3 + s^3." % (p, q))
        return _mk("algebra", seed, st, ans, 4)
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    s, d = rng.randint(5, 30), rng.randint(1, 4)
    # x + y = s, a*x - b*y = t with integer solution (x0, y0)
    x0 = rng.randint(2, 20)
    y0 = s - x0
    t = a * x0 - b * y0
    st = ("Real numbers x and y satisfy x + y = %d and %dx - %dy = %d. "
          "Find x*y." % (s, a, b, t))
    return _mk("algebra", seed, st, x0 * y0, 4)


def _number_theory(rng, seed):
    if seed % 2 == 0:
        a, b, m = rng.randint(3, 40), rng.randint(40, 900), rng.randint(11, 400)
        if math.gcd(a, m) == 1 and pow(a, b, m) == 0:
            m += 1
        ans = pow(a, b, m)
        if ans == 0:
            m += 1
            ans = pow(a, b, m)
        st = "Find the remainder when %d^%d is divided by %d." % (a, b, m)
        return _mk("number_theory", seed, st, ans, 5)
    n = rng.randint(200, 20000)
    divs = sum(1 for k in range(1, n + 1) if n % k == 0)
    st = ("How many positive divisors does %d have, and what is the sum of "
          "that count and the number of its odd divisors?" % n)
    odd = sum(1 for k in range(1, n + 1, 2) if n % k == 0)
    return _mk("number_theory", seed, st, divs + odd, 4)


def _counting(rng, seed):
    if seed % 2 == 0:
        w, h = rng.randint(4, 8), rng.randint(4, 8)
        bx, by = rng.randint(1, w - 1), rng.randint(1, h - 1)
        total = math.comb(w + h, w)
        through = math.comb(bx + by, bx) * math.comb(w - bx + h - by, w - bx)
        ans = total - through
        st = ("A robot moves on a grid from (0,0) to (%d,%d), one step right "
              "or one step up each time. How many paths avoid the point "
              "(%d,%d)?" % (w, h, bx, by))
        return _mk("counting", seed, st, ans, 5)
    words = ["BANANAS", "BALLOON", "PEPPER", "ASSESS", "LETTER", "COFFEE",
             "MISSISSIPPI", "STATISTICS", "COMMITTEE", "TOMATOES",
             "BOOKKEEPER", "ENGINEERING", "PROGRAMMING", "MATHEMATICS"]
    word = rng.choice(words)
    letters = sorted(set(word))
    first, last = rng.sample(letters, 2)
    ans = arrangements(word, first, last)
    if ans == 0:
        raise ValueError("retry")
    st = ("How many distinct arrangements of the letters of %s start with "
          "the letter %s and end with the letter %s?" % (word, first, last))
    return _mk("counting", seed, st, ans, 5)


def arrangements(word, first, last):
    """Distinct arrangements of the letters of word starting with first and
    ending with last (multiset permutations)."""
    from collections import Counter
    c = Counter(word)
    if first == last:
        if c[first] < 2:
            return 0
    c[first] -= 1
    c[last] -= 1
    if c[first] < 0 or c[last] < 0:
        return 0
    n = sum(c.values())
    out = math.factorial(n)
    for v in c.values():
        out //= math.factorial(v)
    return out


def _probability(rng, seed):
    sides = rng.choice([6, 8, 10])
    k = rng.randint(sides + 2, 2 * sides - 1)
    cond = rng.choice(["even", "odd"])
    pairs = [(i, j) for i in range(1, sides + 1) for j in range(1, sides + 1)]
    sel = [p for p in pairs if (p[0] % 2 == 0) == (cond == "even")]
    good = [p for p in sel if p[0] + p[1] >= k]
    if not good:
        k -= 1
        good = [p for p in sel if p[0] + p[1] >= k]
    ans = Fraction(len(good), len(sel))
    st = ("Two fair %d-sided dice (faces 1 to %d) are rolled. Given that the "
          "first die shows an %s number, what is the probability that the "
          "sum is at least %d? Answer as a fraction in lowest terms."
          % (sides, sides, cond, k))
    return _mk("probability", seed, st, _frac(ans), 5)


def _calculus(rng, seed):
    x = sympy.Symbol("x")
    if seed % 2 == 0:
        a, b, c = rng.randint(1, 6), rng.randint(1, 4), rng.randint(1, 3)
        f = (x ** 2 + a) * sympy.exp(b * x)
        val = sympy.simplify(sympy.diff(f, x).subs(x, c))
        st = ("Let f(x) = (x^2 + %d) * e^(%d*x). Find f'(%d). Write e as E "
              "in your answer." % (a, b, c))
        return _mk("calculus", seed, st, val, 4)
    a, b, c = rng.randint(1, 5), rng.randint(1, 6), rng.randint(2, 5)
    f = a * x ** 3 + b * x
    val = sympy.integrate(f, (x, 0, c))
    st = ("Find the integral of %d*x^3 + %d*x from x = 0 to x = %d." % (a, b, c))
    return _mk("calculus", seed, st, val, 3)


def _linear_algebra(rng, seed):
    while True:
        rows = [[rng.randint(-4, 5) for _ in range(3)] for _ in range(3)]
        m = sympy.Matrix(rows)
        if m.det() != 0:
            break
    if seed % 2 == 0:
        ans = m.det()
        st = "Find the determinant of the matrix %s." % rows
        return _mk("linear_algebra", seed, st, ans, 4)
    i, j = rng.randint(0, 2), rng.randint(0, 2)
    inv = m.inv()
    ans = inv[i, j]
    if ans == 0:
        ans = m.det() + 1
        st = "Find the determinant plus 1 of the matrix %s." % rows
        return _mk("linear_algebra", seed, st, ans, 4)
    st = ("Find the entry in row %d, column %d (counting from 1) of the "
          "inverse of the matrix %s. Answer as a fraction in lowest terms."
          % (i + 1, j + 1, rows))
    return _mk("linear_algebra", seed, st, ans, 6)


_GEN = {"algebra": _algebra, "number_theory": _number_theory,
        "counting": _counting, "probability": _probability,
        "calculus": _calculus, "linear_algebra": _linear_algebra}


def _split_of(statement):
    """0 = train side, 1 = holdout side; decided by the statement itself."""
    return int(hashlib.md5(statement.encode("utf-8")).hexdigest(), 16) % 2


def generate(family, seed):
    """Train seeds only produce statements on the train side of the hash
    split and held-out seeds only the other side, so the two sets can never
    share a statement, whatever the parameter ranges are."""
    if family not in _GEN:
        raise ValueError("unknown family: %r" % (family,))
    rng = random.Random("%s:%d" % (family, seed))
    want = 1 if seed >= HOLDOUT_START else 0
    for _ in range(400):
        try:
            p = _GEN[family](rng, seed)
        except ValueError:
            continue
        if str(p.spec["answer"]).strip() in ("0", "0.0"):
            continue
        if _split_of(p.statement) == want:
            return p
    raise RuntimeError("could not make a valid problem")
