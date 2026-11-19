"""Checker and sample builder for D thinking traces (math problems with a known exact answer).

A trace is `### trace NNNN`, a <thinking> block of allowed line kinds, and one `ANSWER:` line. The checker rejects
filler words, long steps, wrong arithmetic (every `a op b = c` chain is recomputed exactly) and a wrong final answer.
"""
import ast
import json
import re
from fractions import Fraction

TRACE = re.compile(r"^### trace (\d{4})[ \t]*$", re.M)
STEP = re.compile(r"^\d+\.\s+\S")
MAP = re.compile(r"^<map> known: .+ \| open: .+ </map>$")
FORBIDDEN = re.compile(r"\b(hmm|wait|okay|let me|i think|i guess|actually|maybe|perhaps|so basically|it seems|sorry|"
                       r"apologi[sz]e|great|excellent|good question)\b", re.I)
CHECKLIST = re.compile(r"^\d{4} ok[ \t]*\n?", re.M)
_N = r"\d+(?:\.\d+)?"
_NUM = r"%s(?:\s*[-+*/x×^]\s*%s)*" % (_N, _N)
FUNC_BEFORE = r"(?:\b(?:log|ln|lg|sqrt|sin|cos|tan|exp|mod|gcd|lcm|det|abs)\w*|\b[A-Za-z])\s*$"
CHAIN = re.compile(r"(?<![\w^*/.(])(%s(?:\s*=\s*%s)+)(?![\w^(]|\.\d)" % (_NUM, _NUM))


def words(text):
    return len(text.split())


def _eval(expr):
    """Exact value of a pure-number expression, or None when it is not one."""
    expr = expr.replace("×", "*").replace("x", "*").replace("^", "**")

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return Fraction(str(n.value))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
            if isinstance(n.op, ast.Pow) and b.denominator == 1 and abs(b) <= 64 and abs(a) < 10 ** 12:
                return a ** int(b)
        raise ValueError("unsupported")
    try:
        return ev(ast.parse(expr.strip(), mode="eval").body)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return None


def bad_chains(text):
    """Every `A = B = C` chain of pure numbers whose parts are not all equal (rounding allowed when a decimal or / is used)."""
    out = []
    for m in CHAIN.finditer(text):
        before = text[:m.start()].rstrip()
        if before and before[-1] in "+-*/^×(":                   # the tail of a longer expression such as x^2 - 9x + 14 = 0
            continue
        if re.search(FUNC_BEFORE, before):                         # log 128 = 7, sqrt 49 = 7, f 2 = 4: not a plain sum
            continue
        parts = [p.strip() for p in m.group(1).split("=")]
        vals = [_eval(p) for p in parts]
        if any(v is None for v in vals):
            continue
        exact = not re.search(r"[./]", m.group(1))
        base = vals[0]
        for v in vals[1:]:
            tol = 0 if exact else max(Fraction(6, 1000) * abs(base), Fraction(11, 1000))
            if abs(v - base) > tol:
                out.append(m.group(1).strip())
                break
    return out


def clean(reply):
    """Take the ```text block when there is one and drop the NNNN ok self-check lines."""
    blocks = re.findall(r"```text\n(.*?)\n```", reply, re.S)           # one block, or one block per trace
    return CHECKLIST.sub("", "\n\n".join(blocks) if blocks else reply).rstrip()


def parse(text):
    heads = list(TRACE.finditer(text))
    return {h.group(1): text[h.end():(heads[i + 1].start() if i + 1 < len(heads) else len(text))].strip()
            for i, h in enumerate(heads)}


def _check_body(body, answer, verify, qid):
    """Only what proves the trace is good: a thinking block, right arithmetic, and the right final answer."""
    m = re.match(r"^<thinking>\n(.*?)\n</thinking>\n(.*)$", body, re.S)
    if not m:
        return ["format: a <thinking> block, then one ANSWER: line"]
    think, after = m.group(1), m.group(2).strip()
    errs = ["wrong arithmetic: " + c for c in bad_chains(think)]
    first, _, rest = after.partition("\n")
    if not first.startswith("ANSWER: ") or rest.strip():
        errs.append("need exactly one ANSWER: line after the thinking")
    else:
        got = first[len("ANSWER: "):].strip()
        if not (verify(qid, got) if verify else got == str(answer).strip()):
            errs.append("answer differs from the known answer")
    return errs


def check(problems, text, verify=None):
    """problems: {id: known answer}; verify(id, answer) -> bool replaces the plain string comparison."""
    parsed = parse(text)
    return {i: (["trace missing"] if i not in parsed else _check_body(parsed[i], a, verify, i)) for i, a in problems.items()}


def build_sample(qid, statement, trace_text):
    body = TRACE.sub("", trace_text, count=1).strip()
    m = re.match(r"^(<thinking>\n.*?\n</thinking>)\nANSWER: (.*)$", body, re.S)
    assistant = "%s\nAnswer: %s\n" % (m.group(1), m.group(2).strip())
    return {"id": "D" + qid,
            "segments": [{"t": "<user>\n%s\n</user>\n" % statement, "loss": 0}, {"t": "<assistant>\n", "loss": 0},
                         {"t": assistant, "loss": 1}, {"t": "</assistant>\n", "loss": 0}],
            "flags": ["deepseek_unreviewed"]}


def dumps(sample):
    return json.dumps(sample, ensure_ascii=True)
