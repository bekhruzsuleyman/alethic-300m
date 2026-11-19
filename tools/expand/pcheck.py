"""Checker for expansion replies of type T (chats) and E (idea evaluation pairs).

It enforces the hard rules of docs/sft/record/prompts/deepseek_TE_chats_*.md and deepseek_E_pairs_*.md
that a script can check: format, turn counts, word limits, verdict lines, pair identity, forbidden words,
tags, markdown, Uzbek apostrophes and simple arithmetic. It cannot check that facts are true.
check(kind, rows, text) -> {example id: [error, ...]}; an empty list means the example passed.
For E both examples of a pair fail together.
"""
import ast
import re

HEADER = re.compile(r"^### ex (\d{4}) \| ([TE]) \| (en|uz|mix) \| deepseek\n", re.M)
TURN = re.compile(r"<(user|assistant)>\n(.*?)\n</\1>", re.S)
FORBIDDEN = re.compile(
    r"\b(great|excellent|amazing|fantastic|brilliant|good question|good idea|you are right to|i'm glad|happy to help|"
    r"of course|sure!|certainly|sorry|apologi[sz]e|ajoyib|zo'r|a'lo|mukammal|yaxshi savol|albatta|kechirasiz|uzr)\b", re.I)
TAGS = re.compile(r"<(call|result|plan|thinking|error|warn|map|goal|todo|done)>")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
CURLY = re.compile("[‘’ʻʼ`´]")
VERDICT = {"en": {"strong": "strong", "risky": "risky", "weak": "weak", "not yet": "not yet"},
           "uz": {"strong": "kuchli", "risky": "xavfli", "weak": "zaif", "not yet": "hali emas"}}
EXPR = re.compile(r"(\d[\d.,]*(?:\s[x×*/+\-]\s\d[\d.,]*)+)\s=\s(\d[\d.,]*)")


def words(text):
    """Word count outside fenced code blocks."""
    return len(re.sub(r"```.*?```", " ", text, flags=re.S).split())


def _num(s):
    s = s.rstrip(".,")
    if re.search(r",\d{3}(\D|$)", s):
        s = s.replace(",", "")
    return float(s.replace(",", "."))


def _eval(expr):
    expr = re.sub(r"\s[x×]\s", " * ", expr)
    expr = re.sub(r"\d[\d.,]*", lambda m: repr(_num(m.group())), expr)
    node = ast.parse(expr, mode="eval")
    for n in ast.walk(node):
        if not isinstance(n, (ast.Expression, ast.BinOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UnaryOp, ast.USub)):
            raise ValueError("bad expression")
    return eval(compile(node, "<expr>", "eval"), {"__builtins__": {}})


def bad_arithmetic(text):
    """Every 'a op b = c' in the text whose result is wrong by more than rounding."""
    out = []
    for m in EXPR.finditer(text):
        try:
            got, want = _eval(m.group(1)), _num(m.group(2))
        except (ValueError, ZeroDivisionError, SyntaxError):
            continue
        if abs(got - want) > max(0.015 * abs(got), 0.06):
            out.append("%s = %s" % (m.group(1), m.group(2)))
    return out


def parse(text):
    """{id: (kind, lang, [(role, text), ...], format_errors)} for every example found, in order."""
    out = {}
    heads = list(HEADER.finditer(text))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[h.end():end].strip()
        turns = TURN.findall(body)
        errs = []
        if "\n".join("<%s>\n%s\n</%s>" % (r, t, r) for r, t in turns) != body:
            errs.append("text outside the turn tags")
        out[h.group(1)] = (h.group(2), h.group(3), turns, errs)
    return out


def _common(ex_id, kind, lang, turns, errs, max_code_lines=8):
    roles = [r for r, _ in turns]
    if not turns or roles != ["user", "assistant"] * (len(turns) // 2) or roles[-1] != "assistant":
        errs.append("turns must alternate user/assistant and end with assistant")
    for role, t in turns:
        if TAGS.search(t):
            errs.append("tool tag in text")
        if EMOJI.search(t):
            errs.append("emoji")
        if re.search(r"^(#|\s*[-*] |\|)|\*\*", t, re.M) and kind == "E" or re.search(r"^#|\*\*|^\|", t, re.M):
            errs.append("markdown")
        if lang in ("uz", "mix") and CURLY.search(t):
            errs.append("non-ASCII apostrophe")
        if role == "assistant":
            plain = re.sub(r"```.*?```", " ", t, flags=re.S)
            if FORBIDDEN.search(plain):
                errs.append("forbidden word: %s" % FORBIDDEN.search(plain).group(0))
            if "?" in plain and kind == "E":
                errs.append("question mark in assistant turn")
            for b in bad_arithmetic(plain):
                errs.append("wrong arithmetic: " + b)
            if kind == "E" and "```" in t:
                errs.append("code block in E")
            for blk in re.findall(r"```.*?\n(.*?)```", t, flags=re.S):
                if len(blk.strip().splitlines()) > max_code_lines:
                    errs.append("code block over %d lines" % max_code_lines)


def _t_row(row):
    i, lang, turns, skill, _topic = row.split(None, 4)
    return {"id": i, "lang": lang, "turns": int(turns), "skill": skill}


def _check_t(row, ex, errs, style="strict"):
    kind, lang, turns, _ = ex
    users = [t for r, t in turns if r == "user"]
    asst = [t for r, t in turns if r == "assistant"]
    if kind != "T" or lang != row["lang"]:
        errs.append("header kind/lang does not match the table")
    if style == "detailed":                                  # a natural chat may be shorter or longer than the plan
        if not 1 <= len(users) <= 3:
            errs.append("need 1 to 3 user turns, got %d" % len(users))
    elif len(users) != row["turns"]:
        errs.append("expected %d user turns, got %d" % (row["turns"], len(users)))
    limit = None if style == "detailed" else 70           # detailed: as short as possible, as long as needed
    for a in asst:
        if limit and words(a) > limit:
            errs.append("assistant turn over %d words" % limit)
    skill = row["skill"]
    if "U" in skill.split("+")[0:1] and asst:
        want = "Ishonchim komil emas." if lang == "uz" else "I am not sure."
        if not asst[0].startswith(want) and not (lang == "mix" and asst[0].startswith(("Ishonchim komil emas.", "I am not sure."))):
            errs.append('U: first answer must start with "I am not sure." / "Ishonchim komil emas."')
    if "L" in skill.split("+")[1:] and len(asst) >= 2:
        if not re.search(r"no live data|Menda jonli ma'lumot yo'q", asst[-1]):
            errs.append("L: must say it has no live data")
    if skill.startswith("N") and not any("=" in a for a in asst):
        errs.append("N: arithmetic with = missing")


def _e_rows(rows):
    out = []
    for r in rows:
        m = re.match(r"\s*(\d+)\s+(\d{4})/(\d{4})\s+(en|uz|mix)\s+.*?(strong|risky|weak|not yet)\s*->\s*(strong|risky|weak|not yet)\s*$", r)
        if not m:
            raise ValueError("cannot read E row: %r" % r)
        out.append({"pair": int(m.group(1)), "hold": m.group(2), "upd": m.group(3), "lang": m.group(4),
                    "first": m.group(5), "new": m.group(6)})
    return out


LABEL = re.compile(r"\s*(Verdict|Xulosa)\b")
CONDITION = re.compile(r"\b(if|unless|would change|different)\b|agar|bo'lsa|o'zgar", re.I)


def _hits(text, lang):
    """{verdict key: [positions]} of the verdict words found in text."""
    out = {}
    for key, w in VERDICT["uz" if lang in ("uz", "mix") else "en"].items():
        ps = [m.start() for m in re.finditer(r"(?<![\w'])%s(?![\w'])" % re.escape(w), text, re.I)]
        if ps:
            out[key] = ps
    return out


def _opening(text):
    return " ".join(re.findall(r"\w+", text.lower())[:3])


def _check_e_pair(row, hold, upd, errs_h, errs_u, style="strict", openings=None):
    """style "strict": fixed label lines (the old format). "natural": free wording; only the verdict word, the worked
    number with =, a condition that would change the answer, and a different opening for every pair are checked."""
    both = []
    lang = row["lang"]
    v = VERDICT["uz" if lang in ("uz", "mix") else "en"]
    pre = ("Xulosa", "Xulosa o'zgarmadi", "Xulosa o'zgardi") if lang in ("uz", "mix") else ("Verdict", "Verdict unchanged", "Verdict changed")
    ht, ut = hold[2], upd[2]
    if len(ht) != 4 or len(ut) != 4:
        both.append("E examples need exactly 2 user and 2 assistant turns")
        return both
    if ht[0] != ut[0] or ht[1] != ut[1]:
        both.append("first two turns of the pair are not identical")
    u1, a1 = ht[0][1], ht[1][1]
    if len(u1.split()) > 60 or len(re.findall(r"\d+", u1)) < 2:
        both.append("first user turn: over 60 words or fewer than 2 numbers")
    first_line = a1.split("\n")[0]
    if style == "strict" and first_line != "%s: %s." % (pre[0], v[row["first"]]):
        both.append("first verdict line is not exactly '%s: %s.'" % (pre[0], v[row["first"]]))
    if style == "natural":
        hits = _hits(a1, lang)
        if LABEL.match(a1):
            both.append("do not start with a fixed label (Verdict: / Xulosa:); say it in your own words")
        if set(hits) != {row["first"]}:
            both.append("first answer must use the verdict word '%s' and no other verdict word" % v[row["first"]])
        if not CONDITION.search(a1):
            both.append("first answer must say what would change the verdict")
        key = _opening(a1)
        if openings is not None:
            if key in openings:
                both.append("opening words repeat an earlier pair of this reply")
            openings.add(key)
    if style != "natural" and words(a1) > 80:
        both.append("first assistant turn over 80 words")
    if "=" not in a1:
        both.append("first assistant turn has no worked number with =")
    last = re.split(r"(?<=[.!])\s+", a1.strip())[-1]
    if style == "strict" and not re.search(r"change|o'zgar", last, re.I):
        both.append("last sentence must say which fact would change the verdict")
    # hold
    h_user, h_asst = ht[2][1], ht[3][1]
    if len(h_user.split()) > 30:
        errs_h.append("hold pushback over 30 words")
    if style == "strict" and h_asst.split("\n")[0] != "%s: %s." % (pre[1], v[row["first"]]):
        errs_h.append("hold verdict line is not exactly '%s: %s.'" % (pre[1], v[row["first"]]))
    if style == "natural":
        if LABEL.match(h_asst) or set(_hits(h_asst, lang)) != {row["first"]}:
            errs_h.append("hold answer: own words, and only the verdict word '%s'" % v[row["first"]])
        if not CONDITION.search(h_asst):
            errs_h.append("hold answer must say what would change the verdict")
    if style != "natural" and words(h_asst) > 50:
        errs_h.append("hold answer over 50 words")
    # update
    u_user, u_asst = ut[2][1], ut[3][1]
    if len(u_user.split()) > 40 or not re.search(r"\d", u_user):
        errs_u.append("update fact over 40 words or without a number")
    if style == "strict" and u_asst.split("\n")[0] != "%s: %s." % (pre[2], v[row["new"]]):
        errs_u.append("update verdict line is not exactly '%s: %s.'" % (pre[2], v[row["new"]]))
    if style == "natural":
        uh = _hits(u_asst, lang)
        if LABEL.match(u_asst) or row["new"] not in uh or not set(uh) <= {row["first"], row["new"]} or \
                (row["first"] in uh and max(uh[row["first"]]) > max(uh[row["new"]])):
            errs_u.append("update answer: own words, ends on the new verdict word '%s'" % v[row["new"]])
    if style != "natural" and words(u_asst) > 60:
        errs_u.append("update answer over 60 words")
    if "=" not in u_asst:
        errs_u.append("update answer has no worked number with =")
    return both


def check(kind, rows, text, style="strict"):
    """Check one reply against its table rows; returns {id: [errors]} for every expected id."""
    parsed = parse(text)
    res = {}
    if kind == "T":
        for row in map(_t_row, rows):
            errs = []
            ex = parsed.get(row["id"])
            if ex is None:
                errs.append("example missing")
            else:
                errs += ex[3]
                _common(row["id"], "T", ex[1], ex[2], errs, 12 if style == "detailed" else 8)
                _check_t(row, ex, errs, style)
            res[row["id"]] = errs
        return res
    openings = set()
    for row in _e_rows(rows):
        eh, eu = parsed.get(row["hold"]), parsed.get(row["upd"])
        errs_h, errs_u = [], []
        if eh is None or eu is None:
            for i, e, es in ((row["hold"], eh, errs_h), (row["upd"], eu, errs_u)):
                if e is None:
                    es.append("example missing")
        else:
            for ex, es in ((eh, errs_h), (eu, errs_u)):
                es += ex[3]
                _common(None, "E", ex[1], ex[2], es)
                if ex[0] != "E" or ex[1] != row["lang"]:
                    es.append("header kind/lang does not match the table")
            both = _check_e_pair(row, eh, eu, errs_h, errs_u, style, openings)
            errs_h += both
            errs_u += both
        # a pair is kept or dropped as a whole
        if errs_h or errs_u:
            errs_h = errs_h or ["pair partner failed"]
            errs_u = errs_u or ["pair partner failed"]
        res[row["hold"]], res[row["upd"]] = errs_h, errs_u
    return res
