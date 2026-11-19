"""SFT traces that teach three tools the other datasets never used: `<map>` (the outline the model writes after reading a
.py file), edit (right line numbers, one line / a block / an insert / a delete) and search(content, source).

Why: SFT-3b edited the wrong lines (44% of its edit calls started at line 1) and no example in the built data used search.
Here every call is run by the real Workspace. The map is the real ast outline of what was read so far. Searches answer
from the real result: Wikipedia answers recorded by tools.skilltraces.record_wiki when that cache exists (nothing in the
facts is written by hand then), else a small hand-written set. Empty result -> the model says it found nothing.

Records: {"id", "kind", "lang", "segments": [{"t", "loss"}]}, like the other built files.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

from tools.errtraces.build import Builder as _Builder, _src
from tools.evalset.prompts import eval_set
from tools.sftbuild.mapper import ast_map
from tools.skilltraces.facts import WIKI
from tools.skilltraces.queries import all_queries

LAST_SETUP: dict = {}


class Builder(_Builder):
    """Same recorder; remembers the starting files so the example can be replayed and checked (see replay.py)."""

    def __init__(self, user, setup=None):
        super().__init__(user, setup)
        LAST_SETUP.clear()
        LAST_SETUP.update(setup or {})


EVAL_PROMPTS = {p["prompt"] for p in eval_set()}
KINDS = ["map_edit", "map_edit_big", "edit_insert", "edit_block", "edit_delete", "search_hit", "search_none", "ask_search", "ask_run", "ask_call"]
DEFAULT_COUNTS = {"map_edit": 500, "map_edit_big": 250, "edit_insert": 150, "edit_block": 150, "edit_delete": 100, "search_hit": 400, "search_none": 250, "ask_search": 200, "ask_run": 200, "ask_call": 150}
CACHE = os.path.join("docs", "sft", "record", "wiki_search_cache.json")

WORDS = ["area", "price", "tax", "shift", "scale", "bonus", "limit", "total", "grade", "score", "speed", "fare", "tip", "wage", "bmi", "cost", "gain", "ratio",
         "boost", "refund", "margin", "fee", "pace", "weight", "volume", "angle", "level", "fuel", "rent", "toll", "penalty", "reward", "stock", "rate_of", "count"]
SUFFIX = ["", "", "_net", "_final", "_for", "_adj", "_total", "_x"]
VARS = ["rate", "factor", "step", "bonus", "limit", "offset", "margin", "base", "gain", "extra", "cap", "unit", "share"]
FILES = ["main.py", "calc.py", "report.py", "pricing.py", "utils.py", "shop.py", "stats.py", "tools.py", "game.py", "billing.py", "helpers.py", "model.py"]
HEAD = ["# helpers for the report", "# pricing rules", "# small utilities", "# do not rename the functions", "# numbers used by the shop", "# v2 of the calculator", ""]
DOC = ["Compute the result.", "Used by the report.", "Returns the adjusted value.", "Keep in sync with the docs.", "Helper."]


def _fn_names(rng, n):
    out = []
    while len(out) < n:
        name = rng.choice(WORDS) + rng.choice(SUFFIX)
        if name not in out:
            out.append(name)
    return out


def _make_func(rng, fn, with_doc=None, debug=False):
    """-> dict(lines, var, k, call_arg, edit_idx) : edit_idx is the index (in lines) of the `var = k` line."""
    t = rng.randrange(6)
    v, k = rng.choice(VARS), rng.randint(2, 40)
    sig, arg = ("x", rng.randint(1, 30))
    body = ["    %s = %d" % (v, k)]
    if t == 0:
        body.append("    return x * %s" % v)
    elif t == 1:
        body.append("    return x + %s" % v)
    elif t == 2:
        body.append("    return x - %s" % v)
    elif t == 3:
        sig, arg = "n", rng.randint(2, 9)
        body += ["    total = 0", "    for i in range(n):", "        total += %s" % v, "    return total"]
    elif t == 4:
        body += ["    if x > %s:" % v, '        return "high"', '    return "low"']
    else:
        body.append("    return max(x, %s)" % v)
    if debug:
        body.insert(1, '    print("debug", %s)' % sig)
    doc = []
    if with_doc if with_doc is not None else rng.random() < 0.3:
        doc = ['    """%s"""' % rng.choice(DOC)]
    lines = ["def %s(%s):" % (fn, sig)] + doc + body
    return {"lines": lines, "var": v, "k": k, "arg": arg, "edit_idx": 1 + len(doc), "tmpl": t, "sig": sig}


def _module(rng, n_funcs, target, debug_target=False, with_doc_target=None):
    names = _fn_names(rng, n_funcs)
    fn_t = names[target]
    out, funcs, starts = [], {}, {}
    out.append(rng.choice(HEAD))
    for i, n in enumerate(names):
        f = _make_func(rng, n, with_doc=with_doc_target if i == target else None, debug=debug_target and i == target)
        out += [""] * rng.choice([1, 1, 2]) if out and out[-1] != "" or True else []
        starts[n] = len(out) + 1
        funcs[n] = f
        out += f["lines"]
    out.append("")
    calls = [n for n in names if n == fn_t or rng.random() < 0.3]
    rng.shuffle(calls)
    for n in calls:
        out.append('print("%s", %s(%d))' % (n, n, funcs[n]["arg"]))
    return names, fn_t, funcs, starts, out


def _run_value(out, fn):
    for line in (out or "").splitlines():
        if line.startswith(fn + " "):
            return line[len(fn) + 1:]
    return None


PLAN_READ = {"en": ["I will read the file, then change the right line.", "First I read the file and map it.", "I need the real line numbers, so I read the file first.",
                    "Let me read the file and find the function.", "Reading the file first."],
             "uz": ["Avval faylni o'qiyman, keyin kerakli qatorni o'zgartiraman.", "Avval faylni o'qib, xaritasini tuzaman.", "Haqiqiy qator raqamlari kerak, shuning uchun faylni o'qiyman.", "Faylni o'qib, funksiyani topaman."]}
PLAN_AFTER = {"en": ["The map says %s is on lines %d-%d.", "%s is on lines %d-%d, so I read just those.", "Per the map, %s covers %d-%d."],
              "uz": ["Xaritaga ko'ra %s %d-%d qatorlarda.", "%s funksiyasi %d-%d qatorlarda, faqat shularni o'qiyman."]}


def _read_map(b, f, total):
    """Read the whole file in chunks of 100 lines, writing the map after each, until the target function is closed in a map."""
    pass


def _open_file(b, f, rng, lang, fn, big):
    """read(f, 1-100) [+ read(f, 101-200)] with a real map after each read. Returns (start, end) of fn from the last map."""
    path = os.path.join(b.dir, f)
    total = len(open(path, encoding="utf-8").read().splitlines())
    b.call("read", "%s, 1-100" % f)
    mp = ast_map(path, min(100, total))
    b.say("<map> %s </map>" % mp)
    span = _span(mp, fn)
    if span is None or span[1] is None:                     # not closed yet: read the next chunk
        b.call("read", "%s, 101-200" % f)
        mp = ast_map(path, total)
        b.say("<map> %s </map>" % mp)
        span = _span(mp, fn)
    return span


def _span(mp, fn):
    m = re.search(r"(?:^|-> )%s (\d+)-(\d*)" % re.escape(fn), mp)
    return (int(m.group(1)), int(m.group(2)) if m.group(2) else None) if m else None


ASK = {"en": ["In %(f)s, set %(v)s to %(k)d in the function %(fn)s.", "Change %(v)s to %(k)d inside %(fn)s in %(f)s.", "%(f)s: %(fn)s should use %(v)s = %(k)d.",
              "open %(f)s and make %(fn)s use %(k)d for %(v)s", "fix %(fn)s in %(f)s, %(v)s must be %(k)d", "Please update %(v)s in %(fn)s (file %(f)s) to %(k)d.",
              "%(fn)s in %(f)s has the wrong %(v)s. It should be %(k)d."],
       "uz": ["%(f)s ichidagi %(fn)s funksiyasida %(v)s ni %(k)d qiling.", "%(f)s faylida %(fn)s ichida %(v)s qiymatini %(k)d ga o'zgartiring.", "%(fn)s (%(f)s) da %(v)s %(k)d bo'lishi kerak.",
              "%(f)s ni oching, %(fn)s funksiyasida %(v)s = %(k)d qiling", "%(fn)s funksiyasida %(v)s noto'g'ri, %(k)d bo'lsin (%(f)s)"]}
DONE = {"en": ["Changed line %(L)d in %(fn)s: %(v)s is now %(k)d. %(fn)s(%(arg)d) gives %(out)s.", "Done. Line %(L)d now reads %(v)s = %(k)d, and %(fn)s(%(arg)d) returns %(out)s.",
               "%(fn)s now uses %(v)s = %(k)d (line %(L)d). Running the file prints %(out)s for %(fn)s(%(arg)d).", "Edited line %(L)d. After the change %(fn)s(%(arg)d) = %(out)s.",
               "Line %(L)d is fixed: %(v)s = %(k)d. The run shows %(out)s for %(fn)s(%(arg)d)."],
        "uz": ["%(fn)s ichida %(L)d-qator o'zgardi: %(v)s endi %(k)d. %(fn)s(%(arg)d) = %(out)s.", "Tayyor. %(L)d-qator endi %(v)s = %(k)d, %(fn)s(%(arg)d) %(out)s qaytaradi.",
               "%(fn)s endi %(v)s = %(k)d ishlatadi (%(L)d-qator). Ishga tushirganda %(fn)s(%(arg)d) uchun %(out)s chiqdi."]}


def _distinct_new(rng, k):
    new = rng.randint(2, 40)
    while new == k:
        new = rng.randint(2, 40)
    return new


def map_edit(rng, lang, big=False):
    n = rng.randint(17, 22) if big else rng.randint(5, 9)
    target = rng.randrange(n)
    names, fn, funcs, starts, lines = _module(rng, n, target)
    f = rng.choice(FILES)
    total = len(lines)
    if big and total <= 100:
        return map_edit(rng, lang, big)
    tf = funcs[fn]
    new = _distinct_new(rng, tf["k"])
    ask = rng.choice(ASK[lang]) % {"f": f, "v": tf["var"], "k": new, "fn": fn}
    b = Builder(ask, {f: _src(lines)})
    b.plan(rng.choice(PLAN_READ[lang]))
    span = _open_file(b, f, rng, lang, fn, big)
    assert span and span[1], (span, fn)
    s, e = span
    b.say(rng.choice(PLAN_AFTER[lang]) % (fn, s, e), loss=1) if rng.random() < 0.5 else None
    shown = b.call("read", "%s, %d-%d" % (f, s, e))
    L = s + tf["edit_idx"]
    assert shown.split("\n")[tf["edit_idx"]].split(None, 1)[1].strip() == "%s = %d" % (tf["var"], tf["k"]), shown
    b.call("edit", "%s, %d-%d" % (f, L, L), "    %s = %d\n" % (tf["var"], new))
    out = b.call("shell", "%s %s" % (rng.choice(["python", "python3"]), f))
    val = _run_value(out, fn)
    assert val is not None, out
    b.say(rng.choice(DONE[lang]) % {"L": L, "fn": fn, "v": tf["var"], "k": new, "arg": tf["arg"], "out": val})
    return b.finish()


def map_edit_big(rng, lang):
    return map_edit(rng, lang, big=True)


INS = {"en": ["In %(f)s, make %(fn)s return 0 for a negative %(s)s.", "Add a guard at the top of %(fn)s in %(f)s: if %(s)s is negative, return 0.",
              "%(fn)s in %(f)s must not accept negative input, return 0 for it.", "guard %(fn)s against negatives (return 0), file %(f)s"],
       "uz": ["%(f)s ichidagi %(fn)s manfiy %(s)s uchun 0 qaytarsin.", "%(f)s dagi %(fn)s boshiga tekshiruv qo'shing: %(s)s manfiy bo'lsa 0 qaytarsin."]}
INS_DONE = {"en": ["Added a guard at lines %(L)d-%(L2)d of %(fn)s. %(fn)s(%(neg)d) now gives %(out)s.", "Inserted the check before line %(L)d. The run prints %(out)s for %(fn)s(%(neg)d).",
                   "%(fn)s now returns 0 for negative input (new lines %(L)d-%(L2)d); %(fn)s(%(neg)d) = %(out)s."],
            "uz": ["%(fn)s ga %(L)d-%(L2)d qatorlarda tekshiruv qo'shildi. %(fn)s(%(neg)d) = %(out)s.", "Tekshiruv %(L)d-qator oldiga qo'shildi, %(fn)s(%(neg)d) uchun %(out)s chiqdi."]}


def _fixed_template(rng, fn, template, debug=False):
    """A module whose target function has one of the simple one-return bodies (templates 0, 1, 2, 5)."""
    for _ in range(200):
        n = rng.randint(4, 8)
        target = rng.randrange(n)
        names, fn_t, funcs, starts, lines = _module(rng, n, target, debug_target=debug, with_doc_target=False)
        if funcs[fn_t]["tmpl"] in template:
            return names, fn_t, funcs, starts, lines
    raise AssertionError("no module")


def edit_insert(rng, lang):
    names, fn, funcs, starts, lines = _fixed_template(rng, None, (0, 1, 2, 5))
    tf = funcs[fn]
    f = rng.choice(FILES)
    # the call at the bottom uses a negative argument for the target so the guard is visible in the run
    neg = -rng.randint(1, 30)
    lines = [l if not l.startswith('print("%s", %s(' % (fn, fn)) else 'print("%s", %s(%d))' % (fn, fn, neg) for l in lines]
    ask = rng.choice(INS[lang]) % {"f": f, "fn": fn, "s": "x"}
    b = Builder(ask, {f: _src(lines)})
    b.plan(rng.choice(PLAN_READ[lang]))
    s, e = _open_file(b, f, rng, lang, fn, False)
    shown = b.call("read", "%s, %d-%d" % (f, s, e))
    L = s + tf["edit_idx"]
    old = "    %s = %d" % (tf["var"], tf["k"])
    b.call("edit", "%s, %d-%d" % (f, L, L), "    if x < 0:\n        return 0\n%s\n" % old)
    out = b.call("shell", "%s %s" % (rng.choice(["python", "python3"]), f))
    val = _run_value(out, fn)
    assert val == "0", (val, out)
    b.say(rng.choice(INS_DONE[lang]) % {"L": L, "L2": L + 1, "fn": fn, "neg": neg, "out": val})
    return b.finish()


BLK = {"en": ["In %(f)s, make %(fn)s add %(c)d to its result.", "%(fn)s in %(f)s should return its value plus %(c)d.", "Add %(c)d to what %(fn)s returns (file %(f)s)."],
       "uz": ["%(f)s ichida %(fn)s natijasiga %(c)d qo'shing.", "%(f)s dagi %(fn)s natijasi %(c)d ga ko'p bo'lsin."]}
BLK_DONE = {"en": ["Rewrote lines %(L)d-%(L2)d of %(fn)s so it adds %(c)d. %(fn)s(%(arg)d) = %(out)s.", "%(fn)s now adds %(c)d: lines %(L)d-%(L2)d changed, and %(fn)s(%(arg)d) gives %(out)s.",
                   "Done. The body of %(fn)s (lines %(L)d-%(L2)d) now ends with + %(c)d; the run prints %(out)s."],
            "uz": ["%(fn)s ning %(L)d-%(L2)d qatorlari qayta yozildi, %(c)d qo'shiladi. %(fn)s(%(arg)d) = %(out)s.", "Tayyor: %(fn)s endi %(c)d qo'shadi, ishga tushirganda %(out)s chiqdi."]}


def edit_block(rng, lang):
    names, fn, funcs, starts, lines = _fixed_template(rng, None, (0, 1, 2))
    tf = funcs[fn]
    f = rng.choice(FILES)
    c = rng.randint(2, 25)
    ask = rng.choice(BLK[lang]) % {"f": f, "fn": fn, "c": c}
    b = Builder(ask, {f: _src(lines)})
    b.plan(rng.choice(PLAN_READ[lang]))
    s, e = _open_file(b, f, rng, lang, fn, False)
    shown = b.call("read", "%s, %d-%d" % (f, s, e))
    L = s + tf["edit_idx"]
    ret = shown.split("\n")[tf["edit_idx"] + 1].split(None, 1)[1]
    assert ret.strip().startswith("return"), shown
    b.call("edit", "%s, %d-%d" % (f, L, L + 1), "    %s = %d\n    %s + %d\n" % (tf["var"], tf["k"], ret.strip(), c))
    out = b.call("shell", "%s %s" % (rng.choice(["python", "python3"]), f))
    val = _run_value(out, fn)
    assert val is not None, out
    b.say(rng.choice(BLK_DONE[lang]) % {"L": L, "L2": L + 1, "fn": fn, "c": c, "arg": tf["arg"], "out": val})
    return b.finish()


DEL = {"en": ["Remove the debug print from %(fn)s in %(f)s.", "%(fn)s in %(f)s prints debug lines. Delete that print.", "get rid of the debug output in %(fn)s (%(f)s)"],
       "uz": ["%(f)s ichidagi %(fn)s dan debug print ni olib tashlang.", "%(fn)s (%(f)s) debug chiqaryapti, o'sha print ni o'chiring."]}
DEL_DONE = {"en": ["Deleted line %(L)d, the debug print. The run now shows only %(fn)s %(out)s for %(fn)s.", "Removed the print on line %(L)d; the output for %(fn)s is just %(out)s now.",
                   "Line %(L)d (the debug print) is gone and %(fn)s(%(arg)d) still gives %(out)s."],
            "uz": ["%(L)d-qator (debug print) o'chirildi. %(fn)s(%(arg)d) = %(out)s.", "%(L)d-qatordagi print olib tashlandi, %(fn)s uchun %(out)s chiqdi."]}


def edit_delete(rng, lang):
    names, fn, funcs, starts, lines = _fixed_template(rng, None, (0, 1, 2, 5), debug=True)
    tf = funcs[fn]
    f = rng.choice(FILES)
    ask = rng.choice(DEL[lang]) % {"f": f, "fn": fn}
    b = Builder(ask, {f: _src(lines)})
    b.plan(rng.choice(PLAN_READ[lang]))
    s, e = _open_file(b, f, rng, lang, fn, False)
    shown = b.call("read", "%s, %d-%d" % (f, s, e))
    L = s + tf["edit_idx"] + 1
    assert "debug" in shown.split("\n")[tf["edit_idx"] + 1], shown
    b.call("edit", "%s, %d-%d" % (f, L, L), "")
    out = b.call("shell", "%s %s" % (rng.choice(["python", "python3"]), f))
    assert "debug" not in out, out
    val = _run_value(out, fn)
    assert val is not None, out
    b.say(rng.choice(DEL_DONE[lang]) % {"L": L, "fn": fn, "arg": tf["arg"], "out": val})
    return b.finish()


# --- search ------------------------------------------------------------------------------------------------------------
def _first_sentence(text, cap=190):
    text = re.sub(r"\s*\(([^()]*)\)", lambda m: m.group(0) if re.fullmatch(r"[0-9A-Za-z ,.\u2013\-/]+", m.group(1)) and any(c.isdigit() for c in m.group(1)) else "", text)
    text = re.sub(r"\s+([,.;])", r"\1", text)                       # pronunciations/translations in brackets are dropped, dates kept
    m = re.search(r"(?<=[a-z0-9\)\]\"'])\.\s", text)
    s = text[:m.start() + 1] if m and m.start() > 25 else text
    s = s if len(s) <= cap else s[:cap].rsplit(" ", 1)[0] + "..."
    return s


def search_pool(cache_path=CACHE):
    """-> (hits, nones). A hit: dict(en, uz, query, rows, fact_en, fact_uz); a none: dict(en, uz, query)."""
    hits, nones = [], []
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
        for q in all_queries():
            rows = cache.get(q["query"])
            if rows is None:
                continue
            if q["expected"] is None:
                if rows == []:
                    nones.append({"en": q["en"], "uz": q["uz"], "query": q["query"]})
                continue
            good = next((r for r in rows if r["title"].lower() == q["expected"].lower()), None)
            if good:
                hits.append({"en": q["en"], "uz": q["uz"], "query": q["query"], "rows": rows, "fact_en": _first_sentence(good["snippet"]),
                             "fact_uz": None, "title": good["title"]})
        if hits and nones:
            return hits, nones
    for i, (title, url, snip, fen, fuz, qen, quz, queries) in enumerate(WIKI):
        others = [WIKI[(i + j) % len(WIKI)] for j in (3, 11)]
        rows = [{"title": title, "url": url, "snippet": snip}] + [{"title": o[0], "url": o[1], "snippet": o[2]} for o in others]
        hits.append({"en": qen, "uz": quz, "query": queries[0], "rows": rows, "fact_en": fen, "fact_uz": fuz, "title": title})
    from tools.skilltraces.queries import invented
    for w in invented(random.Random(1), 60):
        nones.append({"en": ["What is %s?" % w, "Tell me about %s." % w, "who invented %s?" % w, "how do I use %s?" % w, "%s meaning?" % w],
                      "uz": ["%s nima?" % w, "%s haqida ayt." % w], "query": w})
    return hits, nones


def _wiki_json(rows):
    return json.dumps({"query": {"pages": {str(i): {"title": r["title"], "index": i, "extract": r["snippet"]} for i, r in enumerate(rows, 1)}}}) if rows else '{"batchcomplete": ""}'


SPLAN = {"en": ["I am not sure of this, so I will look it up.", "Let me search for it instead of guessing.", "I will check Wikipedia.", "A quick search will settle it.", "I'd better look this up."],
         "uz": ["Bunga ishonchim komil emas, shuning uchun qidirib ko'raman.", "Taxmin qilmay, qidirib ko'raman.", "Vikipediyadan tekshiraman.", "Tez qidiruv aniqlik kiritadi."]}
HIT_FINAL = {"en": ["%(fact)s", "%(fact)s (source: Wikipedia, %(title)s)", "According to Wikipedia: %(fact)s", "The search says: %(fact)s", "%(title)s: %(fact)s",
                    "Found it. %(fact)s", "%(fact)s I took that from the first matching result, %(title)s.", "From Wikipedia's %(title)s page: %(fact)s"],
             "uz": ["%(fact_uz)s", "%(fact_uz)s (manba: Vikipediya, %(title)s)", "Vikipediyaga ko'ra: %(fact_uz)s", "Qidiruv natijasi: %(fact_uz)s", "Topdim. %(fact_uz)s"]}
NONE_FINAL = {"en": ["I searched Wikipedia for \"%(q)s\" and it returned no results, so I can't tell you what it is. I'd rather not guess.", "Nothing came back for \"%(q)s\", so I can't confirm anything about it.",
                     "The search for %(q)s found no results. I don't know what it refers to, and I won't make something up.", "No results for \"%(q)s\". I'm not sure it exists; can you tell me where you saw it?",
                     "I couldn't find \"%(q)s\" anywhere in the search, so I can't answer that reliably.", "Search found nothing on %(q)s. I'd need a source for that before saying more."],
              "uz": ["Vikipediyadan \"%(q)s\" ni qidirdim, natija chiqmadi. Nima ekanini ayta olmayman, taxmin ham qilmayman.", "\"%(q)s\" bo'yicha hech narsa topilmadi, shuning uchun tasdiqlay olmayman.",
                     "%(q)s qidiruvda topilmadi. Bu nima ekanini bilmayman, to'qib chiqarmayman.", "\"%(q)s\" uchun natija yo'q. Qayerdan ko'rganingizni aytsangiz, yana qarab ko'raman."]}


def _search_run(b, query, rows):
    b.ws.search_fetch = lambda url, rows=rows: _wiki_json(rows)
    return b.call("search", "%s, wikipedia" % query)


def search_hit(rng, lang, pool=None):
    hits = (pool or search_pool())[0]
    h = rng.choice(hits)
    qs = h["uz"] if lang == "uz" and h["uz"] else h["en"]
    lang = "uz" if qs is h["uz"] else "en"
    b = Builder(rng.choice(qs))
    b.plan(rng.choice(SPLAN[lang]))
    res = _search_run(b, h["query"], h["rows"])
    assert res and res.startswith("1. "), res
    fact = h["fact_en"]
    fu = h["fact_uz"] or ("%s" % fact)
    text = rng.choice(HIT_FINAL[lang]) % {"fact": fact, "fact_l": fact[0].lower() + fact[1:], "fact_uz": fu, "title": h["title"]}
    b.say(text)
    return b.finish()


def search_none(rng, lang, pool=None):
    nones = (pool or search_pool())[1]
    h = rng.choice(nones)
    qs = h["uz"] if lang == "uz" else h["en"]
    b = Builder(rng.choice(qs))
    b.plan(rng.choice(SPLAN[lang]))
    res = _search_run(b, h["query"], [])
    assert res == "no results", res
    b.say(rng.choice(NONE_FINAL[lang]) % {"q": h["query"]})
    return b.finish()


# --- the user names the tool: "search ...", "run ...", "call ..." -> do exactly that -------------------------------------------
ASK_SEARCH = {"en": ["search %s", "search for %s on wikipedia", "look up %s", "search wikipedia: %s", "Please search %s.", "can you search %s", "search %s for me", "wikipedia %s"],
              "uz": ["%s ni qidir", "vikipediyadan %s ni qidirib ber", "%s haqida qidirib ko'r", "qidir: %s"]}
SEARCH_LEAD = {"en": ["Searching.", "I will search Wikipedia.", "Searching Wikipedia for that.", ""], "uz": ["Qidiraman.", "Vikipediyadan qidiraman.", ""]}


def ask_search(rng, lang, pool=None):
    hits, nones = pool or search_pool()
    is_hit = rng.random() < 0.7
    h = rng.choice(hits if is_hit else nones)
    q = h["query"]
    b = Builder(rng.choice(ASK_SEARCH[lang]) % q)
    lead = rng.choice(SEARCH_LEAD[lang])
    if lead:
        b.plan(lead)
    rows = h["rows"] if is_hit else []
    if rng.random() < 0.12:                                  # the user asks for a source that is not connected: the real error, then wikipedia
        b.segs[0]["t"] = "<user>\n%s\n</user>\n" % (("search %s on brave" if lang == "en" else "%s ni brave dan qidir") % q)
        r = b.call("search", "%s, brave" % q)
        assert r.startswith("error"), r
        b.plan("Brave is not connected, so I will use Wikipedia." if lang == "en" else "Brave ulanmagan, shuning uchun Vikipediyadan qidiraman.")
    res = _search_run(b, q, rows)
    if is_hit:
        fact = h["fact_en"]
        fu = h["fact_uz"] or fact
        b.say(rng.choice(HIT_FINAL[lang]) % {"fact": fact, "fact_l": fact[0].lower() + fact[1:], "fact_uz": fu, "title": h["title"]})
    else:
        assert res == "no results"
        b.say(rng.choice(NONE_FINAL[lang]) % {"q": q})
    return b.finish()


ASK_RUN = {"en": ["run %s", "run the file %s", "execute %s", "python %s", "Please run %s and tell me what it prints.", "can you run %s?", "start %s"],
           "uz": ["%s ni ishga tushir", "%s faylini ishlat", "%s ni run qil", "%s ni ishga tushirib, natijani ayt"]}
RUN_FINAL = {"en": ["It printed %(n)d lines: %(lines)s.", "Ran %(f)s. Output: %(lines)s.", "%(f)s ran fine and printed %(lines)s.", "Done. The output is %(lines)s.", "%(lines)s - that's what %(f)s prints (%(n)d lines)."],
             "uz": ["%(n)ta qator chiqdi: %(lines)s.", "%(f)s ishladi. Natija: %(lines)s.", "Tayyor. Chiqish: %(lines)s."]}


def ask_run(rng, lang):
    n = rng.randint(3, 6)
    names, fn, funcs, starts, lines = _module(rng, n, rng.randrange(n))
    f = rng.choice(FILES)
    b = Builder(rng.choice(ASK_RUN[lang]) % f, {f: _src(lines)})
    if rng.random() < 0.5:
        b.plan(rng.choice(["I will run it.", "Running it now.", "Let me run the file."] if lang == "en" else ["Ishga tushiraman.", "Hozir ishga tushiraman."]))
    out = b.call("shell", "%s %s" % (rng.choice(["python", "python3"]), f))
    got = [l for l in (out or "").splitlines() if l.strip()]
    assert got and "Error" not in out, out
    shown = "; ".join(got[:4]) + ("; ..." if len(got) > 4 else "")
    b.say(rng.choice(RUN_FINAL[lang]).replace("%(n)ta", "%(n)d ta") % {"n": len(got), "lines": shown, "f": f})
    return b.finish()


CALLS = {"en": ["call read(%(f)s, %(a)d-%(z)d)", "use the read tool on %(f)s, lines %(a)d to %(z)d", "read %(f)s lines %(a)d-%(z)d", "use shell to count the lines in %(f)s",
                "use the shell tool: wc -l %(f)s", "call shell and list the files", "use shell: ls"],
         "uz": ["%(f)s ni o'qi, %(a)d-%(z)d qatorlar", "read vositasi bilan %(f)s ning %(a)d-%(z)d qatorlarini ko'rsat", "shell bilan fayllar ro'yxatini ko'rsat", "shell bilan %(f)s dagi qatorlarni sana"]}


def ask_call(rng, lang):
    n = rng.randint(3, 6)
    names, fn, funcs, starts, lines = _module(rng, n, rng.randrange(n))
    f = rng.choice(FILES)
    total = len(lines)
    a = rng.randint(1, max(1, total // 2))
    z = min(total, a + rng.randint(3, 12))
    ask = rng.choice(CALLS[lang]) % {"f": f, "a": a, "z": z}
    b = Builder(ask, {f: _src(lines)})
    if "wc" in ask or "count" in ask or "sana" in ask:
        out = b.call("shell", "wc -l %s" % f)
        b.say(("%s has %s lines." % (f, out.split()[0])) if lang == "en" else "%s da %s ta qator bor." % (f, out.split()[0]))
    elif "list" in ask or "ls" in ask.split() or "ro'yxat" in ask:
        out = b.call("shell", "ls")
        b.say(("The folder has: %s." % ", ".join(out.split())) if lang == "en" else "Papkada: %s." % ", ".join(out.split()))
    else:
        out = b.call("read", "%s, %d-%d" % (f, a, z))
        b.say(("Lines %d-%d of %s are shown above (%d lines)." % (a, z, f, z - a + 1)) if lang == "en" else "%s ning %d-%d qatorlari yuqorida (%d qator)." % (f, a, z, z - a + 1))
    return b.finish()


MAKERS = {"map_edit": map_edit, "map_edit_big": map_edit_big, "edit_insert": edit_insert, "edit_block": edit_block, "edit_delete": edit_delete,
          "search_hit": search_hit, "search_none": search_none, "ask_search": ask_search, "ask_run": ask_run, "ask_call": ask_call}


def build_all(per_kind: int = 100, seed: int = 0, counts: dict | None = None) -> list:
    rng = random.Random(seed)
    pool = search_pool()
    out = []
    for kind in KINDS:
        for i in range(per_kind if counts is None else counts[kind]):
            lang = "uz" if i % 3 == 2 else "en"
            mk = MAKERS[kind]
            for _ in range(30):                                     # never reuse an eval prompt word for word
                segs = mk(rng, lang, pool) if kind in ("search_hit", "search_none", "ask_search") else mk(rng, lang)
                if segs[0]["t"].split("\n")[1] not in EVAL_PROMPTS:
                    break
            out.append({"id": "S%s%05d" % (kind[:2].upper(), len(out)), "kind": kind, "lang": lang, "setup": dict(LAST_SETUP), "segments": segs})
    return out


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/sft/record/built/skill_traces.jsonl"
    data = build_all(seed=37, counts=DEFAULT_COUNTS)
    with open(out, "w", encoding="utf-8") as f:
        for e in data:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print("wrote %d examples to %s" % (len(data), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
