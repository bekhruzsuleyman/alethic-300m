"""F (one file) and G (several files) map examples, built by script from real code.
No model writes a map: names and line numbers come from ast, the answers are known by construction,
and every <map> line is the same format `tools.sftbuild.mapper.ast_map` makes ('name start-end -> ...')."""
import ast
import os
import random
import tempfile

from tools.sftbuild.fileops import numbered
from tools.sftbuild.mapper import ast_map
from tools.sftbuild.seq import Seq

CHUNK = 150          # lines per read while scanning a big file
BIG = 300            # a file with at least this many lines needs a map
SECTION_MAX = 100    # a section read back is at most this long
FLAGS = ["script_built", "real_code"]
SKIP_DIRS = {"test", "tests", "idlelib", "lib2to3", "site-packages", "__pycache__", "turtledemo", "encodings",
             "pydoc_data", "ensurepip", "distutils", "venv", "config-3", "tkinter", "lib-dynload"}

T = {
    "locate": {"en": 'In {path}, which top-level function or class is documented as "{doc}"? Reply with its name and line range.',
               "uz": '{path} faylida "{doc}" deb tavsiflangan funksiya yoki klass qaysi? Nomi va qatorlar oralig\'ini yozing.',
               "mix": 'In {path}: qaysi top-level funksiya "{doc}" deb yozilgan? Name va line range yozing.'},
    "locate_g": {"en": 'The project has these files: {files}. Which file and which top-level function or class is documented as "{doc}"? Reply with the file, the name and the line range.',
                 "uz": 'Loyihada shu fayllar bor: {files}. "{doc}" deb tavsiflangan funksiya yoki klass qaysi faylda? Fayl, nom va qatorlar oralig\'ini yozing.',
                 "mix": 'Project files: {files}. Qaysi file va qaysi top-level funksiya "{doc}" deb yozilgan? File, name va line range yozing.'},
    "control": {"en": "In {path}, lines {a}-{b} hold one function. List its parameter names.",
                "uz": "{path} faylida {a}-{b}-qatorlarda bitta funksiya bor. Uning parametr nomlarini yozing.",
                "mix": "{path}: {a}-{b} lines ichida bitta funksiya bor. Parameter names yozing."},
    "small": {"en": "How many parameters does `{name}` take in {path}?",
              "uz": "{path} faylidagi `{name}` nechta parametr oladi?",
              "mix": "{path} ichidagi `{name}` nechta parameter oladi?"},
}
PLAN = {
    "map": {"en": "The file is long, so I read it in chunks and keep a map, then read only the section I need.",
            "uz": "Fayl uzun, shuning uchun uni bo'laklab o'qiyman va xarita saqlayman, keyin faqat kerakli qismni o'qiyman.",
            "mix": "Fayl uzun, so I read it in chunks, map saqlayman, then faqat kerakli section."},
    "map_g": {"en": "There are several files, so I read each in chunks, keep one map for all of them, then read only the section I need.",
              "uz": "Fayllar ko'p, shuning uchun har birini bo'laklab o'qiyman, hammasi uchun bitta xarita saqlayman, keyin faqat kerakli qismni o'qiyman.",
              "mix": "Fayllar ko'p, so I read each in chunks, bitta map saqlayman, then faqat kerakli section."},
    "control": {"en": "I know the lines, so I read only that part.", "uz": "Qatorlar ma'lum, faqat shu qismni o'qiyman.",
                "mix": "Lines ma'lum, so I read only that part."},
    "small": {"en": "The file is short, so I read it whole.", "uz": "Fayl qisqa, uni butunlay o'qiyman.",
              "mix": "Fayl qisqa, so I read it whole."},
}


def _params(node):
    a = node.args
    out = [x.arg for x in a.posonlyargs + a.args]
    if a.vararg:
        out.append("*" + a.vararg.arg)
    out += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        out.append("**" + a.kwarg.arg)
    return out


def _find(src, name):
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name:
            return n
    raise KeyError(name)


def _doc(node):
    d = ast.get_docstring(node)
    if not d:
        return None
    first = " ".join(d.strip().split("\n\n")[0].split())
    cut = first.find(". ")
    first = first[:cut + 1] if cut > 0 else first
    return first if 20 <= len(first) <= 140 and '"' not in first and "\\" not in first else None


def _head(seq, task, plan):
    seq.add("<user>\n%s\n</user>\n" % task, 0)
    seq.add("<assistant>\n", 0)
    seq.add("<plan> %s </plan>\n" % plan, 1)


def _read(seq, path, lines, a, b, kind="read"):
    seq.add("<call> read(%s, %d-%d) </call>\n" % (path, a, b), 1)
    return seq.add("<result>\n%s\n</result>\n" % numbered(lines[a - 1:b], a), 0, kind)


def _end(seq, final):
    seq.add(final + "\n", 1)
    seq.add("</assistant>\n", 0)


def _final(lang, en, uz):
    return uz if lang == "uz" else en


def _tmp(src):
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    f.write(src)
    f.close()
    return f.name


def _scan(seq, path, src, maps_so_far, order, lang_files=None):
    """Read one file in chunks; after every chunk add the cumulative map (all files read so far)."""
    lines = src.splitlines()
    tmp = _tmp(src)
    try:
        for a in range(1, len(lines) + 1, CHUNK):
            b = min(a + CHUNK - 1, len(lines))
            idx = _read(seq, path, lines, a, b)
            maps_so_far[path] = ast_map(tmp, b)
            line = " | ".join("%s: %s" % (p, maps_so_far[p]) for p in order if p in maps_so_far and maps_so_far[p]) \
                if lang_files else maps_so_far[path]
            seq.set_read_map(idx, seq.add("<map> %s </map>\n" % line, 1, "map"))
    finally:
        os.unlink(tmp)


def build_f_map(path, src, target, lang="en", sid="F0"):
    node = _find(src, target)
    task = T["locate"][lang].format(path=path, doc=_doc(node))
    seq = Seq()
    _head(seq, task, PLAN["map"][lang])
    _scan(seq, path, src, {}, [path])
    _read(seq, path, src.splitlines(), node.lineno, node.end_lineno)
    _end(seq, _final(lang, "`%s`, lines %d-%d." % (target, node.lineno, node.end_lineno),
                     "`%s`, %d-%d-qatorlar." % (target, node.lineno, node.end_lineno)))
    return _tag(seq.samples(sid), "map")


def build_f_control(path, src, target, lang="en", sid="F0"):
    node = _find(src, target)
    seq = Seq()
    _head(seq, T["control"][lang].format(path=path, a=node.lineno, b=node.end_lineno), PLAN["control"][lang])
    _read(seq, path, src.splitlines(), node.lineno, node.end_lineno)
    ps = _params(node)
    en = "`%s` takes: %s." % (target, ", ".join(ps)) if ps else "`%s` takes no parameters." % target
    uz = "`%s` parametrlari: %s." % (target, ", ".join(ps)) if ps else "`%s` parametrsiz." % target
    _end(seq, _final(lang, en, uz))
    return _tag(seq.samples(sid), "control")


def build_f_small(path, src, target, lang="en", sid="F0"):
    node = _find(src, target)
    n = len(_params(node))
    seq = Seq()
    _head(seq, T["small"][lang].format(path=path, name=target), PLAN["small"][lang])
    lines = src.splitlines()
    _read(seq, path, lines, 1, len(lines))
    _end(seq, _final(lang, "`%s` takes %d parameter%s." % (target, n, "" if n == 1 else "s"),
                     "`%s` %d ta parametr oladi." % (target, n)))
    return _tag(seq.samples(sid), "small")


def build_g_map(files, target, lang="en", sid="G0"):
    """files: {name: source}; target: (file name, def name)."""
    fname, dname = target
    node = _find(files[fname], dname)
    task = T["locate_g"][lang].format(files=", ".join(files), doc=_doc(node))
    seq = Seq()
    _head(seq, task, PLAN["map_g"][lang])
    maps, order = {}, list(files)
    for p in order:
        _scan(seq, p, files[p], maps, order, lang_files=True)
    _read(seq, fname, files[fname].splitlines(), node.lineno, node.end_lineno)
    _end(seq, _final(lang, "`%s`: `%s`, lines %d-%d." % (fname, dname, node.lineno, node.end_lineno),
                     "`%s`: `%s`, %d-%d-qatorlar." % (fname, dname, node.lineno, node.end_lineno)))
    return _tag(seq.samples(sid), "map")


def build_g_control(files, target, lang="en", sid="G0"):
    fname, dname = target
    node = _find(files[fname], dname)
    seq = Seq()
    _head(seq, T["control"][lang].format(path=fname, a=node.lineno, b=node.end_lineno), PLAN["control"][lang])
    _read(seq, fname, files[fname].splitlines(), node.lineno, node.end_lineno)
    ps = _params(node)
    _end(seq, _final(lang, "`%s` takes: %s." % (dname, ", ".join(ps)) if ps else "`%s` takes no parameters." % dname,
                     "`%s` parametrlari: %s." % (dname, ", ".join(ps)) if ps else "`%s` parametrsiz." % dname))
    return _tag(seq.samples(sid), "control")


def _tag(samples, kind):
    for s in samples:
        s["flags"] = list(FLAGS) + ["kind:" + kind]
    return samples


# ---- choosing real code -------------------------------------------------------------------------------------------
def _usable(root):
    for d, dirs, names in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("."))
        for n in sorted(names):
            if n.endswith(".py"):
                yield os.path.join(d, n)


def _load(path):
    try:
        src = open(path, encoding="utf-8").read()
        ast.parse(src)
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError, RecursionError):
        return None
    return src if "\t" not in src and all(len(l) <= 140 for l in src.splitlines()) else None


def _targets(src, funcs_only=False):
    """Top-level defs with a usable docstring, section up to SECTION_MAX lines, doc unique in the file."""
    out, seen = [], {}
    nodes = [n for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for n in nodes:
        d = _doc(n)
        if d:
            seen[d] = seen.get(d, 0) + 1
    names = {}
    for n in nodes:
        names[n.name] = names.get(n.name, 0) + 1
    for n in nodes:
        d = _doc(n)
        if d and seen[d] == 1 and names[n.name] == 1 and 3 <= n.end_lineno - n.lineno + 1 <= SECTION_MAX \
                and not (funcs_only and isinstance(n, ast.ClassDef)):
            out.append(n.name)
    return out


def _lang(rng):
    return rng.choice(["en", "en", "en", "uz", "uz", "mix"])


def make_f(root, n, seed=7, kinds_only=False):
    """n F examples: 50% map (big file), 35% control (big file, small task), 15% small file."""
    rng = random.Random(seed)
    big, snippets = [], []
    for p in _usable(root):
        src = _load(p)
        if not src:
            continue
        rel = os.path.relpath(p, root).replace(os.sep, "/")
        nl = len(src.splitlines())
        if BIG <= nl <= 1500:
            t = _targets(src)
            if t:
                big.append((rel, src, t))
        for fn in ast.parse(src).body:
            if isinstance(fn, ast.FunctionDef) and 5 <= fn.end_lineno - fn.lineno + 1 <= 40 and _doc(fn):
                seg = ast.get_source_segment(src, fn)
                if seg and "\t" not in seg:
                    snippets.append((fn.name, seg + "\n"))
    rng.shuffle(big)
    rng.shuffle(snippets)
    n_map, n_control = round(n * 0.50), round(n * 0.35)
    kinds = ["map"] * n_map + ["control"] * n_control + ["small"] * (n - n_map - n_control)
    rng.shuffle(kinds)
    if kinds_only:
        return kinds
    out, used, si = [], set(), 0
    for i, kind in enumerate(kinds):
        lang, sid = _lang(rng), "F%04d" % (i + 1)
        if kind == "small":
            if si >= len(snippets):
                continue
            name, seg = snippets[si]
            si += 1
            out += build_f_small(name + ".py", seg, name, lang, sid)
            continue
        for _ in range(len(big)):
            rel, src, t = big[rng.randrange(len(big))]
            tgt = rng.choice(t)
            if (rel, tgt, kind) in used:
                continue
            node = _find(src, tgt)
            if kind == "control" and (isinstance(node, ast.ClassDef)):
                continue
            used.add((rel, tgt, kind))
            out += (build_f_map if kind == "map" else build_f_control)(rel, src, tgt, lang, sid)
            break
    return out


def make_g(root, n, seed=11):
    """n G examples: 80% multi-file map, 20% control, from directories that hold at least 3 usable files."""
    rng = random.Random(seed)
    pkgs = {}
    for p in _usable(root):
        src = _load(p)
        nl = len(src.splitlines()) if src else 0
        if src and 60 <= nl <= 450 and _targets(src):
            pkgs.setdefault(os.path.dirname(p), []).append((os.path.basename(p), src))
    pkgs = {d: f for d, f in pkgs.items() if len(f) >= 3}
    dirs = sorted(pkgs)
    out, used = [], set()
    for i in range(n):
        if not dirs:
            break
        d = rng.choice(dirs)
        chosen = sorted(rng.sample(pkgs[d], min(4, len(pkgs[d]))))
        files = dict(chosen)
        if sum(len(s.splitlines()) for s in files.values()) < 350:
            continue
        fname = rng.choice(sorted(files))
        t = _targets(files[fname])
        tgt = rng.choice(t)
        # the doc must also be unique across the chosen files, so the answer names exactly one place
        doc = _doc(_find(files[fname], tgt))
        if any(f != fname and _targets(s) and any(_doc(_find(s, x)) == doc for x in _targets(s)) for f, s in files.items()):
            continue
        key = (d, fname, tgt)
        if key in used:
            continue
        used.add(key)
        sid, lang = "G%04d" % (i + 1), _lang(rng)
        funcs = _targets(files[fname], funcs_only=True)
        if rng.random() < 0.8 or not funcs:
            out += build_g_map(files, (fname, tgt), lang, sid)
        else:
            out += build_g_control(files, (fname, rng.choice(funcs)), lang, sid)
    return out
