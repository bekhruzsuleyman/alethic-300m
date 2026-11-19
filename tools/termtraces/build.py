"""Plain-words file and folder requests -> the right tool call, for SFT.

"save file x with ...", "read file x", "make new folder d", "rename a to b", "copy", "move", "show the first 3 lines",
"how many lines", "find lines with word", "delete file", "what folder am I in", "add a line to x" ...

Fixed mapping, so the model can learn it:
  save / create / write a file with text      -> write(f)            (the text is the file body)
  read / open / view / show / look at a file  -> read(f, 1-100)      (a-b when the user names lines)
  make a folder                               -> shell(mkdir d)      (mkdir -p for a path)
  rename / move / copy / delete               -> shell(mv ...) / shell(cp ...) / shell(rm ...)
  first / last N lines, count, find text      -> shell(head ...) / (tail ...) / (wc ...) / (grep ...) / (find ...)
Then a check (`ls`, or the command's own output) and a final sentence built ONLY from the real output.
Every call runs in the real posix Workspace, like tools.errtraces and tools.progtraces.
"""
from __future__ import annotations

import json
import os
import random
import sys

from alethic.agent.protocol import Call
from tools.errtraces.build import Builder, _src

KINDS = ["save_file", "read_file", "read_lines", "make_folder", "folder_and_file", "rename", "copy", "move", "head_tail",
         "count", "find_text", "find_files", "list_folder", "delete", "where", "append"]
STEMS = ["notes", "todo", "ideas", "report", "draft", "plan", "memo", "list", "diary", "budget", "log", "shopping", "tasks", "readme", "summary"]
EXTS = [".txt", ".txt", ".txt", ".md"]
FOLDERS = ["logs", "photos", "backup", "docs", "src", "data", "temp", "output", "images", "archive", "projects", "school", "music", "reports", "tests"]
EN_LINES = ["buy milk", "call Ali", "pay the bill", "water the plants", "read a chapter", "fix the bike", "write to mom", "book a ticket",
            "clean the desk", "plan the trip", "send the report", "learn ten words", "walk for an hour", "cook dinner", "update the notes",
            "check the mail", "visit the market", "backup the files", "wash the car", "call the bank", "study python", "buy bread"]
UZ_LINES = ["non olish", "Alini chaqirish", "hisobni to'lash", "gullarni sug'orish", "bir bob o'qish", "velosipedni tuzatish", "onamga yozish",
            "chipta band qilish", "stolni tozalash", "sayohatni rejalashtirish", "hisobotni yuborish", "o'nta so'z o'rganish", "bir soat yurish",
            "kechki ovqat tayyorlash", "yozuvlarni yangilash", "pochtani tekshirish", "bozorga borish", "fayllarni zaxiralash", "mashinani yuvish",
            "bankka qo'ng'iroq qilish", "python o'rganish", "sut olish"]
WORDS = ["buy", "call", "the", "plan", "report", "file", "bill", "note", "read", "check"]
PLAN = {
    "save": {"en": ["I will save the file.", "I will write the file with that text."], "uz": ["Faylni saqlayman.", "Faylni shu matn bilan yozaman."]},
    "read": {"en": ["I will read the file.", "I will open the file and read it."], "uz": ["Faylni o'qiyman.", "Faylni ochib o'qiyman."]},
    "mkdir": {"en": ["I will make the folder.", "I will create the folder, then check."], "uz": ["Papkani yarataman.", "Papkani yarataman va tekshiraman."]},
    "mv": {"en": ["I will move it, then check the folder.", "I will do it and list the files."], "uz": ["Buni bajaraman va fayllarni ko'raman.", "Bajaraman, keyin papkani tekshiraman."]},
    "show": {"en": ["I will look at the file.", "I will check it with a command."], "uz": ["Faylni ko'raman.", "Buyruq bilan tekshiraman."]},
}


def plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def pick_lines(rng, lang, n):
    return rng.sample(EN_LINES if lang == "en" else UZ_LINES, n)


def fname(rng):
    return rng.choice(STEMS) + rng.choice(EXTS)


def two_names(rng):
    a = fname(rng)
    b = fname(rng)
    while b == a or b.split(".")[0] == a.split(".")[0]:
        b = fname(rng)
    return a, b


def parse_read(out: str) -> list:
    """The text of each numbered line of a read() result."""
    return [l[4:] for l in out.split("\n")]


def names(out: str) -> list:
    return [n for n in out.split("\n") if n]


def join_names(ns, lang):
    return ", ".join(ns) if ns else ("nothing" if lang == "en" else "hech narsa")


def ask(rng, lang, en, uz, **kw):
    return rng.choice(en if lang == "en" else uz).format(**kw)


# --- save / read ----------------------------------------------------------------------------------------------------


def save_file(rng, lang):
    f = fname(rng)
    n = rng.choice([1, 1, 1, 2, 3])
    lines = pick_lines(rng, lang, n)
    text = "; ".join(lines) if n > 1 else lines[0]
    user = ask(rng, lang,
               ["save file {f} with: {t}", "Save a file called {f} with this text: {t}", "create {f} and write {t} in it", "Write \"{t}\" to {f}.",
                "save {t} into {f}", "Please save the file {f}. Content: {t}", "make a new file {f} containing: {t}"],
               ["{f} faylni saqla: {t}", "{f} nomli fayl yarating va ichiga yozing: {t}", "{t} matnini {f} ga saqlang.", "Yangi fayl {f} yarating, ichida: {t}",
                "{f} faylni saqlang. Matn: {t}"], f=f, t=text)
    b = Builder(user)
    b.plan(rng.choice(PLAN["save"][lang]))
    b.call("write", f, _src(lines))
    shown = "; ".join(lines)
    b.say(("Saved %s with %d %s: %s." % (f, n, plural(n, "line", "lines"), shown)) if lang == "en"
          else "%s saqlandi, %d qator: %s." % (f, n, shown))
    return b.finish()


def read_file(rng, lang):
    f = fname(rng)
    n = rng.randint(1, 8)
    lines = pick_lines(rng, lang, n)
    user = ask(rng, lang,
               ["read file {f}", "Read {f}.", "open {f} and read it", "Can you read the file {f}?", "view {f}", "look at {f}", "read {f} please",
                "Show me what is in {f}.", "what is inside {f}?"],
               ["{f} faylni o'qi", "{f} ni o'qing.", "{f} faylni oching va o'qing", "{f} ichida nima bor?", "{f} faylni ko'rsating"], f=f)
    b = Builder(user, {f: _src(lines)})
    b.plan(rng.choice(PLAN["read"][lang]))
    out = b.call("read", "%s, 1-100" % f)
    got = parse_read(out)
    assert got == lines, (got, lines)
    if n == 1:
        b.say(('%s has one line: "%s".' % (f, got[0])) if lang == "en" else '%s da bitta qator bor: "%s".' % (f, got[0]))
    else:
        b.say(('%s has %d lines. The first is "%s" and the last is "%s".' % (f, n, got[0], got[-1])) if lang == "en"
              else '%s da %d qator bor. Birinchisi "%s", oxirgisi "%s".' % (f, n, got[0], got[-1]))
    return b.finish()


def read_lines(rng, lang):
    f = fname(rng)
    n = rng.randint(8, 20)
    lines = pick_lines(rng, lang, min(n, len(EN_LINES)))
    n = len(lines)
    a = rng.randint(1, n - 2)
    z = rng.randint(a + 1, min(n, a + 4))
    user = ask(rng, lang,
               ["read lines {a}-{z} of {f}", "Show me lines {a} to {z} of {f}.", "read {f} from line {a} to line {z}", "open {f} and show lines {a}-{z}"],
               ["{f} ning {a}-{z} qatorlarini o'qing", "{f} faylning {a} dan {z} gacha qatorlarini ko'rsating", "{f} dan {a}-{z} qatorlarni o'qi"], f=f, a=a, z=z)
    b = Builder(user, {f: _src(lines)})
    b.plan(rng.choice(PLAN["read"][lang]))
    out = b.call("read", "%s, %d-%d" % (f, a, z))
    got = parse_read(out)
    assert got == lines[a - 1:z], (got, lines)
    shown = "; ".join('%d: %s' % (a + i, t) for i, t in enumerate(got))
    b.say(("Lines %d-%d of %s: %s." % (a, z, f, shown)) if lang == "en" else "%s ning %d-%d qatorlari: %s." % (f, a, z, shown))
    return b.finish()


# --- folders ----------------------------------------------------------------------------------------------------------


def _setup_files(rng, lang, k):
    files = {}
    for _ in range(k):
        files[fname(rng)] = _src(pick_lines(rng, lang, 2))
    return files


def make_folder(rng, lang):
    d = rng.choice(FOLDERS)
    nested = rng.random() < 0.25
    path = d + "/" + rng.choice(["2024", "old", "new", "week1", "part1", "final"]) if nested else d
    files = _setup_files(rng, lang, rng.randint(0, 3))
    user = ask(rng, lang,
               ["make new folder {p}", "Make a new folder called {p}.", "create a folder {p}", "mkdir {p}", "new folder {p} please", "Create a directory named {p}.",
                "make a folder {p}"],
               ["yangi papka yarating: {p}", "{p} nomli papka oching", "{p} papkani yarating", "yangi papka {p}", "{p} degan papka yarat"], p=path)
    b = Builder(user, files)
    b.plan(rng.choice(PLAN["mkdir"][lang]))
    out = b.call("shell", "mkdir %s%s" % ("-p " if nested else "", path))
    assert out == "", out
    ls = names(b.call("shell", "ls"))
    assert d in ls
    b.say(("Created the folder %s. This folder now has: %s." % (path, join_names(ls, lang))) if lang == "en"
          else "%s papkasi yaratildi. Endi bu yerda: %s." % (path, join_names(ls, lang)))
    return b.finish()


def folder_and_file(rng, lang):
    d = rng.choice(FOLDERS)
    f = fname(rng)
    line = pick_lines(rng, lang, 1)[0]
    user = ask(rng, lang,
               ["make a folder {d} and save {f} in it with: {t}", "Create folder {d}, then a file {f} inside it containing: {t}",
                "new folder {d}, and put a file {f} in it that says {t}"],
               ["{d} papka yarating va ichiga {f} faylni saqlang: {t}", "Yangi papka {d} oching, ichida {f} fayl bo'lsin: {t}"], d=d, f=f, t=line)
    b = Builder(user)
    b.plan(rng.choice(PLAN["mkdir"][lang]))
    assert b.call("shell", "mkdir %s" % d) == ""
    b.call("write", "%s/%s" % (d, f), _src([line]))
    ls = names(b.call("shell", "ls %s" % d))
    assert ls == [f], ls
    b.say(("Made the folder %s and saved %s in it with: %s." % (d, f, line)) if lang == "en" else "%s papkasi yaratildi, ichiga %s saqlandi: %s." % (d, f, line))
    return b.finish()


def list_folder(rng, lang):
    d = rng.choice(FOLDERS)
    inside = sorted({fname(rng) for _ in range(rng.randint(1, 4))})
    setup = {"%s/%s" % (d, n): _src(pick_lines(rng, lang, 1)) for n in inside}
    setup[fname(rng)] = "x\n"
    user = ask(rng, lang,
               ["list folder {d}", "What is in the folder {d}?", "show me the files in {d}", "ls {d}", "list the files inside {d}"],
               ["{d} papkani ko'rsating", "{d} papkada qanday fayllar bor?", "{d} ichidagi fayllar ro'yxati", "{d} papka ichini ko'rsating"], d=d)
    b = Builder(user, {})
    for rel, text in setup.items():  # the Builder only writes top-level files, so nested ones go in through write
        if "/" in rel:
            b.ws.execute(Call("write", rel, text))
        else:
            with open(os.path.join(b.dir, rel), "w", encoding="utf-8") as fh:
                fh.write(text)
    b.plan(rng.choice(PLAN["show"][lang]))
    out = b.call("shell", "ls %s" % d)
    ns = names(out)
    assert ns == inside, (ns, inside)
    b.say(("The folder %s has %d %s: %s." % (d, len(ns), plural(len(ns), "file", "files"), ", ".join(ns))) if lang == "en"
          else "%s papkasida %d ta fayl bor: %s." % (d, len(ns), ", ".join(ns)))
    return b.finish()


# --- rename, copy, move, delete -----------------------------------------------------------------------------------------


def rename(rng, lang):
    a, c = two_names(rng)
    files = {a: _src(pick_lines(rng, lang, 2))}
    if rng.random() < 0.5:
        files[fname(rng)] = "x\n"
    files.pop(c, None)
    user = ask(rng, lang,
               ["rename {a} to {c}", "Rename the file {a} to {c}.", "change the name of {a} to {c}", "rename file {a} -> {c}", "call {a} {c} instead"],
               ["{a} ni {c} deb qayta nomlang", "{a} faylning nomini {c} ga o'zgartiring", "{a} nomini {c} qiling", "{a} faylni {c} deb nomlang"], a=a, c=c)
    b = Builder(user, files)
    b.plan(rng.choice(PLAN["mv"][lang]))
    assert b.call("shell", "mv %s %s" % (a, c)) == ""
    ns = names(b.call("shell", "ls"))
    assert c in ns and a not in ns
    b.say(("Renamed %s to %s. The folder now has: %s." % (a, c, ", ".join(ns))) if lang == "en" else "%s endi %s deb nomlandi. Papkada: %s." % (a, c, ", ".join(ns)))
    return b.finish()


def copy(rng, lang):
    a, c = two_names(rng)
    files = {a: _src(pick_lines(rng, lang, 2))}
    user = ask(rng, lang,
               ["copy {a} to {c}", "Make a copy of {a} called {c}.", "cp {a} {c}", "duplicate the file {a} as {c}", "copy file {a} as {c}"],
               ["{a} ni {c} ga nusxalang", "{a} faylning nusxasini {c} nomi bilan yarating", "{a} dan {c} nomli nusxa oling", "{a} faylni {c} deb nusxa qiling"], a=a, c=c)
    b = Builder(user, files)
    b.plan(rng.choice(PLAN["mv"][lang]))
    assert b.call("shell", "cp %s %s" % (a, c)) == ""
    ns = names(b.call("shell", "ls"))
    assert a in ns and c in ns
    b.say(("Copied %s to %s. Now there are %d files: %s." % (a, c, len(ns), ", ".join(ns))) if lang == "en" else "%s %s ga nusxalandi. Endi %d ta fayl bor: %s." % (a, c, len(ns), ", ".join(ns)))
    return b.finish()


def move(rng, lang):
    a = fname(rng)
    d = rng.choice(FOLDERS)
    files = {a: _src(pick_lines(rng, lang, 2))}
    user = ask(rng, lang,
               ["move {a} into {d}", "Move the file {a} to the folder {d}.", "put {a} in {d}", "mv {a} {d}", "move file {a} to {d}"],
               ["{a} ni {d} papkaga ko'chiring", "{a} faylni {d} ichiga joylang", "{a} ni {d} ga o'tkazing", "{a} faylni {d} papkaga o'tkaz"], a=a, d=d)
    b = Builder(user, files)
    b.plan("I will make the folder if needed, move the file, then check." if lang == "en" else "Kerak bo'lsa papka yarataman, faylni ko'chiraman va tekshiraman.")
    assert b.call("shell", "mkdir %s" % d) == ""
    assert b.call("shell", "mv %s %s" % (a, d)) == ""
    ns = names(b.call("shell", "ls %s" % d))
    assert ns == [a], ns
    b.say(("Moved %s into %s. %s now has: %s." % (a, d, d, a)) if lang == "en" else "%s %s ga ko'chirildi. %s ichida: %s." % (a, d, d, a))
    return b.finish()


def delete(rng, lang):
    k = rng.randint(2, 4)
    files = _setup_files(rng, lang, k)
    if len(files) < 2:
        files[fname(rng) + "x"] = "y\n"
    target = rng.choice(sorted(files))
    user = ask(rng, lang,
               ["delete {f}", "Delete the file {f}.", "remove {f}", "rm {f}", "delete file {f} please", "get rid of {f}"],
               ["{f} ni o'chiring", "{f} faylni o'chir", "{f} faylni o'chirib tashlang", "{f} ni olib tashlang"], f=target)
    b = Builder(user, files)
    b.plan("I will delete the file, then check the folder." if lang == "en" else "Faylni o'chiraman, keyin papkani tekshiraman.")
    assert b.call("shell", "rm %s" % target) == ""
    ns = names(b.call("shell", "ls"))
    assert target not in ns
    b.say(("Deleted %s. The folder now has: %s." % (target, join_names(ns, lang))) if lang == "en" else "%s o'chirildi. Papkada qoldi: %s." % (target, join_names(ns, lang)))
    return b.finish()


# --- look inside a file ---------------------------------------------------------------------------------------------------


def head_tail(rng, lang):
    f = fname(rng)
    n = rng.randint(8, 18)
    lines = pick_lines(rng, lang, n)
    k = rng.randint(2, 5)
    first = rng.random() < 0.5
    user = ask(rng, lang,
               (["show the first {k} lines of {f}", "head {f} {k} lines", "Show me the first {k} lines in {f}."] if first
                else ["show the last {k} lines of {f}", "tail {f}, {k} lines", "Show me the last {k} lines in {f}."]),
               (["{f} ning birinchi {k} qatorini ko'rsating", "{f} faylning boshidagi {k} qatorni ko'rsat"] if first
                else ["{f} ning oxirgi {k} qatorini ko'rsating", "{f} faylning oxiridagi {k} qatorni ko'rsat"]), f=f, k=k)
    b = Builder(user, {f: _src(lines)})
    b.plan(rng.choice(PLAN["show"][lang]))
    out = b.call("shell", "%s -n %d %s" % ("head" if first else "tail", k, f))
    got = out.split("\n")
    assert got == (lines[:k] if first else lines[-k:]), got
    word = ("first" if first else "last") if lang == "en" else ("birinchi" if first else "oxirgi")
    b.say(("The %s %d lines of %s: %s." % (word, k, f, "; ".join(got))) if lang == "en" else "%s ning %s %d qatori: %s." % (f, word, k, "; ".join(got)))
    return b.finish()


def count(rng, lang):
    f = fname(rng)
    n = rng.randint(3, 20)
    lines = pick_lines(rng, lang, min(n, len(EN_LINES)))
    mode = rng.choice(["lines", "lines", "words", "files"])
    if mode == "files":
        files = _setup_files(rng, lang, rng.randint(2, 5))
        user = ask(rng, lang, ["how many files are here?", "count the files in this folder", "How many files do I have?"],
                   ["Bu yerda nechta fayl bor?", "Papkadagi fayllarni sanang", "Nechta faylim bor?"])
        b = Builder(user, files)
        b.plan(rng.choice(PLAN["show"][lang]))
        out = b.call("shell", "ls | wc -l")
        assert out == str(len(files)), (out, files)
        b.say(("There %s %s %s here." % (plural(int(out), "is", "are"), out, plural(int(out), "file", "files"))) if lang == "en" else "Bu yerda %s ta fayl bor." % out)
        return b.finish()
    flag = "-l" if mode == "lines" else "-w"
    thing = ("lines" if mode == "lines" else "words") if lang == "en" else ("qator" if mode == "lines" else "so'z")
    user = ask(rng, lang, ["how many {t} are in {f}?", "count the {t} in {f}", "wc {fl} {f}", "How many {t} does {f} have?"],
               ["{f} da nechta {t} bor?", "{f} dagi {t} sonini sanang", "{f} faylda nechta {t} bor?"], t=thing, f=f, fl=flag)
    b = Builder(user, {f: _src(lines)})
    b.plan(rng.choice(PLAN["show"][lang]))
    out = b.call("shell", "wc %s %s" % (flag, f))
    num, name = out.split()
    assert name == f
    b.say(("%s has %s %s." % (f, num, thing)) if lang == "en" else "%s da %s ta %s bor." % (f, num, thing))
    return b.finish()


def find_text(rng, lang):
    f = fname(rng)
    lines = pick_lines(rng, lang, rng.randint(6, 14))
    word = rng.choice(WORDS if lang == "en" else ["non", "Ali", "to'l", "gul", "o'qish", "yozish", "bozor", "bank"])
    user = ask(rng, lang,
               ["find lines with {w} in {f}", "Which lines of {f} contain \"{w}\"?", "grep {w} {f}", "search {f} for {w}", "find \"{w}\" in {f}"],
               ["{f} dan {w} bor qatorlarni toping", "{f} da \"{w}\" so'zi bor qatorlar qaysi?", "{f} ichidan {w} ni qidiring"], w=word, f=f)
    b = Builder(user, {f: _src(lines)})
    b.plan(rng.choice(PLAN["show"][lang]))
    out = b.call("shell", 'grep -n "%s" %s' % (word, f))
    if out == "":
        b.say(("No line in %s contains \"%s\"." % (f, word)) if lang == "en" else "%s da \"%s\" bor qator topilmadi." % (f, word))
    else:
        hits = out.split("\n")
        shown = "; ".join("line %s" % h.replace(":", ": ", 1) if lang == "en" else "%s-qator: %s" % tuple(h.split(":", 1)) for h in hits)
        b.say(("Found %d %s with \"%s\" in %s: %s." % (len(hits), plural(len(hits), "line", "lines"), word, f, shown)) if lang == "en"
              else "%s da \"%s\" bor %d qator topildi: %s." % (f, word, len(hits), shown))
    return b.finish()


def find_files(rng, lang):
    ext = rng.choice([".py", ".txt", ".md"])
    d = rng.choice(FOLDERS)
    inside = ["%s/%s%s" % (d, rng.choice(STEMS), ext) for _ in range(rng.randint(0, 2))]
    top = ["%s%s" % (rng.choice(STEMS), ext) for _ in range(rng.randint(1, 2))]
    other = ["%s%s" % (rng.choice(STEMS), rng.choice([e for e in [".py", ".txt", ".md", ".json"] if e != ext]))]
    setup = {p: "x\n" for p in inside + top + other}
    user = ask(rng, lang,
               ["find all {e} files", "Find every {e} file in this folder.", "find . -name \"*{e}\"", "where are my {e} files?", "list all {e} files, also in subfolders"],
               ["barcha {e} fayllarni toping", "Papkadagi hamma {e} fayllarni toping.", "{e} fayllar qayerda?", "ichki papkalar bilan barcha {e} fayllarni ko'rsating"], e=ext)
    b = Builder(user, {})
    for rel, text in setup.items():
        b.ws.execute(Call("write", rel, text))
    b.plan(rng.choice(PLAN["show"][lang]))
    out = b.call("shell", 'find . -name "*%s"' % ext)
    found = sorted(out.split("\n")) if out else []
    assert sorted("./" + p for p in set(inside + top)) == found, (found, setup)
    b.say(("Found %d %s files: %s." % (len(found), ext, ", ".join(found))) if lang == "en" else "%d ta %s fayl topildi: %s." % (len(found), ext, ", ".join(found)))
    return b.finish()


def where(rng, lang):
    files = _setup_files(rng, lang, rng.randint(0, 2))
    user = ask(rng, lang, ["what folder am I in?", "pwd", "Where am I?", "Which folder is this?", "show the current folder"],
               ["Men qaysi papkadaman?", "Hozir qaysi papkadamiz?", "Joriy papka qaysi?", "Bu qaysi papka?"])
    b = Builder(user, files)
    b.plan(rng.choice(PLAN["show"][lang]))
    out = b.call("shell", "pwd")
    assert out == "/work", out
    b.say("You are in %s, the work folder." % out if lang == "en" else "Siz %s da, ish papkasidasiz." % out)
    return b.finish()


def append(rng, lang):
    f = fname(rng)
    lines = pick_lines(rng, lang, rng.randint(1, 3))
    extra = rng.choice([l for l in (EN_LINES if lang == "en" else UZ_LINES) if l not in lines])
    user = ask(rng, lang,
               ["add a line to {f}: {t}", "Append \"{t}\" to {f}.", "add {t} at the end of {f}", "put one more line in {f}: {t}"],
               ["{f} ga yangi qator qo'shing: {t}", "{f} ning oxiriga \"{t}\" ni qo'shing", "{f} faylga yana bitta qator yozing: {t}"], f=f, t=extra)
    b = Builder(user, {f: _src(lines)})
    b.plan("I will add the line at the end, then show the file." if lang == "en" else "Qatorni oxiriga qo'shaman, keyin faylni ko'rsataman.")
    assert b.call("shell", "echo \"%s\" >> %s" % (extra, f)) == ""
    out = b.call("shell", "cat %s" % f)
    got = out.split("\n")
    assert got == lines + [extra], got
    b.say(("Added \"%s\" to %s. It now has %d lines; the last is \"%s\"." % (extra, f, len(got), got[-1])) if lang == "en"
          else "\"%s\" %s ga qo'shildi. Endi %d qator bor, oxirgisi \"%s\"." % (extra, f, len(got), got[-1]))
    return b.finish()


MAKERS = {"save_file": save_file, "read_file": read_file, "read_lines": read_lines, "make_folder": make_folder, "folder_and_file": folder_and_file,
          "rename": rename, "copy": copy, "move": move, "head_tail": head_tail, "count": count, "find_text": find_text, "find_files": find_files,
          "list_folder": list_folder, "delete": delete, "where": where, "append": append}
DEFAULT_COUNTS = {"save_file": 140, "read_file": 140, "read_lines": 60, "make_folder": 120, "folder_and_file": 60, "rename": 80, "copy": 60,
                  "move": 60, "head_tail": 70, "count": 70, "find_text": 70, "find_files": 50, "list_folder": 50, "delete": 60, "where": 30, "append": 60}


def build_all(per_kind=60, seed: int = 0, counts: dict | None = None) -> list:
    rng = random.Random(seed)
    out = []
    for kind in KINDS:
        want = (counts or {}).get(kind, per_kind)
        seen = set()
        for i in range(want):
            lang = "en" if i % 2 == 0 else "uz"
            for _ in range(80):                      # a repeated text teaches nothing, so draw again
                segs = MAKERS[kind](rng, lang)
                text = "".join(x["t"] for x in segs)
                if text not in seen:
                    break
            else:
                continue
            seen.add(text)
            out.append({"id": "T%s%04d" % (kind[:2].upper(), len(out)), "kind": kind, "lang": lang, "segments": segs})
    return out


def to_text(example: dict) -> str:
    return "".join(s["t"] for s in example["segments"])


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "term_traces.jsonl"
    data = build_all(seed=29, counts=DEFAULT_COUNTS)
    with open(out, "w", encoding="utf-8") as f:
        for e in data:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print("wrote %d examples to %s" % (len(data), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
