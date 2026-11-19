"""Scripted error-recovery traces for SFT stage B.

Every tool result is produced by the REAL Workspace (the same code the exe runs), never written by hand, so the
model sees exactly the error texts it will meet. The step that caused a failure and the failure itself get loss 0
(the model must not learn to make the mistake); the recovery and the final words get loss 1.

Output: records {"id", "kind", "lang", "segments": [{"t", "loss"}]} like the other built SFT files.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile

from alethic.agent.protocol import Call
from alethic.agent.tools import Workspace

KINDS = ["bug_fix", "edit_range", "read_range", "missing_file", "unknown_tool", "read_no_range", "blocked", "give_up"]
FILES = ["main.py", "app.py", "calc.py", "tool.py", "demo.py", "run.py", "task.py", "script.py"]

# --- small programs with one bug each: (task_en, task_uz, lines, bug line (1-based), replacement lines, what was wrong en/uz)


def _bugs(rng):
    n = rng.choice([3, 5, 8, 12])
    st = rng.choice([5, 10])
    name, tot = rng.choice([("prices", "total"), ("scores", "points"), ("weights", "mass"), ("sizes", "volume")])
    typo = tot[:-2] + tot[-1]
    items = rng.sample(["red", "green", "blue", "gray", "pink", "gold"], 3)
    ages = rng.sample([("ali", 20), ("vali", 31), ("sami", 25), ("nodir", 18)], 2)
    return [
        ("a Python file that prints a Celsius to Fahrenheit table", "Selsiydan Farengeytga o'giradigan jadval chiqaradigan Python fayl",
         ["def to_f(c):", "    return c * 9 / 5 + 32", "", "for c in range(0, %d, %d):" % (n * 10, st), "    f = to_f(c)",
          '    print("C = " + c + " F = " + str(f))'], 6,
         ['    print("C = " + str(c) + " F = " + str(f))'], "an int cannot be added to a str, so I wrapped it in str()", "int ni str ga qo'shib bo'lmaydi, str() bilan o'radim"),
        ("a script that adds up a list of numbers and prints the sum", "sonlar ro'yxatini qo'shib yig'indisini chiqaradigan skript",
         ["%s = [%d, %d, %d, %d]" % (name, n, n + 5, 30, 4), "%s = 0" % tot, "for p in %s:" % name, "    %s += p" % tot, 'print("Total:", %s)' % typo], 5,
         ['print("Total:", %s)' % tot], "the variable name was misspelled on the last line", "oxirgi qatorda o'zgaruvchi nomi xato yozilgan edi"),
        ("a script that prints the last item of a list", "ro'yxatning oxirgi elementini chiqaradigan skript",
         ["items = %r" % items, "last = items[len(items)]", "print(last)"], 2,
         ["last = items[len(items) - 1]"], "list indexes stop at len - 1", "ro'yxat indekslari len - 1 da tugaydi"),
        ("a function that returns the average of a list and prints it for an empty list", "bo'sh ro'yxat bilan ham ishlaydigan, o'rtachani hisoblovchi funksiya",
         ["def average(xs):", "    return sum(xs) / len(xs)", "", "data = []", "print(average(data))"], 2,
         ["    return sum(xs) / len(xs) if xs else 0"], "an empty list divides by zero, so I added a guard", "bo'sh ro'yxat nolga bo'lishga olib keldi, shart qo'shdim"),
        ("a script that prints one person's age from a dictionary", "lug'atdan bitta odamning yoshini chiqaradigan skript",
         ["ages = {%r: %d, %r: %d}" % (ages[0][0], ages[0][1], ages[1][0], ages[1][1]), 'print(ages["%s"])' % ages[1][0].title()], 2,
         ['print(ages["%s"])' % ages[1][0]], "the key is lowercase in the dictionary", "kalit lug'atda kichik harf bilan yozilgan"),
        ("a function that squares a number and prints the square of 7", "sonni kvadratga ko'taradigan va 7 ning kvadratini chiqaradigan funksiya",
         ["def square(n)", "    return n * n", "", "print(square(7))"], 1,
         ["def square(n):"], "the def line was missing a colon", "def qatorida ikki nuqta yetishmagan edi"),
        ("a script that collects names in a list and prints them", "ismlarni ro'yxatga yig'ib chiqaradigan skript",
         ['names = "ali,vali"', 'names.append("sher")', "print(names)"], 1,
         ['names = ["ali", "vali"]'], "names was a str, and only a list has append", "names matn edi, append faqat ro'yxatda bor"),
        ("a script that doubles the number 3.5", "3.5 sonini ikkilantiradigan skript",
         ['raw = "3.5"', "qty = int(raw)", "print(qty * 2)"], 2,
         ["qty = float(raw)"], "int() cannot read a decimal string, so I used float()", "int() o'nli matnni o'qiy olmaydi, float() ishlatdim"),
        ("a function that returns the area of a rectangle", "to'g'ri to'rtburchak yuzini qaytaradigan funksiya",
         ["def area(w, h):", "    return w * h", "", "print(area(5))"], 4,
         ["print(area(5, 3))"], "the call was missing the second argument", "chaqiruvda ikkinchi argument yetishmagan edi"),
        ("a script that prints the area of a circle", "doira yuzini chiqaradigan skript",
         ["radius = %d" % n, "print(math.pi * radius ** 2)"], 1,
         ["import math", "radius = %d" % n], "math was never imported", "math import qilinmagan edi"),
        ("a script that prints every index of a list", "ro'yxatning har bir indeksini chiqaradigan skript",
         ["xs = [4, 8, 15]", "for i in len(xs):", "    print(i)"], 2,
         ["for i in range(len(xs)):"], "a number cannot be looped over, so I used range()", "sonni aylanib bo'lmaydi, range() ishlatdim"),
    ]


# --- existing small files the user already has: (lines, what it does en/uz, (line, new line, request en, request uz))


def _setups(rng):
    g1, g2 = rng.sample(["hello", "hi", "hey", "welcome", "greetings", "good day"], 2)
    who = rng.choice(["world", "team", "friend", "everyone", "class"])
    lim, lim2 = rng.sample(range(3, 40), 2)
    price = rng.choice([80, 120, 200, 450, 999])
    tax, tax2 = rng.sample([0.05, 0.1, 0.15, 0.2, 0.25], 2)
    cols = rng.sample(["red", "green", "blue", "gray", "pink", "black", "white", "orange"], 4)
    return [
        (['GREETING = "%s"' % g1, 'NAME = "%s"' % who, "", "def show():", '    print(GREETING + ", " + NAME)', "", "show()"],
         "prints a greeting", "salomlashuv chiqaradi", (1, 'GREETING = "%s"' % g2, 'set the greeting to "%s"' % g2, "salomlashuvni \"%s\" qilib qo'y" % g2)),
        (["LIMIT = %d" % lim, "", "def squares():", "    for i in range(LIMIT):", "        print(i * i)", "", "squares()"],
         "prints squares of numbers", "sonlar kvadratini chiqaradi", (1, "LIMIT = %d" % lim2, "make it stop at %d" % lim2, "to'xtash chegarasini %d qil" % lim2)),
        (["PRICE = %d" % price, "TAX = %s" % tax, "", "def total():", "    return PRICE * (1 + TAX)", "", "print(total())"],
         "prints a price with tax", "soliq bilan narxni chiqaradi", (2, "TAX = %s" % tax2, "change the tax to %s" % tax2, "soliqni %s qil" % tax2)),
        (["COLORS = %r" % cols[:3], "", "for c in COLORS:", "    print(c.upper())"],
         "prints colors in capitals", "ranglarni bosh harflarda chiqaradi", (1, "COLORS = %r" % cols, "add %s to the colors" % cols[3], "ranglarga %s qo'sh" % cols[3])),
    ]


PLAN = {
    "en": ["I will write one file and run it.", "One file, then I run it with python3.", "I will write the code and test it."],
    "uz": ["Bitta fayl yozaman, python3 bilan ishga tushiraman.", "Kodni yozib, ishga tushirib tekshiraman.", "Faylni yozaman va natijani tekshiraman."],
}
EDIT = {"en": ["I will edit the line.", "I will change that line."], "uz": ["Qatorni o'zgartiraman.", "Shu qatorni tahrirlayman."]}
LOOK = {"en": ["I will read the file, then fix the line.", "First I read the file to see the real lines."], "uz": ["Avval faylni o'qiyman, keyin qatorni to'g'rilayman.", "Avval fayldagi haqiqiy qatorlarni ko'raman."]}
RUN = {"en": ["I will run it.", "I will run the file."], "uz": ["Ishga tushiraman.", "Faylni ishga tushiraman."]}
OK = {"en": ["Done, it runs now.", "Fixed, and it runs without errors."], "uz": ["Tayyor, endi ishlaydi.", "To'g'riladim, endi xatosiz ishlaydi."]}


class Builder:
    """Records one conversation while running every call for real in a scratch folder."""

    def __init__(self, user: str, setup: dict | None = None):
        self.dir = tempfile.mkdtemp(prefix="errtrace")
        self.ws = Workspace(self.dir, mode="posix", python="python3", timeout=10)
        for name, text in (setup or {}).items():
            with open(os.path.join(self.dir, name), "w", encoding="utf-8") as f:
                f.write(text)
        self.segs = [{"t": "<user>\n%s\n</user>\n" % user, "loss": 0}, {"t": "<assistant>\n", "loss": 0}]

    def say(self, text: str, loss: int = 1):
        self.segs.append({"t": text + "\n", "loss": loss})

    def plan(self, text: str):
        self.segs.append({"t": "<plan> %s </plan>\n" % text, "loss": 1})

    def call(self, kind: str, args: str, body: str | None = None, trained: bool = True) -> str | None:
        text = "<call> %s(%s) </call>\n" % (kind, args) + (body if body is not None else "")
        self.segs.append({"t": text, "loss": int(trained)})
        out = self.ws.execute(Call(kind, args, body))
        if out.result is not None:
            self.segs.append({"t": "<result>\n%s\n</result>\n" % out.result, "loss": 0})
        return out.result

    def finish(self) -> list:
        self.segs.append({"t": "</assistant>\n", "loss": 0})
        shutil.rmtree(self.dir, ignore_errors=True)
        return self.segs


def _src(lines):
    return "\n".join(lines) + "\n"


def bug_fix(rng, lang):
    task_en, task_uz, lines, bug, new, why_en, why_uz = rng.choice(_bugs(rng))
    f = rng.choice(FILES)
    user = "Write %s." % task_en if lang == "en" else "%s yozib ber." % (task_uz[0].upper() + task_uz[1:])
    b = Builder(user)
    b.plan(rng.choice(PLAN[lang]))
    b.call("write", f, _src(lines), trained=False)
    first = b.call("shell", "python3 %s" % f, trained=False)
    assert first and ("Error" in first), first
    b.call("read", "%s, 1-%d" % (f, len(lines)))
    b.call("edit", "%s, %d-%d" % (f, bug, bug), _src(new))
    last = b.call("shell", "python3 %s" % f)
    assert last is not None and "Traceback" not in last and "Error" not in last, last
    n = bug
    b.say(("The problem: %s (line %d). It runs now." % (why_en, n)) if lang == "en" else "Muammo: %s (%d-qator). Endi ishlaydi." % (why_uz, n))
    return b.finish()


def edit_range(rng, lang):
    lines, _, _, (idx, new, req_en, req_uz) = rng.choice(_setups(rng))
    f = rng.choice(FILES)
    b = Builder(("In %s, %s." % (f, req_en)) if lang == "en" else "%s ichida %s." % (f, req_uz), {f: _src(lines)})
    bad = rng.choice([(121, 125), (40, 42), (30, 31), (99, 99)])
    if rng.random() < 0.5:
        b.plan(rng.choice(LOOK[lang]))
        b.call("read", "%s, 1-%d" % (f, len(lines)))
        b.call("edit", "%s, %d-%d" % (f, bad[0], bad[1]), _src([new]), trained=False)
    else:
        b.plan(rng.choice(EDIT[lang]))
        b.call("edit", "%s, %d-%d" % (f, bad[0], bad[1]), _src([new]), trained=False)
        b.call("read", "%s, 1-%d" % (f, len(lines)))
    b.call("edit", "%s, %d-%d" % (f, idx, idx), _src([new]))
    out = b.call("shell", "python3 %s" % f)
    assert out is not None and "Error" not in out
    b.say(("Changed line %d in %s. It still runs." % (idx, f)) if lang == "en" else "%s faylda %d-qator o'zgardi. Hali ham ishlaydi." % (f, idx))
    return b.finish()


def read_range(rng, lang):
    lines, what_en, what_uz, _ = rng.choice(_setups(rng))
    f = rng.choice(FILES)
    b = Builder(("What does %s do?" % f) if lang == "en" else "%s nima qiladi?" % f, {f: _src(lines)})
    b.plan("I will read the file." if lang == "en" else "Faylni o'qiyman.")
    start = rng.choice([121, 51, 200])
    b.call("read", "%s, %d-%d" % (f, start, start + 29), trained=False)
    b.call("read", "%s, 1-%d" % (f, len(lines)))
    b.say(("It %s." % what_en) if lang == "en" else "U %s." % what_uz)
    return b.finish()


def missing_file(rng, lang):
    lines, what_en, what_uz, _ = rng.choice(_setups(rng))
    real, guess = rng.sample(FILES, 2)
    b = Builder("Run my script." if lang == "en" else "Skriptimni ishga tushir.", {real: _src(lines)})
    b.plan(rng.choice(RUN[lang]))
    if rng.random() < 0.5:
        b.call("shell", "python3 %s" % guess, trained=False)
    else:
        b.call("read", "%s, 1-20" % guess, trained=False)
    listing = b.call("shell", "ls")
    assert real in listing
    out = b.call("shell", "python3 %s" % real)
    assert out is not None and "Error" not in out
    b.say(("There was no %s, so I ran %s instead. It %s." % (guess, real, what_en)) if lang == "en"
          else "%s degan fayl yo'q ekan, %s ni ishga tushirdim. U %s." % (guess, real, what_uz))
    return b.finish()


def unknown_tool(rng, lang):
    lines, what_en, what_uz, _ = rng.choice(_setups(rng))
    f = rng.choice(FILES)
    which = rng.choice(["list", "run", "show"])
    user = {"list": ("List the files.", "Fayllar ro'yxatini ko'rsat."), "run": ("Run %s." % f, "%s ni ishga tushir." % f), "show": ("Show me %s." % f, "%s ni ko'rsat." % f)}[which]
    b = Builder(user[0] if lang == "en" else user[1], {f: _src(lines)})
    b.plan({"list": "I will list the files.", "run": "I will run it.", "show": "I will read it."}[which] if lang == "en"
           else {"list": "Fayllarni ko'raman.", "run": "Ishga tushiraman.", "show": "Faylni o'qiyman."}[which])
    wrong = {"list": ("bash", "ls"), "run": ("run", "python3 %s" % f), "show": ("cat", f)}[which]
    b.call(wrong[0], wrong[1], trained=False)
    if which == "show":
        b.call("read", "%s, 1-%d" % (f, len(lines)))
    else:
        b.call("shell", wrong[1])
    b.say(("Done. I only have the tools read, write, edit, shell and search." if lang == "en" else "Tayyor. Menda faqat read, write, edit, shell va search bor."))
    return b.finish()


def read_no_range(rng, lang):
    lines, what_en, what_uz, _ = rng.choice(_setups(rng))
    f = rng.choice(FILES)
    b = Builder(("Read %s for me." % f) if lang == "en" else "%s ni o'qib ber." % f, {f: _src(lines)})
    b.plan("I will read the file." if lang == "en" else "Faylni o'qiyman.")
    b.call("read", f, trained=False)
    b.call("read", "%s, 1-%d" % (f, len(lines)))
    b.say(("It %s." % what_en) if lang == "en" else "U %s." % what_uz)
    return b.finish()


def blocked(rng, lang):
    if rng.random() < 0.5:
        folder = rng.choice(["build", "dist", "cache", "old", "tmp", "out", "logs", "backup", "venv", "data", "node_modules", "target"])
        user = ("Delete the %s folder." % folder, "%s papkasini o'chir." % folder)
        cmd, say_en, say_uz = "%s %s" % (rng.choice(["rm -rf", "rm -r"]), folder), "I am not allowed to delete folders. Please delete %s yourself." % folder, "Men papkalarni o'chira olmayman. %s ni o'zingiz o'chiring." % folder
    else:
        path = rng.choice(["/etc/hosts", "/etc/passwd", "/var/log/syslog", "/usr/bin/env", "/root/notes.txt", "/home/user/a.txt"])
        user = ("Show me %s." % path, "%s ni ko'rsat." % path)
        cmd, say_en, say_uz = "cat %s" % path, "I can only open files inside the work folder, so I cannot read %s." % path, "Men faqat ish papkasidagi fayllarni ocha olaman, shuning uchun %s ni o'qiy olmayman." % path
    b = Builder(user[0] if lang == "en" else user[1])
    b.plan("I will run the command." if lang == "en" else "Buyruqni bajaraman.")
    out = b.call("shell", cmd, trained=False)
    assert out.startswith("blocked")
    b.say(say_en if lang == "en" else say_uz)
    return b.finish()


def give_up(rng, lang):
    mod = rng.choice(["fastgraph2", "pyquillz", "tablegen9", "nanoplotx", "colorbeam", "jsonmint", "datafoldr", "pixelwave", "textrank9", "mathbox2", "zipkit3", "parsley9"])
    f = rng.choice(FILES)
    lines = ["import %s" % mod, "", 'print(%s.version())' % mod]
    b = Builder(("Run a script that uses %s." % mod) if lang == "en" else "%s ishlatadigan skriptni ishga tushir." % mod)
    b.plan(rng.choice(PLAN[lang]))
    b.call("write", f, _src(lines), trained=False)
    b.call("shell", "python3 %s" % f, trained=False)
    again = b.call("shell", 'python3 -c "import %s"' % mod)
    assert "ModuleNotFoundError" in again
    b.say(("It needs the package %s, which is not installed here, and I cannot install packages. Install it with: pip install %s. Then run: python3 %s" % (mod, mod, f)) if lang == "en"
          else "Bu %s paketini talab qiladi, u o'rnatilmagan, men paket o'rnata olmayman. Buni o'zingiz bajaring: pip install %s. Keyin: python3 %s" % (mod, mod, f))
    return b.finish()


MAKERS = {"bug_fix": bug_fix, "edit_range": edit_range, "read_range": read_range, "missing_file": missing_file,
          "unknown_tool": unknown_tool, "read_no_range": read_no_range, "blocked": blocked, "give_up": give_up}


def build_all(per_kind: int = 100, seed: int = 0) -> list:
    rng = random.Random(seed)
    out = []
    for kind in KINDS:
        seen = set()
        for i in range(per_kind):
            lang = "en" if i % 2 == 0 else "uz"
            for _ in range(60):                      # a repeated text teaches nothing, so draw again
                segs = MAKERS[kind](rng, lang)
                text = "".join(x["t"] for x in segs)
                if text not in seen:
                    break
            else:
                continue
            seen.add(text)
            out.append({"id": "E%s%04d" % (kind[:2].upper(), len(out)), "kind": kind, "lang": lang, "segments": segs})
    return out


def to_text(example: dict) -> str:
    return "".join(s["t"] for s in example["segments"])


def main() -> int:
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    out = sys.argv[2] if len(sys.argv) > 2 else "error_traces.jsonl"
    data = build_all(per, seed=11)
    with open(out, "w", encoding="utf-8") as f:
        for e in data:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print("wrote %d examples to %s" % (len(data), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
