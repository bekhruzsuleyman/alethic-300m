"""Scripted "do the task for real" traces for SFT: write a program and run it, ask python for a number, list files,
make a note, edit the right line of a longer file.

Why: the gold data has about 34 write-and-run examples, so the model replays one memorised answer. Here every request
uses different numbers, words and files, the code must depend on them, every call is run by the REAL Workspace, and the
final sentence is built from the real output, so it can never claim something the result does not show.

Records: {"id", "kind", "lang", "segments": [{"t", "loss"}]} like the other built SFT files. Nothing here is a failure,
so every call the model makes is trained on; results and the user's text are not.
"""
from __future__ import annotations

import json
import random
import sys

from tools.errtraces.build import PLAN, Builder, _src

KINDS = ["write_run", "quick_calc", "list_files", "make_note", "edit_pos"]
WORDS = ["garden", "window", "planet", "bridge", "forest", "pencil", "silver", "market", "orange", "winter", "harbor", "lantern",
         "canvas", "meadow", "puzzle", "rocket", "violin", "castle", "thunder", "pepper"]
SENTENCES = ["the quick brown fox jumps", "learning to code is fun", "we ship small and test often", "rain falls on the old roof"]
NAMES = ["LIMIT", "SPEED", "PRICE", "WIDTH", "HEIGHT", "COUNT", "TAX", "SIZE", "DELAY", "RATE", "SCORE", "LEVEL", "DEPTH", "GAP", "STEP", "BONUS", "WEIGHT", "ANGLE"]
HEADERS = ["# settings", "# values used below", "# constants", "# do not change the order", "# numbers for the report", "# config"]
FILLERS = ["# note: units are centimetres", "# see the README for details", "# checked on monday", "# keep this short", "# unused for now", "# TODO later"]
PY = ["python", "python3"]


# --- programs: (what it does in English, in Uzbek, the file names it may get, the code lines) -------------------


def _programs(rng):
    a = rng.randint(1, 9)
    b = a + rng.randint(3, 12)
    n = rng.randint(4, 15)
    m = rng.randint(2, 12)
    w, w2 = rng.sample(WORDS, 2)
    xs = rng.sample(range(2, 99), rng.randint(4, 6))
    sent = rng.choice(SENTENCES)
    return [
        ("prints the numbers %d to %d, one per line" % (a, b), "%d dan %d gacha sonlarni har birini alohida qatorda chiqaradi" % (a, b), ["count.py", "numbers.py", "range.py"],
         ["start = %d" % a, "end = %d" % b, "for i in range(start, end + 1):", "    print(i)"]),
        ("prints the sum of the numbers 1 to %d" % n, "1 dan %d gacha sonlar yig'indisini chiqaradi" % n, ["sum.py", "total.py", "adder.py"],
         ["n = %d" % n, "print(sum(range(1, n + 1)))"]),
        ("prints the squares of the numbers 1 to %d" % n, "1 dan %d gacha sonlarning kvadratlarini chiqaradi" % n, ["squares.py", "powers.py", "sq.py"],
         ["n = %d" % n, "for i in range(1, n + 1):", "    print(i * i)"]),
        ("prints the multiplication table of %d from 1 to 5" % m, "%d ning 1 dan 5 gacha ko'paytirish jadvalini chiqaradi" % m, ["table.py", "times.py", "mult.py"],
         ["m = %d" % m, "upto = 5", "for i in range(1, upto + 1):", '    print(m, "x", i, "=", m * i)']),
        ('prints the word "%s" reversed' % w, '"%s" so\'zini teskari chiqaradi' % w, ["rev.py", "reverse.py", "flip.py"],
         ['word = "%s"' % w, "print(word[::-1])"]),
        ('prints the word "%s" in capital letters' % w, '"%s" so\'zini bosh harflarda chiqaradi' % w, ["upper.py", "caps.py", "shout.py"],
         ['word = "%s"' % w, "print(word.upper())"]),
        ('prints how many letters the word "%s" has' % w, '"%s" so\'zida nechta harf borligini chiqaradi' % w, ["length.py", "letters.py", "size.py"],
         ['word = "%s"' % w, "print(len(word))"]),
        ('prints how many vowels the word "%s" has' % w2, '"%s" so\'zida nechta unli harf borligini chiqaradi' % w2, ["vowels.py", "count_vowels.py", "vow.py"],
         ['word = "%s"' % w2, "count = 0", "for ch in word:", '    if ch in "aeiou":', "        count += 1", "print(count)"]),
        ("prints the even numbers from 2 up to %d" % (2 * n), "2 dan %d gacha juft sonlarni chiqaradi" % (2 * n), ["even.py", "evens.py", "pairs.py"],
         ["limit = %d" % (2 * n), "for i in range(2, limit + 1, 2):", "    print(i)"]),
        ("prints the factorial of %d" % min(n, 12), "%d ning faktorialini chiqaradi" % min(n, 12), ["fact.py", "factorial.py", "fac.py"],
         ["n = %d" % min(n, 12), "result = 1", "for i in range(2, n + 1):", "    result *= i", "print(result)"]),
        ("prints the first %d Fibonacci numbers" % n, "birinchi %d ta Fibonachchi sonini chiqaradi" % n, ["fib.py", "fibonacci.py", "seq.py"],
         ["n = %d" % n, "a, b = 0, 1", "for _ in range(n):", "    print(a)", "    a, b = b, a + b"]),
        ("prints the biggest number in the list %r" % xs, "%r ro'yxatidagi eng katta sonni chiqaradi" % xs, ["biggest.py", "maxnum.py", "top.py"],
         ["numbers = %r" % xs, "print(max(numbers))"]),
        ("prints the list %r sorted from small to big" % xs, "%r ro'yxatini kichikdan kattaga saralab chiqaradi" % xs, ["sort.py", "sorted.py", "order.py"],
         ["numbers = %r" % xs, "print(sorted(numbers))"]),
        ("prints the average of the numbers %r" % xs, "%r sonlarining o'rtachasini chiqaradi" % xs, ["average.py", "mean.py", "avg.py"],
         ["numbers = %r" % xs, "print(sum(numbers) / len(numbers))"]),
        ("prints FizzBuzz for the numbers 1 to %d" % n, "1 dan %d gacha FizzBuzz chiqaradi" % n, ["fizz.py", "fizzbuzz.py", "game.py"],
         ["n = %d" % n, "for i in range(1, n + 1):", "    if i % 15 == 0:", '        print("FizzBuzz")', "    elif i % 3 == 0:", '        print("Fizz")',
          "    elif i % 5 == 0:", '        print("Buzz")', "    else:", "        print(i)"]),
        ("counts down from %d to 1 and then prints Go" % n, "%d dan 1 gacha sanaydi va oxirida Go chiqaradi" % n, ["countdown.py", "timer.py", "launch.py"],
         ["start = %d" % n, "for i in range(start, 0, -1):", "    print(i)", 'print("Go")']),
        ('prints the word "%s" %d times, one per line' % (w, min(m, 6)), '"%s" so\'zini %d marta, har birini alohida qatorda chiqaradi' % (w, min(m, 6)), ["repeat.py", "echo.py", "loop.py"],
         ['word = "%s"' % w, "times = %d" % min(m, 6), "for _ in range(times):", "    print(word)"]),
        ('prints how many words are in the sentence "%s"' % sent, '"%s" gapida nechta so\'z borligini chiqaradi' % sent, ["words.py", "wc.py", "sentence.py"],
         ['text = "%s"' % sent, "print(len(text.split()))"]),
        ("prints the cubes of the numbers 1 to %d" % min(n, 9), "1 dan %d gacha sonlarning kublarini chiqaradi" % min(n, 9), ["cubes.py", "cube.py", "pow3.py"],
         ["n = %d" % min(n, 9), "print([i ** 3 for i in range(1, n + 1)])"]),
        ("prints the numbers from 1 to %d that can be divided by %d" % (n * 3, min(m, 7) + 1), "1 dan %d gacha %d ga bo'linadigan sonlarni chiqaradi" % (n * 3, min(m, 7) + 1), ["divisible.py", "multiples.py", "step.py"],
         ["limit = %d" % (n * 3), "d = %d" % (min(m, 7) + 1), "for i in range(1, limit + 1):", "    if i % d == 0:", "        print(i)"]),
    ]


EN_ASK = ["Write {f} that {t} and run it.", "Create a file {f} that {t}, then run it.", "Make a python file called {f} that {t}, and run it.",
          "Please write a script, {f}, that {t}. Run it after.", "I need {f}: a program that {t}. Write it and run it.",
          "Write a Python program that {t}, save it as {f}, and run it.", "Save a script as {f} that {t}, then show me what it prints."]
UZ_ASK = ["{f} fayl yozing: u {t}. Keyin ishga tushiring.", "Python dasturi yozing ({f}), u {t}. Ishga tushiring.", "{f} ni yarating, u {t}, va natijasini ko'rsating.",
          "{f} nomli fayl yozib ishga tushiring, u {t}."]


def describe(out: str, lang: str) -> str:
    """Say what the output really contains: every line when it is short, the first and the last when it is long."""
    lines = out.split("\n")
    if lang == "en":
        if len(lines) == 1:
            return "It prints %s." % lines[0]
        if len(lines) <= 8:
            return "It prints %d lines: %s." % (len(lines), "; ".join(lines))
        return 'It prints %d lines, from %s to %s.' % (len(lines), lines[0], lines[-1])
    if len(lines) == 1:
        return "U %s chiqaradi." % lines[0]
    if len(lines) <= 8:
        return "U %d qator chiqaradi: %s." % (len(lines), "; ".join(lines))
    return "U %d qator chiqaradi, %s dan %s gacha." % (len(lines), lines[0], lines[-1])


def write_run(rng, lang):
    task_en, task_uz, names, code = rng.choice(_programs(rng))
    f = rng.choice(names)
    user = rng.choice(EN_ASK).format(f=f, t=task_en) if lang == "en" else rng.choice(UZ_ASK).format(f=f, t=task_uz)
    b = Builder(user)
    b.plan(rng.choice(PLAN[lang]))
    b.call("write", f, _src(code))
    out = b.call("shell", "%s %s" % (rng.choice(PY), f))
    assert out and "Error" not in out and "Traceback" not in out, out
    d = describe(out, lang)
    b.say(rng.choice(["I wrote %s and ran it. %s", "%s is written and it runs. %s", "Done, %s runs. %s"]) % (f, d) if lang == "en"
          else rng.choice(["%s ni yozib ishga tushirdim. %s", "%s tayyor va ishlaydi. %s"]) % (f, d))
    return b.finish()


# --- ask python for a number -------------------------------------------------------------------------------------


def _calcs(rng):
    x, y = rng.randint(12, 99), rng.randint(12, 99)
    p, base = rng.choice([5, 10, 12, 15, 20, 25, 30, 40]), rng.choice([80, 120, 200, 360, 450, 640, 900])
    e, k = rng.randint(2, 9), rng.randint(2, 5)
    r = rng.randint(4, 30)
    return [
        ("%d * %d" % (x, y), "%d * %d ni" % (x, y), "print(%d * %d)" % (x, y)),
        ("%d%% of %d" % (p, base), "%d ning %d foizini" % (base, p), "print(%d * %d / 100)" % (base, p)),
        ("%d to the power %d" % (e, k + 2), "%d ning %d-darajasini" % (e, k + 2), "print(%d ** %d)" % (e, k + 2)),
        ("the remainder of %d divided by %d" % (x * 3 + y, e + 1), "%d ni %d ga bo'lgandagi qoldiqni" % (x * 3 + y, e + 1), "print(%d %% %d)" % (x * 3 + y, e + 1)),
        ("the square root of %d" % (r * r), "%d ning kvadrat ildizini" % (r * r), "print(%d ** 0.5)" % (r * r)),
        ("%d + %d + %d" % (x, y, base), "%d + %d + %d ni" % (x, y, base), "print(%d + %d + %d)" % (x, y, base)),
    ]


def quick_calc(rng, lang):
    en, uz, code = rng.choice(_calcs(rng))
    user = rng.choice(["Use python to work out %s.", "Compute %s with python.", "Ask python: what is %s?"]) % en if lang == "en" \
        else rng.choice(["Python yordamida %s hisoblang.", "%s python bilan hisoblab bering."]) % uz
    b = Builder(user)
    b.plan("I will ask python." if lang == "en" else "Python'dan so'rayman.")
    out = b.call("shell", '%s -c "%s"' % (rng.choice(PY), code))
    assert out and "Error" not in out, out
    b.say(("The result is %s." % out) if lang == "en" else "Natija: %s." % out)
    return b.finish()


# --- list the folder -----------------------------------------------------------------------------------------------


def list_files(rng, lang):
    stems = rng.sample(["notes", "main", "todo", "data", "report", "draft", "plan", "tool", "demo", "readme", "budget", "ideas"], rng.randint(2, 5))
    files = {s + rng.choice([".py", ".txt", ".md", ".json"]): "x = 1\n" for s in stems}
    user = rng.choice(["What files are in this folder?", "List the files here.", "Show me what is in this folder.", "Which files do I have here?"]) if lang == "en" \
        else rng.choice(["Bu papkada qanday fayllar bor?", "Fayllar ro'yxatini ko'rsating.", "Shu yerda qaysi fayllar bor?"])
    b = Builder(user, files)
    b.plan("I will list the files." if lang == "en" else "Fayllarni ko'raman.")
    out = b.call("shell", "ls")
    names = out.split("\n")
    b.say(("The folder has %d files: %s." % (len(names), ", ".join(names))) if lang == "en" else "Papkada %d ta fayl bor: %s." % (len(names), ", ".join(names)))
    return b.finish()


# --- make a note -----------------------------------------------------------------------------------------------------

NOTES = {
    "rain": (["Rain taps on the window.", "The street smells fresh.", "Puddles shine under the lamps."], ["Yomg'ir derazaga tomchilaydi.", "Ko'cha toza hid taratadi.", "Ko'lmaklar chiroq ostida yaltiraydi."]),
    "tea": (["Tea is hot and green.", "Steam rises from the cup.", "A slow sip ends the day."], ["Choy issiq va ko'k.", "Piyoladan bug' ko'tariladi.", "Sekin ho'plam kunni yakunlaydi."]),
    "books": (["A book is a quiet friend.", "Pages smell like old wood.", "One more chapter before sleep."], ["Kitob jim do'st.", "Sahifalar eski yog'och hidini beradi.", "Uxlashdan oldin yana bir bob."]),
    "code": (["Code starts as an idea.", "Tests keep it honest.", "Small steps make it grow."], ["Kod g'oyadan boshlanadi.", "Testlar uni halol saqlaydi.", "Kichik qadamlar uni o'stiradi."]),
    "the sea": (["The sea is wide and blue.", "Waves repeat the same song.", "Gulls circle above the pier."], ["Dengiz keng va ko'k.", "To'lqinlar bir xil qo'shiq aytadi.", "Chayqalar iskala ustida aylanadi."]),
    "snow": (["Snow covers the roofs.", "Footsteps squeak on the path.", "The air is cold and clear."], ["Qor tomlarni qoplaydi.", "Qadamlar yo'lda g'ijirlaydi.", "Havo sovuq va toza."]),
    "bread": (["Bread is warm from the oven.", "The crust crackles when cut.", "Butter melts on the first slice."], ["Non tandirdan issiq.", "Qobig'i kesilganda qirsillaydi.", "Sariyog' birinchi bo'lakda eriydi."]),
    "the moon": (["The moon is bright tonight.", "Its light is soft and silver.", "The night feels calm."], ["Bugun oy yorqin.", "Nuri yumshoq va kumushsimon.", "Tun sokin tuyuladi."]),
}


def make_note(rng, lang):
    topic = rng.choice(list(NOTES))
    f = rng.choice(["notes.txt", "note.txt", "ideas.txt", "memo.txt", "poem.txt", "draft.txt"])
    en, uz = NOTES[topic]
    lines = list(en if lang == "en" else uz)
    rng.shuffle(lines)
    user = rng.choice(["Create %s with three short lines about %s, then show it.", "Write three lines about %s into %s and print the file."]) \
        if lang == "en" else rng.choice(["%s fayl yarating, unga %s haqida uchta qisqa qator yozing va ko'rsating."])
    user = (user % (f, topic) if user.startswith("Create") else user % (topic, f)) if lang == "en" else user % (f, topic)
    b = Builder(user)
    b.plan("I will write the file and show it." if lang == "en" else "Faylni yozaman va ko'rsataman.")
    b.call("write", f, _src(lines))
    out = b.call("shell", "cat %s" % f)
    got = out.split("\n")
    assert got == lines, (got, lines)
    b.say(('%s has %d lines. The first one is: "%s"' % (f, len(got), got[0])) if lang == "en" else '%s faylida %d qator bor. Birinchisi: "%s"' % (f, len(got), got[0]))
    return b.finish()


# --- edit the right line of a longer file -----------------------------------------------------------------------------


def edit_pos(rng, lang):
    n = rng.randint(8, 40)
    k = rng.randint(3, min(6, n - 3))
    spots = sorted(rng.sample(range(2, n), k))                      # line numbers 2..n-1 hold the constants
    names = rng.sample(NAMES, k)
    vals = [rng.randint(2, 60) for _ in range(k)]
    lines = []
    for i in range(1, n):
        if i in spots:
            j = spots.index(i)
            lines.append("%s = %d" % (names[j], vals[j]))
        elif i == 1:
            lines.append(rng.choice(HEADERS))
        else:
            lines.append(rng.choice(["", "", rng.choice(FILLERS)]))
    lines.append("print(%s)" % " + ".join(names))
    f = rng.choice(["main.py", "config.py", "calc.py", "report.py", "settings.py", "run.py", "demo.py", "score.py"])
    t = rng.randrange(k)
    new = rng.randint(2, 60)
    while new == vals[t]:
        new = rng.randint(2, 60)
    user = rng.choice(["In %s, set %s to %d.", "In %s, change %s to %d.", "Open %s and make %s equal %d."]) % (f, names[t], new) if lang == "en" \
        else rng.choice(["%s ichida %s ni %d qiling.", "%s faylida %s qiymatini %d ga o'zgartiring."]) % (f, names[t], new)
    b = Builder(user, {f: _src(lines)})
    b.plan(rng.choice(["I will read the file, then fix the line.", "First I read the file to see the real lines."]) if lang == "en"
           else rng.choice(["Avval faylni o'qiyman, keyin qatorni to'g'rilayman.", "Avval fayldagi haqiqiy qatorlarni ko'raman."]))
    shown = b.call("read", "%s, 1-100" % f)
    at = spots[t]
    assert shown.split("\n")[at - 1].split(None, 1)[1].startswith("%s = " % names[t]), shown
    b.call("edit", "%s, %d-%d" % (f, at, at), _src(["%s = %d" % (names[t], new)]))
    out = b.call("shell", "%s %s" % (rng.choice(PY), f))
    assert out and "Error" not in out, out
    b.say(("Changed line %d: %s is now %d. It prints %s." % (at, names[t], new, out)) if lang == "en"
          else "%d-qator o'zgardi: %s endi %d. U %s chiqaradi." % (at, names[t], new, out))
    return b.finish()


MAKERS = {"write_run": write_run, "quick_calc": quick_calc, "list_files": list_files, "make_note": make_note, "edit_pos": edit_pos}
DEFAULT_COUNTS = {"write_run": 700, "quick_calc": 200, "list_files": 100, "make_note": 120, "edit_pos": 600}


def build_all(per_kind=100, seed: int = 0, counts: dict | None = None) -> list:
    rng = random.Random(seed)
    out = []
    for kind in KINDS:
        want = (counts or {}).get(kind, per_kind)
        seen = set()
        for i in range(want):
            lang = "en" if i % 2 == 0 else "uz"
            for _ in range(60):                      # a repeated text teaches nothing, so draw again
                segs = MAKERS[kind](rng, lang)
                text = "".join(x["t"] for x in segs)
                if text not in seen:
                    break
            else:
                continue
            seen.add(text)
            out.append({"id": "G%s%04d" % (kind[:2].upper(), len(out)), "kind": kind, "lang": lang, "segments": segs})
    return out


def to_text(example: dict) -> str:
    return "".join(s["t"] for s in example["segments"])


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "prog_traces.jsonl"
    data = build_all(seed=23, counts=DEFAULT_COUNTS)
    with open(out, "w", encoding="utf-8") as f:
        for e in data:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print("wrote %d examples to %s" % (len(data), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
