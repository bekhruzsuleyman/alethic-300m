"""Plain number questions -> ask python -> answer with the real result, for SFT.

Why: a 300M model trained on 6B tokens cannot do 17 * 23 in its head (it said 42; the answer is 391). The cure is not more
math text but a habit: whenever a question needs a number, ask python and report what python printed. The user never
says "use python" here, like in real life, and the numbers change every time so nothing can be memorised.

Every call runs in the real Workspace; the final sentence quotes the real output; `expected` is checked against it.
"""
from __future__ import annotations

import json
import random
import sys

from tools.errtraces.build import Builder
from tools.evalset.prompts import eval_set

PY = ["python3", "python"]
EVAL_PROMPTS = {p["prompt"] for p in eval_set()}
KINDS = ["arith", "order_of_ops", "percent", "percent_change", "discount", "average", "speed", "distance", "units", "money", "geometry", "time_units", "split"]
PLAN = {"en": ["I will calculate it with python.", "I will ask python for the exact number.", "Let me compute it with python."],
        "uz": ["Python bilan hisoblayman.", "Aniq sonni python'dan so'rayman.", "Buni python yordamida hisoblayman."]}


def _ask(rng, lang, en, uz):
    return rng.choice(en if lang == "en" else uz)


def arith(rng, lang):
    a, b, c = rng.randint(12, 99), rng.randint(12, 99), rng.randint(2, 9)
    op = rng.choice(["*", "*", "+", "-"])
    e = "%d %s %d" % (a, op, b)
    q = _ask(rng, lang, ["What is %s?", "Calculate %s.", "how much is %s", "%s = ?", "Can you work out %s?", "%s"], ["%s nechaga teng?", "%s ni hisoblang.", "%s nechchi bo'ladi?", "%s = ?", "%s ni hisoblab bering"]) % e
    return q, "print(%s)" % e, "%s = {r}." % e, "%s = {r}." % e


def order_of_ops(rng, lang):
    a, b, c = rng.randint(2, 20), rng.randint(2, 12), rng.randint(2, 9)
    e = rng.choice(["%d + %d * %d" % (a, b, c), "(%d + %d) * %d" % (a, b, c), "%d * %d - %d" % (a, b, c), "%d - %d * %d + %d" % (a * b, b, c, a)])
    q = _ask(rng, lang, ["What is %s?", "Solve %s.", "Evaluate %s, careful with the order."], ["%s nechaga teng?", "%s ni yeching.", "%s ni hisoblang, tartibga e'tibor bering."]) % e
    return q, "print(%s)" % e, "%s = {r}." % e, "%s = {r}." % e


def percent(rng, lang):
    p = rng.choice([5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 60, 75])
    n = rng.choice([40, 60, 80, 120, 150, 200, 240, 300, 360, 450, 500, 640, 800, 960])
    code = "print(%d * %d // 100)" % (n, p) if n * p % 100 == 0 else "print(format(%d * %d / 100, 'g'))" % (n, p)
    q = _ask(rng, lang, ["What is %d%% of %d?", "Find %d%% of %d.", "how much is %d percent of %d", "Calculate %d%% of %d."],
             ["%d ning %d foizi nechaga teng?" , "%d ning %d foizini toping.", "%d ning %d foizi qancha?"])
    q = q % (p, n) if lang == "en" else q % (n, p)
    return q, code, "%d%% of %d is {r}." % (p, n), "%d ning %d foizi {r} ga teng." % (n, p)


def percent_change(rng, lang):
    a = rng.randint(15, 300)
    b = a + rng.choice([-1, 1]) * rng.randint(3, 60)
    b = max(b, 5)
    q = _ask(rng, lang, ["A price goes from %d to %d. What is the percent change?", "The number of users went from %d to %d. By what percent did it change?",
                         "What is the percentage change from %d to %d?"],
             ["Narx %d dan %d ga o'zgardi. Necha foizga o'zgardi?", "Foydalanuvchilar soni %d dan %d ga o'zgardi. Necha foiz o'zgardi?", "%d dan %d gacha necha foiz o'zgarish bor?"]) % (a, b)
    return q, "print(format(round((%d - %d) / %d * 100, 2), 'g'))" % (b, a, a), "From %d to %d the change is {r}%%." % (a, b), "%d dan %d gacha o'zgarish {r}%%." % (a, b)


def discount(rng, lang):
    price = rng.randrange(20, 600, 5)
    p = rng.choice([10, 15, 20, 25, 30, 40, 50])
    item = rng.choice(["jacket", "backpack", "keyboard", "lamp", "watch"])
    uz_item = {"jacket": "kurtka", "backpack": "ryukzak", "keyboard": "klaviatura", "lamp": "chiroq", "watch": "soat"}[item]
    q = _ask(rng, lang, ["A %s costs $%d and is %d%% off. What is the sale price?", "A $%d %s has a %d%% discount. How much does it cost now?"],
             ["%s narxi $%d, %d%% chegirma bor. Chegirmadan keyingi narxi qancha?", "$%d turadigan %s %d%% arzonlashdi. Endi qancha turadi?"])
    q = q % ((item, price, p) if lang == "en" and q.startswith("A %s") else (price, item, p)) if lang == "en" else (q % (uz_item.capitalize(), price, p) if q.startswith("%s") else q % (price, uz_item, p))
    return q, "print(format(%d * (100 - %d) / 100, '.2f'))" % (price, p), "The sale price is ${r}.", "Chegirmadan keyingi narx: ${r}."


def average(rng, lang):
    xs = [rng.randint(3, 99) for _ in range(rng.randint(3, 6))]
    s = ", ".join(map(str, xs))
    q = _ask(rng, lang, ["What is the average of %s?", "Find the mean of %s.", "Average of these numbers: %s"], ["%s sonlarining o'rtachasi nechaga teng?", "%s sonlarining o'rtacha qiymatini toping."]) % s
    return q, "print(format(sum([%s]) / %d, 'g'))" % (s, len(xs)), "The average of %s is {r}." % s, "%s sonlarining o'rtachasi {r}." % s


def speed(rng, lang):
    a, b = rng.randint(20, 110), rng.randint(20, 110)
    thing_en, thing_uz = rng.choice([("car", "mashina"), ("bus", "avtobus"), ("cyclist", "velosipedchi"), ("train", "poyezd")])
    expl = rng.random() < 0.4
    q = _ask(rng, lang, ["A %s travels %d km in the first hour and %d km in the second hour. What is its average speed over the two hours?%s",
                         "A %s drives %d km in hour one and %d km in hour two. What was the average speed?%s"],
             ["%s birinchi soatda %d km, ikkinchi soatda %d km yurdi. Ikki soatdagi o'rtacha tezligi qancha?%s", "%s 1-soatda %d km, 2-soatda %d km yo'l bosdi. O'rtacha tezlik qancha?%s"])
    q = q % ((thing_en if lang == "en" else thing_uz.capitalize()), a, b, " Explain your calculation." if expl and lang == "en" else (" Hisobni tushuntiring." if expl else ""))
    return q, "print(format((%d + %d) / 2, 'g'))" % (a, b), "Average speed = (%d + %d) km / 2 h = {r} km/h." % (a, b), "O'rtacha tezlik = (%d + %d) km / 2 soat = {r} km/soat." % (a, b)


def distance(rng, lang):
    v, t = rng.randint(30, 130), rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 1.5, 2.5, 3.5, 4.5])
    q = _ask(rng, lang, ["A train goes %d km/h for %s hours. How far does it travel?", "How many kilometers do you cover at %d km/h in %s hours?"],
             ["Poyezd %d km/soat tezlik bilan %s soat yurdi. Qancha masofa bosdi?", "%d km/soat tezlikda %s soatda necha kilometr yuriladi?"]) % (v, t)
    return q, "print(format(%d * %s, 'g'))" % (v, t), "Distance = %d km/h x %s h = {r} km." % (v, t), "Masofa = %d km/soat x %s soat = {r} km." % (v, t)


def units(rng, lang):
    kind = rng.choice(["km_mi", "kg_lb", "c_f", "h_min", "m_cm"])
    n = rng.randint(3, 250) if kind != "h_min" else rng.randint(2, 48)
    table = {
        "km_mi": ("print(format(%d * 0.621371, '.2f'))", ["Convert %d km to miles.", "How many miles is %d km?"], ["%d kilometr necha mil?", "%d km ni milga aylantiring."], "%d km = {r} miles.", "%d km = {r} mil."),
        "kg_lb": ("print(format(%d * 2.20462, '.2f'))", ["Convert %d kg to pounds.", "How many pounds is %d kg?"], ["%d kilogramm necha funt?", "%d kg ni funtga aylantiring."], "%d kg = {r} pounds.", "%d kg = {r} funt."),
        "c_f": ("print(format(%d * 9 / 5 + 32, 'g'))", ["Convert %d degrees Celsius to Fahrenheit.", "What is %d C in Fahrenheit?"], ["%d daraja Selsiy necha daraja Farengeyt?", "%d C ni Farengeytga aylantiring."], "%d C = {r} F.", "%d C = {r} F."),
        "h_min": ("print(%d * 60)", ["How many minutes are in %d hours?", "Convert %d hours to minutes."], ["%d soat necha daqiqa?", "%d soatni daqiqaga aylantiring."], "%d hours = {r} minutes.", "%d soat = {r} daqiqa."),
        "m_cm": ("print(%d * 100)", ["Convert %d meters to centimeters.", "How many centimeters are in %d m?"], ["%d metr necha santimetr?", "%d m ni santimetrga aylantiring."], "%d m = {r} cm.", "%d m = {r} sm.")}
    code, en, uz, ea, ua = table[kind]
    return (rng.choice(en if lang == "en" else uz)) % n, code % n, ea % n, ua % n


def money(rng, lang):
    q_n, price, tax = rng.randint(2, 12), rng.choice([2, 3, 4, 5, 8, 12, 15]), rng.choice([5, 8, 10, 12, 20])
    item_en, item_uz = rng.choice([("notebooks", "daftar"), ("pens", "ruchka"), ("coffees", "kofe"), ("tickets", "chipta")])
    q = _ask(rng, lang, ["%d %s cost $%d each and the tax is %d%%. What is the total?", "I buy %d %s at $%d each, plus %d%% tax. How much do I pay in total?"],
             ["%d ta %s, har biri $%d, soliq %d%%. Jami qancha to'lanadi?", "Har biri $%d dan %d ta %s olsam va %d%% soliq bo'lsa, jami qancha to'laymiz?"])
    if lang == "en":
        q = q % (q_n, item_en, price, tax)
    elif q.startswith("%d ta"):
        q = q % (q_n, item_uz, price, tax)
    else:
        q = q % (price, q_n, item_uz, tax)
    return q, "print(format(%d * %d * (100 + %d) / 100, '.2f'))" % (q_n, price, tax), "The total with tax is ${r}.", "Soliq bilan jami: ${r}."


def geometry(rng, lang):
    kind = rng.choice(["rect_area", "rect_perimeter", "circle", "triangle"])
    a, b = rng.randint(3, 30), rng.randint(3, 30)
    r = rng.randint(2, 15)
    table = {
        "rect_area": ("A rectangle is %d m by %d m. What is its area?" % (a, b), "To'rtburchak %d m ga %d m. Yuzi qancha?" % (a, b), "print(%d * %d)" % (a, b), "Area = %d x %d = {r} square meters." % (a, b), "Yuzi = %d x %d = {r} kvadrat metr." % (a, b)),
        "rect_perimeter": ("What is the perimeter of a %d by %d rectangle?" % (a, b), "%d ga %d o'lchamli to'rtburchakning perimetri qancha?" % (a, b), "print(2 * (%d + %d))" % (a, b), "Perimeter = 2 x (%d + %d) = {r}." % (a, b), "Perimetr = 2 x (%d + %d) = {r}." % (a, b)),
        "circle": ("What is the area of a circle with radius %d?" % r, "Radiusi %d bo'lgan doiraning yuzi qancha?" % r, "print(format(3.14159 * %d ** 2, '.2f'))" % r, "Area = pi x %d^2, with pi = 3.14159, is {r}." % r, "Yuzi = pi x %d^2 (pi = 3.14159) = {r}." % r),
        "triangle": ("A triangle has base %d and height %d. What is its area?" % (a, b), "Uchburchak asosi %d, balandligi %d. Yuzi qancha?" % (a, b), "print(format(%d * %d / 2, 'g'))" % (a, b), "Area = %d x %d / 2 = {r}." % (a, b), "Yuzi = %d x %d / 2 = {r}." % (a, b))}
    en, uz, code, ea, ua = table[kind]
    return (en if lang == "en" else uz), code, ea, ua


def time_units(rng, lang):
    kind = rng.choice(["weeks", "hours", "minutes", "days_years"])
    n = rng.randint(2, 99)
    table = {"weeks": ("How many days are in %d weeks?", "%d haftada necha kun bor?", "print(%d * 7)", "%d weeks have {r} days.", "%d haftada {r} kun bor."),
             "hours": ("How many minutes are in %d hours?", "%d soatda necha daqiqa bor?", "print(%d * 60)", "%d hours have {r} minutes.", "%d soatda {r} daqiqa bor."),
             "minutes": ("How many seconds are in %d minutes?", "%d daqiqada necha soniya bor?", "print(%d * 60)", "%d minutes have {r} seconds.", "%d daqiqada {r} soniya bor."),
             "days_years": ("How many hours are in %d days?", "%d kunda necha soat bor?", "print(%d * 24)", "%d days have {r} hours.", "%d kunda {r} soat bor.")}
    en, uz, code, ea, ua = table[kind]
    return ((en if lang == "en" else uz) % n), code % n, ea % n, ua % n


def split(rng, lang):
    friends = rng.randint(2, 9)
    each = rng.randint(4, 40)
    total = friends * each + rng.choice([0, 0, 0, rng.randint(1, friends - 1)])
    q = _ask(rng, lang, ["The bill is $%d. We split it equally among %d friends. How much does each pay?", "Split $%d between %d people. How much per person?"],
             ["Hisob $%d. Uni %d do'st teng bo'lib to'laydi. Har biri qancha to'laydi?", "$%d ni %d kishi o'rtasida teng bo'ling. Har biriga qancha tushadi?"]) % (total, friends)
    return q, "print(format(%d / %d, '.2f'))" % (total, friends), "Each person pays ${r}.", "Har biri ${r} to'laydi."


MAKERS = {"arith": arith, "order_of_ops": order_of_ops, "percent": percent, "percent_change": percent_change, "discount": discount, "average": average,
          "speed": speed, "distance": distance, "units": units, "money": money, "geometry": geometry, "time_units": time_units, "split": split}
DEFAULT_COUNTS = {"arith": 330, "order_of_ops": 220, "percent": 270, "percent_change": 160, "discount": 160, "average": 160, "speed": 210, "distance": 130,
                  "units": 210, "money": 160, "geometry": 190, "time_units": 130, "split": 130}


FILL = {"en": ["quick one: ", "hey, ", "btw ", "pls ", "ok so ", "hmm, ", "", "", ""],
        "uz": ["tezroq: ", "salom, ", "aytchi, ", "iltimos ", "mayli, ", "", "", ""]}
BARE = {"en": ["{r}", "{r}.", "= {r}", "Answer: {r}", "It's {r}.", "That's {r}.", "Result: {r}", "I get {r}.", "{r}, from the calculation above."],
        "uz": ["{r}", "{r}.", "= {r}", "Javob: {r}", "{r} chiqdi.", "Natija: {r}", "Hisobladim: {r}.", "{r} ga teng."]}
WORKED = {"en": ["%(e)s = {r}.", "I ran %(e)s and got {r}.", "%(e)s comes to {r}.", "Python says %(e)s = {r}. %(full)s"],
          "uz": ["%(e)s = {r}.", "%(e)s ni hisobladim: {r}.", "Python: %(e)s = {r}. %(full)s"]}


def _typo(rng, text):
    words = text.split(" ")
    ok = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    if ok:
        i = rng.choice(ok)
        w = words[i]
        j = rng.randrange(1, len(w) - 2)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def noisy(rng, q, lang):
    r = rng.random()
    if r < 0.5:
        return q
    if r < 0.65:
        return (rng.choice(FILL[lang]) + q[0].lower() + q[1:]).rstrip("?.")
    if r < 0.8:
        return q.lower().rstrip("?.")
    if r < 0.9:
        return _typo(rng, q)
    return rng.choice(FILL[lang]) + q


def one(rng, kind, lang):
    q, code, en_ans, uz_ans = MAKERS[kind](rng, lang)
    inner = code[len("print("):-1]
    code = code if rng.random() < 0.75 else "x = %s; print(x)" % inner
    shown_q = noisy(rng, q, lang)
    b = Builder(shown_q)
    b.plan(rng.choice(PLAN[lang]))
    out = b.call("shell", '%s -c "%s"' % (rng.choice(PY), code))
    assert out and "Error" not in out and "\n" not in out, (code, out)
    full = en_ans if lang == "en" else uz_ans
    r = rng.random()
    if r < 0.40:
        text = full
    elif r < 0.65:
        text = rng.choice(BARE[lang])
    else:
        text = rng.choice(WORKED[lang]) % {"e": inner.replace("format(", "(") if "format(" in inner else inner, "full": full}
    b.say(text.replace("{r}", out))
    return b.finish(), q


def build_all(per_kind=60, seed: int = 0, counts: dict | None = None) -> list:
    rng = random.Random(seed)
    out = []
    for kind in KINDS:
        want = (counts or {}).get(kind, per_kind)
        seen = set()
        for i in range(want):
            lang = "en" if i % 2 == 0 else "uz"
            for _ in range(80):
                segs, q = one(rng, kind, lang)
                text = "".join(x["t"] for x in segs)
                if text not in seen and q not in EVAL_PROMPTS:
                    break
            else:
                continue
            seen.add(text)
            out.append({"id": "M%s%04d" % (kind[:2].upper(), len(out)), "kind": kind, "lang": lang, "segments": segs})
    return out


def to_text(example: dict) -> str:
    return "".join(s["t"] for s in example["segments"])


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "math_traces.jsonl"
    data = build_all(seed=31, counts=DEFAULT_COUNTS)
    with open(path, "w", encoding="utf-8") as f:
        for e in data:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print("wrote %d examples to %s" % (len(data), path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
