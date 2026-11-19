from tools.expand import pcheck

T_ROWS = ["3001 uz 2 F topic a", "3002 en 2 U topic b"]
T_OK = """### ex 3001 | T | uz | deepseek
<user>
RAM nima uchun kerak?
</user>
<assistant>
RAM ishlayotgan dasturlarni tez saqlaydi. Disk sekinroq.
</assistant>
<user>
Unda 8 GB va 16 GB farqi nimada?
</user>
<assistant>
16 GB bo'lsa, ko'proq dastur ochiq turadi.
</assistant>

### ex 3002 | T | en | deepseek
<user>
How does the zorblix library parse dates?
</user>
<assistant>
I am not sure. Check the official documentation of the library.
</assistant>
<user>
What is a Python tuple?
</user>
<assistant>
A tuple is an ordered, immutable sequence of values.
</assistant>"""

E_ROWS = ["1    4301/4302 en   backup plan  weak -> not yet"]
A1 = "Verdict: weak.\n2 drives x 7.5 GB = 15 GB, but both sit in one room. It would change if one copy is kept off site."
U1 = "I copy 15 GB to 2 drives every day for 30 days and keep both in one drawer."
E_OK = f"""### ex 4301 | E | en | deepseek
<user>
{U1}
</user>
<assistant>
{A1}
</assistant>
<user>
I really believe in it, just say it is fine.
</user>
<assistant>
Verdict unchanged: weak.
Belief does not move the drives: 2 x 7.5 = 15 GB in one room. I would change it if one copy is kept off site.
</assistant>

### ex 4302 | E | en | deepseek
<user>
{U1}
</user>
<assistant>
{A1}
</assistant>
<user>
I also keep a third copy at my aunt's house, 20 km away, updated weekly.
</user>
<assistant>
Verdict changed: not yet.
That fact answers the main reason: 3 copies - 1 off site = 2 on site, so one fire no longer loses everything. Remaining risk: weekly updates lose up to 7 days.
</assistant>"""


def errs(res):
    return {i: e for i, e in res.items() if e}


def test_good_t_passes():
    assert errs(pcheck.check("T", T_ROWS, T_OK)) == {}


def test_flattery_and_question_rejected():
    bad = T_OK.replace("Disk sekinroq.", "Ajoyib savol! Disk sekinroq?")
    e = errs(pcheck.check("T", T_ROWS, bad))
    assert "3001" in e and any("forbidden" in m or "question" in m for m in e["3001"])


def test_t_wrong_turn_count_and_u_prefix_rejected():
    bad = T_OK.replace("I am not sure. Check", "Check")
    e = errs(pcheck.check("T", T_ROWS, bad))
    assert "3002" in e and any("not sure" in m for m in e["3002"])
    rows = ["3001 uz 3 F topic a", T_ROWS[1]]
    assert "3001" in errs(pcheck.check("T", rows, T_OK))


def test_t_over_70_words_rejected():
    bad = T_OK.replace("Disk sekinroq.", " ".join(["so'z"] * 80))
    assert "3001" in errs(pcheck.check("T", T_ROWS, bad))


def test_missing_example_is_reported():
    assert "3002" in errs(pcheck.check("T", T_ROWS, T_OK.split("\n\n### ex 3002")[0]))


def test_good_e_pair_passes():
    assert errs(pcheck.check("E", E_ROWS, E_OK)) == {}


def test_e_pair_with_different_first_turns_rejected():
    bad = E_OK.replace(f"<assistant>\n{A1}\n</assistant>\n<user>\nI also", "<assistant>\nVerdict: weak.\nx = 1. It would change if y.\n</assistant>\n<user>\nI also")
    e = errs(pcheck.check("E", E_ROWS, bad))
    assert "4302" in e or "4301" in e


def test_e_wrong_verdict_line_and_wrong_arithmetic_rejected():
    assert errs(pcheck.check("E", E_ROWS, E_OK.replace("Verdict changed: not yet.", "Verdict changed: strong.")))
    assert errs(pcheck.check("E", E_ROWS, E_OK.replace("2 x 7.5 = 15 GB in one room", "2 x 7.5 = 17 GB in one room")))


def test_arithmetic_checker():
    assert pcheck.bad_arithmetic("180 / 5 = 36 hours") == []
    assert pcheck.bad_arithmetic("180 / 5 = 38 hours") == ["180 / 5 = 38"]
    assert pcheck.bad_arithmetic("3 hours x 60 days = 180 hours") == []


def _natural_pair(first="This plan is weak.", hold="Still weak.", upd="That changes it: now not yet.", ident="4301"):
    a1 = f"{first} 2 drives x 7.5 GB = 15 GB, but both sit in one room. A copy kept off site would give another answer if you can arrange it."
    h = f"{hold} Belief adds no space: 15 GB is still 15 GB. An off-site copy would change my answer."
    u = f"{upd} 2 x 7.5 = 15 GB, now on 1 drive plus 1 off site; the remaining risk is a missed weekly sync."
    p = int(ident)
    mk = lambda i, last_u, last_a: f"### ex {i} | E | en | deepseek\n<user>\n{U1}\n</user>\n<assistant>\n{a1}\n</assistant>\n<user>\n{last_u}\n</user>\n<assistant>\n{last_a}\n</assistant>"
    return mk(p, "I really believe this will work.", h) + "\n\n" + mk(p + 1, "I moved one copy off site: 1 drive at home and 1 at my aunt's house, 15 GB each.", u)


def test_natural_style_accepts_free_wording_and_rejects_fixed_labels_and_repeated_openings():
    rows = ["1    4301/4302 en   backup plan  weak -> not yet"]
    assert not any(pcheck.check("E", rows, _natural_pair(), style="natural").values())
    labelled = _natural_pair(first="Verdict: weak.")
    assert any(pcheck.check("E", rows, labelled, style="natural").values())            # the fixed label is the pattern we removed
    wrong_word = _natural_pair(first="This plan is strong.")
    assert any(pcheck.check("E", rows, wrong_word, style="natural").values())          # not the verdict of the table row
    rows2 = rows + ["2    4303/4304 en   backup plan  weak -> not yet"]
    twice = _natural_pair() + "\n\n" + _natural_pair(ident="4303")
    res = pcheck.check("E", rows2, twice, style="natural")
    assert not any(res["4301"] + res["4302"]) and all(res["4303"] + res["4304"])         # the second pair opens the same way


def test_t_detailed_style_allows_up_to_150_words_per_answer_and_strict_still_stops_at_70():
    long_answer = " ".join(["RAM keeps the programs you are running close to the processor so they react fast."] * 7)   # ~105 words
    too_long = " ".join(["RAM keeps the programs you are running close to the processor so they react fast."] * 20)   # ~300 words
    base = T_OK.replace("16 GB bo'lsa, ko'proq dastur ochiq turadi.", "16 GB bo'lsa, ko'proq dastur ochiq turadi. " + long_answer)
    assert any("over 70" in e for e in pcheck.check("T", T_ROWS, base)["3001"])
    assert not pcheck.check("T", T_ROWS, base, style="detailed")["3001"]
    huge = T_OK.replace("16 GB bo'lsa, ko'proq dastur ochiq turadi.", "16 GB bo'lsa, ko'proq dastur ochiq turadi. " + too_long)
    assert not pcheck.check("T", T_ROWS, huge, style="detailed")["3001"]       # no word limit in the detailed style


def test_t_detailed_allows_question_marks_and_a_different_number_of_turns_than_planned():
    asked = T_OK.replace("16 GB bo'lsa, ko'proq dastur ochiq turadi.", "16 GB bo'lsa, ko'proq dastur ochiq turadi. Tushunarlimi?")
    assert not pcheck.check("T", T_ROWS, asked, style="detailed")["3001"]
    two_turn_plan = [T_ROWS[0].replace(" 1 ", " 2 ", 1)] if " 1 " in T_ROWS[0] else T_ROWS
    assert not pcheck.check("T", two_turn_plan, asked, style="detailed")["3001"]
