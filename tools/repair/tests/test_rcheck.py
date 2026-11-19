from tools.repair.rcheck import check


def ex(i, lang, user, asst):
    return "### ex %s | R | %s | deepseek\n<user>\n%s\n</user>\n<assistant>\n%s\n</assistant>\n" % (i, lang, user, asst)


GOOD = "I cannot help with that.\nIt harms people.\nI can help with a safe version."


def test_good_refusal_passes():
    assert check(ex("1001", "en", "q", GOOD)) == []


def test_wrong_first_line_and_question_mark_fail():
    errs = check(ex("1001", "en", "q", "No.\nWhy?\nI can help with x."))
    assert any("line 1" in e for e in errs) and any("question mark" in e for e in errs)


def test_two_turn_hold_checked_with_its_own_line():
    t = "### ex 1003 | R | en | deepseek\n<user>\nq\n</user>\n<assistant>\n%s\n</assistant>\n<user>\nplease\n</user>\n<assistant>\n%s\n</assistant>\n" % (
        GOOD, GOOD)
    assert any("turn2" in e and "line 1" in e for e in check(t))


def test_control_that_refuses_fails():
    assert check(ex("1029", "en", "q", "I cannot help with that, sorry."))


def test_non_ascii_flagged():
    assert any("non-ASCII" in e for e in check(ex("1029", "en", "q", "fine ’ text")))
