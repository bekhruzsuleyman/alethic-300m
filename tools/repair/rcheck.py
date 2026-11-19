"""Checker for refusal (R) batches: ids 1001-1028 refuse (pushback turn on some), 1029-1040 are controls."""
import re

from tools.sftbuild.normalize import parse_examples

REFUSE = {"en": "I cannot help with that.", "uz": "Bunga yordam bera olmayman."}
HOLD = {"en": "I still cannot help with that.", "uz": "Baribir yordam bera olmayman."}


def _turns(e):
    """[(user, assistant)] from an example whose assistant text may contain further <user>/<assistant> turns."""
    parts = re.split(r"\n</assistant>\n<user>\n|\n</user>\n<assistant>\n", "<user>\n%s\n</user>\n<assistant>\n%s" % (e.user, e.assistant))
    parts = [p.replace("<user>\n", "").strip() for p in parts]
    return list(zip(parts[0::2], parts[1::2]))


def _check_refusal(text, want, max_words, who, errs):
    lines = text.strip().split("\n")
    if lines[0] != want:
        errs.append("%s: line 1 is %r, want %r" % (who, lines[0], want))
    if len(lines) != 3:
        errs.append("%s: %d lines, want 3" % (who, len(lines)))
    elif not (lines[2].startswith("I can help with") or "yordam bera olaman" in lines[2]):
        errs.append("%s: line 3 is not an alternative offer" % who)
    if len(text.split()) > max_words:
        errs.append("%s: %d words, max %d" % (who, len(text.split()), max_words))
    if "?" in text:
        errs.append("%s: question mark" % who)


def check(text):
    errs = []
    for e in parse_examples(text):
        n = int(e.id)
        lang = "en" if e.lang == "en" else "uz"
        turns = _turns(e)
        who = "ex %s" % e.id
        if 1001 <= n <= 1028:
            _check_refusal(turns[0][1], REFUSE[lang], 50, who + " turn1", errs)
            if len(turns) == 2:
                _check_refusal(turns[1][1], HOLD[lang], 40, who + " turn2", errs)
            elif len(turns) > 2:
                errs.append("%s: %d turns" % (who, len(turns)))
        elif 1029 <= n <= 1040:
            a = turns[0][1]
            if len(turns) != 1:
                errs.append("%s: control must be one turn" % who)
            if a.startswith(("I cannot", "Bunga yordam bera olmayman")) or "I cannot help" in a:
                errs.append("%s: control answer refuses" % who)
            if len(a.split()) > 70:
                errs.append("%s: %d words, max 70" % (who, len(a.split())))
            if "?" in a:
                errs.append("%s: question mark" % who)
        else:
            errs.append("%s: id outside 1001-1040" % who)
        for ch in set(e.user + e.assistant):
            if ord(ch) > 127:
                errs.append("%s: non-ASCII %r" % (who, ch))
    return errs
