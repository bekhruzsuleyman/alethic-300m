import os
import re
import tempfile

import pytest

from alethic.agent.shellemu import Emu
from alethic.agent.tools import Workspace
from tools.evalset.prompts import eval_set
from tools.termtraces.build import KINDS, build_all, to_text


@pytest.fixture(scope="module")
def data():
    return build_all(per_kind=40, seed=7)


def calls(e):
    return re.findall(r"<call> (\w+)\((.*?)\) </call>", to_text(e))


def results(e):
    return [s["t"][len("<result>\n"):-len("\n</result>\n")] for s in e["segments"] if s["t"].startswith("<result>")]


def final_say(e):
    return e["segments"][-2]["t"]


def of(data, kind):
    return [e for e in data if e["kind"] == kind]


def test_every_kind_in_both_languages_with_unique_ids(data):
    assert {e["kind"] for e in data} == set(KINDS)
    assert len({e["id"] for e in data}) == len(data)
    for k in KINDS:
        assert {e["lang"] for e in of(data, k)} == {"en", "uz"}, k


def test_same_seed_same_data_other_seed_differs():
    assert build_all(per_kind=3, seed=1) == build_all(per_kind=3, seed=1) != build_all(per_kind=3, seed=2)


def test_structure_nothing_leaks_a_real_path_and_every_call_is_trained(data):
    for e in data:
        segs = e["segments"]
        assert segs[0]["t"].startswith("<user>\n") and segs[0]["loss"] == 0 and segs[1]["t"] == "<assistant>\n" and segs[-1]["t"] == "</assistant>\n"
        text = to_text(e)
        assert "/tmp" not in text and "errtrace" not in text and "Traceback" not in text and "error:" not in text
        for r in results(e):
            for bad in ("unexpected EOF", "syntax error", "command not found", "No such file", "cannot", "Is a directory", "not specified", "blocked"):
                assert bad not in r, (bad, text)
        for s in segs:
            if s["t"].startswith("<result>") or s["t"].startswith("<user>"):
                assert s["loss"] == 0
            elif s["t"].startswith("<call>") or s["t"].startswith("<plan>"):
                assert s["loss"] == 1


FIRST_TOOL = {
    "save_file": ("write", None), "read_file": ("read", "1-100"), "read_lines": ("read", None), "make_folder": ("shell", "mkdir"),
    "folder_and_file": ("shell", "mkdir"), "rename": ("shell", "mv"), "copy": ("shell", "cp"), "move": ("shell", "mkdir"),
    "head_tail": ("shell", ("head", "tail")), "count": ("shell", ("wc", "ls | wc")), "find_text": ("shell", "grep"),
    "find_files": ("shell", "find"), "list_folder": ("shell", "ls"), "delete": ("shell", "rm"), "where": ("shell", "pwd"), "append": ("shell", "echo"),
}


def test_the_same_kind_of_request_always_uses_the_same_tool(data):
    for e in data:
        tool, hint = FIRST_TOOL[e["kind"]]
        first = calls(e)[0]
        assert first[0] == tool, (e["kind"], first)
        if hint:
            hints = hint if isinstance(hint, tuple) else (hint,)
            assert any(first[1].startswith(h) or first[1].endswith(h) for h in hints), (e["kind"], first)


def test_save_file_writes_exactly_the_text_the_user_gave(data):
    for e in of(data, "save_file"):
        user = e["segments"][0]["t"]
        body = next(s["t"] for s in e["segments"] if s["t"].startswith("<call> write("))
        body = body.split("\n", 1)[1]
        for line in body.strip().split("\n"):
            assert line in user, (line, user)


def test_final_sentence_only_names_things_the_results_or_request_show(data):
    for e in data:
        user = e["segments"][0]["t"]
        shown = user + " ".join(results(e)) + " ".join(a for _, a in calls(e)) + to_text(e).split("</plan>")[0]
        words = re.findall(r"[\w.\-/]+\.(?:txt|md|py)", final_say(e))
        for w in words:
            assert w in shown, (e["kind"], w, final_say(e))
        for num in re.findall(r"\b\d+\b", final_say(e)):
            bodies = [x["t"].split("\n", 1)[1].strip().count("\n") + 1 for x in e["segments"] if x["t"].startswith("<call> write(")]
            ok = num in shown or bool(results(e) and num == str(len(results(e)[-1].split("\n")))) or int(num) in bodies
            assert ok, (e["kind"], num, final_say(e), shown)


def test_count_quotes_the_real_number_and_folder_listings_are_exact(data):
    for e in of(data, "count"):
        res = results(e)[0].split()[0]
        assert res in final_say(e)
    for e in of(data, "make_folder") + of(data, "rename") + of(data, "copy") + of(data, "delete"):
        names = [n for n in results(e)[-1].split("\n") if n]
        for n in names:
            assert n in final_say(e)


def test_find_text_reports_every_real_hit_and_no_hit_is_said_plainly(data):
    hits = empties = 0
    for e in of(data, "find_text"):
        out = results(e)[0]
        if not out:
            empties += 1
            assert "No line" in final_say(e) or "topilmadi" in final_say(e)
        else:
            hits += 1
            for line in out.split("\n"):
                assert line.split(":", 1)[1] in final_say(e)
    assert hits and empties


def test_the_users_own_words_are_in_the_requests(data):
    users = "\n".join(e["segments"][0]["t"] for e in data)
    for phrase in ("save file ", "read file ", "make new folder ", "rename ", "copy ", "move ", "delete ", "first 3 lines", "how many lines", "what folder am I in"):
        assert phrase in users or phrase.capitalize() in users, phrase


def test_requests_are_diverse(data):
    users = [e["segments"][0]["t"] for e in data]
    assert len(set(users)) > 0.8 * len(users)
    assert len({re.sub(r"\S+\.(txt|md)", "F", u) for u in users}) > 60


def test_eval_prompts_never_appear_in_training(data):
    text = "".join(to_text(e) for e in data)
    for p in eval_set():
        assert "<user>\n%s\n</user>" % p["prompt"] not in text


def test_the_windows_emulation_can_run_every_shell_command_in_the_data(data):
    for e in data:
        for kind, args in calls(e):
            if kind != "shell":
                continue
            d = tempfile.mkdtemp()
            ws = Workspace(d, mode="windows", python=None)
            for need in re.findall(r"[\w\-/]+\.(?:txt|md)|\b(?:%s)\b" % "|".join(["logs", "photos", "backup", "docs", "src", "data", "temp", "output"]), args):
                if "." in need:
                    os.makedirs(os.path.dirname(os.path.join(d, need)) or d, exist_ok=True)
                    open(os.path.join(d, need), "w").write("x\n")
            assert Emu(ws.resolve, ws.root).run(args) is not None, args
