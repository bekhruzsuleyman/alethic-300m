import json
import os

from alethic.agent import folderpick as fp


def test_recent_folders_are_remembered_newest_first_without_duplicates_and_capped(tmp_path):
    f = str(tmp_path / "recent.json")
    dirs = []
    for i in range(9):
        d = tmp_path / ("p%d" % i)
        d.mkdir()
        dirs.append(str(d))
        fp.remember(f, str(d))
    fp.remember(f, dirs[3])
    got = fp.load_recent(f)
    assert got[0] == dirs[3] and len(got) == fp.MAX_RECENT and len(set(got)) == len(got)
    assert got[1] == dirs[8]


def test_missing_or_broken_files_and_deleted_folders_are_ignored(tmp_path):
    assert fp.load_recent(str(tmp_path / "none.json")) == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert fp.load_recent(str(bad)) == []
    gone = tmp_path / "gone"
    gone.mkdir()
    f = str(tmp_path / "r.json")
    fp.remember(f, str(gone))
    gone.rmdir()
    assert fp.load_recent(f) == []
    assert json.loads(open(f).read())          # the file itself is still valid json


def test_storage_path_prefers_appdata_on_windows_and_the_home_folder_elsewhere():
    assert fp.storage_path({"APPDATA": r"C:\Users\b\AppData\Roaming"}, "/home/b").replace("\\", "/").endswith("Roaming/Alethic/recent_folders.json")
    assert fp.storage_path({}, "/home/b").replace("\\", "/") == "/home/b/.alethic/recent_folders.json"


def test_risky_folders_get_a_warning():
    home = os.path.expanduser("~")
    assert fp.risk(os.path.abspath(os.sep), home)
    assert fp.risk(home, home)
    assert fp.risk(os.path.dirname(home), home)
    assert fp.risk(r"C:\Windows", home, windows=True) and fp.risk(r"C:\Program Files", home, windows=True)
    assert fp.risk("/usr", home) and fp.risk("/etc", home)
    assert fp.risk(os.path.join(home, "projects", "x"), home) is None


def test_options_list_recent_folders_then_the_default_and_the_other_ways(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    opts = fp.options([a, b], default=str(tmp_path / "work"))
    keys = [k for k, _, _ in opts]
    assert keys == ["1", "2", "d", "b", "t"]
    assert opts[0][2] == a and opts[2][2] == str(tmp_path / "work")
    assert [k for k, _, _ in fp.options([], default="")] == ["b", "t"]


def test_choose_parses_numbers_letters_and_typed_paths():
    opts = fp.options(["/x/a", "/x/b"], default="/x/work")
    assert fp.parse_answer("2", opts) == ("folder", "/x/b")
    assert fp.parse_answer("d", opts) == ("folder", "/x/work")
    assert fp.parse_answer("B", opts) == ("browse", "")
    assert fp.parse_answer("t", opts) == ("type", "")
    assert fp.parse_answer("", opts) == ("folder", "/x/a")                 # Enter = the most recent folder
    assert fp.parse_answer("/some/where", opts) == ("folder", "/some/where")
    assert fp.parse_answer('"C:\\My Files"', opts) == ("folder", "C:\\My Files")
    assert fp.parse_answer("9", opts) == ("bad", "")
    assert fp.parse_answer("q", opts) == ("cancel", "")


def run_pick(tmp_path, answers, browse=None, recent=None, default=None):
    said, it = [], iter(answers)
    f = str(tmp_path / "recent.json")
    for r in recent or []:
        fp.remember(f, r)
    got = fp.pick(said.append, lambda prompt="": next(it), browse or (lambda start: ""), f, default or str(tmp_path / "work"),
                  home=str(tmp_path / "home"))
    return got, "\n".join(said), f


def test_pick_returns_a_recent_folder_and_remembers_it(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    got, said, f = run_pick(tmp_path, ["1"], recent=[str(d)])
    assert got == str(d) and "proj" in said and fp.load_recent(f)[0] == str(d)


def test_pick_browse_uses_the_dialog_and_falls_back_to_typing_when_it_is_closed(tmp_path):
    d = tmp_path / "chosen"
    d.mkdir()
    got, _, _ = run_pick(tmp_path, ["b"], browse=lambda start: str(d))
    assert got == str(d)
    e = tmp_path / "typed"
    e.mkdir()
    got, said, _ = run_pick(tmp_path, ["b", str(e)], browse=lambda start: "")
    assert got == str(e) and "type" in said.lower()


def test_pick_creates_a_missing_folder_only_after_a_yes(tmp_path):
    new = tmp_path / "newone"
    got, _, _ = run_pick(tmp_path, [str(new), "n", "q"])
    assert got is None and not new.exists()
    got, _, _ = run_pick(tmp_path, [str(new), "y"])
    assert got == str(new) and new.is_dir()


def test_pick_warns_before_a_risky_folder_and_goes_back_on_no(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    ok = tmp_path / "ok"
    ok.mkdir()
    got, said, _ = run_pick(tmp_path, [str(home), "n", str(ok)])
    assert got == str(ok) and "whole home folder" in said.lower()
    got, _, _ = run_pick(tmp_path, [str(home), "y"])
    assert got == str(home)


def test_pick_cancel_and_end_of_input_return_none(tmp_path):
    assert run_pick(tmp_path, ["q"])[0] is None

    def eof(prompt=""):
        raise EOFError
    said = []
    assert fp.pick(said.append, eof, lambda s: "", str(tmp_path / "r.json"), str(tmp_path / "w"), home=str(tmp_path)) is None


def test_bad_answers_ask_again(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    got, said, _ = run_pick(tmp_path, ["9", "zzz?", str(d)])
    assert got == str(d)


def test_empty_enter_picks_default_when_no_recent_folders(tmp_path):
    opts = fp.options([], default=str(tmp_path / "work"))
    assert fp.parse_answer("", opts) == ("folder", str(tmp_path / "work"))


class _FakeKeys:
    def __init__(self, *keys):
        self.keys = list(keys)
        self.mouse = True

    def usable(self):
        return True

    def session(self):
        import contextlib
        return contextlib.nullcontext()

    def read(self):
        return self.keys.pop(0)


def test_pick_uses_arrow_keys_on_tty(tmp_path):
    import io
    from alethic.agent.ui import UI

    a, b = tmp_path / "first", tmp_path / "second"
    a.mkdir()
    b.mkdir()
    recent = str(tmp_path / "recent.json")
    fp.remember(recent, str(b))
    fp.remember(recent, str(a))

    ui = UI(out=io.StringIO(), color=False, unicode=True, isatty=True, fancy=True, size=lambda: (80, 24),
            keys=_FakeKeys("down", "enter"))
    got = fp.pick(ui.line, lambda p="": "", lambda s: "", recent, default=str(tmp_path / "work"),
                  home=str(tmp_path / "home"), ui=ui)
    assert got == str(b)

    ui_new = UI(out=io.StringIO(), color=False, unicode=True, isatty=True, fancy=True, size=lambda: (80, 24),
                keys=_FakeKeys("enter", "y"))
    target_new = tmp_path / "brand_new"
    got_new = fp.pick(ui_new.line, lambda p="": "", lambda s: "", str(tmp_path / "r2.json"),
                      default=str(target_new), home=str(tmp_path / "home"), ui=ui_new)
    assert got_new == str(target_new) and target_new.is_dir()

    ui_esc = UI(out=io.StringIO(), color=False, unicode=True, isatty=True, fancy=True, size=lambda: (80, 24),
                keys=_FakeKeys("esc"))
    assert fp.pick(ui_esc.line, lambda p="": "", lambda s: "", recent, default=str(tmp_path / "work"),
                   home=str(tmp_path / "home"), ui=ui_esc) is None


def test_windows_browse_dialog_works_without_tkinter(tmp_path, monkeypatch):
    import subprocess
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["start"] = kw["env"]["ALETHIC_BROWSE_START"]
        return subprocess.CompletedProcess(cmd, 0, stdout=str(tmp_path), stderr="")

    monkeypatch.setattr("os.name", "nt")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert fp.browse_dialog(str(tmp_path)) == os.path.normpath(str(tmp_path))
    assert seen["cmd"][0] == "powershell" and seen["start"] == str(tmp_path)
