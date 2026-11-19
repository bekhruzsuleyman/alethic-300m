"""The Windows shell emulation must print exactly what the real posix shell printed when the training data was made.
Every case runs the same command in real bash and in the emulation, on two identical folders, and compares the
output and the files left behind."""
import os

import pytest

from alethic.agent.tools import Workspace


def fixture(root):
    os.makedirs(os.path.join(root, "d"))
    os.makedirs(os.path.join(root, "sub"))
    files = {
        "f.txt": "a\nb\nc\n",
        "g.txt": "".join("line %d\n" % i for i in range(1, 16)),
        "sub/x.py": "print(1)\n",
        "sub/y.txt": "hello world\nHello\nbye\n",
        ".hidden": "h\n",
        "nonl.txt": "one\ntwo",
    }
    for name, text in files.items():
        with open(os.path.join(root, name), "w", newline="\n") as f:
            f.write(text)


def tree(root):
    out = {}
    for base, dirs, names in os.walk(root):
        for d in dirs:
            out[os.path.relpath(os.path.join(base, d), root)] = None
        for n in names:
            with open(os.path.join(base, n), "rb") as f:
                out[os.path.relpath(os.path.join(base, n), root)] = f.read()
    return out


def pair(tmp_path):
    a, b = str(tmp_path / "posix"), str(tmp_path / "win")
    os.makedirs(a)
    os.makedirs(b)
    fixture(a)
    fixture(b)
    return Workspace(a, mode="posix", python=None), Workspace(b, mode="windows", python=None)


ORDERLESS = ("find", "grep -r", "grep -rn")  # the real tools list a folder in disk order; compare as sets of lines

READS = [
    "cat f.txt", "cat f.txt g.txt", "cat nofile", "cat d", "cat nonl.txt",
    "head f.txt", "head -n 2 g.txt", "head -3 g.txt", "head g.txt", "head -n 2 f.txt g.txt", "head nofile", "head -n -13 g.txt",
    "tail g.txt", "tail -n 2 g.txt", "tail -2 g.txt", "tail -n +14 g.txt", "tail nofile", "tail -n 2 f.txt g.txt",
    "wc f.txt", "wc -l f.txt", "wc -w f.txt", "wc -c f.txt", "wc g.txt", "wc -l f.txt g.txt", "wc nofile", "wc -lw g.txt", "wc nonl.txt",
    "grep a f.txt", "grep -n b f.txt", "grep -c a f.txt", "grep -i hello sub/y.txt", "grep -v a f.txt",
    "grep -l hello sub/y.txt f.txt", "grep a f.txt g.txt", "grep -r hello .", "grep -rn hello sub", "grep zzz f.txt",
    "grep a nofile", "grep -w line g.txt", "grep -e a -e b f.txt", "grep -F 'a' f.txt", "grep -E 'a|b' f.txt", "grep 'a\\|b' f.txt",
    "grep -r hello", "grep -r hello sub/y.txt", "grep hello d", "grep -h a f.txt g.txt", "grep -c line g.txt f.txt",
    "find .", "find . -name '*.py'", "find sub -type f", "find . -type d", "find . -maxdepth 1 -name '*.txt'", "find nofile",
    "find", "find . -iname 'F.TXT'", "find sub", "find . -mindepth 2",
    "ls", "ls sub", "ls -a", "ls nofile", "ls f.txt", "ls sub d", "ls f.txt sub", "ls -1", "ls *.txt", "ls *.zzz", "ls sub/*.py",
    "cat *.txt", "wc -l *.txt", "grep -n line *.txt", "echo hello", "echo hello world", "echo 'a  b'", 'echo "x  y"', "echo", "echo -n hi",
    "cat g.txt | head -n 2", "grep line g.txt | wc -l", "ls | wc -l", "cat f.txt | grep b", "head -n 3 g.txt | tail -n 1",
    "cat g.txt | wc", "cat f.txt | wc -l", "ls | head -n 2", "grep line g.txt | head -n 2 | tail -n 1", "cat nofile | wc -l",
]

WRITES = [
    "mkdir newdir", "mkdir -p a/b/c", "mkdir d", "mkdir x/y", "mkdir a b", "mkdir -p d",
    "touch n.txt", "touch x/y", "touch f.txt n1 n2",
    "mv f.txt h.txt", "mv f.txt d", "mv nofile z", "mv f.txt g.txt", "mv sub sub2", "mv f.txt x/y", "mv f.txt g.txt d",
    "cp f.txt h.txt", "cp f.txt d", "cp nofile z", "cp d z", "cp -r sub sub2", "cp -r sub d", "cp f.txt x/y", "cp f.txt g.txt d",
    "rm f.txt", "rm nofile", "rm d", "rm -f nofile", "rm f.txt g.txt", "rm *.txt", "rm -f f.txt nofile",
    "echo hello > o.txt", "echo hi >> f.txt", "echo 'a b' > o.txt", "cat f.txt > o.txt", "echo x > f.txt",
    "echo one > o.txt && echo two",  # unsupported: not run by the emulation, checked below
]
WRITES = [w for w in WRITES if "&&" not in w]


def norm(cmd, text):
    if cmd.split()[0] == "find" or cmd.startswith("grep -r") or cmd.startswith("grep -rn"):
        return sorted(text.split("\n"))
    return text


@pytest.mark.parametrize("cmd", READS + WRITES)
def test_emulation_matches_real_shell(tmp_path, cmd):
    posix, win = pair(tmp_path)
    expected = posix._run(cmd)
    got = win._emulate(cmd)
    assert got is not None, "not handled: " + cmd
    assert norm(cmd, got) == norm(cmd, expected), cmd
    assert tree(posix.root) == tree(win.root), cmd


@pytest.mark.parametrize("cmd", [
    "python x.py", "sort f.txt", "echo $HOME", "ls -l", "cat f.txt && cat g.txt", "git status", "echo `ls`", "cat < f.txt",
    "echo a ; echo b", "head -c 3 f.txt", "grep -P a f.txt", "find . -exec cat {} +", "cp -p f.txt h.txt", "cat f.txt 2>&1",
])
def test_unsupported_commands_fall_back_to_the_real_shell(tmp_path, cmd):
    _, win = pair(tmp_path)
    before = tree(win.root)
    assert win._emulate(cmd) is None
    assert tree(win.root) == before  # and nothing was changed on the way


def test_nothing_outside_the_work_folder_is_touched(tmp_path):
    _, win = pair(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret\n")
    for cmd in ("cat ../secret.txt", "cat %s" % secret, "mv f.txt ../out.txt", "cp ../secret.txt in.txt", "echo x > ../o.txt",
                "rm ../secret.txt", "head ../secret.txt", "grep top ../secret.txt", "ls ..", "find ..", "mkdir ../zz", "touch ../zz"):
        out = win._emulate(cmd)
        assert out is not None and "outside the work folder" in out, (cmd, out)
    assert secret.read_text() == "top secret\n"
    assert not (tmp_path / "out.txt").exists() and not (tmp_path / "o.txt").exists() and not (tmp_path / "zz").exists()
    assert not os.path.exists(os.path.join(win.root, "in.txt"))


def test_glob_never_leaves_the_folder(tmp_path):
    _, win = pair(tmp_path)
    (tmp_path / "outside.txt").write_text("x\n")
    out = win._emulate("cat ../*.txt")
    assert out is not None and "outside" in out
    assert "x" != out


def test_unquoted_star_expands_and_quoted_star_does_not(tmp_path):
    posix, win = pair(tmp_path)
    assert win._emulate("find . -name '*.py'") == posix._run("find . -name '*.py'")
    assert win._emulate("echo '*.txt'") == "*.txt"
    assert win._emulate("echo *.txt") == posix._run("echo *.txt")


def test_shell_tool_uses_the_emulation_on_windows(tmp_path, monkeypatch):
    from alethic.agent.protocol import Call
    _, win = pair(tmp_path)

    def boom(*a, **k):
        raise AssertionError("fell through to cmd.exe")
    monkeypatch.setattr(win, "_run", boom)
    win.auto = True
    out = win.execute(Call("shell", "grep -n b f.txt"))
    assert out.result == "2:b" and out.ok
    out = win.execute(Call("shell", "mv f.txt moved.txt"))
    assert out.ok and os.path.exists(os.path.join(win.root, "moved.txt"))


def test_safe_list_covers_make_folder_and_file_but_not_overwriting_commands():
    from alethic.agent.tools import SAFE
    assert "mkdir" in SAFE and "touch" in SAFE
    assert "mv" not in SAFE and "cp" not in SAFE and "rm" not in SAFE
