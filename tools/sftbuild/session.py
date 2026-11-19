"""Build samples from a multi-window project session script, running every step for real."""
import os
import re

from tools.sftbuild.fileops import FileOpError, apply_edit, numbered as _number
from tools.sftbuild.mapper import ast_map
from tools.sftbuild.policy import check
from tools.sftbuild.sandbox import fresh_dir, run
from tools.sftbuild.seq import Seq
from tools.sftbuild.terminal import ROOT

MARKERS = r"\n(?=USER: |ASSISTANT: |PLAN: |STEP: |FINAL: )"


class SessionError(Exception):
    pass


def _traceback(workdir, spec):
    files, cmd = spec
    for name, content in files.items():
        with open(os.path.join(workdir, name), "w") as f:
            f.write(content)
    _, out = run(cmd, workdir)
    for name in files:
        os.remove(os.path.join(workdir, name))
    return out.replace(workdir, ROOT)


class _Window:
    def __init__(self, seq):
        self.seq, self.open, self.flags, self.last_end = seq, False, set(), {}

    def asst(self):
        if not self.open:
            self.seq.add("<assistant>\n", 0)
            self.open = True

    def close(self):
        if self.open:
            self.seq.add("</assistant>\n", 0)
            self.open = False


def _edit(workdir, part):
    path = re.search(r"PATH: (\S+)", part).group(1)
    m = re.search(r"BUG_LINE: (.*?) \| NEW_LINE: (.*)$", part, re.S)
    old, new = m.group(1), m.group(2).rstrip("\n")
    try:
        first, last = apply_edit(workdir, path, old, new)
    except FileOpError as exc:
        raise SessionError(str(exc)) from exc
    return "<call> edit(%s, %d-%d) </call>\n%s\n" % (path, first, last, new)


def _step(w, part, workdir):
    kind = re.match(r"STEP: (\w+)", part).group(1)
    if kind == "write":
        path = re.search(r"PATH: (\S+)", part).group(1)
        failed = "ATTEMPT: failed" in part
        body = re.search(r"CONTENT:\n<<<\n(.*?)\n>>>", part, re.S).group(1) + "\n"
        full = os.path.join(workdir, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(body)
        w.seq.add("<call> write(%s) </call>\n%s" % (path, body), 0 if failed else 1)
    elif kind == "edit":
        w.seq.add(_edit(workdir, part), 1)
    elif kind == "read":
        path = re.search(r"PATH: (\S+)", part).group(1)
        m = re.search(r"lines (\d+)-(\w+)", part)
        lines = open(os.path.join(workdir, path)).read().splitlines()
        a = int(m.group(1)) if m else 1
        b = len(lines) if (not m or m.group(2) == "end") else min(int(m.group(2)), len(lines))
        w.seq.add("<call> read(%s, %d-%d) </call>\n" % (path, a, b), 1)
        idx = w.seq.add("<result>\n%s\n</result>\n" % _number(lines[a - 1:b], a), 0, "read")
        scanning = a == 1 or w.last_end.get(path) == a - 1
        w.last_end[path] = b
        if scanning and path.endswith(".py"):
            mp = ast_map(os.path.join(workdir, path), b)
            if mp is None:
                w.flags.add("no_map")
            else:
                w.seq.set_read_map(idx, w.seq.add("<map> %s </map>\n" % mp, 1, "map"))
    elif kind in ("shell", "run"):
        cmd = re.search(r"CMD: (.*)", part).group(1)
        why = check(cmd)
        if why:
            raise SessionError("%s: %s" % (cmd, why))
        _, out = run(cmd, workdir)
        w.seq.add("<call> shell(%s) </call>\n" % cmd, 1)
        w.seq.add("<result>\n%s\n</result>\n" % out.replace(workdir, ROOT), 0)


def build_session(text: str, workdir: str, traceback=None) -> list:
    try:
        fresh_dir(workdir)
    except ValueError as exc:
        raise SessionError(str(exc)) from exc
    samples = []
    windows = [p for p in re.split(r"(?m)^(?=### WINDOW )", text) if p.startswith("### WINDOW ")]
    for wi, win in enumerate(windows, 1):
        win = re.split(r"\n### (?:PLANTED_BUG|CHANGES|EDITS)", win)[0]
        head, *stages = re.split(r"\n(?=### STAGE )", win)
        goal, todo, done = (re.search(r"^%s: (.*)$" % k, head, re.M).group(1)
                            for k in ("GOAL", "TODO", "DONE_SO_FAR"))
        w = _Window(Seq())
        if wi > 1:
            w.seq.add("<goal>\n%s\n</goal>\n<todo>\n%s\n</todo>\n<done>\n%s\n</done>\n" % (goal, todo, done), 0)
        for st in stages:
            body = st.split("\n", 1)[1]
            blocks = re.findall(r"<<<\n.*?\n>>>", body, re.S)  # file content must never be split on markers
            for i, blk in enumerate(blocks):
                body = body.replace(blk, "\x00%d\x00" % i, 1)
            for part in re.split(MARKERS, "\n" + body.strip("\n")):
                part = part.strip("\n")
                for i, blk in enumerate(blocks):
                    part = part.replace("\x00%d\x00" % i, blk)
                if part.startswith("USER: "):
                    w.close()
                    msg = part[6:]
                    if "{TRACEBACK}" in msg:
                        msg = msg.replace("{TRACEBACK}", _traceback(workdir, traceback))
                    w.seq.add("<user>\n%s\n</user>\n" % msg, 0)
                elif part.startswith("ASSISTANT: "):
                    w.asst()
                    msg = part[11:]
                    g = re.match(r"GOAL: (.*?) TODO: (.*)$", msg, re.S)
                    w.seq.add(("<goal>\n%s\n</goal>\n<todo>\n%s\n</todo>\n" % g.groups()) if g else msg + "\n", 1)
                elif part.startswith("PLAN: "):
                    w.asst()
                    w.seq.add("<plan> %s </plan>\n" % part[6:], 1)
                elif part.startswith("FINAL: "):
                    w.asst()
                    w.seq.add(part[7:] + "\n", 1)
                elif part.startswith("STEP: "):
                    w.asst()
                    _step(w, part, workdir)
        w.close()
        for s in w.seq.samples("W%d" % wi):
            s["window"], s["flags"] = wi, sorted(w.flags)
            samples.append(s)
    return samples
