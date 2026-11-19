"""Run the model's tool calls inside one folder and return the result text the model should see next."""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass

from alethic.agent.policy import check
from alethic.agent.protocol import Call
from alethic.agent.shellemu import Emu

ROOT_SHOWN = "/work"  # what results show instead of the real folder, like the training data
SAFE = ("python", "python3", "py", "ls", "cat", "echo", "pwd", "pytest", "head", "tail", "wc", "grep", "find", "dir", "type", "mkdir", "touch")
PY_AT_START = re.compile(r"(^|&&|\|\||;|\|)(\s*)(?:python3?|py(?:\s+-3)?)(?=\s|$)")
WORK_PATH = re.compile(r"/work(?:\\{1,2}[^\s\\'\"<>|:*?]+)+")  # /work\\a\\b.py: the rest of a Windows path after the folder
RANGE = re.compile(r"^(.*?),\s*(\d+)\s*-\s*(\d+)$")


@dataclass
class Outcome:
    result: str | None  # text for the <result> block; None when the call has no result (write, edit)
    ok: bool = True


def python_command(frozen: bool, executable: str, which) -> str | None:
    """The command that runs Python scripts: this Python, or (inside the .exe, where sys.executable is the exe itself)
    one found on the computer: `py`, then `python`, then `python3`. None if there is no Python."""
    if not frozen:
        return '"%s"' % executable
    if which("py"):
        return "py -3"
    for name in ("python", "python3"):
        found = which(name)
        if found and "WindowsApps" not in found:  # the Microsoft Store stub is not a real Python
            return '"%s"' % found
    return None


def rewrite_python(cmd: str, python: str) -> str:
    """Run `python`, `python3` or `py` at the start of a command with the Python that exists on this computer."""
    return PY_AT_START.sub(lambda m: m.group(1) + m.group(2) + python, cmd)


def numbered(lines, start):
    return "\n".join("%-4d%s" % (start + i, l) for i, l in enumerate(lines))


class Workspace:
    def __init__(self, root, timeout=20, cap=20_000, approve=None, mode=None, python="auto", search_fetch=None):
        self.root = os.path.abspath(root)
        self.search_fetch = search_fetch
        os.makedirs(self.root, exist_ok=True)
        self.timeout, self.cap, self.approve, self.auto = timeout, cap, approve, False
        self.mode = mode or ("windows" if os.name == "nt" else "posix")
        if python == "auto":
            frozen = bool(getattr(sys, "frozen", False))
            found = python_command(frozen, sys.executable, shutil.which)
            self.python = found if found is not None else ('"%s" --run-python' % sys.executable if frozen else None)
        else:
            self.python = python

    # paths -------------------------------------------------------------
    def resolve(self, path: str) -> str:
        path = path.strip().strip("\"'")
        if path.startswith(ROOT_SHOWN + "/"):
            path = path[len(ROOT_SHOWN) + 1:]
        if os.path.isabs(path) or re.match(r"^[A-Za-z]:", path):
            raise ValueError("path %r is outside the work folder" % path)
        full = os.path.realpath(os.path.join(self.root, path))
        root = os.path.realpath(self.root)
        if full != root and not full.startswith(root + os.sep):
            raise ValueError("path %r is outside the work folder" % path)
        return full

    # dispatch ----------------------------------------------------------
    def execute(self, call: Call) -> Outcome:
        fn = getattr(self, "_" + call.kind, None) if call.kind in ("shell", "read", "write", "edit", "search") else None
        if fn is None:
            return Outcome("error: unknown tool %s" % call.kind, False)
        try:
            return fn(call)
        except (ValueError, OSError) as exc:
            return Outcome("error: %s" % self._hide_root(str(exc)), False)

    def _hide_root(self, text: str) -> str:
        """Show the work folder as /work/..., like the training data, so a result never carries the user's real path.
        On Windows the folder appears with \\, with / and doubled (\\\\) inside Python's error text; all become /work/name."""
        for form in (self.root.replace("\\", "\\\\"), self.root, self.root.replace("\\", "/")):
            text = text.replace(form, ROOT_SHOWN)
        return WORK_PATH.sub(lambda m: m.group(0).replace("\\\\", "/").replace("\\", "/"), text)

    def _search(self, call):
        from alethic.agent import websearch
        return Outcome(websearch.run_search(call.args, self.search_fetch or websearch.http_fetch))

    def _read(self, call):
        m = RANGE.match(call.args)
        if not m:
            raise ValueError("read needs a path and a line range, like read(a.py, 1-20)")
        a, z = int(m.group(2)), int(m.group(3))
        with open(self.resolve(m.group(1)), encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        if not 1 <= a <= z or a > len(lines):
            raise ValueError("range %d-%d is outside the file (%d lines)" % (a, z, len(lines)))
        return Outcome(numbered(lines[a - 1:min(z, len(lines))], a))

    def _write(self, call):
        m = RANGE.match(call.args)          # the model sometimes writes write(a.py, 1-16): the range means nothing here
        full = self.resolve(m.group(1) if m else call.args)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8", newline="\n") as f:
            f.write(call.body or "")
        return Outcome(None)

    def _edit(self, call):
        m = RANGE.match(call.args)
        if not m:
            raise ValueError("edit needs a path and a line range, like edit(a.py, 3-3)")
        full = self.resolve(m.group(1))
        a, z = int(m.group(2)), int(m.group(3))
        with open(full, encoding="utf-8") as f:
            lines = f.read().splitlines()
        if not 1 <= a <= z <= len(lines):
            raise ValueError("lines %d-%d are outside the file (%d lines). Use lines 1-%d, or write(%s) to replace the whole file"
                             % (a, z, len(lines), max(1, len(lines)), m.group(1).strip()))
        new = (call.body or "").rstrip("\n").split("\n") if call.body else []
        with open(full, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines[:a - 1] + new + lines[z:]) + "\n")
        return Outcome(None)

    # shell -------------------------------------------------------------
    def _needs_approval(self, cmd: str) -> bool:
        first = cmd.strip().split(None, 1)[0] if cmd.strip() else ""
        return not (self.auto or first in SAFE)

    def _shell(self, call):
        cmd = call.args
        why = check(cmd)
        if why:
            return Outcome("blocked: " + why, False)
        if self.python is None and PY_AT_START.search(cmd):
            return Outcome("error: no Python found on this computer. Install Python 3 from python.org and tick 'Add to PATH'", False)
        if self.approve and self._needs_approval(cmd) and not self.approve(call):
            return Outcome("(declined by the user)", False)
        out = self._emulate(cmd) if self.mode == "windows" else None
        if out is None:
            out = self._run(rewrite_python(cmd, self.python) if self.python else cmd)
        out = self._hide_root(out)
        if len(out) > self.cap:
            out = out[:self.cap] + "\n[truncated]"
        return Outcome(out)

    def _run(self, cmd: str) -> str:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1", LC_ALL="C.UTF-8")
        args = cmd if self.mode == "windows" else ["bash", "-c", cmd]
        try:
            r = subprocess.run(args, cwd=self.root, capture_output=True, timeout=self.timeout, env=env,
                               shell=self.mode == "windows", stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return "(timed out after %ds)" % self.timeout
        return (r.stdout + r.stderr).decode("utf-8", "replace").replace("\r\n", "\n").strip()  # Windows line endings never reach the model

    def _emulate(self, cmd: str):
        """Windows has no ls/cat/grep/mv/...: do them in Python, same output as the posix shell. None = run it in cmd."""
        return Emu(self.resolve, self.root).run(cmd)
