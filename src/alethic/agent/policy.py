"""Commands the agent may not run. The training harness blocked the same things, so the model rarely tries them."""
from __future__ import annotations

import re
import shlex

BLOCKED = {"sudo", "su", "curl", "wget", "ssh", "scp", "nc", "powershell", "pwsh", "del", "erase", "rmdir", "rd",
           "format", "shutdown", "reg", "taskkill", "chmod", "chown"}
SHELLS = {"sh", "bash", "zsh", "dash", "cmd"}
SEPARATORS = {"|", "||", "&&", ";", "&"}
DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


def check(cmd: str):
    """Return a reason if the command is blocked, else None."""
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    try:
        tokens = list(lex)
    except ValueError:
        return "unparsable command"
    start = True
    for i, tok in enumerate(tokens):
        if tok in SEPARATORS:
            start = True
            continue
        if start:
            if tok.lower() in BLOCKED:
                return "blocked command: " + tok
            if tok in SHELLS and i > 0 and tokens[i - 1] == "|":
                return "pipe into a shell"
            if tok == "rm" and any(t.startswith("-") and "r" in t.lower() for t in tokens[i + 1:]):
                return "recursive delete is not allowed"
        start = False
        if tok.startswith("/") or tok.startswith("~") or DRIVE.match(tok):
            return "absolute path: " + tok
        if ".." in re.split(r"[\\/]", tok):
            return "parent path: " + tok
    return None
