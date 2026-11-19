"""Harness command policy: block what a training example must never show."""
import shlex

BLOCKED = {"sudo", "su", "curl", "wget", "ssh", "scp", "nc"}
SHELLS = {"sh", "bash", "zsh", "dash"}
SEPARATORS = {"|", "||", "&&", ";", "&"}


def check(cmd: str):
    """Return a reason string if the command is blocked, else None."""
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
            if tok in BLOCKED:
                return "blocked command: " + tok
            if tok in SHELLS and i > 0 and tokens[i - 1] == "|":
                return "pipe into a shell"
            if tok == "rm" and any(t.startswith("-") and "r" in t.lower() for t in tokens[i + 1:]):
                return "recursive delete needs confirmation"
        start = False
        if tok.startswith("/"):
            return "absolute path: " + tok
        if ".." in tok.split("/"):
            return "parent path: " + tok
    return None
