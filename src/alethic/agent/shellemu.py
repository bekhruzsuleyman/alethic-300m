"""Shell commands for Windows, written in Python, so `mv a b`, `grep -n x f`, `head -n 3 f`, `ls | wc -l` and
`echo hi > f` work there and print exactly what the posix shell printed when the training data was made.

Only plain commands: words, quotes, `*` globs, `|` pipes and `>` / `>>` into a file. Anything else (variables,
`&&`, `;`, `<`, backticks, flags that are not supported) returns None, so the caller runs it in cmd.exe as before.
Every path goes through `resolve`, which refuses anything outside the work folder.
"""
from __future__ import annotations

import errno
import fnmatch
import glob
import os
import re
import shutil

WHY = {errno.ENOENT: "No such file or directory", errno.EISDIR: "Is a directory", errno.ENOTDIR: "Not a directory",
       errno.EEXIST: "File exists", errno.ENOTEMPTY: "Directory not empty", errno.EACCES: "Permission denied"}


class Unsupported(Exception):
    """This command line needs the real shell."""


def why(exc: OSError) -> str:
    return WHY.get(exc.errno, os.strerror(exc.errno) if exc.errno else str(exc))


def lex(cmd: str):
    """Split into ('word', text, globby) and ('op', '|' | '>' | '>>') tokens. None when the line needs a real shell."""
    tokens, cur, has, glob_ok, quote, i = [], "", False, False, None, 0

    def flush():
        nonlocal cur, has, glob_ok
        if has or cur:
            tokens.append(("word", cur, glob_ok))
        cur, has, glob_ok = "", False, False

    while i < len(cmd):
        c = cmd[i]
        if quote == "'":
            if c == "'":
                quote = None
            else:
                cur += c
        elif quote == '"':
            if c == '"':
                quote = None
            elif c == "\\" and i + 1 < len(cmd) and cmd[i + 1] in '"\\$`':
                cur += cmd[i + 1]
                i += 1
            elif c in "$`":
                return None
            else:
                cur += c
        elif c in "'\"":
            quote, has = c, True
        elif c == "\\":
            if i + 1 < len(cmd):
                cur += cmd[i + 1]
                has = True
                i += 1
        elif c.isspace():
            flush()
        elif c in "$`&;<()~" and not (c == "~" and cur):
            return None
        elif c == "|":
            if cmd[i + 1:i + 2] == "|":
                return None
            flush()
            tokens.append(("op", "|"))
        elif c == ">":
            if cur.isdigit() and not has:
                return None  # 2>&1 and friends
            flush()
            if cmd[i + 1:i + 2] == ">":
                tokens.append(("op", ">>"))
                i += 1
            else:
                tokens.append(("op", ">"))
        elif c == "#" and not cur and not has:
            break
        elif c in "*?[":
            cur += c
            glob_ok = True
        else:
            cur += c
        i += 1
    if quote:
        return None
    flush()
    return tokens


class Emu:
    def __init__(self, resolve, root):
        self.resolve, self.root = resolve, root
        self.commands = {"pwd": self.pwd, "ls": self.ls, "cat": self.cat, "mkdir": self.mkdir, "touch": self.touch,
                         "echo": self.echo, "mv": self.mv, "cp": self.cp, "rm": self.rm, "head": self.head,
                         "tail": self.tail, "wc": self.wc, "grep": self.grep, "find": self.find}

    # the command line ----------------------------------------------------
    def run(self, cmd: str):
        """The text the shell would have printed, or None if this line is not for the emulation."""
        tokens = lex(cmd)
        if not tokens:
            return None
        stages, cur = [], []
        for kind, *rest in tokens:
            if kind == "op" and rest[0] == "|":
                stages.append(cur)
                cur = []
            else:
                cur.append((kind, *rest))
        stages.append(cur)
        redirect = None
        last = stages[-1]
        for i, tok in enumerate(last):
            if tok[0] == "op":
                if i != len(last) - 2 or last[-1][0] != "word" or redirect:
                    return None
                redirect = (tok[1], last[-1][1])
                stages[-1] = last[:i]
                break
        words = []
        for stage in stages:
            if not stage or any(t[0] == "op" for t in stage):
                return None
            words.append(self._expand(stage))
        if any(w[0] not in self.commands for w in words):
            return None
        try:
            data, errs = None, []
            for w in words:
                data, err = self.commands[w[0]](w[1:], data)
                errs.append(err)
            out = data or ""
            if redirect:
                mode, name = redirect
                try:
                    with open(self.resolve(name), "a" if mode == ">>" else "w", encoding="utf-8", newline="\n") as f:
                        f.write(out)
                except ValueError as exc:
                    errs.append("%s\n" % exc)
                except OSError as exc:
                    errs.append("%s: %s\n" % (name, why(exc)))
                out = ""
            return (out + "".join(errs)).strip()
        except Unsupported:
            return None

    def _expand(self, stage):
        out = []
        for _, text, globby in stage:
            if globby and not (text.startswith(("/", "\\")) or ".." in text or ":" in text):
                hits = sorted(glob.glob(text, root_dir=self.root))
                out.extend(h.replace("\\", "/") for h in hits) if hits else out.append(text)
            else:
                out.append(text)
        return out

    # helpers --------------------------------------------------------------
    def _path(self, name, cmd):
        """Resolve inside the work folder; ValueError text becomes the command's error."""
        return self.resolve(name)

    def _read(self, name):
        with open(self.resolve(name), "rb") as f:
            return f.read().decode("utf-8", "replace")

    @staticmethod
    def _split(text):
        """Lines of a text; the last one has no newline when the text did not end with one."""
        return text.splitlines(keepends=True) if text else []

    @staticmethod
    def _opts(args, flags, valued=""):
        """GNU-style options: -abc, -n 3, -n3. Returns (flag set, {letter: value}, operands). Unknown letters: Unsupported."""
        got, vals, rest, i, done = set(), {}, [], 0, False
        while i < len(args):
            a = args[i]
            if done or not a.startswith("-") or a == "-":
                rest.append(a)
            elif a == "--":
                done = True
            else:
                j = 1
                while j < len(a):
                    ch = a[j]
                    if ch in valued:
                        if a[j + 1:]:
                            vals.setdefault(ch, []).append(a[j + 1:])
                        elif i + 1 < len(args):
                            i += 1
                            vals.setdefault(ch, []).append(args[i])
                        else:
                            raise Unsupported
                        break
                    if ch not in flags:
                        raise Unsupported
                    got.add(ch)
                    j += 1
            i += 1
        return got, vals, rest

    # simple commands ---------------------------------------------------------
    def pwd(self, args, data):
        return self.root + "\n", ""

    def echo(self, args, data):
        nl = True
        if args and args[0] == "-n":
            nl, args = False, args[1:]
        elif args and args[0].startswith("-") and len(args[0]) > 1 and set(args[0][1:]) <= set("neE"):
            raise Unsupported
        return " ".join(args) + ("\n" if nl else ""), ""

    def cat(self, args, data):
        if not args:
            return data or "", ""
        out, err = [], []
        for a in args:
            try:
                out.append(self._read(a))
            except ValueError as exc:
                err.append("cat: %s\n" % exc)
            except OSError as exc:
                err.append("cat: %s: %s\n" % (a, why(exc)))
        return "".join(out), "".join(err)

    def mkdir(self, args, data):
        got, _, names = self._opts(args, "p")
        err = []
        for n in names:
            try:
                full = self.resolve(n)
                if "p" in got:
                    os.makedirs(full, exist_ok=True)
                else:
                    os.mkdir(full)
            except ValueError as exc:
                err.append("mkdir: %s\n" % exc)
            except OSError as exc:
                err.append("mkdir: cannot create directory \u2018%s\u2019: %s\n" % (n, why(exc)))
        return "", "".join(err)

    def touch(self, args, data):
        _, _, names = self._opts(args, "")
        err = []
        for n in names:
            try:
                full = self.resolve(n)
                with open(full, "a"):
                    pass
                os.utime(full, None)
            except ValueError as exc:
                err.append("touch: %s\n" % exc)
            except OSError as exc:
                err.append("touch: cannot touch '%s': %s\n" % (n, why(exc)))
        return "", "".join(err)

    def rm(self, args, data):
        got, _, names = self._opts(args, "f")
        err = []
        for n in names:
            try:
                full = self.resolve(n)
                if os.path.isdir(full) and not os.path.islink(full):
                    err.append("rm: cannot remove '%s': Is a directory\n" % n)
                    continue
                os.remove(full)
            except ValueError as exc:
                err.append("rm: %s\n" % exc)
            except OSError as exc:
                if not ("f" in got and exc.errno == errno.ENOENT):
                    err.append("rm: cannot remove '%s': %s\n" % (n, why(exc)))
        return "", "".join(err)

    def _targets(self, cmd, names):
        """(sources, destination) for mv and cp."""
        if len(names) < 2:
            raise Unsupported
        return names[:-1], names[-1]

    def mv(self, args, data):
        _, _, names = self._opts(args, "f")
        srcs, dest = self._targets("mv", names)
        err = []
        try:
            dest_full = self.resolve(dest)
        except ValueError as exc:
            return "", "mv: %s\n" % exc
        into = os.path.isdir(dest_full)
        if len(srcs) > 1 and not into:
            return "", "mv: target '%s' is not a directory\n" % dest
        for s in srcs:
            try:
                src = self.resolve(s)
                if not os.path.lexists(src):
                    err.append("mv: cannot stat '%s': No such file or directory\n" % s)
                    continue
                target = os.path.join(dest_full, os.path.basename(src)) if into else dest_full
                if not os.path.isdir(os.path.dirname(target)):
                    err.append("mv: cannot move '%s' to '%s': No such file or directory\n" % (s, dest))
                    continue
                if os.path.isdir(target) and not os.path.isdir(src):
                    err.append("mv: cannot overwrite directory '%s' with non-directory\n" % dest)
                    continue
                shutil.move(src, target) if not os.path.isfile(target) else os.replace(src, target)
            except ValueError as exc:
                err.append("mv: %s\n" % exc)
            except OSError as exc:
                err.append("mv: cannot move '%s' to '%s': %s\n" % (s, dest, why(exc)))
        return "", "".join(err)

    def cp(self, args, data):
        got, _, names = self._opts(args, "rRf")
        recursive = bool(got & {"r", "R"})
        srcs, dest = self._targets("cp", names)
        err = []
        try:
            dest_full = self.resolve(dest)
        except ValueError as exc:
            return "", "cp: %s\n" % exc
        into = os.path.isdir(dest_full)
        if len(srcs) > 1 and not into:
            return "", "cp: target '%s' is not a directory\n" % dest
        for s in srcs:
            try:
                src = self.resolve(s)
                if not os.path.exists(src):
                    err.append("cp: cannot stat '%s': No such file or directory\n" % s)
                    continue
                if os.path.isdir(src) and not recursive:
                    err.append("cp: -r not specified; omitting directory '%s'\n" % s)
                    continue
                target = os.path.join(dest_full, os.path.basename(src)) if into else dest_full
                if not os.path.isdir(os.path.dirname(target)):
                    err.append("cp: cannot create %s '%s': No such file or directory\n"
                               % ("directory" if os.path.isdir(src) else "regular file", dest))
                    continue
                if os.path.isdir(src):
                    shutil.copytree(src, target, dirs_exist_ok=True)
                else:
                    shutil.copyfile(src, target)
            except ValueError as exc:
                err.append("cp: %s\n" % exc)
            except OSError as exc:
                err.append("cp: cannot create regular file '%s': %s\n" % (dest, why(exc)))
        return "", "".join(err)

    # reading ----------------------------------------------------------------
    def ls(self, args, data):
        got, _, names = self._opts(args, "a1A")
        names = names or ["."]
        out, err, files, dirs = [], [], [], []
        for n in names:
            try:
                full = self.resolve(n)
            except ValueError as exc:
                err.append("ls: %s\n" % exc)
                continue
            if not os.path.lexists(full):
                err.append("ls: cannot access '%s': No such file or directory\n" % n)
            elif os.path.isdir(full):
                dirs.append((n, full))
            else:
                files.append(n)
        blocks = []
        if files:
            blocks.append("\n".join(sorted(files)) + "\n")
        for n, full in sorted(dirs):
            entries = os.listdir(full)
            if "a" in got:
                entries += [".", ".."]
            elif "A" not in got:
                entries = [e for e in entries if not e.startswith(".")]
            body = "\n".join(sorted(entries)) + "\n" if entries else ""
            blocks.append(("%s:\n" % n if len(names) > 1 else "") + body)
        text = "\n".join(blocks)
        return text, "".join(err)

    def head(self, args, data):
        return self._headtail(args, data, "head")

    def tail(self, args, data):
        return self._headtail(args, data, "tail")

    def _headtail(self, args, data, which):
        count, rest, i = 10, [], 0
        while i < len(args):
            a = args[i]
            if a == "-n" and i + 1 < len(args):
                count, i = args[i + 1], i + 1
            elif re.fullmatch(r"-n[+-]?\d+", a):
                count = a[2:]
            elif re.fullmatch(r"-\d+", a):
                count = a[1:]
            elif a.startswith("-") and a != "-" and not (a == "--"):
                raise Unsupported
            else:
                rest.append(a)
            i += 1
        count = str(count)
        if not re.fullmatch(r"[+-]?\d+", count):
            raise Unsupported
        n, plus, minus = abs(int(count)), count.startswith("+"), count.startswith("-")

        def pick(lines):
            if which == "head":
                return lines[:len(lines) - n] if minus else lines[:n]
            if plus:
                return lines[max(n - 1, 0):]
            return lines[len(lines) - n:] if n else []

        if not rest:
            return "".join(pick(self._split(data or ""))), ""
        out, err = [], []
        for k, name in enumerate(rest):
            try:
                text = self._read(name)
            except ValueError as exc:
                err.append("%s: %s\n" % (which, exc))
                continue
            except IsADirectoryError:
                if len(rest) > 1:
                    out.append(("\n" if k else "") + "==> %s <==\n" % name)
                err.append("%s: error reading '%s': Is a directory\n" % (which, name))
                continue
            except OSError as exc:
                err.append("%s: cannot open '%s' for reading: %s\n" % (which, name, why(exc)))
                continue
            if len(rest) > 1:
                out.append(("\n" if out else "") + "==> %s <==\n" % name)
            body = "".join(pick(self._split(text)))
            out.append(body + ("\n" if body and not body.endswith("\n") and len(rest) > 1 else ""))
        return "".join(out), "".join(err)

    def wc(self, args, data):
        got, _, names = self._opts(args, "lwcm")
        cols = [c for c in "lwc" if c in got or (c == "c" and "m" in got)] or list("lwc")
        rows, err = [], []
        sources = names if names else [None]
        for name in sources:
            try:
                raw = (data or "").encode("utf-8") if name is None else open(self.resolve(name), "rb").read()
            except ValueError as exc:
                err.append("wc: %s\n" % exc)
                continue
            except OSError as exc:
                err.append("wc: %s: %s\n" % (name, why(exc)))
                continue
            text = raw.decode("utf-8", "replace")
            counts = {"l": raw.count(b"\n"), "w": len(text.split()), "c": len(raw)}
            rows.append((counts, name, len(raw)))
        if len(rows) > 1:
            rows.append(({c: sum(r[0][c] for r in rows) for c in "lwc"}, "total", sum(r[2] for r in rows)))
        if len(sources) == 1 and len(cols) == 1:
            width = 1
        elif names:
            width = len(str(sum(os.path.getsize(self.resolve(n)) for n in names if os.path.isfile(self.resolve(n))))) or 1
        else:
            width = 7
        lines = []
        for counts, name, _ in rows:
            line = " ".join(str(counts[c]).rjust(width) for c in cols)
            lines.append(line + (" " + name if name else ""))
        return "\n".join(lines) + "\n" if lines else "", "".join(err)

    # grep -------------------------------------------------------------------
    @staticmethod
    def _bre(p):
        """Basic regular expression (grep's default) to Python: + ? | ( ) { } are plain, their backslash forms are operators."""
        out, i = "", 0
        while i < len(p):
            c = p[i]
            if c == "\\" and i + 1 < len(p):
                nxt = p[i + 1]
                out += nxt if nxt in "+?|(){}" else "\\" + nxt
                i += 2
                continue
            out += "\\" + c if c in "+?|(){}" else c
            i += 1
        return out

    def grep(self, args, data):
        got, vals, rest = self._opts(args, "nivclrRwFEhHs", valued="e")
        if vals.get("e"):
            pats = vals["e"]
        elif rest:
            pats, rest = [rest[0]], rest[1:]
        else:
            raise Unsupported
        parts = [re.escape(p) if "F" in got else (p if "E" in got else self._bre(p)) for p in pats]
        body = "|".join("(?:%s)" % p for p in parts)
        if "w" in got:
            body = r"(?<!\w)(?:%s)(?!\w)" % body
        try:
            rx = re.compile(body, re.I if "i" in got else 0)
        except re.error:
            raise Unsupported
        recursive = bool(got & {"r", "R"})
        operands = rest or (["."] if recursive else [])
        strip_dot = recursive and not rest
        files, err = [], []
        for op in operands or [None]:
            if op is None:
                files.append((None, None))
                continue
            try:
                full = self.resolve(op)
            except ValueError as exc:
                err.append("grep: %s\n" % exc)
                continue
            if os.path.isdir(full):
                if not recursive:
                    err.append("grep: %s: Is a directory\n" % op)
                    continue
                for base, dirs, names in os.walk(full):
                    dirs.sort()
                    for nm in sorted(names):
                        shown = os.path.join(os.path.relpath(base, full), nm) if base != full else nm
                        shown = shown.replace("\\", "/")
                        files.append((("" if strip_dot else op.rstrip("/") + "/") + shown, os.path.join(base, nm)))
            elif not os.path.exists(full):
                err.append("grep: %s: No such file or directory\n" % op)
            else:
                files.append((op, full))
        has_dir = recursive and any(op is not None and os.path.isdir(self.resolve(op)) for op in operands)
        show = ("H" in got or len(operands) > 1 or has_dir) and "h" not in got
        out = []
        for name, full in files:
            if full is None:
                text = data or ""
            else:
                with open(full, "rb") as f:
                    text = f.read().decode("utf-8", "replace")
            hits = []
            for n, line in enumerate(text.split("\n")[:-1] if text.endswith("\n") else text.split("\n"), 1):
                if bool(rx.search(line)) != ("v" in got):
                    hits.append((n, line))
            prefix = (name + ":") if show and name else ""
            if "l" in got:
                if hits:
                    out.append(name + "\n")
            elif "c" in got:
                out.append("%s%d\n" % (prefix, len(hits)))
            else:
                for n, line in hits:
                    out.append("%s%s%s\n" % (prefix, "%d:" % n if "n" in got else "", line))
        return "".join(out), "".join(err)

    # find -------------------------------------------------------------------
    def find(self, args, data):
        paths, i = [], 0
        while i < len(args) and not args[i].startswith(("-", "!", "(")):
            paths.append(args[i])
            i += 1
        tests, maxd, mind = [], None, 0
        while i < len(args):
            a = args[i]
            if a in ("-name", "-iname", "-type", "-maxdepth", "-mindepth") and i + 1 < len(args):
                v = args[i + 1]
                if a == "-maxdepth":
                    maxd = int(v) if v.isdigit() else self._unsupported()
                elif a == "-mindepth":
                    mind = int(v) if v.isdigit() else self._unsupported()
                elif a == "-type":
                    if v not in ("f", "d"):
                        raise Unsupported
                    tests.append((a, v))
                else:
                    tests.append((a, v))
                i += 2
            elif a == "-print":
                i += 1
            else:
                raise Unsupported
        out, err = [], []
        for p in paths or ["."]:
            try:
                full = self.resolve(p)
            except ValueError as exc:
                err.append("find: %s\n" % exc)
                continue
            if not os.path.lexists(full):
                err.append("find: \u2018%s\u2019: No such file or directory\n" % p)
                continue
            self._walk(p, full, 0, tests, maxd, mind, out)
        return "".join(l + "\n" for l in out), "".join(err)

    @staticmethod
    def _unsupported():
        raise Unsupported

    def _walk(self, shown, full, depth, tests, maxd, mind, out):
        if depth >= mind and self._match(shown, full, tests):
            out.append(shown)
        if os.path.isdir(full) and not os.path.islink(full) and (maxd is None or depth < maxd):
            for name in sorted(os.listdir(full)):
                self._walk(shown.rstrip("/") + "/" + name if shown != "/" else "/" + name, os.path.join(full, name),
                           depth + 1, tests, maxd, mind, out)

    @staticmethod
    def _match(shown, full, tests):
        base = os.path.basename(shown.rstrip("/")) or shown
        for kind, v in tests:
            if kind == "-name" and not fnmatch.fnmatchcase(base, v):
                return False
            if kind == "-iname" and not fnmatch.fnmatchcase(base.lower(), v.lower()):
                return False
            if kind == "-type" and (os.path.isdir(full) != (v == "d")):
                return False
        return True
