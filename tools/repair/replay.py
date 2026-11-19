"""Re-run the tool calls of a hand-written example against a real working directory and replace the
hand-invented <result>/<error> blocks by what the tools really return."""
import ast
import operator
import os
import re

from tools.sftbuild import fileops
from tools.sftbuild.sandbox import run as sandbox_run

ROOT = "/work"
CALL = re.compile(r"^<call> (\w+)\((.*)\) </call>$")
TAG = re.compile(r"^<(call|result|error|warn|done|thinking|goal|todo|plan|map)\b")
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}


class ReplayError(ValueError):
    pass


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    raise ReplayError("calc accepts arithmetic only")


def calc(expr):
    try:
        v = _eval(ast.parse(expr.strip(), mode="eval"))
    except SyntaxError as e:
        raise ReplayError("bad calc expression %r" % expr) from e
    return repr(round(v, 10)) if isinstance(v, float) else str(v)


def _list(workdir, path):
    full = fileops._resolve(workdir, path)
    names = sorted(n + ("/" if os.path.isdir(os.path.join(full, n)) else "") for n in os.listdir(full))
    return "\n".join(names) if names else "(empty)"


def _read(workdir, args):
    path, rng = [a.strip() for a in args.rsplit(",", 1)]
    a, b = (int(x) for x in rng.split("-"))
    total = len(open(fileops._resolve(workdir, path)).read().splitlines())
    if a > total:
        raise ReplayError("read starts at line %d but %s has %d lines" % (a, path, total))
    a, b, lines = fileops.read_lines(workdir, path, a, min(b, total))  # the harness stops at the last line
    return fileops.numbered(lines, a)


def _error_line(out, rc):
    m = re.findall(r'File "([^"]+)", line (\d+)', out) or [(a, b) for a, b in re.findall(r"^(\S+):(\d+)$", out, re.M)]
    exc = re.findall(r"^(\w+(?:Error|Exception|Warning)\b.*)$", out, re.M)
    if not m or not exc:
        return "<error> command exited with code %d </error>" % rc
    path, line = m[-1] if 'File "' in out else m[0]
    path = path[len(ROOT) + 1:] if path.startswith(ROOT + "/") else path
    return "<error> %s, line %s: %s </error>" % (path, line, exc[-1])


def _shell(workdir, cmd):
    rc, out = sandbox_run(cmd, workdir)
    out = out.replace(workdir, ROOT)
    res = "<result> %s </result>" % (out if out.strip() else "(no output)")
    return [res] + ([_error_line(out, rc)] if rc != 0 or "Traceback" in out else [])


def _replace_range(workdir, path, a, b, new_lines):
    full = fileops._resolve(workdir, path)
    lines = open(full).read().splitlines()
    if not 1 <= a <= b <= len(lines):
        raise ReplayError("edit range %d-%d is outside %s (%d lines)" % (a, b, path, len(lines)))
    lines[a - 1:b] = new_lines
    open(full, "w").write("\n".join(lines) + "\n")


def replay(body, files, workdir):
    for name, content in files.items():
        full = fileops._resolve(workdir, name)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "w").write(content)
    src, out, i = body.split("\n"), [], 0
    while i < len(src):
        m = CALL.match(src[i])
        if not m:
            out.append(src[i])
            i += 1
            continue
        name, args = m.group(1), m.group(2)
        out.append(src[i])
        i += 1
        if name in ("write", "edit"):
            content = []
            while i < len(src) and not TAG.match(src[i]):
                content.append(src[i])
                i += 1
            out.extend(content)
            if name == "write":
                fileops.write_file(workdir, args.strip(), "\n".join(content) + "\n")
            else:
                path, rng = [a.strip() for a in args.rsplit(",", 1)]
                a, b = (int(x) for x in rng.split("-"))
                _replace_range(workdir, path, a, b, content)
            continue
        if i < len(src) and src[i].startswith("<result>"):  # drop the hand-written result
            while not src[i].rstrip().endswith("</result>"):
                i += 1
            i += 1
            if i < len(src) and src[i].startswith("<error>"):
                i += 1
        if name == "calc":
            out.append("<result> %s </result>" % calc(args))
        elif name == "list":
            out.append("<result> %s </result>" % _list(workdir, args.strip()))
        elif name == "read":
            out.append("<result> %s </result>" % _read(workdir, args))
        elif name == "shell":
            out.extend(_shell(workdir, args))
        else:
            raise ReplayError("unknown tool %r" % name)
    return "\n".join(out)
