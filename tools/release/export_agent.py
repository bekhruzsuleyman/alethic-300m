"""Build a clean public copy of the agent harness only (the alethic-agent repo).

    python -m tools.release.export_agent SRC_REPO OUT_DIR

Copies src/alethic/agent plus every alethic module it imports, the agent tests, the agent launchers and the
installer source; removes private names with the same rules as the full export; scans the result.
"""
import ast
import os
import shutil
import sys

from tools.release import export_public as ex

AGENT_PACKAGE = "alethic.agent"
TEST_PREFIXES = ("test_agent_", "test_shellemu")
CONFIG_FILES = ["model_gqa.yaml"]  # read by a harness test
DEPLOY_FILES = ["21_build_agent_exe.bat", "31_agent_sft3c.bat", "34_preview_ui.bat", "setup_installer.cs"]

README = """# Alethic Agent

A small terminal agent harness for the Alethic-300M language model. It runs the model on your own computer and lets it
read, write and edit files in one folder, run safe shell commands, and search Wikipedia. It asks before anything risky.

- Model weights: https://huggingface.co/bekhruzsuleyman/alethic-300m (SFT-3c is in `sft/sft3c/`)
- Training code, data tools and results: https://github.com/bekhruzsuleyman/alethic-300m

## What is in here

| Path | What it is |
| --- | --- |
| `src/alethic/agent` | the harness: terminal UI, tool loop, safe-command policy, shell emulation, search |
| `src/alethic/model` | the model code the harness loads |
| `tests/` | the harness tests |
| `deploy/31_agent_sft3c.bat` | Windows double-click launcher (downloads the weights on the first run) |
| `deploy/21_build_agent_exe.bat`, `deploy/setup_installer.cs` | builds a Windows installer |

## How it works

The model writes `<plan>`, `<call> tool(args) </call>` and a final answer. The harness runs the call in the work folder,
returns `<result>..</result>`, and shows everything as UI blocks, never as raw tags. Tools: `read`, `write`, `edit`,
`shell`, `search(content, source)`. Reading and listing run freely; deleting, moving and unknown commands ask first.

## Limits

Alethic-300M is small. On my 50-prompt check, SFT-3c passed 19 of 35 graded prompts, and edits at a given line failed
every time. Check anything that matters.

## Tests

    pip install -e .
    pytest

## License

MIT, see `LICENSE`.
"""


def _imports(path):
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append(node.module)
            out += ["%s.%s" % (node.module, a.name) for a in node.names]
    return [m for m in out if m == "alethic" or m.startswith("alethic.")]


def _module_files(root, mod):
    """Files that make up one module name: its .py or package __init__, plus parent package __init__ files."""
    parts = mod.split(".")
    found = []
    for i in range(1, len(parts) + 1):
        base = os.path.join(root, "src", *parts[:i])
        if os.path.isfile(base + ".py"):
            found.append(base + ".py")
        elif os.path.isfile(os.path.join(base, "__init__.py")):
            found.append(os.path.join(base, "__init__.py"))
    return found


def _package_files(root, mod):
    base = os.path.join(root, "src", *mod.split("."))
    out = []
    for r, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ex.SKIP_DIRS]
        out += [os.path.join(r, f) for f in files if f.endswith(".py")]
    return out


def module_closure(root, starts, extra_files=()):
    todo = []
    for s in starts:
        todo += _package_files(root, s)
    todo += list(extra_files)
    seen = set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        for m in _imports(p):
            todo += _module_files(root, m)
    rels = {os.path.relpath(p, root).replace(os.sep, "/") for p in seen if p.startswith(os.path.join(root, "src"))}
    return rels


def _copy(root, out, rel):
    sp = os.path.join(root, rel)
    dp = os.path.join(out, rel)
    os.makedirs(os.path.dirname(dp), exist_ok=True)
    if ex.is_text(rel):
        with open(sp, encoding="utf-8") as fh:
            text = ex.clean_text(fh.read(), rel.startswith("deploy"))
        with open(dp, "wb") as fh:
            fh.write(text.encode("utf-8"))
    else:
        shutil.copyfile(sp, dp)


def export(root, out):
    if os.path.exists(out):
        ex.force_rmtree(out)
    os.makedirs(out)
    files = set(module_closure(root, [AGENT_PACKAGE]))
    tests_dir = os.path.join(root, "tests")
    if os.path.isdir(tests_dir):
        for f in sorted(os.listdir(tests_dir)):
            if f.startswith(TEST_PREFIXES) and f.endswith(".py"):
                files.add("tests/" + f)
    for f in DEPLOY_FILES:
        if os.path.isfile(os.path.join(root, "deploy", f)):
            files.add("deploy/" + f)
    for f in CONFIG_FILES:
        if os.path.isfile(os.path.join(root, "configs", f)):
            files.add("configs/" + f)
    if os.path.isfile(os.path.join(root, "pyproject.toml")):
        files.add("pyproject.toml")
    for rel in sorted(files):
        _copy(root, out, rel)
    for name, text in (("README.md", README), ("LICENSE", ex.LICENSE),
                       (".gitignore", "__pycache__/\n*.pyc\n.venv/\n/work/\n/build/\n/dist/\n*.pt\n*.zip\n")):
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    with open(os.path.join(out, ".alethic_export_ok"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("made by tools.release.export_agent; publish scripts check this file\n")
    return len(files) + 3


def scan(out):
    problems = ex.scan(out)
    import re
    rx = [re.compile(p, re.I) for p in ex.FORBIDDEN]
    for r, _, fs in os.walk(out):
        for f in fs:
            if f.endswith(".cs"):
                p = os.path.join(r, f)
                for i, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
                    for x in rx:
                        if x.search(line):
                            problems.append((os.path.relpath(p, out), i, x.pattern))
    return problems


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    n = export(argv[1], argv[2])
    problems = scan(argv[2])
    print("exported %d files to %s" % (n, argv[2]))
    for rel, line, pat in problems[:40]:
        print("PRIVATE?  %s:%d  matches %s" % (rel, line, pat))
    print("scan: %d problems" % len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
