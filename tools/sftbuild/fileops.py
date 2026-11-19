"""Read and edit files inside a build working directory, shared by the terminal and session builders."""
import os


class FileOpError(ValueError):
    pass


def numbered(lines, start):
    return "\n".join("%-4d%s" % (start + i, l) for i, l in enumerate(lines))


def _resolve(workdir, path):
    root = os.path.realpath(workdir)
    full = os.path.realpath(os.path.join(root, path))
    if full != root and not full.startswith(root + os.sep):
        raise FileOpError("path %r is outside the working directory" % path)
    return full


def read_lines(workdir, path, a, b):
    """Lines a..b (1-based, inclusive) of a file; a range that runs past the end stops at the last line,
    like the harness. Returns (a, last line returned, lines)."""
    lines = open(_resolve(workdir, path)).read().splitlines()
    if not 1 <= a <= b or a > len(lines):
        raise FileOpError("range %d-%d is outside %s (%d lines)" % (a, b, path, len(lines)))
    b = min(b, len(lines))
    return a, b, lines[a - 1:b]


def apply_edit(workdir, path, old, new):
    """Replace the one whole line (or run of whole lines) `old` by `new`. Returns (first, last) line numbers."""
    full = _resolve(workdir, path)
    text = open(full).read()
    if text.count(old) != 1:
        raise FileOpError("edit text found %d times (ambiguous or missing) in %s: %r" % (text.count(old), path, old[:60]))
    i = text.index(old)
    s0 = text.rfind("\n", 0, i) + 1
    e0 = text.find("\n", i + len(old))
    e0 = len(text) if e0 < 0 else e0
    if text[s0:i].strip() or text[i + len(old):e0].strip():
        raise FileOpError("edit does not cover whole lines in %s: %r" % (path, old[:60]))
    first = text.count("\n", 0, s0) + 1
    last = first + old.count("\n")
    with open(full, "w") as f:
        f.write(text[:s0] + new + text[e0:])
    return first, last


def write_file(workdir, path, content):
    """Create or overwrite a file (and its folders) inside the working directory."""
    full = _resolve(workdir, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
