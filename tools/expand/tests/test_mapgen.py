import re
import ast

from tools.expand import mapgen


def module(n_funcs=30, prefix="f"):
    """n_funcs top-level functions of 8 lines each with a unique docstring: about 240 lines."""
    out = []
    for i in range(n_funcs):
        out += ["def %s%d(x, y=1, *rest):" % (prefix, i), '    """Compute value number %s%d from the input."""' % (prefix, i),
                "    a = x + %d" % i, "    b = a * y", "    c = b - 1", "    d = c + len(rest)", "    return d", ""]
    return "\n".join(out) + "\n"


def flat(sample):
    return [(s["t"], s["loss"]) for s in sample["segments"]]


def text(samples):
    return "\n".join(t for s in samples for t, _ in flat(s))


def test_map_example_has_growing_maps_true_answer_and_stubbed_reads():
    src = module(30)
    samples = mapgen.build_f_map("big.py", src, target="f7", lang="en")
    assert len(samples) >= 2
    maps = [t for s in samples for t, _ in flat(s) if t.startswith("<map> ")]
    assert maps and "f0 1-7" in maps[-1]
    node = [n for n in ast.parse(src).body if n.name == "f7"][0]
    assert ("`f7`, lines %d-%d." % (node.lineno, node.end_lineno)) in text(samples)
    last = flat(samples[-1])
    assert any("(dropped; see map)" in t for t, _ in last)
    assert all(loss == 0 for t, loss in last if t.startswith("<result>"))
    assert any(t.startswith("<map> ") and loss == 1 for t, loss in flat(samples[0]))


def test_control_reads_only_the_section_and_never_maps():
    src = module(30)
    samples = mapgen.build_f_control("big.py", src, target="f9", lang="en")
    assert len(samples) == 1
    t = text(samples)
    assert "<map>" not in t and "<call> read(big.py, 73-79) </call>" in t
    assert "takes: x, y, *rest." in t


def test_small_file_is_read_whole_without_a_map():
    src = "def add(a, b=2):\n    \"\"\"Add two numbers together.\"\"\"\n    return a + b\n"
    samples = mapgen.build_f_small("add.py", src, target="add", lang="en")
    t = text(samples)
    assert "<map>" not in t and "read(add.py, 1-3)" in t and "`add` takes 2 parameters." in t


def test_multifile_map_joins_files_and_answer_names_file():
    files = {"a.py": module(20, "a"), "b.py": module(20, "b"), "c.py": module(20, "c")}
    samples = mapgen.build_g_map(files, ("b.py", "b11"), lang="en")
    last_map = [t for s in samples for t, _ in flat(s) if t.startswith("<map> ")][-1]
    assert " | " in last_map and "a.py: " in last_map and "b.py: " in last_map and "c.py: " in last_map
    assert "`b.py`: `b11`, lines 89-95." in text(samples)
    assert "<call> read(b.py, 89-95) </call>" in text(samples)


def test_make_f_is_deterministic_and_mix_has_controls_and_small_files(tmp_path):
    for i in range(6):
        (tmp_path / ("m%d.py" % i)).write_text(module(30, "g%d_" % i))
    a = mapgen.make_f(str(tmp_path), 60, seed=3)
    b = mapgen.make_f(str(tmp_path), 60, seed=3)
    assert [s["id"] for s in a] == [s["id"] for s in b]
    counts = {}
    for k in mapgen.make_f(str(tmp_path), 60, seed=3, kinds_only=True):
        counts[k] = counts.get(k, 0) + 1
    assert counts["control"] >= 0.3 * 60 and counts["small"] >= 0.1 * 60 and counts["map"] >= 0.4 * 60
    assert all(s["flags"] for s in a)


def test_g_picks_packages_with_several_files(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    for n in "abc":
        (pkg / (n + ".py")).write_text(module(20, n))
    (tmp_path / "lonely.py").write_text(module(20, "z"))
    out = mapgen.make_g(str(tmp_path), 10, seed=1)
    maps = [t for s in out for t, _ in flat(s) if t.startswith("<map> ")]
    assert maps and all(re.search(r"[abc]\.py: ", t) for t in maps)


def test_g_with_class_only_files_does_not_crash_on_controls(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    for n in "abc":
        body = []
        for i in range(20):
            body += ["class %s%d:" % (n, i), '    """Holds the data number %s%d for later use."""' % (n, i),
                     "    x = 1", "    y = 2", "    z = 3", "    w = 4", "    v = 5", ""]
        (pkg / (n + ".py")).write_text("\n".join(body) + "\n")
    out = mapgen.make_g(str(tmp_path), 40, seed=2)
    assert out
