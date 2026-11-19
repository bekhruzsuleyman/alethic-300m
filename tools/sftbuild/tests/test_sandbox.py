import pytest
import tempfile

from tools.sftbuild.sandbox import run


def test_lc_all_c():
    with tempfile.TemporaryDirectory() as d:
        assert run("echo $LC_ALL", d) == (0, "C")


def test_timeout_returns_124():
    with tempfile.TemporaryDirectory() as d:
        rc, _ = run("sleep 20", d, timeout=1)
        assert rc == 124


def test_output_cap_marker():
    with tempfile.TemporaryDirectory() as d:
        rc, out = run("python3 -c \"print('x' * 500)\"", d, cap=100)
        assert out.endswith("[truncated]") and len(out) <= 100 + len("\n[truncated]")


def test_empty_output_is_empty_string():
    with tempfile.TemporaryDirectory() as d:
        assert run("true", d) == (0, "")


def test_exit_code_and_stderr_combined():
    with tempfile.TemporaryDirectory() as d:
        rc, out = run("echo oops >&2; exit 3", d)
        assert (rc, out) == (3, "oops")


def test_proxy_vars_hidden_by_default_and_passed_with_network(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:1")
    with tempfile.TemporaryDirectory() as d:
        assert run("echo ${HTTPS_PROXY:-none}", d) == (0, "none")
        assert run("echo ${HTTPS_PROXY:-none}", d, network=True) == (0, "http://proxy.invalid:1")


def test_timings_are_normalized_so_builds_are_reproducible():
    with tempfile.TemporaryDirectory() as d:
        rc, out = run("echo 'Ran 11 tests in 0.002s'; echo '3 passed in 0.12s'", d)
        assert out == "Ran 11 tests in 0.001s\n3 passed in 0.001s"


def test_no_bytecode_files_are_written():
    with tempfile.TemporaryDirectory() as d:
        open(d + "/m.py", "w").write("x = 1\n")
        run("python -c 'import m'", d)
        assert run("ls -a", d)[1].split() == [".", "..", "m.py"]


def test_same_size_edit_in_same_second_is_picked_up():
    with tempfile.TemporaryDirectory() as d:
        open(d + "/m.py", "w").write("value = 'aaaa'\n")
        open(d + "/main.py", "w").write("import m\nprint(m.value)\n")
        assert run("python main.py", d) == (0, "aaaa")
        open(d + "/m.py", "w").write("value = 'bbbb'\n")  # same size, same second
        assert run("python main.py", d) == (0, "bbbb")


import shutil

TOOLS = {"rustc": "rustc --version", "go": "go version", "gcc": "gcc --version", "node": "node --version",
         "tsc": "tsc --version", "cargo": "cargo --version"}


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_toolchains_found_on_the_builder_path_work_inside_the_sandbox(tmp_path, tool):
    if shutil.which(tool) is None:
        pytest.skip("%s is not installed here" % tool)
    rc, out = run(TOOLS[tool], str(tmp_path))
    assert rc == 0, out


def test_go_and_rust_builds_do_not_litter_the_working_directory(tmp_path):
    if shutil.which("go") is None or shutil.which("rustc") is None:
        pytest.skip("go or rustc missing")
    (tmp_path / "m.go").write_text('package main\nimport "fmt"\nfunc main() { fmt.Println("hi") }\n')
    (tmp_path / "m.rs").write_text('fn main() { println!("hi"); }\n')
    assert run("go run m.go", str(tmp_path))[1] == "hi"
    assert run("rustc m.rs -o app && ./app", str(tmp_path))[1] == "hi"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["app", "m.go", "m.rs"]
