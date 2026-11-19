import textwrap

from tools.sftbuild.mapper import ast_map, map_end_accuracy

SRC = textwrap.dedent('''\
    import os
    X = 1


    def a():
        return 1


    class B:
        def m(self):
            return 2

        def n(self):
            return 3


    def c():
        y = 1
        return y
''')
# lines: a 5-6, B 9-14, c 17-19


def _path(tmp_path, text=SRC):
    p = tmp_path / "m.py"
    p.write_text(text)
    return str(p)


def test_closed_and_open(tmp_path):
    assert ast_map(_path(tmp_path), 17) == "a 5-6 -> B 9-14 -> c 17-"


def test_all_closed_when_read_to_end(tmp_path):
    assert ast_map(_path(tmp_path), 19) == "a 5-6 -> B 9-14 -> c 17-19"


def test_only_top_level_functions_and_classes(tmp_path):
    assert "m " not in ast_map(_path(tmp_path), 19) and "X" not in ast_map(_path(tmp_path), 19)


def test_nothing_started_yet(tmp_path):
    assert ast_map(_path(tmp_path), 3) == ""


def test_syntax_error_returns_none(tmp_path):
    assert ast_map(_path(tmp_path, "def (:\n"), 10) is None


def test_accuracy_counts_only_closed_entries(tmp_path):
    p = _path(tmp_path)
    assert map_end_accuracy("a 5-6 -> B 9-13 -> c 17-", p) == 0.5
    assert map_end_accuracy("a 5-6 -> B 9-14 -> c 17-19", p) == 1.0
