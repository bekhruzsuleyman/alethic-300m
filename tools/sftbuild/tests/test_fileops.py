import pytest

from tools.sftbuild.fileops import FileOpError, apply_edit, numbered, read_lines


def test_numbered_matches_the_read_format():
    assert numbered(["a", "b"], 9) == "9   a\n10  b"


def test_read_lines_returns_the_requested_range(tmp_path):
    (tmp_path / "f.txt").write_text("a\nb\nc\n")
    assert read_lines(str(tmp_path), "f.txt", 2, 3) == (2, 3, ["b", "c"])


def test_read_lines_past_the_end_stops_at_the_last_line(tmp_path):
    (tmp_path / "f.txt").write_text("a\nb\n")
    assert read_lines(str(tmp_path), "f.txt", 1, 9) == (1, 2, ["a", "b"])


def test_read_lines_starting_past_the_end_raises(tmp_path):
    (tmp_path / "f.txt").write_text("a\nb\n")
    with pytest.raises(FileOpError, match="range"):
        read_lines(str(tmp_path), "f.txt", 3, 4)


def test_apply_edit_replaces_one_whole_line(tmp_path):
    (tmp_path / "f.txt").write_text("a\nb\nc\n")
    assert apply_edit(str(tmp_path), "f.txt", "b", "B") == (2, 2)
    assert (tmp_path / "f.txt").read_text() == "a\nB\nc\n"


@pytest.mark.parametrize("old", ["x", "a"])
def test_apply_edit_missing_or_ambiguous_raises(tmp_path, old):
    (tmp_path / "f.txt").write_text("a\na\nc\n")
    with pytest.raises(FileOpError, match="found"):
        apply_edit(str(tmp_path), "f.txt", old, "z")


def test_apply_edit_partial_line_raises(tmp_path):
    (tmp_path / "f.txt").write_text("hello world\n")
    with pytest.raises(FileOpError, match="whole lines"):
        apply_edit(str(tmp_path), "f.txt", "hello", "bye")


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "sub/../../x.txt"])
def test_paths_cannot_leave_the_workdir(tmp_path, path):
    work = tmp_path / "w"
    work.mkdir()
    (tmp_path / "outside.txt").write_text("secret\n")
    with pytest.raises(FileOpError, match="outside"):
        read_lines(str(work), path, 1, 1)
    with pytest.raises(FileOpError, match="outside"):
        apply_edit(str(work), path, "secret", "x")


def test_write_file_creates_folders_and_content(tmp_path):
    from tools.sftbuild.fileops import write_file
    write_file(str(tmp_path), "src/main.rs", "fn main() {}\n")
    assert (tmp_path / "src" / "main.rs").read_text() == "fn main() {}\n"


@pytest.mark.parametrize("path", ["../x.txt", "/tmp/x.txt", "a/../../x.txt"])
def test_write_file_cannot_leave_the_workdir(tmp_path, path):
    from tools.sftbuild.fileops import write_file
    work = tmp_path / "w"
    work.mkdir()
    with pytest.raises(FileOpError, match="outside"):
        write_file(str(work), path, "x\n")
    assert not (tmp_path / "x.txt").exists()
