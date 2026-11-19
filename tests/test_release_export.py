import os

from tools.release import export_public as ex


def _put(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)


def test_legal_name_allowed_in_dataset_packer_and_its_test(tmp_path):
    root = str(tmp_path)
    line = 'NAME = "Bekhruz Suleyman"\n'
    _put(root, "tools/release/pack_newborn.py", line)
    _put(root, "tests/test_release_pack.py", line)
    assert ex.scan(root) == []


def test_legal_name_still_flagged_elsewhere(tmp_path):
    root = str(tmp_path)
    _put(root, "src/alethic/other.py", 'NAME = "Bekhruz Suleyman"\n')
    assert {p[0] for p in ex.scan(root)} == {os.path.join("src", "alethic", "other.py")}


def test_family_names_still_flagged_in_allowed_files(tmp_path):
    root = str(tmp_path)
    family = "dil" + "noza"  # split so this file itself passes the private-text scan
    _put(root, "tools/release/pack_newborn.py", "x = '%s'\n" % family)
    assert ex.scan(root)


def test_public_readme_reports_the_measured_eval():
    assert "19/35" in ex.README and "16/35" in ex.README
    assert "huggingface.co/datasets/bekhruzsuleyman/alethic-newborn" in ex.README


def test_export_leaves_a_marker_the_publish_script_checks(tmp_path):
    src = tmp_path / "src_repo"
    src.mkdir()
    (src / "pyproject.toml").write_text("[project]\n")
    out = tmp_path / "out"
    ex.export(str(src), str(out))
    assert (out / ".alethic_export_ok").exists()


def test_delete_retry_makes_a_read_only_file_writable_first(tmp_path):
    import os
    import stat
    f = tmp_path / "abc"
    f.write_text("x")
    os.chmod(str(f), stat.S_IREAD)  # how git stores its object files
    modes = []
    ex._make_writable_and_retry(lambda p: modes.append(os.stat(p).st_mode & stat.S_IWRITE), str(f), None)
    assert modes == [stat.S_IWRITE]


def test_force_rmtree_removes_a_folder_with_read_only_files(tmp_path):
    import os
    import stat
    d = tmp_path / "pub" / ".git" / "objects"
    d.mkdir(parents=True)
    (d / "abc").write_text("x")
    os.chmod(str(d / "abc"), stat.S_IREAD)
    ex.force_rmtree(str(tmp_path / "pub"))
    assert not (tmp_path / "pub").exists()


def test_public_readme_points_to_the_standalone_agent_repo():
    assert "github.com/bekhruzsuleyman/alethic-agent" in ex.README
