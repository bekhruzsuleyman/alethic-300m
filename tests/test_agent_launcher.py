import os

from alethic.agent.cli import build_parser
from alethic.agent.launcher import app_dirs, build_argv, ensure_weights


def norm(p):
    return os.path.normpath(p)


def test_frozen_app_keeps_data_next_to_the_exe_and_bundle_inside():
    bundle, data = app_dirs(frozen=True, exe=os.path.join("C:\\", "app", "alethic-agent.exe"), meipass="BUNDLE", source=__file__)
    assert bundle == "BUNDLE" and norm(data) == norm(os.path.join("C:\\", "app"))


def test_source_run_uses_the_repo_root_for_both():
    src = os.path.join("repo", "src", "alethic", "agent", "launcher.py")
    bundle, data = app_dirs(frozen=False, exe="python", meipass=None, source=src)
    assert norm(bundle) == norm(data) == "repo"


def test_default_arguments_point_at_weights_tokenizer_config_and_work_folder():
    a = build_parser().parse_args(build_argv([], "B", "D"))
    assert norm(a.weights) == norm("D/model-sft3b.pt") and a.folder is None and norm(a.suggest) == norm("D/work")
    assert a.tokenizer.endswith("alethic-tokenizer-v1.model") and a.tokenizer.startswith("B")
    assert norm(a.model) == norm("B/configs/model_gqa.yaml")


def test_arguments_typed_by_the_user_win_over_the_defaults():
    a = build_parser().parse_args(build_argv(["--effort", "high", "--folder", "other"], "B", "D"))
    assert a.effort == "high" and a.folder == "other"


def test_weights_are_downloaded_once_and_copied_next_to_the_app(tmp_path):
    cached = tmp_path / "cache.pt"
    cached.write_bytes(b"weights")
    target, calls, said = tmp_path / "app" / "model.pt", [], []
    ensure_weights(str(target), lambda: calls.append(1) or str(cached), say=said.append)
    assert target.read_bytes() == b"weights" and calls == [1] and said
    assert not (tmp_path / "app" / "model.pt.part").exists()
    ensure_weights(str(target), lambda: calls.append(1) or str(cached), say=said.append)
    assert calls == [1]


def test_selftest_reports_modules_that_fail_to_import():
    from alethic.agent.launcher import check_imports

    got = check_imports(["json", "no_such_module_xyz"])
    assert list(got) == ["no_such_module_xyz"] and got["no_such_module_xyz"]
    assert check_imports(["json", "os"]) == {}


def test_a_read_only_install_folder_moves_the_data_to_the_users_own_folder():
    exe = os.path.join("C:\\", "Program Files", "alethic", "alethic-agent.exe")
    bundle, data = app_dirs(frozen=True, exe=exe, meipass="BUNDLE", source=__file__, writable=lambda d: False,
                            fallback=os.path.join("C:\\", "Users", "x", "alethic-agent"))
    assert bundle == "BUNDLE" and norm(data) == norm(os.path.join("C:\\", "Users", "x", "alethic-agent"))


def test_a_writable_install_folder_keeps_the_data_next_to_the_exe():
    exe = os.path.join("C:\\", "app", "alethic-agent.exe")
    _, data = app_dirs(frozen=True, exe=exe, meipass="BUNDLE", source=__file__, writable=lambda d: True, fallback="elsewhere")
    assert norm(data) == norm(os.path.join("C:\\", "app"))


def test_a_failed_download_gives_a_message_a_person_can_act_on(tmp_path):
    import pytest
    from alethic.agent.launcher import WeightsError

    def offline():
        raise OSError("getaddrinfo failed")
    target = tmp_path / "model.pt"
    with pytest.raises(WeightsError) as e:
        ensure_weights(str(target), offline, say=lambda m: None)
    msg = str(e.value)
    assert "internet" in msg and "model.pt" in msg and str(tmp_path) in msg and not target.exists()


def test_the_harness_uses_the_sft3b_best_checkpoint_and_names_the_local_file_after_it():
    from alethic.agent import launcher
    assert launcher.WEIGHTS_FILE == "runs/sft3b/best/model.pt"
    assert launcher.WEIGHTS_NAME == "model-sft3b.pt"       # an older model.pt next to the exe is never reused by mistake


def test_the_download_message_names_the_model_being_fetched(tmp_path):
    cached = tmp_path / "c.pt"
    cached.write_bytes(b"w")
    said = []
    ensure_weights(str(tmp_path / "app" / "model-sft3b.pt"), lambda: str(cached), say=said.append)
    assert "SFT-3b" in said[0] and "SFT-2" not in said[0]


def test_the_label_in_the_title_defaults_to_sft3b_and_can_be_changed():
    from alethic.agent.cli import agent_title
    assert build_parser().parse_args(build_argv([], "B", "D")).label == "SFT-3b"
    assert build_parser().parse_args(build_argv(["--label", "SFT-2"], "B", "D")).label == "SFT-2"
    assert agent_title("SFT-3b") == "Alethic agent \u00b7 SFT-3b \u00b7 300M"

