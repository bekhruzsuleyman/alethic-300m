"""The publish launchers must build and publish the SAME folder, and never delete a .git they did not just verify."""
import os
import re

DEPLOY = os.path.join(os.path.dirname(__file__), "..", "deploy")


def _read(name):
    return open(os.path.join(DEPLOY, name), encoding="utf-8").read().replace("\r\n", "\n")


def _folder_of(text, pattern):
    return re.search(pattern, text).group(1)


def test_full_repo_copy_is_built_where_it_is_published():
    built = _folder_of(_read("32_make_public_copy.bat"), r"export_public \. \.\.\\(alethic-public)")
    published = _folder_of(_read("33_publish_public_repo.bat"), r'cd /d "%~dp0\.\.\\\.\.\\(alethic-public)"')
    assert built == published == "alethic-public"


def test_agent_copy_is_built_where_it_is_published():
    text = _read("38_publish_agent_repo.bat")
    built = _folder_of(text, r"export_agent \. \.\.\\(alethic-agent-public)")
    published = _folder_of(text, r'cd /d "%~dp0\.\.\\\.\.\\(alethic-agent-public)"')
    assert built == published == "alethic-agent-public"


def test_agent_publish_checks_the_folder_before_touching_git():
    text = _read("38_publish_agent_repo.bat")
    after_cd = text[text.index('cd /d "%~dp0..\\..\\alethic-agent-public"'):]
    assert after_cd.index("if errorlevel 1 goto nofolder") < after_cd.index("rmdir")
    assert after_cd.index("if not exist .alethic_export_ok goto nofolder") < after_cd.index("rmdir")


def test_full_publish_checks_the_folder_marker_too():
    text = _read("33_publish_public_repo.bat")
    assert text.index("if not exist .alethic_export_ok") < text.index("rmdir")
