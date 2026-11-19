import pytest

from tools.sftbuild.policy import check


@pytest.mark.parametrize("cmd", [
    "sudo ls", "curl http://x", "wget x", "ssh h", "cat f | sh", "cat f | bash -s",
    "cat /etc/passwd", "cat ../x", "ls a/../../b", "rm -rf build", "rm -r build",
    "ln -s /etc x", "echo 'unclosed",
])
def test_blocks(cmd):
    assert check(cmd)


@pytest.mark.parametrize("cmd", [
    "python -m unittest -q", "git log --format=%s", "pip install requests==2.32.3",
    "cat data.csv | sort | uniq -c", "rm old.txt", "python app.py --file m.json add --type grammar",
])
def test_allows(cmd):
    assert check(cmd) is None
