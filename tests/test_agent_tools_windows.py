from alethic.agent.protocol import Call
from alethic.agent.tools import Workspace


def test_command_output_has_unix_line_endings_like_the_training_data(tmp_path):
    ws = Workspace(str(tmp_path), mode="posix")
    out = ws.execute(Call("shell", "printf 'a\\r\\nb\\r\\n'"))
    assert out.result == "a\nb"


def test_a_windows_work_folder_is_always_shown_as_slash_work(tmp_path):
    ws = Workspace(str(tmp_path), mode="posix")
    ws.root = r"C:\Users\Bekhruz\run\V32"
    cases = {
        r"can't open file 'C:\\Users\\Bekhruz\\run\\V32\\fizzbuzz.py': [Errno 2] No such file": "can't open file '/work/fizzbuzz.py': [Errno 2] No such file",
        r'File "C:\Users\Bekhruz\run\V32\main.py", line 2': 'File "/work/main.py", line 2',
        "C:/Users/Bekhruz/run/V32/sub/x.py": "/work/sub/x.py",
        r"error: [Errno 2] 'C:\\Users\\Bekhruz\\run\\V32\\sub\\x.py'": "error: [Errno 2] '/work/sub/x.py'",
        r'File "/work\main.py", line 2': 'File "/work/main.py", line 2',
    }
    for raw, want in cases.items():
        assert ws._hide_root(raw) == want, raw


def test_text_that_only_looks_like_the_work_folder_is_left_alone(tmp_path):
    ws = Workspace(str(tmp_path), mode="posix")
    assert ws._hide_root("the /workshop is open and 5 \\ 2 stays") == "the /workshop is open and 5 \\ 2 stays"


def test_shell_results_and_errors_both_go_through_the_same_hiding(tmp_path):
    ws = Workspace(str(tmp_path), mode="posix")
    shown = ws.execute(Call("shell", "pwd")).result
    err = ws.execute(Call("read", "nope.py, 1-3")).result
    assert str(tmp_path) not in shown and str(tmp_path) not in err and "/work" in shown + err
