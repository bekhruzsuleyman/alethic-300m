import io
import json
import threading

from alethic.agent.cli import Session
from alethic.agent.jsonio import JsonBridge, serve


class Script:
    def __init__(self, *texts):
        self.texts = list(texts)
        self.temperature, self.last_tokens = 0.0, 3

    def generate(self, history, stop, max_new, on_text):
        t = self.texts.pop(0)
        on_text(t)
        return t


def run(tmp_path, lines, *texts, auto=True, delay=None):
    out = io.StringIO()
    bridge = JsonBridge(out)
    eng = Script(*texts)
    s = Session(eng.generate, eng, str(tmp_path), bridge, count_tokens=len, auto=auto, mode="posix", title="T")
    serve(s, io.StringIO("".join(json.dumps(l) + "\n" for l in lines)), out)
    return [json.loads(x) for x in out.getvalue().splitlines()], s


def kinds(ev):
    return [e["t"] for e in ev]


def test_hello_lists_the_commands_the_folder_and_the_title(tmp_path):
    ev, _ = run(tmp_path, [])
    h = ev[0]
    assert h["t"] == "hello" and h["title"] == "T" and h["folder"] == str(tmp_path)
    assert any(c["name"] == "clear" for c in h["commands"])


def test_an_answer_is_text_events_then_done_with_stats(tmp_path):
    ev, _ = run(tmp_path, [{"t": "submit", "text": "hi"}], "Hello there.\n</assistant>")
    assert "".join(e["text"] for e in ev if e["t"] == "text") == "Hello there.\n"
    assert ev[-1]["t"] == "done" and ev[-1]["tokens"] >= 1


def test_code_fences_become_colored_code_events(tmp_path):
    ev, _ = run(tmp_path, [{"t": "submit", "text": "x"}], "```python\ndef f():\n    pass\n```\n</assistant>")
    k = kinds(ev)
    assert "code_open" in k and "code" in k and "code_close" in k
    code = [e for e in ev if e["t"] == "code"][0]
    assert any(style for _, style in code["runs"]) and code["runs"][0][0] == "def"


def test_a_tool_call_sends_call_then_tool_with_rows(tmp_path):
    ev, _ = run(tmp_path, [{"t": "submit", "text": "x"}], "<call> shell(echo hello) </call>", "ok\n</assistant>")
    c = next(e for e in ev if e["t"] == "call")
    t = next(e for e in ev if e["t"] == "tool")
    assert c["head"] == "Running command in terminal" and t["ok"] is True
    assert "hello" in json.dumps(t["rows"]) and t["summary"].startswith("hello")


def test_clear_sends_clear_and_forgets_the_conversation(tmp_path):
    ev, s = run(tmp_path, [{"t": "submit", "text": "hi"}, {"t": "submit", "text": "/clear"}], "Hi.\n</assistant>")
    assert "clear" in kinds(ev) and s.agent.turns == 0


def test_open_sends_the_whole_file_as_colored_rows(tmp_path):
    (tmp_path / "a.py").write_text("print(1)\nprint(2)\n")
    ev, _ = run(tmp_path, [{"t": "submit", "text": "/open a.py"}])
    d = next(e for e in ev if e["t"] == "doc")
    assert d["name"] == "a.py" and len(d["rows"]) == 2


def test_a_command_that_is_not_safe_waits_for_the_answer(tmp_path):
    (tmp_path / "a.txt").write_text("x\n")
    out = io.StringIO()
    bridge = JsonBridge(out)
    eng = Script("<call> shell(cp a.txt b.txt) </call>", "ok\n</assistant>")
    s = Session(eng.generate, eng, str(tmp_path), bridge, count_tokens=len, auto=False, mode="posix", title="T")

    class Pipe:
        def __init__(self):
            self.q = __import__("queue").Queue()

        def __iter__(self):
            return self

        def __next__(self):
            v = self.q.get()
            if v is None:
                raise StopIteration
            return v

    pipe = Pipe()
    th = threading.Thread(target=serve, args=(s, pipe, out))
    th.start()
    pipe.q.put(json.dumps({"t": "submit", "text": "go"}) + "\n")
    for _ in range(100):
        if '"t": "confirm"' in out.getvalue():
            break
        threading.Event().wait(0.05)
    assert '"t": "confirm"' in out.getvalue() and not (tmp_path / "b.txt").exists()
    pipe.q.put(json.dumps({"t": "confirm", "answer": "y"}) + "\n")
    for _ in range(100):
        if '"t": "done"' in out.getvalue():
            break
        threading.Event().wait(0.05)
    pipe.q.put(None)
    th.join(5)
    assert (tmp_path / "b.txt").exists()


def test_status_events_carry_a_word_and_a_tip(tmp_path):
    ev, _ = run(tmp_path, [{"t": "submit", "text": "hi"}], "Hello.\n</assistant>")
    s = [e for e in ev if e["t"] == "status"][-1]
    assert s["word"] and s["tip"]


def test_quit_ends_the_program(tmp_path):
    ev, _ = run(tmp_path, [{"t": "submit", "text": "/quit"}])
    assert ev[-1]["t"] == "quit"


def test_the_ink_tips_only_mention_things_that_exist_on_that_screen():
    from alethic.agent.jsonio import INK_TIPS
    assert INK_TIPS and all(len(t) <= 78 for t in INK_TIPS)
    assert not any(w in t for t in INK_TIPS for w in ("Click", "beside", "/mouse", "resize"))
