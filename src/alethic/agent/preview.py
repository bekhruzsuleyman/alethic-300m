"""A fake session with no model, to look at the screen design: the logo animation, the spinner and vibe words,
tool blocks, code in answers and the file viewer. Run with deploy\\34_preview_ui.bat."""
from __future__ import annotations

import time

from alethic.agent.protocol import Call
from alethic.agent.render import Renderer

CODE = '''def greet(name):
    """Say hello."""
    return "hello, " + name


if __name__ == "__main__":
    print(greet("world"))
'''
SEARCH = "1. Tokyo (en.wikipedia.org/wiki/Tokyo)\n   Tokyo is the capital and most populous city of Japan.\n2. Japan (en.wikipedia.org/wiki/Japan)\n   Japan is an island country in East Asia."
ANSWER = ("I wrote the file and ran it. The command printed the greeting. Here is the core of it:\n"
          "```python\ndef greet(name):\n    return \"hello, \" + name\n```\n"
          "Tokyo is the capital of Japan.\n")


def _say(ui, r, text, sleep, per=4, tokens=None):
    """Stream `text` a few characters at a time, counting each piece as a token."""
    for i in range(0, len(text), per):
        tokens["n"] += 1
        ui.tick(tokens["n"])
        ui.stream(r.feed(text[i:i + per]))
        sleep(0.03)


def run(ui, sleep=time.sleep, viewer=True) -> None:
    ui.banner("Alethic agent · preview · no model", "preview-folder", auto=False, effort="medium")
    t = {"n": 0}
    ui.line(ui.paint(ui.arrow + " ", "brand", "bold") + ui.paint("write hello.py, run it, and tell me the capital of Japan", "bold"))
    ui.begin_turn()
    try:
        r = Renderer()
        _say(ui, r, "<plan> write the file, run it, look the capital up </plan>\n", sleep, tokens=t)
        ui.stream(r.finish())
        sleep(0.6)
        steps = [(Call("write", "hello.py", CODE), "wrote hello.py (7 lines)", True),
                 (Call("shell", "python3 hello.py", None), "hello, world", True),
                 (Call("read", "hello.py, 1-100", None), "\n".join("%d   %s" % (i + 1, ln) for i, ln in enumerate(CODE.rstrip("\n").split("\n"))), True),
                 (Call("search", "capital of Japan, wikipedia"), SEARCH, True)]
        for call, result, ok in steps:
            ui.call(call)
            sleep(0.8)
            ui.result(result, ok=ok, call=call)
            t["n"] += 40
            ui.tick(t["n"])
            sleep(0.4)
        r = Renderer()
        _say(ui, r, ANSWER, sleep, tokens=t)
        ui.stream(r.finish())
    finally:
        ui.end_turn()
    if viewer:
        ui.line()
        ui.info("now try it: click a ▸ header to hide or show its block. type /open to see the viewer, q to quit.")
        while True:
            try:
                line = ui.prompt()
            except KeyboardInterrupt:
                return
            if line is None or line == "q":
                return
            if line in ("/open", "/expand"):
                ui.view_file("hello.py", CODE * 8)
            elif line.startswith("/details"):
                ui.set_details("off" not in line)


class _Demo:
    """A model that is not a model: it answers every message with the same little script (write, run, read, explain)."""

    def __init__(self, pause: float):
        self.pause, self.n = pause, 0
        self.temperature, self.last_tokens = 0.0, 0

    def generate(self, history, stop, max_new, on_text):
        i = self.n % 4
        self.n += 1
        text = ["<plan> write the file, run it, read it back </plan>\n<call> write(hello.py) </call>\n" + CODE + "<call>",
                "<call> shell(python3 hello.py) </call>",
                "<call> read(hello.py, 1-100) </call>",
                ANSWER + "</assistant>"][i]
        for k in range(0, len(text), 5):
            on_text(text[k:k + 5])
            time.sleep(self.pause)
        return text


def demo_session(folder: str, pause: float = 0.02, bridge=None):
    from alethic.agent.cli import Session, agent_title
    if bridge is None:
        from alethic.agent.tui import TuiBridge
        bridge = TuiBridge()
    demo = _Demo(pause)
    session = Session(demo.generate, demo, folder, bridge, count_tokens=len, auto=True, title=agent_title("preview"))
    return bridge, session


def main_textual() -> None:
    import tempfile

    from alethic.agent.tui import AgentApp
    with tempfile.TemporaryDirectory() as folder:
        bridge, session = demo_session(folder)
        AgentApp(session, bridge).run()


def main_json() -> None:
    import sys
    import tempfile

    from alethic.agent.jsonio import JsonBridge, serve
    with tempfile.TemporaryDirectory() as folder:
        bridge, session = demo_session(folder, bridge=JsonBridge(sys.stdout))
        serve(session, sys.stdin, sys.stdout)


def main() -> None:
    import sys
    if "--json" in sys.argv:
        return main_json()
    if "--textual" in sys.argv:
        try:
            import textual  # noqa: F401
            return main_textual()
        except ImportError:
            pass
    from alethic.agent.ui import UI, enable_ansi, use_utf8_output
    use_utf8_output()
    ui = UI(vt=enable_ansi())
    run(ui)


if __name__ == "__main__":
    main()
