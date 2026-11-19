import io

from alethic.agent import preview
from alethic.agent.ui import UI


def make():
    out = io.StringIO()
    return UI(out=out, color=False, unicode=True, isatty=False, fancy=True, animate=False, size=lambda: (100, 30)), out


def test_the_preview_shows_every_part_of_the_new_screen_without_a_model():
    ui, out = make()
    preview.run(ui, sleep=lambda s: None, viewer=False)
    text = out.getvalue()
    for part in ("Alethic agent", "▸ plan", "▸ Running command in terminal", "▸ Writing hello.py", "▸ Reading hello.py",
                 '▸ Searching wikipedia for "capital of Japan"', "$ python3 hello.py", "tool calls"):
        assert part in text, part


def test_the_preview_counts_tool_calls_and_tokens_like_a_real_turn():
    ui, out = make()
    preview.run(ui, sleep=lambda s: None, viewer=False)
    assert ui.last_file == "hello.py" and ui._calls == 4


def test_the_preview_waits_between_pieces_so_you_can_watch_it():
    slept = []
    ui, out = make()
    preview.run(ui, sleep=slept.append, viewer=False)
    assert len(slept) > 20 and sum(slept) < 15


def test_the_textual_preview_runs_a_scripted_conversation_with_real_tool_calls(tmp_path):
    import asyncio

    from alethic.agent.tui import AgentApp
    from alethic.agent.tui_widgets import ToolBlock

    async def go():
        bridge, session = preview.demo_session(str(tmp_path), pause=0)
        app = AgentApp(session, bridge)
        async with app.run_test(size=(110, 32)) as pilot:
            app.query_one("#prompt").value = "write hello.py and run it"
            await pilot.press("enter")
            for _ in range(100):
                await pilot.pause(0.05)
                if not app.busy:
                    break
            assert (tmp_path / "hello.py").exists()
            assert len(app.query(ToolBlock)) == 3
    asyncio.run(go())


def test_the_json_preview_answers_a_message_with_tool_events(tmp_path):
    import io
    import json

    from alethic.agent.jsonio import JsonBridge, serve
    out = io.StringIO()
    bridge, session = preview.demo_session(str(tmp_path), pause=0, bridge=JsonBridge(out))
    serve(session, io.StringIO(json.dumps({"t": "submit", "text": "go"}) + "\n"), out)
    kinds = [json.loads(x)["t"] for x in out.getvalue().splitlines()]
    assert kinds.count("tool") == 3 and kinds[-1] == "done" and (tmp_path / "hello.py").exists()
