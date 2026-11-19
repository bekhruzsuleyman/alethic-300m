import asyncio
import threading

from alethic.agent.cli import Session
from alethic.agent.tui import AgentApp, TuiBridge
from alethic.agent.tui_widgets import AnswerText, ConfirmScreen, ToolBlock


class Script:
    def __init__(self, *texts, gate=None):
        self.texts, self.gate = list(texts), gate
        self.temperature, self.last_tokens = 0.0, 3

    def generate(self, history, stop, max_new, on_text):
        if self.gate:
            self.gate.wait(5)
        t = self.texts.pop(0)
        on_text(t)
        return t


def make(tmp_path, *texts, auto=True, gate=None):
    eng = Script(*texts, gate=gate)
    bridge = TuiBridge()
    s = Session(eng.generate, eng, str(tmp_path), bridge, count_tokens=len, auto=auto, mode="posix", title="Alethic test")
    return AgentApp(s, bridge), s


def run(coro):
    return asyncio.run(coro)


async def say(pilot, app, text):
    box = app.query_one("#prompt")
    box.value = text
    await pilot.press("enter")
    await settle(pilot, app)


async def settle(pilot, app, limit=60):
    for _ in range(limit):
        await pilot.pause(0.05)
        if not app.busy:
            break
    await pilot.pause(0.05)


def chat_text(app):
    return "\n".join(str(w.render()) if hasattr(w, "render") else "" for w in app.query("#chat *"))


def test_the_banner_names_the_folder_and_the_prompt_is_ready(tmp_path):
    async def go():
        app, s = make(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.1)
            assert "Alethic test" in chat_text(app) and str(tmp_path) in chat_text(app)
            assert app.query_one("#prompt").disabled is False
    run(go())


def test_an_answer_appears_in_the_chat(tmp_path):
    async def go():
        app, s = make(tmp_path, "Hello there.\n</assistant>")
        async with app.run_test(size=(100, 30)) as pilot:
            await say(pilot, app, "hi")
            assert "Hello there." in chat_text(app) and "hi" in chat_text(app)
    run(go())


def test_the_input_stays_visible_but_locked_while_the_model_answers(tmp_path):
    async def go():
        gate = threading.Event()
        app, s = make(tmp_path, "Done.\n</assistant>", gate=gate)
        async with app.run_test(size=(100, 30)) as pilot:
            box = app.query_one("#prompt")
            box.value = "hi"
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert app.busy and box.disabled is True and box.display is True
            assert app.query_one("#status").display is True
            gate.set()
            await settle(pilot, app)
            assert box.disabled is False
    run(go())


def test_a_tool_call_makes_a_block_and_a_click_on_its_header_hides_and_shows_it(tmp_path):
    async def go():
        app, s = make(tmp_path, "<call> shell(echo hello) </call>", "ok\n</assistant>")
        async with app.run_test(size=(100, 30)) as pilot:
            await say(pilot, app, "run it")
            block = app.query_one(ToolBlock)
            assert block.body.display is True and "Running command in terminal" in str(block.header.render())
            await pilot.click(block.header)
            await pilot.pause(0.1)
            assert block.body.display is False and block.open_ is False
            await pilot.click(block.header)
            await pilot.pause(0.1)
            assert block.body.display is True
    run(go())


def test_details_off_collapses_every_block_and_details_on_opens_them(tmp_path):
    async def go():
        app, s = make(tmp_path, "<call> shell(echo hello) </call>", "ok\n</assistant>")
        async with app.run_test(size=(100, 30)) as pilot:
            await say(pilot, app, "run it")
            await say(pilot, app, "/details off")
            assert app.query_one(ToolBlock).open_ is False
            await say(pilot, app, "/details on")
            assert app.query_one(ToolBlock).open_ is True
    run(go())


def test_clear_really_empties_the_chat_and_forgets_the_conversation(tmp_path):
    async def go():
        app, s = make(tmp_path, "Hello there.\n</assistant>")
        async with app.run_test(size=(100, 30)) as pilot:
            await say(pilot, app, "hi")
            assert "Hello there." in chat_text(app)
            await say(pilot, app, "/clear")
            assert "Hello there." not in chat_text(app) and len(app.query(".user")) == 0
            assert s.agent.turns == 0
            assert len(app.query(AnswerText)) == 0
    run(go())


def test_the_chat_scrolls_with_page_keys_and_stays_at_the_end_for_new_text(tmp_path):
    async def go():
        app, s = make(tmp_path, *["line %d\n</assistant>" % i for i in range(30)])
        async with app.run_test(size=(100, 20)) as pilot:
            for i in range(30):
                await say(pilot, app, "q%d" % i)
            chat = app.query_one("#chat")
            assert chat.max_scroll_y > 0 and chat.scroll_y >= chat.max_scroll_y - 1
            end = chat.scroll_y
            await pilot.press("pageup")
            await pilot.pause(0.2)
            assert chat.scroll_y < end
            await pilot.press("pagedown")
            await pilot.pause(0.2)
            assert chat.scroll_y >= end - 1
    run(go())


def test_open_shows_the_whole_file_on_the_right_and_the_edge_can_be_dragged(tmp_path):
    (tmp_path / "a.py").write_text("print(1)\nprint(2)\n")

    async def go():
        app, s = make(tmp_path)
        async with app.run_test(size=(120, 30)) as pilot:
            await say(pilot, app, "/open a.py")
            doc = app.query_one("#doc")
            assert doc.display is True and doc.region.x > app.query_one("#chat").region.x
            assert "print(2)" in str(app.query_one("#doc-text").render())
            before = doc.region.width
            await pilot.pause(0.1)
            await app.set_doc_width(before - 20)
            await pilot.pause(0.1)
            assert doc.region.width == before - 20
            await pilot.click("#doc-close")
            await pilot.pause(0.1)
            assert doc.display is False
    run(go())


def test_a_command_that_is_not_on_the_safe_list_asks_first(tmp_path):
    (tmp_path / "a.txt").write_text("x\n")

    async def go():
        app, s = make(tmp_path, "<call> shell(cp a.txt new.txt) </call>", "done\n</assistant>", auto=False)
        async with app.run_test(size=(100, 30)) as pilot:
            box = app.query_one("#prompt")
            box.value = "make it"
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause(0.05)
                if isinstance(app.screen, ConfirmScreen):
                    break
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await settle(pilot, app)
            assert (tmp_path / "new.txt").exists()
    run(go())


def test_declining_the_prompt_runs_nothing(tmp_path):
    (tmp_path / "a.txt").write_text("x\n")

    async def go():
        app, s = make(tmp_path, "<call> shell(cp a.txt no.txt) </call>", "ok\n</assistant>", auto=False)
        async with app.run_test(size=(100, 30)) as pilot:
            app.query_one("#prompt").value = "make it"
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause(0.05)
                if isinstance(app.screen, ConfirmScreen):
                    break
            await pilot.press("n")
            await settle(pilot, app)
            assert not (tmp_path / "no.txt").exists()
    run(go())


def test_typing_a_slash_shows_the_commands_and_tab_completes(tmp_path):
    async def go():
        app, s = make(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            box = app.query_one("#prompt")
            box.focus()
            await pilot.press("/", "c", "l")
            await pilot.pause(0.1)
            menu = app.query_one("#menu")
            assert menu.display is True and menu.option_count >= 1
            await pilot.press("tab")
            await pilot.pause(0.1)
            assert box.value.startswith("/clear")
    run(go())


def test_a_tip_and_the_status_row_show_while_working(tmp_path):
    async def go():
        gate = threading.Event()
        app, s = make(tmp_path, "Done.\n</assistant>", gate=gate)
        async with app.run_test(size=(100, 30)) as pilot:
            app.query_one("#prompt").value = "hi"
            await pilot.press("enter")
            await pilot.pause(0.3)
            assert str(app.query_one("#tip").render()).strip() != ""
            assert "tok/s" in str(app.query_one("#status").render()) or "token" in str(app.query_one("#status").render())
            gate.set()
            await settle(pilot, app)
    run(go())


def test_folder_command_with_a_path_switches_the_folder_and_starts_a_new_conversation(tmp_path):
    other = tmp_path / "other"
    other.mkdir()

    async def go():
        app, s = make(tmp_path, "Hi.\n</assistant>")
        async with app.run_test(size=(100, 30)) as pilot:
            await say(pilot, app, "hello")
            await say(pilot, app, "/folder %s" % other)
            assert s.ws.root == str(other) and s.agent.turns == 0
    run(go())


def test_the_folder_dialog_lists_recent_folders_and_takes_a_typed_path(tmp_path, monkeypatch):
    target = tmp_path / "typed"
    target.mkdir()
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))

    async def go():
        app, s = make(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            app.query_one("#prompt").value = "/folder"
            await pilot.press("enter")
            from alethic.agent.tui_widgets import FolderScreen
            for _ in range(40):
                await pilot.pause(0.05)
                if isinstance(app.screen, FolderScreen):
                    break
            assert isinstance(app.screen, FolderScreen)
            app.screen.query_one("#folder-input").value = str(target)
            await pilot.press("enter")
            await settle(pilot, app)
            assert s.ws.root == str(target)
    run(go())
