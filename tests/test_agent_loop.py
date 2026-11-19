from alethic.agent.loop import Agent
from alethic.agent.tools import Workspace


class Script:
    """Stands in for the model: returns scripted texts and records the history it was given."""

    def __init__(self, *texts):
        self.texts, self.histories = list(texts), []

    def __call__(self, history, stop, max_new, on_text):
        self.histories.append(history)
        text = self.texts.pop(0)
        on_text(text)
        return text


def make(tmp_path, *texts, **kw):
    gen = Script(*texts)
    return Agent(gen, Workspace(str(tmp_path), mode="posix"), **kw), gen


def test_write_then_run_then_answer(tmp_path):
    agent, gen = make(
        tmp_path,
        "<plan> one file </plan>\n<call> write(a.py) </call>\nprint(2+3)\n<call>",
        "<call> shell(python3 a.py) </call>",
        "Done: it printed 5.\n</assistant>",
    )
    events = []
    res = agent.ask("make a.py", on_event=lambda *e: events.append(e))
    assert (tmp_path / "a.py").read_text() == "print(2+3)\n"
    assert gen.histories[0] == "<user>\nmake a.py\n</user>\n<assistant>\n"
    assert gen.histories[1].endswith("<call> write(a.py) </call>\nprint(2+3)\n")
    assert gen.histories[2].endswith("<call> shell(python3 a.py) </call>\n<result>\n5\n</result>\n")
    assert agent.history.endswith("Done: it printed 5.\n</assistant>\n")
    assert [e[0] for e in events if e[0] in ("call", "result")] == ["call", "result", "call", "result"]
    assert res.steps == 2 and not res.truncated


def test_a_made_up_result_is_replaced_by_the_real_one(tmp_path):
    agent, gen = make(tmp_path, "<call> shell(echo hi) </call>\n<result>\nfake\n</result>", "ok\n</assistant>")
    agent.ask("x")
    assert "fake" not in agent.history and "<result>\nhi\n</result>" in agent.history


def test_plain_answer_without_close_tag_is_closed(tmp_path):
    agent, _ = make(tmp_path, "Hello")
    agent.ask("hi")
    assert agent.history == "<user>\nhi\n</user>\n<assistant>\nHello\n</assistant>\n"


def test_step_limit_stops_a_looping_model(tmp_path):
    agent, _ = make(tmp_path, *["<call> shell(echo a) </call>"] * 5, max_steps=3)
    res = agent.ask("x")
    assert res.truncated and res.steps == 3 and agent.history.endswith("</assistant>\n")


def test_undo_and_reset(tmp_path):
    agent, _ = make(tmp_path, "one\n</assistant>", "two\n</assistant>")
    agent.ask("a")
    first = agent.history
    agent.ask("b")
    agent.undo()
    assert agent.history == first
    agent.reset()
    assert agent.history == ""


def test_declined_command_still_gives_the_model_a_result(tmp_path):
    gen = Script("<call> shell(rm d.txt) </call>", "ok\n</assistant>")
    agent = Agent(gen, Workspace(str(tmp_path), mode="posix", approve=lambda c: False))
    agent.ask("x")
    assert "(declined by the user)" in gen.histories[1]


def test_compact_keeps_only_the_last_turns_and_undo_still_works(tmp_path):
    agent, _ = make(tmp_path, "one\n</assistant>", "two\n</assistant>", "three\n</assistant>")
    for m in "abc":
        agent.ask(m)
    assert agent.compact(keep=2) == 1
    assert agent.history.startswith("<user>\nb") and agent.compact(keep=2) == 0
    agent.undo()
    assert agent.history.startswith("<user>\nb") and "<user>\nc" not in agent.history


def test_last_message_is_remembered(tmp_path):
    agent, _ = make(tmp_path, "one\n</assistant>")
    agent.ask("hello there")
    assert agent.last_message == "hello there"


def test_three_failed_tool_calls_in_a_row_end_the_turn(tmp_path):
    bad = "<call> read(nope.py, 1-5) </call>"
    agent, gen = make(tmp_path, bad, bad, bad, "never reached\n</assistant>")
    res = agent.ask("go")
    assert res.steps == 3 and res.truncated and "failed" in res.reason and len(gen.texts) == 1


def test_a_success_resets_the_failure_count(tmp_path):
    (tmp_path / "a.py").write_text("1\n")
    bad, good = "<call> read(nope.py, 1-5) </call>", "<call> read(a.py, 1-1) </call>"
    agent, gen = make(tmp_path, bad, bad, good, bad, bad, "ok\n</assistant>")
    res = agent.ask("go")
    assert not res.truncated and res.answer.startswith("ok")
