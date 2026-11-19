from types import SimpleNamespace

from alethic.agent.commands import run_command
from alethic.agent.loop import Agent
from alethic.agent.menu import COMMANDS, suggest, takes_argument
from alethic.agent.tools import Workspace


def names(buffer):
    return [s.text for s in suggest(buffer)]


def test_only_a_slash_opens_the_menu():
    assert suggest("") == [] and suggest("hello") == [] and suggest("what is /new") == []


def test_a_bare_slash_lists_every_command():
    got = names("/")
    assert "/new" in got and "/effort" in got and "/quit" in got and len(got) == len(COMMANDS)


def test_typing_filters_by_prefix():
    assert set(names("/e")) == {"/effort", "/expand"}
    assert names("/eff") == ["/effort"]
    assert set(names("/c")) == {"/clear", "/compact", "/context", "/cat"}
    assert set(names("/t")) == {"/tools", "/temp", "/tokens"}
    assert names("/status") == ["/status"]


def test_unknown_prefix_gives_no_suggestions():
    assert suggest("/zzz") == []


def test_effort_offers_its_three_levels_after_a_space():
    assert names("/effort ") == ["/effort low", "/effort medium", "/effort high"]
    assert names("/effort h") == ["/effort high"]


def test_other_commands_with_text_after_them_have_no_menu():
    assert suggest("/cat a.py") == [] and suggest("/temp 0.7") == []


def test_every_suggestion_has_a_description():
    assert all(s.description for s in suggest("/"))


def test_takes_argument_knows_which_commands_need_text():
    assert takes_argument("/effort") and takes_argument("/cat") and takes_argument("/temp")
    assert not takes_argument("/new") and not takes_argument("/status")


def test_every_menu_command_is_handled_by_run_command(tmp_path):
    ws = Workspace(str(tmp_path), mode="posix")
    agent = Agent(lambda h, s, m, t: "ok\n</assistant>", ws)
    c = SimpleNamespace(agent=agent, ws=ws, engine=SimpleNamespace(temperature=0.0), count_tokens=len,
                        max_seq_len=1000, title="t", effort="medium")
    for cmd in COMMANDS:
        r = run_command("/" + cmd.name, c)
        assert "unknown command" not in r.message, cmd.name


def test_open_expand_and_folder_are_in_the_menu():
    from alethic.agent.menu import COMMANDS
    names = {c.name for c in COMMANDS}
    assert {"open", "expand", "folder"} <= names
