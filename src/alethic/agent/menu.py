"""The list of slash commands and the suggestions shown while you type a "/"."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    name: str
    usage: str        # what is shown in the menu, e.g. "/effort [level]"
    description: str


@dataclass(frozen=True)
class Suggestion:
    text: str         # what goes into the input line when it is picked
    usage: str
    description: str


COMMANDS = [
    Command("new", "/new", "start a new conversation (the model forgets everything)"),
    Command("clear", "/clear", "like /new, and clear the screen"),
    Command("undo", "/undo", "remove your last message and the answer to it"),
    Command("retry", "/retry", "run your last message again"),
    Command("compact", "/compact", "forget the oldest turns to free context space"),
    Command("effort", "/effort [level]", "low, medium or high: how many tool calls and how long each step may be"),
    Command("status", "/status", "model, folder, permissions, effort and context use"),
    Command("context", "/context", "a bar of the context window and who uses it"),
    Command("tools", "/tools", "the tools the model can call"),
    Command("auto", "/auto", "turn permission prompts off or on"),
    Command("files", "/files", "list the files in the work folder"),
    Command("cat", "/cat <file>", "show a file from the work folder"),
    Command("open", "/open [file]", "open a whole file in a viewer next to the chat (default: the last file touched)"),
    Command("expand", "/expand [file]", "same as /open"),
    Command("details", "/details [on|off]", "show or hide the code and output blocks of tool calls"),
    Command("mouse", "/mouse [on|off]", "click a tool header to hide or show it (on: select text with Shift+drag)"),
    Command("folder", "/folder", "pick another work folder (starts a new conversation)"),
    Command("save", "/save [file]", "save the raw conversation text into the work folder"),
    Command("temp", "/temp <number>", "sampling temperature, 0 = always the most likely token"),
    Command("tokens", "/tokens", "how many tokens the conversation uses so far"),
    Command("help", "/help", "list every command with a short explanation"),
    Command("quit", "/quit", "leave"),
]

CHOICES = {"effort": [("low", "3 tool calls, short steps"), ("medium", "10 tool calls, normal steps"),
                      ("high", "20 tool calls, long steps")]}


def takes_argument(text: str) -> bool:
    """True for a command that is usually followed by text, like /effort or /cat."""
    name = text.lstrip("/").strip()
    return any(c.name == name and ("[" in c.usage or "<" in c.usage) for c in COMMANDS)


def suggest(buffer: str) -> list:
    """Commands that match what is typed so far. Empty when the line is not a slash command."""
    if not buffer.startswith("/"):
        return []
    head, space, tail = buffer[1:].partition(" ")
    if space:
        # after "/effort " offer its levels; every other command takes free text
        choices = CHOICES.get(head)
        if not choices:
            return []
        return [Suggestion("/%s %s" % (head, value), "/%s %s" % (head, value), note)
                for value, note in choices if value.startswith(tail.strip())]
    return [Suggestion("/" + c.name, c.usage, c.description) for c in COMMANDS if c.name.startswith(head)]
