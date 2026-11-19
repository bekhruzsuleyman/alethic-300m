"""Slash commands typed at the prompt."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from alethic.agent import effort as effort_levels

HELP = """commands
  /new            start a new conversation (the model forgets everything)
  /clear          like /new, and clear the screen
  /undo           remove your last message and the answer to it
  /retry          run your last message again
  /compact        forget the oldest turns to free context space
  /effort [level] low, medium or high: how many tool calls and how long each step may be
  /status         model, folder, permissions, effort and context use
  /context        a bar of the context window and who uses it
  /tools          the tools the model can call
  /auto           turn permission prompts off or on (off = it runs commands without asking)
  /files          list the files in the work folder
  /cat <file>     show a file from the work folder
  /open [file]    open a whole file in a viewer next to the chat (default: the last file touched); also /expand
  /details [on|off]  show or hide the code and output blocks of tool calls (hidden: one line per call)
  /mouse [on|off]  clicking a ▸ header toggles its block (while on, select text with Shift+drag)
  /folder         pick another work folder (starts a new conversation)
  /save <file>    save the raw conversation text into the work folder
  /temp <number>  sampling temperature, 0 = always the most likely token
  /tokens         how many tokens the conversation uses so far
  /quit           leave"""


@dataclass
class CommandResult:
    message: str = ""
    quit: bool = False
    clear: bool = False      # the screen should be cleared
    rerun: str | None = None  # a message the program should run again


def context_breakdown(history: str, count) -> dict:
    """Tokens used by what you wrote, by tool results, and by the model's own text (answers, calls, files)."""
    you = sum(count(m) for m in re.findall(r"<user>\n.*?\n</user>", history, re.S))
    results = sum(count(m) for m in re.findall(r"<result>\n.*?\n</result>", history, re.S))
    total = count(history)
    return {"you": you, "tool results": results, "model": max(0, total - you - results), "total": total}


TOOLS = """tools the model can call
  shell(cmd)       run a command in the work folder (asks first unless it is on the safe list)
  read(f, a-b)     show lines a to b of a file, with line numbers
  write(f)         create or replace a file with the text that follows the call
  edit(f, a-b)     replace lines a to b of a file with the text that follows the call
  search(words, source)  look up words on a source (wikipedia); "no results" when nothing is found
blocked: sudo, curl, wget, ssh, recursive rm, absolute paths, .. paths"""


def run_command(line: str, ctx):
    """Run a /command and return its result, or None if the line is a normal message."""
    if not line.startswith("/"):
        return None
    name, _, arg = line[1:].partition(" ")
    arg = arg.strip()
    if name in ("quit", "exit", "q"):
        return CommandResult("", quit=True)
    if name == "help":
        return CommandResult(HELP)
    if name == "new":
        ctx.agent.reset()
        return CommandResult("new conversation")
    if name == "clear":
        ctx.agent.reset()
        return CommandResult("new conversation", clear=True)
    if name == "retry":
        last = ctx.agent.last_message
        if not last:
            return CommandResult("nothing to retry yet")
        ctx.agent.undo()
        return CommandResult("retrying: " + last, rerun=last)
    if name == "compact":
        before = ctx.count_tokens(ctx.agent.history)
        dropped = ctx.agent.compact(keep=2)
        return CommandResult("dropped %d old turn%s, %d -> %d tokens" % (dropped, "" if dropped == 1 else "s", before, ctx.count_tokens(ctx.agent.history)))
    if name == "effort":
        if not arg:
            steps, new = effort_levels.LEVELS[ctx.effort]
            return CommandResult("effort: %s (up to %d tool calls, %d tokens per step). levels: low, medium, high" % (ctx.effort, steps, new))
        if arg not in effort_levels.LEVELS:
            return CommandResult("usage: /effort low|medium|high")
        ctx.effort = arg
        effort_levels.apply(ctx.agent, arg)
        return CommandResult("effort: %s (up to %d tool calls, %d tokens per step)" % ((arg,) + effort_levels.LEVELS[arg]))
    if name == "tools":
        return CommandResult(TOOLS)
    if name == "context":
        b = context_breakdown(ctx.agent.history, ctx.count_tokens)
        used, cap = b["total"], ctx.max_seq_len
        filled = min(20, round(20 * used / cap)) if cap else 0
        rows = ["[%s%s] %d / %d tokens (%d%%)" % ("#" * filled, "-" * (20 - filled), used, cap, round(100 * used / cap) if cap else 0)]
        rows += ["  %-13s %d" % (k, b[k]) for k in ("you", "tool results", "model")]
        return CommandResult("\n".join(rows))
    if name == "status":
        used = ctx.count_tokens(ctx.agent.history)
        steps, new = ctx.agent.max_steps, ctx.agent.max_new
        n = ctx.agent.turns
        files = sum(len(f) for _, _, f in os.walk(ctx.ws.root))
        return CommandResult("\n".join([
            ctx.title,
            "  folder       %s (%d files)" % (ctx.ws.root, files),
            "  permissions  %s" % ("auto: runs commands without asking" if ctx.ws.auto else "ask before commands"),
            "  effort       %s (up to %d tool calls, %d tokens per step)" % (ctx.effort, steps, new),
            "  temperature  %s" % ctx.engine.temperature,
            "  context      %d / %d tokens, %d turn%s" % (used, ctx.max_seq_len, n, "" if n == 1 else "s"),
        ]))
    if name == "undo":
        ctx.agent.undo()
        return CommandResult("removed the last exchange")
    if name == "auto":
        ctx.ws.auto = not ctx.ws.auto
        return CommandResult("auto mode %s" % ("on: commands run without asking" if ctx.ws.auto else "off: it asks before commands"))
    if name == "files":
        found = []
        for base, dirs, files in os.walk(ctx.ws.root):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            found += [os.path.relpath(os.path.join(base, f), ctx.ws.root).replace("\\", "/") for f in sorted(files)]
        return CommandResult("\n".join(found[:60]) or "(the work folder is empty)")
    if name in ("open", "expand"):
        rel = arg or getattr(getattr(ctx, "ui", None), "last_file", "")
        if not rel:
            return CommandResult("no file yet. use /open <file>")
        try:
            with open(ctx.ws.resolve(rel), encoding="utf-8", errors="replace") as f:
                text = f.read()
        except (ValueError, OSError) as exc:
            return CommandResult("error: %s" % exc)
        ctx.ui.view_file(rel, text)
        return CommandResult("")
    if name == "details":
        ui = getattr(ctx, "ui", None)
        if ui is None:
            return CommandResult("not available here")
        value = {"on": True, "off": False}.get(arg.lower(), not ui.details)
        (getattr(ui, "set_details", None) or (lambda v: setattr(ui, "details", v)))(value)
        return CommandResult("tool blocks %s. click a ▸ header to change one" % ("shown" if value else "hidden"))
    if name == "mouse":
        ui = getattr(ctx, "ui", None)
        if ui is None:
            return CommandResult("not available here")
        ui.mouse = {"on": True, "off": False}.get(arg.lower(), not ui.mouse)
        return CommandResult("mouse clicks %s%s" % ("on: click a ▸ header to hide or show its block" if ui.mouse else "off",
                                                    "" if ui.mouse else ". you can select text with the mouse again"))
    if name == "folder":
        choose = getattr(ctx, "choose_folder", None)
        return CommandResult(choose(arg) if choose else "changing the folder is not available here")
    if name == "cat":
        try:
            with open(ctx.ws.resolve(arg), encoding="utf-8", errors="replace") as f:
                return CommandResult("".join(f.readlines()[:200]).rstrip("\n"))
        except (ValueError, OSError) as exc:
            return CommandResult("error: %s" % exc)
    if name == "save":
        try:
            with open(ctx.ws.resolve(arg or "conversation.txt"), "w", encoding="utf-8") as f:
                f.write(ctx.agent.history)
            return CommandResult("saved %s" % (arg or "conversation.txt"))
        except (ValueError, OSError) as exc:
            return CommandResult("error: %s" % exc)
    if name == "temp":
        try:
            ctx.engine.temperature = max(0.0, float(arg))
            return CommandResult("temperature %s" % ctx.engine.temperature)
        except ValueError:
            return CommandResult("usage: /temp 0.7")
    if name == "tokens":
        return CommandResult("%d tokens" % ctx.count_tokens(ctx.agent.history))
    return CommandResult("unknown command /%s, try /help" % name)
