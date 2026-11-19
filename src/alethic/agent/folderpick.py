"""Choosing the folder the model may work in: arrow-key selector, recent folders, Browse dialog, or typed path.

The functions that decide things (options, parse_answer, risk, the recent list) are pure; pick() uses arrow keys
in an interactive terminal and falls back to print/ask lines in plain pipes or tests.
"""
from __future__ import annotations

import json
import os

from alethic.agent import brand

MAX_RECENT = 6
WINDOWS_SYSTEM = ("c:\\windows", "c:\\program files", "c:\\program files (x86)", "c:\\programdata", "c:\\users")
POSIX_SYSTEM = ("/usr", "/etc", "/bin", "/sbin", "/lib", "/var", "/boot", "/sys", "/proc", "/dev", "/opt")


def storage_path(env=None, home=None) -> str:
    env = os.environ if env is None else env
    home = home or os.path.expanduser("~")
    if env.get("APPDATA"):
        return os.path.join(env["APPDATA"], "Alethic", "recent_folders.json")
    return os.path.join(home, ".alethic", "recent_folders.json")


def load_recent(path: str) -> list:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [p for p in data if isinstance(p, str) and os.path.isdir(p)][:MAX_RECENT]
    except (OSError, ValueError, TypeError):
        return []


def remember(path: str, folder: str) -> None:
    try:
        try:
            with open(path, encoding="utf-8") as f:
                old = [p for p in json.load(f) if isinstance(p, str)]
        except (OSError, ValueError, TypeError):
            old = []
        new = [folder] + [p for p in old if p != folder]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(new[:MAX_RECENT], f)
    except OSError:
        pass          # a list of recent folders is a convenience; never stop the program for it


def risk(folder: str, home: str, windows=None):
    """A warning when the folder is too big or too important to hand to a small model, else None."""
    windows = (os.name == "nt") if windows is None else windows
    norm = lambda p: os.path.normcase(os.path.normpath(p)).rstrip("\\/") or os.sep  # noqa: E731
    f, h = norm(folder), norm(home)
    if windows:
        f, h = f.lower(), h.lower()
    if os.path.dirname(f) == f or f.endswith(":") or f in ("", os.sep):
        return "that is a whole drive"
    if f == h:
        return "that is your whole home folder"
    if f == norm(os.path.dirname(h)):
        return "that folder holds every user's files"
    system = WINDOWS_SYSTEM if windows else POSIX_SYSTEM
    if any(f == s or f.startswith(s + ("\\" if windows else "/")) for s in system if s != "c:\\users"):
        return "that is a system folder"
    return None


def options(recent: list, default: str = "") -> list:
    """[(key, label, value)]: numbered recent folders, d = the suggested folder, b = browse, t = type a path."""
    opts = [(str(i + 1), p, p) for i, p in enumerate(recent)]
    if default and default not in recent:
        opts.append(("d", default + "  (new folder here)", default))
    opts.append(("b", "Browse…  (open a folder window)", ""))
    opts.append(("t", "Type a path", ""))
    return opts


def _looks_like_path(text: str) -> bool:
    return any(c in text for c in ("/", "\\", ":")) or text.startswith(("~", "."))


def parse_answer(answer: str, opts: list):
    a = answer.strip().strip("\"'")
    if not a:
        first = next((o for o in opts if o[0] in ("1", "d")), None)
        return ("folder", first[2]) if first else ("bad", "")
    if a.lower() in ("q", "quit", "cancel"):
        return ("cancel", "")
    low = a.lower()
    for key, _, value in opts:
        if low == key:
            return ("browse", "") if key == "b" else ("type", "") if key == "t" else ("folder", value)
    return ("folder", a) if _looks_like_path(a) else ("bad", "")


def _browse_windows(start: str) -> str:
    """Open the native Windows Explorer folder picker without requiring tkinter."""
    import subprocess
    init = start if os.path.isdir(start) else os.path.expanduser("~")
    script = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "[System.Windows.Forms.Application]::EnableVisualStyles();"
        "$o=New-Object System.Windows.Forms.Form -Property @{TopMost=$true;Opacity=0;ShowInTaskbar=$false;FormBorderStyle='None';Width=1;Height=1;StartPosition='CenterScreen'};"
        "$o.Show();$o.Activate();"
        "$d=New-Object System.Windows.Forms.FolderBrowserDialog;$d.Description='Choose the folder Alethic may work in';$d.SelectedPath=$env:ALETHIC_BROWSE_START;"
        "if($d.ShowDialog($o) -eq 'OK'){[Console]::OutputEncoding=[Text.Encoding]::UTF8;[Console]::Write($d.SelectedPath)};"
        "$o.Close();$o.Dispose()"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", script], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=dict(os.environ, ALETHIC_BROWSE_START=init))
    out = (r.stdout or "").strip()
    return os.path.normpath(out) if r.returncode == 0 and out else ""


def browse_dialog(start: str) -> str:
    """The system folder window (native Explorer on Windows, tkinter elsewhere). Empty string if closed."""
    if os.name == "nt":
        try:
            return _browse_windows(start)
        except Exception:
            return ""
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            chosen = filedialog.askdirectory(initialdir=start if os.path.isdir(start) else os.path.expanduser("~"),
                                             title="Choose the folder Alethic may work in")
        finally:
            root.destroy()
        return os.path.normpath(chosen) if chosen else ""
    except Exception:
        return ""


def _render_menu(ui, opts: list, selected: int, note: str = "") -> str:
    cols = max(32, ui._size()[0] - 1)
    w = min(84, cols)
    rule = ui.paint(ui.g["rule"] * w, brand.RULE)
    lines = [
        "",
        "  " + ui.paint(ui.g["star"] + " Alethic", brand.AMBER, "bold"),
        "  " + ui.paint("Where should I work? I can only read and write inside the folder you pick.", brand.MUTED),
        "  " + rule,
    ]
    for i, (_, label, _) in enumerate(opts):
        if i == selected:
            row = "  " + ui.paint(ui.g["arrow"] + " ", brand.AMBER, "bold") + ui.paint(label, brand.CREAM, "bold")
        else:
            row = "    " + ui.paint(label, brand.MUTED)
        lines.append(row)
    lines.append("  " + rule)
    if note:
        lines.append("  " + ui.paint(note, "yellow"))
    lines.append("  " + ui.paint("↑/↓ move · Enter select · q/Esc cancel", brand.MUTED))
    return "\x1b[H\x1b[J" + "\n".join(lines) + "\n"


def _ask_key_yes_no(ui, prompt_text: str) -> bool:
    ui._write("\x1b[2K\r  " + ui.paint(prompt_text + " ", "yellow") + ui.paint("[y] yes · [n/Esc] back", brand.MUTED))
    while True:
        k = ui.keys.read()
        low = k.lower() if len(k) == 1 else k
        if low == "y":
            return True
        if low in ("n", "esc", "ctrl-c", "ctrl-d", "enter"):
            return False


def _validate_arrow(ui, value: str, home: str) -> tuple[str | None, str]:
    folder = os.path.abspath(os.path.expanduser(value))
    if not os.path.isdir(folder):
        if os.path.exists(folder):
            return None, "That is a file, not a folder: %s" % folder
        if not _ask_key_yes_no(ui, "%s does not exist. Create it?" % folder):
            return None, ""
        os.makedirs(folder, exist_ok=True)
    warn = risk(folder, home)
    if warn:
        if not _ask_key_yes_no(ui, "Careful: %s (%s). Use anyway?" % (warn, folder)):
            return None, ""
    return folder, ""


def _read_arrow_choice(ui, opts: list, note: str, selected: int):
    ui._write("\x1b[?25l")
    try:
        with ui.keys.session():
            while True:
                ui._write(_render_menu(ui, opts, selected, note))
                key, note = ui.keys.read(), ""
                if key in ("up", "k"):
                    selected = (selected - 1) % len(opts)
                elif key in ("down", "j"):
                    selected = (selected + 1) % len(opts)
                elif key in ("home", "end"):
                    selected = 0 if key == "home" else len(opts) - 1
                elif key in ("q", "esc", "ctrl-c", "ctrl-d"):
                    return None, selected
                elif len(key) == 1 and key.isdigit() and 1 <= int(key) <= len(opts):
                    return opts[int(key) - 1], int(key) - 1
                elif key == "enter":
                    return opts[selected], selected
    finally:
        ui._write("\x1b[?25h")


def _pick_arrows(ui, browse, recent_file: str, default: str, home: str):
    selected, note = 0, ""
    if hasattr(ui.keys, "mouse"):
        ui.keys.mouse = False
    while True:
        opts = options(load_recent(recent_file), default)
        selected = max(0, min(selected, len(opts) - 1))
        choice, selected = _read_arrow_choice(ui, opts, note, selected)
        if choice is None:
            return None
        opt_key, _, chosen_value = choice
        note = ""
        if opt_key == "b":
            recent = load_recent(recent_file)
            fallback_dir = os.path.dirname(default) if default else home
            chosen_value = browse(recent[0] if recent else (fallback_dir if os.path.isdir(fallback_dir) else home))
            if not chosen_value:
                note = "No folder chosen in dialog."
                continue
        elif opt_key == "t":
            ui._write("\n  " + ui.paint("Path (leave empty to go back): ", brand.CREAM))
            try:
                chosen_value = input().strip().strip("\"'")
            except (EOFError, KeyboardInterrupt):
                chosen_value = ""
            if not chosen_value or chosen_value.lower() in ("q", "quit", "cancel"):
                continue
        with ui.keys.session():
            ui._write("\x1b[?25l")
            try:
                folder, note = _validate_arrow(ui, chosen_value, home)
            finally:
                ui._write("\x1b[?25h")
        if folder:
            remember(recent_file, folder)
            return folder


def pick(say, ask, browse, recent_file: str, default: str = "", home: str = "", first=None, ui=None):
    """Ask where to work. Returns the folder (it exists and is remembered) or None when the user cancels."""
    home = home or os.path.expanduser("~")
    if first is None and ui is not None and ui.isatty and ui.vt and ui.keys and ui.keys.usable():
        return _pick_arrows(ui, browse, recent_file, default, home)
    while True:
        opts = options(load_recent(recent_file), default)
        if first is None:
            say("")
            say("Where should I work? I can only read and write inside the folder you pick.")
            for key, label, _ in opts:
                say("  %s  %s" % (key.ljust(2), label))
        default_key = opts[0][0] if opts and opts[0][0] in ("1", "d") else "b"
        try:
            answer, first = (first, None) if first is not None else (ask("choose (Enter = %s, q = cancel): " % default_key), None)
            kind, value = parse_answer(answer, opts)
            if kind == "browse":
                value = browse(load_recent(recent_file)[0] if load_recent(recent_file) else home)
                if not value:
                    say("no folder chosen. type a path instead.")
                    kind = "type"
                else:
                    kind = "folder"
            if kind == "type":
                kind, value = parse_answer(ask("path: "), [("x", "", "")])
                if kind == "bad":
                    kind = "folder"
                    value = ""
            if kind == "cancel":
                return None
            if kind == "bad" or not value:
                say("I did not understand that. type a number, d, b, t, or a full path.")
                continue
            folder = os.path.abspath(os.path.expanduser(value))
            if not os.path.isdir(folder):
                if os.path.exists(folder):
                    say("that is a file, not a folder.")
                    continue
                if ask("%s does not exist. create it? [y/N]: " % folder).strip().lower() not in ("y", "yes"):
                    continue
                os.makedirs(folder, exist_ok=True)
            warn = risk(folder, home)
            if warn:
                say("careful: %s. the model could change or delete anything inside %s" % (warn, folder))
                if ask("use it anyway? [y/N]: ").strip().lower() not in ("y", "yes"):
                    continue
            remember(recent_file, folder)
            return folder
        except EOFError:
            return None
