"""Read single key presses (no Enter needed) on Windows and on Linux/macOS, and name them: "a", "enter", "up"..."""
from __future__ import annotations

import contextlib
import os
import sys

CONTROL = {"\r": "enter", "\n": "enter", "\x7f": "backspace", "\x08": "backspace", "\t": "tab", "\x03": "ctrl-c",
           "\x04": "ctrl-d", "\x15": "ctrl-u", "\x17": "ctrl-w", "\x01": "home", "\x05": "end"}
LETTER_KEYS = {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end"}
TILDE_KEYS = {"1": "home", "3": "delete", "4": "end", "5": "pgup", "6": "pgdn", "7": "home", "8": "end"}
WINDOWS_KEYS = {"H": "up", "P": "down", "K": "left", "M": "right", "G": "home", "O": "end", "S": "delete", "I": "pgup", "Q": "pgdn"}


def _named(ch: str) -> str:
    if ch in CONTROL:
        return CONTROL[ch]
    return ch if ord(ch) >= 32 else "unknown"


def _sequence(params: str, final: str) -> str:
    """The key for an escape sequence ESC [ params final (arrows, Home/End, PgUp/PgDn, mouse clicks/wheel)."""
    if params.startswith("<"):                      # mouse report: button;column;row, M = press, m = release
        parts = params[1:].split(";")
        if final == "M" and len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            if parts[0] == "0":
                return "click:%s:%s" % (parts[1], parts[2])
            if parts[0] in ("64", "68", "72", "80"):
                return "wheel-up"
            if parts[0] in ("65", "69", "73", "81"):
                return "wheel-down"
        return "unknown"
    if final == "~":
        return TILDE_KEYS.get(params, "unknown")
    return LETTER_KEYS.get(final, "unknown") if not params else "unknown"


def decode_unix(read, ready) -> str:
    """One key from a terminal byte stream. read() gives the next byte, ready(t) says if one is waiting."""
    b = read()
    if b == 0x1b:
        if not ready(0.03):
            return "esc"
        b2 = read()
        if b2 == ord("O"):
            return LETTER_KEYS.get(chr(read()), "unknown")
        if b2 != ord("["):
            return "unknown"
        params = ""
        while True:
            c = read()
            if 0x40 <= c <= 0x7e:
                break
            params += chr(c)
        return _sequence(params, chr(c))
    if b >= 0xC0:
        extra = 1 if b < 0xE0 else 2 if b < 0xF0 else 3
        raw = bytes([b] + [read() for _ in range(extra)])
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return "unknown"
    if b >= 0x80:
        return "unknown"
    return _named(chr(b))


def decode_windows(getwch, ready=None) -> str:
    """One key from msvcrt.getwch(): arrows and Home/End arrive as a prefix character followed by a code.
    In VT input mode (needed for the mouse) they arrive as escape sequences instead; `ready()` says if more is waiting."""
    ch = getwch()
    if ch in ("\x00", "\xe0"):
        return WINDOWS_KEYS.get(getwch(), "unknown")
    if ch == "\x1b":
        if not (ready and ready()):
            return "esc"
        if getwch() != "[":
            return "unknown"
        params = ""
        while True:
            c = getwch()
            if "@" <= c <= "~":
                return _sequence(params, c)
            params += c
    return _named(ch)


@contextlib.contextmanager
def _windows_vt_input(enabled: bool):
    """Windows: ask the console to deliver keys and mouse as VT escape sequences (ENABLE_VIRTUAL_TERMINAL_INPUT)."""
    old = None
    if enabled:
        try:
            import ctypes
            k = ctypes.windll.kernel32
            h = k.GetStdHandle(-10)
            mode = ctypes.c_uint()
            if k.GetConsoleMode(h, ctypes.byref(mode)):
                old = (k, h, mode.value)
                # Enable VT input + mouse input and disable QuickEdit while capturing clicks/wheel.
                k.SetConsoleMode(h, (mode.value & ~0x0040) | 0x0290)
        except Exception:
            old = None
    try:
        yield
    finally:
        if old:
            try:
                old[0].SetConsoleMode(old[1], old[2])
            except Exception:
                pass


class KeyReader:
    """Platform wrapper: `with keys.session(): keys.read()` gives key names until the block ends."""

    def __init__(self, stdin=None):
        self.stdin = stdin or sys.stdin
        self.mouse = False       # set before session(): the console then sends escape sequences, which carry mouse clicks
        try:
            import msvcrt  # noqa: F401
            self.kind = "windows"
        except ImportError:
            try:
                import termios  # noqa: F401
                self.kind = "unix"
            except ImportError:
                self.kind = ""

    def usable(self) -> bool:
        try:
            return bool(self.kind) and self.stdin.isatty()
        except (AttributeError, ValueError):
            return False

    @contextlib.contextmanager
    def session(self):
        if self.kind == "windows":
            with _windows_vt_input(self.mouse):
                yield
            return
        if self.kind != "unix":
            yield
            return
        import termios
        import tty
        fd = self.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            yield
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def read(self) -> str:
        if self.kind == "windows":
            import msvcrt
            return decode_windows(msvcrt.getwch, msvcrt.kbhit)
        import select
        fd = self.stdin.fileno()

        def one() -> int:
            data = os.read(fd, 1)
            if not data:
                raise EOFError
            return data[0]
        try:
            return decode_unix(one, lambda t=0.0: bool(select.select([fd], [], [], t)[0]))
        except EOFError:
            return "ctrl-d"
