"""Full-screen terminal transitions and centered Alethic model loading screen."""
from __future__ import annotations

import os
import sys
import threading
import time

from alethic.agent import brand

LOAD_TITLE = "Loading Alethic Model"


def clear_console(ui) -> None:
    """Clear the visible screen and scrollback buffer when running in a terminal."""
    if not getattr(ui, "isatty", False):
        return
    if getattr(ui, "vt", False):
        ui._write("\x1b[2J\x1b[3J\x1b[H")
        return
    if os.name == "nt":
        os.system("cls")


def maximize_console() -> None:
    """Maximize the host console window on Windows so the chat opens full-screen."""
    if os.name != "nt":
        return
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return
        import ctypes
        k = ctypes.windll.kernel32
        u = ctypes.windll.user32
        hwnd = k.GetConsoleWindow()
        # Windows Terminal hosts cmd inside a CASCADIA window while GetConsoleWindow() is hidden.
        if os.environ.get("WT_SESSION") or not hwnd or not u.IsWindowVisible(hwnd):
            fg = u.GetForegroundWindow()
            if fg:
                buf = ctypes.create_unicode_buffer(128)
                if u.GetClassNameW(fg, buf, 128) and buf.value in ("CASCADIA_HOSTING_WINDOW_CLASS", "ConsoleWindowClass"):
                    hwnd = fg
        if hwnd:
            u.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
    except Exception:
        pass


def _pad_left(width: int, visible_len: int) -> str:
    return " " * max(0, (width - visible_len) // 2)


def _sweep_bar(width: int, step: int, bar_ch: str, ui) -> str:
    """Centered brand rule with an animated blue->lens->amber highlight gliding across it."""
    bar_w = max(18, min(36, width - 8))
    span = bar_w + 6
    pos = (step % span) - 3
    runs = []
    for i in range(bar_w):
        dist = abs(i - pos)
        if dist == 0:
            style = brand.LENS
        elif dist == 1:
            style = brand.AMBER if i > pos else brand.DISC_BLUE
        elif dist == 2:
            style = brand.DISC_AMBER if i > pos else brand.DISC_BLUE
        else:
            style = brand.RULE
        if runs and runs[-1][1] == style:
            runs[-1] = (runs[-1][0] + bar_ch, style)
        else:
            runs.append((bar_ch, style))
    return _pad_left(width, bar_w) + ui.paint_row(runs)


def render_loading_frame(cols: int, rows: int, elapsed: float, step: int, label: str, ui,
                         title: str = LOAD_TITLE) -> str:
    """Build one centered ANSI frame matching the Alethic brand palette."""
    w = max(24, cols - 1)
    h = max(12, rows)
    fancy = bool(ui.unicode and ui.fancy)
    gaps = brand.intro_gaps()
    gap = 0 if step >= len(gaps) else gaps[step]
    mark = brand.mark_rows(fancy, gap)

    spin = brand.spinner(elapsed, fancy)
    dots = "." * (step % 4)
    heading_plain = "%s  %s%s" % (spin, title, dots.ljust(3))
    heading = (ui.paint(spin, brand.AMBER, "bold")
               + "  "
               + ui.paint(title, brand.CREAM, "bold")
               + ui.paint(dots.ljust(3), brand.AMBER))
    sub_plain = "%s · 300M · %.1fs" % (label or "SFT-3c", elapsed)
    sub = ui.paint(sub_plain, brand.MUTED)

    block = []
    if w >= 30 and h >= 12:
        for row in mark:
            vis = sum(len(t) for t, _ in row)
            block.append(_pad_left(w, vis) + ui.paint_row(row))
        block.append("")
    block.append(_pad_left(w, len(heading_plain)) + heading)
    block.append(_sweep_bar(w, step, ui.g["rule"], ui))
    block.append(_pad_left(w, len(sub_plain)) + sub)

    top = max(1, (h - len(block)) // 2 + 1)
    out = ["\x1b[H"]
    for i in range(1, top):
        out.append("\x1b[%d;1H\x1b[2K" % i)
    for idx, line in enumerate(block):
        out.append("\x1b[%d;1H\x1b[2K%s" % (top + idx, line))
    for i in range(top + len(block), h + 1):
        out.append("\x1b[%d;1H\x1b[2K" % i)
    return "".join(out)


def run_loading(ui, label: str, loader, title: str = LOAD_TITLE):
    """Run `loader()` while showing a centered Alethic animation; plain message when non-interactive."""
    if not (ui.isatty and ui.vt and ui.animate):
        ui.info("loading the model…")
        return loader()

    result = {}
    done = threading.Event()

    def worker() -> None:
        try:
            result["value"] = loader()
        except BaseException as exc:
            result["error"] = exc
        finally:
            done.set()

    clear_console(ui)
    t0 = time.time()
    step = 0
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    ui._write("\x1b[?25l")
    try:
        while True:
            cols, rows = ui._size()
            frame = render_loading_frame(cols, rows, time.time() - t0, step, label, ui, title=title)
            ui._write(frame)
            step += 1
            if done.is_set():
                break
            done.wait(0.07)
    finally:
        ui._write("\x1b[?25h")
    th.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")
