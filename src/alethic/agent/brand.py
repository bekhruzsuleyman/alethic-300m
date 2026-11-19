"""The Alethic look in the terminal: colors, the two-disc mark, the spinner.

Everything here is a pure function (no printing, no clock reads), so it is easy to test.
Colors are written as "#RRGGBB"; style_code() turns one into a truecolor code or the nearest of the 256 colors.
"""
from __future__ import annotations

# palette (dark terminal)
AMBER = "#FFB020"        # accent: prompt arrow, tool arrows, spinner
CREAM = "#F5E9C8"        # main text
MUTED = "#A39E8A"        # secondary text
RULE = "#3A3730"         # thin lines
BLOCK_BG = "#1C1B17"     # background of code and tool blocks (a little lighter than the terminal)
DISC_BLUE = "#4F6BFF"    # left disc
DISC_AMBER = "#F2B01E"   # right disc
LENS = "#F5E9C8"         # where the discs overlap
# code colors, taken from the logo colors
KEYWORD = "#7B95FF"
STRING = "#9CCB8B"
NUMBER = "#F2B01E"
COMMENT = "#7A7566"
NAME = "#F5E9C8"
FUNC = "#8FD3FF"
BUILTIN = "#FF6F96"
DECORATOR = "#3DDCC1"

NAMED = {"dim": "2", "bold": "1", "italic": "3", "red": "31", "green": "32", "yellow": "33", "blue": "34",
         "magenta": "35", "cyan": "36", "brand": "38;5;208"}

MARK_WIDTH = 17
_HALF = [2, 4, 5, 4, 2]          # half width of a disc on each of its 5 rows
_A_CENTER, _B_CENTER = 5, 11
SPIN_UNI, SPIN_ASCII, SPIN_SECONDS = "◐◓◑◒", "-\\|/", 0.12


def rgb_to_256(r: int, g: int, b: int) -> int:
    """The nearest color in the 256-color table (6x6x6 cube or the gray ramp)."""
    if abs(r - g) < 10 and abs(g - b) < 10:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return 232 + round((r - 8) / 247 * 23)
    q = lambda v: round(v / 255 * 5)  # noqa: E731
    return 16 + 36 * q(r) + 6 * q(g) + q(b)


def style_code(style: str, truecolor: bool) -> str:
    """ANSI parameters for a style name ("bold", "dim"...) or a "#RRGGBB" color."""
    if style.startswith("#") and len(style) == 7:
        r, g, b = (int(style[i:i + 2], 16) for i in (1, 3, 5))
        return "38;2;%d;%d;%d" % (r, g, b) if truecolor else "38;5;%d" % rgb_to_256(r, g, b)
    return NAMED[style]


def bg_code(color: str, truecolor: bool) -> str:
    """ANSI parameters for a background "#RRGGBB"."""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return "48;2;%d;%d;%d" % (r, g, b) if truecolor else "48;5;%d" % rgb_to_256(r, g, b)


def mark_rows(unicode: bool = True, gap: int = 0) -> list:
    """The logo: two overlapping discs, 5 rows. Each row is [(text, style)]; `gap` moves the right disc away
    (used by the start-up animation, where the discs slide together)."""
    fill = ("█", "█", "█") if unicode else ("#", "=", "%")
    width = MARK_WIDTH + gap
    rows = []
    for y in range(5):
        cells = []
        for x in range(width):
            in_a = abs(x - _A_CENTER) <= _HALF[y]
            in_b = abs(x - (_B_CENTER + gap)) <= _HALF[y]
            if in_a and in_b:
                cells.append((fill[2], LENS))
            elif in_a:
                cells.append((fill[0], DISC_BLUE))
            elif in_b:
                cells.append((fill[1], DISC_AMBER))
            else:
                cells.append((" ", ""))
        row = []
        for ch, st in cells:
            if row and row[-1][1] == st:
                row[-1] = (row[-1][0] + ch, st)
            else:
                row.append((ch, st))
        rows.append(row)
    return rows


def intro_gaps() -> list:
    """How far apart the discs are in each frame of the start-up animation; the last frame is the finished logo."""
    return [8, 6, 4, 3, 2, 1, 0]


def spinner(t: float, unicode: bool = True) -> str:
    """One frame of the little cycling spinner for time t (seconds)."""
    frames = SPIN_UNI if unicode else SPIN_ASCII
    return frames[int(t / SPIN_SECONDS + 1e-6) % len(frames)]
