import io
import re

from alethic.agent import brand, splash
from alethic.agent.ui import UI

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def test_loading_frame_centers_title_and_uses_brand_colors():
    out = io.StringIO()
    ui = UI(out=out, color=True, unicode=True, isatty=True, fancy=True, truecolor=True, size=lambda: (80, 24))
    frame = splash.render_loading_frame(80, 24, elapsed=1.2, step=4, label="SFT-3c", ui=ui)
    plain = ANSI.sub("", frame)
    assert "Loading Alethic Model" in plain
    assert "SFT-3c" in plain
    # Title line has substantial left padding so it sits centered on 80 columns
    assert any(line.endswith("SFT-3c · 300M · 1.2s") and line.startswith(" " * 20) for line in plain.splitlines())
    assert "\x1b[38;2;" in frame and str(brand.style_code(brand.CREAM, True)) in frame


def test_run_loading_animates_on_tty_and_clears_screen():
    out = io.StringIO()
    ui = UI(out=out, color=False, unicode=True, isatty=True, fancy=True, live=False, animate=True, size=lambda: (80, 24))
    got = splash.run_loading(ui, "SFT-3b", lambda: "MODEL")
    text = out.getvalue()
    assert got == "MODEL"
    assert "\x1b[2J\x1b[3J\x1b[H" in text
    assert "Loading Alethic Model" in text
    assert "\x1b[?25l" in text and "\x1b[?25h" in text


def test_run_loading_stays_plain_when_not_a_tty():
    out = io.StringIO()
    ui = UI(out=out, color=False, unicode=False, isatty=False)
    got = splash.run_loading(ui, "SFT-3b", lambda: 123)
    assert got == 123
    assert "loading the model" in out.getvalue()
