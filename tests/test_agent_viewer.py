from alethic.agent import brand, viewer


def plain(row):
    return "".join(t for t, _ in row)


CODE = "\n".join("x%d = %d  # line" % (i, i) for i in range(1, 101)) + "\n"


def test_widths_put_the_document_on_the_right_and_respect_the_chosen_width():
    chat, doc = viewer.widths(120)
    assert chat + doc + 1 == 120 and doc > chat and chat >= 20 and doc >= 30
    chat, doc = viewer.widths(120, 80)
    assert doc == 80 and chat == 39
    assert viewer.widths(120, 200)[0] == 20 and viewer.widths(120, 5)[1] == 30      # clamped
    assert viewer.widths(60) == (0, 60)


def test_arrows_resize_the_document_in_steps_inside_the_limits():
    d = viewer.widths(120)[1]
    assert viewer.resize(None, "left", 120) == d + 4 and viewer.resize(None, "]", 120) == d - 4
    assert viewer.resize(None, "x", 120) is None
    wide = 120
    for _ in range(40):
        wide = viewer.resize(wide, "left", 120)
    assert viewer.widths(120, wide)[0] == 20
    assert viewer.resize(None, "left", 60) == 60              # narrow window: nothing to resize


def test_scrolling_moves_by_line_and_page_and_stays_inside_the_file():
    assert viewer.scroll(0, "down", 20, 100) == 1
    assert viewer.scroll(0, "up", 20, 100) == 0
    assert viewer.scroll(10, "pgdn", 20, 100) == 29       # a page minus one line of overlap
    assert viewer.scroll(10, "pgup", 20, 100) == 0
    assert viewer.scroll(10, "end", 20, 100) == 80        # last page: the file ends at the bottom row
    assert viewer.scroll(80, "down", 20, 100) == 80
    assert viewer.scroll(50, "home", 20, 100) == 0
    assert viewer.scroll(5, "x", 20, 100) == 5
    assert viewer.scroll(0, "down", 20, 10) == 0          # a short file never scrolls


def test_the_document_pane_has_numbered_colored_lines_and_exact_width():
    rows = viewer.document_rows("a.py", "def add(a, b):\n    return a + b\n", 0, 10, 40)
    assert len(rows) == 10 and {len(plain(r)) for r in rows} == {40}
    assert plain(rows[0]).startswith("  1  def add(a, b):") and plain(rows[1]).startswith("  2      return")
    styles = {t: s for r in rows[:2] for t, s in r}
    assert styles["def"] == brand.KEYWORD and styles["add"] == brand.FUNC
    assert plain(rows[5]).strip() == ""                    # below the end of the file: empty rows


def test_scrolled_document_starts_at_the_right_line_and_long_lines_are_cut():
    rows = viewer.document_rows("a.py", CODE, 49, 5, 30)
    assert plain(rows[0]).strip().startswith("50") and "x50 = 50" in plain(rows[0])
    long = viewer.document_rows("a.py", "y = " + "9" * 200 + "\n", 0, 3, 30)
    assert plain(long[0]).endswith("…") and len(plain(long[0])) == 30


def test_multiline_strings_keep_their_color_when_scrolled_into_the_middle():
    code = '"""\nline two\n"""\nx = 1\n'
    rows = viewer.document_rows("a.py", code, 1, 3, 30)
    assert [s for t, s in rows[0] if "line two" in t] == [brand.STRING]


def test_chat_pane_shows_the_tail_of_the_transcript_wrapped_to_its_width():
    text = "\n".join("line %d" % i for i in range(50)) + "\n" + "w" * 70
    rows = viewer.chat_rows(text, 5, 30)
    assert len(rows) == 5 and {len(plain(r)) for r in rows} == {30}
    joined = "\n".join(plain(r).rstrip() for r in rows)
    assert "line 49" in joined and "line 3" not in joined and "w" * 10 in joined


def test_chat_lines_padded_with_spaces_do_not_wrap_into_empty_rows():
    rows = viewer.chat_rows("  hello" + " " * 80 + "\nnext", 4, 30)
    assert [plain(r).strip() for r in rows if plain(r).strip()] == ["hello", "next"]


def test_screen_has_the_chat_on_the_left_the_document_on_the_right_and_a_status_bar():
    rows = viewer.screen("a.py", CODE, "hello\nworld", top=0, cols=100, lines=12, unicode=True)
    assert len(rows) == 12 and all(len(plain(r)) == 100 for r in rows)
    chat_w, doc_w = viewer.widths(100)
    header = plain(rows[0])
    assert header[:chat_w].strip() == "chat" and header[chat_w + 1:].strip() == "a.py"
    body = plain(rows[2])
    assert body[chat_w] == "│" and "x2 = 2" in body[chat_w + 1:] and "x2 = 2" not in body[:chat_w]
    assert "hello" in "\n".join(plain(r)[:chat_w] for r in rows)
    bar = plain(rows[-1])
    assert "1-10 of 100" in bar and "q" in bar and "resize" in bar


def test_screen_follows_a_custom_document_width():
    rows = viewer.screen("a.py", CODE, "c", top=0, cols=100, lines=8, unicode=True, doc=70)
    assert plain(rows[2])[100 - 70 - 1] == "│"


def test_narrow_window_shows_only_the_document():
    rows = viewer.screen("a.py", CODE, "chat chat", top=0, cols=50, lines=8, unicode=True)
    assert "chat chat" not in "\n".join(plain(r) for r in rows) and all(len(plain(r)) == 50 for r in rows)


def test_ascii_mode_uses_ascii_divider():
    rows = viewer.screen("a.py", "x = 1\n", "hi", top=0, cols=100, lines=6, unicode=False)
    assert all(plain(r).isascii() for r in rows)


def test_run_scrolls_with_keys_and_leaves_the_alternate_screen():
    import io
    out = io.StringIO()
    keys = iter(["down", "left", "down", "pgdn", "q"])
    viewer.run("a.py", CODE, "chat", read=lambda: next(keys), write=out.write, size=lambda: (100, 12), unicode=True, color=False)
    text = out.getvalue()
    assert text.startswith("\x1b[?1049h") and text.rstrip().endswith("\x1b[?1049l") or "\x1b[?1049l" in text
    assert "x1 = 1" in text and "x12 = 12" in text          # first screen, then after scrolling
    assert text.index("\x1b[?1049h") < text.index("\x1b[?1049l")


def test_plain_print_for_dumb_terminals_shows_the_whole_file_with_numbers():
    out = viewer.plain_text("a.py", "a = 1\nb = 2\n")
    assert out.splitlines() == ["1  a = 1", "2  b = 2"]
