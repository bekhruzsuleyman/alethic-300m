from alethic.agent import screenmap as sm


def test_lines_are_wrapped_to_the_window_and_blocks_mark_their_header_row():
    segs = ["intro\n", sm.BlockMark("A", header="▸ run", body=["x", "y" * 25]), "outro\n"]
    lines = sm.physical_lines(segs, cols=10)
    assert [t for t, _ in lines] == ["intro", "▸ run", "x", "yyyyyyyyyy", "yyyyyyyyyy", "yyyyy", "outro"]
    assert [m for _, m in lines] == [None, "A", None, None, None, None, None]


def test_a_long_header_marks_only_its_first_row_and_text_without_a_newline_joins_the_next():
    lines = sm.physical_lines(["ab", "cd\n", sm.BlockMark("B", header="h" * 25, body=[])], cols=10)
    assert [t for t, _ in lines] == ["abcd", "hhhhhhhhhh", "hhhhhhhhhh", "hhhhh"]
    assert [m for _, m in lines] == [None, "B", None, None]


def test_ansi_codes_do_not_count_as_width():
    lines = sm.physical_lines(["\x1b[1mhello\x1b[0m\n"], cols=10)
    assert [t for t, _ in lines] == ["hello"]


def test_a_short_screen_is_top_aligned_and_a_long_one_is_bottom_aligned():
    lines = [("l%d" % i, "B%d" % i if i == 2 else None) for i in range(1, 6)]     # 5 lines, header of B2 on line 2
    assert sm.hit(lines, row=2, rows=24, frame_h=4) == "B2"               # everything fits: line i is on row i
    assert sm.hit(lines, row=1, rows=24, frame_h=4) is None
    assert sm.hit(lines, row=9, rows=24, frame_h=4) is None               # empty or frame rows
    # 5 lines + frame of 4 = 9 rows, window of 6 rows: 3 rows scrolled away, so line 2 is hidden and line 5 is on row 2
    assert sm.hit(lines, row=1, rows=6, frame_h=4) is None
    lines2 = [("l%d" % i, "H%d" % i) for i in range(1, 6)]
    assert sm.hit(lines2, row=1, rows=6, frame_h=4) == "H4" and sm.hit(lines2, row=2, rows=6, frame_h=4) == "H5"
    assert sm.hit(lines2, row=3, rows=6, frame_h=4) is None               # the frame starts here
    assert sm.hit(lines2, row=1, rows=6, frame_h=4, scroll=2) == "H2"


def test_wrap_ansi_lines_keeps_color_codes_across_wraps():
    wrapped = sm.wrap_ansi_lines("\x1b[33mabcdefghij\x1b[0m\n", cols=4)
    assert [sm.ANSI.sub("", row) for row in wrapped] == ["abcd", "efgh", "ij"]
    assert wrapped[0].startswith("\x1b[33m") and wrapped[0].endswith("\x1b[0m")
    assert wrapped[1].startswith("\x1b[33m") and wrapped[1].endswith("\x1b[0m")
