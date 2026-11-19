from alethic.agent import brand


def test_named_style_keeps_its_ansi_code():
    assert brand.style_code("bold", True) == "1"
    assert brand.style_code("dim", False) == "2"


def test_hex_style_is_truecolor_or_the_nearest_256_color():
    assert brand.style_code("#FFB020", True) == "38;2;255;176;32"
    code = brand.style_code("#FFB020", False)
    assert code.startswith("38;5;") and 16 <= int(code.split(";")[2]) <= 255


def test_rgb_to_256_maps_black_white_and_gray_sensibly():
    assert brand.rgb_to_256(0, 0, 0) == 16
    assert brand.rgb_to_256(255, 255, 255) == 231
    assert 232 <= brand.rgb_to_256(128, 128, 128) <= 255


def test_mark_rows_all_have_the_same_width_and_five_rows():
    for uni in (True, False):
        rows = brand.mark_rows(uni)
        assert len(rows) == 5
        widths = {sum(len(t) for t, _ in r) for r in rows}
        assert widths == {brand.MARK_WIDTH}


def test_mark_is_two_discs_blue_left_amber_right_and_a_lens_where_they_overlap():
    rows = brand.mark_rows(True)
    styles = {s for r in rows for _, s in r}
    assert {brand.DISC_BLUE, brand.DISC_AMBER, brand.LENS} <= styles
    mid = rows[2]
    flat = [(ch, s) for t, s in mid for ch in t]
    first_styled = next(s for ch, s in flat if ch != " ")
    last_styled = [s for ch, s in flat if ch != " "][-1]
    assert first_styled == brand.DISC_BLUE and last_styled == brand.DISC_AMBER


def test_plain_mark_uses_ascii_only():
    text = "".join(t for r in brand.mark_rows(False) for t, _ in r)
    assert text.isascii() and "#" in text


def test_discs_slide_together_gap_makes_rows_wider_and_zero_is_final():
    final = brand.mark_rows(True, gap=0)
    wide = brand.mark_rows(True, gap=8)
    assert sum(len(t) for t, _ in wide[0]) == brand.MARK_WIDTH + 8
    assert final == brand.mark_rows(True)
    # with a gap the discs do not overlap: no lens cells
    assert brand.LENS not in {s for r in wide for _, s in r}


def test_intro_gaps_end_at_zero_and_never_grow():
    gaps = brand.intro_gaps()
    assert gaps[-1] == 0 and gaps == sorted(gaps, reverse=True)


def test_spinner_cycles_with_the_clock_and_has_an_ascii_version():
    uni = [brand.spinner(t * 0.12, True) for t in range(8)]
    assert len(set(uni)) == 4 and uni[0] == uni[4]
    assert all(c.isascii() for c in (brand.spinner(0.0, False), brand.spinner(0.13, False)))
