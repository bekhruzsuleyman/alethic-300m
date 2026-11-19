from alethic.agent import blocks, brand
from alethic.agent.protocol import Call


def plain(row):
    return "".join(t for t, _ in row)


def flat(rows):
    return "\n".join(plain(r) for r in rows)


def test_block_rows_all_have_the_same_width_a_title_and_no_border_characters():
    rows = blocks.box("terminal", [[("hello", "")], blocks.SEP, [("out", "")]], 30, True)
    assert {len(plain(r)) for r in rows} == {30}
    assert plain(rows[0]).strip() == "terminal" and plain(rows[1]).strip() == "hello" and plain(rows[2]).strip() == ""
    assert not set("╭╮╰╯│├┤─+|") & set(flat(rows))


def test_block_cuts_long_lines_with_an_ellipsis():
    rows = blocks.box("t", [[("x" * 100, "")]], 20, True)
    assert plain(rows[1]).rstrip().endswith("…") and len(plain(rows[1])) == 20


def test_shell_view_names_the_action_and_boxes_command_and_output():
    v = blocks.tool_view(Call("shell", 'python3 -c "print(1+1)"'), "2", True, 60, True)
    assert plain(v.header) == "▸ Running command in terminal"
    text = flat(v.rows)
    assert '$ python3 -c "print(1+1)"' in text and "\n  2" in text and "├" not in text
    assert all(len(plain(r)) == 60 for r in v.rows)


def test_long_output_is_cut_and_failed_output_is_red():
    out = "\n".join("line %d" % i for i in range(30))
    v = blocks.tool_view(Call("shell", "ls"), out, True, 60, True)
    assert "… +20 more lines" in flat(v.rows)
    bad = blocks.tool_view(Call("shell", "false"), "boom", False, 60, True)
    assert ("boom", "red") in [(t.strip(), s) for r in bad.rows for t, s in r if t.strip()]


def test_a_call_still_waiting_for_its_result_has_only_a_header():
    v = blocks.tool_view(Call("shell", "sleep 3"), None, True, 60, True)
    assert v.rows == [] and plain(v.header).startswith("▸ Running")


def test_read_view_shows_file_and_range_with_numbered_colored_code():
    res = "1   def add(a, b):\n2       return a + b"
    v = blocks.tool_view(Call("read", "calc.py, 1-100"), res, True, 60, True)
    assert plain(v.header) == "▸ Reading calc.py lines 1-100"
    styles = {t.strip(): s for r in v.rows for t, s in r if t.strip()}
    assert styles["def"] == brand.KEYWORD and styles["add"] == brand.FUNC and styles["1"] == brand.MUTED


def test_write_and_edit_views_show_the_new_code_and_the_result():
    w = blocks.tool_view(Call("write", "hello.py", 'print("hi")\n'), "wrote hello.py (1 line)", True, 60, True)
    assert plain(w.header) == "▸ Writing hello.py"
    assert 'print("hi")' in flat(w.rows) and "wrote hello.py" in flat(w.rows)
    e = blocks.tool_view(Call("edit", "calc.py, 5-6", "x = 1\n"), "edited", True, 60, True)
    assert plain(e.header) == "▸ Editing calc.py lines 5-6"


def test_long_code_is_cut_and_points_to_the_viewer():
    body = "\n".join("x%d = %d" % (i, i) for i in range(40))
    v = blocks.tool_view(Call("write", "big.py", body), "wrote", True, 60, True)
    assert "… +26 more lines" in flat(v.rows) and "/open" in flat(v.rows)


def test_search_view_names_query_and_source_and_lists_results():
    res = "1. Tokyo (en.wikipedia.org/wiki/Tokyo)\n   Tokyo is the capital of Japan."
    v = blocks.tool_view(Call("search", "capital of Japan, wikipedia"), res, True, 70, True)
    assert plain(v.header) == '▸ Searching wikipedia for "capital of Japan"'
    assert "Tokyo" in flat(v.rows) and "capital of Japan." in flat(v.rows)
    none = blocks.tool_view(Call("search", "blorptangle"), "no results", True, 70, True)
    assert "no results" in flat(none.rows) and "wikipedia" in plain(none.header)


def test_unknown_call_still_gets_a_block():
    v = blocks.tool_view(Call("?", "weird"), None, True, 60, True)
    assert plain(v.header).startswith("▸ ")


def test_no_row_is_wider_than_the_window_at_any_width():
    for w in (24, 31, 40, 80, 130):
        v = blocks.tool_view(Call("shell", "echo " + "z" * 200), "o" * 300, True, w, True)
        assert all(len(plain(r)) <= w for r in v.rows)


def feed_all(p, *chunks):
    ev = []
    for c in chunks:
        ev += p.feed(c)
    return ev + p.finish()


def test_fence_parser_splits_text_and_code():
    ev = feed_all(blocks.FenceParser(), "Here:\n```python\nx = 1\nprint(x)\n```\nDone.")
    kinds = [e[0] for e in ev]
    assert kinds.count("open") == 1 and kinds.count("close") == 1
    assert ("open", "python") in ev and ("code", "x = 1") in ev and ("code", "print(x)") in ev
    assert "".join(e[1] for e in ev if e[0] == "text") == "Here:\nDone."


def test_fence_parser_works_when_the_fence_is_split_across_chunks():
    ev = feed_all(blocks.FenceParser(), "a\n``", "`rust\nlet x", " = 1;\n``", "`\nb")
    assert ("open", "rust") in ev and ("code", "let x = 1;") in ev and ("close",) in ev
    assert "".join(e[1] for e in ev if e[0] == "text") == "a\nb"


def test_unterminated_fence_is_closed_at_the_end_and_inline_ticks_are_text():
    ev = feed_all(blocks.FenceParser(), "```\nx = 1")
    assert ev[-1] == ("close",) and ("code", "x = 1") in ev
    ev = feed_all(blocks.FenceParser(), "use `ls` and ```not a fence``` here")
    assert [e[0] for e in ev] == ["text"] * len(ev)


def test_text_before_a_possible_fence_is_not_held_back_forever():
    p = blocks.FenceParser()
    ev = p.feed("hello wor")
    assert "".join(e[1] for e in ev) == "hello wor"
