from alethic.agent.lineedit import LineEditor


def type_(ed, text):
    for ch in text:
        ed.feed(ch)


def test_typing_builds_the_buffer_and_enter_submits_it():
    ed = LineEditor()
    type_(ed, "hi there")
    r = ed.feed("enter")
    assert r.kind == "submit" and r.text == "hi there"


def test_backspace_delete_and_cursor_keys_edit_in_the_middle():
    ed = LineEditor()
    type_(ed, "abcd")
    ed.feed("left"); ed.feed("left")
    ed.feed("backspace")          # removes b -> acd, cursor after a
    assert ed.buffer == "acd" and ed.cursor == 1
    ed.feed("delete")             # removes c -> ad
    assert ed.buffer == "ad"
    ed.feed("end"); type_(ed, "!")
    ed.feed("home"); type_(ed, ">")
    assert ed.buffer == ">ad!"


def test_ctrl_u_clears_the_line_and_ctrl_w_deletes_a_word():
    ed = LineEditor()
    type_(ed, "one two three")
    ed.feed("ctrl-w")
    assert ed.buffer == "one two "
    ed.feed("ctrl-u")
    assert ed.buffer == "" and ed.cursor == 0


def test_ctrl_c_interrupts_and_ctrl_d_leaves_only_on_an_empty_line():
    assert LineEditor().feed("ctrl-c").kind == "interrupt"
    assert LineEditor().feed("ctrl-d").kind == "eof"
    ed = LineEditor()
    type_(ed, "ab"); ed.feed("home")
    assert ed.feed("ctrl-d") is None and ed.buffer == "b"


def test_up_and_down_walk_through_earlier_messages_and_back_to_the_draft():
    ed = LineEditor(history=["first", "second"])
    type_(ed, "dra")
    ed.feed("up");   assert ed.buffer == "second"
    ed.feed("up");   assert ed.buffer == "first"
    ed.feed("up");   assert ed.buffer == "first"
    ed.feed("down"); assert ed.buffer == "second"
    ed.feed("down"); assert ed.buffer == "dra"


def test_a_slash_opens_the_menu_and_arrows_move_the_selection():
    ed = LineEditor()
    ed.feed("/")
    assert ed.suggestions() and ed.selected == 0
    ed.feed("down"); assert ed.selected == 1
    ed.feed("up");   ed.feed("up")
    assert ed.selected == len(ed.suggestions()) - 1   # wraps round to the last one


def test_up_in_an_open_menu_does_not_touch_history():
    ed = LineEditor(history=["old"])
    ed.feed("/"); ed.feed("up")
    assert ed.buffer == "/"


def test_typing_narrows_the_menu_and_resets_the_selection():
    ed = LineEditor()
    ed.feed("/"); ed.feed("down"); ed.feed("down")
    type_(ed, "ef")
    assert [s.text for s in ed.suggestions()] == ["/effort"] and ed.selected == 0


def test_tab_completes_the_selected_command_and_adds_a_space_when_it_takes_text():
    ed = LineEditor()
    type_(ed, "/ef"); ed.feed("tab")
    assert ed.buffer == "/effort " and ed.cursor == len("/effort ")
    ed2 = LineEditor(); type_(ed2, "/sta"); ed2.feed("tab")
    assert ed2.buffer == "/status"


def test_enter_on_a_command_without_text_runs_it():
    ed = LineEditor()
    type_(ed, "/sta")
    r = ed.feed("enter")
    assert r.kind == "submit" and r.text == "/status"


def test_enter_on_a_command_that_wants_text_fills_it_in_and_waits():
    ed = LineEditor()
    type_(ed, "/e")
    assert ed.feed("enter") is None
    assert ed.buffer == "/effort " and [s.text for s in ed.suggestions()][0] == "/effort low"


def test_enter_on_a_fully_typed_command_submits_it_even_if_it_takes_text():
    ed = LineEditor()
    type_(ed, "/effort")
    r = ed.feed("enter")
    assert r.kind == "submit" and r.text == "/effort"


def test_picking_an_effort_level_runs_it():
    ed = LineEditor()
    type_(ed, "/effort ")
    ed.feed("down")
    r = ed.feed("enter")
    assert r.kind == "submit" and r.text == "/effort medium"


def test_escape_closes_the_menu_until_you_type_again():
    ed = LineEditor()
    type_(ed, "/e")
    ed.feed("esc")
    assert ed.suggestions() == []
    r = ed.feed("enter")
    assert r.kind == "submit" and r.text == "/e"          # no menu, so enter just sends the line
    ed = LineEditor(); type_(ed, "/e"); ed.feed("esc"); ed.feed("f")
    assert ed.suggestions()


def test_a_normal_message_never_shows_a_menu():
    ed = LineEditor()
    type_(ed, "write a.py")
    assert ed.suggestions() == []


def test_window_scrolls_so_the_cursor_stays_visible():
    ed = LineEditor()
    type_(ed, "x" * 50)
    text, col = ed.window(20)
    assert len(text) <= 20 and col <= 20 and text.endswith("x")
    ed.feed("home")
    text, col = ed.window(20)
    assert col == 0 and text.startswith("x")
