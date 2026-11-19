from alethic.agent.protocol import scan


def test_plain_answer_is_done_without_call():
    s = scan("Tokyo.\n</assistant>")
    assert s.call is None and s.done


def test_shell_call_is_ready_as_soon_as_it_is_closed():
    text = "<plan> run it </plan>\n<call> shell(python3 a.py) </call>"
    s = scan(text)
    assert (s.call.kind, s.call.args, s.call.body) == ("shell", "python3 a.py", None)
    assert s.cut == len(text) and not s.done


def test_cut_includes_the_newline_after_the_call():
    text = "<call> read(a.py, 1-3) </call>\nlater text"
    assert scan(text).cut == text.index("\n") + 1


def test_unfinished_header_is_not_a_call():
    s = scan("<call> shell(python3 a")
    assert s.call is None and not s.done


def test_args_may_contain_parentheses():
    s = scan('<call> shell(python3 -c "print((1+2))") </call>')
    assert s.call.args == 'python3 -c "print((1+2))"'


def test_write_waits_for_the_next_call_and_cuts_before_it():
    head = "<call> write(a.py) </call>\nprint(1)\n"
    assert scan(head).call is None
    s = scan(head + "<call>")
    assert (s.call.kind, s.call.args, s.call.body) == ("write", "a.py", "print(1)\n")
    assert s.cut == len(head)


def test_write_body_also_ends_at_the_assistant_close_tag():
    head = "<call> write(a.py) </call>\nprint(1)\n"
    s = scan(head + "</assistant>")
    assert s.call.body == "print(1)\n" and s.cut == len(head)


def test_edit_has_a_body_like_write():
    s = scan("<call> edit(t.py, 6-6) </call>\n    x = 1\n<call>")
    assert (s.call.kind, s.call.args, s.call.body) == ("edit", "t.py, 6-6", "    x = 1\n")


def test_unknown_tool_still_parses():
    assert scan("<call> fly(now) </call>").call.kind == "fly"
