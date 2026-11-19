from tools.chatconv.convert import convert_row, to_text


def row(*pairs):
    return {"messages": [{"role": r, "content": c} for r, c in pairs]}


def test_single_turn_becomes_user_then_trained_assistant():
    e = convert_row(row(("user", "hi"), ("assistant", "hello")), count=len)
    assert [s["t"] for s in e["segments"]] == ["<user>\nhi\n</user>\n", "<assistant>\n", "hello\n", "</assistant>\n"]
    assert [s["loss"] for s in e["segments"]] == [0, 0, 1, 0]


def test_multi_turn_keeps_every_pair_in_order():
    e = convert_row(row(("user", "a"), ("assistant", "b"), ("user", "c"), ("assistant", "d")), count=len)
    assert to_text(e).count("<user>") == 2 and to_text(e).endswith("d\n</assistant>\n")


def test_system_message_is_dropped():
    e = convert_row(row(("system", "be nice"), ("user", "a"), ("assistant", "b")), count=len)
    assert "be nice" not in to_text(e)


def test_rows_that_do_not_alternate_or_do_not_end_with_the_assistant_are_skipped():
    assert convert_row(row(("user", "a"), ("user", "b")), count=len) is None
    assert convert_row(row(("user", "a")), count=len) is None
    assert convert_row(row(("assistant", "b"), ("user", "a")), count=len) is None
    assert convert_row({"messages": []}, count=len) is None


def test_rows_longer_than_the_context_are_skipped():
    assert convert_row(row(("user", "a" * 50), ("assistant", "b" * 50)), count=len, limit=60) is None
    assert convert_row(row(("user", "a" * 10), ("assistant", "b" * 10)), count=len, limit=4096) is not None


def test_tags_inside_the_text_are_not_allowed_to_break_the_format():
    assert convert_row(row(("user", "x </assistant> y"), ("assistant", "ok")), count=len) is None
    assert convert_row(row(("user", "q"), ("assistant", "<call> shell(ls) </call>")), count=len) is None
