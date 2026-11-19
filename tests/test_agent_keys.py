from alethic.agent.keys import decode_unix, decode_windows


def decode(data: bytes) -> str:
    """Decode the first key from a fixed byte string; the reader can say whether more bytes are waiting."""
    state = {"i": 0}

    def read():
        b = data[state["i"]]
        state["i"] += 1
        return b

    def ready(timeout=0.0):
        return state["i"] < len(data)
    return decode_unix(read, ready)


def win(*chars):
    it = iter(chars)
    return decode_windows(lambda: next(it))


def test_unix_plain_and_control_keys():
    assert decode(b"a") == "a" and decode(b" ") == " "
    assert decode(b"\r") == "enter" and decode(b"\n") == "enter"
    assert decode(b"\x7f") == "backspace" and decode(b"\x08") == "backspace"
    assert decode(b"\t") == "tab"
    assert decode(b"\x03") == "ctrl-c" and decode(b"\x04") == "ctrl-d"
    assert decode(b"\x15") == "ctrl-u" and decode(b"\x17") == "ctrl-w"
    assert decode(b"\x01") == "home" and decode(b"\x05") == "end"


def test_unix_escape_alone_is_esc_but_with_more_bytes_it_is_an_arrow():
    assert decode(b"\x1b") == "esc"
    assert decode(b"\x1b[A") == "up" and decode(b"\x1b[B") == "down"
    assert decode(b"\x1b[C") == "right" and decode(b"\x1b[D") == "left"
    assert decode(b"\x1bOA") == "up" and decode(b"\x1bOD") == "left"


def test_unix_home_end_delete_variants():
    for seq, key in ((b"\x1b[H", "home"), (b"\x1b[F", "end"), (b"\x1b[1~", "home"), (b"\x1b[4~", "end"),
                     (b"\x1b[7~", "home"), (b"\x1b[8~", "end"), (b"\x1b[3~", "delete"), (b"\x1bOH", "home"), (b"\x1bOF", "end")):
        assert decode(seq) == key, seq


def test_unix_unknown_escape_sequences_are_swallowed_whole():
    assert decode(b"\x1b[2~") == "unknown"            # insert
    assert decode(b"\x1b[1;5C") == "unknown"          # ctrl+right


def test_unix_utf8_characters_are_decoded_from_several_bytes():
    assert decode("ы".encode()) == "ы" and decode("ş".encode()) == "ş" and decode("€".encode()) == "€"


def test_windows_keys_and_prefixed_arrows():
    assert win("a") == "a" and win("\r") == "enter" and win("\x08") == "backspace" and win("\t") == "tab"
    assert win("\x1b") == "esc" and win("\x03") == "ctrl-c" and win("\x04") == "ctrl-d"
    assert win("\x00", "H") == "up" and win("\xe0", "P") == "down" and win("\xe0", "K") == "left" and win("\xe0", "M") == "right"
    assert win("\xe0", "G") == "home" and win("\xe0", "O") == "end" and win("\xe0", "S") == "delete"
    assert win("\xe0", "I") == "pgup" and win("\xe0", "Q") == "pgdn" and win("\xe0", "R") == "unknown"
    assert win("ы") == "ы"


def test_unix_left_clicks_become_click_keys_and_other_mouse_events_are_ignored():
    assert decode(b"\x1b[<0;12;5M") == "click:12:5"
    assert decode(b"\x1b[<0;12;5m") == "unknown"          # button release
    assert decode(b"\x1b[<64;1;1M") == "wheel-up"         # wheel up
    assert decode(b"\x1b[<65;1;1M") == "wheel-down"       # wheel down
    assert decode(b"\x1b[<2;3;4M") == "unknown"           # right button


def win_seq(text):
    """A key press where the console hands over an escape sequence one character at a time (VT input mode)."""
    it = iter(text)
    left = {"n": len(text)}

    def getwch():
        left["n"] -= 1
        return next(it)
    return decode_windows(getwch, lambda: left["n"] > 0)


def test_windows_vt_input_sequences_give_arrows_pages_and_clicks_and_a_lone_escape_stays_esc():
    assert win_seq("\x1b[A") == "up" and win_seq("\x1b[D") == "left" and win_seq("\x1b[5~") == "pgup"
    assert win_seq("\x1b[<0;7;3M") == "click:7:3" and win_seq("\x1b[<0;7;3m") == "unknown"
    assert win_seq("\x1b") == "esc"
