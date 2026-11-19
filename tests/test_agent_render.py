from alethic.agent.render import Renderer


def flat(spans):
    return [(s, t) for s, t in spans if t]


def feed_all(text, step):
    r, out = Renderer(), []
    for i in range(0, len(text), step):
        out += r.feed(text[i:i + step])
    out += r.finish()
    merged = []
    for s, t in out:
        if merged and merged[-1][0] == s:
            merged[-1] = (s, merged[-1][1] + t)
        else:
            merged.append((s, t))
    return merged


def test_plan_is_shown_dim_without_tags():
    assert feed_all("<plan> do x </plan>\nHello", 100) == [("plan", " do x "), ("text", "\nHello")]


def test_calls_and_file_bodies_are_hidden():
    text = "<call> write(a.py) </call>\nprint(1)\n<call> shell(python3 a.py) </call>\nDone"
    assert feed_all(text, 100) == [("text", "\nDone")]


def test_close_tag_is_hidden_and_map_is_dim():
    assert feed_all("<map> f 1-3 </map>\nok\n</assistant>", 100) == [("map", " f 1-3 "), ("text", "\nok\n")]


def test_partial_tag_is_held_back_until_complete():
    r = Renderer()
    assert flat(r.feed("Hi <pl")) == [("text", "Hi ")]
    assert flat(r.feed("an> x </plan>")) == [("plan", " x ")]


def test_a_lone_less_than_sign_is_plain_text():
    assert feed_all("a < b and c", 100) == [("text", "a < b and c")]


def test_chunking_does_not_change_the_output():
    text = "<plan> p </plan>\n<call> write(a.py) </call>\nx\n<call> shell(ls) </call>\nAll <b>done</b>.\n</assistant>"
    assert feed_all(text, 1) == feed_all(text, 100) == feed_all(text, 7)


import re

LEAK = re.compile(r"</?(user|assistant|call|result|plan|map|think|system|tool)>|<\|[^|>]*\|>")


def test_unknown_protocol_tags_never_reach_the_screen_at_any_split():
    text = "Hi <user> x </user> there <think>hmm</think> <|endoftext|> ok </assistant>"
    for step in (1, 2, 3, 5, 100):
        shown = "".join(t for _, t in feed_all(text, step))
        assert not LEAK.search(shown), (step, shown)
        assert "Hi" in shown and "there" in shown and "ok" in shown


def test_unfinished_tag_at_the_end_is_dropped_not_printed():
    for tail in ("<ca", "<call", "</assist", "<|endof", "</"):
        shown = "".join(t for _, t in feed_all("Done " + tail, 1))
        assert shown == "Done ", (tail, shown)


def test_ordinary_angle_brackets_in_answers_stay():
    shown = "".join(t for _, t in feed_all("if a < b and x <= 3: use <div> and List<int>", 2))
    assert shown == "if a < b and x <= 3: use <div> and List<int>"


def test_thinking_blocks_are_shown_as_dim_thinking_text_never_as_tags():
    text = "<thinking>\nAverage = 60 km\n</thinking>\nThe answer is 50."
    for step in (1, 3, 100):
        spans = feed_all(text, step)
        assert [s for s, _ in spans] == ["thinking", "text"], (step, spans)
        assert "Average = 60 km" in spans[0][1] and "<" not in "".join(t for _, t in spans)
        assert spans[1][1].strip() == "The answer is 50."


def test_think_and_thought_spellings_and_an_unclosed_block_never_show_a_tag():
    for open_, close in (("<think>", "</think>"), ("<thinking>", "</thinking>"), ("<thought>", "</thought>"), ("<reasoning>", "</reasoning>")):
        shown = "".join(t for _, t in feed_all("%s hmm %s ok" % (open_, close), 2))
        assert "<" not in shown and "hmm" in shown and "ok" in shown, open_
    shown = "".join(t for _, t in feed_all("<thinking> never closed", 3))
    assert "<" not in shown
