from tools.sftbuild.normalize import Example, parse_examples

LONG = """### ex 0211 | type: A | lang: en | diff: easy | topic: x | source: deepseek
```text
Why does ice float?
```
```text
Ice is less dense than water.
```
"""
SHORT = """### ex 0211 | A | en | deepseek
<user>
Why does ice float?
</user>
<assistant>
Ice is less dense than water.
</assistant>
"""


def test_long_header():
    ex = parse_examples(LONG)[0]
    assert (ex.id, ex.type, ex.lang, ex.source) == ("0211", "A", "en", "deepseek")


def test_short_header():
    ex = parse_examples(SHORT)[0]
    assert (ex.id, ex.type, ex.lang, ex.source) == ("0211", "A", "en", "deepseek")


def test_plain_tags_and_text_fence_agree():
    a, b = parse_examples(LONG)[0], parse_examples(SHORT)[0]
    assert a.user == b.user == "Why does ice float?"
    assert a.assistant == b.assistant == "Ice is less dense than water."


def test_fenced_text_and_four_backtick_code():
    t = "### ex 0050 | type: C | lang: code | diff: easy | topic: x | source: deepseek\n```text\nwrite add\n```\n````text\n```python\ndef add(a, b):\n    return a + b\n```\n````\n"
    ex = parse_examples(t)[0]
    assert ex.user == "write add"
    assert ex.assistant == "```python\ndef add(a, b):\n    return a + b\n```"


def test_call_line_inside_content_survives():
    t = "### ex 0001 | A | en | deepseek\n<user>\nq\n</user>\n<assistant>\n<call> x </call>\nend\n</assistant>\n"
    assert parse_examples(t)[0].assistant == "<call> x </call>\nend"


def test_multiple_examples():
    assert [e.id for e in parse_examples(LONG + "\n" + SHORT.replace("0211", "0212"))] == ["0211", "0212"]


def test_tags_wrapped_in_four_backtick_fence():
    t = "### ex 0336 | A | en | deepseek\n````text\n<user>\nq\n</user>\n<assistant>\n```python\nx = 1\n```\n</assistant>\n````\n"
    ex = parse_examples(t)[0]
    assert ex.user == "q"
    assert ex.assistant == "```python\nx = 1\n```"
