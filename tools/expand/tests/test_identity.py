import re

from tools.expand import identity
from tools.sftbuild.plain import build_plain

TEXT = identity.make_text()
EXAMPLES = re.split(r"\n(?=### ex )", TEXT.strip())


def test_about_sixty_examples_in_both_languages_with_unique_ids_and_questions():
    ids = re.findall(r"^### ex (\d+) \| T \| (en|uz)", TEXT, re.M)
    assert 55 <= len(ids) <= 70 and len({i for i, _ in ids}) == len(ids)
    assert {l for _, l in ids} == {"en", "uz"}
    users = re.findall(r"<user>\n(.*?)\n</user>", TEXT, re.S)
    assert len(set(users)) >= 45


def test_every_example_that_asks_who_it_is_names_alethic_and_never_another_model():
    assert not re.search(r"\bI am (ChatGPT|GPT|Claude|Gemini|Llama|DeepSeek)", TEXT)
    assert not re.search(r"Men (ChatGPT|GPT|Claude|Gemini|Llama|DeepSeek)", TEXT)
    for ex in EXAMPLES:
        a = " ".join(re.findall(r"<assistant>\n(.*?)\n</assistant>", ex, re.S))
        assert "Alethic" in a or "Bekhruz" in a or "internet" in a.lower() or "xotira" in a.lower() or "internet" in a, ex[:80]


def test_required_facts_appear_somewhere():
    for fact in ["300M", "Bekhruz Suleyman", "Bekhruz Suleyman", "https://huggingface.co/bekhruzsuleyman/alethic-300m", "open-source", "ochiq manba"]:
        assert fact in TEXT, fact
    assert "’" not in TEXT and "‘" not in TEXT                      # ASCII apostrophes only


def test_builds_with_the_plain_builder_and_only_assistant_text_is_learned():
    samples = build_plain(TEXT)
    assert len(samples) >= 55 and all(any(s["loss"] for s in x["segments"]) for x in samples)
