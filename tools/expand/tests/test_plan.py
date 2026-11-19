import re
import pytest

from tools.expand import plan

KINDS = ["T", "E", "R", "P", "Z"]


@pytest.mark.parametrize("kind", KINDS)
def test_rows_are_deterministic_and_have_the_right_count(kind):
    a = plan.make_rows(kind, 40, 3001, seed=1)
    b = plan.make_rows(kind, 40, 3001, seed=1)
    c = plan.make_rows(kind, 40, 3001, seed=2)
    assert a == b and a != c
    assert len(a) == (20 if kind == "E" else 40)   # E counts examples, rows are pairs


def test_t_rows_match_the_checker_regex_and_ids_are_consecutive():
    rows = plan.make_rows("T", 30, 3001, seed=1)
    ids = [int(re.match(r"(\d{4})\s+(en|uz|mix)\s+(\d)\s+([A-Z+]+)\s", r).group(1)) for r in rows]
    assert ids == list(range(3001, 3031))


def test_t_skill_mix_and_language_mix():
    rows = plan.make_rows("T", 100, 3001, seed=3)
    skills = {r.split()[3] for r in rows}
    assert {"F", "C", "S", "U", "K", "N", "P", "S+L"} <= skills
    langs = [r.split()[1] for r in rows]
    assert min(langs.count(l) for l in ("en", "uz", "mix")) >= 25
    # topic surface is never repeated inside one plan
    topics = [r.split(None, 4)[4] for r in rows]
    assert len(set(topics)) == len(topics)


def test_e_rows_are_pairs_with_valid_verdict_changes():
    rows = plan.make_rows("E", 40, 4301, seed=1, first_pair=1)
    pat = re.compile(r"^(\d+)\s+(\d{4})/(\d{4})\s+(en|uz|mix)\s+(.*?)\s+(strong|risky|weak|not yet) -> (strong|risky|weak|not yet)\s*$")
    for i, r in enumerate(rows):
        m = pat.match(r)
        assert m, r
        assert int(m.group(1)) == i + 1
        assert int(m.group(3)) == int(m.group(2)) + 1
        assert m.group(6) != m.group(7)
    assert int(pat.match(rows[0]).group(2)) == 4301
    assert int(pat.match(rows[1]).group(2)) == 4303


def test_e_pair_numbers_can_continue_across_batches():
    rows = plan.make_rows("E", 20, 4321, seed=1, first_pair=11)
    assert rows[0].split()[0] == "11"


def test_r_rows_are_70_percent_refuse_and_30_percent_control():
    rows = plan.make_rows("R", 50, 5401, seed=1)
    kinds = [r.split()[3] for r in rows]
    assert kinds.count("refuse") == 35 and kinds.count("control") == 15
    two = [r for r in rows if r.split()[3] == "refuse" and r.split()[2] == "2"]
    assert 8 <= len(two) <= 16


def test_p_rows_cover_every_stack_and_use_allowed_stacks_only():
    rows = plan.make_rows("P", 60, 5801, seed=1)
    stacks = {r.split()[2] for r in rows}
    assert {"python", "rust", "go", "c", "javascript", "typescript", "html_css"} <= stacks
    assert any("+" in s for s in stacks)
    assert stacks <= plan.ALLOWED_STACKS
    kinds = {r.split()[3] for r in rows}
    assert kinds == {"project", "project_fix"}


def test_z_rows_have_all_four_kinds():
    rows = plan.make_rows("Z", 40, 6701, seed=1)
    kinds = {r.split()[1] for r in rows}
    assert kinds == {"inject_file", "inject_control", "inject_search", "inject_paste"}


def test_addendum_names_the_base_prompt_and_embeds_the_table():
    rows = plan.make_rows("T", 5, 3001, seed=1)
    text = plan.addendum("T", rows, base="deepseek_TE_chats_0801-0880.md", batch=1)
    assert "deepseek_TE_chats_0801-0880.md" in text
    assert rows[0] in text and rows[-1] in text
    assert "3001" in text and "3005" in text
    assert "READY" not in text    # the addendum starts the work, it does not restart the handshake


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        plan.make_rows("Q", 5, 1, seed=1)
