from tools.release import update_card as uc

OLD = "---\nlicense: mit\ntags:\n- code\n---\n\n# old body\n"
BODY = "# Alethic-300M\n\nnew body\n"


def test_keeps_the_existing_front_matter_and_replaces_the_body():
    out = uc.merge_card(OLD, BODY)
    assert out.startswith("---\nlicense: mit\ntags:\n- code\n---\n")
    assert "old body" not in out and out.rstrip().endswith("new body")


def test_old_card_without_front_matter_gets_none_invented():
    out = uc.merge_card("# just text\n", BODY)
    assert out == BODY


def test_body_must_not_carry_its_own_front_matter():
    import pytest
    with pytest.raises(uc.CardError):
        uc.merge_card(OLD, "---\nlicense: apache-2.0\n---\nx")


def test_model_card_body_points_to_the_standalone_agent_repo():
    from pathlib import Path
    body = Path(uc.BODY_PATH).read_text(encoding="utf-8")
    assert "github.com/bekhruzsuleyman/alethic-agent" in body
