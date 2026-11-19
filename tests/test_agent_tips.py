import re

from alethic.agent import menu, tips


def test_tips_are_short_unique_and_only_mention_real_commands():
    assert len(tips.TIPS) >= 10 and len(set(tips.TIPS)) == len(tips.TIPS)
    names = {c.name for c in menu.COMMANDS}
    for t in tips.TIPS:
        assert len(t) <= 78, t
        for cmd in re.findall(r"/(\w+)", t):
            assert cmd in names, (t, cmd)


def test_the_tip_changes_with_work_not_with_time_and_never_repeats_back_to_back():
    seen = [tips.tip_at(step, seed=3) for step in range(0, 200)]
    assert all(a != b for a, b in zip(seen[::tips.EVERY], seen[tips.EVERY::tips.EVERY]))
    assert tips.tip_at(5, 3) == tips.tip_at(5, 3)
    assert set(seen) == set(tips.TIPS)
    assert tips.tip_at(0, 1) != tips.tip_at(0, 2) or tips.tip_at(1, 1) != tips.tip_at(1, 2)
