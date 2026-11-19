import re

from alethic.agent import vibe

# words Claude Code's spinner uses: ours must not look copied
BANNED = {"Pondering", "Cooking", "Brewing", "Noodling", "Conjuring", "Tinkering", "Scheming", "Crunching", "Vibing", "Inferring",
          "Synthesizing", "Generating", "Calculating", "Processing", "Thinking", "Cogitating", "Percolating", "Ruminating",
          "Hallucinating", "Spelunking", "Manifesting", "Daydreaming", "Whittling", "Simmering", "Marinating", "Churning",
          "Musing", "Puzzling", "Reticulating", "Flibbertigibbeting", "Coding"}


def test_words_are_clean_original_and_numerous():
    assert len(set(vibe.WORDS)) == len(vibe.WORDS) >= 24
    assert len(set(vibe.RUNNING)) == len(vibe.RUNNING) >= 8
    for w in vibe.WORDS + vibe.RUNNING:
        assert re.fullmatch(r"[A-Z][a-z]+", w) and len(w) <= 14, w
    assert not BANNED & set(vibe.WORDS + vibe.RUNNING)


def test_the_word_changes_every_80_tokens_not_on_a_clock():
    v = vibe.Vibes(seed=1)
    first = v.word("generating")
    assert v.word("generating") == first            # asking again, even much later, never changes it
    v.tokens(79)
    assert v.word("generating") == first
    v.tokens(80)
    second = v.word("generating")
    assert second != first
    v.tokens(159)
    assert v.word("generating") == second
    v.tokens(160)
    assert v.word("generating") not in (second,)


def test_a_tool_call_moves_to_the_next_word_and_running_uses_its_own_pool():
    v = vibe.Vibes(seed=2)
    a = v.word("generating")
    v.call()
    assert v.word("generating") != a
    assert v.word("running") in vibe.RUNNING
    r1 = v.word("running")
    v.call()
    assert v.word("running") != r1 and v.word("running") in vibe.RUNNING


def test_never_repeats_back_to_back_and_every_word_gets_a_turn():
    v = vibe.Vibes(seed=3)
    seen = []
    for _ in range(300):
        seen.append(v.word("generating"))
        v.call()
    assert all(a != b for a, b in zip(seen, seen[1:]))
    assert set(seen) == set(vibe.WORDS)


def test_other_seeds_give_other_orders_and_the_same_seed_is_repeatable():
    def run(seed):
        v = vibe.Vibes(seed=seed)
        out = []
        for _ in range(12):
            out.append(v.word("generating"))
            v.call()
        return out
    assert run(1) != run(2) and run(5) == run(5)


def test_token_counts_only_ever_move_forward_and_a_new_turn_resets_the_count():
    v = vibe.Vibes(seed=4)
    v.tokens(200)
    w = v.word("generating")
    v.tokens(100)                       # going backwards is ignored
    assert v.word("generating") == w
    v.new_turn()
    v.tokens(10)                        # a new turn counts from zero again, so no jump
    assert v.word("generating") != "" 
