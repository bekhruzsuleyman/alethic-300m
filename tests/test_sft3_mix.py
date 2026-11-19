import json

from tools.sft3.mix import build_stage_b, report, write_mix


def put(path, ids, tag="x"):
    path.write_text("".join(json.dumps({"id": i, "segments": [{"t": "<user>\n%s\n</user>\n" % tag, "loss": 0}]}) + "\n" for i in ids))


def setup(tmp_path):
    b = tmp_path / "built"
    b.mkdir()
    put(b / "maps_F.jsonl", ["F1.0", "F1.1", "F2.0"])
    put(b / "plain_a.jsonl", ["P1.0"])
    put(b / "plain_batch1_0001-0100_codefixed.jsonl", ["DUP1.0"])
    put(b / "identity_en_uz_built.jsonl", ["I9.0"])
    put(b / "identity_variants.jsonl", ["I1", "I2"])
    put(b / "error_traces.jsonl", ["E1", "E2", "E3"])
    replay = tmp_path / "a.jsonl"
    put(replay, ["S%d" % i for i in range(50)])
    return b, replay


def test_gold_is_repeated_with_the_same_ids_so_a_copy_never_lands_in_validation(tmp_path):
    b, replay = setup(tmp_path)
    mix = build_stage_b(b, replay, gold_repeat=3, replay_n=10, seed=1)
    ids = [s["id"] for s in mix["gold"]]
    assert len(ids) == 4 * 3 and ids.count("F1.0") == 3 and "DUP1.0" not in ids


def test_identity_errors_and_replay_are_in_their_own_parts(tmp_path):
    b, replay = setup(tmp_path)
    mix = build_stage_b(b, replay, gold_repeat=2, replay_n=10, seed=1)
    assert [s["id"] for s in mix["identity"]] == ["I9.0", "I1", "I2"]
    assert len(mix["errors"]) == 3 and len(mix["replay"]) == 10
    assert len({s["id"] for s in mix["replay"]}) == 10


def test_replay_is_reproducible_and_capped_by_what_exists(tmp_path):
    b, replay = setup(tmp_path)
    a = build_stage_b(b, replay, replay_n=10, seed=1)["replay"]
    assert a == build_stage_b(b, replay, replay_n=10, seed=1)["replay"] and a != build_stage_b(b, replay, replay_n=10, seed=2)["replay"]
    assert len(build_stage_b(b, replay, replay_n=999, seed=1)["replay"]) == 50


def test_write_and_report(tmp_path):
    b, replay = setup(tmp_path)
    mix = build_stage_b(b, replay, gold_repeat=3, replay_n=10, seed=1)
    write_mix(mix, tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["errors.jsonl", "gold.jsonl", "identity.jsonl", "replay.jsonl"]
    r = report(mix)
    assert r["total"] == 12 + 3 + 3 + 10 and abs(sum(r["share"].values()) - 1) < 1e-9


def test_lines_with_unicode_line_separators_inside_text_do_not_break_json(tmp_path):
    """Real chat text can hold U+2028, U+0085 or \\x1c: str.splitlines() would cut such a line in half."""
    b, _ = setup(tmp_path)
    bad = tmp_path / "chat.jsonl"
    rows = [{"id": "S%d" % i, "segments": [{"t": "a b\x85c\x1cd\x0be\x0cf %d" % i, "loss": 1}]} for i in range(5)]
    bad.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    mix = build_stage_b(b, bad, replay_n=5, seed=1)
    assert len(mix["replay"]) == 5 and all(" " in s["segments"][0]["t"] for s in mix["replay"])


def test_take_first_copies_only_the_first_n_whole_lines(tmp_path):
    from tools.sft3.mix import take_first
    src, dst = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    src.write_text("".join(json.dumps({"id": i, "t": "x y"}, ensure_ascii=False) + "\n" for i in range(10)), encoding="utf-8")
    assert take_first(src, dst, 4) == 4
    assert [json.loads(l)["id"] for l in dst.read_text(encoding="utf-8").split("\n") if l] == [0, 1, 2, 3]
    assert take_first(src, dst, 99) == 10


def test_program_traces_are_their_own_part_used_once_and_never_counted_as_gold(tmp_path):
    b, replay = setup(tmp_path)
    put(b / "prog_traces.jsonl", ["G%d" % i for i in range(5)])
    mix = build_stage_b(b, replay, gold_repeat=3, replay_n=10, seed=1)
    assert [s["id"] for s in mix["progs"]] == ["G0", "G1", "G2", "G3", "G4"]
    assert not any(s["id"].startswith("G") for s in mix["gold"]) and len(mix["gold"]) == 12
    write_mix(mix, tmp_path / "out")
    assert (tmp_path / "out" / "progs.jsonl").exists()
    assert report(mix)["total"] == 12 + 3 + 3 + 10 + 5


def test_without_a_program_traces_file_there_is_no_progs_part(tmp_path):
    b, replay = setup(tmp_path)
    assert "progs" not in build_stage_b(b, replay, replay_n=10, seed=1)


def test_terminal_traces_are_their_own_part_used_once_and_never_counted_as_gold(tmp_path):
    b, replay = setup(tmp_path)
    put(b / "term_traces.jsonl", ["T%d" % i for i in range(4)])
    mix = build_stage_b(b, replay, gold_repeat=3, replay_n=10, seed=1)
    assert [s["id"] for s in mix["terms"]] == ["T0", "T1", "T2", "T3"]
    assert not any(s["id"].startswith("T") for s in mix["gold"]) and len(mix["gold"]) == 12
    write_mix(mix, tmp_path / "out")
    assert (tmp_path / "out" / "terms.jsonl").exists()
    assert report(mix)["total"] == 12 + 3 + 3 + 10 + 4


def test_without_a_terminal_traces_file_there_is_no_terms_part(tmp_path):
    b, replay = setup(tmp_path)
    assert "terms" not in build_stage_b(b, replay, replay_n=10, seed=1)


def test_math_traces_are_their_own_part_used_once_and_never_counted_as_gold(tmp_path):
    b, replay = setup(tmp_path)
    put(b / "math_traces.jsonl", ["M%d" % i for i in range(6)])
    mix = build_stage_b(b, replay, gold_repeat=3, replay_n=10, seed=1)
    assert [s["id"] for s in mix["maths"]] == ["M0", "M1", "M2", "M3", "M4", "M5"]
    assert not any(s["id"].startswith("M") for s in mix["gold"]) and len(mix["gold"]) == 12
    write_mix(mix, tmp_path / "out")
    assert (tmp_path / "out" / "maths.jsonl").exists()
    assert report(mix)["total"] == 12 + 3 + 3 + 10 + 6


def test_without_a_math_traces_file_there_is_no_maths_part(tmp_path):
    b, replay = setup(tmp_path)
    assert "maths" not in build_stage_b(b, replay, replay_n=10, seed=1)


def test_skill_traces_are_their_own_part_used_once_and_never_counted_as_gold(tmp_path):
    b, replay = setup(tmp_path)
    put(b / "skill_traces.jsonl", ["SK%d" % i for i in range(5)])
    mix = build_stage_b(b, replay, gold_repeat=3, replay_n=10, seed=1)
    assert [s["id"] for s in mix["skills"]] == ["SK0", "SK1", "SK2", "SK3", "SK4"]
    assert not any(s["id"].startswith("SK") for s in mix["gold"])
    write_mix(mix, tmp_path / "out")
    assert (tmp_path / "out" / "skills.jsonl").exists()


def test_gemini_sessions_are_a_part_used_twice_and_never_counted_as_gold(tmp_path):
    b, replay = setup(tmp_path)
    put(b / "gemini_sessions.jsonl", ["G%d" % i for i in range(4)])
    mix = build_stage_b(b, replay, gold_repeat=3, replay_n=10, seed=1)
    assert sorted(s["id"] for s in mix["sessions"]) == ["G0", "G0", "G1", "G1", "G2", "G2", "G3", "G3"]
    assert not any(s["id"].startswith("G") for s in mix["gold"])
