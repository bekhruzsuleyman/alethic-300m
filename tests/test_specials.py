"""specials.json: stable ids, next free slot, 64-slot limit, append-only, no text injection by name."""
from __future__ import annotations

import json

import pytest

from alethic.tokenizer.specials import (
    INITIAL_NAMES,
    NUM_SLOTS,
    SpecialsError,
    SpecialsRegistry,
    read_names,
)


def test_specials(tiny_codec, tmp_path):
    path = tmp_path / "specials.json"
    codec = tiny_codec

    # The first entries are image = 0, img_start = 1, img_end = 2 (ids 4, 5, 6).
    assert read_names(path) == list(INITIAL_NAMES)
    assert [codec.special_token(n) for n in INITIAL_NAMES] == [4, 5, 6]

    # Same name -> same id, also after reloading from disk.
    assert codec.special_token("image") == 4
    # A new name takes the next free slot and is saved at once.
    assert codec.special_token("thinking") == 7
    assert codec.special_token("thinking") == 7
    assert SpecialsRegistry(path).special_token("thinking") == 7
    assert read_names(path)[-1] == "thinking"

    # Text containing <thinking> stays ordinary text.
    text = "please <thinking> about it </thinking>"
    ids = codec.encode(text)
    assert 7 not in ids
    assert not any(4 <= i < 4 + NUM_SLOTS for i in ids)
    assert codec.decode(ids) == text

    # Fill every slot; the 65th name raises.
    for i in range(len(read_names(path)), NUM_SLOTS):
        assert codec.special_token(f"name{i}") == 4 + i
    with pytest.raises(SpecialsError, match="all 64"):
        codec.special_token("one_too_many")
    assert len(read_names(path)) == NUM_SLOTS

    # Renaming or removing an entry fails.
    saved = read_names(path)       # e.g. the copy kept next to a checkpoint
    renamed = saved.copy()
    renamed[3] = "thoughts"
    path.write_text(json.dumps({"format": 1, "names": renamed}), encoding="utf-8")
    with pytest.raises(SpecialsError, match="slot 3"):
        SpecialsRegistry(path, reference=saved)
    removed = saved[:3] + saved[4:]
    path.write_text(json.dumps({"format": 1, "names": removed}), encoding="utf-8")
    with pytest.raises(SpecialsError):
        SpecialsRegistry(path, reference=saved)
    path.write_text(json.dumps({"format": 1, "names": ["picture"] + saved[1:]}), encoding="utf-8")
    with pytest.raises(SpecialsError, match="must start with"):
        SpecialsRegistry(path)


def test_specials_edit_under_running_registry(tmp_path):
    """A registry refuses to save if the file was renamed under it."""
    path = tmp_path / "specials.json"
    reg = SpecialsRegistry(path)
    reg.special_token("thinking")
    names = read_names(path)
    names[3] = "renamed"
    path.write_text(json.dumps({"format": 1, "names": names}), encoding="utf-8")
    with pytest.raises(SpecialsError):
        reg.special_token("another")
