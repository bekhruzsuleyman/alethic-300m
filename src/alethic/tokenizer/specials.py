"""Reserved special slots <sp00>..<sp63> and the append-only name registry, specials.json.

The tokenizer has 64 reserved ids right after the 4 control ids (unk 0, bos 1, eos 2, pad 3),
so slot s has id 4 + s. A name such as "image" is bound to a slot the first time
special_token() is asked for it, and that binding never changes. Plain text such as
"<thinking>" is never turned into a special id; only a call to special_token() produces one.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from alethic.config import write_json_atomic

NUM_CONTROL = 4                     # unk 0, bos 1, eos 2, pad 3
NUM_SLOTS = 64                      # <sp00> .. <sp63>
FIRST_SLOT_ID = NUM_CONTROL         # slot s has id 4 + s
INITIAL_NAMES = ("image", "img_start", "img_end")   # first entries, used only in M8
FORMAT_VERSION = 1


class SpecialsError(RuntimeError):
    """specials.json was changed in a forbidden way, or all 64 slots are used."""


def slot_piece(slot: int) -> str:
    """Return the text of a slot's piece, e.g. slot 5 -> '<sp05>'."""
    if not 0 <= slot < NUM_SLOTS:
        raise SpecialsError(f"slot {slot} is outside 0..{NUM_SLOTS - 1}")
    return f"<sp{slot:02d}>"


def slot_id(slot: int) -> int:
    """Return the token id of a slot (4 + slot)."""
    slot_piece(slot)  # range check
    return FIRST_SLOT_ID + slot


def all_slot_pieces() -> list[str]:
    """The 64 slot pieces in id order; they are the tokenizer's user_defined_symbols."""
    return [slot_piece(s) for s in range(NUM_SLOTS)]


def validate_names(names: list[str]) -> None:
    """Check a names list: unique non-empty strings, at most 64, starting with the initial names."""
    if len(names) > NUM_SLOTS:
        raise SpecialsError(f"{len(names)} names but only {NUM_SLOTS} slots exist")
    if any(not isinstance(n, str) or not n for n in names):
        raise SpecialsError("every special name must be a non-empty string")
    if len(set(names)) != len(names):
        raise SpecialsError("special names must be unique")
    if tuple(names[: len(INITIAL_NAMES)]) != INITIAL_NAMES:
        raise SpecialsError(
            f"specials.json must start with {list(INITIAL_NAMES)}; got {names[:len(INITIAL_NAMES)]}"
        )


def check_append_only(old: list[str], new: list[str]) -> None:
    """Fail unless `new` keeps every entry of `old` in the same slot (only appending is allowed)."""
    for slot, name in enumerate(old):
        if slot >= len(new):
            raise SpecialsError(f"entry {name!r} (slot {slot}) was removed; specials.json is append-only")
        if new[slot] != name:
            raise SpecialsError(
                f"slot {slot} was {name!r} and is now {new[slot]!r}; names never change slot "
                "and are never renamed or removed"
            )


def read_names(path: str | Path) -> list[str]:
    """Read the names list from a specials.json file and validate it."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != FORMAT_VERSION or not isinstance(data.get("names"), list):
        raise SpecialsError(f"{path} is not a specials.json file (format {FORMAT_VERSION})")
    names = list(data["names"])
    validate_names(names)
    return names


class SpecialsRegistry:
    """The name -> slot table stored in specials.json. Slot = position in the names list."""

    def __init__(self, path: str | Path, reference: list[str] | None = None) -> None:
        """Load specials.json (creating it with the initial names if missing).

        `reference` is an older copy of the names (e.g. the one saved next to a checkpoint);
        loading fails if the file on disk is not an append-only extension of it.
        """
        self.path = Path(path)
        if self.path.exists():
            self._names = read_names(self.path)
        else:
            self._names = list(INITIAL_NAMES)
            self._save()
        if reference is not None:
            check_append_only(reference, self._names)

    @property
    def names(self) -> tuple[str, ...]:
        """All registered names in slot order."""
        return tuple(self._names)

    def special_token(self, name: str) -> int:
        """Return the id for `name`; a new name takes the next free slot and is saved at once."""
        if name in self._names:
            return slot_id(self._names.index(name))
        if len(self._names) >= NUM_SLOTS:
            raise SpecialsError(
                f"all {NUM_SLOTS} special slots are used; cannot register {name!r}"
            )
        new = self._names + [name]
        validate_names(new)
        on_disk = read_names(self.path) if self.path.exists() else []
        check_append_only(on_disk, new)  # refuse if someone edited the file under us
        self._names = new
        self._save()
        return slot_id(len(new) - 1)

    def sha256(self) -> str:
        """SHA-256 of the file on disk; it goes into manifest.json next to each checkpoint."""
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def _save(self) -> None:
        """Write the names atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.path, {"format": FORMAT_VERSION, "names": self._names})


def special_token(name: str, path: str | Path) -> int:
    """Shortcut: id of `name` in the registry stored at `path` (registers new names)."""
    return SpecialsRegistry(path).special_token(name)
