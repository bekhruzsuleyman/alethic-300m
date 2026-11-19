"""Thin wrapper around a SentencePiece model: batch encode, decode and special_token.

It rewrites no text, so the saved .model file alone is enough to tokenize (llama.cpp, Ollama).
EOS is not added here; the data stage appends id 2 after each document.
"""
from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

from alethic.tokenizer.specials import (
    FIRST_SLOT_ID,
    NUM_SLOTS,
    SpecialsRegistry,
    slot_piece,
)

CONTROL_PIECES = {0: "<unk>", 1: "<s>", 2: "</s>", 3: "<pad>"}
UNK_ID, BOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3


class CodecError(RuntimeError):
    """The model file does not have the expected id layout."""


class Codec:
    """Encode and decode text with one frozen SentencePiece model."""

    def __init__(self, model_path: str | Path, specials_path: str | Path | None = None) -> None:
        """Load the model and check its id layout; specials_path enables special_token()."""
        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.specials = SpecialsRegistry(specials_path) if specials_path else None
        self.check_layout()

    @property
    def vocab_size(self) -> int:
        """Number of ids in the model."""
        return self.sp.get_piece_size()

    @property
    def eos_id(self) -> int:
        """End-of-document id (2)."""
        return EOS_ID

    def encode(self, text: str) -> list[int]:
        """Token ids for one string, with no BOS or EOS added."""
        return self.sp.encode(text, out_type=int)

    def encode_batch(self, texts: list[str], num_threads: int = -1) -> list[list[int]]:
        """Token ids for many strings, encoded by SentencePiece's own thread pool."""
        return self.sp.encode(texts, out_type=int, num_threads=num_threads)

    def decode(self, ids: list[int]) -> str:
        """Text for a list of ids (control ids such as EOS decode to nothing)."""
        return self.sp.decode(list(ids))

    def piece(self, token_id: int) -> str:
        """The vocabulary piece for one id, e.g. '<0x0A>' or '▁▁▁▁'."""
        return self.sp.id_to_piece(token_id)

    def is_byte(self, token_id: int) -> bool:
        """True if the id is one of the 256 byte-fallback pieces."""
        return self.sp.is_byte(token_id)

    def special_token(self, name: str) -> int:
        """Id of the reserved slot registered for `name` (a new name takes the next free slot)."""
        if self.specials is None:
            raise CodecError("Codec was created without specials_path; cannot map special names")
        token_id = self.specials.special_token(name)
        expected = slot_piece(token_id - FIRST_SLOT_ID)
        if self.piece(token_id) != expected:
            raise CodecError(f"id {token_id} is {self.piece(token_id)!r}, expected {expected!r}")
        return token_id

    def check_layout(self) -> None:
        """Fail loudly unless ids 0-3 are the control pieces, 4-67 the slots, and bytes exist."""
        if self.vocab_size >= 65536:
            raise CodecError(f"vocab_size {self.vocab_size} does not fit uint16")
        for token_id, piece in CONTROL_PIECES.items():
            if self.piece(token_id) != piece:
                raise CodecError(f"id {token_id} is {self.piece(token_id)!r}, expected {piece!r}")
        for slot in range(NUM_SLOTS):
            token_id = FIRST_SLOT_ID + slot
            if self.piece(token_id) != slot_piece(slot):
                raise CodecError(f"id {token_id} is {self.piece(token_id)!r}, expected {slot_piece(slot)!r}")
        for probe in ("\u2581", "\ue000", "a\u2581 \ue0000"):
            if self.decode(self.encode(probe)) != probe:
                raise CodecError("model lacks the U+2581 escape rules; retrain with u2581_escape: true")
        missing = [b for b in range(256) if not self.is_byte(self.sp.piece_to_id(f"<0x{b:02X}>"))]
        if missing:
            raise CodecError(f"byte-fallback pieces missing for {len(missing)} bytes; train with byte_fallback")
