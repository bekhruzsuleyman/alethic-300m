"""Text in, text out: tokenizes the history, asks a decoder for next-token logits, stops on a stop rule."""
from __future__ import annotations

import torch

EOS_ID = 2
LOOP = 16  # the same LOOP tokens twice in a row = the model is stuck repeating itself


def _looping(ids: list) -> bool:
    return len(ids) >= 2 * LOOP and ids[-LOOP:] == ids[-2 * LOOP:-LOOP]


class Engine:
    def __init__(self, decoder, sp, max_seq_len, temperature=0.0, top_k=40):
        self.decoder, self.sp, self.max_seq_len = decoder, sp, max_seq_len
        self.temperature, self.top_k = temperature, top_k
        self.last_tokens = 0

    def _pick(self, logits: torch.Tensor) -> int:
        if self.temperature <= 0:
            return int(logits.argmax())
        top = torch.topk(logits, min(self.top_k, logits.numel()))
        probs = torch.softmax(top.values / self.temperature, dim=-1)
        return int(top.indices[torch.multinomial(probs, 1)])

    @torch.inference_mode()
    def generate(self, history: str, stop, max_new: int, on_text) -> str:
        """Generate after `history` until EOS, stop(text), max_new tokens, a loop or a full context. Returns the text."""
        ids = self.sp.encode(history)
        start, shown, self.last_tokens = len(ids), "", 0
        for _ in range(max_new):
            if len(ids) >= self.max_seq_len:
                break
            nxt = self._pick(self.decoder.logits(ids))
            if nxt == EOS_ID:
                break
            ids.append(nxt)
            self.last_tokens += 1
            text = self.sp.decode(ids[start:])
            if text.endswith("�"):  # half of a multi-byte character: wait for the rest
                continue
            if text[len(shown):]:
                on_text(text[len(shown):])
            shown = text
            if stop(shown) or _looping(ids[start:]):
                break
        return shown
