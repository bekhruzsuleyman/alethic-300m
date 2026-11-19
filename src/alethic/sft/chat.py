"""Chat with an SFT checkpoint in the console: python -m alethic.sft.chat --weights model.pt --tokenizer x.model"""
from __future__ import annotations

import argparse
import sys

import sentencepiece as spm
import torch

from alethic.config import load_config
from alethic.model.transformer import ModelConfig, Transformer
from alethic.sft.data import CLOSE_TAG, EOS_ID


def load(weights: str, model_cfg: str, device: str = "cpu") -> Transformer:
    """Build the model and load the saved best-epoch weights."""
    model = Transformer(ModelConfig.from_dict(load_config(model_cfg))).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True)["model"])
    return model.eval()


@torch.no_grad()
def reply(model, sp, history: str, device: str, max_new: int, out=sys.stdout) -> str:
    """Greedy answer to the conversation so far; prints text as it is produced; returns the answer text."""
    ids = sp.encode(history + "<assistant>\n")
    start, shown = len(ids), ""
    for _ in range(max_new):
        if len(ids) >= model.cfg.max_seq_len:
            break
        nxt = int(model(torch.tensor([ids], device=device))[0, -1].argmax())
        if nxt == EOS_ID:
            break
        ids.append(nxt)
        text = sp.decode(ids[start:])
        out.write(text[len(shown):])
        out.flush()
        shown = text
        if text.rstrip().endswith(CLOSE_TAG):
            break
    out.write("\n")
    return shown


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model", default="configs/model_gqa.yaml")
    p.add_argument("--max-new", type=int, default=200)
    a = p.parse_args(argv)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    model = load(a.weights, a.model)
    print("Alethic SFT-1. Type a message and press Enter. Empty line = new conversation. Ctrl+C = quit.\n")
    history = ""
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            history = ""
            print("(new conversation)")
            continue
        history += f"<user>\n{msg}\n</user>\n"
        print("bot> ", end="")
        ans = reply(model, sp, history, "cpu", a.max_new)
        history += f"<assistant>\n{ans.strip()}\n" if ans.rstrip().endswith(CLOSE_TAG) else f"<assistant>\n{ans}\n</assistant>\n"


if __name__ == "__main__":
    main()
