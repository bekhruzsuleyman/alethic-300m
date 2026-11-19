"""Turn open chat rows ({"messages": [{"role", "content"}]}) into Alethic segments. Stage A of the SFT plan.

Only the assistant's words are trained (loss 1). A row is skipped if the roles do not alternate, it does not end with
the assistant, it is longer than the context, or its text contains one of our own tags (it would confuse the harness).
"""
from __future__ import annotations

import json
import os
import random
import sys

CONTEXT = 4096
TAGS = ("<user>", "</user>", "<assistant>", "</assistant>", "<call>", "</call>", "<result>", "</result>", "<plan>", "<map>")
TOKENIZER = os.path.join("data", "tokenizer", "alethic-tokenizer-v1", "alethic-tokenizer-v1.model")


def convert_row(row: dict, count, limit: int = CONTEXT):
    msgs = [m for m in (row.get("messages") or []) if m.get("role") != "system"]
    if not msgs or len(msgs) % 2 or msgs[-1].get("role") != "assistant":
        return None
    segs = []
    for i, m in enumerate(msgs):
        want = "user" if i % 2 == 0 else "assistant"
        text = (m.get("content") or "").strip()
        if m.get("role") != want or not text or any(t in text for t in TAGS):
            return None
        if want == "user":
            segs += [{"t": "<user>\n%s\n</user>\n" % text, "loss": 0}, {"t": "<assistant>\n", "loss": 0}]
        else:
            segs += [{"t": text + "\n", "loss": 1}, {"t": "</assistant>\n", "loss": 0}]
    ex = {"segments": segs}
    return ex if count(to_text(ex)) <= limit else None


def to_text(ex: dict) -> str:
    return "".join(s["t"] for s in ex["segments"])


def main() -> int:
    """Download the Smol-SmolTalk train shards, keep a random ~100k that fit, write reports\\stage_a_chat.jsonl."""
    import pyarrow.parquet as pq
    import sentencepiece as spm
    from huggingface_hub import HfApi, hf_hub_download

    target = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER)
    count = lambda t: len(sp.encode(t))
    repo = "HuggingFaceTB/smol-smoltalk"
    shards = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith("data/train-") and f.endswith(".parquet"))
    rng, kept, seen, skipped = random.Random(5), 0, 0, 0
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "stage_a_chat.jsonl"), "w", encoding="utf-8") as out:
        for name in shards:
            print("shard", name, "…", flush=True)
            for batch in pq.ParquetFile(hf_hub_download(repo, name, repo_type="dataset")).iter_batches(batch_size=500):
                for r in batch.to_pylist():
                    seen += 1
                    if rng.random() > 0.3 or kept >= target:      # about every third row, until we have enough
                        continue
                    ex = convert_row(r, count)
                    if ex is None:
                        skipped += 1
                        continue
                    ex["id"] = "S%06d" % kept
                    out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    kept += 1
            print("  kept %d so far (seen %d, skipped %d)" % (kept, seen, skipped), flush=True)
            if kept >= target:
                break
    print("\nSaved %d examples to reports\\stage_a_chat.jsonl" % kept)
    return 0


if __name__ == "__main__":
    sys.exit(main())
