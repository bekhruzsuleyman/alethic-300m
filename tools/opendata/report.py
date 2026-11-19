"""Measure open datasets before we train on them: license, size, and how many examples fit the 4096-token window.

Run by deploy/22_open_data_report.bat on the user's computer (it needs internet and the Alethic tokenizer).
"""
from __future__ import annotations

import json
import os
import re
import sys

DATA_EXT = (".parquet", ".jsonl", ".json")
TOKENIZER = os.path.join("data", "tokenizer", "alethic-tokenizer-v1", "alethic-tokenizer-v1.model")
MAX_BYTES, SAMPLE_ROWS, CONTEXT = 700_000_000, 1000, 4096

# (dataset id, what it is, optional pattern for the file to sample)
DATASETS = [
    ("HuggingFaceTB/smol-smoltalk", "chat, built for models under 1B", None),
    ("GasaiAI/Gasai-Agent-5k", "agent traces: read/bash/edit/write", None),
    ("domofon/Gasai-Agent-CoderForge", "agent traces: read/bash/edit/write", None),
    ("TIGER-Lab/SWE-Next-SFT-Trajectories", "agent traces: SWE tasks", None),
    ("NousResearch/hermes-function-calling-v1", "JSON function calling", "func-calling"),
    ("Salesforce/xlam-function-calling-60k", "JSON function calling", None),
    ("UAzimov/uzbek-instruct-llm", "Uzbek chat", None),
    ("behbudiy/translation-instruction", "Uzbek/English translation and instructions", None),
    ("allenai/tulu-3-sft-mixture", "multilingual chat mixture", None),
]


def flatten_text(obj) -> str:
    """Every string inside a row (dicts, lists, plain text) joined with newlines. Numbers and None are ignored."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return "\n".join(t for t in (flatten_text(v) for v in obj.values()) if t)
    if isinstance(obj, (list, tuple)):
        return "\n".join(t for t in (flatten_text(v) for v in obj) if t)
    return ""


def summarize(lengths, limit: int = CONTEXT) -> dict:
    n = len(lengths)
    if not n:
        return {"n": 0, "median": 0, "p90": 0, "fit": 0, "fit_pct": 0.0, "fit_2048": 0}
    s = sorted(lengths)
    fit = sum(1 for x in s if x <= limit)
    return {"n": n, "median": s[n // 2], "p90": s[min(n - 1, int(n * 0.9))], "fit": fit,
            "fit_pct": 100.0 * fit / n, "fit_2048": sum(1 for x in s if x <= 2048)}


def pick_file(files: dict, max_bytes: int = MAX_BYTES, prefer: str | None = None):
    """Choose one data file to sample: matching `prefer` if given, then names with 'train', then the smallest."""
    data = {f: s for f, s in files.items() if f.lower().endswith(DATA_EXT) and s <= max_bytes}
    if prefer:
        data = {f: s for f, s in data.items() if re.search(prefer, f)} or data
    train = {f: s for f, s in data.items() if "train" in f.lower()}
    pool = train or data
    return min(pool, key=pool.get) if pool else None


def read_rows(path: str, limit: int = SAMPLE_ROWS):
    low = path.lower()
    if low.endswith(".parquet"):
        import pyarrow.parquet as pq
        rows = []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=200):
            rows += batch.to_pylist()
            if len(rows) >= limit:
                break
        return rows[:limit]
    if low.endswith(".jsonl"):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) >= limit:
                    break
        return rows
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit] if isinstance(data, list) else list(data.values())[:limit]


def report_one(api, hf_download, count_tokens, repo: str, note: str, prefer) -> list:
    lines = ["", "== %s  (%s)" % (repo, note)]
    info = api.dataset_info(repo, files_metadata=True)
    card = getattr(info, "card_data", None)
    lic = (card.get("license") if hasattr(card, "get") else getattr(card, "license", None)) if card else None
    files = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
    lines.append("license: %s   gated: %s   data files: %d" % (lic, getattr(info, "gated", False), sum(1 for f in files if f.lower().endswith(DATA_EXT))))
    name = pick_file(files, prefer=prefer)
    if not name:
        lines.append("no data file small enough to sample (limit %d MB)" % (MAX_BYTES // 1_000_000))
        return lines
    path = hf_download(repo_id=repo, filename=name, repo_type="dataset")
    rows = read_rows(path)
    texts = [flatten_text(r) for r in rows]
    s = summarize([count_tokens(t) for t in texts])
    lines.append("sampled %d rows from %s (%d MB)" % (s["n"], name, files[name] // 1_000_000))
    lines.append("tokens per row: median %d, 90th percentile %d" % (s["median"], s["p90"]))
    lines.append("fit in %d tokens: %d of %d (%.0f%%)   fit in 2048: %d" % (CONTEXT, s["fit"], s["n"], s["fit_pct"], s["fit_2048"]))
    if texts:
        lines.append("first row starts: " + texts[0][:300].replace("\n", " | "))
    return lines


def main() -> int:
    import sentencepiece as spm
    from huggingface_hub import HfApi, hf_hub_download

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER)
    api, out = HfApi(), ["Open dataset report"]
    for repo, note, prefer in DATASETS:
        print("checking", repo, "…", flush=True)
        try:
            out += report_one(api, hf_hub_download, lambda t: len(sp.encode(t)), repo, note, prefer)
        except Exception as exc:
            out += ["", "== %s" % repo, "FAILED: %s: %s" % (type(exc).__name__, str(exc)[:300])]
    text = "\n".join(out)
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "open_data_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text)
    print("\nSaved: reports\\open_data_report.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
