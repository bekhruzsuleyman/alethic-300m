"""Tokenizer evaluation on held-out documents (M1). Every number in the report is per source.

Gates (plan): round trip 100.00% for every source; Python newline + 4/8/12 spaces encodes to at most
2 tokens in at least 99% of cases. Flag (owner decides): Uzbek tokens per word above 3.0.
Also reported: bytes per token, tokens per word, byte-fallback share, top 30 learned pieces.
"""
from __future__ import annotations

import re
from collections import Counter

from alethic.tokenizer.codec import Codec

FIRST_LEARNED_ID = 324          # 4 control + 64 slots + 256 bytes
INDENT = re.compile(rb"\n( +)(?! )")


def indentation_cases(codec: Codec, texts: list[str], widths=(4, 8, 12), max_tokens: int = 2) -> dict:
    """For every newline followed by exactly 4, 8 or 12 spaces in Python text, count the pure
    indentation tokens inside [start, end) (excluding a trailing word token that absorbs spaces
    along with the line's first code token); return per width cases and share with <= max_tokens."""
    stats = {w: [0, 0] for w in widths}       # width -> [cases, ok]
    for text in texts:
        raw = text.encode("utf-8")
        spans = [(m.start(), m.end(), len(m.group(1))) for m in INDENT.finditer(raw)
                 if len(m.group(1)) in stats]
        if not spans:
            continue
        pieces = codec.sp.encode(text, return_type="proto").pieces     # begin/end = byte offsets
        bounds = [(p.begin, p.end) for p in pieces if p.end > p.begin]
        for start, end, width in spans:
            n = sum(1 for b, e in bounds if b >= start and e <= end)
            stats[width][0] += 1
            stats[width][1] += n <= max_tokens
    return {str(w): {"cases": c, "ok": ok, "share": (ok / c if c else None)} for w, (c, ok) in stats.items()}


def source_stats(codec: Codec, texts: list[str], top_n: int) -> dict:
    """Round trip, bytes/token, tokens/word, byte-fallback share and top learned pieces for one source."""
    ids_all = codec.encode_batch(texts)
    failures = [i for i, (t, ids) in enumerate(zip(texts, ids_all)) if codec.decode(ids) != t]
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    n_words = sum(len(t.split()) for t in texts)
    n_tokens = sum(len(ids) for ids in ids_all)
    counts = Counter(i for ids in ids_all for i in ids)
    n_byte_tokens = sum(c for i, c in counts.items() if 68 <= i < FIRST_LEARNED_ID)
    top = [(codec.piece(i), c) for i, c in counts.most_common() if i >= FIRST_LEARNED_ID][:top_n]
    return {
        "docs": len(texts), "bytes": n_bytes, "words": n_words, "tokens": n_tokens,
        "roundtrip_share": 1.0 - len(failures) / max(1, len(texts)),
        "roundtrip_failures": [texts[i][:200] for i in failures[:3]],
        "bytes_per_token": n_bytes / max(1, n_tokens),
        "tokens_per_word": n_tokens / max(1, n_words),
        "byte_fallback_share": n_byte_tokens / max(1, n_tokens),
        "top_learned_pieces": top,
    }


def evaluate_tokenizer(codec: Codec, heldout: dict[str, list[str]], cfg_eval: dict) -> dict:
    """Run every rule; report["gates"] says whether the freeze may happen."""
    report = {"sources": {s: source_stats(codec, texts, int(cfg_eval["top_pieces"]))
                          for s, texts in sorted(heldout.items())}}
    report["indentation"] = indentation_cases(codec, heldout.get("python", []),
                                              max_tokens=int(cfg_eval["indentation_max_tokens"]))
    limit = float(cfg_eval["uzbek_max_tokens_per_word"])
    uz = {s: report["sources"][s]["tokens_per_word"] for s in ("uzbek_latn", "uzbek_cyrl")
          if s in report["sources"]}
    min_share = float(cfg_eval["indentation_min_share"])
    too_few = [s for s, t in heldout.items() if len(t) < int(cfg_eval["heldout_docs_per_source"])]
    report["gates"] = {
        "roundtrip_100": all(v["roundtrip_share"] == 1.0 for v in report["sources"].values()),
        "indentation": all(v["cases"] > 0 and v["share"] >= min_share
                           for v in report["indentation"].values()),
        "heldout_docs_enough": not too_few,
    }
    report["flags"] = {"uzbek_over_limit": {s: v for s, v in uz.items() if v > limit},
                       "sources_with_too_few_heldout_docs": too_few}
    report["freeze_allowed"] = all(report["gates"].values()) and not report["flags"]["uzbek_over_limit"]
    return report
