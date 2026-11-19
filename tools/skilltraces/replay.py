"""Replay a training file and check that every <result> and every <map> is what the real tools produce.

    python -m tools.skilltraces.replay FILE.jsonl [--search-cache docs/sft/record/wiki_search_cache.json] [--limit N]

Each record may carry "setup": {"file name": "starting text"}: the files that existed before the first call. Every
<call> is run in a fresh scratch folder by the real Workspace (python3), then compared with the <result> that follows;
the <map> after a read of a .py file must equal the real ast outline. search() uses the cache when it has the query,
else the real Wikipedia API (needs internet). Exit code 1 and a list of the first problems when anything differs."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile

from alethic.agent.protocol import Call
from alethic.agent.tools import Workspace
from alethic.agent import websearch
from tools.sftbuild.mapper import ast_map

CALL = re.compile(r"<call> (\w+)\((.*?)\) </call>\n(.*)\Z", re.S)


def _fetcher(cache):
    def fetch(url):
        q = re.search(r"gsrsearch=([^&]+)", url).group(1)
        import urllib.parse
        q = urllib.parse.unquote_plus(q)
        if q in cache:
            rows = cache[q]
            return json.dumps({"query": {"pages": {str(i): {"title": r["title"], "index": i, "extract": r["snippet"]} for i, r in enumerate(rows, 1)}}}) if rows else '{"batchcomplete": ""}'
        return websearch.http_fetch(url)
    return fetch


def check(example, cache=None, search_only=False):
    """-> list of problems (empty when the example replays exactly)."""
    problems = []
    root = tempfile.mkdtemp(prefix="replay")
    try:
        for name, text in (example.get("setup") or {}).items():
            os.makedirs(os.path.dirname(os.path.join(root, name)) or root, exist_ok=True)
            with open(os.path.join(root, name), "w", encoding="utf-8") as f:
                f.write(text)
        ws = Workspace(root, mode="posix", python="python3", timeout=10, search_fetch=_fetcher(cache or {}))
        segs, last_read = example["segments"], None
        for i, s in enumerate(segs):
            t = s["t"]
            m = CALL.match(t) if t.startswith("<call> ") else None
            if m:
                kind, args, body = m.group(1), m.group(2), m.group(3)
                if search_only and kind != "search":
                    continue
                out = ws.execute(Call(kind, args, body if kind in ("write", "edit") else None)).result
                nxt = segs[i + 1]["t"] if i + 1 < len(segs) else ""
                want = "<result>\n%s\n</result>\n" % out if out is not None else None
                if want is None and nxt.startswith("<result>"):
                    problems.append("%s: call %s(%s) gives no result but the file has one" % (example["id"], kind, args))
                elif want is not None and nxt != want:
                    problems.append("%s: result differs for %s(%s)\n   file: %r\n   real: %r" % (example["id"], kind, args, nxt[:200], want[:200]))
                if kind == "read":
                    mm = re.match(r"(.+?),\s*(\d+)-(\d+)$", args.strip())
                    last_read = (mm.group(1).strip(), int(mm.group(3))) if mm else None
            elif not search_only and t.startswith("<map>") and last_read and last_read[0].endswith(".py"):
                path = os.path.join(root, last_read[0])
                total = len(open(path, encoding="utf-8").read().splitlines())
                real = ast_map(path, min(last_read[1], total))
                if real is None or t.strip() != "<map> %s </map>" % real:
                    problems.append("%s: map differs\n   file: %r\n   real: %r" % (example["id"], t.strip()[:200], real))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return problems


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--search-cache")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--search-only", action="store_true", help="check only the search results (works on Windows, no bash needed)")
    a = ap.parse_args(argv)
    cache = json.load(open(a.search_cache, encoding="utf-8")) if a.search_cache and os.path.exists(a.search_cache) else {}
    bad, n = [], 0
    for line in open(a.file, encoding="utf-8"):
        if a.limit and n >= a.limit:
            break
        n += 1
        bad += check(json.loads(line), cache, a.search_only)
    print("checked %d examples, %d problems" % (n, len(bad)))
    for p in bad[:20]:
        print(p)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
