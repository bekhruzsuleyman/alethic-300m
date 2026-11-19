"""Ask the real Wikipedia search API every question in queries.py and save the answers to a JSON cache.
Run on a computer with internet (the build sandbox has none): double-click deploy\\28_record_wiki_search.bat.
Resumable: a query already in the cache is not asked again. Result text is the API's own intro extract (plain text)."""
import json
import os
import sys
import time

from tools.skilltraces.queries import all_queries

from alethic.agent.websearch import http_fetch, parse_wiki as parse, wiki_url as url_for   # one implementation for harness and recorder

CACHE = os.path.join("docs", "sft", "record", "wiki_search_cache.json")


def record(cache_path=CACHE, fetch=http_fetch, pause=0.25, say=print):
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    todo = [q["query"] for q in all_queries() if q["query"] not in cache]
    say("%d questions to ask, %d already saved" % (len(todo), len(cache)))
    for i, q in enumerate(todo, 1):
        try:
            cache[q] = parse(fetch(url_for(q)))
        except Exception as exc:                                  # network hiccup: keep what we have, next run resumes
            say("stopped at %r: %s" % (q, exc))
            break
        if i % 20 == 0:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            say("%d / %d" % (i, len(todo)))
        time.sleep(pause)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    say("saved %d queries to %s" % (len(cache), cache_path))
    return cache


if __name__ == "__main__":
    record()
    sys.exit(0)
