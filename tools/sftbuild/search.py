"""The `search` tool of the harness, for building samples.

Sources: `docs.python.org` (Sphinx search index of the Python docs, ranked here) and `any` (exact-title lookup of
the invented UNIQUE code on en.wikipedia.org, which must find nothing). Results are recorded to a JSON cache so a
build is reproducible and replayable without a network. Fetching is injected, so tests never touch the network.
"""
import json
import math
import os
import re
import unicodedata
import urllib.parse
import urllib.request
from html.parser import HTMLParser

import snowballstemmer

DOCS_ROOT = "https://docs.python.org/3/"
WIKI_API = "https://en.wikipedia.org/w/api.php?action=query&format=json&titles="
NO_RESULTS = "no results"
STOP = {"python", "has", "no", "attribute", "module", "error", "object", "the", "a", "an", "of", "in", "to", "is", "not",
        "name", "unexpected", "keyword", "argument", "cannot", "import", "and", "for", "with", "how", "what", "from"}
_STEM = snowballstemmer.stemmer("english").stemWord


class SearchError(Exception):
    pass


class SearchMiss(SearchError):
    pass


def fold(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s.casefold()) if not unicodedata.combining(c))


def parse_searchindex(text):
    m = re.search(r"setIndex\((.*)\)\s*;?\s*$", text.strip(), re.S)
    return json.loads(m.group(1) if m else text)


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip, self.parts = 0, []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def strip_html(html):
    p = _Text()
    p.feed(html)
    return re.sub(r"\s+", " ", "".join(p.parts)).strip()


def _docs_of(entry):
    return entry if isinstance(entry, list) else [entry]


def _rank(index, query, k=3):
    words = [w for w in re.findall(r"\w+", query.lower()) if w not in STOP]
    scores = {}
    for w in words:
        stem = _STEM(w)
        docs = set(_docs_of(index["terms"].get(stem, [])))
        titles = set(_docs_of(index.get("titleterms", {}).get(stem, [])))
        weight = 1.0 / math.log(2 + len(docs | titles))
        for d in docs:
            scores[d] = scores.get(d, 0) + weight
        for d in titles:
            scores[d] = scores.get(d, 0) + 3 * weight
    need = max(1, math.ceil(len(words) / 2))
    hit_counts = {d: sum(1 for w in words if d in set(_docs_of(index["terms"].get(_STEM(w), []))) |
                         set(_docs_of(index.get("titleterms", {}).get(_STEM(w), [])))) for d in scores}
    ranked = sorted((d for d in scores if hit_counts[d] >= need), key=lambda d: (-hit_counts[d], -scores[d], d))
    return ranked[:k], words


def _snippet(text, words, index, width=240):
    rare = sorted(words, key=lambda w: len(set(_docs_of(index["terms"].get(_STEM(w), [])))))
    low = text.lower()
    pos = next((low.find(w) for w in rare if low.find(w) >= 0), 0)
    start = max(0, pos - width // 3)
    return text[start:start + width].strip()


def http_fetch(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "alethic-sftbuild/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


class Searcher:
    def __init__(self, fetch=http_fetch, cache_path=None, mode="live"):
        if mode not in ("live", "record", "replay"):
            raise ValueError("mode must be live, record or replay")
        self.fetch, self.cache_path, self.mode = fetch, cache_path, mode
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            self.cache = json.load(open(cache_path))
        self._index = None

    def search(self, source, query, unique=None):
        key = "%s|%s|%s" % (source, query, unique or "")
        if self.mode == "replay":
            if key not in self.cache:
                raise SearchMiss("no recorded result for %s" % key)
            return self.cache[key]
        if source == "docs.python.org":
            out = self._docs(query)
        elif source == "any":
            out = self._any(unique)
        else:
            raise SearchError("source %r is not on the allowlist" % source)
        if self.mode == "record" and self.cache_path:
            self.cache[key] = out
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=1, sort_keys=True)
        return out

    def _docs(self, query):
        if self._index is None:
            self._index = parse_searchindex(self.fetch(DOCS_ROOT + "searchindex.js"))
        ranked, words = _rank(self._index, query)
        lines = []
        for n, d in enumerate(ranked, 1):
            url = DOCS_ROOT + self._index["docnames"][d] + ".html"
            text = strip_html(self.fetch(url))
            lines.append("%d. %s (%s)\n   %s" % (n, self._index["titles"][d], url.replace("https://", ""),
                                                 _snippet(text, words, self._index)))
        return "\n".join(lines) if lines else NO_RESULTS

    def _any(self, unique):
        if not unique:
            raise SearchError("source 'any' needs the unique code of the skeleton")
        data = json.loads(self.fetch(WIKI_API + urllib.parse.quote(unique)))
        if any("missing" not in p for p in data["query"]["pages"].values()):
            raise SearchError("the unique code %r exists on the web; choose another" % unique)
        return NO_RESULTS
