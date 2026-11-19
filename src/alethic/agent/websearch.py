"""The `search(content, source)` tool: ask a real search source and show the top results as numbered lines.

Sources: wikipedia (the Wikipedia search API, intro extract of each hit). Others are not connected yet and say so.
The network call is injected, so tests and the training-data builder never need a connection."""
import json
import urllib.parse
import urllib.request

SOURCES = ("wikipedia",)
KNOWN_NOT_CONNECTED = ("python", "web", "brave")
DEFAULT_SOURCE = "wikipedia"
NO_RESULTS = "no results"
API = "https://en.wikipedia.org/w/api.php?"
AGENT = "AlethicAgent/0.1 (https://bekhruz.org; local research tool)"


def split_args(args: str):
    """'capital of France, wikipedia' -> ('capital of France', 'wikipedia'); no known source word at the end -> default source."""
    text = args.strip()
    head, sep, tail = text.rpartition(",")
    if sep and tail.strip().lower() in SOURCES + KNOWN_NOT_CONNECTED:
        return head.strip().strip("\"'"), tail.strip().lower()
    return text.strip("\"'"), DEFAULT_SOURCE


def wiki_url(query, limit=3):
    return API + urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search", "gsrsearch": query, "gsrlimit": limit,
        "prop": "extracts", "exintro": 1, "explaintext": 1, "exchars": 260, "exlimit": limit, "redirects": 1})


def parse_wiki(text):
    pages = json.loads(text).get("query", {}).get("pages", {})
    out = []
    for p in sorted(pages.values(), key=lambda p: p.get("index", 99)):
        snip = " ".join((p.get("extract") or "").split())
        if snip:
            out.append({"title": p["title"], "url": "en.wikipedia.org/wiki/" + p["title"].replace(" ", "_"), "snippet": snip})
    return out


def format_results(rows):
    return "\n".join("%d. %s (%s)\n   %s" % (i, r["title"], r["url"], r["snippet"]) for i, r in enumerate(rows, 1)) if rows else NO_RESULTS


def http_fetch(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def run_search(args, fetch=http_fetch):
    """-> result text. Raises ValueError (shown to the model as 'error: ...') for a bad source or a network failure."""
    query, source = split_args(args)
    if not query:
        raise ValueError("search needs words to look for, like search(capital of France, wikipedia)")
    if source in KNOWN_NOT_CONNECTED:
        raise ValueError("source %r is not connected yet. Use: %s" % (source, ", ".join(SOURCES)))
    try:
        return format_results(parse_wiki(fetch(wiki_url(query))))
    except (OSError, ValueError) as exc:
        raise ValueError("could not reach %s: %s" % (source, exc))
