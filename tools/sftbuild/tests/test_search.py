import json

import pytest
import snowballstemmer

from tools.sftbuild.search import SearchError, SearchMiss, Searcher, fold, parse_searchindex, strip_html

STEM = snowballstemmer.stemmer("english").stemWord


def index():
    t = lambda *ws: {STEM(w): d for w, d in ws}
    return {
        "docnames": ["library/unittest", "whatsnew/3.13"],
        "titles": ["unittest - Unit testing framework", "What's New In Python 3.13"],
        "terms": {STEM("unittest"): [0, 1], STEM("makesuite"): 1, STEM("removed"): [1], STEM("testcase"): 0},
        "titleterms": {STEM("unittest"): 0},
    }


PAGES = {
    "https://docs.python.org/3/searchindex.js": "Search.setIndex(%s)" % json.dumps(index()),
    "https://docs.python.org/3/whatsnew/3.13.html":
        "<html><body><script>var x=1</script><p>Some intro text. unittest: the makeSuite function was removed; "
        "use loadTestsFromTestCase instead. More text follows here.</p></body></html>",
    "https://docs.python.org/3/library/unittest.html": "<p>unittest is a framework. TestCase is the base class.</p>",
}


def fake_fetch(url):
    if url in PAGES:
        return PAGES[url]
    if url.startswith("https://en.wikipedia.org/w/api.php") and "KRT-9183" in url:
        return json.dumps({"query": {"pages": {"-1": {"title": "KRT-9183", "missing": ""}}}})
    if url.startswith("https://en.wikipedia.org/w/api.php") and "Python" in url:
        return json.dumps({"query": {"pages": {"23862": {"title": "Python"}}}})
    raise AssertionError("unexpected url " + url)


def test_fold_removes_accents_and_case():
    assert fold("É") == "e"
    assert fold("Ünïcode") == "unicode"


def test_parse_searchindex_reads_the_js_wrapper():
    assert parse_searchindex(PAGES["https://docs.python.org/3/searchindex.js"])["docnames"][1] == "whatsnew/3.13"


def test_strip_html_drops_scripts_and_tags():
    assert strip_html("<p>a <b>b</b></p><script>bad()</script><style>x{}</style>") == "a b"


def test_docs_search_ranks_the_rare_term_first_and_cuts_a_snippet():
    s = Searcher(fake_fetch)
    out = s.search("docs.python.org", "python unittest has no attribute makeSuite")
    assert out.startswith("1. What's New In Python 3.13")
    assert "whatsnew/3.13.html" in out
    assert "loadTestsFromTestCase" in out
    assert len(out.split("\n")[1]) <= 3 + 260


def test_docs_search_with_no_match_says_no_results():
    s = Searcher(fake_fetch)
    assert s.search("docs.python.org", "python zzzzqq nothingmatches") == "no results"


def test_any_source_does_an_exact_title_lookup_of_the_unique_code():
    s = Searcher(fake_fetch)
    assert s.search("any", "handshake KRT-9183 rejected", unique="KRT-9183") == "no results"


def test_any_source_fails_if_the_code_exists_on_the_web():
    s = Searcher(fake_fetch)
    with pytest.raises(SearchError, match="exists"):
        s.search("any", "Python", unique="Python")


def test_any_source_needs_the_unique_code():
    with pytest.raises(SearchError, match="unique"):
        Searcher(fake_fetch).search("any", "x y z w")


def test_unknown_source_is_refused():
    with pytest.raises(SearchError, match="source"):
        Searcher(fake_fetch).search("example.com", "a b c d")


def test_record_then_replay_without_network(tmp_path):
    cache = str(tmp_path / "c.json")
    rec = Searcher(fake_fetch, cache_path=cache, mode="record")
    first = rec.search("docs.python.org", "python unittest has no attribute makeSuite")

    def no_network(url):
        raise AssertionError("replay must not fetch")

    again = Searcher(no_network, cache_path=cache, mode="replay")
    assert again.search("docs.python.org", "python unittest has no attribute makeSuite") == first


def test_replay_miss_raises(tmp_path):
    s = Searcher(fake_fetch, cache_path=str(tmp_path / "c.json"), mode="replay")
    with pytest.raises(SearchMiss):
        s.search("docs.python.org", "python unittest has no attribute makeSuite")


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        Searcher(fake_fetch, mode="nope")
