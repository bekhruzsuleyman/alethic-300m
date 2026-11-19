from alethic.agent import brand, highlight as hl


def styles(tokens):
    return {t: s for t, s in tokens if t.strip()}


def test_text_is_never_changed_or_lost():
    for lang, line in [("python", 'def add(a, b):  # sum "x"'), ("rust", 'fn main() { println!("hi {}", 1); }'),
                       ("js", "const x = `a${b}` + 'c'; // note"), ("html", '<a href="x">hi</a>'),
                       ("json", '{"a": [1, 2.5, true, null]}'), ("bash", 'echo "$HOME" | grep -n foo > out.txt')]:
        h = hl.Highlighter(lang)
        assert "".join(t for t, _ in h.line(line)) == line


def test_python_keywords_names_strings_numbers_and_comments_get_colors():
    s = styles(hl.Highlighter("python").line('def add(a, b):  # sum\n'.rstrip("\n")))
    assert s["def"] == brand.KEYWORD
    assert s["add"] == brand.FUNC
    assert s["# sum"] == brand.COMMENT
    s = styles(hl.Highlighter("python").line('x = 42 + len("hi")'))
    assert s["42"] == brand.NUMBER and s["len"] == brand.BUILTIN and s['"hi"'] == brand.STRING


def test_python_triple_quoted_string_runs_over_lines():
    h = hl.Highlighter("python")
    a = h.line('"""start')
    b = h.line("still a string  # not a comment")
    c = h.line('end""" + x')
    assert a[0][1] == brand.STRING
    assert all(s == brand.STRING for _, s in b)
    assert c[0][1] == brand.STRING and ("x", "") in [(t, s) for t, s in c if t.strip() == "x"]


def test_rust_keywords_macros_types_and_lifetimes():
    s = styles(hl.Highlighter("rust").line("fn main() { let v: Vec<i32> = vec![1]; println!(\"x\"); }"))
    assert s["fn"] == brand.KEYWORD and s["let"] == brand.KEYWORD
    assert s["main"] == brand.FUNC
    assert s["Vec"] == brand.BUILTIN and s["i32"] == brand.BUILTIN
    assert s["println!"] == brand.BUILTIN and s["vec!"] == brand.BUILTIN
    assert hl.Highlighter("rust").line("fn f<'a>(x: &'a str) {}")[0][0] == "fn"
    assert styles(hl.Highlighter("rust").line("fn f<'a>() {}"))["'a"] == brand.DECORATOR


def test_block_comment_state_carries_over_in_rust_and_js():
    for lang in ("rust", "js"):
        h = hl.Highlighter(lang)
        h.line("/* open")
        mid = h.line("still comment")
        assert mid and mid[0][1] == brand.COMMENT
        after = h.line("close */ let x = 1;")
        assert after[0][1] == brand.COMMENT


def test_json_keys_and_values():
    s = styles(hl.Highlighter("json").line('{"name": "a", "n": 3, "ok": true}'))
    assert s['"name"'] == brand.FUNC and s['"a"'] == brand.STRING
    assert s["3"] == brand.NUMBER and s["true"] == brand.KEYWORD


def test_html_tags_attributes_and_strings():
    s = styles(hl.Highlighter("html").line('<a href="x">hi</a>'))
    assert s["<a"] == brand.KEYWORD and s['"x"'] == brand.STRING and s["href"] == brand.BUILTIN


def test_shell_command_flags_variables_comments():
    s = styles(hl.Highlighter("bash").line('grep -n "$HOME" file.txt # find'))
    assert s["grep"] == brand.FUNC and s["-n"] == brand.BUILTIN
    assert s["# find"] == brand.COMMENT


def test_unknown_language_is_plain_and_aliases_work():
    assert hl.Highlighter("klingon").line("x = 1") == [("x = 1", "")]
    assert hl.normalize("py") == "python" and hl.normalize("RS") == "rust" and hl.normalize("ts") == "js"
    assert hl.normalize("sh") == "bash" and hl.normalize("") == ""


def test_language_from_a_file_name():
    assert hl.lang_for_path("main.rs") == "rust" and hl.lang_for_path("a/b/calc.py") == "python"
    assert hl.lang_for_path("index.HTML") == "html" and hl.lang_for_path("notes.txt") == ""


def test_highlight_whole_text_gives_one_token_list_per_line():
    rows = hl.highlight("a = 1\nb = 2\n", "python")
    assert len(rows) == 2 and rows[1][0][0] == "b"


def test_a_very_long_line_does_not_hang():
    import time
    t = time.time()
    hl.Highlighter("python").line("x = '" + "a" * 20000)
    assert time.time() - t < 1.0


def test_shell_pipe_starts_a_new_command():
    s = styles(hl.Highlighter("bash").line("cat a.txt | sort -r"))
    assert s["cat"] == brand.FUNC and s["sort"] == brand.FUNC and s["a.txt"] == ""
