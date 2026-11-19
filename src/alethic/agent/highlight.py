"""Syntax colors for code, with no outside library (fast, and it ships inside the exe).

Highlighter(lang).line(text) returns [(text, color)] for one line and remembers things that run over several lines
(triple-quoted strings, block comments). The text is never changed: the pieces always join back to the line.
"""
from __future__ import annotations

import re

from alethic.agent import brand as b

ALIASES = {"py": "python", "python3": "python", "python": "python", "rs": "rust", "rust": "rust",
           "js": "js", "javascript": "js", "ts": "js", "typescript": "js", "jsx": "js", "tsx": "js", "mjs": "js",
           "html": "html", "htm": "html", "xml": "html", "svg": "html", "css": "css", "json": "json",
           "sh": "bash", "bash": "bash", "shell": "bash", "zsh": "bash", "console": "bash", "terminal": "bash",
           "powershell": "bash", "ps1": "bash", "bat": "bash", "cmd": "bash",
           "toml": "config", "yaml": "config", "yml": "config", "ini": "config", "cfg": "config"}
EXT = {".py": "python", ".rs": "rust", ".js": "js", ".ts": "js", ".jsx": "js", ".tsx": "js", ".mjs": "js",
       ".html": "html", ".htm": "html", ".xml": "html", ".svg": "html", ".css": "css", ".json": "json",
       ".sh": "bash", ".bat": "bash", ".cmd": "bash", ".ps1": "bash", ".toml": "config", ".yaml": "config",
       ".yml": "config", ".ini": "config", ".cfg": "config"}

KEYWORDS = {
    "python": set("and as assert async await break class continue def del elif else except finally for from global if import in is lambda "
                  "nonlocal not or pass raise return try while with yield True False None match case".split()),
    "rust": set("as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod move mut pub ref "
                "return self Self static struct super trait true type unsafe use where while".split()),
    "js": set("async await break case catch class const continue default delete do else export extends false finally for from function if "
              "import in instanceof let new null of return static super switch this throw true try typeof undefined var void while yield "
              "interface type enum implements".split()),
    "bash": set("if then else elif fi for while do done case esac in function return exit export set unset local source".split()),
    "json": {"true", "false", "null"},
    "config": {"true", "false", "null", "yes", "no"},
}
BUILTINS = {
    "python": set("print len range int str float list dict set tuple open sum min max abs enumerate zip map filter sorted isinstance type bool "
                  "input format repr round any all super self cls reversed iter next object Exception ValueError TypeError KeyError".split()),
    "rust": set("i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64 bool char str String Vec Option Result Box Rc Arc HashMap "
                "HashSet Some None Ok Err".split()),
    "js": set("console Math JSON Object Array String Number Boolean Promise Map Set Date document window require module".split()),
}

# every rule is (group name, regex). Order matters: the first rule that matches at a position wins.
STR_DQ = r'"(?:[^"\\]|\\.)*(?:"|$)'
STR_SQ = r"'(?:[^'\\]|\\.)*(?:'|$)"
NUM = r"\b(?:0[xX][0-9a-fA-F_]+|\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?)\b"
WORD = r"[A-Za-z_][A-Za-z0-9_]*"

RULES = {
    "python": [("comment", r"#.*"), ("tri", r"[rRbBfFuU]{0,2}(?:\"\"\"|''')"),
               ("str", r"[rRbBfFuU]{0,2}(?:" + STR_DQ + "|" + STR_SQ + ")"), ("deco", r"@" + WORD + r"(?:\.\w+)*"),
               ("num", NUM), ("word", WORD)],
    "rust": [("comment", r"//.*"), ("block", r"/\*"), ("str", STR_DQ), ("life", r"'" + WORD + r"(?!')"),
             ("chr", r"'(?:\\.|[^\\'])'"), ("macro", WORD + r"!(?=[\(\[\{])"), ("num", NUM), ("word", WORD)],
    "js": [("comment", r"//.*"), ("block", r"/\*"), ("str", STR_DQ + "|" + STR_SQ + r"|`(?:[^`\\]|\\.)*(?:`|$)"),
           ("num", NUM), ("word", WORD)],
    "json": [("key", r'"(?:[^"\\]|\\.)*"(?=\s*:)'), ("str", STR_DQ), ("num", r"-?\d[\d.eE+-]*"), ("word", WORD)],
    "html": [("comment", r"<!--.*?(?:-->|$)"), ("tag", r"</?[A-Za-z][\w:-]*|/?>"), ("str", STR_DQ + "|" + STR_SQ),
             ("attr", r"[A-Za-z_:][\w:.-]*(?==)"), ("ent", r"&\w+;")],
    "css": [("block", r"/\*"), ("str", STR_DQ + "|" + STR_SQ), ("at", r"@[\w-]+"), ("hex", r"#[0-9a-fA-F]{3,8}\b"),
            ("sel", r"[.#][A-Za-z_][\w-]*"), ("prop", r"[a-z-]+(?=\s*:)"), ("num", r"-?\d+(?:\.\d+)?(?:px|em|rem|%|vh|vw|s|ms)?")],
    "bash": [("comment", r"(?:(?<=\s)|^)#.*"), ("str", STR_DQ + "|" + STR_SQ), ("var", r"\$\{[^}]*\}|\$[A-Za-z_]\w*|%[A-Za-z_]\w*%"),
             ("flag", r"(?<![\w-])-{1,2}[A-Za-z][\w-]*"), ("op", r"\|\||&&|[|;<>&]+"), ("num", NUM), ("word", r"[A-Za-z_./~][\w./~:-]*")],
    "config": [("comment", r"#.*"), ("str", STR_DQ + "|" + STR_SQ), ("key", r"^\s*[\w.-]+(?=\s*[:=])"), ("num", NUM), ("word", WORD)],
}
COMPILED = {lang: re.compile("|".join("(?P<%s>%s)" % (n, rx) for n, rx in rules)) for lang, rules in RULES.items()}
BLOCK_END = {"python": None, "rust": "*/", "js": "*/", "css": "*/"}


def normalize(lang: str) -> str:
    return ALIASES.get((lang or "").strip().lower(), "")


def lang_for_path(path: str) -> str:
    dot = path.rfind(".")
    return EXT.get(path[dot:].lower(), "") if dot >= 0 else ""


class Highlighter:
    def __init__(self, lang: str):
        self.lang = normalize(lang) or (lang if lang in RULES else "")
        self.state = None        # ("tri", quote) or ("block", end) while a construct runs over lines
        self.expect_name = False  # the next word is the name of a def/fn/class
        self.expect_cmd = True    # shell: the next word is a command

    def line(self, text: str) -> list:
        if self.lang not in COMPILED:
            return [(text, "")]
        out, pos = [], 0
        if self.state:
            kind, closer = self.state
            i = text.find(closer)
            style = b.STRING if kind == "tri" else b.COMMENT
            if i < 0:
                return [(text, style)] if text else []
            out.append((text[:i + len(closer)], style))
            pos, self.state = i + len(closer), None
        rx = COMPILED[self.lang]
        self.expect_cmd = True
        for m in rx.finditer(text, pos):
            if m.start() > pos:
                out.append((text[pos:m.start()], ""))
                pos = m.start()
            kind, tok = m.lastgroup, m.group()
            style = self._style(kind, tok, m)
            if kind == "tri":
                q = tok[-3:]
                j = text.find(q, m.end())
                end = len(text) if j < 0 else j + 3
                if j < 0:
                    self.state = ("tri", q)
                out.append((text[m.start():end], b.STRING))
                pos = end
                return out + self._rest(text, pos)
            if kind == "block":
                j = text.find("*/", m.end())
                end = len(text) if j < 0 else j + 2
                if j < 0:
                    self.state = ("block", "*/")
                out.append((text[m.start():end], b.COMMENT))
                return out + self._rest(text, end)
            out.append((tok, style))
            pos = m.end()
        if pos < len(text):
            out.append((text[pos:], ""))
        return out

    def _rest(self, text: str, pos: int) -> list:
        """Highlight the part of the line after a multi-line construct ended on it."""
        if pos >= len(text) or self.state:
            return []
        saved = self.state
        out = Highlighter(self.lang).line(text[pos:])
        self.state = saved
        return out

    def _style(self, kind: str, tok: str, m) -> str:
        lang = self.lang
        if kind in ("comment", "block"):
            return b.COMMENT
        if kind in ("str", "chr", "tri"):
            return b.STRING
        if kind in ("num", "hex", "ent"):
            return b.NUMBER
        if kind in ("deco", "life", "var"):
            return b.DECORATOR
        if kind in ("macro", "attr", "prop", "flag"):
            return b.BUILTIN
        if kind in ("key", "sel"):
            return b.FUNC
        if kind in ("tag", "at", "op"):
            if kind == "op":
                self.expect_cmd = True   # after a pipe or && the next word is a command again
            return b.KEYWORD
        if kind == "word":
            if lang == "bash":
                if self.expect_cmd and tok not in KEYWORDS["bash"]:
                    self.expect_cmd = False
                    return b.FUNC
                self.expect_cmd = False
                return b.KEYWORD if tok in KEYWORDS["bash"] else ""
            if self.expect_name:
                self.expect_name = False
                return b.FUNC
            if tok in ("def", "class", "fn", "function", "struct", "enum", "trait") and lang in ("python", "rust", "js"):
                self.expect_name = True
                return b.KEYWORD
            if tok in KEYWORDS.get(lang, ()):
                return b.KEYWORD
            if tok in BUILTINS.get(lang, ()):
                return b.BUILTIN
        return ""


def highlight(code: str, lang: str) -> list:
    """Whole text at once: one token list per line."""
    h = Highlighter(lang)
    lines = code.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [h.line(ln) for ln in lines]
