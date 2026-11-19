"""Build samples from terminal skeletons by running every step for real."""
import os
import re

from tools.sftbuild.fileops import FileOpError, apply_edit, numbered, read_lines, write_file
from tools.sftbuild.policy import check
from tools.sftbuild.sandbox import fresh_dir, run
from tools.sftbuild.seq import Seq

ROOT = "/work"  # path shown in recorded results instead of the real working directory
USER_CAUSED_FAIL = {"0621", "0639"}  # the failure comes from the user's request, not the assistant


class SkeletonError(Exception):
    def __init__(self, example_id, cmd, detail):
        super().__init__("ex %s: %s: %s" % (example_id, cmd, detail))


def _is_install(cmd: str) -> bool:
    return "pip install" in cmd or "-m venv" in cmd


def build_terminal(text: str, workdir: str, user_caused=USER_CAUSED_FAIL, searcher=None) -> list:
    out = []
    for b in re.split(r"\n(?=### ex)", text.strip()):
        eid = re.search(r"### ex (\d+)", b).group(1)
        kind = re.search(r"^KIND: (\w+)", b, re.M).group(1)
        try:
            fresh_dir(workdir)
        except ValueError as exc:
            raise SkeletonError(eid, "workdir", str(exc)) from exc
        for m in re.finditer(r"^--- (\S+)\n<<<\n(.*?)\n>>>", b, re.S | re.M):
            path = os.path.join(workdir, m.group(1))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(m.group(2) + "\n")
        b = re.sub(r"^--- \S+\n<<<\n.*?\n>>>\n?", "", b, flags=re.S | re.M)  # file contents are not skeleton lines
        blocks = []

        def grab(m):
            blocks.append(m.group(2))
            return "STEP: write | PATH: %s | BLOCK: %d" % (m.group(1), len(blocks) - 1)

        b = re.sub(r"^STEP: write \| PATH: (\S+)\n<<<\n(.*?)\n>>>", grab, b, flags=re.S | re.M)  # content is never a step
        unique = re.search(r"^UNIQUE: (\S+)$", b, re.M)
        unique = unique.group(1) if unique else None
        user = re.search(r"^USER: (.*)$", b, re.M).group(1)
        plan = re.search(r"^PLAN: (.*)$", b, re.M).group(1)
        final = re.search(r"^FINAL: (.*)$", b, re.M).group(1)
        steps = re.findall(r"^STEP: (shell|read|edit|search|write) \| (.*)$", b, re.M)
        shell_expects = [re.search(r"\| EXPECT: (OK|FAIL)$", rest).group(1) for kind_, rest in steps if kind_ == "shell"]
        own_fail = kind in ("fix", "project_fix") and eid not in user_caused and "FAIL" in shell_expects
        seen_fail = False
        seq = Seq()
        seq.add("<user>\n%s\n</user>\n" % user, 0)
        seq.add("<assistant>\n", 0)
        if plan != "NONE":
            seq.add("<plan> %s </plan>\n" % plan, 0 if own_fail and kind == "fix" else 1)
        for step_kind, rest in steps:
            if step_kind == "shell":
                m = re.match(r"(.*) \| EXPECT: (OK|FAIL)$", rest)
                cmd, expect = m.group(1), m.group(2)
                why = check(cmd)
                if why:
                    raise SkeletonError(eid, cmd, why)
                rc, res = run(cmd, workdir, timeout=60 if _is_install(cmd) else 10, network=_is_install(cmd))
                if (rc == 0) != (expect == "OK"):
                    raise SkeletonError(eid, cmd, "exit %d, expected %s" % (rc, expect))
                res = res.replace(workdir, ROOT)
                masked = own_fail and expect == "FAIL"
                seen_fail = seen_fail or expect == "FAIL"
                seq.add("<call> shell(%s) </call>\n" % cmd, 0 if masked else 1)
                seq.add("<result>\n%s\n</result>\n" % res, 0)
            elif step_kind == "read":
                m = re.match(r"PATH: (\S+) \(lines (\d+)-(\d+)\)$", rest)
                if not m:
                    raise SkeletonError(eid, "read", "bad read step: %r" % rest[:60])
                try:
                    a, z, lines = read_lines(workdir, m.group(1), int(m.group(2)), int(m.group(3)))
                except FileOpError as exc:
                    raise SkeletonError(eid, "read", str(exc)) from exc
                seq.add("<call> read(%s, %s-%s) </call>\n" % (m.group(1), m.group(2), m.group(3)), 1)
                seq.add("<result>\n%s\n</result>\n" % numbered(lines, a), 0)
            elif step_kind == "edit":
                m = re.match(r"PATH: (\S+) \| BUG_LINE: (.*?) \| NEW_LINE: (.*)$", rest)
                if not m:
                    raise SkeletonError(eid, "edit", "bad edit step: %r" % rest[:60])
                try:
                    first, last = apply_edit(workdir, m.group(1), m.group(2), m.group(3))
                except FileOpError as exc:
                    raise SkeletonError(eid, "edit", str(exc)) from exc
                seq.add("<call> edit(%s, %d-%d) </call>\n%s\n" % (m.group(1), first, last, m.group(3)), 1)
            elif step_kind == "write":
                m = re.match(r"PATH: (\S+) \| BLOCK: (\d+)$", rest)
                body = blocks[int(m.group(2))] + "\n"
                try:
                    write_file(workdir, m.group(1), body)
                except FileOpError as exc:
                    raise SkeletonError(eid, "write", str(exc)) from exc
                buggy = own_fail and kind == "project_fix" and not seen_fail  # code written before the first failing run
                seq.add("<call> write(%s) </call>\n%s" % (m.group(1), body), 0 if buggy else 1)
            else:
                m = re.match(r"QUERY: (.*?) \| SOURCE: (\S+) \| EXPECT_IN_RESULT: (.*)$", rest)
                if not m:
                    raise SkeletonError(eid, "search", "bad search step: %r" % rest[:60])
                if searcher is None:
                    raise SkeletonError(eid, "search", "no searcher configured for this build")
                from tools.sftbuild.search import NO_RESULTS, SearchError, fold
                query, source, expect = m.groups()
                try:
                    res = searcher.search(source, query, unique=unique)
                except SearchError as exc:
                    raise SkeletonError(eid, "search", str(exc)) from exc
                if expect.strip() == "NONE":
                    if res != NO_RESULTS:
                        raise SkeletonError(eid, "search", "EXPECT_IN_RESULT: NONE but the search found: %r" % res[:80])
                else:
                    missing = [t.strip() for t in expect.split(",") if fold(t.strip()) not in fold(res)]
                    if missing:
                        raise SkeletonError(eid, "search", "EXPECT_IN_RESULT %s not in the result: %r" % (missing, res[:80]))
                seq.add("<call> search(%s) </call>\n" % query, 1)
                seq.add("<result>\n%s\n</result>\n" % res, 0)
        seq.add(final + "\n", 1)
        seq.add("</assistant>\n", 0)
        out.extend(seq.samples("X" + eid))
    return out
