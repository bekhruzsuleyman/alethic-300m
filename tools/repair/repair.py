"""Apply a cleanup plan to the checked batch files: remove, relabel, patch, rewrite, and re-run hand-written
tool results for real. Output is the tag format (### ex ... / <user> / <assistant>) the plain builder reads."""
import dataclasses
import os
import re
import shutil
import tempfile

from tools.repair.replay import replay
from tools.sftbuild.normalize import parse_examples

THINKING = re.compile(r"<thinking>.*?</thinking>\n?", re.S)


class RepairError(ValueError):
    pass


def to_instant(e):
    if not THINKING.search(e.assistant):
        raise RepairError("ex %s has no <thinking> block to drop" % e.id)
    return dataclasses.replace(e, type="A", assistant=THINKING.sub("", e.assistant).strip("\n"))


def _patch(e, patches):
    for field, old, new in patches:
        value = getattr(e, field)
        if value.count(old) != 1:
            raise RepairError("ex %s: patch text %r matches %d times in %s" % (e.id, old, value.count(old), field))
        e = dataclasses.replace(e, **{field: value.replace(old, new)})
    return e


def _render(e):
    return "### ex %s | type: %s | lang: %s | source: %s\n<user>\n%s\n</user>\n<assistant>\n%s\n</assistant>\n" % (
        e.id, e.type, e.lang, e.source, e.user, e.assistant)


def repair(text, plan, workroot):
    examples = parse_examples(text)
    have = {e.id for e in examples}
    for key in ("remove", "instant"):
        missing = set(plan.get(key, ())) - have
        if missing:
            raise RepairError("%s lists ids that are not in the file: %s" % (key, sorted(missing)))
    report = {"removed": [], "instant": [], "patched": [], "rewritten": [], "replayed": [], "changed_results": []}
    out = []
    for e in examples:
        if e.id in plan.get("remove", ()):
            report["removed"].append(e.id)
            continue
        if e.id in plan.get("instant", ()):
            e = to_instant(e)
            report["instant"].append(e.id)
        if e.id in plan.get("rewrite", {}):
            user, asst = plan["rewrite"][e.id]
            e = dataclasses.replace(e, user=user, assistant=asst)
            report["rewritten"].append(e.id)
        if e.id in plan.get("patch", {}):
            e = _patch(e, plan["patch"][e.id])
            report["patched"].append(e.id)
        if e.id in plan.get("fixtures", {}):
            wd = tempfile.mkdtemp(prefix="replay_", dir=workroot)
            try:
                new = replay(e.assistant, plan["fixtures"][e.id], wd)
            finally:
                shutil.rmtree(wd, ignore_errors=True)
            if new != e.assistant:
                report["changed_results"].append(e.id)
            e = dataclasses.replace(e, assistant=new, source="replay")
            report["replayed"].append(e.id)
        out.append(_render(e))
    return "\n".join(out), report
