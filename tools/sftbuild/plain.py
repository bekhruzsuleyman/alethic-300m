"""Plain question/answer examples (controls, mixed, uncertain, batches) as samples."""
import re

from tools.sftbuild.normalize import parse_examples
from tools.sftbuild.seq import Seq

# harness-written blocks, in the multi-line form or the inline form the old batches use
RESULT = re.compile(r"(<(?:result|error|warn)>.*?</(?:result|error|warn)>\n?)", re.S)


def build_plain(text: str) -> list:
    out = []
    for ex in parse_examples(text):
        seq = Seq()
        seq.add("<user>\n%s\n</user>\n" % ex.user, 0)
        seq.add("<assistant>\n", 0)
        flags = []
        # later turns are written inside the assistant text: split them out so user text and tags get no loss
        turns = re.split(r"\n</assistant>\n<user>\n(.*?)\n</user>\n<assistant>\n", ex.assistant, flags=re.S)
        chunks = [(turns[0], None)] + [(turns[i + 1], turns[i]) for i in range(1, len(turns) - 1, 2)]
        for chunk, next_user in chunks:
            if next_user is not None:
                seq.add("</assistant>\n<user>\n%s\n</user>\n<assistant>\n" % next_user, 0)
            _add_assistant(seq, chunk, ex, flags)
        seq.add("</assistant>\n", 0)
        for s in seq.samples("P" + ex.id):
            s["flags"] = flags
            out.append(s)
    return out


def _add_assistant(seq, text, ex, flags):
    for part in RESULT.split(text + "\n"):
        if not part:
            continue
        if RESULT.fullmatch(part):  # a result written by hand, not a real run
            flags[:] = [] if ex.source == "replay" else ["hand_result"]
            seq.add(part if part.endswith("\n") else part + "\n", 0)
        elif part.strip():
            seq.add(part.lstrip("\n") if part.endswith("\n") else part + "\n", 1)
