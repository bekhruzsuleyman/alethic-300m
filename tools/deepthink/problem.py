"""Problem schema and verifier dispatch for deep-thinking data."""
import json
from dataclasses import asdict, dataclass

DOMAINS = ("math", "code", "poly", "design")

_VERIFIERS = {}


@dataclass(frozen=True)
class Problem:
    id: str
    domain: str
    statement: str
    verifier: str
    spec: dict
    source: str
    license: str

    def __post_init__(self):
        if self.domain not in DOMAINS:
            raise ValueError("bad domain: %r" % (self.domain,))


def register(name, fn):
    """fn(problem, answer, workdir) -> (bool, message)"""
    _VERIFIERS[name] = fn


def verify(problem, answer, workdir):
    fn = _VERIFIERS.get(problem.verifier)
    if fn is None:
        raise ValueError("unknown verifier: %r" % (problem.verifier,))
    return fn(problem, answer, workdir)


def _exact(problem, answer, workdir):
    ok = answer.strip() == str(problem.spec["answer"]).strip()
    return ok, "" if ok else "answer differs"


register("exact", _exact)


def dump_jsonl(path, problems):
    with open(path, "w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps(asdict(p), ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Problem(**json.loads(line)))
    return out
