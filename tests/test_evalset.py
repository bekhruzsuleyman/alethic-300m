import subprocess
import sys

from tools.evalset.prompts import eval_set


def test_exactly_fifty_unique_prompts_with_unique_ids():
    s = eval_set()
    assert len(s) == 50 and len({e["id"] for e in s}) == 50 and len({e["prompt"] for e in s}) == 50


def test_eval_prompts_are_not_in_the_training_traces():
    from tools.errtraces.build import build_all, to_text
    train = "".join(to_text(e) for e in build_all(per_kind=60, seed=1))
    for e in eval_set():
        assert "<user>\n%s\n</user>" % e["prompt"] not in train


def test_every_tool_expectation_is_what_a_correct_solution_prints(tmp_path):
    """Run the reference answer for tasks that have an expected output: the file after the right fix must print it."""
    fixes = {"Hi Ali5": ('print("Hi " + name + 5)', 'print("Hi " + name + "5")'), "0": ("print(sum(xs) / len(xs))", "print(sum(xs) / len(xs) if xs else 0)"),
             "Plan": ('TITLE = "Draft"', 'TITLE = "Plan"'), "0\n1\n2": ("LIMIT = 9", "LIMIT = 3"), "start\ndone": ('print("start")', 'print("start")\nprint("done")')}
    for e in eval_set():
        if e["kind"] != "tool" or not e["expect"] or not e["files"]:
            continue
        d = tmp_path / e["id"]
        d.mkdir()
        for name, text in e["files"].items():
            if e["expect"] in fixes and fixes[e["expect"]][0] in text:
                text = text.replace(*fixes[e["expect"]])
            (d / name).write_text(text)
        out = subprocess.run([sys.executable, e["run"]], cwd=d, capture_output=True, text=True).stdout.strip()
        assert out == e["expect"].strip(), (e["id"], out)
