import os
import tempfile
import unittest

from tools.deepthink.problem import Problem, dump_jsonl, load_jsonl, verify


def mk(verifier="exact", spec=None):
    return Problem(id="p1", domain="math", statement="2+2?", verifier=verifier,
                   spec=spec if spec is not None else {"answer": "4"},
                   source="gen", license="own")


class ProblemTests(unittest.TestCase):
    def test_exact_answer_verifier_accepts_true_rejects_false(self):
        p = mk()
        self.assertTrue(verify(p, "4", "")[0])
        self.assertTrue(verify(p, "  4 \n", "")[0])
        self.assertFalse(verify(p, "5", "")[0])

    def test_unknown_verifier_raises(self):
        with self.assertRaises(ValueError):
            verify(mk(verifier="nope"), "4", "")

    def test_roundtrip_jsonl_is_lossless(self):
        ps = [mk(), mk(spec={"answer": "x", "nested": [1, 2]})]
        d = tempfile.mkdtemp()
        path = os.path.join(d, "p.jsonl")
        dump_jsonl(path, ps)
        self.assertEqual(load_jsonl(path), ps)

    def test_bad_domain_rejected(self):
        with self.assertRaises(ValueError):
            Problem(id="x", domain="cooking", statement="s", verifier="exact",
                    spec={}, source="s", license="own")


if __name__ == "__main__":
    unittest.main()
