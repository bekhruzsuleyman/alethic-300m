import os
import unittest

from tools.deepthink import mathgen
from tools.deepthink.problem import Problem, verify

TRAIN_SEEDS = range(50)


def wrong_forms(ans):
    return ["(%s)+1" % ans, "2*(%s)" % ans, "(%s)-1" % ans]


class MathGenTests(unittest.TestCase):
    def test_generate_is_deterministic_per_seed(self):
        for fam in mathgen.FAMILIES:
            a, b = mathgen.generate(fam, 7), mathgen.generate(fam, 7)
            self.assertEqual(a, b)
            self.assertNotEqual(a.statement, mathgen.generate(fam, 8).statement)

    def test_true_answer_verifies_for_every_family(self):
        for fam in mathgen.FAMILIES:
            for s in TRAIN_SEEDS:
                p = mathgen.generate(fam, s)
                self.assertEqual(p.domain, "math")
                ok, msg = verify(p, p.spec["answer"], "")
                self.assertTrue(ok, (fam, s, p.spec["answer"], msg))

    def test_wrong_answers_fail_for_every_family(self):
        for fam in mathgen.FAMILIES:
            for s in TRAIN_SEEDS:
                p = mathgen.generate(fam, s)
                self.assertNotEqual(p.spec["answer"].strip(), "0", (fam, s))
                for w in wrong_forms(p.spec["answer"]):
                    self.assertFalse(verify(p, w, "")[0], (fam, s, w))

    def test_equal_by_simplify_passes(self):
        p = Problem(id="m", domain="math", statement="s", verifier="sympy",
                    spec={"answer": "2*(x+1)"}, source="t", license="own")
        self.assertTrue(verify(p, "2*x+2", "")[0])
        self.assertFalse(verify(p, "2*x+3", "")[0])

    def test_malicious_answer_is_rejected_not_executed(self):
        marker = "/tmp/deepthink_pwned_marker"
        if os.path.exists(marker):
            os.remove(marker)
        p = Problem(id="m", domain="math", statement="s", verifier="sympy",
                    spec={"answer": "4"}, source="t", license="own")
        evil = "__import__('os').system('touch %s')" % marker
        self.assertFalse(verify(p, evil, "")[0])
        self.assertFalse(os.path.exists(marker))
        self.assertFalse(verify(p, "open('/etc/passwd')", "")[0])

    def test_garbage_answer_is_false_not_crash(self):
        p = mathgen.generate("algebra", 1)
        for junk in ["", "I think it is big", "((", "1/0"]:
            self.assertFalse(verify(p, junk, "")[0])

    def test_holdout_seeds_are_disjoint_and_marked(self):
        self.assertTrue(mathgen.is_holdout(mathgen.generate("algebra", 900001)))
        self.assertFalse(mathgen.is_holdout(mathgen.generate("algebra", 5)))
        with self.assertRaises(ValueError):
            mathgen.train_seeds(900001)

    def test_holdout_statements_not_in_train_sample(self):
        for fam in mathgen.FAMILIES:
            train = {mathgen.generate(fam, s).statement for s in range(300)}
            hold = {mathgen.generate(fam, s).statement for s in range(900000, 900060)}
            self.assertFalse(train & hold, fam)

    def test_every_problem_needs_several_steps(self):
        for fam in mathgen.FAMILIES:
            p = mathgen.generate(fam, 3)
            self.assertGreaterEqual(p.spec["steps"], 3)
            self.assertLessEqual(p.spec["steps"], 8)


class CountingFormulaTests(unittest.TestCase):
    def test_arrangements_match_bruteforce(self):
        import itertools
        for word in ["BANANA", "PEPPER", "ASSESS", "LETTER"]:
            perms = set(itertools.permutations(word))
            for f in set(word):
                for l in set(word):
                    brute = sum(1 for q in perms if q[0] == f and q[-1] == l)
                    self.assertEqual(mathgen.arrangements(word, f, l), brute,
                                     (word, f, l))


if __name__ == "__main__":
    unittest.main()
