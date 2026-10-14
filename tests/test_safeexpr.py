import os
import unittest
from pathlib import Path

from sira import worldlab_lib as wl

NS = wl._num_namespace()


def ev(e, **pt):
    return wl.safe_eval(e, {**NS, **pt})


class SafeEval(unittest.TestCase):
    def test_math(self):
        self.assertAlmostEqual(ev("sin(x)**2 + cos(x)**2", x=0.7), 1.0)
        self.assertEqual(ev("2**10 - 24"), 1000)
        self.assertEqual(ev("sum([1, 2, 3]) + max(1, 5)"), 11)
        self.assertEqual(ev("factorial(5)"), 120)
        self.assertAlmostEqual(ev("-x + +x*pi", x=2.0), 2 * 3.141592653589793 - 2)

    def test_rejects_attacks(self):
        for bad in ["().__class__", "__import__('os')", "[c for c in ().__class__.__bases__]", "(lambda: 1)()", "open('f')", "x.real",
                    "'a'*10", "1 if x else 2", "{1: 2}", "sin(x=1)", "evil(1)", "x @ x", "x < 2", "sin(*[1])", "(x := 3)", "1j", "True"]:
            with self.subTest(bad=bad), self.assertRaises((ValueError, SyntaxError)):
                ev(bad, x=1.0)

    def test_dos_limits(self):
        for bad in ["9**9**9", "2**4097", "factorial(10**7)", "comb(10**6, 5)", "1+" * 3000 + "1", "+".join(["x"] * 500)]:
            with self.subTest(bad=bad), self.assertRaises((ValueError, SyntaxError, RecursionError)):
                ev(bad, x=1.0)

    def test_check_expr_names(self):
        wl.check_expr("x + sin(y)", {"x", "y", "sin"})
        with self.assertRaises(ValueError):
            wl.check_expr("x + z", {"x"})

    def test_no_eval_left(self):
        import ast
        tree = ast.parse(Path(wl.__file__).read_text(encoding="utf-8"))
        bad = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("eval", "exec", "compile")]
        self.assertEqual(bad, [])


class MathVerify(unittest.TestCase):
    def test_chain_ok_and_bad(self):
        self.assertTrue(wl.math_verify(["(x+1)**2", "x**2+2*x+1", "x*x+2*x+1"])["ok"])
        r = wl.math_verify(["(x+1)^2", "x^2+1"])
        self.assertFalse(r["ok"])
        self.assertEqual(r["first_bad"], 1)

    def test_identity_mode(self):
        self.assertTrue(wl.math_verify(["sin(x)**2 + cos(x)**2 = 1"], mode="identity")["ok"])
        self.assertFalse(wl.math_verify(["x**2 = x"], mode="identity")["ok"])

    def test_malicious_steps_are_rejected_not_executed(self):
        marker = "/tmp/sira_pwned_marker"
        if os.path.exists(marker):
            os.remove(marker)
        for evil in ["__import__('os').system('touch %s')" % marker, "().__class__.__bases__", "x.__class__"]:
            r = wl.math_verify(["x", evil])
            self.assertFalse(r["ok"], evil)
            self.assertEqual(r["details"][1].get("method"), "rejected")
        self.assertFalse(os.path.exists(marker))


if __name__ == "__main__":
    unittest.main()
