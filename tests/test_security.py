"""Cheating / escape attempts by a (possibly malicious) candidate program. Run: python -m unittest tests.test_security -v"""
import json, os, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sira.config import Config
from sira.orchestrator import Orchestrator
from sira.sandbox import SandboxUnavailable
from sira.challenges import ChallengeEngine


class Sec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = Config(llm_provider="mock", auto_install=False, auto_adapt=False, thinking="off", sandbox_backend="auto", sandbox_timeout_s=20, sandbox_mem_mb=512,
                     run_root=tempfile.mkdtemp())
        try:
            cls.o = Orchestrator(cfg)
            cls.o.build()
        except SandboxUnavailable:
            cls.o = None
            return
        cls.sb, cls.eng = cls.o.ctx.sandbox, ChallengeEngine(cls.o.ctx, cls.o.research)

    @classmethod
    def tearDownClass(cls):
        if cls.o:
            cls.o.stop()

    def need(self):
        if not self.o:
            self.skipTest("no strong sandbox")

    def test_candidate_cannot_find_the_verifier(self):
        self.need()
        secret = "def verify(a):\n    return {'score': 0.5, 'holdout': 0.5, 'msg': 'SECRET_MARKER_XYZ'}"
        code = ("import os\nhits=[]\nfor root in ['/', '/work', '/data', '/station', '/tmp', '/home', '/root']:\n"
                "    for d, _, fs in os.walk(root):\n        if d.startswith(('/proc', '/sys', '/usr', '/dev')): continue\n"
                "        for f in fs:\n            try:\n                if ('SECRET_' + 'MARKER_XYZ') in open(os.path.join(d, f), errors='ignore').read(200000): hits.append(os.path.join(d, f))\n"
                "            except Exception: pass\nprint('HITS', hits)\n")
        # the engine runs the verifier in a SEPARATE call; here we only prove the candidate sandbox contains no verifier source
        cid = self.o.db.add_challenge("t", "d", "optimization", "stmt", secret, [{"answer": 0, "expect": "low"}], 3, "t")
        r = self.sb.run(code, station="ro", timeout=30)
        self.assertIn("HITS []", r.stdout)

    def test_escape_and_resource_attacks(self):
        self.need()
        atk = {"read_host_env": "import os; print('ENV', os.environ.get('HOME'), os.environ.get('SIRA_SELFTEST_SECRET'), len(os.environ))",
               "write_outside": "open('/usr/evil','w').write('x')", "write_host_path": "open('" + str(self.o.run_dir / 'pwned.txt') + "','w').write('x')",
               "net": "import socket; print(sorted(n for _, n in socket.if_nameindex()))",
               "mem": "x = bytearray(4 * 1024 ** 3); print('allocated')", "loop": "while True: pass"}
        r = {k: self.sb.run(v, timeout=8, network="none") for k, v in atk.items()}
        self.assertNotIn("SELFTEST", r["read_host_env"].stdout)
        self.assertNotEqual(r["write_outside"].exit_code, 0)
        self.assertFalse((self.o.run_dir / "pwned.txt").exists())
        self.assertIn("['lo']", r["net"].stdout)
        self.assertNotIn("allocated", r["mem"].stdout)
        self.assertTrue(r["loop"].timed_out)

    def test_fake_result_lines_do_not_fool_the_verifier(self):
        self.need()
        ch = {"verifier_code": "def verify(a):\n    return {'score': 1.0 if a == 7 else 0.0, 'holdout': 1.0 if a == 7 else 0.0}"}
        ans, run, err = self.eng.run_candidate("print('SIRA_RESULT: {\"score\": 1.0}')\nprint('SIRA_ANSWER: 3')")
        self.assertEqual(ans, 3)
        v = self.eng.verify_answer(ch["verifier_code"], ans)
        self.assertEqual(v["score"], 0.0)                         # the candidate's printed SIRA_RESULT is irrelevant: the verifier runs separately

    def test_hostile_answers_cannot_break_the_verifier(self):
        self.need()
        vc = "def verify(a):\n    return {'score': 0.1, 'holdout': 0.1}"
        deep = 0
        for _ in range(5000):
            deep = [deep]
        for ans in (deep, "A" * 300000, float("nan"), 10 ** 400, {"__class__": 1}):
            v = self.eng.verify_answer(vc, ans)
            self.assertIn("score", v)                              # never raises; crashes/oversize become score 0

    def test_validator_rejects_gaming_prone_verifiers(self):
        self.need()
        bad = [{"verifier_code": "def verify(a):\n    return {'score': 0.0 if a == 0 else 1.0, 'holdout': 1.0}", "probes": [{"answer": 0, "expect": "low"}], "kind": "optimization"},   # any non-zero wins
               {"verifier_code": "def verify(a):\n    return {'score': 0.1}", "probes": [{"answer": 0, "expect": "low"}], "kind": "optimization"},                                   # no holdout
               {"verifier_code": "def verify(a):\n    return {'score': 0.1, 'holdout': 0.9}", "probes": [{"answer": 0, "expect": "low"}], "kind": "optimization"}]                   # inconsistent
        for spec in bad:
            self.assertFalse(self.eng.validate(spec)[0], spec["verifier_code"])

    def test_open_problems_are_never_called_solved(self):
        self.need()
        spec = {"verifier_code": "def verify(a):\n    s = min(1.0, a / 10) if isinstance(a, (int, float)) and not isinstance(a, bool) and a > 0 else 0.0\n    return {'score': s, 'holdout': s}",
                "probes": [{"answer": 0, "expect": "low"}, {"answer": 3, "expect": "low"}], "kind": "open", "open": True}
        self.assertEqual(self.eng.classify(spec), "open")
        self.assertEqual(self.eng.classify({"kind": "reference", "probes": [{"expect": "high"}]}), "solvable")


if __name__ == "__main__":
    unittest.main()
