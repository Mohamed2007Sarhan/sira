"""Run:  python -m unittest tests.test_core -v   (needs bwrap or docker for the sandbox tests)"""
import hashlib, json, os, sys, tempfile, threading, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sira.config import Config
from sira.db import DB
from sira.bus import EventBus
from sira.control import RunControl, StopReason, Stopped
from sira.evalcore import EvalCore, extract_answer
from sira.llm import extract_json
from sira.sandbox import make_sandbox, SandboxUnavailable
from sira.verifier import check_expect, validate_expect
from sira.web import WebTools, WebError


def mk(cfg=None):
    d = tempfile.mkdtemp()
    cfg = cfg or Config(sandbox_timeout_s=30, sandbox_mem_mb=1024, sandbox_disk_mb=64, sandbox_pids=64)
    db = DB(d + "/t.db")
    return d, cfg, db, RunControl(cfg, db, EventBus())


class T(unittest.TestCase):
    def test_stop_reasons_only_three(self):
        d, cfg, db, c = mk(Config(budget_usd=1))
        c.add_cost(usd=2)
        self.assertEqual(c.stop_reason, StopReason.BUDGET_EXHAUSTED)
        c.request_stop(StopReason.HUMAN_STOP)                      # first reason wins
        self.assertEqual(c.stop_reason, StopReason.BUDGET_EXHAUSTED)
        with self.assertRaises(Stopped):
            c.checkpoint()
        self.assertEqual({s.value for s in StopReason}, {"goal_reached", "budget_exhausted", "human_stop"})

    def test_budget_persists(self):
        d, cfg, db, c = mk()
        c.add_cost(usd=1.5, gpu_s=10); c.persist()
        c2 = RunControl(cfg, db, EventBus())
        self.assertAlmostEqual(c2.spent_usd, 1.5)

    def test_json_extract(self):
        self.assertEqual(extract_json('x {"a":{"b":"}"}} y'), {"a": {"b": "}"}})

    def test_expect(self):
        ok, _ = check_expect({"a": 1e-5, "b": True}, validate_expect([{"metric": "a", "op": "<", "value": 1e-3}, {"metric": "b", "op": "true"}]))
        self.assertTrue(ok)
        ok, _ = check_expect({"a": "nan"}, validate_expect([{"metric": "a", "op": "<", "value": 1}]))
        self.assertFalse(ok)
        with self.assertRaises(ValueError):
            validate_expect([])

    def test_web_blocks_private(self):
        d, cfg, db, c = mk()
        w = WebTools(cfg, c, d + "/data")
        for u in ("http://127.0.0.1/", "http://169.254.169.254/", "file:///etc/passwd", "http://192.168.1.1/"):
            with self.assertRaises(WebError):
                w.fetch(u)

    def test_evalcore_oracle_and_garbage(self):
        d, cfg, db, c = mk()
        try:
            sb = make_sandbox(cfg, d)
        except SandboxUnavailable:
            self.skipTest("no strong sandbox")
        ec = EvalCore(d, db)
        probs = ec.make_set("eval", 50)
        ans = {p.id: ("```python\n" + p.meta["ref_src"] + "```") if p.kind == "code" else f"ANSWER: {p.answer}" for p in probs}
        self.assertEqual(ec.score(probs, ans, sb).total, 1.0)
        bad = {p.id: "ANSWER: zzz" for p in probs}
        self.assertEqual(ec.score(probs, bad, sb).total, 0.0)
        train = {hashlib.sha1(p.prompt.encode()).hexdigest() for p in ec.make_set("train", 200)}
        self.assertFalse(any(hashlib.sha1(p.prompt.encode()).hexdigest() in train for p in ec.make_set("eval", 200)))

    def test_sandbox_isolation(self):
        d, cfg, db, c = mk()
        try:
            sb = make_sandbox(cfg, d, c)
        except SandboxUnavailable:
            self.skipTest("no strong sandbox")
        rep = sb.selftest()
        self.assertTrue(rep["passed"], [x for x in rep["checks"] if not x["ok"] and x["critical"]])

    def test_weak_sandbox_refused_by_default(self):
        d, cfg, db, c = mk(Config(sandbox_backend="subprocess"))
        with self.assertRaises(SandboxUnavailable):
            make_sandbox(cfg, d)


if __name__ == "__main__":
    unittest.main()


class TNew(unittest.TestCase):
    """Worlds, math lab, workshop, station security, backups, brain map."""

    @classmethod
    def setUpClass(cls):
        from sira.orchestrator import Orchestrator
        cfg = Config(llm_provider="mock", auto_install=False, auto_adapt=False, sandbox_backend="auto", sandbox_timeout_s=40, sandbox_mem_mb=1024, run_root=tempfile.mkdtemp())
        try:
            cls.o = Orchestrator(cfg)
            cls.o.build()
        except SandboxUnavailable:
            cls.o = None

    @classmethod
    def tearDownClass(cls):
        if cls.o:
            cls.o.stop()

    def need(self):
        if not self.o:
            self.skipTest("no strong sandbox")
        return self.o.ctx

    def test_worlds_conserve_their_laws(self):
        from sira import worldlab_lib as w
        for name, steps, dt, key in [("pendulum", 2000, 0.01, "energy"), ("lotka_volterra", 2000, 0.005, "V"),
                                     ("sir", 1000, 0.05, "population"), ("heat1d", 300, 0.4, "total_heat")]:
            self.assertLess(w.load_world(name).check(steps, dt)[key], 1e-6, name)

    def test_wrong_invented_law_is_caught(self):
        from sira import worldlab_lib as w
        t = w.LawWorld("s").set_state(x=1.0, v=0.0).add_law("spring", lambda s: {"x": s["v"], "v": -4 * s["x"]})
        t.add_invariant("right", lambda s: 0.5 * s["v"] ** 2 + 2 * s["x"] ** 2).add_invariant("wrong", lambda s: 0.5 * s["v"] ** 2 + s["x"] ** 2)
        d = t.check(1000, 0.01)
        self.assertLess(d["right"], 1e-6)
        self.assertGreater(d["wrong"], 0.1)

    def test_math_verify_finds_wrong_step(self):
        from sira import worldlab_lib as w
        r = w.math_verify(["(x+1)**2", "x**2+2*x+1", "x**2+2*x+2"], ["x"])
        self.assertEqual(r["first_bad"], 2)
        self.assertTrue(w.math_verify(["sin(x)**2+cos(x)**2 = 1"], ["x"], mode="identity")["ok"])

    def test_create_tool_world_and_security(self):
        ctx = self.need()
        w = ctx.workshop
        self.assertEqual(w.create_tool("t", "double", "doubles x", "def run(a):\n    return {'y': a['x'] * 2}\n", {"x": 2}, {"y": 4})["status"], "active")
        self.assertEqual(w.use_tool("t", "double", {"x": 5})["output"]["y"], 10)
        self.assertEqual(w.create_tool("t", "bad_one", "broken", "def run(a):\n    return 1/0\n", {})["status"], "rejected")
        src = ("import worldlab\ndef build(**p):\n    w = worldlab.LawWorld('c')\n    w.set_state(x=1.0, v=0.0)\n"
               "    w.add_law('s', lambda s: {'x': s['v'], 'v': -s['x']})\n    w.add_invariant('E', lambda s: 0.5*s['v']**2 + 0.5*s['x']**2)\n    return w\n")
        self.assertEqual(w.create_world("t", "cell_world", "physics", "d", src)["status"], "active")
        self.assertEqual(w.create_world("t", "cell2", "x", "d", "def build(**p):\n    return 5\n")["status"], "rejected")
        # symlink / path escape through the persistent station
        w.station_run("t", command="ln -s /etc/passwd /station/evil")
        self.assertIn("error", w.station_files("t", "read", "evil"))
        self.assertIn("error", w.station_files("t", "read", "../../../etc/passwd"))

    def test_backup_restore(self):
        ctx = self.need()
        from sira.backup import BackupManager
        snap = self.o.backup.create("manual-test", include_weights=False)
        self.assertTrue(BackupManager.verify(snap)[0])
        new = BackupManager.restore_to_new_run(snap, tempfile.mkdtemp() + "/r")
        self.assertTrue((new / "sira.db").exists())
        (snap / "sira.db").write_bytes(b"corrupt")
        self.assertFalse(BackupManager.verify(snap)[0])

    def test_brainmap_selection(self):
        from sira import brainmap as B
        cfg = {"hidden_size": 64, "num_hidden_layers": 3, "vocab_size": 100, "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 128}
        d = tempfile.mkdtemp()
        # structure from config only (no weights on disk)
        (Path(d) / "config.json").write_text(json.dumps(cfg))
        st = B.read_structure(d)
        self.assertEqual(len(st["layers"]), 3)
        ops = B.build_ops(st, st, [{"target_layer": 2, "donor_layer": 0, "components": ["mlp.down"], "mode": "blend", "alpha": 0.3}])
        self.assertEqual(ops[0]["target"], "model.layers.2.mlp.down_proj.weight")
        with self.assertRaises(ValueError):
            B.build_ops(st, st, [{"target_layer": 9, "donor_layer": 0}])

    def test_web_policy_lists(self):
        ctx = self.need()
        ctx.cfg.web_mode, ctx.cfg.web_allow = "allowlist", ["arxiv.org"]
        with self.assertRaises(WebError):
            ctx.web._check_url("https://example.com/")
        ctx.cfg.web_mode = "blocklist"
        ctx.cfg.web_block = ["example.com"]
        with self.assertRaises(WebError):
            ctx.web._check_url("https://www.example.com/")
        ctx.cfg.web_mode = "open"


class TProviders(unittest.TestCase):
    """LLM providers against a local fake server (Ollama native API, OpenAI-compatible, Anthropic)."""

    @classmethod
    def setUpClass(cls):
        import http.server
        cls.seen = []
        outer = cls

        class H(http.server.BaseHTTPRequestHandler):
            def _j(s, o):
                b = json.dumps(o).encode()
                s.send_response(200); s.send_header("content-type", "application/json"); s.end_headers(); s.wfile.write(b)

            def do_GET(s):
                s._j({"models": [{"name": "llama3.1:8b"}]}) if s.path == "/api/tags" else s.send_error(404)

            def do_POST(s):
                body = json.loads(s.rfile.read(int(s.headers["content-length"])))
                outer.seen.append((s.path, body, {k.lower(): v for k, v in s.headers.items()}))
                if s.path == "/api/chat":
                    s._j({"message": {"content": '{"ok": true}'}, "prompt_eval_count": 11, "eval_count": 5})
                elif s.path.endswith("/chat/completions"):
                    s._j({"choices": [{"message": {"content": '{"ok": true}'}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
                elif s.path == "/v1/messages":
                    s._j({"content": [{"type": "text", "text": '{"ok": true}'}], "usage": {"input_tokens": 9, "output_tokens": 4}})
                else:
                    s.send_error(404)

            def log_message(*a):
                pass
        cls.srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    def llm(self, **kw):
        from sira.llm import LLM
        d, cfg, db, c = mk(Config(**kw))
        return LLM(cfg, c)

    def test_ollama_native(self):
        l = self.llm(llm_provider="ollama", llm_model="llama3.1:8b", llm_base_url=self.base, llm_json_mode=True, llm_num_ctx=8192)
        ok, msg = l.check()
        self.assertTrue(ok, msg)
        path, body, _ = self.seen[-1]
        self.assertEqual((path, body["format"], body["options"]["num_ctx"], body["stream"]), ("/api/chat", "json", 8192, False))
        self.assertEqual((l.control.tokens_in, l.control.tokens_out, l.control.spent_usd), (11, 5, 0.0))

    def test_ollama_missing_model_is_explained(self):
        ok, msg = self.llm(llm_provider="ollama", llm_model="nope:1b", llm_base_url=self.base).check()
        self.assertFalse(ok)
        self.assertIn("ollama pull nope:1b", msg)

    def test_openai_compatible_family(self):
        os.environ["GROQ_API_KEY"] = "k123"
        for prov in ("openai_compatible", "lmstudio", "vllm", "llamacpp", "groq", "openrouter", "together", "deepseek"):
            l = self.llm(llm_provider=prov, llm_model="m", llm_base_url=self.base + "/v1")
            self.assertTrue(l.check()[0], prov)
        self.llm(llm_provider="groq", llm_model="m", llm_base_url=self.base + "/v1").chat("s", [{"role": "user", "content": "x"}])
        self.assertEqual(self.seen[-1][2].get("authorization"), "Bearer k123")

    def test_anthropic_and_validation(self):
        os.environ["ANTHROPIC_API_KEY"] = "a1"
        self.assertTrue(self.llm(llm_provider="anthropic", llm_model="m", llm_base_url=self.base).check()[0])
        self.assertEqual(self.seen[-1][2].get("x-api-key"), "a1")
        with self.assertRaises(ValueError):
            Config(llm_provider="openai_compatible").validate()
        with self.assertRaises(ValueError):
            Config(llm_provider="nonsense").validate()


class TBootstrap(unittest.TestCase):
    def test_torch_wheel_selection(self):
        from sira.bootstrap import torch_install_args
        self.assertIn("whl/cpu", torch_install_args({"cuda": False}, None)[-1])
        self.assertIn("cu126", torch_install_args({"cuda": True}, "12.6")[-1])
        self.assertIn("cu118", torch_install_args({"cuda": True}, "11.8")[-1])

    def test_adapt_to_hardware(self):
        from sira.bootstrap import adapt_config
        c = Config(parallel_experiments=6, sandbox_mem_mb=4096, micro_batch=4)
        ch = adapt_config(c, {"cpu_count": 2, "ram_gb": 8.0, "cuda": False, "gpus": []})
        self.assertEqual(c.parallel_experiments, 1)
        self.assertLessEqual(c.sandbox_mem_mb, 2000)
        big = Config()
        adapt_config(big, {"cpu_count": 32, "ram_gb": 256, "cuda": True, "gpus": [{"vram_gb": 80.0}]})
        self.assertEqual((big.micro_batch, big.grad_accum), (8, 2))

    def test_algorithm_catalog_and_probe(self):
        from sira.bootstrap import probe
        r = probe()
        names = {a["name"] for a in r["algorithms"]}
        self.assertTrue({"lora", "dora", "rslora", "qlora", "grpo", "dpo", "rft", "ties", "dare", "transplant"} <= names)
        self.assertFalse(next(a for a in r["algorithms"] if a["name"] == "qlora")["usable"] and not r["hw"]["cuda"])


class TStats(unittest.TestCase):
    def test_exact_paired_test(self):
        from sira.trainer import binom_tail
        self.assertAlmostEqual(binom_tail(9, 12), 0.0730, places=3)
        self.assertAlmostEqual(binom_tail(0, 5), 1.0)
        self.assertEqual(binom_tail(0, 0), 1.0)
        self.assertLess(binom_tail(10, 10), 0.001)


class TCla(unittest.TestCase):
    def test_cla_cache_equivalence_and_savings(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from sira import cla_model as C
        C.register()
        base = dict(vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
        torch.manual_seed(0)
        sizes = {}
        for f in (1, 2, 3, 6):
            m = (LlamaForCausalLM(LlamaConfig(**base)) if f == 1 else C.ClaForCausalLM(C.make_config(base, cla_factor=f))).eval()
            sizes[f] = sum(p.numel() for p in m.parameters())
            x = torch.randint(0, 200, (1, 14))
            with torch.no_grad():
                full = m(input_ids=x).logits
                o = m(input_ids=x[:, :9], use_cache=True)
                pk, outs = o.past_key_values, [o.logits]
                for t in range(9, 14):
                    o = m(input_ids=x[:, t:t + 1], past_key_values=pk, use_cache=True)
                    pk = o.past_key_values
                    outs.append(o.logits)
            self.assertLess(float((full - torch.cat(outs, 1)).abs().max()), 1e-5, f"CLA-{f}")
            self.assertEqual(C.kv_cache_bytes_per_token(6, 2, 16, f), 2 * (-(-6 // f)) * 2 * 16 * 2)
        self.assertTrue(sizes[1] > sizes[2] > sizes[3] > sizes[6])

    def test_conversion_math(self):
        import subprocess, torch
        from safetensors import safe_open
        src = "/tmp/tiny"
        if not os.path.isdir(src):
            self.skipTest("tiny model fixture missing")
        d = tempfile.mkdtemp()
        json.dump({"model_path": src, "out_dir": d + "/o", "kv_heads": 0, "cla_factor": 2, "cla_init": "mean"}, open(d + "/a.json", "w"))
        r = subprocess.run([sys.executable, "-m", "sira.modelworker", "convert_arch", d + "/a.json"], capture_output=True, text=True,
                           cwd=str(Path(__file__).resolve().parent.parent))
        self.assertEqual(r.returncode, 0, r.stderr[-300:])
        with safe_open(d + "/o/model.safetensors", "pt") as h, safe_open(src + "/model.safetensors", "pt") as s:
            self.assertNotIn("model.layers.1.self_attn.k_proj.weight", set(h.keys()))
            exp = (s.get_tensor("model.layers.0.self_attn.k_proj.weight").float() + s.get_tensor("model.layers.1.self_attn.k_proj.weight").float()) / 2
            self.assertTrue(torch.allclose(h.get_tensor("model.layers.0.self_attn.k_proj.weight").float(), exp, atol=1e-6))


class TSociety(unittest.TestCase):
    def test_engine_separates_understanding_from_shortcuts(self):
        from sira import society as S
        oracle = S.evaluate_policy(S.oracle_policy, 60, 1)
        naive = S.evaluate_policy(S.naive_policy, 60, 1)
        rand = S.evaluate_policy(S.random_policy_factory(2), 60, 1)
        self.assertEqual(oracle["life_score"], 1.0)
        self.assertLess(naive["life_score"], 0.3)             # trusting + eager + reality-only is NOT rewarded
        self.assertLess(rand["life_score"], 0.6)
        self.assertEqual(naive["false_belief"], 0.0)          # tracks reality, has no theory of mind

    def test_episodes_deterministic(self):
        import random
        from sira import society as S
        for cls in S.EPISODES:
            self.assertEqual(cls(random.Random(9)).intro(), cls(random.Random(9)).intro())


class TChallenges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sira.orchestrator import Orchestrator
        from sira.challenges import ChallengeEngine
        cfg = Config(llm_provider="mock", auto_install=False, auto_adapt=False, thinking="off", sandbox_backend="auto", sandbox_timeout_s=40, run_root=tempfile.mkdtemp())
        try:
            cls.o = Orchestrator(cfg)
            cls.o.build()
            cls.eng = ChallengeEngine(cls.o.ctx, cls.o.research)
        except SandboxUnavailable:
            cls.o = None

    @classmethod
    def tearDownClass(cls):
        if cls.o:
            cls.o.stop()

    def need(self):
        if not self.o:
            self.skipTest("no strong sandbox")

    def test_verifier_sanity_rejects_bad_verifiers(self):
        self.need()
        good = {"verifier_code": "def verify(a):\n    s = max(0.0, 1 - abs(a - 10) / 10) if isinstance(a, int) and not isinstance(a, bool) else 0.0\n    return {'score': s, 'holdout': s}",
                "probes": [{"answer": 2, "expect": "low"}, {"answer": 10, "expect": "high"}], "kind": "reference", "statement": "Return the integer ten."}
        gameable = {"verifier_code": "def verify(a):\n    s = min(a, 10) / 10 if isinstance(a, int) else 0.0\n    return {'score': s, 'holdout': s}",
                    "probes": [{"answer": 2, "expect": "low"}, {"answer": 10, "expect": "high"}], "kind": "reference", "statement": "Return a number, the bigger the better."}
        self.assertFalse(self.eng.validate(gameable)[0])           # any huge integer would score 1.0
        self.assertTrue(self.eng.validate(good)[0])
        self.assertFalse(self.eng.validate({"verifier_code": "def verify(a):\n    return {'score': 1.0}", "probes": [{"answer": 1, "expect": "low"}], "statement": "s1"})[0])     # gameable
        self.assertFalse(self.eng.validate({"verifier_code": "def verify(a):\n    return 1/0", "probes": [{"answer": 1, "expect": "low"}]})[0])                # crashes
        self.assertFalse(self.eng.validate({"verifier_code": "import random\ndef verify(a):\n    return {'score': random.random() * 0.4}", "probes": [{"answer": 1, "expect": "low"}]})[0])

    def test_progress_reward_and_gaming_penalty(self):
        self.need()
        cid = self.o.db.add_challenge("t", "d", "optimization", "Answer a number.", "def verify(a):\n    return {'score': 0.5, 'holdout': 0.1, 'msg': 'x'}", [{"answer": 0, "expect": "low"}], 3, "t")
        v = self.eng.verify_answer(self.o.db.challenge(cid)["verifier_code"], 1)
        self.assertGreater(v["score"] - v["holdout"], 0.3)      # exactly the gaming condition that costs a penalty

    def test_full_cycle_demo(self):
        self.need()
        res = self.eng.cycle(1, 2, 3)
        self.assertTrue(res)
        led = self.eng.ledger.totals()
        self.assertGreater(led["reward"], 0)
        self.assertTrue(any(c["status"] == "solved" for c in self.o.db.challenges()))


class TQuestGrading(unittest.TestCase):
    def test_value_grading_and_calibration_bounds(self):
        from sira.quest import QuestMaestro
        qm = QuestMaestro.__new__(QuestMaestro)
        qm.ctx = None
        q = {"kind": "value", "truth": 0.001234, "ref": 1}
        self.assertEqual(QuestMaestro.grade(qm, q, {"answer": 0.0012})[0], 1.0)
        self.assertEqual(QuestMaestro.grade(qm, q, {"answer": 0.002})[0], 0.5)
        self.assertEqual(QuestMaestro.grade(qm, q, {"answer": 5})[0], 0.0)
        s = {"kind": "status", "truth": "REFUTED", "ref": 3}
        self.assertEqual(QuestMaestro.grade(qm, s, {"answer": "refuted"})[0], 1.0)
        self.assertEqual(QuestMaestro.grade(qm, s, {"answer": "VERIFIED"})[0], 0.0)

    def test_quest_end_to_end_with_demo_brain(self):
        from sira.orchestrator import Orchestrator
        cfg = Config(mode="quest", quest_goal="reach AGI", llm_provider="mock", auto_install=False, auto_adapt=False, thinking="off", sandbox_backend="auto",
                     sandbox_timeout_s=40, run_root=tempfile.mkdtemp(), quest_questions=8, society_episodes=5, attempts_per_challenge=2,
                     challenge_domains_per_cycle=1, challenges_per_domain=2, quest_pass_rounds=50)
        try:
            o = Orchestrator(cfg)
            o.build()
        except SandboxUnavailable:
            self.skipTest("no strong sandbox")
        st = o.quest.cycle()
        self.assertEqual(st["rounds"], 1)
        h = st["history"][0]
        self.assertGreater(h["exam"], h["blank"])               # notes beat a blank student: measured learning
        self.assertTrue(o.db.society_runs())
        o.stop()


class TMemoryQuarantine(unittest.TestCase):
    def test_lessons_are_quarantined_until_a_mechanical_retest_passes(self):
        from sira.orchestrator import Orchestrator
        cfg = Config(mode="quest", quest_goal="g", llm_provider="mock", auto_install=False, auto_adapt=False, thinking="off", sandbox_backend="auto", sandbox_timeout_s=30,
                     run_root=tempfile.mkdtemp())
        try:
            o = Orchestrator(cfg)
            o.build()
        except SandboxUnavailable:
            self.skipTest("no strong sandbox")
        qm = o.quest
        q = {"kind": "status", "prompt": "Statement: X\nIs this supported?", "format": "word", "truth": "VERIFIED", "ref": 1}
        # (a) a lesson that does NOT change the student's behaviour must be rejected
        qm.ask = lambda qq, notes: {"answer": "UNVERIFIED", "confidence": 0.5}
        qm._controls = {"status": []}
        qm.autopsy(q, {"answer": "UNVERIFIED"}, "truth is VERIFIED", [])
        less = o.db.lessons(10, include_inactive=True)
        self.assertEqual(less[0]["status"], "rejected")
        self.assertEqual(o.db.lessons(10), [])                     # rejected lessons never reach a prompt
        # (b) a lesson that fixes the answer but BREAKS a previously correct one must be rejected (conflict check)
        good_ctrl = {"kind": "status", "prompt": "Statement: Y", "format": "word", "truth": "REFUTED", "ref": 2}
        qm.ask = lambda qq, notes: {"answer": ("VERIFIED" if qq is q else "VERIFIED"), "confidence": 0.5} if any(n["id"] == "lesson" for n in notes) else {"answer": "UNVERIFIED", "confidence": 0.5}
        qm._controls = {"status": [good_ctrl]}
        qm.autopsy(q, {"answer": "UNVERIFIED"}, "truth is VERIFIED", [])
        self.assertEqual(o.db.lessons(10, include_inactive=True)[0]["status"], "rejected")
        # (c) a lesson that fixes it without regressions is adopted
        qm.ask = lambda qq, notes: {"answer": qq["truth"], "confidence": 0.9} if any(n["id"] == "lesson" for n in notes) else {"answer": "UNVERIFIED", "confidence": 0.5}
        qm.autopsy(q, {"answer": "UNVERIFIED"}, "truth is VERIFIED", [])
        top = o.db.lessons(10)[0]
        self.assertEqual((top["status"], top["source"]), ("active", "llm-explanation"))
        self.assertTrue(any(m["source"] == "mechanical-ground-truth" for m in o.db.mistakes(10)))
        o.stop()
