"""Fault injection: API errors, malformed replies, slowness, network and download failures, corrupted models, worker crashes.
Run: python -m unittest tests.test_faults -v"""
import http.server, json, os, sys, tempfile, threading, time, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sira.config import Config
from sira.db import DB
from sira.bus import EventBus
from sira.control import RunControl, StopReason, Stopped
from sira.llm import LLM, LLMError, extract_json
from sira.web import WebTools, WebError


class Faulty(http.server.BaseHTTPRequestHandler):
    plan = []            # list of ("status", body, delay) consumed per request
    hits = 0

    def do_POST(self):
        Faulty.hits += 1
        self.rfile.read(int(self.headers.get("content-length", 0)))
        status, body, delay = Faulty.plan.pop(0) if Faulty.plan else (200, {"message": {"content": '{"ok": true}'}}, 0)
        time.sleep(delay)
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status); self.send_header("content-type", "application/json"); self.end_headers(); self.wfile.write(raw)

    def do_GET(self):
        self.send_response(500); self.end_headers(); self.wfile.write(b"boom")

    def log_message(self, *a): pass


def mk(**kw):
    d = tempfile.mkdtemp()
    cfg = Config(**kw)
    db = DB(d + "/t.db")
    return d, cfg, RunControl(cfg, db, EventBus())


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.HTTPServer(("127.0.0.1", 0), Faulty)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    def llm(self, **kw):
        d, cfg, c = mk(llm_provider="ollama", llm_model="m", llm_base_url=self.base, llm_timeout_s=2, **kw)
        c.sleep = lambda s: None                    # no real waiting between retries in the test
        return LLM(cfg, c), c

    def test_retries_on_500_and_429_then_succeeds(self):
        Faulty.plan, Faulty.hits = [(500, {"error": "x"}, 0), (429, {"error": "slow down"}, 0)], 0
        l, c = self.llm()
        self.assertEqual(l.chat("s", [{"role": "user", "content": "x"}]), '{"ok": true}')
        self.assertEqual(Faulty.hits, 3)

    def test_auth_error_is_not_retried(self):
        Faulty.plan, Faulty.hits = [(401, {"error": "bad key"}, 0)] * 5, 0
        l, c = self.llm()
        with self.assertRaises(LLMError) as cm:
            l.chat("s", [{"role": "user", "content": "x"}])
        self.assertIn("authentication", str(cm.exception))
        self.assertEqual(Faulty.hits, 1)

    def test_malformed_and_empty_replies_do_not_crash_json_extraction(self):
        for raw in (b"not json at all", b"", b"{truncated"):
            Faulty.plan = [(200, raw, 0)] * 8
            l, c = self.llm()
            try:
                out = l.chat("s", [{"role": "user", "content": "x"}])
            except (LLMError, ValueError):
                continue                                         # a clean, typed failure is acceptable
            with self.assertRaises(ValueError):
                extract_json(out)

    def test_slow_server_times_out_cleanly(self):
        Faulty.plan = [(200, {"message": {"content": "x"}}, 5)] * 9
        l, c = self.llm()
        t = time.time()
        with self.assertRaises(LLMError):
            l.chat("s", [{"role": "user", "content": "x"}])
        self.assertLess(time.time() - t, 60)

    def test_server_down_gives_actionable_message(self):
        d, cfg, c = mk(llm_provider="ollama", llm_model="m", llm_base_url="http://127.0.0.1:9")
        ok, msg = LLM(cfg, c).check()
        self.assertFalse(ok)
        self.assertIn("ollama serve", msg)

    def test_stop_interrupts_llm_retry_loop(self):
        Faulty.plan = [(500, {"error": "x"}, 0)] * 20
        d, cfg, c = mk(llm_provider="ollama", llm_model="m", llm_base_url=self.base, llm_timeout_s=2)
        l = LLM(cfg, c)
        threading.Timer(0.5, lambda: c.request_stop(StopReason.HUMAN_STOP)).start()
        with self.assertRaises(Stopped):
            l.chat("s", [{"role": "user", "content": "x"}])

    def test_every_call_is_logged(self):
        Faulty.plan = []
        l, c = self.llm()
        d = tempfile.mkdtemp()
        l.log_path = Path(d) / "calls.jsonl"
        l.chat("ROLE=solver\nsys", [{"role": "user", "content": "hi"}])
        rec = json.loads(l.log_path.read_text().splitlines()[0])
        self.assertEqual((rec["role"], rec["reply"]), ("solver", '{"ok": true}'))

    def test_web_failures(self):
        d, cfg, c = mk()
        w = WebTools(cfg, c, d + "/data")
        with self.assertRaises(WebError):
            w.fetch("http://this-host-does-not-exist.invalid/")        # DNS failure
        cfg.block_private_ips = False
        with self.assertRaises(WebError):
            w.fetch(self.base + "/x")                                   # HTTP 500
        with self.assertRaises(WebError):
            w.download(self.base + "/x", "f.bin")                       # failed download leaves no partial file
        self.assertFalse((Path(d) / "data" / "f.bin").exists())
        with self.assertRaises(WebError):
            w.download("http://127.0.0.1:1/", "../evil")                # unsafe filename refused before any request

    def test_corrupted_or_missing_models_fail_with_a_clear_error(self):
        from sira import intake
        d = tempfile.mkdtemp()
        with self.assertRaises(intake.IntakeError):
            intake.analyze_model(d)                                      # no config.json
        (Path(d) / "config.json").write_text("{not json")
        with self.assertRaises(Exception):
            intake.analyze_model(d)
        (Path(d) / "config.json").write_text(json.dumps({"architectures": ["BertModel"], "model_type": "bert", "hidden_size": 8}))
        rep = intake.analyze_model(d)
        self.assertFalse(rep["supported"])                              # encoder-only: refused, not crashed
        (Path(d) / "config.json").write_text(json.dumps({"architectures": ["FooForCausalLM"], "auto_map": {"x": "y"}}))
        self.assertFalse(intake.analyze_model(d)["supported"])           # remote code refused

    def test_worker_crash_becomes_training_error_not_a_hang(self):
        from sira.trainer import TrainingMaestro, TrainingError
        from sira.orchestrator import Orchestrator
        cfg = Config(llm_provider="mock", auto_install=False, auto_adapt=False, thinking="off", sandbox_backend="auto", run_root=tempfile.mkdtemp(), base_model=tempfile.mkdtemp())
        try:
            o = Orchestrator(cfg)
            o.build()
        except Exception:
            self.skipTest("no sandbox")
        tm = o.training
        with self.assertRaises(TrainingError):
            tm.run_worker("generate", {"model_path": "/nonexistent/model", "adapter": None, "prompts_path": "/x", "out_path": "/y", "n_samples": 1, "temperature": 0, "max_new_tokens": 2,
                                       "dtype": "float32", "device": "cpu", "batch_size": 1}, "crash test")
        o.stop()

    def test_resume_keeps_budget_and_generations(self):
        from sira.orchestrator import Orchestrator
        cfg = Config(llm_provider="mock", auto_install=False, auto_adapt=False, thinking="off", sandbox_backend="auto", run_root=tempfile.mkdtemp())
        try:
            o = Orchestrator(cfg)
            o.build()
        except Exception:
            self.skipTest("no sandbox")
        o.control.add_cost(usd=1.25, gpu_s=100)
        o.db.add_generation(None, "m", "base", 0.5, {}, True, "x")
        o.control.persist()
        rd = o.run_dir
        o.stop()
        o2 = Orchestrator(Config.load(rd / "config.json"), run_dir=rd)
        self.assertAlmostEqual(o2.control.spent_usd, 1.25)
        self.assertEqual(len(o2.db.generations()), 1)
        o2.stop()


if __name__ == "__main__":
    unittest.main()
