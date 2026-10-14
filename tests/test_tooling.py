import contextlib
import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from sira.__main__ import main
from sira.configcheck import check_config
from sira.db import DB
from sira.doctor import run_doctor, verify_manifest
from sira.report import list_runs, make_server, render_html, render_markdown, summarize

EVIL = "<script>alert(1)</script>"


def make_run(root: Path, name="run_a") -> Path:
    d = root / name
    d.mkdir(parents=True)
    db = DB(d / "sira.db")
    for i, (sc, acc) in enumerate([(0.2, 1), (0.35, 1), (0.3, 0), (0.5, 1)]):
        db.conn.execute("INSERT INTO generations(parent,path,method,score,accepted,created) VALUES(?,?,?,?,?,0)", (i or None, "p", "lora", sc, acc))
    db.conn.execute("INSERT INTO claims(text,status,created) VALUES('a','verified',0)")
    db.conn.execute("INSERT INTO claims(text,status,created) VALUES('b','refuted',0)")
    db.conn.execute("INSERT INTO experiments(agent,exit_code,timed_out,created) VALUES('x',0,0,0)")
    db.conn.close()
    (d / "config.json").write_text(json.dumps({"goal_text": EVIL, "mode": "model", "llm_provider": "mock"}))
    return d


def get(url, headers=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=5) as r:
            return r.status, r.read().decode(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), dict(e.headers)


class ConfigCheck(unittest.TestCase):
    def test_good_config(self):
        r = check_config({"llm_provider": "mock", "llm_model": "m", "base_model": "x/y", "target_score": 0.8})
        self.assertTrue(r["ok"], r)

    def test_typos_types_ranges(self):
        r = check_config({"llm_provider": "mock", "buget_usd": 5, "target_score": 1.5, "parallel_experiments": "3", "llm_timeout_s": True})
        self.assertFalse(r["ok"])
        self.assertTrue(any("budget_usd" in w for w in r["warnings"]))          # suggestion
        self.assertTrue(any("target_score" in e for e in r["errors"]))
        self.assertTrue(any("parallel_experiments" in e for e in r["errors"]))
        self.assertTrue(any("llm_timeout_s" in e for e in r["errors"]))

    def test_provider_and_modes(self):
        self.assertFalse(check_config({"llm_provider": "nope"})["ok"])
        self.assertFalse(check_config({"llm_provider": "openai_compatible"})["ok"])
        self.assertFalse(check_config({"llm_provider": "mock", "mode": "quest", "quest_goal": ""})["ok"])
        self.assertTrue(any("no model" in w for w in check_config({"llm_provider": "mock", "mode": "model"})["warnings"]))

    def test_bad_files(self):
        self.assertFalse(check_config("/nonexistent/x.json")["ok"])
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{oops")
        self.assertFalse(check_config(f.name)["ok"])


class Doctor(unittest.TestCase):
    def test_fast_doctor_shape(self):
        rep = run_doctor(None, fast=True)
        names = {c["name"] for c in rep["checks"]}
        self.assertTrue({"python", "install integrity", "sandbox backend", "brain"} <= names)
        self.assertEqual(sum(rep["summary"].values()), len(rep["checks"]))
        self.assertEqual(next(c for c in rep["checks"] if c["name"] == "install integrity")["status"], "ok")

    def test_manifest_detects_tampering(self):
        with tempfile.TemporaryDirectory() as t:
            pkg = Path(__file__).resolve().parent.parent / "sira"
            for p in pkg.rglob("*"):
                if p.is_file():
                    (Path(t) / p.relative_to(pkg)).parent.mkdir(parents=True, exist_ok=True)
                    (Path(t) / p.relative_to(pkg)).write_bytes(p.read_bytes())
            self.assertTrue(verify_manifest(t)[0])
            (Path(t) / "sandbox.py").write_text("# tampered\n")
            (Path(t) / "extra.py").write_text("x=1\n")
            ok, bad, _ = verify_manifest(t)
            self.assertFalse(ok)
            self.assertIn("sandbox.py", bad)
            self.assertIn("unlisted:extra.py", bad)


class Reports(unittest.TestCase):
    def test_summary_and_escaping(self):
        with tempfile.TemporaryDirectory() as t:
            run = make_run(Path(t))
            s = summarize(run)
            self.assertEqual(len(s["generations"]), 4)
            self.assertEqual(s["best_score"], 0.5)                              # rejected 0.3 never counts as best
            self.assertEqual(s["claims"], {"verified": 1, "refuted": 1})
            self.assertEqual(s["experiments"]["total"], 1)
            h = render_html(s)
            self.assertNotIn(EVIL, h)
            self.assertIn("&lt;script&gt;", h)
            self.assertIn("<polyline", h)
            self.assertIn("| 4 |", render_markdown(s))
            self.assertEqual(list_runs(t)[0]["generations"], 4)

    def test_missing_db(self):
        with tempfile.TemporaryDirectory() as t, self.assertRaises(FileNotFoundError):
            summarize(t)

    def test_read_only(self):
        with tempfile.TemporaryDirectory() as t:
            run = make_run(Path(t))
            before = (run / "sira.db").read_bytes()
            summarize(run)
            self.assertEqual(before, (run / "sira.db").read_bytes())


class Dashboard(unittest.TestCase):
    def test_server_auth_and_routes(self):
        with tempfile.TemporaryDirectory() as t:
            make_run(Path(t), "run_a")
            make_run(Path(t), "run_b")
            srv = make_server(t, "127.0.0.1", 0, token="s3cret")
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            try:
                self.assertEqual(get(base + "/healthz")[0], 200)
                self.assertEqual(get(base + "/")[0], 401)
                self.assertEqual(get(base + "/?token=wrong")[0], 401)
                code, body, hdr = get(base + "/?token=s3cret")
                self.assertEqual(code, 200)
                self.assertIn("run_a", body)
                self.assertIn("default-src 'none'", hdr.get("Content-Security-Policy", ""))
                self.assertEqual(get(base + "/run/run_a", {"Authorization": "Bearer s3cret"})[0], 200)
                self.assertEqual(get(base + "/run/..%2F..%2Fetc?token=s3cret")[0], 404)
                self.assertEqual(get(base + "/run/nope?token=s3cret")[0], 404)
                self.assertEqual(len(json.loads(get(base + "/api/runs?token=s3cret")[1])), 2)
                self.assertEqual(json.loads(get(base + "/api/run/run_b?token=s3cret")[1])["run"], "run_b")
            finally:
                srv.shutdown()
                srv.server_close()

    def test_refuses_public_bind_without_token(self):
        with self.assertRaises(ValueError):
            make_server(".", "0.0.0.0", 0)


class CLI(unittest.TestCase):
    def run_cli(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(list(argv))
        return code, buf.getvalue()

    def test_commands(self):
        with tempfile.TemporaryDirectory() as t:
            run = make_run(Path(t))
            cfg = Path(t) / "c.json"
            cfg.write_text(json.dumps({"llm_provider": "mock", "base_model": "a/b"}))
            self.assertEqual(self.run_cli("--check-config", "--config", str(cfg))[0], 0)
            cfg.write_text(json.dumps({"target_score": 7}))
            self.assertEqual(self.run_cli("--check-config", "--config", str(cfg), "--json")[0], 1)
            self.assertEqual(self.run_cli("--check-config")[0], 2)
            code, out = self.run_cli("--doctor", "--fast", "--json")
            self.assertIn(code, (0, 1))
            self.assertIn("checks", json.loads(out))
            code, out = self.run_cli("--report", str(run))
            self.assertEqual(code, 0)
            self.assertTrue((run / "report.html").exists())
            self.assertIn("SIRA run report", self.run_cli("--report", str(run), "--format", "md")[1])
            self.assertEqual(json.loads(self.run_cli("--report", str(run), "--format", "json")[1])["run"], "run_a")
            self.assertIn("run_a", self.run_cli("--list-runs", t)[1])
            self.assertEqual(self.run_cli("--report", t + "/missing")[0], 2)
            self.assertEqual(self.run_cli("--serve", t, "--host", "0.0.0.0")[0], 2)
            self.assertEqual(self.run_cli("--version")[1].strip(), "2.1.0")


if __name__ == "__main__":
    unittest.main()
