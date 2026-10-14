"""Entry point:  python -m sira [--config file.json] [--headless] [--selftest] [--build-sandbox]"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .config import Config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sira")
    ap.add_argument("--config", help="JSON config file")
    ap.add_argument("--resume", help="resume a run folder (or a restored backup) - keeps budget, knowledge, generations")
    ap.add_argument("--headless", action="store_true", help="run without GUI (Ctrl+C = human stop)")
    ap.add_argument("--selftest", action="store_true", help="run the sandbox isolation self-test and exit")
    ap.add_argument("--test-llm", action="store_true", help="test the agents' brain connection (provider/model/key) and exit")
    ap.add_argument("--validate-real", action="store_true", help="run the validation harness against your real brain (and target) model; writes report.json/md and logs every call")
    ap.add_argument("--validate-trials", type=int, default=5)
    ap.add_argument("--validate-rounds", type=int, default=3)
    ap.add_argument("--validate-out", default="validation_report")
    ap.add_argument("--version", action="store_true")
    ap.add_argument("--verify-install", action="store_true", help="check every package file against the release manifest (detects corruption)")
    ap.add_argument("--bootstrap", action="store_true", help="probe this machine, install missing packages, show usable algorithms, and exit")
    ap.add_argument("--build-sandbox", action="store_true", help="build the Docker sandbox image and exit")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config) if args.config else Config()
    if args.resume:
        cfg = Config.load(Path(args.resume) / "config.json")

    if args.build_sandbox:
        df = Path(__file__).resolve().parent.parent / "docker" / "Dockerfile.sandbox"
        return subprocess.call(["docker", "build", "-t", cfg.docker_image, "-f", str(df), str(df.parent)])

    if args.verify_install:
        import hashlib
        root = Path(__file__).resolve().parent
        mf = root / "manifest.json"
        if not mf.exists():
            print("no manifest.json in this installation")
            return 2
        want = json.loads(mf.read_text())["files"]
        bad = [f for f, h in want.items() if not (root / f).exists() or hashlib.sha256((root / f).read_bytes()).hexdigest() != h]
        print("installation OK: all %d files match the manifest" % len(want) if not bad else f"CORRUPTED OR MODIFIED FILES: {bad}")
        return 0 if not bad else 1

    if args.version:
        from . import __version__
        print(__version__)
        return 0

    if args.validate_real:
        from .validate_real import run_validation
        rep = run_validation(cfg, args.validate_out, args.validate_trials, args.validate_rounds)
        print(json.dumps({k: rep[k] for k in ("verdict", "roles", "checks") if k in rep}, indent=2, default=str))
        print(f"report: {Path(args.validate_out) / 'report.md'}")
        return 0 if str(rep.get("verdict", "")).startswith("PASS") else 1

    if args.test_llm:
        from .bus import EventBus
        from .control import RunControl
        from .db import DB
        from .llm import LLM
        ctl = RunControl(cfg, DB(tempfile.mkdtemp() + "/t.db"), EventBus())
        ok, msg = LLM(cfg, ctl).check()
        print(("OK: " if ok else "FAILED: ") + msg)
        return 0 if ok else 1

    if args.bootstrap:
        from .bootstrap import Bootstrap
        from .bus import EventBus
        bus = EventBus()
        rep = Bootstrap(cfg, bus).run()
        for ev in bus.drain():
            if ev["kind"] == "bootstrap":
                print(ev["msg"])
        print("\nUsable algorithms:", ", ".join(rep["usable_algorithms"]))
        return 0 if not rep["missing"] else 1

    if args.selftest:
        from .sandbox import make_sandbox, SandboxUnavailable
        try:
            sb = make_sandbox(cfg, tempfile.mkdtemp(prefix="sira_selftest_"))
        except SandboxUnavailable as e:
            print("UNAVAILABLE:", e)
            return 2
        rep = sb.selftest()
        print(f"backend={rep['backend']} strong={rep['strong']} passed={rep['passed']}")
        for c in rep["checks"]:
            print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}{'' if c['critical'] else ' (non-critical)'} {c['detail']}")
        return 0 if rep["passed"] else 1

    if args.headless:
        import signal
        from .orchestrator import Orchestrator
        o = Orchestrator(cfg, run_dir=args.resume) if args.resume else Orchestrator(cfg)
        hits = {"n": 0}

        def on_signal(signum, frame):            # never raise inside arbitrary code: just record the request (repeated signals are fine)
            hits["n"] += 1

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        o.start()
        asked = False
        while o.thread.is_alive():
            for ev in o.bus.drain():
                if ev["kind"] != "agent_status":
                    print(f"{time.strftime('%H:%M:%S', time.localtime(ev['t']))} [{ev['agent']}] {ev['msg']}", flush=True)
            if hits["n"] and not asked:
                print("Human stop requested - finishing safely (final report, backup)...", flush=True)
                o.stop()
                asked = True
            if hits["n"] >= 5:                   # impatient: stop waiting (state is already persisted every cycle)
                break
            time.sleep(0.3)
        o.thread.join(timeout=60)
        for ev in o.bus.drain():
            if ev["kind"] != "agent_status":
                print(f"[{ev['agent']}] {ev['msg']}", flush=True)
        print(f"run folder: {o.run_dir}")
        return 0

    from .gui import run_gui
    run_gui(cfg, resume_dir=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
