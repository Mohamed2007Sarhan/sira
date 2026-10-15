"""One-shot health check: `python -m sira --doctor`. Pure standard library; never raises, every problem becomes a FAIL/WARN line."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

PKG = Path(__file__).resolve().parent


def verify_manifest(root=PKG):
    """-> (ok|None if no manifest, problems, n_files)."""
    root = Path(root)
    mf = root / "manifest.json"
    if not mf.exists():
        return None, [], 0
    want = json.loads(mf.read_text())["files"]
    bad = [f for f, h in want.items() if not (root / f).exists() or __import__("hashlib").sha256((root / f).read_bytes()).hexdigest() != h]
    bad += ["unlisted:" + str(p.relative_to(root)) for p in sorted(root.rglob("*.py")) if str(p.relative_to(root)) not in want]
    return not bad, bad, len(want)


def run_doctor(config_path=None, fast: bool = False) -> dict:
    checks = []

    def add(name, status, detail=""):
        checks.append({"name": name, "status": status, "detail": str(detail)})

    # python
    v = sys.version_info
    add("python", "ok" if v >= (3, 10) else "fail", f"{v.major}.{v.minor}.{v.micro}" + ("" if v >= (3, 10) else " (need >= 3.10)"))
    # install integrity
    try:
        ok, bad, n = verify_manifest()
        add("install integrity", "warn" if ok is None else "ok" if ok else "fail",
            "no manifest.json" if ok is None else f"{n} files match the manifest" if ok else "modified/unlisted: " + ", ".join(bad[:6]))
    except Exception as e:
        add("install integrity", "fail", e)
    # config
    cfg = None
    from .config import Config
    if config_path:
        from .configcheck import check_config
        rep = check_config(config_path)
        for e in rep["errors"]:
            add("config", "fail", e)
        for w in rep["warnings"]:
            add("config", "warn", w)
        if rep["ok"]:
            add("config", "ok", str(config_path))
            try:
                cfg = Config.load(config_path)
            except Exception as e:
                add("config", "fail", e)
    cfg = cfg or Config()
    # optional modules
    has = lambda m: importlib.util.find_spec(m) is not None
    add("tkinter (GUI)", "ok" if has("tkinter") else "warn", "available" if has("tkinter") else "missing - headless/--serve still work (apt install python3-tk)")
    train = [m for m in ("torch", "transformers", "peft", "accelerate", "safetensors", "numpy") if not has(m)]
    add("training stack", "ok" if not train else "warn", "complete" if not train else "missing: " + ", ".join(train) + " (auto-installed at first start, or: python -m sira --bootstrap)")
    add("sympy (symbolic verification)", "ok" if has("sympy") else "warn", "available" if has("sympy") else "missing - numeric checks only")
    # gpu / disk
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
            add("gpu", "ok", out.replace("\n", "; ") or "nvidia-smi returned nothing")
        except Exception as e:
            add("gpu", "warn", e)
    else:
        add("gpu", "warn", "no NVIDIA GPU detected - CPU works for tiny models only")
    try:
        probe = Path(cfg.run_root) if Path(cfg.run_root).exists() else Path.cwd()
        free = shutil.disk_usage(probe).free / 1e9
        add("disk", "ok" if free >= 5 else "warn", f"{free:.1f} GB free at {probe}")
    except Exception as e:
        add("disk", "warn", e)
    # sandbox
    docker, bwrap = shutil.which("docker"), shutil.which("bwrap")
    if not (docker or bwrap):
        weak = getattr(cfg, "allow_weak_sandbox", False)
        add("sandbox backend", "warn" if weak else "fail", "no bubblewrap/Docker" + (" (weak backend explicitly allowed)" if weak else " - experiments are refused (apt install bubblewrap)"))
    else:
        add("sandbox backend", "ok", ", ".join(n for n, p in (("bubblewrap", bwrap), ("docker", docker)) if p))
        if fast:
            add("sandbox self-test", "warn", "skipped (--fast)")
        else:
            tmp = tempfile.mkdtemp(prefix="sira_doctor_")
            try:
                from .sandbox import make_sandbox, SandboxUnavailable
                try:
                    rep = make_sandbox(cfg, tmp).selftest()
                    bad = [c for c in rep["checks"] if not c["ok"] and c["critical"]]
                    why = next((c["detail"].strip().splitlines()[-1][:200] for c in bad if c.get("detail", "").strip()), "")
                    add("sandbox self-test", "ok" if rep["passed"] and rep["strong"] else "fail",
                        f"backend={rep['backend']} strong={rep['strong']} passed={rep['passed']}" + (f" failed: {[c['name'] for c in bad]}" if bad else "") + (f" - first error: {why}" if why else ""))
                except SandboxUnavailable as e:
                    add("sandbox self-test", "fail", str(e)[:200])
            except Exception as e:
                add("sandbox self-test", "fail", f"{type(e).__name__}: {e}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    # brain
    try:
        from .llm import LOCAL_PROVIDERS, PRESETS
        p = cfg.llm_provider
        if p == "mock":
            add("brain", "ok", "mock (offline demo)")
        elif p == "ollama":
            base = (cfg.llm_base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
            base = base if "://" in base else "http://" + base
            try:
                with urllib.request.urlopen(base + "/api/tags", timeout=2) as r:
                    names = [m.get("name") for m in json.loads(r.read().decode()).get("models", [])]
                add("brain", "ok" if cfg.llm_model in names else "warn", f"ollama at {base}; " + (f"model {cfg.llm_model} present" if cfg.llm_model in names else f"model {cfg.llm_model} not pulled (ollama pull {cfg.llm_model})"))
            except Exception as e:
                add("brain", "warn", f"ollama not reachable at {base}: {type(e).__name__}")
        elif p in LOCAL_PROVIDERS:
            add("brain", "warn", f"local provider '{p}' - run --test-llm to verify the server")
        else:
            env = cfg.llm_api_key_env or {"anthropic": "ANTHROPIC_API_KEY"}.get(p) or PRESETS.get(p, ("", "OPENAI_API_KEY", False))[1]
            add("brain", "ok" if os.environ.get(env) else "warn", f"{p}/{cfg.llm_model}: " + (f"{env} is set (run --test-llm for a live check)" if os.environ.get(env) else f"{env} is NOT set"))
    except Exception as e:
        add("brain", "warn", e)
    summary = {s: sum(1 for c in checks if c["status"] == s) for s in ("ok", "warn", "fail")}
    return {"ok": summary["fail"] == 0, "summary": summary, "checks": checks}


def format_report(rep: dict) -> str:
    tag = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}
    lines = [f"{tag[c['status']]} {c['name']}: {c['detail']}" for c in rep["checks"]]
    s = rep["summary"]
    lines.append(f"\n{s['ok']} ok, {s['warn']} warnings, {s['fail']} failures -> " + ("READY" if rep["ok"] else "NOT READY (fix the FAIL lines)"))
    return "\n".join(lines)
