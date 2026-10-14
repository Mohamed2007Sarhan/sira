"""Environment bootstrapper: at start-up SIRA looks at the machine, installs what the chosen algorithms need,
and adapts its own settings to the hardware. Installs are limited to the PyPI package names in algorithms.PACKAGES
(plus the sandbox science libraries) - never arbitrary names."""
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import re
import shutil
import subprocess
import sys

from .algorithms import ALGORITHMS, CORE_TRAINING, PACKAGES, SANDBOX_LIBS, algorithm_status
from .intake import detect_hardware

DIST = {"huggingface_hub": "huggingface-hub"}


def installed_version(pkg: str):
    try:
        return md.version(DIST.get(pkg, pkg))
    except md.PackageNotFoundError:
        return None


def cuda_runtime_version() -> str | None:
    """Highest CUDA version the installed NVIDIA driver supports (from nvidia-smi's header)."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15).stdout
        m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def torch_install_args(hw: dict, cuda_ver: str | None) -> list:
    """pip arguments for PyTorch matching this machine (CPU wheel is ~10x smaller than the default CUDA wheel)."""
    if not hw.get("cuda"):
        return ["torch", "--index-url", "https://download.pytorch.org/whl/cpu"]
    major, minor = (int(x) for x in (cuda_ver or "12.4").split("."))
    v = major * 10 + minor
    for tag, need in (("cu128", 128), ("cu126", 126), ("cu124", 124), ("cu121", 121), ("cu118", 118)):
        if v >= need:
            return ["torch", "--index-url", f"https://download.pytorch.org/whl/{tag}"]
    return ["torch"]


def probe(sandbox_libs: dict | None = None) -> dict:
    hw = detect_hardware()
    pk = {p: installed_version(p) for p in PACKAGES}
    try:
        disk = shutil.disk_usage(os.getcwd())
        disk_free = round(disk.free / 1e9, 1)
    except OSError:
        disk_free = None
    try:
        import tkinter  # noqa: F401
        tk = True
    except Exception:
        tk = False
    return {"python": platform.python_version(), "os": f"{platform.system()} {platform.release()}", "machine": platform.machine(),
            "hw": hw, "cuda_driver": cuda_runtime_version(), "packages": pk, "tkinter": tk, "disk_free_gb": disk_free,
            "bwrap": bool(shutil.which("bwrap")), "docker": bool(shutil.which("docker")), "sandbox_libs": sandbox_libs or {},
            "algorithms": algorithm_status(pk, hw.get("cuda", False))}


def wanted_packages(cfg, hw: dict) -> list:
    want = list(CORE_TRAINING)
    if hw.get("cuda") and platform.system() == "Linux":
        want.append("bitsandbytes")          # QLoRA / 8-bit optimizers
    want.append("sentencepiece")
    seen = []
    for p in want:
        if p not in seen:
            seen.append(p)
    return seen


def pip_install(args: list, timeout: int = 1800) -> tuple[bool, str]:
    """pip install into the running interpreter. On 'externally-managed-environment' (Debian/Ubuntu system Python, PEP 668)
    retry once with --break-system-packages (--user when not in a venv, so system files stay untouched)."""
    base = [sys.executable, "-m", "pip", "install", "--no-input", "--disable-pip-version-check"]
    in_venv = sys.prefix != sys.base_prefix

    def go(extra):
        r = subprocess.run(base + extra + args, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout + r.stderr)[-1500:]
    try:
        ok, out = go([])
        if not ok and "externally-managed-environment" in out:
            ok, out = go(["--break-system-packages"] + ([] if in_venv else ["--user"]))
        if ok:                                   # make fresh installs visible to THIS process (user-site may not exist at start-up)
            import importlib
            import site
            us = site.getusersitepackages()
            if os.path.isdir(us) and us not in sys.path:
                sys.path.append(us)
            importlib.invalidate_caches()
        return ok, out
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def adapt_config(cfg, hw: dict) -> dict:
    """Tune settings to this machine. Returns {field: (old, new)} for everything changed (the GUI shows it)."""
    changes = {}

    def setv(k, v):
        old = getattr(cfg, k)
        if old != v:
            setattr(cfg, k, v)
            changes[k] = (old, v)
    cpus = hw.get("cpu_count") or 1
    setv("parallel_experiments", max(1, min(cfg.parallel_experiments, max(1, cpus // 2))))
    ram = hw.get("ram_gb")
    if ram:
        setv("sandbox_mem_mb", int(max(256, min(cfg.sandbox_mem_mb, ram * 1000 * 0.25))))
    try:
        free = shutil.disk_usage(os.getcwd()).free / 1e6
        setv("station_mb", int(max(64, min(cfg.station_mb, free * 0.1))))
        setv("sandbox_disk_mb", int(max(64, min(cfg.sandbox_disk_mb, free * 0.1))))
    except OSError:
        pass
    if hw.get("cuda") and hw["gpus"]:
        v = hw["gpus"][0]["vram_gb"]
        mb, ga, seq = (1, 16, 1024) if v < 12 else (2, 8, 1024) if v < 24 else (4, 4, 2048) if v < 80 else (8, 2, 4096)
        setv("micro_batch", mb)
        setv("grad_accum", ga)
        setv("max_seq_len", min(max(cfg.max_seq_len, 512), seq) if cfg.max_seq_len > seq else cfg.max_seq_len)
    else:
        setv("micro_batch", 1 if cfg.micro_batch > 1 else cfg.micro_batch)
    return changes


class Bootstrap:
    def __init__(self, cfg, bus):
        self.cfg, self.bus = cfg, bus

    def say(self, msg, level="info"):
        self.bus.emit("bootstrap", "bootstrap", msg, level=level)
        self.bus.status("bootstrap", msg[:90], role="bootstrap", parent="orchestrator")

    def run(self) -> dict:
        cfg = self.cfg
        rep = probe()
        hw = rep["hw"]
        self.say(f"machine: {rep['os']} {rep['machine']}, python {rep['python']}, CPUs {hw.get('cpu_count')}, RAM {hw.get('ram_gb')} GB, "
                 f"GPU: {[g['name'] + ' ' + str(g['vram_gb']) + 'GB' for g in hw['gpus']] or 'none'}, free disk {rep['disk_free_gb']} GB")
        want = wanted_packages(cfg, hw)
        missing = [p for p in want if not rep["packages"].get(p)]
        rep.update({"wanted": want, "missing": missing, "installed_now": [], "failed": {}})
        if missing and cfg.auto_install:
            self.say(f"installing missing packages: {missing}")
            if "torch" in missing:
                ok, out = pip_install(torch_install_args(hw, rep["cuda_driver"]))
                (rep["installed_now"] if ok else rep["failed"].__setitem__("torch", out[-300:]) or []).append("torch") if ok else None
                missing = [p for p in missing if p != "torch"]
            if missing:
                ok, out = pip_install(missing)
                if ok:
                    rep["installed_now"] += missing
                else:                                    # one by one, so a single bad package does not block the rest
                    for p in missing:
                        ok1, out1 = pip_install([p])
                        (rep["installed_now"].append(p) if ok1 else rep["failed"].__setitem__(p, out1[-300:]))
            rep["packages"] = {p: installed_version(p) for p in PACKAGES}
            rep["missing"] = [p for p in want if not rep["packages"].get(p)]
        for p in rep["missing"]:
            self.say(f"MISSING: {p} - {PACKAGES[p][0]}" + ("" if cfg.auto_install else "  (auto_install is off)"), "warn")
        if rep["failed"]:
            for k, v in rep["failed"].items():
                self.say(f"could not install {k}: {v.strip().splitlines()[-1][:200] if v.strip() else 'unknown error'}", "error")
        rep["algorithms"] = algorithm_status(rep["packages"], hw.get("cuda", False))
        rep["usable_algorithms"] = [a["name"] for a in rep["algorithms"] if a["usable"]]
        if cfg.auto_adapt:
            rep["adapted"] = adapt_config(cfg, hw)
            for k, (o, n) in rep["adapted"].items():
                self.say(f"adapted {k}: {o} -> {n}")
        self.say(f"usable algorithms: {', '.join(rep['usable_algorithms'])}")
        self.bus.status("bootstrap", "done", role="bootstrap", parent="orchestrator")
        return rep

    def sandbox_libs(self, sandbox) -> dict:
        """Check the science libraries INSIDE the sandbox; install missing ones into the agents' persistent /station/pylib."""
        res = {}
        for lib in SANDBOX_LIBS:
            r = sandbox.run(f"import {lib}; print({lib}.__version__)", station="ro", timeout=60, network="none")
            res[lib] = r.stdout.strip() if r.ok else None
        miss = [k for k, v in res.items() if not v]
        if miss and self.cfg.auto_install:
            self.say(f"installing sandbox libraries into /station/pylib: {miss}")
            cmd = "mkdir -p /station/pylib && python3 -m pip install --no-input --no-cache-dir --disable-pip-version-check --target /station/pylib " + " ".join(miss) + " 2>&1 | tail -3"
            r = sandbox.run(cmd, entry="shell", station="rw", network="open", timeout=900)
            for lib in miss:
                r2 = sandbox.run(f"import {lib}; print({lib}.__version__)", station="ro", timeout=60, network="none")
                res[lib] = r2.stdout.strip() if r2.ok else None
        return res
