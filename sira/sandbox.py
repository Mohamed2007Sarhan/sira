"""Sandbox: runs UNTRUSTED model-written Python in an isolated environment.

Backends (strongest first):
  docker      - no host FS, no caps, read-only root, memory/cpu/pids limits, network none|bridge
  bwrap       - Linux namespaces (bubblewrap): own mount/pid/net/ipc/uts/user ns, ro system, rlimits
  subprocess  - WEAK: rlimits + scrubbed env only. Does NOT hide the host filesystem.
                Refused unless cfg.allow_weak_sandbox is True.

Before any experiment the orchestrator runs `selftest()` which *proves* (by probing from inside)
that the host filesystem, the host environment, extra files and the network are unreachable.
"""
from __future__ import annotations

import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
MAX_CODE_BYTES = 400_000
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


class SandboxUnavailable(RuntimeError):
    pass


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration: float
    backend: str
    truncated: bool = False
    out_files: dict = field(default_factory=dict)   # small text files the code wrote to ./out

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


# --------------------------------------------------------------------------- process runner
def _dir_size(path, limit_files=20000):
    total, n = 0, 0
    stack = [str(path)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    n += 1
                    if n > limit_files:
                        return 1 << 62          # absurd number of files => treat as over quota
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def _run_capped(cmd, *, env, cwd, timeout, stdin_bytes, max_out, preexec, control, on_timeout=None,
                watch_dir=None, max_disk=0, extra_watch=None):
    """Run a command, capture capped output, kill the whole process group on timeout/stop."""
    t0 = time.time()
    p = subprocess.Popen(
        cmd, stdin=subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=cwd,
        preexec_fn=preexec, start_new_session=True,
    )
    if control:
        control.register_proc(p)
    bufs = {"out": bytearray(), "err": bytearray()}
    counters = {"total": 0, "trunc": False}
    hard_cap = max_out * 25

    def reader(stream, key):
        try:
            while True:
                chunk = stream.read1(65536) if hasattr(stream, "read1") else stream.read(65536)
                if not chunk:
                    break
                counters["total"] += len(chunk)
                room = max_out - len(bufs[key])
                if room > 0:
                    bufs[key].extend(chunk[:room])
                if len(chunk) > room:
                    counters["trunc"] = True
                if counters["total"] > hard_cap:       # output flood: kill it
                    counters["trunc"] = True
                    _kill_group(p)
                    break
        except Exception:
            pass

    threads = [threading.Thread(target=reader, args=(p.stdout, "out"), daemon=True),
               threading.Thread(target=reader, args=(p.stderr, "err"), daemon=True)]
    for t in threads:
        t.start()
    if stdin_bytes is not None:
        def feed():
            try:
                p.stdin.write(stdin_bytes)
                p.stdin.close()
            except Exception:
                pass
        threading.Thread(target=feed, daemon=True).start()

    timed_out = False
    last_disk_check = 0.0
    try:
        while True:
            try:
                p.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if (watch_dir or extra_watch) and time.time() - last_disk_check > 1.0:
                    last_disk_check = time.time()
                    over = (watch_dir and max_disk and _dir_size(watch_dir) > max_disk) or \
                           (extra_watch and _dir_size(extra_watch[0]) > extra_watch[1])
                    if over:
                        bufs["err"].extend(b"\n[SANDBOX] disk quota exceeded - process killed\n")
                        counters["trunc"] = True
                        _kill_group(p)
                        p.wait(timeout=10)
                        break
                if time.time() - t0 > timeout:
                    timed_out = True
                    if on_timeout:
                        on_timeout()
                    _kill_group(p)
                    p.wait(timeout=10)
                    break
                if control and control.stopped:
                    _kill_group(p)
                    p.wait(timeout=10)
                    break
    finally:
        for t in threads:
            t.join(timeout=3)
        for st in (p.stdout, p.stderr, p.stdin):
            try:
                if st:
                    st.close()
            except Exception:
                pass
        if control:
            control.unregister_proc(p)
    return (p.returncode if p.returncode is not None else -9,
            bytes(bufs["out"]).decode("utf-8", "replace"),
            bytes(bufs["err"]).decode("utf-8", "replace"),
            timed_out, time.time() - t0, counters["trunc"])


def _kill_group(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def _count_user_procs() -> int:
    uid = os.getuid()
    n = 0
    try:
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    if os.stat(f"/proc/{d}").st_uid == uid:
                        n += 1
                except OSError:
                    pass
    except OSError:
        return 0
    return n


def _rlimit_preexec(cfg, timeout):
    mem = int(cfg.sandbox_mem_mb) * 1024 * 1024
    fsize = int(cfg.sandbox_max_file_mb) * 1024 * 1024
    nproc = _count_user_procs() + int(cfg.sandbox_pids) + 16
    cpu = int(timeout) + 5

    def fn():
        for lim, val in ((resource.RLIMIT_CORE, 0), (resource.RLIMIT_AS, mem),
                         (resource.RLIMIT_FSIZE, fsize), (resource.RLIMIT_CPU, cpu),
                         (resource.RLIMIT_NOFILE, 512), (resource.RLIMIT_NPROC, nproc)):
            try:
                resource.setrlimit(lim, (val, val))
            except (ValueError, OSError):
                pass
    return fn


# --------------------------------------------------------------------------- base class
class Sandbox:
    name = "base"
    strong = True

    def __init__(self, cfg, run_dir, control=None):
        self.cfg, self.control = cfg, control
        self.run_dir = Path(run_dir)
        self.tmp_root = self.run_dir / "sandbox_tmp"
        self.data_dir = self.run_dir / "data_store"       # web downloads live here (read-only for experiments)
        self.station_dir = self.run_dir / "station"       # the agents' persistent computer (home dir, libs, tools, notes)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.station_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.station_dir, 0o777)
        self.runs = 0

    # subclasses implement
    def _exec(self, work: Path, timeout: int, network: str, env: dict, stdin_bytes, kill_name: str, entry: str, station: str):
        raise NotImplementedError

    @property
    def station_max(self) -> int:
        return int(self.cfg.station_mb) * 1024 * 1024

    @staticmethod
    def available(cfg) -> tuple[bool, str]:
        raise NotImplementedError

    def _base_env(self, seed):
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": "/tmp",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONPATH": "/work:/station/pylib",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        if seed is not None:
            env["SIRA_SEED"] = str(int(seed))
        return env

    def run(self, code: str, files: dict | None = None, stdin: str | bytes | None = None,
            timeout: int | None = None, seed: int | None = None, network: str | None = None,
            entry: str = "python", station: str = "none") -> SandboxResult:
        """entry: python|shell.  station: none|ro|rw  (persistent computer mounted at /station)."""
        if entry not in ("python", "shell") or station not in ("none", "ro", "rw"):
            raise ValueError("bad entry/station")
        if self.control:
            self.control.checkpoint()
        code_b = code.encode("utf-8")
        if len(code_b) > MAX_CODE_BYTES:
            raise ValueError("code too large")
        timeout = int(timeout or self.cfg.sandbox_timeout_s)
        network = network or self.cfg.sandbox_network
        if network not in ("none", "open"):
            raise ValueError("network must be none|open")
        work = self.tmp_root / uuid.uuid4().hex
        (work / "out").mkdir(parents=True)
        try:
            (work / ("main.py" if entry == "python" else "main.sh")).write_bytes(code_b)
            shutil.copy(PKG_DIR / "worldlab_lib.py", work / "worldlab.py")
            for name, content in (files or {}).items():
                if not SAFE_NAME.match(name) or name in ("main.py", "main.sh", "worldlab.py"):
                    raise ValueError(f"unsafe file name: {name!r}")
                data = content if isinstance(content, bytes) else str(content).encode("utf-8")
                (work / name).write_bytes(data)
            os.chmod(work, 0o777)
            os.chmod(work / "out", 0o777)
            for f in work.iterdir():
                if f.is_file():
                    os.chmod(f, 0o644)
            stdin_b = stdin.encode("utf-8") if isinstance(stdin, str) else stdin
            t_start = time.time() - 1
            kill_name = "sira-" + work.name[:12]
            code_, out, err, to, dur, trunc = self._exec(work, timeout, network, self._base_env(seed),
                                                          stdin_b, kill_name, entry, station)
            out_files = {}
            for i, f in enumerate(sorted((work / "out").iterdir())):
                if i >= 20:
                    break
                if f.is_file() and not f.is_symlink():
                    try:
                        out_files[f.name] = f.read_bytes()[:20_000].decode("utf-8", "replace")
                    except OSError:
                        pass
            self.runs += 1
            if station == "rw":
                note = self._enforce_station_quota(t_start)
                if note:
                    err += note
            return SandboxResult(code_, out, err, to, dur, self.name, trunc, out_files)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _enforce_station_quota(self, since: float) -> str:
        """If the persistent station exceeds its quota, delete files written in this run (largest first)."""
        if _dir_size(self.station_dir) <= self.station_max:
            return ""
        files = []
        for root, _, names in os.walk(self.station_dir):
            for n in names:
                fp = os.path.join(root, n)
                try:
                    st = os.lstat(fp)
                    if st.st_mtime >= since:
                        files.append((st.st_size, fp))
                except OSError:
                    pass
        removed = 0
        for size, fp in sorted(files, reverse=True):
            if _dir_size(self.station_dir) <= self.station_max:
                break
            try:
                os.unlink(fp)
                removed += 1
            except OSError:
                pass
        return f"\n[SANDBOX] station quota ({self.cfg.station_mb} MB) exceeded - removed {removed} file(s) written by this run\n"

    # ---------------------------------------------------------------- self test
    def selftest(self) -> dict:
        checks: list[dict] = []

        def add(name, ok, detail="", critical=True):
            checks.append({"name": name, "ok": bool(ok), "critical": critical, "detail": "" if ok else detail})

        host_dir = Path(tempfile.mkdtemp(prefix="sira_host_secret_"))
        sentinel = host_dir / "secret.txt"
        token = uuid.uuid4().hex
        sentinel.write_text(token)
        os.environ["SIRA_SELFTEST_SECRET"] = token
        try:
            probe = f'''
import os, json, socket
res = {{}}
def t(name, fn):
    try:
        res[name] = fn()
    except BaseException as e:
        res[name] = "blocked:" + type(e).__name__
t("read_host_file", lambda: open({str(sentinel)!r}).read())
t("env_secret", lambda: os.environ.get("SIRA_SELFTEST_SECRET"))
t("write_host_dir", lambda: open({str(host_dir / "written_by_sandbox.txt")!r}, "w").write("x"))
t("write_work", lambda: (open("out/probe.txt", "w").write("ok"), "ok")[1])
t("ifaces", lambda: sorted(n for _, n in socket.if_nameindex()))
t("hostname_file", lambda: len(open("/etc/shadow").read()))
print(json.dumps(res))
'''
            r = self.run(probe, timeout=30, network="none")
            try:
                res = json.loads(r.stdout.strip().splitlines()[-1])
            except Exception:
                res = {}
                add("probe_runs", False, f"probe failed: rc={r.exit_code} err={r.stderr[-300:]}")
            if res:
                add("hello/probe_runs", True)
                add("host_file_unreadable", res.get("read_host_file") != token,
                    "experiment code could READ a host file!" if res.get("read_host_file") == token else "")
                add("host_env_hidden", res.get("env_secret") in (None, ""),
                    "host environment variable leaked into sandbox")
                add("host_not_writable", not (host_dir / "written_by_sandbox.txt").exists(),
                    "experiment code WROTE into a host directory!")
                add("workspace_writable", res.get("write_work") == "ok", str(res.get("write_work")))
                ifs = res.get("ifaces")
                add("network_none_enforced", isinstance(ifs, list) and set(ifs) <= {"lo"},
                    f"interfaces visible: {ifs}")
                add("etc_shadow_unreadable", str(res.get("hostname_file", "")).startswith("blocked"),
                    "/etc/shadow readable", critical=True)
            # file written by probe must be harvested
            add("out_files_collected", "probe.txt" in r.out_files)

            r = self.run("while True: pass", timeout=3)
            add("timeout_enforced", r.timed_out and r.duration < 12, f"timed_out={r.timed_out} dur={r.duration:.1f}")

            r = self.run(f"x = bytearray({int(self.cfg.sandbox_mem_mb) * 3} * 1024 * 1024)\nprint('allocated')", timeout=30)
            add("memory_limit_enforced", ("allocated" not in r.stdout) and r.exit_code != 0,
                f"rc={r.exit_code} out={r.stdout[:40]!r}")

            r = self.run("import sys\nfor _ in range(200):\n    sys.stdout.write('x' * 100000)\n", timeout=30)
            add("output_capped", r.truncated and len(r.stdout) <= self.cfg.sandbox_max_output_bytes,
                f"len={len(r.stdout)} truncated={r.truncated}")

            r = self.run(
                "import subprocess, sys, time\nprocs=[]\n"
                f"try:\n    for i in range({int(self.cfg.sandbox_pids) * 4}):\n"
                "        procs.append(subprocess.Popen(['sleep','3']))\n"
                "    print('spawned', len(procs))\nexcept BaseException as e:\n    print('limited', len(procs), type(e).__name__)\n"
                "for p in procs:\n    p.kill()\n", timeout=40)
            add("process_limit_enforced", "limited" in r.stdout,
                f"{r.stdout.strip()[:80]} (needs Docker or a non-root user for kernel-enforced limits)",
                critical=False)

            r = self.run("import sys; print(sys.stdin.read().upper())", stdin="abc", timeout=10)
            add("stdin_works", r.stdout.strip() == "ABC", r.stdout[:30])
            r = self.run("import worldlab, os; print(worldlab.seed())", seed=7, timeout=10)
            add("worldlab_available", r.ok and r.stdout.strip() == "7", r.stderr[-200:])
        finally:
            os.environ.pop("SIRA_SELFTEST_SECRET", None)
            shutil.rmtree(host_dir, ignore_errors=True)
        passed = all(c["ok"] for c in checks if c["critical"])
        return {"backend": self.name, "strong": self.strong, "passed": passed, "checks": checks}


# --------------------------------------------------------------------------- docker
class DockerSandbox(Sandbox):
    name = "docker"
    strong = True

    @staticmethod
    def available(cfg):
        docker = shutil.which("docker")
        if not docker:
            return False, "docker not installed"
        try:
            r = subprocess.run([docker, "info"], capture_output=True, timeout=10)
            if r.returncode != 0:
                return False, "docker daemon not reachable"
            r = subprocess.run([docker, "image", "inspect", cfg.docker_image], capture_output=True, timeout=15)
            if r.returncode != 0:
                return False, (f"docker image '{cfg.docker_image}' missing - build it: "
                               f"python -m sira --build-sandbox")
        except Exception as e:
            return False, f"docker check failed: {e}"
        return True, "ok"

    def _exec(self, work, timeout, network, env, stdin_bytes, kill_name, entry, station):
        c = self.cfg
        cmd = ["docker", "run", "--rm", "-i", "--name", kill_name,
               "--network", "none" if network == "none" else "bridge",
               "--read-only", "--tmpfs", "/tmp:rw,size=256m",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--pids-limit", str(c.sandbox_pids),
               "--memory", f"{c.sandbox_mem_mb}m", "--memory-swap", f"{c.sandbox_mem_mb}m",
               "--cpus", str(c.sandbox_cpus), "--user", "65534:65534",
               "--ulimit", f"fsize={int(c.sandbox_max_file_mb) * 1024 * 1024}",
               "--ulimit", "core=0",
               "-v", f"{work}:/work:rw", "-v", f"{self.data_dir}:/data:ro", "-w", "/work"]
        if station != "none":
            cmd += ["-v", f"{self.station_dir}:/station:{station}"]
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += [c.docker_image] + (["python", "-u", "main.py"] if entry == "python" else ["bash", "main.sh"])
        host_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/tmp"),
                    "DOCKER_HOST": os.environ.get("DOCKER_HOST", "")}
        host_env = {k: v for k, v in host_env.items() if v}
        return _run_capped(
            cmd, env=host_env, cwd=str(work), timeout=timeout, stdin_bytes=stdin_bytes,
            max_out=c.sandbox_max_output_bytes, preexec=None, control=self.control,
            watch_dir=work, max_disk=c.sandbox_disk_mb * 1024 * 1024,
            extra_watch=(self.station_dir, self.station_max) if station == "rw" else None,
            on_timeout=lambda: subprocess.run(["docker", "kill", kill_name], capture_output=True, timeout=15))


# --------------------------------------------------------------------------- bubblewrap
class BwrapSandbox(Sandbox):
    name = "bwrap"
    strong = True

    @staticmethod
    def _python() -> str:
        return os.path.realpath(sys.executable)

    @classmethod
    def _base_args(cls, network: str, tmp_mb: int = 256):
        # root is an in-memory tmpfs (capped) - nothing written there ever reaches the host disk
        args = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
                "--size", str(32 * 1024 * 1024), "--tmpfs", "/"]
        if network == "open":
            args.append("--share-net")
        for d in ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32"):
            if os.path.islink(d):
                args += ["--symlink", os.readlink(d), d]
            elif os.path.isdir(d):
                args += ["--ro-bind", d, d]
        py = cls._python()
        extra = {os.path.dirname(os.path.dirname(py)), os.path.realpath(sys.base_prefix)}
        for p in sorted(extra):
            if p and not p.startswith("/usr") and p not in ("/", "/bin", "/lib") and os.path.isdir(p):
                args += ["--ro-bind", p, p]
        for f in ("/etc/ssl", "/etc/ca-certificates", "/etc/resolv.conf", "/etc/hosts",
                  "/etc/nsswitch.conf", "/etc/localtime", "/etc/alternatives"):
            if os.path.exists(f):
                args += ["--ro-bind", f, f]
        args += ["--proc", "/proc", "--dev", "/dev", "--size", str(tmp_mb * 1024 * 1024), "--tmpfs", "/tmp"]
        return args

    @classmethod
    def available(cls, cfg):
        if not shutil.which("bwrap"):
            return False, "bubblewrap (bwrap) not installed (apt install bubblewrap)"
        try:
            r = subprocess.run(cls._base_args("none") + [cls._python(), "-c", "print('ok')"],
                               capture_output=True, timeout=20)
            if r.returncode != 0 or b"ok" not in r.stdout:
                return False, f"bwrap cannot create namespaces here: {r.stderr.decode(errors='replace')[:200]}"
        except Exception as e:
            return False, f"bwrap check failed: {e}"
        return True, "ok"

    def _exec(self, work, timeout, network, env, stdin_bytes, kill_name, entry, station):
        c = self.cfg
        cmd = self._base_args(network, c.sandbox_tmp_mb)
        cmd += ["--bind", str(work), "/work", "--ro-bind", str(self.data_dir), "/data", "--chdir", "/work"]
        if station != "none":
            cmd += ["--bind" if station == "rw" else "--ro-bind", str(self.station_dir), "/station"]
        cmd += ["--clearenv"]
        for k, v in env.items():
            cmd += ["--setenv", k, v]
        cmd += ([self._python(), "-u", "main.py"] if entry == "python" else ["/usr/bin/bash", "main.sh"])
        return _run_capped(
            cmd, env={"PATH": "/usr/bin:/bin"}, cwd=str(work), timeout=timeout, stdin_bytes=stdin_bytes,
            max_out=c.sandbox_max_output_bytes, preexec=_rlimit_preexec(c, timeout), control=self.control,
            watch_dir=work, max_disk=c.sandbox_disk_mb * 1024 * 1024,
            extra_watch=(self.station_dir, self.station_max) if station == "rw" else None)


# --------------------------------------------------------------------------- weak fallback
class SubprocessSandbox(Sandbox):
    name = "subprocess"
    strong = False

    @staticmethod
    def available(cfg):
        return True, "ok (WEAK: no filesystem/network isolation)"

    def _exec(self, work, timeout, network, env, stdin_bytes, kill_name, entry, station):
        c = self.cfg
        env = dict(env)
        env["HOME"] = str(work)
        env["PYTHONPATH"] = f"{work}:{self.station_dir}/pylib"
        link = work / "data"
        try:
            link.symlink_to(self.data_dir, target_is_directory=True)
        except OSError:
            pass
        cmd = [os.path.realpath(sys.executable), "-u", "main.py"] if entry == "python" else ["bash", "main.sh"]
        return _run_capped(
            cmd, env=env, cwd=str(work), timeout=timeout, stdin_bytes=stdin_bytes,
            max_out=c.sandbox_max_output_bytes, preexec=_rlimit_preexec(c, timeout), control=self.control,
            watch_dir=work, max_disk=c.sandbox_disk_mb * 1024 * 1024)


# --------------------------------------------------------------------------- factory
def make_sandbox(cfg, run_dir, control=None) -> Sandbox:
    order = {"auto": ["docker", "bwrap", "subprocess"], "docker": ["docker"], "bwrap": ["bwrap"],
             "subprocess": ["subprocess"]}[cfg.sandbox_backend]
    classes = {"docker": DockerSandbox, "bwrap": BwrapSandbox, "subprocess": SubprocessSandbox}
    reasons = []
    for name in order:
        cls = classes[name]
        ok, why = cls.available(cfg)
        if not ok:
            reasons.append(f"{name}: {why}")
            continue
        if not cls.strong and not cfg.allow_weak_sandbox:
            reasons.append("subprocess: weak backend refused (set allow_weak_sandbox=true to accept the risk)")
            continue
        return cls(cfg, run_dir, control)
    raise SandboxUnavailable("No acceptable sandbox backend. " + " | ".join(reasons))
