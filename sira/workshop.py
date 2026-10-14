"""Workshop: how the agents EXTEND THEMSELVES while running - without touching the immutable core.

  create_tool   - new Python tool (def run(args)->dict), smoke-tested in the sandbox, usable by every agent
  create_world  - a new simulated reality with its own laws (module exposing build(**params) -> worldlab.World)
  station_*     - the agents' persistent computer (files, shell, pip installs into /station/pylib)
  math_verify   - step-by-step checking of derivations (sympy or numeric), failures become 'mistakes'
  concepts      - verified claims synthesised into scientific concepts

Everything an agent creates is code that only ever runs INSIDE the sandbox. Optional human approval
(cfg.approve_new_tools) puts new tools/worlds in 'pending' until you enable them in the GUI.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,30}$")
PKG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*(\[[A-Za-z0-9,_\-]+\])?([=<>!~]=?[A-Za-z0-9.*]+)?$")
MAX_SRC = 100_000
RESERVED = {"worldlab", "main", "inputs", "args", "sys", "os", "json", "math", "random", "time"}

TOOL_RUNNER = '''
import json, sys
import tool_{name} as _t
_args = json.load(open("args.json"))
_res = _t.run(_args)
json.dumps(_res)
print("SIRA_TOOL_OUT: " + json.dumps(_res))
'''

WORLD_SMOKE = '''
import json, worldlab, math
w = worldlab.load_world("{name}")
assert isinstance(w, worldlab.World), "build() must return a worldlab.World"
tr = w.run(30, 0.01)
assert len(tr["states"]) == 31
drift = {{k: worldlab.relative_drift(v) for k, v in tr["invariants"].items() if v}}
bad = [k for k, v in drift.items() if not math.isfinite(v)]
assert not bad, "non-finite invariants: %s" % bad
worldlab.report(ok=True, n_invariants=len(drift), **{{"drift_" + k: v for k, v in drift.items()}})
'''


def _safe_join(root: Path, rel: str) -> Path:
    """Resolve rel inside root; refuse anything that escapes (including through symlinks made by sandbox code)."""
    p = (root / rel.lstrip("/")).resolve()
    r = root.resolve()
    if p != r and r not in p.parents:
        raise ValueError("path escapes the station")
    return p


class Workshop:
    def __init__(self, ctx):
        self.ctx = ctx

    # ------------------------------------------------------------------ module injection
    def module_files(self) -> dict:
        """Active agent-made tools/worlds, injected into every sandbox run (so experiments can import them)."""
        files = {}
        for t in self.ctx.db.tools("active")[:20]:
            files[f"tool_{t['name']}.py"] = t["code"]
        for w in self.ctx.db.worlds("active")[:20]:
            files[f"world_{w['name']}.py"] = w["source"] if "source" in w else ""
        return files

    # ------------------------------------------------------------------ tools
    def create_tool(self, agent, name, description, code, example_args, expect=None) -> dict:
        db, cfg = self.ctx.db, self.ctx.cfg
        if not isinstance(name, str) or not NAME_RE.match(name) or name in RESERVED:
            return {"error": "name must match ^[a-z][a-z0-9_]{2,30}$ and not be reserved"}
        if not isinstance(code, str) or "def run(" not in code or len(code) > MAX_SRC:
            return {"error": f"code must define `def run(args: dict) -> dict` and be < {MAX_SRC} chars"}
        if not isinstance(example_args, dict):
            return {"error": "example_args must be an object used for the smoke test"}
        files = {f"tool_{name}.py": code, "args.json": json.dumps(example_args)}
        files.update({k: v for k, v in self.module_files().items() if k != f"tool_{name}.py"})
        r = self.ctx.sandbox.run(TOOL_RUNNER.format(name=name), files=files, station="ro", timeout=60, network="none")
        out = self._tool_out(r.stdout)
        if not r.ok or out is None:
            self.ctx.db.add_mistake("tool", f"tool '{name}' failed its smoke test: {(r.stderr or r.stdout)[-300:]}", "")
            return {"status": "rejected", "reason": "smoke test failed", "stderr": r.stderr[-800:], "stdout": r.stdout[-400:]}
        if isinstance(expect, dict):
            for k, v in expect.items():
                if out.get(k) != v:
                    return {"status": "rejected", "reason": f"smoke test output {k}={out.get(k)!r}, expected {v!r}", "output": out}
        status = "pending" if cfg.approve_new_tools else "active"
        db.put_tool(name, str(description)[:300], code, example_args, status, agent)
        self.ctx.bus.emit("tool_created", agent, f"new tool '{name}' ({status}): {str(description)[:100]}", {"name": name}, level="warn" if status == "pending" else "info")
        return {"status": status, "name": name, "smoke_output": out,
                "note": "awaiting human approval in the GUI" if status == "pending" else "available to all agents via use_tool and import tool_" + name}

    @staticmethod
    def _tool_out(stdout):
        for line in reversed((stdout or "").splitlines()):
            if line.startswith("SIRA_TOOL_OUT:"):
                try:
                    d = json.loads(line[len("SIRA_TOOL_OUT:"):])
                    return d if isinstance(d, dict) else {"value": d}
                except ValueError:
                    return None
        return None

    def use_tool(self, agent, name, args) -> dict:
        t = self.ctx.db.tool(str(name))
        if not t or t["status"] != "active":
            return {"error": f"no active tool '{name}'. Use list_tools."}
        files = {"args.json": json.dumps(args if isinstance(args, dict) else {})}
        files.update(self.module_files())
        r = self.ctx.sandbox.run(TOOL_RUNNER.format(name=name), files=files, station="ro", timeout=90, network="none")
        self.ctx.db.bump_tool(name)
        out = self._tool_out(r.stdout)
        if out is None:
            return {"error": "tool failed", "stderr": r.stderr[-800:], "stdout": r.stdout[-300:]}
        return {"output": out}

    def list_tools(self) -> dict:
        return {"tools": [{"name": t["name"], "description": t["description"], "example_args": json.loads(t["example_args"] or "{}"),
                           "status": t["status"], "uses": t["uses"]} for t in self.ctx.db.tools()]}

    # ------------------------------------------------------------------ worlds
    def create_world(self, agent, name, domain, description, source) -> dict:
        import importlib.util  # noqa: F401  (kept for clarity: the module only ever imports inside the sandbox)
        db, cfg = self.ctx.db, self.ctx.cfg
        from . import worldlab_lib
        if not isinstance(name, str) or not NAME_RE.match(name) or name in worldlab_lib.BUILTIN_WORLDS or name in RESERVED:
            return {"error": "bad or reserved world name (^[a-z][a-z0-9_]{2,30}$)"}
        if not isinstance(source, str) or "def build(" not in source or len(source) > MAX_SRC:
            return {"error": "source must define `def build(**params) -> worldlab.World` (use worldlab.World or worldlab.LawWorld)"}
        files = {f"world_{name}.py": source}
        files.update({k: v for k, v in self.module_files().items() if k != f"world_{name}.py"})
        r = self.ctx.sandbox.run(WORLD_SMOKE.format(name=name), files=files, station="ro", timeout=60, network="none")
        from .verifier import parse_result
        m = parse_result(r.stdout)
        if not r.ok or not m or m.get("ok") is not True:
            db.add_mistake("world", f"world '{name}' failed its smoke test: {(r.stderr or r.stdout)[-300:]}", "")
            return {"status": "rejected", "reason": "smoke test failed", "stderr": r.stderr[-900:]}
        status = "pending" if cfg.approve_new_tools else "active"
        db.put_world(name, str(domain)[:60], str(description)[:400], source, status, agent)
        self.ctx.bus.emit("world_created", agent, f"new world '{name}' [{domain}] ({status}): {str(description)[:100]}", {"name": name},
                          level="warn" if status == "pending" else "info")
        return {"status": status, "name": name, "invariant_drifts": {k: v for k, v in m.items() if k.startswith("drift_")},
                "use": f"worldlab.load_world('{name}') inside any experiment"}

    def list_worlds(self) -> dict:
        from . import worldlab_lib
        built = [{"name": k, **v, "kind": "built-in"} for k, v in worldlab_lib.list_builtin_worlds().items()]
        custom = [{"name": w["name"], "domain": w["domain"], "description": w["description"], "status": w["status"],
                   "kind": "agent-made"} for w in self.ctx.db.worlds()]
        return {"worlds": built + custom,
                "hint": "terra_incognita = empty world with NO laws: LawWorld(name).set_state(...).add_law(...).add_invariant(...). "
                        "If a science you need has no world, create one with create_world."}

    # ------------------------------------------------------------------ math lab
    def math_verify(self, agent, steps, symbols=None, mode="chain") -> dict:
        if not isinstance(steps, list) or not steps or len(steps) > 60 or any(not isinstance(s, str) or len(s) > 400 for s in steps):
            return {"error": "steps: list of 1..60 strings (<=400 chars each)"}
        symbols = [s for s in (symbols or ["x"]) if isinstance(s, str) and re.match(r"^[A-Za-z_]\w{0,8}$", s)][:6] or ["x"]
        if mode not in ("chain", "identity"):
            return {"error": "mode must be chain|identity"}
        code = ("import json, worldlab\na = json.load(open('args.json'))\n"
                "print('SIRA_TOOL_OUT: ' + json.dumps(worldlab.math_verify(a['steps'], a['symbols'], a['mode'])))\n")
        r = self.ctx.sandbox.run(code, files={"args.json": json.dumps({"steps": steps, "symbols": symbols, "mode": mode})},
                                 timeout=60, network="none")
        out = self._tool_out(r.stdout)
        if out is None:
            return {"error": "math check crashed", "stderr": r.stderr[-500:]}
        if not out["ok"]:
            bad = out["details"][out["first_bad"]]
            self.ctx.db.add_mistake("math", f"step {out['first_bad']} '{bad['text']}' is wrong ({json.dumps(bad.get('counterexample'))[:200]})", "")
            self.ctx.bus.emit("math", agent, f"derivation error at step {out['first_bad']}: {bad['text'][:80]}", level="warn")
        return out

    # ------------------------------------------------------------------ station (the agents' computer)
    def station_run(self, agent, command=None, python=None, network=False, timeout=120) -> dict:
        if bool(command) == bool(python):
            return {"error": "give exactly one of: command (shell) | python (code)"}
        net = "open" if (network and self.ctx.cfg.sandbox_network == "open") else "none"
        r = self.ctx.sandbox.run(command or python, entry="shell" if command else "python", station="rw",
                                 timeout=min(int(timeout), 600), network=net, files=self.module_files())
        self.ctx.bus.emit("station", agent, f"$ {(command or python)[:120]} -> exit {r.exit_code}")
        if not r.ok and r.stderr:
            self.ctx.db.add_mistake("station", f"command failed: {(command or python)[:120]} :: {r.stderr[-200:]}", "")
        return {"exit_code": r.exit_code, "timed_out": r.timed_out, "stdout": r.stdout[-4000:], "stderr": r.stderr[-1500:],
                "network": net}

    def station_install(self, agent, packages) -> dict:
        cfg = self.ctx.cfg
        if not cfg.allow_agent_installs:
            return {"error": "package installs are disabled by the human (allow_agent_installs=false)"}
        if cfg.sandbox_network != "open":
            return {"error": "the sandbox network is 'none'. Ask the human to set sandbox_network=open in Setup/Network, "
                             "or use web_download to fetch pure-python wheels into /data and install them with station_run."}
        if not isinstance(packages, list) or not packages or len(packages) > 8 or not all(isinstance(p, str) and PKG_RE.match(p) for p in packages):
            return {"error": "packages: list of 1..8 valid pip requirement strings"}
        cmd = "mkdir -p /station/pylib && python3 -m pip install --no-input --no-cache-dir --disable-pip-version-check --target /station/pylib " + \
              " ".join(f"'{p}'" for p in packages) + " 2>&1 | tail -15"
        r = self.ctx.sandbox.run(cmd, entry="shell", station="rw", network="open", timeout=600)
        self.ctx.bus.emit("station", agent, f"pip install {packages} -> exit {r.exit_code}", level="info" if r.ok else "warn")
        return {"exit_code": r.exit_code, "output": r.stdout[-2000:], "importable_from": "/station/pylib (already on PYTHONPATH)"}

    def station_files(self, agent, action, path="", content=None) -> dict:
        root = self.ctx.sandbox.station_dir
        try:
            if action == "list":
                base = _safe_join(root, path or "")
                if not base.exists():
                    return {"error": "no such directory"}
                items = []
                for p in sorted(base.rglob("*"))[:300]:
                    if p.is_symlink():
                        continue
                    items.append({"path": str(p.relative_to(root)), "bytes": p.stat().st_size if p.is_file() else None})
                return {"items": items}
            p = _safe_join(root, path)
            if action == "read":
                if not p.is_file() or p.is_symlink():
                    return {"error": "not a regular file"}
                fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(fd, "rb") as f:
                    return {"path": path, "content": f.read(60_000).decode("utf-8", "replace")}
            if action == "write":
                if not isinstance(content, str) or len(content) > 1_000_000:
                    return {"error": "content must be a string < 1 MB"}
                p.parent.mkdir(parents=True, exist_ok=True)
                if p.is_symlink():
                    return {"error": "refusing to write through a symlink"}
                p.write_text(content, encoding="utf-8")
                os.chmod(p, 0o666)
                return {"written": path, "bytes": len(content.encode())}
        except (ValueError, OSError) as e:
            return {"error": str(e)}
        return {"error": "action must be list|read|write"}

    # ------------------------------------------------------------------ concepts & mistakes
    def record_concept(self, agent, title, summary, claim_ids) -> dict:
        ids = []
        for c in claim_ids if isinstance(claim_ids, list) else []:
            try:
                ids.append(int(c))
            except (TypeError, ValueError):
                pass
        verified = {c["id"] for c in self.ctx.db.claims("verified", 100000)}
        ids = [i for i in ids if i in verified]
        if not ids:
            return {"error": "a concept must cite at least one VERIFIED claim id (concepts are built from verified evidence only)"}
        cid = self.ctx.db.add_concept(str(title)[:120], str(summary)[:2500], ids)
        self.ctx.bus.emit("concept", agent, f"CONCEPT #{cid}: {str(title)[:100]} (from claims {ids})")
        return {"concept_id": cid, "claims": ids}

    def record_mistake(self, agent, area, description, fix="") -> dict:
        mid = self.ctx.db.add_mistake(str(area)[:30], str(description)[:600], str(fix)[:600])
        return {"mistake_id": mid}
