"""Experiments and the Verifier.

Rule of the whole system: NOTHING becomes knowledge unless reality (the sandbox) says so.
  * web pages / papers / the LLM's own beliefs  ->  'lead' (unverified)
  * a claim becomes 'verified' only if ALL hold:
      1. a sandbox experiment ran cleanly and reported metrics via worldlab.report(...)
      2. the machine-checkable `expect` conditions hold on those metrics (checked HERE, outside the sandbox)
      3. the identical code, re-run in fresh sandboxes with different seeds, satisfies them again
      4. an independent critic audit finds the experiment is a genuine, falsifiable computation
  * if the conditions fail, the claim is stored as 'refuted' (negative knowledge is knowledge).
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from .llm import extract_json

RESULT_RE = re.compile(r"^SIRA_RESULT:\s*(\{.*\})\s*$", re.M)
OPS = ("<", "<=", ">", ">=", "==", "!=", "approx", "true", "false")


def parse_result(stdout: str):
    ms = RESULT_RE.findall(stdout or "")
    if not ms:
        return None
    try:
        d = json.loads(ms[-1])
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def validate_expect(expect) -> list:
    if not isinstance(expect, list) or not expect:
        raise ValueError("expect must be a non-empty list of {metric, op, value[, tol]} conditions")
    out = []
    for c in expect:
        if not isinstance(c, dict) or "metric" not in c or c.get("op") not in OPS:
            raise ValueError(f"bad condition {c!r}; ops allowed: {OPS}")
        if c["op"] not in ("true", "false"):
            if not isinstance(c.get("value"), (int, float)) or isinstance(c.get("value"), bool):
                raise ValueError(f"condition {c!r} needs a numeric 'value'")
        out.append({"metric": str(c["metric"]), "op": c["op"], "value": c.get("value"), "tol": c.get("tol")})
    return out


def check_expect(metrics: dict, expect: list) -> tuple[bool, list]:
    details, ok_all = [], True
    for c in expect:
        m, op, val = c["metric"], c["op"], c["value"]
        got = metrics.get(m)
        ok = False
        if op in ("true", "false"):
            ok = (got is True) if op == "true" else (got is False)
        elif isinstance(got, (int, float)) and not isinstance(got, bool) and math.isfinite(got):
            if op == "<":
                ok = got < val
            elif op == "<=":
                ok = got <= val
            elif op == ">":
                ok = got > val
            elif op == ">=":
                ok = got >= val
            elif op == "==":
                ok = got == val
            elif op == "!=":
                ok = got != val
            elif op == "approx":
                tol = c["tol"] if isinstance(c.get("tol"), (int, float)) else 1e-6
                ok = abs(got - val) <= tol * max(abs(val), 1e-300)
        details.append({"metric": m, "op": op, "value": val, "got": got, "ok": bool(ok)})
        ok_all &= ok
    return ok_all, details


CRITIC_SYSTEM = """ROLE=critic_audit
You are an adversarial scientific auditor. You receive a CLAIM, the machine-checked CONDITIONS, the experiment
CODE and the metrics from several independent runs. Decide whether the experiment is a GENUINE test.
Reject (valid=false) if any of these hold:
 - metrics are hard-coded, computed from the claim text, or do not come from a real computation;
 - the experiment cannot fail (tautology) or the thresholds are so loose that any outcome passes;
 - the metric does not actually measure what the claim says (wrong units, wrong quantity, cherry-picked range);
 - the code reads files/data that make the result trivially true.
Reply with ONE JSON object only: {"valid": true|false, "reason": "<one or two sentences>"}"""


class ExperimentRunner:
    def __init__(self, ctx):
        self.ctx = ctx

    def run(self, agent: str, hypothesis: str, code: str, seed: int | None = None, parent: int | None = None,
            files: dict | None = None, timeout: int | None = None) -> dict:
        ctx = self.ctx
        seed = int(seed if seed is not None else 1234)
        ctx.bus.emit("experiment", agent, f"running experiment: {hypothesis[:90]}")
        mods = ctx.workshop.module_files() if getattr(ctx, "workshop", None) else {}
        res = ctx.sandbox.run(code, files={**mods, **(files or {})}, seed=seed, timeout=timeout, station="ro")
        if res.exit_code != 0 and not res.timed_out and res.stderr.strip() and agent != "verifier":
            ctx.db.add_mistake("code", f"experiment crashed: {res.stderr.strip().splitlines()[-1][:220]}", "")
        metrics = parse_result(res.stdout)
        eid = ctx.db.add_experiment(
            agent=agent, hypothesis=hypothesis, code=code, stdout=res.stdout[-20000:], stderr=res.stderr[-8000:],
            exit_code=res.exit_code, timed_out=int(res.timed_out), result=json.dumps(metrics) if metrics else None,
            duration=res.duration, backend=res.backend, seed=seed, parent=parent)
        ctx.bus.emit("experiment", agent,
                     f"experiment #{eid}: exit={res.exit_code} timeout={res.timed_out} metrics={metrics}",
                     {"id": eid}, level="info" if res.ok else "warn")
        return {"experiment_id": eid, "exit_code": res.exit_code, "timed_out": res.timed_out,
                "duration_s": round(res.duration, 2), "metrics": metrics,
                "stdout_tail": res.stdout[-2500:], "stderr_tail": res.stderr[-1500:],
                "out_files": list(res.out_files.keys()),
                "hint": None if metrics else "No SIRA_RESULT line: end your code with worldlab.report(name=value, ...)"}


class Verifier:
    def __init__(self, ctx):
        self.ctx = ctx
        self.runner = ExperimentRunner(ctx)

    def verify(self, agent: str, claim: str, experiment_id: int, expect) -> dict:
        ctx = self.ctx
        claim = (claim or "").strip()
        if len(claim) < 8:
            return {"status": "rejected", "reason": "claim text too short"}
        try:
            expect = validate_expect(expect)
        except ValueError as e:
            return {"status": "rejected", "reason": str(e)}
        exp = ctx.db.experiment(int(experiment_id)) if isinstance(experiment_id, (int, float, str)) and str(experiment_id).lstrip("-").isdigit() else None
        if not exp:
            return {"status": "rejected", "reason": f"experiment {experiment_id} does not exist"}
        if exp["exit_code"] != 0 or exp["timed_out"]:
            return {"status": "rejected", "reason": "the experiment did not finish cleanly"}
        m0 = json.loads(exp["result"]) if exp["result"] else None
        if not m0:
            return {"status": "rejected", "reason": "experiment reported no metrics (worldlab.report)"}

        runs = [{"seed": exp["seed"], "metrics": m0}]
        ok0, det0 = check_expect(m0, expect)
        evidence = {"experiment_id": exp["id"], "expect": expect, "runs": runs, "checks": [det0]}

        ctx.bus.emit("verify", "verifier", f"verifying claim: {claim[:90]}")
        # ---- independent replications with different seeds -------------------------------------------------
        passes = [ok0]
        for i in range(ctx.cfg.replication_runs):
            seed = int(hashlib.sha256(f"{exp['id']}:{i}:{exp['seed']}".encode()).hexdigest()[:8], 16)
            r = self.runner.run("verifier", f"replication {i + 1} of #{exp['id']}: {exp['hypothesis'][:60]}",
                                exp["code"], seed=seed, parent=exp["id"])
            m = r["metrics"]
            runs.append({"seed": seed, "metrics": m, "experiment_id": r["experiment_id"]})
            if r["exit_code"] != 0 or r["timed_out"] or not m:
                passes.append(False)
                evidence["checks"].append([])
            else:
                ok, det = check_expect(m, expect)
                passes.append(ok)
                evidence["checks"].append(det)

        if not any(passes):
            status, reason = "refuted", "the measured metrics contradict the claim in every run"
        elif not all(passes):
            status, reason = "inconclusive", f"not reproducible: {sum(passes)}/{len(passes)} runs satisfied the conditions"
        else:
            status, reason = "verified", "all independent runs satisfy the conditions"

        # ---- adversarial audit (only needed to confirm a positive result) -----------------------------------
        if status == "verified":
            audit = self._audit(claim, expect, exp["code"], runs)
            evidence["audit"] = audit
            if not audit.get("valid"):
                status, reason = "inconclusive", "critic audit rejected the experiment: " + str(audit.get("reason"))[:300]

        evidence["reason"] = reason
        if status in ("refuted", "inconclusive") and getattr(ctx, "thinker", None):
            t = ctx.thinker.think("verifier", f"Why is this claim {status}: {claim[:200]}",
                                  {"reason": reason, "expect": expect, "runs": [r["metrics"] for r in runs], "code_head": exp["code"][:1500]})
            if t:
                evidence["thinking"] = {"decision": t["decision"], "analysis": t["analysis"]}
                ctx.db.add_mistake("verification", f"{status}: {claim[:120]} -> {t['decision'][:200]}", "")
        cid = ctx.db.add_claim(claim, status, source=f"experiment#{exp['id']} by {agent}", evidence=evidence)
        ctx.bus.emit("verify", "verifier", f"claim #{cid} -> {status.upper()}: {claim[:80]} ({reason})",
                     {"claim_id": cid, "status": status}, level="info" if status == "verified" else "warn")
        return {"status": status, "claim_id": cid, "reason": reason,
                "runs_passed": f"{sum(passes)}/{len(passes)}", "checks": evidence["checks"][0]}

    def _audit(self, claim, expect, code, runs) -> dict:
        user = json.dumps({"claim": claim, "conditions": expect, "code": code[:12000],
                           "runs": [{"seed": r["seed"], "metrics": r["metrics"]} for r in runs]}, default=str)
        for _ in range(3):
            txt = self.ctx.llm.chat(CRITIC_SYSTEM, [{"role": "user", "content": user}], max_tokens=600, temperature=0.0)
            try:
                d = extract_json(txt)
                if "valid" in d:
                    return {"valid": bool(d["valid"]), "reason": str(d.get("reason", ""))}
            except ValueError:
                continue
        return {"valid": False, "reason": "audit produced no parseable verdict"}
