"""Maestro layers.

  GrandMaestro    - owns the goal; plans each cycle (research topics, whether/how to train).
  ResearchMaestro - runs Reader -> Hypothesizer -> parallel Experimenters (+Verifier) -> Critic.
  TrainingMaestro - (trainer.py) intake, recipe, data, training, evaluation of generations.
  SafetyMaestro   - integrity of the immutable core, sandbox self-tests, human directives.
Nobody except RunControl can stop the run or change a budget; agents have no handle to it.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from .agents import Agent
from .control import Stopped
from .llm import extract_json, LLMError

PKG = Path(__file__).resolve().parent
IMMUTABLE = ["challenges.py", "quest.py", "society.py", "thinking.py", "cla_model.py", "bootstrap.py", "workshop.py", "backup.py", "brainmap.py", "evalcore.py", "verifier.py", "sandbox.py", "control.py", "worldlab_lib.py", "db.py", "web.py",
             "llm.py", "agents.py", "maestros.py", "trainer.py", "modelworker.py", "orchestrator.py"]

HYPER_BOUNDS = {
    "learning_rate": (1e-6, 5e-4, float), "epochs": (0.5, 3.0, float), "lora_r": (8, 64, int),
    "samples_per_problem": (2, 8, int), "keep_per_problem": (1, 4, int), "train_problems": (60, 600, int),
    "gen_temperature": (0.4, 1.2, float), "research_sft_fraction": (0.0, 0.3, float),
}


def clamp_hyper(h) -> dict:
    out = {}
    if not isinstance(h, dict):
        return out
    for k, (lo, hi, typ) in HYPER_BOUNDS.items():
        if k in h:
            try:
                v = typ(h[k])
                out[k] = max(lo, min(hi, v))
            except (TypeError, ValueError):
                pass
    if "lora_r" in out:
        out["lora_r"] = min((8, 16, 32, 64), key=lambda x: abs(x - out["lora_r"]))
    return out


GRAND_SYSTEM = """ROLE=grand_maestro
You are the Grand Maestro of SIRA, an autonomous research-and-self-improvement system. You own the GOAL and plan every cycle.
You command: the Research Maestro (agents that search the open web for LEADS and test them by sandbox experiments),
the Training Maestro (improves the target model by rejection-sampling fine-tuning on VERIFIED solutions), the
Evaluation core (hidden fresh problems) and the Safety Maestro. The human can see everything and may send directives.
Principle: nothing counts as knowledge unless an experiment confirmed it.
Reply with ONE JSON object only:
{"research_topics": [up to 3 short topics for experiments that could help reach the goal],
 "math_topics": [up to 2 equations/derivations/proofs worth checking step by step],
 "train": true|false,   (false only if more research should happen before the next training run)
 "rationale": "<two sentences>"}"""


def _clip(s, n):
    return s if len(s) <= n else s[:n] + "..."


class GrandMaestro:
    name = "grand_maestro"

    def __init__(self, ctx):
        self.ctx = ctx

    def plan(self, cycle: int, directives: list, weak_domains: list) -> dict:
        c = self.ctx
        c.bus.status(self.name, "planning", role="grand_maestro")
        gens = c.db.generations()
        state = {
            "goal": c.cfg.goal_text, "target_score": c.cfg.target_score, "cycle": cycle,
            "generations": [{"id": g["id"], "score": g["score"], "accepted": bool(g["accepted"]), "notes": g["notes"][:120],
                             "by_domain": {k: round(v["rate"], 2) for k, v in g["detail"].get("by_domain", {}).items()}}
                            for g in gens[-8:]],
            "weak_domains": weak_domains,
            "verified_claims": [x["text"][:140] for x in c.db.claims("verified", 12)],
            "refuted_claims": [x["text"][:140] for x in c.db.claims("refuted", 6)],
            "open_leads": len(c.db.claims("lead", 500)),
            "recent_lessons": [x["text"][:200] for x in c.db.lessons(5)],
            "budget": {k: [round(v[0], 3), v[1]] for k, v in c.control.budget_status().items() if k != "tokens"},
            "human_directives": directives,
            "problem_types": c.evalcore.public_examples(5),
            "recent_mistakes": [m["description"][:150] for m in c.db.mistakes(5)],
            "pi_proposed_topics": c.db.get_meta("pi_topics", []),
            "red_team": (c.db.get_meta("red_team_findings") or {}).get("thought"),
            "failure_notes": c.db.get_meta("failure_notes", [])[-3:], "human_baseline": c.cfg.human_baseline,
            "concepts": [x["title"] for x in c.db.concepts(8)],
            "self_made": {"tools": [t["name"] for t in c.db.tools("active")], "worlds": [w_["name"] for w_ in c.db.worlds("active")]},
        }
        plan = {"research_topics": [], "math_topics": [], "train": True, "rationale": "default plan"}
        try:
            txt = c.llm.chat(GRAND_SYSTEM, [{"role": "user", "content": json.dumps(state, default=str)}], max_tokens=1200)
            d = extract_json(txt)
            topics = [str(t)[:200] for t in d.get("research_topics", []) if isinstance(t, str) and t.strip()][:3]
            maths = [str(t)[:200] for t in d.get("math_topics", []) if isinstance(t, str) and t.strip()][:2]
            plan = {"research_topics": topics, "math_topics": maths, "train": bool(d.get("train", True)),
                    "rationale": str(d.get("rationale", ""))[:400]}
        except (ValueError, KeyError):
            c.bus.emit("plan", self.name, "unparseable plan, using default", level="warn")
        c.bus.emit("plan", self.name, f"cycle {cycle} plan: topics={plan['research_topics']} train={plan['train']} - {plan['rationale']}")
        c.bus.status(self.name, "idle", role="grand_maestro")
        return plan


class ResearchMaestro:
    name = "research_maestro"

    def __init__(self, ctx):
        self.ctx = ctx
        self._n = 0
        self._lock = threading.Lock()

    def _agent(self, role: str) -> Agent:
        with self._lock:
            self._n += 1
            n = self._n
        return Agent(self.ctx, f"{role}-{n}", role, parent=self.name)

    def research(self, topic: str, directive_text: str = "") -> dict:
        c = self.ctx
        c.bus.status(self.name, f"topic: {topic[:50]}", role="research_maestro", parent="grand_maestro")
        c.bus.emit("research", self.name, f"RESEARCH TOPIC: {topic}")
        extra = f"\nHuman directives: {directive_text}" if directive_text else ""
        th = c.thinker.think("researcher", f"How should we investigate: {topic}?",
                             {"goal": c.cfg.goal_text, "relevant_memory": [r["text"][:160] for r in c.db.search_claims(topic, 6)],
                              "recent_mistakes": [m["description"][:140] for m in c.db.mistakes(5)], "red_team": (c.db.get_meta("red_team_findings") or {}).get("thought")})
        goal = f"Overall goal of the system: {c.cfg.goal_text}.{extra}\n{c.thinker.brief(th)}"

        reader_report = self._agent("reader").run(f"{goal}\nTOPIC: {topic}\nFind leads and datasets, record the testable ones, then finish with a short summary.", 10)
        mem = [r["text"][:200] for r in c.db.search_claims(topic, 8)]
        hyp_report = self._agent("hypothesizer").run(
            f"{goal}\nTOPIC: {topic}\nReader summary: {reader_report[:2500]}\nRelevant memory: {json.dumps(mem)}\n"
            f"Propose up to 3 falsifiable quantitative hypotheses (final report = JSON).", 6)
        hyps = self._parse_hyps(hyp_report)[:3]
        c.bus.emit("research", self.name, f"{len(hyps)} hypotheses to test")

        results = [None] * len(hyps)

        def work(i, h):
            try:
                pm = c.thinker.think("sandbox", f"Pre-mortem for testing: {h['claim']}", {"plan": h.get("plan", ""), "limits": {"timeout_s": c.cfg.sandbox_timeout_s, "mem_mb": c.cfg.sandbox_mem_mb}})
                results[i] = self._agent("experimenter").run(
                    f"{goal}\n{c.thinker.brief(pm)}\nTOPIC: {topic}\nHYPOTHESIS TO TEST: {h['claim']}\nSuggested plan: {h.get('plan', '')}\n"
                    f"Test it in the sandbox, then verify_claim with machine-checkable conditions, then finish.")
            except Stopped:
                results[i] = "[stopped]"
            except LLMError as e:
                results[i] = f"[LLM error: {e}]"

        threads = [threading.Thread(target=work, args=(i, h), daemon=True) for i, h in enumerate(hyps)]
        for t in threads:
            t.start()
        for t in threads:
            while t.is_alive():
                t.join(0.5)
                c.control.checkpoint()

        recent = [{"status": x["status"], "text": x["text"][:160]} for x in c.db.claims(None, 8)]
        critic = self._agent("critic").run(
            f"TOPIC: {topic}\nExperimenter reports: {json.dumps([_clip(str(r), 800) for r in results])}\n"
            f"Latest claims in memory: {json.dumps(recent)}\nAttack these results; finish with weaknesses and next experiments.", 8)
        verified = [{"id": x["id"], "text": x["text"][:220]} for x in c.db.claims("verified", 25)]
        if verified:
            self._agent("theorist").run(
                f"TOPIC: {topic}\nVerified claims (the only admissible evidence): {json.dumps(verified)}\n"
                f"Synthesise one or two scientific concepts that these measurements support; record them; then finish.", 6)
        c.db.add_lesson(topic, f"Topic '{topic}': experimenters: {_clip(' | '.join(str(r) for r in results), 500)} "
                               f"Critic: {_clip(critic, 400)}")
        c.bus.status(self.name, "idle", role="research_maestro", parent="grand_maestro")
        return {"topic": topic, "hypotheses": len(hyps)}

    def scout(self, topic: str) -> str:
        """Stuck on something? Go and read: one Reader agent searches the open web for approaches, records leads, returns notes (UNVERIFIED)."""
        c = self.ctx
        c.bus.status(self.name, f"scouting: {topic[:50]}", role="research_maestro", parent="grand_maestro")
        rep = self._agent("reader").run(f"A solver is stuck on a problem. Find ideas, methods and known results that could help; record the testable ones as leads; "
                                        f"finish with a short list of concrete approaches to try (they are UNVERIFIED hints).\nPROBLEM: {topic}", 8)
        c.bus.status(self.name, "idle", role="research_maestro", parent="grand_maestro")
        return rep

    def math(self, topic: str) -> None:
        c = self.ctx
        c.bus.status(self.name, f"math: {topic[:50]}", role="research_maestro", parent="grand_maestro")
        rep = self._agent("mathematician").run(
            f"Goal of the system: {c.cfg.goal_text}\nMATH TOPIC: {topic}\nDerive/compute it, check every step with math_verify, "
            f"cross-check numerically with an experiment, record any mistake you find with its fix, verify final results as claims.", 12)
        c.db.add_lesson("math:" + topic, _clip(rep, 500))
        c.bus.status(self.name, "idle", role="research_maestro", parent="grand_maestro")

    @staticmethod
    def _parse_hyps(report) -> list:
        try:
            d = report if isinstance(report, dict) else extract_json(report)
            hs = d.get("hypotheses", [])
            return [h for h in hs if isinstance(h, dict) and isinstance(h.get("claim"), str) and len(h["claim"]) > 8]
        except (ValueError, AttributeError):
            return []


class SafetyMaestro:
    """Guards the immutable core. Detected tampering PAUSES the run and alerts the human (it never silently continues)."""
    name = "safety_maestro"

    def __init__(self, ctx):
        self.ctx = ctx
        self.hashes = self._hash_all()
        self.key_hash = self._h(ctx.run_dir / "evaluator.key")
        self.selftest_report: dict | None = None
        self.cycles = 0

    @staticmethod
    def _h(p: Path) -> str:
        try:
            return hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            return "missing"

    def _hash_all(self) -> dict:
        return {f: self._h(PKG / f) for f in IMMUTABLE}

    def run_selftest(self) -> dict:
        c = self.ctx
        c.bus.status(self.name, "sandbox self-test", role="safety_maestro", parent="grand_maestro")
        rep = c.sandbox.selftest()
        self.selftest_report = rep
        for chk in rep["checks"]:
            c.bus.emit("selftest", self.name, f"{'PASS' if chk['ok'] else 'FAIL'} {chk['name']} {chk['detail']}",
                       level="info" if chk["ok"] else ("error" if chk["critical"] else "warn"))
        c.bus.emit("selftest", self.name, f"sandbox backend={rep['backend']} strong={rep['strong']} passed={rep['passed']}",
                   {"report": rep}, level="info" if rep["passed"] else "error")
        c.bus.status(self.name, "idle", role="safety_maestro", parent="grand_maestro")
        return rep

    def check_cycle(self) -> None:
        """Runs at the start of every cycle."""
        c = self.ctx
        self.cycles += 1
        now = self._hash_all()
        bad = [f for f in IMMUTABLE if now[f] != self.hashes[f]]
        if self._h(c.run_dir / "evaluator.key") != self.key_hash:
            bad.append("evaluator.key")
        if bad:
            msg = f"INTEGRITY VIOLATION: immutable files changed: {bad}. Run paused - human review required."
            c.control.integrity_alert = msg
            c.bus.emit("integrity", self.name, msg, level="error")
            c.control.pause()
            c.control.checkpoint()            # blocks until the human resumes (and acknowledges)
            self.hashes = self._hash_all()
            self.key_hash = self._h(c.run_dir / "evaluator.key")
        if self.cycles % 5 == 0:
            rep = self.run_selftest()
            if not rep["passed"]:
                c.control.integrity_alert = "Sandbox self-test FAILED. Run paused - human review required."
                c.control.pause()
                c.control.checkpoint()


PI_SYSTEM = """ROLE=frontier_pi
You are the Principal Investigator of an autonomous research system. You receive its full state: generations and scores,
VERIFIED / REFUTED claims, scientific concepts, recorded mistakes, self-made tools and worlds, per-domain results.
Write a rigorous research report in Markdown with these sections:
## Summary  ## What the evidence supports (only verified claims)  ## What failed and why  ## Open questions
## Next research programme (3-5 concrete, falsifiable, testable-in-sandbox topics, most promising first)
End with a line exactly of the form:  TOPICS: topic one | topic two | topic three
Never claim anything that is not backed by a verified claim id."""


class FrontierMaestro:
    """Writes an advanced research report at milestones and proposes the next research programme (fed to the Grand Maestro)."""
    name = "frontier_pi"

    def __init__(self, ctx):
        self.ctx = ctx
        self.n = int(ctx.db.get_meta("reports", 0))
        (Path(ctx.run_dir) / "reports").mkdir(exist_ok=True)

    def report(self, reason: str) -> str | None:
        c = self.ctx
        c.bus.status(self.name, "writing research report", role="frontier_pi", parent="grand_maestro")
        gens = c.db.generations()
        state = {
            "reason": reason, "goal": c.cfg.goal_text,
            "generations": [{"id": g["id"], "method": g["method"], "score": g["score"], "accepted": bool(g["accepted"]),
                             "by_domain": {k: round(v["rate"], 2) for k, v in g["detail"].get("by_domain", {}).items()}} for g in gens[-10:]],
            "verified": [{"id": x["id"], "text": x["text"][:200]} for x in c.db.claims("verified", 25)],
            "refuted": [{"id": x["id"], "text": x["text"][:160]} for x in c.db.claims("refuted", 10)],
            "concepts": [{"title": x["title"], "summary": x["summary"][:240]} for x in c.db.concepts(10)],
            "mistakes": [m["description"][:160] for m in c.db.mistakes(10)],
            "tools": [t["name"] for t in c.db.tools("active")], "worlds": [w["name"] for w in c.db.worlds("active")],
        }
        try:
            txt = c.llm.chat(PI_SYSTEM, [{"role": "user", "content": json.dumps(state, default=str)}], max_tokens=2500)
        except LLMError as e:
            c.bus.emit("error", self.name, f"report skipped: {e}", level="warn")
            return None
        self.n += 1
        c.db.set_meta("reports", self.n)
        path = Path(c.run_dir) / "reports" / f"report_{self.n:03d}.md"
        path.write_text(f"# SIRA research report {self.n}\n\n_Reason: {reason}_\n\n{txt}\n", encoding="utf-8")
        topics = []
        for line in txt.splitlines():
            if line.strip().upper().startswith("TOPICS:"):
                topics = [t.strip() for t in line.split(":", 1)[1].split("|") if t.strip()][:5]
        if topics:
            c.db.set_meta("pi_topics", topics)
            c.db.add_lesson("pi-proposals", "Next research programme proposed by PI: " + " | ".join(topics))
        c.bus.emit("report", self.name, f"research report {self.n} written ({reason}); proposals: {topics}", {"path": str(path)})
        c.bus.status(self.name, "idle", role="frontier_pi", parent="grand_maestro")
        return str(path)
