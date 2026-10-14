"""Thinking layer: every layer of the architecture (Evaluation Core, Researcher, Sandbox, Verifier, Failure Analysis,
Experience Memory, Improvement Engine, Red Team) deliberates before it acts. A deliberation is a structured record:
analysis, risks, alternatives, decision, confidence, next checks. It is stored (Thinking tab), fed into the next
layer's prompt, and never overrides the immutable evaluator/verifier - thoughts are ADVICE, checks stay mechanical.

cfg.thinking: off | low | medium | high
  low    - only failure analysis and experience memory
  medium - all layers, one pass
  high   - all layers, two passes (draft, then self-critique and revision)"""
from __future__ import annotations

import json

from .llm import extract_json, LLMError

LAYERS = {
    "evaluation_core": "Evaluation Core. Judge coverage and difficulty of the hidden fresh problem sets: which domains/difficulties are under-tested, "
                       "where could the model be fooled by contamination or format luck? Advice only: the evaluator itself is immutable.",
    "researcher": "AI Researcher (search + retrieval + planning). Plan the investigation: what to search, what would count as a testable lead, "
                  "which prior verified claims or mistakes are relevant, what is the cheapest decisive experiment?",
    "sandbox": "Sandbox. Pre-mortem before running an experiment: how can this experiment fail or mislead (bugs, units, seeds, trivial thresholds, "
               "resource limits)? What is the minimal design that can falsify the claim?",
    "verifier": "Verifier. Post-analysis of a claim that was refuted or not reproducible: is it the hypothesis, the experiment design, noise, or a bug? "
                "What single follow-up would settle it?",
    "failure_analysis": "Failure Analysis. A training round or generation failed or was rejected: find the most likely cause from the numbers "
                        "(data quality, learning rate, too little signal, noise level of the test) and propose ONE concrete change.",
    "memory": "Experience Memory. Distil the last cycle into at most three durable lessons that would change future behaviour "
              "(what worked, what failed and why, what to avoid).",
    "improvement_engine": "Improvement Engine. Decide how the next training round should differ from the last (algorithm, data mix, "
                          "learning rate, which domains), justified by the measured per-domain results.",
    "red_team": "Red Team. Attack the current best model: from the failure statistics, what kinds of problems break it, what stress families "
                "should be added next, and what weakness might the evaluator not be measuring at all?",
}

SYSTEM = """ROLE=thinker
You are the deliberation step of ONE layer of an autonomous research and self-improvement system.
{layer}
Think carefully but concisely. Use ONLY the facts in the context; if the context is insufficient say what is missing.
Reply with ONE JSON object:
{{"analysis": "<3-6 sentences>", "risks": ["..."], "alternatives": ["..."], "decision": "<one concrete decision>",
  "confidence": <0..1>, "next_checks": ["..."]}}"""

CRITIC = """ROLE=thinker
You are reviewing a draft deliberation of the layer below. Find flaws (unsupported claims, missing alternatives, overconfidence)
and return the IMPROVED deliberation in the same JSON format.
{layer}"""

TOKENS = {"low": 600, "medium": 900, "high": 1600}


class Thinker:
    def __init__(self, ctx):
        self.ctx = ctx

    def enabled(self, layer: str) -> bool:
        e = self.ctx.cfg.thinking
        return e != "off" and (e != "low" or layer in ("failure_analysis", "memory"))

    def think(self, layer: str, question: str, context: dict) -> dict | None:
        """Returns the deliberation (or None when disabled / unparseable). Never raises for LLM trouble."""
        c = self.ctx
        if layer not in LAYERS or not self.enabled(layer):
            return None
        effort = c.cfg.thinking
        c.bus.status(f"think:{layer}", "deliberating", role="thinker", parent="grand_maestro")
        user = json.dumps({"question": question, "context": context}, default=str, ensure_ascii=False)[:14000]
        try:
            txt = c.llm.chat(SYSTEM.format(layer=LAYERS[layer]), [{"role": "user", "content": user}], max_tokens=TOKENS[effort], temperature=0.3)
            d = self._parse(txt)
            if d and effort == "high":
                txt2 = c.llm.chat(CRITIC.format(layer=LAYERS[layer]),
                                  [{"role": "user", "content": user + "\n\nDRAFT:\n" + json.dumps(d)}], max_tokens=TOKENS[effort], temperature=0.2)
                d = self._parse(txt2) or d
        except LLMError as e:
            c.bus.emit("error", f"think:{layer}", f"deliberation skipped: {e}", level="warn")
            d = None
        c.bus.status(f"think:{layer}", "idle", role="thinker", parent="grand_maestro")
        if not d:
            return None
        d["layer"], d["question"] = layer, question[:300]
        c.db.add_thought(layer, question[:300], d)
        c.bus.emit("thinking", f"think:{layer}", f"{d.get('decision', '')[:200]}  (confidence {d.get('confidence')})", {"thought": d})
        return d

    @staticmethod
    def _parse(txt):
        try:
            d = extract_json(txt)
        except ValueError:
            return None
        out = {"analysis": str(d.get("analysis", ""))[:1500], "decision": str(d.get("decision", ""))[:600],
               "risks": [str(x)[:200] for x in d.get("risks", [])][:5], "alternatives": [str(x)[:200] for x in d.get("alternatives", [])][:5],
               "next_checks": [str(x)[:200] for x in d.get("next_checks", [])][:5]}
        try:
            out["confidence"] = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
        except (TypeError, ValueError):
            out["confidence"] = 0.5
        return out if out["decision"] or out["analysis"] else None

    @staticmethod
    def brief(d: dict | None) -> str:
        """Compact text to inject into another layer's prompt."""
        if not d:
            return ""
        return (f"[{d['layer']} thinking] decision: {d['decision']} | risks: {'; '.join(d['risks'][:3])} | "
                f"checks: {'; '.join(d['next_checks'][:3])}")


class RedTeam:
    """Automated red team: finds what breaks the best model from the failure statistics of its latest evaluation, then deliberates.
    (The human red team of the original design stays available through GUI directives.)"""

    def __init__(self, ctx):
        self.ctx = ctx

    def attack(self, eval_id: int | None, label: str) -> dict | None:
        c = self.ctx
        if not eval_id:
            return None
        items = c.db.eval_items(eval_id)
        if not items:
            return None
        by, fmt = {}, {}
        for i in items:
            d = by.setdefault(i["domain"], [0, 0])
            d[1] += 1
            d[0] += i["passed"]
            if not i["passed"]:
                r = i["reason"]
                kind = "format (no ANSWER/code block)" if r.startswith("no ") else "wrong answer" if r.startswith("got") else "hidden tests failed" if "tests passed" in r else "other"
                fmt[kind] = fmt.get(kind, 0) + 1
        stats = {"evaluated": label, "pass_rate_by_domain": {k: round(v[0] / v[1], 3) for k, v in by.items()}, "failure_kinds": fmt,
                 "n": len(items), "sample_failures": [{"domain": i["domain"], "reason": i["reason"][:100]} for i in items if not i["passed"]][:8]}
        d = c.thinker.think("red_team", "What breaks the current best model, and which stress problems should be added next?", stats)
        findings = {"stats": stats, "thought": d}
        c.db.set_meta("red_team_findings", findings)
        c.bus.emit("redteam", "red_team", f"weakest domains: {sorted(stats['pass_rate_by_domain'].items(), key=lambda kv: kv[1])[:2]}; failure kinds {fmt}")
        return findings
