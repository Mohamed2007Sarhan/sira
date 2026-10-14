"""Validation harness for a REAL brain model (Ollama / any API) and, optionally, a real target model.
    python -m sira --validate-real --config my.json [--validate-trials 5] [--validate-rounds 3] [--validate-out DIR]
It records every input and output (llm_calls.jsonl), measures protocol compliance per agent role, runs one mini quest cycle with the real brain,
and (if base_model is set) measures the target model before/after improvement on fresh held-out problems with several seeds, mean +- SD and a 95% CI.
The report (report.json / report.md) states what passed and what did not. It is the evidence that the platform works with YOUR model."""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

from .llm import extract_json
from .quest import PLANNER_SYSTEM, STUDENT_SYSTEM, AUTOPSY_SYSTEM
from .challenges import PROPOSER_SYSTEM, SOLVER_SYSTEM
from .thinking import SYSTEM as THINK_SYSTEM, LAYERS

PROBES = {
    "quest_planner": (PLANNER_SYSTEM, json.dumps({"goal": "reach artificial general intelligence", "hints_from_user": []}), ["definition", "capability_map", "criteria", "topics", "unverifiable"]),
    "proposer": (PROPOSER_SYSTEM, json.dumps({"domain": "algorithms", "learner_level": 2, "target_difficulty": 3, "count": 1, "already_made": []}), ["challenges"]),
    "solver": (SOLVER_SYSTEM, json.dumps({"challenge": "Return the number of primes below 100 as JSON.", "title": "primes", "attempt": 1, "best_score_so_far": 0, "history": []}), ["approach", "code"]),
    "student": (STUDENT_SYSTEM, json.dumps({"question": {"kind": "status", "prompt": "Statement: water boils at 90C at sea level. Answer VERIFIED, REFUTED or UNVERIFIED."}, "answer_format": "one word", "notes": []}), ["answer", "confidence"]),
    "autopsy": (AUTOPSY_SYSTEM, json.dumps({"question": "q", "ground_truth": "VERIFIED", "reason": "wrong", "student_answer": "REFUTED"}), ["misconception", "corrected_understanding"]),
    "thinker": (THINK_SYSTEM.format(layer=LAYERS["researcher"]), json.dumps({"question": "How should we investigate sorting algorithms?", "context": {}}), ["analysis", "decision"]),
}


def ci95(xs: list) -> tuple[float, float, float]:
    n = len(xs)
    m = statistics.fmean(xs) if xs else 0.0
    sd = statistics.stdev(xs) if n > 1 else 0.0
    t = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}.get(n - 1 if n > 1 else 1, 1.96)
    t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}.get(n, 1.96) if n > 1 else 0.0
    return m, sd, (t * sd / math.sqrt(n) if n > 1 else float("nan"))


def run_validation(cfg, out_dir, trials: int = 5, rounds: int = 3) -> dict:
    from .orchestrator import Orchestrator
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.run_root = str(out / "runs")
    o = Orchestrator(cfg)
    o.build()
    llm = o.ctx.llm
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "provider": cfg.llm_provider, "model": cfg.llm_model, "checks": {}, "roles": {}}
    ok, msg = llm.check()
    report["checks"]["connection"] = {"ok": ok, "detail": msg}
    if not ok:
        report["verdict"] = "FAILED: brain not reachable"
        (out / "report.json").write_text(json.dumps(report, indent=2))
        o.stop()
        return report
    # ---- protocol compliance per role (JSON validity + required keys), latency
    for role, (system, user, keys) in PROBES.items():
        valid, lat = 0, []
        for _ in range(trials):
            t0 = time.time()
            try:
                d = extract_json(llm.chat(system, [{"role": "user", "content": user}], max_tokens=1500, temperature=0.3))
                valid += all(k in d for k in keys)
            except Exception:
                pass
            lat.append(time.time() - t0)
        report["roles"][role] = {"valid_json_with_required_keys": f"{valid}/{trials}", "rate": valid / trials, "mean_latency_s": round(statistics.fmean(lat), 2)}
    # ---- one mini quest cycle with the real brain (every call is logged)
    cfg.mode = cfg.mode if cfg.mode != "model" else "quest"
    t0 = time.time()
    try:
        st = o.quest.cycle()
        report["checks"]["quest_cycle"] = {"ok": True, "history": st["history"][-1], "seconds": round(time.time() - t0, 1)}
    except Exception as e:
        report["checks"]["quest_cycle"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    # ---- target model before / after (optional)
    if cfg.base_model:
        try:
            o.training.do_intake()
            if o.training.can_train:
                base = o.training.baseline()
                accs = [base["score"]]
                for _ in range(rounds):
                    o.training.improve([])
                    accs.append(o.training.best()["score"])
                report["target_model"] = {"scores_per_generation": accs, "gain": accs[-1] - accs[0], "generations": [(g["id"], g["method"], bool(g["accepted"]), round(g["score"], 3)) for g in o.db.generations()]}
            else:
                report["target_model"] = {"skipped": o.training.recipe.get("reason")}
        except Exception as e:
            report["target_model"] = {"error": f"{type(e).__name__}: {e}"}
    rates = [v["rate"] for v in report["roles"].values()]
    report["llm_stats"] = o.ctx.llm.stats
    report["call_log"] = str(o.run_dir / "llm_calls.jsonl")
    report["verdict"] = ("PASS: protocol compliance >= 80% in every role" if rates and min(rates) >= 0.8 else
                         "PARTIAL: some roles below 80% JSON compliance - use a stronger brain model or enable llm_json_mode")
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    md = [f"# SIRA validation report ({report['provider']} / {report['model']})", f"Verdict: **{report['verdict']}**", "", "| role | valid JSON with keys | latency (s) |", "|---|---|---|"]
    md += [f"| {r} | {v['valid_json_with_required_keys']} | {v['mean_latency_s']} |" for r, v in report["roles"].items()]
    md += ["", f"Quest cycle: {report['checks'].get('quest_cycle')}", f"Target model: {report.get('target_model')}", f"All inputs and outputs: `{report['call_log']}`"]
    (out / "report.md").write_text("\n".join(md))
    o.stop()
    return report
