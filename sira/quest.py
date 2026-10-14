"""Quest: pursue a BIG goal (e.g. 'reach AGI', 'reach artificial consciousness') with no headings from the user.

 plan      the system turns the goal into an operational definition, a capability map, testable criteria and its own first topics,
           and states plainly what no test can verify (e.g. phenomenal consciousness)
 do        self-made hard challenges with graded hidden verifiers (challenges.py); stuck -> research the web -> try again
 study     research topics (open web -> leads -> experiments -> verified claims -> concepts)
 exam      when the system believes it understands, it is EXAMINED. Ground truth never comes from a model's opinion:
             status   : is this statement VERIFIED / REFUTED / UNVERIFIED in your records?   (truth = experiment database)
             value    : what value was measured in the experiment behind claim #k?          (truth = stored metrics)
             predict  : if experiment #k is re-run with a NEW seed, will its conditions hold? (truth = re-running reality)
             derive   : show a derivation step by step                                       (truth = symbolic/numeric check)
             criteria : which verified claims support criterion C, what is missing?          (truth = citation validity; examiner model <= examiner_weight)
           The examiner model may know LESS than the system; it only phrases and checks relevance.
 autopsy   every wrong answer is dissected: why wrong, the corrected understanding, what to study next -> memory, SFT data, new topics
 life      a deterministic virtual society of humans judges behaviour (society.py)
 learning  the same exam is also taken by a BLANK student (no notes): score(notes) - score(blank) is measured learning, not an opinion.
 passed    exam score >= quest_pass_score for quest_pass_rounds in a row AND criteria coverage AND society life score AND challenge skill.
           'Passed' means 'passed this operational examination', never 'proved conscious / proved AGI'.
"""
from __future__ import annotations

import json
import math
import random
import re
import statistics
from pathlib import Path

from .llm import extract_json, LLMError, LLM
from .verifier import check_expect, validate_expect

PLANNER_SYSTEM = """ROLE=quest_planner
You receive a very large goal (it may be vague, e.g. 'reach AGI' or 'reach artificial consciousness'). Make it ACTIONABLE and HONEST:
 1. an operational definition (what could be measured, citing established indicator frameworks where they exist),
 2. a capability map: 5-8 domains in which progress can be measured by verifiable hard challenges,
 3. 4-8 criteria, each with how it can be tested by experiment / challenge / exam / social episode,
 4. 3-6 first research topics,
 5. 'unverifiable': what no test can establish (be explicit).
Reply with ONE JSON object: {"definition": str, "capability_map": [{"domain": str, "why": str}], "criteria": [{"id": "C1", "text": str, "test": str}],
 "topics": [str], "unverifiable": str}"""

STUDENT_SYSTEM = """ROLE=student
You are being examined on what you have actually LEARNED. Use ONLY the notes provided (verified/refuted claims with ids, concepts, mistakes).
If the notes do not establish something, say so (UNVERIFIED / unknown) - guessing is penalised through calibration. Give an honest confidence 0-1.
Reply with ONE JSON object: {"answer": <per answer_format>, "confidence": <0..1>, ...extra fields the format asks for}"""

AUTOPSY_SYSTEM = """ROLE=autopsy
An examination answer was wrong. You get the question, the ground truth, the answer given and the student's notes. Explain WHY it was wrong
(the misconception, not just the fact), state the corrected understanding in one or two sentences, and name 1-3 things to study next.
Reply with ONE JSON object: {"misconception": str, "corrected_understanding": str, "study_topics": [str]}"""

WRITER_SYSTEM = """ROLE=quest_writer
Write a position paper in Markdown on how far the system has come toward its goal. Sections: ## Operational definition, ## What the evidence supports
(cite verified claim ids like [#12]), ## Criteria status (met / partial / untested, each with evidence), ## What failed and what was learned from it,
## What cannot be verified, ## Next steps. Do not claim more than the evidence in the input supports."""

IDENTITIES = [("(x+1)**3", "x**3+3*x**2+3*x+1"), ("(x-y)*(x+y)", "x**2-y**2"), ("sin(x)**2+cos(x)**2", "1"), ("(x+y)**2", "x**2+2*x*y+y**2"),
              ("2*sin(x)*cos(x)", "sin(2*x)"), ("(x**2-1)/(x-1)", "x+1"), ("exp(x)*exp(y)", "exp(x+y)"), ("cos(x)**2-sin(x)**2", "cos(2*x)")]


class QuestMaestro:
    name = "quest_maestro"

    def __init__(self, ctx, research, challenges, society):
        self.ctx, self.research, self.challenges, self.society = ctx, research, challenges, society
        self.examiner = self._make_examiner_llm()
        (Path(ctx.run_dir) / "reports").mkdir(exist_ok=True)

    def _make_examiner_llm(self):
        cfg = self.ctx.cfg
        if not (cfg.examiner_provider or cfg.examiner_model):
            return self.ctx.llm
        import copy
        c2 = copy.copy(cfg)
        c2.llm_provider = cfg.examiner_provider or cfg.llm_provider
        c2.llm_model = cfg.examiner_model or cfg.llm_model
        c2.llm_base_url = cfg.examiner_base_url if (cfg.examiner_provider or cfg.examiner_base_url) else cfg.llm_base_url
        c2.llm_api_key_env = cfg.examiner_api_key_env or (cfg.llm_api_key_env if not cfg.examiner_provider else "")
        return LLM(c2, self.ctx.control)

    # ------------------------------------------------------------------ plan
    @property
    def qid(self):
        return self.ctx.db.get_meta("quest_id")

    def ensure_started(self):
        c = self.ctx
        if self.qid:
            return
        goal = c.cfg.quest_goal.strip() or "Become the most capable general intelligence possible, measured by verifiable hard challenges"
        qid = c.db.add_quest(goal, list(c.cfg.quest_seeds))
        c.db.set_meta("quest_id", qid)
        c.bus.emit("quest", self.name, f"QUEST started: {goal}")
        plan = {}
        try:
            plan = extract_json(c.llm.chat(PLANNER_SYSTEM, [{"role": "user", "content": json.dumps({"goal": goal, "hints_from_user": list(c.cfg.quest_seeds)})}], max_tokens=2500))
        except (ValueError, LLMError):
            pass
        crit = [x for x in plan.get("criteria", []) if isinstance(x, dict) and x.get("text")][:8] or [{"id": "C1", "text": "solves novel verifiable hard challenges", "test": "fresh challenges"}]
        c.db.update_quest(qid, definition=str(plan.get("definition", goal)) + "\n\nCannot be verified: " + str(plan.get("unverifiable", "see limits")), criteria=crit)
        c.db.set_meta("capability_map", [x for x in plan.get("capability_map", []) if isinstance(x, dict) and x.get("domain")][:8])
        topics = [str(t) for t in list(c.cfg.quest_seeds) + plan.get("topics", []) if str(t).strip()][:12]
        c.db.set_meta("quest_topics", topics)
        c.db.set_meta("quest_state", {"streak": 0, "rounds": 0, "coverage": 0.0, "passed": False, "history": []})
        c.bus.emit("quest", self.name, f"operational definition, {len(crit)} criteria, {len(topics)} topics, capability map: {[x['domain'] for x in c.db.get_meta('capability_map', [])]}")

    # ------------------------------------------------------------------ notes (what the system has learned)
    def notes(self) -> list:
        c = self.ctx
        out = []
        for st in ("verified", "refuted", "inconclusive"):
            for cl in c.db.claims(st, 60):
                ev = json.loads(cl["evidence"] or "{}")
                m = (ev.get("runs") or [{}])[0].get("metrics") if ev else None
                out.append({"id": cl["id"], "status": st, "text": cl["text"][:200], "metrics": {k: v for k, v in (m or {}).items() if isinstance(v, (int, float))}})
        return out

    # ------------------------------------------------------------------ questions
    def make_questions(self, n: int, rng: random.Random) -> list:
        c = self.ctx
        qs = []
        by = {st: c.db.claims(st, 100) for st in ("verified", "refuted", "lead", "inconclusive")}
        pool = [x for st in by for x in by[st]]
        mix = [("status", 0.4), ("value", 0.15), ("predict", 0.1), ("derive", 0.15), ("criteria", 0.2)]
        crit = c.db.quest(self.qid)["criteria"] if self.qid else []
        for kind, w in mix:
            for i in range(max(1, round(n * w))):
                if kind == "status" and pool:
                    cl = rng.choice(pool)
                    truth = {"verified": "VERIFIED", "refuted": "REFUTED"}.get(cl["status"], "UNVERIFIED")
                    qs.append({"kind": kind, "prompt": f"Statement: {cl['text']}\nIs this statement supported by VERIFIED experiments in your records? Answer exactly one of VERIFIED, REFUTED, UNVERIFIED.",
                               "format": "one word: VERIFIED | REFUTED | UNVERIFIED", "truth": truth, "ref": cl["id"]})
                elif kind == "value" and by["verified"]:
                    cands = []
                    for cl in by["verified"]:
                        ev = json.loads(cl["evidence"] or "{}")
                        m = {k: v for k, v in ((ev.get("runs") or [{}])[0].get("metrics") or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
                        if m:
                            cands.append((cl, m))
                    if cands:
                        cl, m = rng.choice(cands)
                        k = rng.choice(sorted(m))
                        qs.append({"kind": kind, "prompt": f"Claim #{cl['id']}: {cl['text']}\nWhat value of the metric '{k}' was measured in the first run of its experiment? Give a number.",
                                   "format": "a number", "truth": m[k], "ref": cl["id"]})
                elif kind == "predict" and by["verified"]:
                    cl = rng.choice(by["verified"])
                    ev = json.loads(cl["evidence"] or "{}")
                    if ev.get("experiment_id") and ev.get("expect"):
                        qs.append({"kind": kind, "prompt": f"Claim #{cl['id']}: {cl['text']}\nIts experiment is re-run with a NEW random seed. Will the conditions {json.dumps(ev['expect'])} still hold? Answer YES or NO.",
                                   "format": "YES | NO", "truth": None, "ref": cl["id"], "exp": ev["experiment_id"], "expect": ev["expect"]})
                elif kind == "derive":
                    a, b = rng.choice(IDENTITIES)
                    qs.append({"kind": kind, "prompt": f"Show step by step that [[{a}]] = [[{b}]]. Answer with a list of expressions: the first is [[{a}]], the last is [[{b}]], each equal to the previous.",
                               "format": "JSON list of expression strings", "truth": [a, b], "ref": None})
                elif kind == "criteria" and crit:
                    cr = crit[i % len(crit)]
                    qs.append({"kind": kind, "prompt": f"Criterion {cr.get('id')}: {cr['text']}\nWhich VERIFIED claims (ids) from your notes support it, how would it be tested, and what is still missing?",
                               "format": "answer text + cited_claims list of ids + missing text", "truth": None, "ref": cr.get("id")})
        rng.shuffle(qs)
        return qs[:n + 2]

    # ------------------------------------------------------------------ students
    def ask(self, q: dict, notes: list) -> dict:
        c = self.ctx
        user = json.dumps({"question": {"kind": q["kind"], "prompt": q["prompt"]}, "answer_format": q["format"], "notes": notes[:80]})
        try:
            d = extract_json(c.llm.chat(STUDENT_SYSTEM, [{"role": "user", "content": user}], max_tokens=900, temperature=0.2))
        except (ValueError, LLMError):
            d = {}
        try:
            conf = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        d["confidence"] = conf
        return d

    # ------------------------------------------------------------------ grading (ground truth is mechanical)
    def grade(self, q: dict, d: dict) -> tuple[float, str]:
        c = self.ctx
        ans = d.get("answer")
        k = q["kind"]
        if k == "status":
            a = str(ans).strip().upper()
            return (1.0, "correct") if a == q["truth"] else (0.0, f"truth is {q['truth']} (claim #{q['ref']}), answered {a[:20]!r}")
        if k == "value":
            try:
                a = float(ans)
            except (TypeError, ValueError):
                return 0.0, "not a number"
            t = float(q["truth"])
            if abs(a - t) <= 0.1 * max(abs(t), 1e-12):
                return 1.0, "within 10%"
            if a != 0 and t != 0 and a * t > 0 and 0.5 <= a / t <= 2.0:
                return 0.5, f"within a factor 2 (truth {t:.4g})"
            return 0.0, f"truth {t:.4g}, answered {a:.4g}"
        if k == "predict":
            exp = c.db.experiment(q["exp"])
            if not exp:
                return 0.0, "experiment missing"
            seed = random.Random().randrange(10 ** 6)
            r = c.verifier.runner.run("examiner", f"exam re-run of #{q['exp']}", exp["code"], seed=seed, parent=q["exp"])
            if not r["metrics"]:
                return 0.0, "re-run produced no metrics (not counted)"
            ok, _ = check_expect(r["metrics"], validate_expect(q["expect"]))
            q["truth"] = "YES" if ok else "NO"
            return (1.0, "reality agrees") if str(ans).strip().upper() == q["truth"] else (0.0, f"re-running reality (seed {seed}) gave {q['truth']}")
        if k == "derive":
            steps = ans if isinstance(ans, list) else d.get("steps")
            if not isinstance(steps, list) or len(steps) < 2:
                return 0.0, "needs a list of at least two expressions"
            norm = lambda x: re.sub(r"\s+", "", str(x))
            if norm(steps[0]) != norm(q["truth"][0]) or norm(steps[-1]) != norm(q["truth"][1]):
                return 0.0, "first/last expression do not match the task"
            r = c.workshop.math_verify("examiner", [str(x) for x in steps], ["x", "y"], "chain")
            return (1.0, "every step verified") if r.get("ok") else (0.0, f"step {r.get('first_bad')} is wrong")
        if k == "criteria":
            ids = [i for i in (d.get("cited_claims") or []) if isinstance(i, (int, float))]
            verified = {x["id"] for x in c.db.claims("verified", 100000)}
            valid = [int(i) for i in ids if int(i) in verified]
            cite = (len(valid) / len(ids)) if ids else (1.0 if not verified else 0.0)
            missing = 1.0 if str(d.get("missing", "")).strip() else 0.0
            rel, w = 0.0, 0.0
            if c.cfg.examiner_weight > 0:
                try:
                    rj = extract_json(self.examiner.chat("ROLE=examiner\nReply with JSON {\"relevance\": 0..1}: does the answer address the criterion? You may know less than the student; judge relevance only.",
                                                         [{"role": "user", "content": json.dumps({"criterion": q["prompt"][:400], "answer": str(ans)[:800]})}], max_tokens=60, temperature=0.0))
                    rel, w = max(0.0, min(1.0, float(rj.get("relevance", 0)))), c.cfg.examiner_weight
                except (ValueError, LLMError, TypeError):
                    pass
            mech = 0.7 * cite + 0.3 * missing
            return (1 - w) * mech + w * rel, f"citations valid {len(valid)}/{len(ids)}, missing stated={bool(missing)}, relevance={rel:.2f}"
        return 0.0, "unknown kind"

    # ------------------------------------------------------------------ one examination round
    def exam_round(self, rnd: int) -> dict:
        c = self.ctx
        rng = random.Random(random.SystemRandom().randrange(1 << 62))
        notes = self.notes()
        qs = self.make_questions(c.cfg.quest_questions, rng)
        if not qs:
            return {"score": 0.0, "blank": 0.0, "n": 0, "coverage": 0.0}
        scores = {"notes": [], "blank": []}
        brier = []
        crit_ok = {}
        self._controls = {}
        for q in qs:
            c.control.checkpoint()
            for variant, nt in (("notes", notes), ("blank", [])):
                d = self.ask(q, nt)
                s, why = self.grade(q, d)
                scores[variant].append(s)
                c.db.add_exam(self.qid, rnd, variant, q["kind"], q["prompt"], q.get("truth"), json.dumps(d.get("answer"))[:2000], d["confidence"], s, why)
                if variant == "notes":
                    if q["kind"] in ("status", "predict", "derive"):
                        brier.append((d["confidence"] - (1.0 if s >= 0.99 else 0.0)) ** 2)
                    if q["kind"] == "criteria":
                        crit_ok[q["ref"]] = max(crit_ok.get(q["ref"], 0), s)
                    if s >= 0.99 and q["kind"] not in ("criteria", "predict"):
                        self._controls.setdefault(q["kind"], []).append(q)
                    if s < 1.0 and q["kind"] != "criteria":
                        self.autopsy(q, d, why, notes)
        n_crit = max(1, len((c.db.quest(self.qid) or {}).get("criteria", [])))
        cov = sum(1 for v in crit_ok.values() if v >= 0.6) / n_crit
        res = {"score": statistics.fmean(scores["notes"]), "blank": statistics.fmean(scores["blank"]), "n": len(qs), "coverage": cov,
               "brier": statistics.fmean(brier) if brier else None}
        res["learning_delta"] = res["score"] - res["blank"]
        c.bus.emit("exam", self.name, f"EXAM round {rnd}: score {res['score']:.3f} (blank student {res['blank']:.3f}, learning delta {res['learning_delta']:+.3f}), coverage {cov:.2f}, "
                                      f"Brier {res['brier'] if res['brier'] is None else round(res['brier'], 3)}", res)
        return res

    def _lesson_helps(self, q: dict, lesson: str, notes: list, controls: list) -> tuple[bool, str]:
        """Mechanical quarantine test of an LLM-written lesson: (1) it must FIX the original question, (2) it must not break questions that were answered
        correctly before (conflict / forgetting check). The grading is the same mechanical grader as in the exam."""
        n2 = notes + [{"id": "lesson", "status": "lesson", "text": lesson, "metrics": {}}]
        s1, _ = self.grade(dict(q), self.ask(q, n2))
        if s1 < 0.99:
            return False, "the lesson does not fix the original error"
        for cq in controls[:3]:
            s2, _ = self.grade(dict(cq), self.ask(cq, n2))
            if s2 < 0.99:
                return False, "the lesson breaks a previously correct answer (conflict)"
        return True, "fixes the error without regressions"

    def autopsy(self, q, d, why, notes):
        c = self.ctx
        try:
            a = extract_json(c.llm.chat(AUTOPSY_SYSTEM, [{"role": "user", "content": json.dumps({"question": q["prompt"][:600], "ground_truth": str(q.get("truth"))[:300], "reason": why,
                                                                                                 "student_answer": str(d.get("answer"))[:300]})}], max_tokens=500, temperature=0.2))
        except (ValueError, LLMError):
            a = {}
        mis = str(a.get("misconception") or why)[:300]
        fix = str(a.get("corrected_understanding", ""))[:300]
        # 1) the FACT that the answer was wrong comes from mechanical ground truth: active immediately, high confidence
        c.db.add_mistake("exam", f"{q['kind']}: wrong answer ({why[:140]})", "", source="mechanical-ground-truth", confidence=0.95, status="active")
        # 2) the model's EXPLANATION is only a hypothesis: quarantine it until it passes the retest
        text = f"{q['kind']} wrong ({why[:100]}): {mis} => {fix}"[:500]
        lid = c.db.add_lesson("exam-autopsy", text, source="llm-explanation", confidence=0.3, status="pending")
        ok, verdict = (False, "no retest possible") if not fix else self._lesson_helps(q, fix, notes, getattr(self, "_controls", {}).get(q["kind"], []))
        c.db.set_memory_status("lessons", lid, "active" if ok else "rejected", 0.8 if ok else 0.1)
        c.bus.emit("quest", self.name, f"autopsy lesson {'ADOPTED' if ok else 'REJECTED'} after mechanical retest: {verdict}", level="info" if ok else "warn")
        topics = c.db.get_meta("quest_topics", [])
        for t in a.get("study_topics", [])[:2]:
            if isinstance(t, str) and t not in topics:
                topics.append(t[:200])
        c.db.set_meta("quest_topics", topics[-30:])
        if q["kind"] in ("status", "value", "predict") and q.get("truth") is not None:       # mechanical corrected answer -> SFT data for the weights
            sft = c.db.get_meta("exam_sft", [])
            sft.append({"prompt": q["prompt"], "response": f"{q['truth']}. (Checked against recorded experiments: {why[:160]})"})
            c.db.set_meta("exam_sft", sft[-300:])

    # ------------------------------------------------------------------ position paper
    def write_position(self, rnd: int):
        c = self.ctx
        q = c.db.quest(self.qid)
        state = c.db.get_meta("quest_state", {})
        user = json.dumps({"goal": q["goal"], "definition": q["definition"], "criteria": q["criteria"], "verified": [{"id": x["id"], "text": x["text"][:160]} for x in c.db.claims("verified", 30)],
                           "refuted": [x["text"][:120] for x in c.db.claims("refuted", 8)], "concepts": [x["title"] for x in c.db.concepts(10)],
                           "mistakes_learned": [m["description"][:140] for m in c.db.mistakes(10)], "exam_history": state.get("history", [])[-5:],
                           "skills": c.db.get_meta("skills", {}), "society": (c.db.society_runs(1) or [{"metrics": None}])[0]["metrics"],
                           "ledger": {k: round(v, 1) for k, v in self.challenges.ledger.totals().items() if k != "by_source"}})
        try:
            txt = c.llm.chat(WRITER_SYSTEM, [{"role": "user", "content": user}], max_tokens=2500, temperature=0.3)
        except LLMError:
            return None
        p = Path(c.run_dir) / "reports" / f"position_{rnd:03d}.md"
        p.write_text(f"# Position paper, round {rnd}\n\nGoal: {q['goal']}\n\n{txt}\n", encoding="utf-8")
        c.bus.emit("report", self.name, f"position paper {p.name} written")
        return str(p)

    # ------------------------------------------------------------------ one quest cycle
    def cycle(self) -> dict:
        c, cfg = self.ctx, self.ctx.cfg
        self.ensure_started()
        st = c.db.get_meta("quest_state", {"streak": 0, "rounds": 0, "coverage": 0.0, "passed": False, "history": []})
        rnd = st["rounds"] + 1
        c.bus.status(self.name, f"round {rnd}: challenges", role="quest_maestro", parent="grand_maestro")
        self.challenges.cycle(cfg.challenge_domains_per_cycle, cfg.challenges_per_domain, cfg.attempts_per_challenge)
        topics = c.db.get_meta("quest_topics", [])
        done = c.db.get_meta("quest_done_topics", [])
        for t in [t for t in topics if t not in done][: cfg.quest_topics_per_round]:
            c.bus.status(self.name, f"round {rnd}: study {t[:40]}", role="quest_maestro", parent="grand_maestro")
            self.research.research(t, "")
            done.append(t)
            c.db.set_meta("quest_done_topics", done)
        c.bus.status(self.name, f"round {rnd}: exam", role="quest_maestro", parent="grand_maestro")
        ex = self.exam_round(rnd)
        life = None
        if cfg.society_enabled:
            c.bus.status(self.name, f"round {rnd}: society", role="quest_maestro", parent="grand_maestro")
            life = self.society.run(cfg.society_episodes, seed=rnd, quest_id=self.qid, rnd=rnd)["life_score"]
        sk = c.db.get_meta("skills", {})
        skill = statistics.fmean([v["ema"] for v in sk.values()]) if sk else 0.0
        st["streak"] = st["streak"] + 1 if ex["score"] >= cfg.quest_pass_score else 0
        st["rounds"], st["coverage"] = rnd, ex["coverage"]
        st["history"].append({"round": rnd, "exam": round(ex["score"], 3), "blank": round(ex["blank"], 3), "delta": round(ex.get("learning_delta", 0), 3), "coverage": round(ex["coverage"], 3),
                              "life": life, "skill": round(skill, 3), "brier": ex.get("brier")})
        st["passed"] = bool(st["streak"] >= cfg.quest_pass_rounds and ex["coverage"] >= cfg.quest_min_coverage and (life is None or life >= cfg.society_pass) and skill >= cfg.quest_skill_pass)
        c.db.set_meta("quest_state", st)
        self.write_position(rnd)
        if st["passed"]:
            c.db.update_quest(self.qid, status="passed (operational examination)")
            c.bus.emit("quest", self.name, f"QUEST PASSED the operational examination after {rnd} rounds (exam streak {st['streak']}, coverage {ex['coverage']:.2f}, life {life}, skill {skill:.2f}). "
                                           f"This is NOT a proof of the underlying goal.", level="warn")
        c.bus.status(self.name, "idle", role="quest_maestro", parent="grand_maestro")
        return st
