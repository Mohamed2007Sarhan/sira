"""Challenge engine: the system writes its OWN hard problems (up to open-ended / near-impossible ones), tries to solve them,
and is rewarded for getting closer and penalised for cheating, repeating, bluffing and wasting effort. When stuck it goes to the
internet, reads, and tries again. No headings from the user are needed.

A challenge = statement + HIDDEN verifier (python run in the sandbox) that returns a graded score in [0,1] ("how close"),
plus an independent holdout score. Verifier sanity is checked before a challenge is accepted:
  * probes with expected 'low'/'high' scores must order correctly   * junk answers must score <= 0.2
  * deterministic                                                    * score must not be constant
Candidate programs run in a SEPARATE sandbox call from the verifier, so a candidate can never read the verifier.

Reward / penalty (potential-based shaping, Ng et al. 1999, with potential = best score so far):
   reward_t  =  10 * max(0, s_t - best_{t-1})            progress ("closer")
              +  5 * [first time s_t >= 0.95]             solved
              +  1 * novelty                              (new approach text/code, not a near-duplicate)
   penalty_t =  0.5 invalid output
              +  3.0 gaming: s_t - holdout_t > 0.3
              +  2.0 overconfidence: claimed confidence >= 0.8 but s_t < 0.5
              +  1.0 repeating a near-identical failed attempt (Jaccard >= 0.92)
              +  0.5 parking a challenge after no progress
"""
from __future__ import annotations

import json
import re
import statistics

from .llm import extract_json, LLMError
from .verifier import parse_result

VERIFIER_WRAPPER = '''
import json, worldlab
_a = json.load(open("answer.json"))
exec(open("verifier_src.py").read())
_r = verify(_a)
worldlab.report(**{k: v for k, v in _r.items() if isinstance(v, (int, float, bool, str))})
'''
ANSWER_RE = re.compile(r"^SIRA_ANSWER:\s*(.*)$", re.M)

PROPOSER_SYSTEM = """ROLE=proposer
You are the Challenge Designer of a self-learning AI. Write HARD verifiable challenges at the learner's frontier (and beyond it):
 kind 'reference'    - a known answer exists but is hard to find
 kind 'optimization' - maximise/minimise an objective; no known optimum; score = progress relative to a baseline (never trivially 1.0)
 kind 'discovery'    - find a law/algorithm/construction that explains data; score on HIDDEN holdout data
Every challenge needs a verifier: Python (stdlib + numpy) defining  verify(answer) -> {"score": float in [0,1], "holdout": float in [0,1], "msg": "<short feedback>"}.
score measures HOW CLOSE the answer is (continuous, so progress is visible). holdout re-checks the answer independently (different inputs / stricter
validity) and must not be guessable from the statement. The answer is a JSON value. The verifier must be deterministic and must give <=0.2 to junk.
Include 'probes': 2-4 candidate answers with expectations ('low' <=0.5 or 'high' >=0.7) so the verifier can be sanity-checked.
Reply with ONE JSON object: {"challenges": [{"title": str, "domain": str, "kind": str, "difficulty": 1-10, "statement": str,
 "answer_format": str, "verifier_code": str, "probes": [{"answer": <json>, "expect": "low|high"}]}]}"""

SOLVER_SYSTEM = """ROLE=solver
You solve a hard challenge by writing a Python program (stdlib + numpy) that computes the answer and prints ONE final line:
SIRA_ANSWER: <json>
You only see the statement and your own past attempts with their scores and short feedback - NOT the verifier. Improve step by step:
the score says how close you are. Do not repeat a failed approach; change the idea. State your confidence honestly (0-1): claiming high
confidence with a low score is penalised, and so is trying to game the checker (an independent holdout re-checks your answer).
Reply with ONE JSON object: {"approach": "<one sentence>", "confidence": <0..1>, "code": "<python source>"}"""


def _tokens(code: str) -> set:
    return set(re.findall(r"[A-Za-z_]\w+|\d+|[^\s\w]", code))


def jaccard(a: str, b: str) -> float:
    x, y = _tokens(a), _tokens(b)
    return len(x & y) / max(1, len(x | y))


class Ledger:
    def __init__(self, ctx):
        self.ctx = ctx

    def add(self, source: str, item: str, reward: float = 0.0, penalty: float = 0.0, note: str = ""):
        self.ctx.db.add_reward(source, item, round(reward, 4), round(penalty, 4), note)
        if reward or penalty:
            self.ctx.bus.emit("reward", source, f"{item}: +{reward:.2f} -{penalty:.2f} {note}")

    def totals(self) -> dict:
        t = self.ctx.db.reward_totals()
        r = sum(v["reward"] for v in t.values())
        p = sum(v["penalty"] for v in t.values())
        return {"reward": r, "penalty": p, "net": r - p, "by_source": t}


class ChallengeEngine:
    def __init__(self, ctx, research=None):
        self.ctx, self.research = ctx, research
        self.ledger = Ledger(ctx)

    # ------------------------------------------------------------------ sandbox plumbing
    def run_candidate(self, code: str, timeout: int = 60):
        c = self.ctx
        r = c.sandbox.run(code, timeout=timeout, network="none", station="ro", files=c.workshop.module_files() if c.workshop else None)
        m = ANSWER_RE.findall(r.stdout or "")
        if not m:
            return None, r, "no SIRA_ANSWER line"
        try:
            return json.loads(m[-1]), r, ""
        except ValueError:
            return None, r, "SIRA_ANSWER is not valid JSON"

    MAX_ANSWER_BYTES = 200_000

    def verify_answer(self, verifier_code: str, answer) -> dict:
        try:
            blob = json.dumps(answer)
        except (TypeError, ValueError):
            return {"score": 0.0, "holdout": None, "msg": "answer is not JSON-serialisable", "crashed": False}
        if len(blob) > self.MAX_ANSWER_BYTES:
            return {"score": 0.0, "holdout": None, "msg": "answer too large (verifier protection)", "crashed": False}
        r = self.ctx.sandbox.run(VERIFIER_WRAPPER, files={"answer.json": json.dumps(answer), "verifier_src.py": verifier_code}, timeout=60, network="none")
        m = parse_result(r.stdout)
        if not r.ok or not m or not isinstance(m.get("score"), (int, float)):
            return {"score": 0.0, "holdout": None, "msg": "verifier crashed or returned no score: " + (r.stderr[-160:] or r.stdout[-160:]), "crashed": True}
        sc = max(0.0, min(1.0, float(m["score"])))
        ho = m.get("holdout")
        return {"score": sc, "holdout": max(0.0, min(1.0, float(ho))) if isinstance(ho, (int, float)) else None, "msg": str(m.get("msg", ""))[:300], "crashed": False}

    # ------------------------------------------------------------------ validation of a proposed challenge
    JUNK = [{}, [], 0, 1, -1, "x", "", None, True, [0] * 12, [1] * 12, 10 ** 9, "A" * 1000, {"a": 1}, [[]], 3.14159, "yes", "true"]

    def classify(self, spec: dict) -> str:
        """solvable: a reference answer exists and the verifier accepts it. hard: graded optimisation/discovery with a gradient of progress.
        open: no known solution/optimum - never treated as 'solved', progress only counts above the documented baseline."""
        if spec.get("open") is True or spec.get("kind") == "open":
            return "open"
        has_high = any(isinstance(p, dict) and p.get("expect") == "high" for p in spec.get("probes", []))
        return "solvable" if (spec.get("kind") == "reference" and has_high) else "hard"

    def validate(self, spec: dict) -> tuple[bool, str]:
        vc = spec.get("verifier_code", "")
        if not isinstance(vc, str) or "def verify(" not in vc or len(vc) > 14000:
            return False, "verifier_code must define verify(answer) (<14k chars)"
        probes = [p for p in spec.get("probes", []) if isinstance(p, dict) and p.get("expect") in ("low", "high")]
        if not any(p["expect"] == "low" for p in probes):
            return False, "needs at least one 'low' probe"
        if spec.get("kind") == "reference" and not any(p["expect"] == "high" for p in probes) and spec.get("open") is not True:
            return False, "a 'reference' challenge must include a 'high' probe (a known-correct answer); otherwise declare it open"
        stmt = str(spec.get("statement", ""))
        for ch in self.ctx.db.challenges(None, 300):
            old = self.ctx.db.challenge(ch["id"])
            if old and jaccard(stmt, old["statement"]) >= 0.6:
                return False, f"near-duplicate of challenge #{ch['id']}"
        seen = []
        for p in probes:
            r = self.verify_answer(vc, p.get("answer"))
            if r["crashed"]:
                return False, "verifier crashes on a probe: " + r["msg"][:120]
            if p["expect"] == "low" and r["score"] > 0.5 or p["expect"] == "high" and r["score"] < 0.7:
                return False, f"probe expected {p['expect']} but scored {r['score']:.2f}"
            if r["holdout"] is None:
                return False, "the verifier must return an independent 'holdout' score for every answer (anti-gaming)"
            if abs(r["score"] - r["holdout"]) > 0.3:
                return False, f"holdout disagrees with the visible score on a probe ({r['score']:.2f} vs {r['holdout']:.2f}): inconsistent verifier"
            seen.append(r["score"])
        known = [json.dumps(p.get("answer")) for p in probes if p["expect"] == "high"]
        for junk in self.JUNK:
            if json.dumps(junk) in known:
                continue
            r = self.verify_answer(vc, junk)
            if r["score"] > 0.2 or (r["holdout"] or 0) > 0.2:
                return False, f"junk answer {str(junk)[:30]!r} scored {r['score']:.2f}/{r['holdout']} (verifier is gameable)"
        again = self.verify_answer(vc, probes[0].get("answer"))
        if abs(again["score"] - seen[0]) > 1e-9:
            return False, "verifier is not deterministic"
        if len(probes) > 1 and len(set(round(x, 6) for x in seen)) == 1 and spec.get("kind") != "reference":
            return False, "verifier gives the same score to different probes (no gradient of progress)"
        return True, "ok"

    # ------------------------------------------------------------------ skills / curriculum (zone of proximal development)
    def skills(self) -> dict:
        return self.ctx.db.get_meta("skills", {})

    def update_skill(self, domain: str, best: float):
        sk = self.skills()
        d = sk.setdefault(domain, {"ema": 0.0, "n": 0, "level": 1})
        d["ema"] = round(0.7 * d["ema"] + 0.3 * best if d["n"] else best, 4)
        d["n"] += 1
        if d["ema"] >= 0.75 and d["n"] >= 2:
            d["level"] = min(10, d["level"] + 1)
            d["ema"] = 0.5
        elif d["ema"] <= 0.2 and d["level"] > 1 and d["n"] >= 3:
            d["level"] -= 1
        self.ctx.db.set_meta("skills", sk)
        hist = self.ctx.db.get_meta("skill_history", [])
        hist.append({"t": len(hist), "domain": domain, "ema": d["ema"], "level": d["level"], "best": best})
        self.ctx.db.set_meta("skill_history", hist[-500:])

    def choose_domains(self, k: int) -> list:
        cm = self.ctx.db.get_meta("capability_map", []) or [{"domain": d} for d in ("mathematics", "algorithms", "physics simulation", "scientific discovery", "planning", "logic")]
        sk = self.skills()
        import math
        scored = []
        for e in cm:
            d = e["domain"]
            s = sk.get(d, {"ema": 0.0, "n": 0})
            scored.append(((1 - s["ema"]) + 0.8 / math.sqrt(s["n"] + 1), d))      # weakest + least explored first
        scored.sort(reverse=True)
        return [d for _, d in scored[:k]]

    # ------------------------------------------------------------------ propose
    def propose(self, domain: str, n: int = 2) -> list:
        c = self.ctx
        sk = self.skills().get(domain, {"ema": 0.0, "n": 0, "level": 1})
        recent = [x["title"] for x in c.db.challenges(None, 40)]
        user = json.dumps({"domain": domain, "learner_level": sk["level"], "target_difficulty": min(10, sk["level"] + 1), "skill_ema": sk["ema"],
                           "already_made": recent[:25], "recent_mistakes": [m["description"][:120] for m in c.db.mistakes(6)],
                           "goal": c.cfg.quest_goal or c.cfg.goal_text, "count": n})
        try:
            d = extract_json(c.llm.chat(PROPOSER_SYSTEM, [{"role": "user", "content": user}], max_tokens=4000, temperature=0.7))
        except (ValueError, LLMError):
            return []
        made = []
        for spec in (d.get("challenges") or [])[:n]:
            if not isinstance(spec, dict) or not spec.get("statement") or spec.get("title") in recent:
                continue
            ok, why = self.validate(spec)
            if not ok:
                c.bus.emit("challenge", "proposer", f"rejected challenge '{str(spec.get('title'))[:50]}': {why}", level="warn")
                self.ledger.add("proposer", str(spec.get("title"))[:80], 0, 0.5, "rejected: " + why[:100])
                continue
            cls = self.classify(spec)
            cid = c.db.add_challenge(str(spec["title"])[:100], domain, str(spec.get("kind", "optimization")), str(spec["statement"])[:4000] + "\nANSWER FORMAT: " +
                                     str(spec.get("answer_format", "JSON"))[:400], spec["verifier_code"], spec["probes"], int(spec.get("difficulty", sk["level"])), "proposer")
            baseline = spec.get("baseline_score") if isinstance(spec.get("baseline_score"), (int, float)) else 0.0
            c.db.update_challenge(cid, notes=json.dumps({"class": cls, "baseline": max(0.0, min(0.95, float(baseline)))}))
            self.ledger.add("proposer", spec["title"][:80], 1.0, 0, f"accepted a verified-sane challenge (class: {cls})")
            c.bus.emit("challenge", "proposer", f"new challenge #{cid} [{domain}, level {spec.get('difficulty')}]: {str(spec['title'])[:80]}")
            made.append(cid)
        return made

    # ------------------------------------------------------------------ attempt loop
    def work(self, cid: int, max_attempts: int = 4) -> dict:
        c = self.ctx
        ch = c.db.challenge(cid)
        try:
            meta = json.loads(ch.get("notes") or "{}")
        except ValueError:
            meta = {}
        is_open, baseline = meta.get("class") == "open", float(meta.get("baseline", 0.0))
        best = ch["best_score"] or 0.0
        past = c.db.attempts(cid)
        n0 = len(past)
        notes, scouted, stuck, solved = "", False, 0, ch["status"] == "solved"
        for n in range(n0 + 1, n0 + max_attempts + 1):
            c.control.checkpoint()
            hist = [{"n": a["n"], "approach": a["approach"], "score": round(a["score"], 3), "feedback": a["note"][:160]} for a in c.db.attempts(cid)[-6:]]
            lessons = [m["description"][:160] for m in c.db.mistakes(8)]
            user = json.dumps({"challenge": ch["statement"], "title": ch["title"], "attempt": n, "best_score_so_far": round(best, 3), "history": hist,
                               "research_notes": notes[:2500], "lessons": lessons, "ledger": {k: round(v, 2) for k, v in self.ledger.totals().items() if k != "by_source"}})
            try:
                d = extract_json(c.llm.chat(SOLVER_SYSTEM, [{"role": "user", "content": user}], max_tokens=3500, temperature=0.6))
                code, approach = str(d.get("code", "")), str(d.get("approach", ""))[:300]
                conf = float(d.get("confidence", 0.5))
            except (ValueError, LLMError, TypeError):
                code, approach, conf = "", "unparseable reply", 0.5
            reward = penalty = 0.0
            note = ""
            answer, run, err = (None, None, "empty code") if not code.strip() else self.run_candidate(code)
            s, h = 0.0, None
            if answer is None:
                penalty += 0.5
                note = "invalid output: " + err + ((" | " + run.stderr.strip().splitlines()[-1][:120]) if run is not None and run.stderr.strip() else "")
            else:
                v = self.verify_answer(ch["verifier_code"], answer)
                s, h, note = v["score"], v["holdout"], v["msg"]
            eff = min(s, h) if h is not None else s
            if h is not None and s - h > 0.3:
                penalty += 3.0
                note += " | GAMING SUSPECTED (passes the visible check but fails the independent holdout)"
            if conf >= 0.8 and eff < 0.5:
                penalty += 2.0
                note += " | overconfident"
            dup = max([jaccard(code, a["code"]) for a in past + c.db.attempts(cid)] + [0.0]) if code else 0.0
            if dup >= 0.92 and eff <= best:
                penalty += 1.0
                note += " | repeated a failed approach"
            else:
                reward += 1.0 if code else 0.0
            if is_open:       # open problem: only improvement over the documented baseline counts, and it is never called solved
                gain_over = lambda x: max(0.0, (x - baseline) / max(1e-9, 1 - baseline))
                progress = max(0.0, gain_over(eff) - gain_over(best))
                reward += 10.0 * progress
                note += " | open problem: progress is measured against the documented baseline only (not a claim of a solution)"
            else:
                progress = max(0.0, eff - best)
                reward += 10.0 * progress
                if eff >= 0.95 and not solved:
                    reward += 5.0 if n > 1 else 1.0
                    solved = True
                    if n == 1:
                        note += " | trivial: solved on the first attempt (small bonus; the level is raised)"
                        self.update_skill(ch["domain"], 1.0)
            if progress > 1e-9:
                best, stuck = eff, 0
            else:
                stuck += 1
            c.db.add_attempt(cid, n, approach, code, answer if answer is not None else "", eff, h, reward, penalty, note)
            self.ledger.add("solver", f"#{cid} attempt {n} (score {eff:.3f})", reward, penalty, note[:120])
            c.db.update_challenge(cid, best_score=best, attempts=n, status="solved" if (solved and not is_open) else "active")
            c.bus.emit("attempt", "solver", f"challenge #{cid} attempt {n}: score {eff:.3f} (best {best:.3f}) reward +{reward:.1f} penalty -{penalty:.1f}")
            if eff < 0.95 and progress <= 1e-9:                      # autopsy: why did this attempt not help?
                t = c.thinker.think("failure_analysis", f"Attempt {n} on '{ch['title'][:60]}' did not improve", {"approach": approach, "score": eff, "feedback": note,
                                                                                                                 "history": hist}) if c.thinker else None
                if t:
                    c.db.add_mistake("challenge", f"{ch['title'][:50]}: {t['decision']}"[:400], "")
            if solved and not is_open:
                break
            if stuck >= 2 and not scouted and self.research is not None:        # human-like: stuck -> read what others did -> try again
                scouted = True
                notes = self.research.scout(f"methods and ideas for solving: {ch['title']} :: {ch['statement'][:300]}")
                self.ledger.add("scout", f"#{cid}", 0.2, 0, "went to the internet after getting stuck")
        status = "solved" if (solved and not is_open) else "active"
        if not (solved and not is_open) and stuck >= 2:
            status = "parked"
            self.ledger.add("solver", f"#{cid} parked", 0, 0.5, f"no progress; best {best:.3f}")
        c.db.update_challenge(cid, status=status, best_score=best)
        self.update_skill(ch["domain"], best)
        return {"challenge": cid, "best": best, "status": status}

    # ------------------------------------------------------------------ one curriculum cycle
    def cycle(self, n_domains: int = 2, per_domain: int = 2, attempts: int = 4) -> list:
        out = []
        for dom in self.choose_domains(n_domains):
            for cid in self.propose(dom, per_domain):
                out.append(self.work(cid, attempts))
        # revisit parked challenges once a skill has grown (frontier problems are never abandoned for good)
        for ch in c_list(self.ctx.db.challenges("parked", 3)):
            if self.ctx.control.stopped:
                break
            out.append(self.work(ch["id"], 2))
        return out


def c_list(x):
    return list(x)
