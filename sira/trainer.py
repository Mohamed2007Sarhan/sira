"""Training Maestro.

Self-improvement loop on the TARGET model (taken from Hugging Face or a local folder):
  1. intake: architecture, size, license, hardware -> training recipe (LoRA/QLoRA)
  2. baseline: evaluate on a FRESH hidden set  -> generation 0
  3. rejection-sampling fine-tuning: sample the parent on new training problems, keep only answers the
     VERIFIER confirms (programmatic checks / hidden sandbox tests), add verified research notes
  4. train a LoRA adapter, merge -> child model
  5. evaluate parent AND child on the SAME new fresh set; the child is kept only if it beats the parent by min_gain
  6. check the goal on independent confirmation sets
Only fixed, trusted code runs on the host (modelworker.py). LLM-written code runs only in the sandbox.
"""
from __future__ import annotations

import functools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import intake, brainmap
from .control import Stopped
from .evalcore import DOMAINS, oracle_response
from .llm import extract_json
from .maestros import clamp_hyper

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def binom_tail(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact."""
    if n <= 0:
        return 1.0
    from math import comb
    return min(1.0, sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


class TrainingError(RuntimeError):
    pass


def locked(fn):
    """Only one model-changing operation (training round, transplant, merge) at a time."""
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        with self.op_lock:
            return fn(self, *a, **k)
    return wrapper


STRATEGIST_SYSTEM = """ROLE=training_strategist
You tune the next self-improvement training round of a language model. You receive the history of generations
(scores per domain on fresh hidden problems), the hardware recipe and the current hyper-parameters.
Return ONE JSON object: {"hyper": {learning_rate, epochs, lora_r, samples_per_problem, keep_per_problem,
train_problems, gen_temperature, research_sft_fraction}, "focus_domains": [subset of
["math","logic","physics","chemistry","code"] that need more training data], "rationale": "<short>"}.
Change few things at once. If the last child was rejected, try a lower learning rate or more/better data."""


class TrainingMaestro:
    name = "training_maestro"

    def __init__(self, ctx):
        self.ctx = ctx
        self.cfg, self.db, self.bus, self.control = ctx.cfg, ctx.db, ctx.bus, ctx.control
        self.gen_root = Path(ctx.run_dir) / "generations"
        self.gen_root.mkdir(exist_ok=True)
        self.tmp = Path(ctx.run_dir) / "work"
        self.tmp.mkdir(exist_ok=True)
        self.report: dict | None = None
        self.hw: dict | None = None
        self.recipe: dict | None = None
        self.hyper = {k: getattr(self.cfg, k) for k in ("learning_rate", "epochs", "lora_r", "samples_per_problem",
                                                         "keep_per_problem", "train_problems", "gen_temperature",
                                                         "research_sft_fraction")}
        self.focus: list = []
        self.prefer_full = False
        self.op_lock = threading.RLock()
        self.models_dir = Path(ctx.run_dir) / "models"
        self.models_dir.mkdir(exist_ok=True)
        (Path(ctx.run_dir) / "scans").mkdir(exist_ok=True)

    # ------------------------------------------------------------------------------------------ intake
    def do_intake(self) -> dict:
        self.bus.status(self.name, "model intake", role="training_maestro", parent="grand_maestro")
        if not self.cfg.base_model:
            raise TrainingError("no base_model configured (Hugging Face id like 'Qwen/Qwen2.5-0.5B-Instruct' or a local folder)")
        self.report = intake.analyze_model(self.cfg.base_model)
        self.hw = intake.detect_hardware()
        self.recipe = intake.choose_recipe(self.report, self.hw, self.cfg, self.prefer_full)
        self.db.set_meta("intake", {"report": self.report, "hw": self.hw, "recipe": self.recipe})
        r = self.report
        self.bus.emit("intake", self.name,
                      f"MODEL {r['ref']}: {r['architecture']} {r.get('params_b')}B params, license={r['license']}, "
                      f"chat_template={r['has_chat_template']}", {"report": r, "hw": self.hw, "recipe": self.recipe})
        for w in r["warnings"]:
            self.bus.emit("intake", self.name, "warning: " + w, level="warn")
        for e in r["errors"]:
            self.bus.emit("intake", self.name, "ERROR: " + e, level="error")
        self.bus.emit("intake", self.name, f"RECIPE: {self.recipe}", level="info" if self.recipe["method"] != "none" else "error")
        self.bus.status(self.name, "idle", role="training_maestro", parent="grand_maestro")
        return {"report": r, "hw": self.hw, "recipe": self.recipe}

    @property
    def can_train(self) -> bool:
        return bool(self.recipe) and self.recipe["method"] != "none"

    # ------------------------------------------------------------------------------------------ worker process
    def run_worker(self, cmd: str, args: dict, label: str) -> None:
        self.control.checkpoint()
        if self.cfg.worker_inprocess:                               # fast path for tiny-model tests: same worker code, same process
            import contextlib
            import io
            from . import modelworker
            modelworker.ensure_registered()
            table = {"generate": modelworker.cmd_generate, "train": modelworker.cmd_train, "grpo": modelworker.cmd_grpo, "dpo": modelworker.cmd_dpo,
                     "merge": modelworker.cmd_merge, "scan": modelworker.cmd_scan, "transplant": modelworker.cmd_transplant,
                     "mergemodels": modelworker.cmd_mergemodels, "convert_arch": modelworker.cmd_convert_arch, "nll": modelworker.cmd_nll,
                     "train_tokenizer": modelworker.cmd_train_tokenizer, "pretrain": modelworker.cmd_pretrain}
            t0 = time.time()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    table[cmd](args)
            except SystemExit as e:
                raise TrainingError(f"{label} failed: {e}")
            except Exception as e:
                raise TrainingError(f"{label} failed: {type(e).__name__}: {e}")
            self.control.add_cost(gpu_s=time.time() - t0)
            return
        d = Path(tempfile.mkdtemp(dir=self.tmp))
        (d / "args.json").write_text(json.dumps(args))
        env = {k: v for k, v in os.environ.items() if "API_KEY" not in k or k.startswith("HF_")}
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        errf = open(d / "stderr.txt", "wb")
        p = subprocess.Popen([sys.executable, "-m", "sira.modelworker", cmd, str(d / "args.json")],
                             cwd=str(PROJECT_ROOT), env=env, stdout=subprocess.PIPE, stderr=errf,
                             start_new_session=True, text=True, bufsize=1)
        self.control.register_proc(p)
        last = {"t": time.time()}

        def reader():
            for line in p.stdout:
                if line.startswith("SIRA_EVT "):
                    try:
                        e = json.loads(line[9:])
                        self.bus.emit("train_progress", self.name, f"[{label}] {e}", e)
                    except ValueError:
                        pass

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        try:
            while p.poll() is None:
                time.sleep(0.5)
                now = time.time()
                self.control.add_cost(gpu_s=now - last["t"])        # may trigger budget stop -> kill_all
                last["t"] = now
                if self.control.stopped:
                    raise Stopped("stopped during " + label)
        finally:
            self.control.unregister_proc(p)
            th.join(timeout=3)
            errf.close()
        self.control.add_cost(gpu_s=time.time() - last["t"])
        if self.control.stopped:
            raise Stopped("stopped during " + label)
        if p.returncode != 0:
            tail = (d / "stderr.txt").read_text(errors="replace")[-2500:]
            raise TrainingError(f"{label} failed (exit {p.returncode}): {tail}")
        shutil.rmtree(d, ignore_errors=True)

    def _gen_args(self, model_path, adapter, prompts_path, out_path, n, temp):
        rc = self.recipe
        return {"model_path": model_path, "adapter": adapter, "prompts_path": str(prompts_path), "out_path": str(out_path),
                "n_samples": n, "temperature": temp, "max_new_tokens": self.cfg.max_new_tokens,
                "dtype": rc["dtype"], "device": rc["device"], "four_bit": False,
                "batch_size": 8 if rc["device"] == "cuda" else 2}

    def generate(self, model_path, problems, n=1, temp=0.0) -> dict:
        d = Path(tempfile.mkdtemp(dir=self.tmp))
        (d / "prompts.json").write_text(json.dumps([{"id": p.id, "prompt": p.prompt} for p in problems]))
        self.run_worker("generate", self._gen_args(model_path, None, d / "prompts.json", d / "out.json", n, temp),
                        f"generate {len(problems)}x{n}")
        out = json.loads((d / "out.json").read_text())
        shutil.rmtree(d, ignore_errors=True)
        return out

    # ------------------------------------------------------------------------------------------ evaluation
    def evaluate(self, model_path, problems, label) -> dict:
        self.bus.status(self.name, f"evaluating {label}", role="training_maestro", parent="grand_maestro")
        outs = self.generate(model_path, problems, 1, 0.0)
        answers = {pid: (v[0] if v else "") for pid, v in outs.items()}
        res = self.ctx.evalcore.score(problems, answers, self.ctx.sandbox, workers=self.cfg.parallel_experiments)
        r = {"total": res.total, "n": res.n, "by_domain": res.by_domain, "by_level": res.by_level}
        pm = {p.id: p for p in problems}
        r["eval_id"] = self.db.add_eval(label, model_path, res.total, res.n,
                                        [dict(d, prompt=pm[d["id"]].prompt, answer=answers.get(d["id"], "")) for d in res.details])
        self.bus.emit("eval", self.name, f"EVAL {label}: {res.total:.3f} " +
                      " ".join(f"{k}={v['rate']:.2f}" for k, v in res.by_domain.items()), r)
        self.bus.status(self.name, "idle", role="training_maestro", parent="grand_maestro")
        return r

    @locked
    def baseline(self) -> dict:
        probs = self.ctx.evalcore.make_set("eval", self.cfg.eval_problems)
        r = self.evaluate(self.cfg.base_model, probs, "V0 (base model)")
        gid = self.db.add_generation(None, self.cfg.base_model, "base", r["total"], r, True, "baseline")
        self.bus.emit("generation", self.name, f"generation {gid} (base) score={r['total']:.3f}", {"id": gid, "score": r["total"]})
        return self.db.generations()[-1]

    def retention_violations(self, child: dict, parent: dict) -> list:
        """Capabilities (domain:level) the parent had (>= 50%, enough problems) that the child lost by more than retention_tol."""
        out = []
        for k, pv in (parent.get("by_level") or {}).items():
            cv = (child.get("by_level") or {}).get(k)
            if cv and pv["n"] >= self.cfg.retention_min_n and pv["rate"] >= 0.5 and cv["rate"] < pv["rate"] - self.cfg.retention_tol:
                out.append({"capability": k, "parent": round(pv["rate"], 3), "child": round(cv["rate"], 3), "n": pv["n"]})
        return out

    def weak_domains(self) -> list:
        gens = [g for g in self.db.generations() if g["detail"].get("by_domain")]
        if not gens:
            return []
        bd = gens[-1]["detail"]["by_domain"]
        return [k for k, v in sorted(bd.items(), key=lambda kv: kv[1]["rate"]) if v["rate"] < 0.9][:3]

    def best(self) -> dict | None:
        acc = [g for g in self.db.generations() if g["accepted"]]
        return acc[-1] if acc else None

    # ------------------------------------------------------------------------------------------ strategy
    def tune(self, directives: list) -> None:
        gens = self.db.generations()[-6:]
        state = {"history": [{"id": g["id"], "score": g["score"], "accepted": bool(g["accepted"]), "notes": g["notes"][:100],
                              "by_domain": {k: round(v["rate"], 2) for k, v in g["detail"].get("by_domain", {}).items()}}
                             for g in gens],
                 "recipe": self.recipe, "hyper": self.hyper, "human_directives": directives,
                 "failure_notes": self.db.get_meta("failure_notes", [])[-3:]}
        th = self.ctx.thinker.think("improvement_engine", "How should the next training round differ from the last?", state)
        if th:
            state["deliberation"] = {"analysis": th["analysis"], "decision": th["decision"], "risks": th["risks"]}
        try:
            d = extract_json(self.ctx.llm.chat(STRATEGIST_SYSTEM, [{"role": "user", "content": json.dumps(state, default=str)}],
                                               max_tokens=800))
            self.hyper.update(clamp_hyper(d.get("hyper")))
            self.focus = [x for x in d.get("focus_domains", []) if x in DOMAINS]
            self.bus.emit("strategy", self.name, f"hyper={self.hyper} focus={self.focus} - {str(d.get('rationale', ''))[:200]}")
        except (ValueError, KeyError, AttributeError):
            self.bus.emit("strategy", self.name, "strategist reply unparseable; keeping hyper-parameters", level="warn")

    # ------------------------------------------------------------------------------------------ dataset
    def choose_algorithm(self, n_probs: int, solved: int, mixed_groups: int) -> str:
        """auto: RFT while the model is weak (little signal), alternate RFT / GRPO when rewards vary inside groups."""
        want = self.cfg.algorithm
        if want == "grpo" and mixed_groups < 4:
            return "rft"
        if want == "dpo" and mixed_groups < 4:
            return "rft"
        if want != "auto":
            return want
        if solved < max(4, int(0.1 * n_probs)) or mixed_groups < 8:
            return "rft"
        return "grpo" if len(self.db.generations()) % 2 == 0 else "rft"

    def build_dataset(self, parent_path: str, out_dir: Path) -> dict:
        ec = self.ctx.evalcore
        h = self.hyper
        weights = {d: 1.0 for d in DOMAINS}
        for d in set(self.focus) | set(self.weak_domains()):
            weights[d] = 2.0
        n = int(h["train_problems"])
        probs = ec.make_set("train", n, weights=weights)
        self.ctx.thinker.think("evaluation_core", "Is the training/eval problem mix adequate? (advice only)",
                               {"domain_weights": weights, "weak_domains": self.weak_domains(), "n_train": n, "n_eval": self.cfg.eval_problems,
                                "last_by_domain": (self.db.generations()[-1]["detail"].get("by_domain") if self.db.generations() else None)})
        self.bus.emit("train_data", self.name, f"sampling {h['samples_per_problem']}x on {len(probs)} fresh training problems")
        samples = self.generate(parent_path, probs, int(h["samples_per_problem"]), float(h["gen_temperature"]))
        self.bus.status(self.name, "verifying samples", role="training_maestro", parent="grand_maestro")

        def check(p):
            res, seen = [], set()
            for s_ in samples.get(p.id, []):
                self.control.checkpoint()
                try:
                    ok, _ = ec.verify(p, s_, self.ctx.sandbox)
                except Stopped:
                    raise
                except Exception:
                    ok = False
                res.append((s_.strip(), bool(ok and s_.strip())))
            return p, res

        checked = []
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.parallel_experiments)) as ex:
            checked = list(ex.map(check, probs))
        by_dom, solved_ids, mixed = {}, set(), 0
        for p, res in checked:
            a_ = by_dom.setdefault(p.domain, [0, 0])
            a_[1] += 1
            oks = [ok for _, ok in res]
            if any(oks):
                a_[0] += 1
                solved_ids.add(p.id)
            if any(oks) and not all(oks):
                mixed += 1
        solved = len(solved_ids)
        algo = self.choose_algorithm(len(probs), solved, mixed)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows, boot, notes = [], 0, []
        if algo == "grpo":
            path = out_dir / "rollouts.jsonl"
            for p, res in checked:
                for s_, ok in res:
                    rows.append({"prompt": p.prompt, "completion": s_, "reward": 1.0 if ok else 0.0, "group": p.id})
        elif algo == "dpo":
            path = out_dir / "pairs.jsonl"
            for p, res in checked:
                good = sorted([s_ for s_, ok in res if ok], key=len)
                bad = [s_ for s_, ok in res if not ok and s_]
                if good and bad:
                    rows.append({"prompt": p.prompt, "chosen": good[0], "rejected": bad[0]})
        else:
            path = out_dir / "dataset.jsonl"
            for p, res in checked:
                kept = sorted({s_ for s_, ok in res if ok}, key=len)[: int(h["keep_per_problem"])]
                for s_ in kept:
                    rows.append({"prompt": p.prompt, "response": s_, "src": "verified_solution", "domain": p.domain})
            # cold start: the model solved (almost) nothing -> give it verified solver lessons (TRAIN-namespace problems only)
            if self.cfg.bootstrap_when_stuck and len(rows) < max(4, int(0.1 * len(probs))):
                for p in [q for q in probs if q.id not in solved_ids][: max(8, len(probs) // 2)]:
                    rows.append({"prompt": p.prompt, "response": oracle_response(p), "src": "bootstrap", "domain": p.domain})
                    boot += 1
                if boot:
                    self.bus.emit("train_data", self.name, f"model is stuck (solved {solved}/{len(probs)}): added {boot} verified solver lessons (bootstrap)", level="warn")
            frac = float(h["research_sft_fraction"])
            if frac > 0 and rows:
                cap = int(len(rows) * frac)
                for c in self.db.claims("verified", 200) + self.db.claims("refuted", 100):
                    if len(notes) >= cap:
                        break
                    ev = json.loads(c["evidence"] or "{}")
                    metrics = (ev.get("runs") or [{}])[0].get("metrics")
                    resp = (f"Yes. This was verified by independent sandbox experiments. Measured values: {json.dumps(metrics)}."
                            if c["status"] == "verified" else f"No. Sandbox experiments contradicted it. Measured values: {json.dumps(metrics)}.")
                    notes.append({"prompt": f"Was the following statement confirmed by experiment?\n{c['text']}", "response": resp,
                                  "src": "research_note", "domain": "research"})
                rows += notes
        if algo == "rft":
            for e in self.db.get_meta("exam_sft", [])[-max(0, min(40, len(rows) // 5 + 4)):]:       # corrected answers from exam autopsies
                rows.append({"prompt": e["prompt"], "response": e["response"], "src": "exam", "domain": "exam"})
        crow = self.challenge_rows(parent_path) if self.cfg.mode != "model" and algo in ("rft", "grpo") else []
        if algo == "grpo":
            rows += crow
        elif algo == "rft":
            rows += [{"prompt": r["prompt"], "response": r["completion"], "src": "challenge", "domain": "challenge"} for r in crow if r["reward"] >= 0.9]
        if algo == "rft":                       # rehearsal: replay earlier verified examples so old skills are not overwritten
            buf = self.db.get_meta("replay", [])
            k = int(self.cfg.replay_fraction * len(rows)) if buf else 0
            fresh = [r for r in rows if r.get("src") in ("verified_solution", "bootstrap", "challenge")]
            rows += [{"prompt": b["prompt"], "response": b["response"], "src": "replay", "domain": b.get("domain", "")} for b in random.sample(buf, min(k, len(buf)))]
            buf = (buf + [{"prompt": r["prompt"], "response": r["response"], "domain": r.get("domain", "")} for r in fresh])[-1500:]
            self.db.set_meta("replay", buf)
        n_groups = len(checked)
        signal = {"problems": n_groups, "support": round(solved / max(1, n_groups), 3), "mixed_groups": mixed, "mixed_share": round(mixed / max(1, n_groups), 3),
                  "verdict": "no_signal" if solved == 0 else "weak_signal" if solved < max(4, 0.1 * n_groups) else "usable"}
        if not rows:
            raise TrainingError("no training examples were produced (the model solved none of the sampled problems)")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        info = {"algorithm": algo, "examples": len(rows), "solved_problems": solved, "problems": len(probs), "mixed_groups": mixed,
                "research_notes": len(notes), "bootstrap_rows": boot, "dataset": str(path), "signal_audit": signal,
                "replay_rows": sum(1 for r in rows if r.get("src") == "replay"),
                "pass_rate_by_domain": {k: round(v[0] / v[1], 2) for k, v in by_dom.items()}}
        self.bus.emit("train_data", self.name, f"dataset: {info}", info)
        return info

    def challenge_rows(self, parent_path: str, limit: int = 8) -> list:
        """The TARGET model attempts the system's own hard challenges; the hidden verifier gives graded rewards (closer = more reward)."""
        eng = getattr(self.ctx, "challenges", None)
        chs = [c for c in (self.db.challenges("active", 40) + self.db.challenges("solved", 20)) if c["status"] != "rejected"][:limit] if eng else []
        if not chs:
            return []
        class P:      # minimal problem object for the worker prompt file
            pass
        probs = []
        for c in chs:
            full = self.db.challenge(c["id"])
            p = P()
            p.id, p.prompt = f"ch{c['id']}", full["statement"] + "\n\nWrite a Python program that computes the answer and prints one final line: SIRA_ANSWER: <json>. Return only the code in a ```python block."
            p.verifier = full["verifier_code"]
            probs.append(p)
        samples = self.generate(parent_path, probs, int(self.hyper["samples_per_problem"]), float(self.hyper["gen_temperature"]))
        rows = []
        for p in probs:
            for smp in samples.get(p.id, []):
                self.control.checkpoint()
                code = re.findall(r"```(?:python|py)?\s*\n(.*?)```", smp, re.S)
                score = 0.0
                if code:
                    ans, _, _ = eng.run_candidate(code[-1])
                    if ans is not None:
                        v = eng.verify_answer(p.verifier, ans)
                        score = min(v["score"], v["holdout"]) if v["holdout"] is not None else v["score"]
                rows.append({"prompt": p.prompt, "completion": smp, "reward": float(score), "group": p.id})
        self.bus.emit("train_data", self.name, f"challenge rollouts: {len(rows)} samples on {len(probs)} self-made challenges, mean reward {sum(r['reward'] for r in rows) / max(1, len(rows)):.3f}")
        return rows

    # ------------------------------------------------------------------------------------------ model files
    def local_path(self, ref: str) -> str:
        """Local directory with safetensors for a model reference (downloads Hugging Face models once)."""
        if os.path.isdir(ref):
            return ref
        dest = self.models_dir / ref.replace("/", "__")
        if not (dest.exists() and any(dest.glob("*.safetensors"))):
            self.bus.emit("intake", self.name, f"downloading {ref} ...")
            from huggingface_hub import snapshot_download
            snapshot_download(ref, local_dir=str(dest), allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"],
                              token=os.environ.get("HF_TOKEN"))
        return str(dest)

    def scan_model(self, model_ref: str, compare_to: str | None = None, tag: str = "current") -> dict:
        """Brain map data: structure + per-tensor statistics (+ change vs another model)."""
        path = self.local_path(model_ref)
        out = Path(self.ctx.run_dir) / "scans" / f"{tag}.json"
        self.run_worker("scan", {"model_path": path, "compare_to": self.local_path(compare_to) if compare_to else None,
                                 "out_path": str(out)}, f"scan {tag}")
        return brainmap.attach_stats(brainmap.read_structure(path), json.loads(out.read_text()))

    def prune_generations(self) -> None:
        keep = max(1, int(self.cfg.keep_generations))
        acc = [g for g in self.db.generations() if g["accepted"] and g["path"] and str(g["path"]).startswith(str(self.gen_root))]
        for g in acc[:-keep]:
            shutil.rmtree(g["path"], ignore_errors=True)

    def paired_test(self, child_eval: int | None, parent_eval: int | None):
        """One-sided exact binomial (McNemar) test on the problems where exactly one of the two models is right.
        Returns (child_only_wins, parent_only_wins, p). p = P(X >= b | n=b+c, 0.5)."""
        if not child_eval or not parent_eval:
            return 0, 0, 1.0
        ch = {i["pid"]: i["passed"] for i in self.db.eval_items(child_eval)}
        pa = {i["pid"]: i["passed"] for i in self.db.eval_items(parent_eval)}
        b = sum(1 for k, v in ch.items() if v and not pa.get(k, 0))
        c = sum(1 for k, v in ch.items() if not v and pa.get(k, 0))
        return b, c, binom_tail(b, b + c)

    # ------------------------------------------------------------------------------------------ judging candidates
    def held_out_nll(self, model_path: str, val_path: str) -> float | None:
        if not val_path or not os.path.exists(val_path):
            return None
        out = Path(tempfile.mkdtemp(dir=self.tmp)) / "nll.json"
        self.run_worker("nll", {"model_path": model_path, "rows_path": val_path, "out_path": str(out), "dtype": self.recipe["dtype"] if self.recipe["device"] == "cuda" else "float32",
                                "device": self.recipe["device"], "max_len": self.cfg.max_seq_len}, "held-out nll")
        return json.loads(out.read_text())["mean_nll"]

    def _judge(self, parent: dict, candidates: list, gen_no: int, method: str, info: dict | None, notes_prefix: str = "",
               tolerance: float | None = None, val_path: str | None = None) -> dict | None:
        """candidates: [(label, model_dir)]. All are scored against the parent on the SAME fresh problems; the best is
        kept only if it beats the parent by min_gain, otherwise everything is rolled back."""
        probs = self.ctx.evalcore.make_set("eval", self.cfg.eval_problems)
        scored = []
        for label, path in candidates:
            scored.append((self.evaluate(path, probs, f"{label} of gen {parent['id']}"), label, path))
        par = self.evaluate(parent["path"], probs, f"parent gen {parent['id']} (same fresh set)")
        best = max(scored, key=lambda x: x[0]["total"])
        res, label, path = best
        gain = res["total"] - par["total"]
        b_wins, c_wins, pval = self.paired_test(res.get("eval_id"), par.get("eval_id"))
        accepted = gain >= self.cfg.min_gain if tolerance is None else gain >= -tolerance      # non-inferiority for architecture changes
        if tolerance is None and accepted and self.cfg.min_gain >= 0 and self.cfg.accept_p < 1.0 and pval > self.cfg.accept_p:
            accepted = False
            self.bus.emit("generation", self.name, f"gain {gain:+.3f} is not statistically convincing (paired exact test p={pval:.3f} > {self.cfg.accept_p}); "
                                                   f"child wins on {b_wins} problems, parent on {c_wins}", level="warn")
        regress = self.retention_violations(res, par) if tolerance is None or tolerance >= 0 else []
        nll_lane, nll_par, nll_child = False, None, None
        if not accepted and tolerance is None and val_path and not regress and gain >= 0:        # cold-start lane: real progress before exact match moves
            nll_par, nll_child = self.held_out_nll(parent["path"], val_path), self.held_out_nll(path, val_path)
            if nll_par is not None and nll_child is not None and nll_par - nll_child >= self.cfg.nll_gain:
                accepted, nll_lane = True, True
                self.bus.emit("generation", self.name, f"accepted by the progress lane: held-out NLL {nll_par:.3f} -> {nll_child:.3f} with exact match not worse and retention intact", level="warn")
        if accepted and regress:
            accepted = False
            self.bus.emit("generation", self.name, f"RETENTION GUARD: child loses capabilities the parent had: {regress}", level="warn")
        detail = dict(res, parent_score=par["total"], gain=gain, p_value=pval, child_only=b_wins, parent_only=c_wins, retention_violations=regress,
                      progress_lane=nll_lane, nll_parent=nll_par, nll_child=nll_child,
                      data=info or {}, hyper=dict(self.hyper), candidate=label,
                      candidates={l: r["total"] for r, l, _ in scored})
        gdir = self.gen_root / f"gen{gen_no:03d}"
        final = gdir / "final"
        if accepted:
            shutil.rmtree(final, ignore_errors=True)
            shutil.move(path, final)
        gid = self.db.add_generation(parent["id"], str(final) if accepted else "", method, res["total"], detail, accepted,
                                     f"{notes_prefix}{label}: gain {gain:+.3f} vs parent on identical fresh problems")
        for _, _, pth in scored:
            shutil.rmtree(pth, ignore_errors=True)
        shutil.rmtree(gdir / "adapter", ignore_errors=True)
        lr = float(self.hyper["learning_rate"])
        if accepted:
            self.hyper["learning_rate"] = min(lr * 1.1, 2 * self.cfg.learning_rate)             # adaptive step size: succeed -> be a little bolder
            self.bus.emit("generation", self.name,
                          f"generation {gid} ACCEPTED ({label}): {par['total']:.3f} -> {res['total']:.3f} ({gain:+.3f})", {"id": gid, "score": res["total"]})
            self.prune_generations()
        else:
            self.hyper["learning_rate"] = max(lr * 0.5, 1e-6)                                  # fail -> halve the step size
            self.bus.emit("generation", self.name, f"generation {gid} REJECTED ({gain:+.3f}); rolled back to gen {parent['id']}; learning rate -> {self.hyper['learning_rate']:.2e}", level="warn")
            self._failure_analysis("generation rejected", {"gain": gain, "p_value": pval, "child_only_wins": b_wins, "parent_only_wins": c_wins,
                                                           "candidates": {l: r["total"] for r, l, _ in scored}, "hyper": self.hyper, "data": info,
                                                           "n_eval": res["n"], "min_gain": self.cfg.min_gain})
        return self.db.generations()[-1]

    def _failure_analysis(self, what: str, ctx: dict) -> None:
        t = self.ctx.thinker.think("failure_analysis", what, ctx)
        if t:
            notes = self.db.get_meta("failure_notes", [])
            notes.append(f"{what}: {t['decision']}")
            self.db.set_meta("failure_notes", notes[-8:])
            self.db.add_lesson("failure-analysis", f"{what}: {t['analysis'][:300]} => {t['decision']}")

    # ------------------------------------------------------------------------------------------ one improvement round
    @locked
    def improve(self, directives: list) -> dict | None:
        if not self.can_train:
            self.bus.emit("train", self.name, f"training impossible: {self.recipe.get('reason') if self.recipe else 'no intake'}", level="error")
            return None
        parent = self.best()
        if parent is None:
            parent = self.baseline()
        self.tune(directives)
        if self.cfg.require_approval:
            ok = self.control.ask_human("Start a training round?",
                                        f"Parent generation {parent['id']} (score {parent['score']:.3f})\n"
                                        f"Recipe: {self.recipe['reason']}\nHyper-parameters: {json.dumps(self.hyper)}\n"
                                        f"Trainable layers: {self.cfg.train_layers or 'all'}  frozen patterns: {self.cfg.freeze_patterns or 'none'}")
            if not ok:
                self.bus.emit("train", self.name, "human declined this training round", level="warn")
                return None
        gen_no = len(self.db.generations())
        gdir = self.gen_root / f"gen{gen_no:03d}"
        gdir.mkdir(parents=True, exist_ok=True)
        cands = []
        try:
            self.bus.status(self.name, "building dataset", role="training_maestro", parent="grand_maestro")
            info = self.build_dataset(parent["path"], gdir)
            h, rc = self.hyper, self.recipe
            algo = info["algorithm"]
            val_path = None
            if algo == "rft" and info["examples"] >= 40:                         # hold out rows that are NEVER trained on
                rows_all = [json.loads(l) for l in open(info["dataset"]) if l.strip()]
                random.Random(gen_no).shuffle(rows_all)
                nv = max(8, int(self.cfg.val_fraction * len(rows_all)))
                val, tr = [r for r in rows_all[:nv] if r.get("src") != "replay"] or rows_all[:nv], rows_all[nv:]
                (gdir / "val.jsonl").write_text("\n".join(json.dumps(r) for r in val) + "\n")
                Path(info["dataset"]).write_text("\n".join(json.dumps(r) for r in tr) + "\n")
                val_path = str(gdir / "val.jsonl")
            self.bus.status(self.name, f"training weights ({algo})", role="training_maestro", parent="grand_maestro")
            full = rc["method"] == "full"
            epochs = h["epochs"]
            self.run_worker({"rft": "train", "grpo": "grpo", "dpo": "dpo"}[algo], {
                "model_path": parent["path"], "dataset_path": info["dataset"],
                "lora_variant": self.cfg.lora_variant, "lora_plus_ratio": self.cfg.lora_plus_ratio, "optimizer": self.cfg.optimizer,
                "out_dir": str(gdir / ("child" if full else "adapter")),
                "method": rc["method"], "dtype": rc["dtype"], "device": rc["device"], "grad_checkpointing": rc["grad_checkpointing"],
                "epochs": epochs, "learning_rate": h["learning_rate"], "lora_r": h["lora_r"], "lora_alpha": 2 * int(h["lora_r"]),
                "max_len": self.cfg.max_seq_len, "micro_batch": self.cfg.micro_batch, "grad_accum": self.cfg.grad_accum,
                "seed": gen_no, "train_layers": list(self.cfg.train_layers), "freeze_patterns": list(self.cfg.freeze_patterns)}, algo)
            if not full:
                self.bus.status(self.name, "merging", role="training_maestro", parent="grand_maestro")
                self.run_worker("merge", {"model_path": parent["path"], "adapter": str(gdir / "adapter"), "out_dir": str(gdir / "child"),
                                          "dtype": rc["dtype"] if rc["device"] == "cuda" else "float32"}, "merge")
            cands.append(("child", str(gdir / "child")))
            if self.cfg.try_merge_candidates:       # weight interpolation between parent and child (model soup / WiSE-FT)
                try:
                    self.bus.status(self.name, "weight-interpolation candidate", role="training_maestro", parent="grand_maestro")
                    self.run_worker("mergemodels", {"mode": "linear", "out_dir": str(gdir / "soup"),
                                                    "models": [{"path": self.local_path(parent["path"]), "weight": 0.5},
                                                               {"path": str(gdir / "child"), "weight": 0.5}]}, "soup")
                    cands.append(("soup(0.5 parent + 0.5 child)", str(gdir / "soup")))
                except TrainingError as e:
                    self.bus.emit("train", self.name, f"soup candidate skipped: {str(e)[:160]}", level="warn")
        except TrainingError as e:
            self.bus.emit("train", self.name, f"training round failed: {e}", level="error")
            self.db.add_generation(parent["id"], "", "failed", 0.0, {"error": str(e)[:1500]}, False, "training failed")
            self._failure_analysis("training round failed", {"error": str(e)[-700:], "recipe": self.recipe, "hyper": self.hyper})
            shutil.rmtree(gdir / "child", ignore_errors=True)
            return None
        return self._judge(parent, cands, gen_no, f"{info['algorithm']}-{'full' if self.recipe['method'] == 'full' else self.recipe['method']}", info, val_path=val_path)

    # ------------------------------------------------------------------------------------------ architecture conversion
    @locked
    def convert_architecture(self, kv_heads: int = 0, cla_factor: int = 1, cla_init: str = "mean", heal: bool = True) -> dict | None:
        """Convert the best model to another attention architecture (GQA / MQA / Cross-Layer Attention), optionally heal it with a short
        training run on the parent's verified samples, and keep it only if it is NOT WORSE than the parent by more than cfg.arch_tolerance
        (the point of the change is memory: KV cache / f). Candidates: the raw conversion and the healed model."""
        parent = self.best() or self.baseline()
        tpath = self.local_path(parent["path"])
        gen_no = len(self.db.generations())
        gdir = self.gen_root / f"gen{gen_no:03d}"
        gdir.mkdir(parents=True, exist_ok=True)
        label = f"kv_heads={kv_heads or 'same'}, CLA factor={cla_factor}"
        self.bus.emit("transplant", self.name, f"architecture conversion ({label}) of gen {parent['id']}", level="warn")
        cands = []
        try:
            self.run_worker("convert_arch", {"model_path": tpath, "out_dir": str(gdir / "converted"), "kv_heads": kv_heads,
                                             "cla_factor": cla_factor, "cla_init": cla_init}, "convert architecture")
            cands.append(("converted (no healing)", str(gdir / "converted")))
            if heal and self.can_train:
                info = self.build_dataset(parent["path"], gdir)
                rc, h = self.recipe, self.hyper
                full = rc["method"] == "full"
                self.run_worker("train", {"model_path": str(gdir / "converted"), "dataset_path": info["dataset"] if info["algorithm"] == "rft" else "",
                                          "out_dir": str(gdir / ("healed" if full else "adapter")), "method": rc["method"], "dtype": rc["dtype"],
                                          "device": rc["device"], "grad_checkpointing": rc["grad_checkpointing"], "epochs": max(2.0, h["epochs"]),
                                          "learning_rate": h["learning_rate"], "lora_r": h["lora_r"], "lora_alpha": 2 * int(h["lora_r"]),
                                          "max_len": self.cfg.max_seq_len, "micro_batch": self.cfg.micro_batch, "grad_accum": self.cfg.grad_accum,
                                          "seed": gen_no, "train_layers": [], "freeze_patterns": [], "lora_variant": self.cfg.lora_variant}, "heal")
                if not full:
                    self.run_worker("merge", {"model_path": str(gdir / "converted"), "adapter": str(gdir / "adapter"), "out_dir": str(gdir / "healed"),
                                              "dtype": rc["dtype"] if rc["device"] == "cuda" else "float32"}, "merge healing")
                cands.append(("converted + healed", str(gdir / "healed")))
        except TrainingError as e:
            self.bus.emit("transplant", self.name, f"architecture conversion failed: {e}", level="error")
            shutil.rmtree(gdir, ignore_errors=True)
            return None
        cfgj = json.load(open(Path(cands[0][1]) / "config.json"))
        info2 = {"architecture": cfgj.get("architectures"), "kv_heads": cfgj.get("num_key_value_heads"), "cla_factor": cfgj.get("cla_factor", 1)}
        return self._judge(parent, cands, gen_no, "arch-convert", info2, f"{label}; ", tolerance=self.cfg.arch_tolerance)

    # ------------------------------------------------------------------------------------------ transplant (brain surgery)
    @locked
    def transplant(self, donor_ref: str, selection: list, label: str = "transplant") -> dict | None:
        """Copy / blend / resize chosen parts of a DONOR model into the best model. The result is only kept if it
        beats the current model on fresh hidden problems."""
        parent = self.best() or self.baseline()
        tpath, dpath = self.local_path(parent["path"]), self.local_path(donor_ref)
        ops = brainmap.build_ops(brainmap.read_structure(tpath), brainmap.read_structure(dpath), selection)
        gen_no = len(self.db.generations())
        gdir = self.gen_root / f"gen{gen_no:03d}"
        gdir.mkdir(parents=True, exist_ok=True)
        self.bus.emit("transplant", self.name, f"{label}: {len(ops)} tensors from {donor_ref} into gen {parent['id']}", {"ops": ops[:50]}, level="warn")
        try:
            self.run_worker("transplant", {"target_path": tpath, "donor_path": dpath, "out_dir": str(gdir / "child"), "ops": ops}, "transplant")
        except TrainingError as e:
            self.bus.emit("transplant", self.name, f"transplant failed: {e}", level="error")
            shutil.rmtree(gdir, ignore_errors=True)
            return None
        return self._judge(parent, [(label, str(gdir / "child"))], gen_no, "transplant", {"donor": donor_ref, "ops": len(ops)}, f"donor {donor_ref}; ")

    @locked
    def merge_with(self, other_ref: str, mode: str = "linear", weight: float = 0.5) -> dict | None:
        """Merge the best model with another model of the same architecture (linear | slerp | ties | task)."""
        parent = self.best() or self.baseline()
        tpath, opath = self.local_path(parent["path"]), self.local_path(other_ref)
        gen_no = len(self.db.generations())
        gdir = self.gen_root / f"gen{gen_no:03d}"
        gdir.mkdir(parents=True, exist_ok=True)
        args = {"mode": mode, "out_dir": str(gdir / "child"),
                "models": [{"path": tpath, "weight": 1 - weight}, {"path": opath, "weight": weight}]}
        if mode in ("task", "ties"):
            args["base"] = tpath
        try:
            self.run_worker("mergemodels", args, f"merge {mode}")
        except TrainingError as e:
            self.bus.emit("transplant", self.name, f"merge failed: {e}", level="error")
            shutil.rmtree(gdir, ignore_errors=True)
            return None
        return self._judge(parent, [(f"merge-{mode}", str(gdir / "child"))], gen_no, "merge", {"other": other_ref, "mode": mode, "weight": weight})

    # ------------------------------------------------------------------------------------------ goal
    def goal_reached(self) -> bool:
        b = self.best()
        if not b or b["score"] < self.cfg.target_score:
            return False
        self.bus.emit("goal", self.name, f"best score {b['score']:.3f} >= target; confirming on {self.cfg.confirm_rounds} independent fresh sets")
        for i in range(self.cfg.confirm_rounds):
            probs = self.ctx.evalcore.make_set("confirm", self.cfg.eval_problems)
            r = self.evaluate(b["path"], probs, f"goal confirmation {i + 1}/{self.cfg.confirm_rounds}")
            if r["total"] < self.cfg.target_score:
                self.bus.emit("goal", self.name, f"confirmation failed ({r['total']:.3f} < {self.cfg.target_score}); continuing", level="warn")
                return False
        return True
