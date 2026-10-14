"""Orchestrator: builds the run, then loops until one of exactly three things happens:
GOAL_REACHED, BUDGET_EXHAUSTED, HUMAN_STOP. Every other failure is logged and retried."""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path

from .agents import Context
from .bus import EventBus
from .config import Config
from .control import RunControl, StopReason, Stopped
from .db import DB
from .evalcore import EvalCore
from .llm import LLM, LLMError
from .maestros import GrandMaestro, ResearchMaestro, SafetyMaestro, FrontierMaestro
from .backup import BackupManager
from .bootstrap import Bootstrap
from .thinking import Thinker, RedTeam
from .challenges import ChallengeEngine
from .society import SocietyRunner
from .quest import QuestMaestro
from .sandbox import make_sandbox, SandboxUnavailable
from .trainer import TrainingMaestro, TrainingError
from .scratch import ScratchBuilder
from .verifier import Verifier
from .web import WebTools
from .workshop import Workshop


class Orchestrator:
    def __init__(self, cfg: Config, run_dir: str | Path | None = None):
        cfg.validate()
        self.cfg = cfg
        if run_dir is None:
            run_dir = Path(cfg.run_root) / time.strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(self.run_dir / "config.json")
        self.bus = EventBus(self.run_dir / "events.jsonl")
        self.db = DB(self.run_dir / "sira.db")
        self.control = RunControl(cfg, self.db, self.bus)
        self.ctx: Context | None = None
        self.thread: threading.Thread | None = None
        self.ready = threading.Event()
        self.error: str | None = None
        self.cycle = int(self.db.get_meta("cycle", 0))

    # ---------------------------------------------------------------- build
    def build(self) -> None:
        cfg, bus = self.cfg, self.bus
        bus.status("orchestrator", "building", role="orchestrator")
        self.bootstrap = Bootstrap(cfg, bus)
        self.env_report = self.bootstrap.run()                     # probe machine, install what is missing, adapt settings
        self.db.set_meta("environment", {k: v for k, v in self.env_report.items() if k != "algorithms"})
        llm = LLM(cfg, self.control)
        llm.log_path = self.run_dir / "llm_calls.jsonl"
        sandbox = make_sandbox(cfg, self.run_dir, self.control)
        web = WebTools(cfg, self.control, sandbox.data_dir)
        ec = EvalCore(self.run_dir, self.db, cfg.domains)
        ctx = Context(cfg=cfg, control=self.control, bus=bus, db=self.db, llm=llm, sandbox=sandbox, web=web,
                      evalcore=ec, verifier=None, run_dir=self.run_dir,
                      exp_sem=threading.Semaphore(max(1, cfg.parallel_experiments)))
        ctx.verifier = Verifier(ctx)
        ctx.workshop = Workshop(ctx)
        ctx.thinker = Thinker(ctx)
        self.ctx = ctx
        self.grand = GrandMaestro(ctx)
        self.research = ResearchMaestro(ctx)
        self.training = TrainingMaestro(ctx)
        self.safety = SafetyMaestro(ctx)
        self.frontier = FrontierMaestro(ctx)
        self.redteam = RedTeam(ctx)
        self.challenges = ChallengeEngine(ctx, self.research)
        ctx.challenges = self.challenges
        self.society = SocietyRunner(ctx)
        self.quest = QuestMaestro(ctx, self.research, self.challenges, self.society)
        self.model_enabled = self.cfg.mode == "model" or (self.cfg.mode in ("both", "auto") and bool(self.cfg.base_model or self.cfg.scratch_mode))
        self.backup = BackupManager(ctx, self.run_dir)
        for n, r, p in (("grand_maestro", "grand_maestro", ""), ("research_maestro", "research_maestro", "grand_maestro"),
                        ("training_maestro", "training_maestro", "grand_maestro"),
                        ("safety_maestro", "safety_maestro", "grand_maestro")):
            bus.status(n, "idle", role=r, parent=p)
        self.env_report["sandbox_libs"] = self.bootstrap.sandbox_libs(sandbox)
        bus.emit("bootstrap", "bootstrap", f"sandbox science libraries: {self.env_report['sandbox_libs']}", {"report": self.env_report})
        ok, msg = llm.check()
        bus.emit("llm", "llm", ("brain OK: " if ok else "BRAIN PROBLEM: ") + msg, level="info" if ok else "error")
        rep = self.safety.run_selftest()
        if not rep["passed"]:
            raise SandboxUnavailable("sandbox self-test failed; see the Sandbox tab. Experiments will NOT run on an unsafe sandbox.")
        self.ready.set()

    # ---------------------------------------------------------------- main loop
    def start(self) -> None:
        self.thread = threading.Thread(target=self.run, name="orchestrator", daemon=True)
        self.thread.start()

    def run(self) -> None:
        bus = self.bus
        try:
            try:
                self.build()
                bus.emit("run", "orchestrator", f"run started in {self.run_dir}")
                self._initial_phase()
                while True:
                    self.control.checkpoint()
                    self._cycle()
            except Stopped:
                pass
            except (SandboxUnavailable, TrainingError, ValueError) as e:     # setup problems: show and wait for the human
                self.error = str(e)
                bus.emit("fatal", "orchestrator", f"SETUP PROBLEM: {e}", level="error")
                bus.emit("fatal", "orchestrator", "Fix the problem and start a new run, or press STOP.", level="error")
                while not self.control.stopped:
                    time.sleep(0.5)
        finally:
            self._finalize()

    def _initial_phase(self) -> None:
        if not self.model_enabled:
            self.bus.emit("run", "orchestrator", f"mode '{self.cfg.mode}' without a model: research, challenges, exams and society only")
            return
        if self.cfg.scratch_mode:
            self.training.prefer_full = True
            self.bus.status("scratch_builder", "starting", role="scratch_builder", parent="training_maestro")
            while True:
                try:
                    path = self.db.get_meta("scratch_model") or ScratchBuilder(self.training).build()
                    self.cfg.base_model = path
                    from .intake import detect_hardware
                    if not detect_hardware().get("cuda"):
                        self.cfg.allow_cpu_training = True
                    break
                except TrainingError as e:
                    self.bus.emit("error", "orchestrator", f"scratch build failed: {e}", level="error")
                    self.control.sleep(60)
        while True:
            try:
                self.training.do_intake()
                if self.training.can_train and not self.db.generations():
                    self.training.baseline()
                return
            except LLMError:
                raise
            except TrainingError as e:
                if "no base_model" in str(e):
                    raise
                bus_msg = f"intake/baseline failed: {e}"
                self.bus.emit("error", "orchestrator", bus_msg, level="error")
                self.control.sleep(60)
            except Exception as e:
                self.bus.emit("error", "orchestrator", f"intake/baseline error: {type(e).__name__}: {e}", level="error")
                self.control.sleep(60)

    def _cycle(self) -> None:
        bus, c = self.bus, self.control
        self.cycle += 1
        self.db.set_meta("cycle", self.cycle)
        try:
            self.safety.check_cycle()
            directives = c.pop_directives()
            weak = self.training.weak_domains()
            if self.cfg.mode != "model":
                st = self.quest.cycle()
                if st["passed"] and (not self.model_enabled or self.cfg.mode == "auto"):
                    c.request_stop(StopReason.GOAL_REACHED)
                    raise Stopped("quest passed")
            if not self.model_enabled and self.cfg.mode != "model":
                c.persist()
                self._reflect()
                self.backup.maybe_auto()
                return
            plan = self.grand.plan(self.cycle, directives, weak)
            dtext = " | ".join(directives)
            for mt in plan.get("math_topics", []):
                c.checkpoint()
                try:
                    self.research.math(mt)
                except LLMError as e:
                    bus.emit("error", "research_maestro", f"LLM error: {e}", level="error")
                    c.sleep(30)
            for topic in plan["research_topics"]:
                c.checkpoint()
                try:
                    self.research.research(topic, dtext)
                except LLMError as e:
                    bus.emit("error", "research_maestro", f"LLM error: {e}", level="error")
                    c.sleep(30)
            force = self.cycle % max(1, self.cfg.force_train_every) == 0      # always keep teaching itself
            if (plan["train"] or force) and self.training.can_train:
                before = max([g["id"] for g in self.db.generations() if g["accepted"]] or [-1])
                self.training.improve(directives)
                after = max([g["id"] for g in self.db.generations() if g["accepted"]] or [-1])
                self.backup.maybe_auto()
                if after != before:
                    self.frontier.report(f"milestone: generation {after} accepted")
                    ev = self.db.evals(1)
                    if ev:
                        self.redteam.attack(ev[0]["id"], ev[0]["label"])
                if self.training.goal_reached() and (self.cfg.mode != "both" or (self.db.get_meta("quest_state", {}) or {}).get("passed")):
                    c.request_stop(StopReason.GOAL_REACHED)
                    raise Stopped("goal reached")
            elif plan["train"]:
                bus.emit("train", "training_maestro", f"cannot train: {self.training.recipe.get('reason')}", level="error")
                c.sleep(30)
            self._reflect()
            if self.cycle % max(1, self.cfg.report_every_cycles) == 0:
                self.frontier.report(f"periodic report (cycle {self.cycle})")
            self.backup.maybe_auto()
            c.persist()
        except Stopped:
            raise
        except LLMError as e:
            bus.emit("error", "orchestrator", f"LLM error (will retry): {e}", level="error")
            c.sleep(60)
        except Exception as e:
            bus.emit("error", "orchestrator", f"cycle {self.cycle} error: {type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}", level="error")
            c.sleep(30)

    def _reflect(self) -> None:
        """Experience Memory layer: distil the cycle into lessons."""
        d = self.db
        ctx = {"cycle": self.cycle, "recent_claims": [{"status": c["status"], "text": c["text"][:140]} for c in d.claims(None, 8)],
               "recent_mistakes": [m["description"][:140] for m in d.mistakes(5)],
               "recent_generations": [{"id": g["id"], "method": g["method"], "score": g["score"], "accepted": bool(g["accepted"]), "notes": g["notes"][:100]}
                                      for g in d.generations()[-3:]]}
        t = self.ctx.thinker.think("memory", "Which durable lessons follow from this cycle?", ctx)
        if t:
            d.add_lesson("reflection", f"cycle {self.cycle}: {t['decision']} | analysis: {t['analysis'][:300]}")

    def _finalize(self) -> None:
        c = self.control
        c.kill_all()
        c.persist()
        try:
            if getattr(self, "backup", None):
                self.backup.maybe_auto(force=True)
        except Exception:
            pass
        best = self.training.best() if getattr(self, "training", None) else None
        report = {"stop_reason": c.stop_reason.value if c.stop_reason else None, "cycles": self.cycle,
                  "best_generation": best, "budget": {k: v for k, v in c.budget_status().items()},
                  "verified_claims": len(self.db.claims("verified", 100000)),
                  "refuted_claims": len(self.db.claims("refuted", 100000)),
                  "generations": [{k: g[k] for k in ("id", "parent", "method", "score", "accepted", "notes")} for g in self.db.generations()],
                  "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        (self.run_dir / "final_report.json").write_text(json.dumps(report, indent=2, default=str))
        self.bus.emit("run", "orchestrator", f"RUN ENDED: {report['stop_reason']}. Report: {self.run_dir / 'final_report.json'}", level="warn")
        self.bus.status("orchestrator", f"ended ({report['stop_reason']})", role="orchestrator")

    def stop(self) -> None:
        self.control.request_stop(StopReason.HUMAN_STOP)
