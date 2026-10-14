"""RunControl: the ONLY authority on stopping.

The run can end for exactly three reasons: GOAL_REACHED, BUDGET_EXHAUSTED, HUMAN_STOP.
Agents never receive a handle that can change limits or stop reasons.
"""
from __future__ import annotations

import os
import queue
import signal
import threading
import time
from enum import Enum


class StopReason(str, Enum):
    GOAL_REACHED = "goal_reached"
    BUDGET_EXHAUSTED = "budget_exhausted"
    HUMAN_STOP = "human_stop"


class Stopped(Exception):
    """Raised inside any worker when the run is ending."""


class RunControl:
    def __init__(self, cfg, db, bus):
        self.cfg, self.db, self.bus = cfg, db, bus
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._run = threading.Event()      # set => running, clear => paused
        self._run.set()
        self.stop_reason: StopReason | None = None
        self.limits = {"usd": cfg.budget_usd, "gpu_h": cfg.budget_gpu_hours, "wall_h": cfg.budget_wall_hours}
        self.spent_usd = float(db.get_meta("spent_usd", 0.0))
        self.spent_gpu_s = float(db.get_meta("spent_gpu_s", 0.0))
        self._wall_prior = float(db.get_meta("wall_s", 0.0))
        self._t0 = time.time()
        self.tokens_in = int(db.get_meta("tokens_in", 0))
        self.tokens_out = int(db.get_meta("tokens_out", 0))
        self.directives: "queue.Queue[str]" = queue.Queue()
        self.approvals: "queue.Queue[dict]" = queue.Queue()
        self._procs: set = set()
        self.integrity_alert: str | None = None

    # ---- stop / pause --------------------------------------------------------
    def request_stop(self, reason: StopReason) -> None:
        with self._lock:
            if self.stop_reason is None:
                self.stop_reason = reason
                self.bus.emit("control", "control", f"STOP: {reason.value}", level="warn")
                self._stop.set()
                self._run.set()          # wake paused workers so they can exit
        self.kill_all()
        self.persist()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    @property
    def paused(self) -> bool:
        return not self._run.is_set()

    def pause(self) -> None:
        if not self.stopped:
            self._run.clear()
            self.bus.emit("control", "control", "PAUSED by human", level="warn")

    def resume(self) -> None:
        self.integrity_alert = None
        self._run.set()
        self.bus.emit("control", "control", "RESUMED", level="info")

    def checkpoint(self) -> None:
        """Call frequently. Raises Stopped on stop, blocks while paused."""
        self.check_budget()
        while True:
            if self._stop.is_set():
                raise Stopped(self.stop_reason.value if self.stop_reason else "stopped")
            if self._run.wait(timeout=0.25):
                if self._stop.is_set():
                    raise Stopped(self.stop_reason.value if self.stop_reason else "stopped")
                return

    def sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            self.checkpoint()
            time.sleep(min(0.25, max(0.0, end - time.time())))

    # ---- budget --------------------------------------------------------------
    @property
    def wall_s(self) -> float:
        return self._wall_prior + (time.time() - self._t0)

    def add_cost(self, usd: float = 0.0, gpu_s: float = 0.0, tin: int = 0, tout: int = 0) -> None:
        with self._lock:
            self.spent_usd += usd
            self.spent_gpu_s += gpu_s
            self.tokens_in += tin
            self.tokens_out += tout
        self.check_budget()

    def set_limits(self, usd=None, gpu_h=None, wall_h=None) -> None:
        with self._lock:
            if usd is not None and usd > 0:
                self.limits["usd"] = float(usd)
            if gpu_h is not None and gpu_h > 0:
                self.limits["gpu_h"] = float(gpu_h)
            if wall_h is not None and wall_h > 0:
                self.limits["wall_h"] = float(wall_h)
        self.bus.emit("control", "human", f"budget limits now {self.limits}", level="warn")

    def budget_status(self) -> dict:
        return {
            "usd": (self.spent_usd, self.limits["usd"]),
            "gpu_h": (self.spent_gpu_s / 3600, self.limits["gpu_h"]),
            "wall_h": (self.wall_s / 3600, self.limits["wall_h"]),
            "tokens": (self.tokens_in, self.tokens_out),
        }

    def check_budget(self) -> None:
        b = self.budget_status()
        if any(b[k][0] >= b[k][1] for k in ("usd", "gpu_h", "wall_h")):
            self.request_stop(StopReason.BUDGET_EXHAUSTED)

    def persist(self) -> None:
        self.db.set_meta("spent_usd", self.spent_usd)
        self.db.set_meta("spent_gpu_s", self.spent_gpu_s)
        self.db.set_meta("wall_s", self.wall_s)
        self.db.set_meta("tokens_in", self.tokens_in)
        self.db.set_meta("tokens_out", self.tokens_out)

    # ---- human interaction ------------------------------------------------------
    def send_directive(self, text: str) -> None:
        self.directives.put(text)
        self.bus.emit("human", "human", f"directive: {text}", level="warn")

    def pop_directives(self) -> list:
        out = []
        while True:
            try:
                out.append(self.directives.get_nowait())
            except queue.Empty:
                return out

    def ask_human(self, title: str, detail: str) -> bool:
        """Blocks until the human answers in the GUI (or the run stops)."""
        box = {"title": title, "detail": detail, "answer": None, "ev": threading.Event()}
        self.approvals.put(box)
        self.bus.emit("approval", "control", f"awaiting human approval: {title}", level="warn")
        while not box["ev"].wait(0.25):
            if self._stop.is_set():
                raise Stopped("stopped while awaiting approval")
        return bool(box["answer"])

    # ---- child processes ---------------------------------------------------------
    def register_proc(self, p) -> None:
        with self._lock:
            self._procs.add(p)
        if self._stop.is_set():
            self._kill(p)

    def unregister_proc(self, p) -> None:
        with self._lock:
            self._procs.discard(p)

    @staticmethod
    def _kill(p) -> None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    def kill_all(self) -> None:
        with self._lock:
            procs = list(self._procs)
        for p in procs:
            self._kill(p)
