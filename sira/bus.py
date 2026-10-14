"""Thread-safe event bus: the GUI reads it, a JSONL file records it (full audit trail)."""
from __future__ import annotations

import json
import queue
import threading
import time


class EventBus:
    def __init__(self, log_path=None):
        self.q: "queue.Queue[dict]" = queue.Queue()
        self._lock = threading.Lock()
        self._fh = open(log_path, "a", encoding="utf-8") if log_path else None
        self.agent_status: dict[str, dict] = {}

    def emit(self, kind: str, agent: str, msg: str, data: dict | None = None, level: str = "info") -> None:
        ev = {"t": time.time(), "kind": kind, "agent": agent, "level": level, "msg": msg, "data": data or {}}
        with self._lock:
            if self._fh:
                try:
                    self._fh.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
                    self._fh.flush()
                except Exception:
                    pass
            if kind == "agent_status":
                self.agent_status[agent] = {"status": msg, "t": ev["t"], **(data or {})}
        self.q.put(ev)

    def status(self, agent: str, state: str, **data) -> None:
        self.emit("agent_status", agent, state, data)

    def drain(self, limit: int = 500) -> list:
        out = []
        while len(out) < limit:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                break
        return out

    def close(self) -> None:
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None
