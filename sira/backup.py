"""Backups: consistent snapshots of everything that matters (database, config, station, reports, best weights),
each with a sha256 manifest. Restoring always happens into a NEW run folder, so a snapshot can never be damaged.
Weights are hard-linked when possible (instant, no extra disk) - generation folders are immutable once written."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clone(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class BackupManager:
    def __init__(self, ctx, run_dir):
        self.ctx, self.run_dir = ctx, Path(run_dir)
        self.root = self.run_dir / "backups"
        self.root.mkdir(exist_ok=True)
        self.last_time = 0.0
        self.last_gen_seen = -1

    # ------------------------------------------------------------------ create
    def create(self, label: str = "auto", include_weights: bool | None = None) -> Path | None:
        ctx, cfg = self.ctx, self.ctx.cfg
        include_weights = cfg.backup_weights if include_weights is None else include_weights
        name = time.strftime("%Y%m%d_%H%M%S") + "_" + "".join(c for c in label if c.isalnum() or c in "-_")[:30]
        dest = self.root / name
        try:
            dest.mkdir()
            ctx.control.persist()
            src = ctx.db.conn
            with ctx.db.lock:
                dst = sqlite3.connect(str(dest / "sira.db"))
                src.backup(dst)
                dst.close()
            shutil.copy2(self.run_dir / "config.json", dest / "config.json")
            if (self.run_dir / "evaluator.key").exists():
                shutil.copy2(self.run_dir / "evaluator.key", dest / "evaluator.key")
                os.chmod(dest / "evaluator.key", 0o600)
            for sub in ("station", "reports"):
                p = self.run_dir / sub
                if p.exists():
                    shutil.copytree(p, dest / sub, symlinks=False, ignore=shutil.ignore_patterns("*.part"))
            gens = ctx.db.generations()
            acc = [g for g in gens if g["accepted"]]
            best = acc[-1] if acc else None
            weights_info = None
            if include_weights and best and best["path"] and os.path.isdir(best["path"]) and str(best["path"]).startswith(str(self.run_dir)):
                for f in Path(best["path"]).rglob("*"):
                    if f.is_file():
                        _clone(f, dest / "weights" / f.relative_to(best["path"]))
                weights_info = {"generation": best["id"]}
            manifest = {"label": label, "created": time.time(), "cycle": ctx.db.get_meta("cycle", 0), "best_generation": best["id"] if best else None,
                        "best_score": best["score"] if best else None, "weights": weights_info,
                        "spent_usd": ctx.control.spent_usd, "files": {}}
            for f in sorted(dest.rglob("*")):
                if f.is_file() and f.name != "manifest.json":
                    manifest["files"][str(f.relative_to(dest))] = _sha(f)
            (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
        except Exception as e:
            ctx.bus.emit("backup", "backup", f"backup FAILED: {type(e).__name__}: {e}", level="error")
            shutil.rmtree(dest, ignore_errors=True)
            return None
        self.last_time = time.time()
        self.prune()
        ctx.bus.emit("backup", "backup", f"backup '{name}' ok ({len(manifest['files'])} files, weights={bool(weights_info)})")
        return dest

    def prune(self) -> None:
        keep = max(2, int(self.ctx.cfg.backup_keep))
        auto = [p for p in sorted(self.root.iterdir()) if p.is_dir() and not p.name.split("_", 2)[-1].startswith("manual")]
        for p in auto[:-keep]:
            shutil.rmtree(p, ignore_errors=True)

    def maybe_auto(self, force: bool = False) -> None:
        gens = self.ctx.db.generations()
        newest_acc = max([g["id"] for g in gens if g["accepted"]] or [-1])
        due = (time.time() - self.last_time) >= self.ctx.cfg.backup_every_min * 60
        if force or due or newest_acc != self.last_gen_seen:
            self.last_gen_seen = newest_acc
            self.create("auto" if not force else "final")

    # ------------------------------------------------------------------ inspect / verify
    def list(self) -> list:
        out = []
        for p in sorted(self.root.iterdir(), reverse=True):
            mf = p / "manifest.json"
            if mf.exists():
                m = json.loads(mf.read_text())
                out.append({"name": p.name, "path": str(p), "label": m["label"], "created": m["created"], "best_generation": m["best_generation"],
                            "best_score": m["best_score"], "weights": bool(m["weights"]), "files": len(m["files"])})
        return out

    @staticmethod
    def verify(snapshot: str | Path) -> tuple[bool, list]:
        snap = Path(snapshot)
        m = json.loads((snap / "manifest.json").read_text())
        bad = [rel for rel, h in m["files"].items() if not (snap / rel).exists() or _sha(snap / rel) != h]
        return not bad, bad

    # ------------------------------------------------------------------ restore / rollback
    @staticmethod
    def restore_to_new_run(snapshot: str | Path, new_run_dir: str | Path) -> Path:
        """Create a fresh run folder from a verified snapshot. Start it with Orchestrator(cfg, run_dir=new_run_dir)."""
        snap, new = Path(snapshot), Path(new_run_dir)
        ok, bad = BackupManager.verify(snap)
        if not ok:
            raise ValueError(f"snapshot is corrupted: {bad[:3]}")
        if new.exists() and any(new.iterdir()):
            raise ValueError("target run folder must be empty")
        new.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snap / "sira.db", new / "sira.db")
        shutil.copy2(snap / "config.json", new / "config.json")
        for sub in ("station", "reports"):
            if (snap / sub).exists():
                shutil.copytree(snap / sub, new / sub)
        m = json.loads((snap / "manifest.json").read_text())
        if m["weights"]:
            dst = new / "generations" / "restored" / "final"
            for f in (snap / "weights").rglob("*"):
                if f.is_file():
                    _clone(f, dst / f.relative_to(snap / "weights"))
            con = sqlite3.connect(str(new / "sira.db"))
            con.execute("UPDATE generations SET path=? WHERE id=?", (str(dst), m["best_generation"]))
            con.commit()
            con.close()
        if (snap / "evaluator.key").exists():          # same key => train/eval namespaces stay disjoint
            shutil.copy2(snap / "evaluator.key", new / "evaluator.key")
            os.chmod(new / "evaluator.key", 0o600)
        return new

    def rollback_to(self, gen_id: int) -> str:
        """Mark every generation after gen_id as not accepted, so gen_id becomes the best model again."""
        gens = {g["id"]: g for g in self.ctx.db.generations()}
        g = gens.get(int(gen_id))
        if not g or not g["accepted"]:
            raise ValueError("that generation does not exist or was never accepted")
        if g["method"] != "base" and not os.path.isdir(g["path"]):
            raise ValueError("the weights of that generation were pruned (keep_generations); restore from a backup instead")
        self.ctx.db._x("UPDATE generations SET accepted=0, notes=notes || ' [rolled back by human]' WHERE id>? AND accepted=1", (int(gen_id),))
        self.ctx.bus.emit("backup", "human", f"rolled back to generation {gen_id}", level="warn")
        return g["path"]
