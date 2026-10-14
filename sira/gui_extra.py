"""Additional GUI tabs (mixin for App): Brain Map, Worlds & Tools, Computer, Tests, Concepts & Reports,
Backups, Network & Policy, Statistics. Long operations run in background threads."""
from __future__ import annotations

import json
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

from . import brainmap
from .backup import BackupManager

METRICS = ("norm", "std", "absmax", "frac_small", "delta_rel")
CELL_W, CELL_H, GAP = 34, 20, 4


def _color(v, lo, hi):
    if v is None:
        return "#d9d9d9"
    t = 0.0 if hi <= lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))
    r = int(235 - 215 * t)
    g = int(240 - 150 * t)
    b = int(250 - 40 * t)
    return f"#{r:02x}{g:02x}{b:02x}"


class ExtraTabs:
    # ------------------------------------------------------------------ helpers
    def _bg(self, fn, label):
        def run():
            try:
                self.orch.bus.emit("gui", "human", f"{label} started", level="warn")
                fn()
                self.orch.bus.emit("gui", "human", f"{label} finished")
            except Exception as e:
                self.orch.bus.emit("error", "human", f"{label} failed: {type(e).__name__}: {e}", level="error")
        threading.Thread(target=run, daemon=True).start()

    def _need_run(self) -> bool:
        if not self.orch or not self.orch.ready.is_set():
            messagebox.showinfo("Not ready", "Start a run first and wait until the sandbox self-test finishes.")
            return False
        return True

    def _extra_tabs(self):
        self._tab_brain()
        self._tab_worlds()
        self._tab_computer()
        self._tab_tests()
        self._tab_concepts()
        self._tab_backups()
        self._tab_policy()
        self._tab_stats()
        self._tab_env()
        self._tab_thinking()
        self._tab_quest()
        self._tab_challenges()
        self._tab_society()
        self._extra_last = 0.0

    # =================================================================== BRAIN MAP
    def _tab_brain(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Brain map")
        bar = ttk.Frame(t)
        bar.pack(fill="x", padx=8, pady=4)
        ttk.Button(bar, text="Load map", command=self._brain_load).pack(side="left")
        ttk.Button(bar, text="Scan statistics", command=self._brain_scan).pack(side="left", padx=4)
        ttk.Label(bar, text="Color by:").pack(side="left", padx=(10, 2))
        self.metric = tk.StringVar(value="norm")
        cb = ttk.Combobox(bar, textvariable=self.metric, values=METRICS, state="readonly", width=11)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self._brain_draw())
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(bar, text="Train ONLY selected layers", command=lambda: self._layers_mode("only")).pack(side="left")
        ttk.Button(bar, text="Freeze selected", command=lambda: self._layers_mode("freeze")).pack(side="left", padx=4)
        ttk.Button(bar, text="Train all", command=lambda: self._layers_mode("all")).pack(side="left")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(bar, text="Transplant from donor model...", command=self._transplant_dialog).pack(side="left")
        ttk.Button(bar, text="Merge with model...", command=self._merge_dialog).pack(side="left", padx=4)
        ttk.Button(bar, text="Convert architecture (CLA / GQA / MQA)...", command=self._arch_dialog).pack(side="left", padx=4)
        self.brain_state = ttk.Label(t, text="Click a layer header to select/deselect the layer, click a cell to inspect one tensor.")
        self.brain_state.pack(fill="x", padx=8)
        fr = ttk.Frame(t)
        fr.pack(fill="both", expand=True, padx=8, pady=4)
        self.brain_cv = tk.Canvas(fr, bg="white", height=330, scrollregion=(0, 0, 2000, 330))
        hs = ttk.Scrollbar(fr, orient="horizontal", command=self.brain_cv.xview)
        self.brain_cv.configure(xscrollcommand=hs.set)
        self.brain_cv.pack(fill="both", expand=True)
        hs.pack(fill="x")
        self.brain_info = ScrolledText(t, height=9, font=("Courier", 9))
        self.brain_info.pack(fill="x", padx=8, pady=4)
        self.brain_struct = None
        self.sel_layers: set = set()
        self.sel_comp = None
        self._cells = {}

    def _best_ref(self):
        b = self.orch.training.best()
        return b["path"] if b else self.orch.cfg.base_model

    def _brain_load(self):
        if not self._need_run():
            return

        def job():
            tm = self.orch.training
            path = tm.local_path(self._best_ref())
            self.brain_struct = brainmap.read_structure(path)
            self.brain_path = path
        self._bg(job, "load brain map")

    def _brain_scan(self):
        if not self._need_run():
            return

        def job():
            tm = self.orch.training
            accepted = [g for g in tm.db.generations() if g["accepted"]]
            prev = accepted[-2]["path"] if len(accepted) > 1 and accepted[-2]["path"] else None
            try:
                prev = prev if prev and Path(prev).is_dir() else None
            except OSError:
                prev = None
            self.brain_struct = tm.scan_model(self._best_ref(), compare_to=prev, tag="gui")
        self._bg(job, "scan model statistics")

    def _layers_mode(self, mode):
        if not self.brain_struct:
            return messagebox.showinfo("Brain map", "Load the map first.")
        n = len(self.brain_struct["layers"])
        cfg = self.orch.cfg
        if mode == "only":
            cfg.train_layers = sorted(self.sel_layers)
        elif mode == "freeze":
            cfg.train_layers = [i for i in range(n) if i not in self.sel_layers]
        else:
            cfg.train_layers = []
        self.orch.bus.emit("human", "human", f"trainable layers set to {cfg.train_layers or 'ALL'}", level="warn")
        self._brain_draw()

    def _brain_draw(self):
        cv, st = self.brain_cv, self.brain_struct
        cv.delete("all")
        self._cells = {}
        if not st:
            cv.create_text(20, 20, anchor="nw", text="No map loaded yet (press 'Load map').", fill="#777")
            return
        metric = self.metric.get()
        vals = []
        for e in list(st["special"].values()) + [c for L in st["layers"] for c in L["components"].values()]:
            v = (e.get("stats") or {}).get(metric)
            if v is not None:
                vals.append(v)
        lo, hi = (min(vals), max(vals)) if vals else (0, 1)
        trainable = set(self.orch.cfg.train_layers)
        x = 10

        def cell(x, y, w, e, label, tag):
            v = (e.get("stats") or {}).get(metric)
            rid = cv.create_rectangle(x, y, x + w, y + CELL_H, fill=_color(v, lo, hi), outline="#999")
            cv.create_text(x + w / 2, y + CELL_H / 2, text=label, font=("TkDefaultFont", 7))
            self._cells[rid] = (tag, e)

        y0 = 34
        for key in ("embed",):
            if key in st["special"]:
                cv.create_text(x + 28, 14, text="embed", font=("TkDefaultFont", 8, "bold"))
                cell(x, y0, 56, st["special"][key], "tokens", ("special", key))
        x += 70
        comps = brainmap.COMPONENT_ORDER
        for L in st["layers"]:
            i = L["index"]
            sel = i in self.sel_layers
            hid = cv.create_rectangle(x, 4, x + CELL_W, 26, fill="#ffd9a8" if sel else ("#e8f5e8" if (not trainable or i in trainable) else "#f5e0e0"),
                                      outline="#e08000" if sel else "#999", width=2 if sel else 1)
            cv.create_text(x + CELL_W / 2, 15, text=f"L{i}", font=("TkDefaultFont", 8, "bold"))
            self._cells[hid] = (("layer", i), {})
            for r, c in enumerate(comps):
                if c in L["components"]:
                    cell(x, y0 + r * (CELL_H + GAP), CELL_W, L["components"][c], c.split(".")[-1][:5], ("comp", i, c))
            x += CELL_W + GAP + 4
        x += 10
        for key, lab in (("final_norm", "norm"), ("lm_head", "head")):
            if key in st["special"]:
                cv.create_text(x + 28, 14, text=key, font=("TkDefaultFont", 8, "bold"))
                cell(x, y0, 56, st["special"][key], lab, ("special", key))
                x += 70
        ylab = y0 + len(comps) * (CELL_H + GAP) + 8
        cv.create_text(10, ylab, anchor="nw", fill="#555", font=("TkDefaultFont", 8),
                       text=f"color = {metric}  (low {lo:.3g} -> high {hi:.3g})   rows: " + ", ".join(comps) +
                            ("   | green header = trainable, red = frozen" if trainable else ""))
        cv.configure(scrollregion=(0, 0, x + 40, ylab + 30))
        cv.tag_bind("all", "<Button-1>", self._brain_click)
        self.brain_state.config(text=f"model: {st['model']}  |  {st['total_params'] / 1e6:.2f}M parameters  |  layers: {len(st['layers'])}  |  "
                                     f"selected layers: {sorted(self.sel_layers) or '-'}  |  trainable: {sorted(trainable) or 'ALL'}")

    def _brain_click(self, ev):
        cv = self.brain_cv
        x, y = cv.canvasx(ev.x), cv.canvasy(ev.y)
        hit = [i for i in cv.find_overlapping(x, y, x, y) if i in self._cells]
        if not hit:
            return
        tag, e = self._cells[hit[-1]]
        if tag[0] == "layer":
            self.sel_layers ^= {tag[1]}
        else:
            self.sel_comp = tag
            self.brain_info.delete("1.0", "end")
            self.brain_info.insert("end", json.dumps({"where": list(tag), "tensor": e.get("name"), "shape": e.get("shape"),
                                                      "parameters": e.get("numel"), "stats": e.get("stats", "run 'Scan statistics' for numbers")}, indent=2))
        self._brain_draw()

    def _dialog(self, title, fields):
        """Tiny modal form -> dict or None. fields: [(key, label, default, choices|None)]"""
        top = tk.Toplevel(self)
        top.title(title)
        top.transient(self)
        vars_ = {}
        for r, (k, lab, dflt, choices) in enumerate(fields):
            ttk.Label(top, text=lab).grid(row=r, column=0, sticky="w", padx=8, pady=4)
            v = tk.StringVar(value=str(dflt))
            (ttk.Combobox(top, textvariable=v, values=choices, state="readonly", width=24) if choices else ttk.Entry(top, textvariable=v, width=44)).grid(row=r, column=1, padx=8)
            vars_[k] = v
        out = {}

        def ok():
            out.update({k: v.get() for k, v in vars_.items()})
            top.destroy()
        ttk.Button(top, text="Run", command=ok).grid(row=len(fields), column=0, columnspan=2, pady=8)
        self.wait_window(top)
        return out or None

    def _transplant_dialog(self):
        if not self._need_run():
            return
        if not self.brain_struct:
            return messagebox.showinfo("Transplant", "Load the brain map first and select the target layer(s).")
        d = self._dialog("Transplant from a donor model", [
            ("donor", "Donor model (HF id or local folder)", "", None),
            ("donor_start", "Donor layer for the FIRST selected target layer", 0, None),
            ("components", "Components", "all", ["all", "selected cell only", "attention only", "mlp only"]),
            ("mode", "Mode", "blend", ["blend", "copy", "resize"]),
            ("alpha", "Blend strength alpha (0..1)", 0.5, None),
            ("special", "Also transfer", "nothing", ["nothing", "embed", "lm_head", "final_norm"])])
        if not d or not d["donor"].strip():
            return
        sel = sorted(self.sel_layers)
        if not sel and d["special"] == "nothing":
            return messagebox.showinfo("Transplant", "Select at least one target layer (click its header).")
        comps = {"all": "all", "attention only": ["attn.q", "attn.k", "attn.v", "attn.o"], "mlp only": ["mlp.gate", "mlp.up", "mlp.down"],
                 "selected cell only": [self.sel_comp[2]] if self.sel_comp and self.sel_comp[0] == "comp" else "all"}[d["components"]]
        selection = [{"target_layer": i, "donor_layer": int(d["donor_start"]) + (i - sel[0]), "components": comps, "mode": d["mode"],
                      "alpha": float(d["alpha"])} for i in sel]
        if d["special"] != "nothing":
            selection.append({"special": d["special"], "mode": d["mode"], "alpha": float(d["alpha"])})
        self._bg(lambda: self.orch.training.transplant(d["donor"].strip(), selection, "transplant (human)"), "transplant")

    def _merge_dialog(self):
        if not self._need_run():
            return
        d = self._dialog("Merge with another model (same architecture)", [
            ("other", "Other model (HF id or local folder)", "", None),
            ("mode", "Method", "slerp", ["linear", "slerp", "task", "ties"]), ("weight", "Weight of the other model (0..1)", 0.5, None)])
        if d and d["other"].strip():
            self._bg(lambda: self.orch.training.merge_with(d["other"].strip(), d["mode"], float(d["weight"])), "merge")

    def _arch_dialog(self):
        if not self._need_run():
            return
        d = self._dialog("Convert the current best model to another attention architecture", [
            ("preset", "Preset", "CLA-2 (2 layers share K/V)", ["CLA-2 (2 layers share K/V)", "CLA-3 (3 layers share K/V)", "GQA (halve KV heads)", "MQA (1 KV head)", "custom"]),
            ("kv_heads", "Custom: KV heads (0 = unchanged)", 0, None), ("cla_factor", "Custom: layers sharing K/V (1 = none)", 1, None),
            ("init", "K/V initialisation for shared groups", "mean", ["mean", "first"]), ("heal", "Heal with a short training run", "yes", ["yes", "no"])])
        if not d:
            return
        try:
            st = brainmap.read_structure(self.orch.training.local_path(self._best_ref()))
            heads, kv_now = st["config"].get("num_attention_heads") or 1, st["config"].get("num_key_value_heads") or st["config"].get("num_attention_heads") or 1
        except Exception:
            kv_now = 1
        kv, f = {"CLA-2 (2 layers share K/V)": (0, 2), "CLA-3 (3 layers share K/V)": (0, 3), "GQA (halve KV heads)": (max(1, kv_now // 2), 1),
                 "MQA (1 KV head)": (1, 1)}.get(d["preset"], (int(d["kv_heads"]), int(d["cla_factor"])))
        self._bg(lambda: self.orch.training.convert_architecture(kv, f, d["init"], d["heal"] == "yes"), "architecture conversion")

    # =================================================================== WORLDS & TOOLS
    def _tab_worlds(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Worlds & Tools")
        pw = ttk.PanedWindow(t, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=8, pady=8)
        left = ttk.LabelFrame(pw, text="Worlds (black box: built-in + created by the agents)")
        right = ttk.LabelFrame(pw, text="Tools created by the agents")
        pw.add(left, weight=1)
        pw.add(right, weight=1)
        self.w_tree = ttk.Treeview(left, columns=("name", "domain", "kind", "status"), show="headings", height=10)
        for c, wd in zip(("name", "domain", "kind", "status"), (140, 110, 80, 70)):
            self.w_tree.heading(c, text=c)
            self.w_tree.column(c, width=wd)
        self.w_tree.pack(fill="both", expand=True)
        self.w_tree.bind("<<TreeviewSelect>>", self._world_show)
        b = ttk.Frame(left)
        b.pack(fill="x")
        for txt, st in (("Approve/enable", "active"), ("Disable", "disabled")):
            ttk.Button(b, text=txt, command=lambda s=st: self._set_status("world", s)).pack(side="left", padx=2, pady=3)
        ttk.Button(b, text="Check laws (invariant drift)", command=self._world_check).pack(side="left", padx=6)
        self.w_text = ScrolledText(left, height=10, font=("Courier", 9))
        self.w_text.pack(fill="both", expand=True)
        self.t_tree = ttk.Treeview(right, columns=("name", "status", "uses", "by"), show="headings", height=10)
        for c, wd in zip(("name", "status", "uses", "by"), (140, 80, 50, 120)):
            self.t_tree.heading(c, text=c)
            self.t_tree.column(c, width=wd)
        self.t_tree.pack(fill="both", expand=True)
        self.t_tree.bind("<<TreeviewSelect>>", self._tool_show)
        b = ttk.Frame(right)
        b.pack(fill="x")
        for txt, st in (("Approve/enable", "active"), ("Disable", "disabled")):
            ttk.Button(b, text=txt, command=lambda s=st: self._set_status("tool", s)).pack(side="left", padx=2, pady=3)
        self.t_text = ScrolledText(right, height=10, font=("Courier", 9))
        self.t_text.pack(fill="both", expand=True)

    def _set_status(self, kind, status):
        tree = self.w_tree if kind == "world" else self.t_tree
        sel = tree.selection()
        if not sel or not self.orch:
            return
        name = tree.item(sel[0])["values"][0]
        if kind == "world":
            if self.orch.db.world(name):
                self.orch.db.set_world_status(name, status)
        else:
            self.orch.db.set_tool_status(name, status)
        self.orch.bus.emit("human", "human", f"{kind} '{name}' -> {status}", level="warn")

    def _world_show(self, _=None):
        sel = self.w_tree.selection()
        if not sel or not self.orch:
            return
        name = self.w_tree.item(sel[0])["values"][0]
        w = self.orch.db.world(name)
        self.w_text.delete("1.0", "end")
        if w:
            self.w_text.insert("end", f"# {w['name']} [{w['domain']}] by {w['created_by']}\n# {w['description']}\n\n{w['source']}")
        else:
            from . import worldlab_lib
            info = worldlab_lib.list_builtin_worlds().get(name, {})
            self.w_text.insert("end", f"# built-in world '{name}'\n# {info.get('description', '')}\n\nUse in any experiment:\n  import worldlab\n  w = worldlab.load_world('{name}')\n  w.check(1000, 0.01)   # invariant drift = do the laws hold?")

    def _world_check(self):
        sel = self.w_tree.selection()
        if not sel or not self._need_run():
            return
        name = self.w_tree.item(sel[0])["values"][0]

        def job():
            code = (f"import worldlab, json\nw = worldlab.load_world({name!r})\n"
                    "print(json.dumps({k: v for k, v in w.check(1000, 0.01).items()}))\n")
            r = self.orch.ctx.sandbox.run(code, files=self.orch.ctx.workshop.module_files(), station="ro", timeout=60)
            self.orch.bus.emit("gui", "human", f"world '{name}' invariant drift: {r.stdout.strip()[-300:] or r.stderr[-300:]}", level="warn")
        self._bg(job, f"check world {name}")

    def _tool_show(self, _=None):
        sel = self.t_tree.selection()
        if sel and self.orch:
            t = self.orch.db.tool(self.t_tree.item(sel[0])["values"][0])
            if t:
                self.t_text.delete("1.0", "end")
                self.t_text.insert("end", f"# {t['name']}: {t['description']}\n# example args: {t['example_args']}\n\n{t['code']}")

    # =================================================================== COMPUTER
    def _tab_computer(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Computer")
        top = ttk.Frame(t)
        top.pack(fill="x", padx=8, pady=4)
        ttk.Label(top, text="The agents' persistent computer (/station). You can use it too:").pack(side="left")
        self.sh_mode = tk.StringVar(value="shell")
        ttk.Combobox(top, textvariable=self.sh_mode, values=("shell", "python"), state="readonly", width=8).pack(side="left", padx=6)
        self.sh_entry = ttk.Entry(top, width=70)
        self.sh_entry.pack(side="left")
        self.sh_entry.bind("<Return>", lambda e: self._sh_run())
        ttk.Button(top, text="Run", command=self._sh_run).pack(side="left", padx=4)
        pw = ttk.PanedWindow(t, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=8, pady=4)
        self.f_tree = ttk.Treeview(pw, columns=("bytes",), show="tree headings")
        self.f_tree.heading("#0", text="/station")
        self.f_tree.heading("bytes", text="bytes")
        self.f_tree.column("#0", width=260)
        self.f_tree.column("bytes", width=70)
        self.f_tree.bind("<<TreeviewSelect>>", self._file_show)
        pw.add(self.f_tree, weight=1)
        self.sh_out = ScrolledText(pw, font=("Courier", 9))
        pw.add(self.sh_out, weight=2)

    def _sh_run(self):
        if not self._need_run():
            return
        cmd, mode = self.sh_entry.get().strip(), self.sh_mode.get()
        if not cmd:
            return

        def job():
            w = self.orch.ctx.workshop
            r = w.station_run("human", command=cmd if mode == "shell" else None, python=cmd if mode == "python" else None)
            self.ui(lambda: self._sh_print(f"$ {cmd}\n{r.get('stdout', '')}{r.get('stderr', '')}[exit {r.get('exit_code')}]\n\n"))
        self._bg(job, "computer command")

    def _sh_print(self, text):
        self.sh_out.insert("end", text)
        self.sh_out.see("end")

    def _file_show(self, _=None):
        sel = self.f_tree.selection()
        if sel and self.orch and self.orch.ctx:
            r = self.orch.ctx.workshop.station_files("human", "read", sel[0])
            self.sh_out.delete("1.0", "end")
            self.sh_out.insert("end", r.get("content", r.get("error", "")))

    def _refresh_files(self):
        if not (self.orch and self.orch.ctx):
            return
        r = self.orch.ctx.workshop.station_files("human", "list", "")
        keep = set(self.f_tree.selection())
        self.f_tree.delete(*self.f_tree.get_children())
        for it in r.get("items", []):
            parent = str(Path(it["path"]).parent)
            parent = "" if parent == "." else parent
            if parent and not self.f_tree.exists(parent):
                parent = ""
            self.f_tree.insert(parent, "end", iid=it["path"], text=Path(it["path"]).name, values=(it["bytes"] if it["bytes"] is not None else "",), open=True)
        present = [i for i in keep if self.f_tree.exists(i)]
        if present:
            self.f_tree.selection_set(present)

    # =================================================================== TESTS
    def _tab_tests(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Tests")
        pw = ttk.PanedWindow(t, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=8, pady=8)
        self.ev_tree = ttk.Treeview(pw, columns=("id", "label", "score", "n"), show="headings")
        for c, wd in zip(("id", "label", "score", "n"), (40, 250, 60, 40)):
            self.ev_tree.heading(c, text=c)
            self.ev_tree.column(c, width=wd)
        self.ev_tree.bind("<<TreeviewSelect>>", self._eval_select)
        pw.add(self.ev_tree, weight=1)
        right = ttk.Frame(pw)
        pw.add(right, weight=2)
        bar = ttk.Frame(right)
        bar.pack(fill="x")
        self.ev_filter = tk.StringVar(value="all")
        ttk.Combobox(bar, textvariable=self.ev_filter, values=("all", "failed", "passed", "math", "logic", "physics", "chemistry", "code"),
                     state="readonly", width=10).pack(side="left")
        self.ev_filter.trace_add("write", lambda *_: self._eval_select())
        self.ev_items = ttk.Treeview(right, columns=("pid", "domain", "ok", "reason"), show="headings", height=9)
        for c, wd in zip(("pid", "domain", "ok", "reason"), (90, 80, 40, 300)):
            self.ev_items.heading(c, text=c)
            self.ev_items.column(c, width=wd)
        self.ev_items.pack(fill="x")
        self.ev_items.bind("<<TreeviewSelect>>", self._eval_item)
        self.ev_text = ScrolledText(right, font=("Courier", 9))
        self.ev_text.pack(fill="both", expand=True)
        self._ev_rows = {}

    def _eval_select(self, _=None):
        sel = self.ev_tree.selection()
        if not sel or not self.orch:
            return
        items = self.orch.db.eval_items(int(sel[0]))
        f = self.ev_filter.get()
        self._ev_rows = {i["pid"]: i for i in items}
        self.ev_items.delete(*self.ev_items.get_children())
        for i in items:
            if f == "failed" and i["passed"] or f == "passed" and not i["passed"] or f in ("math", "logic", "physics", "chemistry", "code") and i["domain"] != f:
                continue
            self.ev_items.insert("", "end", iid=i["pid"], values=(i["pid"], i["domain"], "yes" if i["passed"] else "NO", i["reason"]))

    def _eval_item(self, _=None):
        sel = self.ev_items.selection()
        if sel and sel[0] in self._ev_rows:
            i = self._ev_rows[sel[0]]
            self.ev_text.delete("1.0", "end")
            self.ev_text.insert("end", f"PROMPT:\n{i['prompt']}\n\nMODEL ANSWER:\n{i['answer']}\n\nRESULT: {'PASS' if i['passed'] else 'FAIL'} - {i['reason']}")

    # =================================================================== CONCEPTS / REPORTS / MISTAKES
    def _tab_concepts(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Concepts & Reports")
        pw = ttk.PanedWindow(t, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=8, pady=8)
        left = ttk.Frame(pw)
        pw.add(left, weight=1)
        ttk.Label(left, text="Scientific concepts (built from VERIFIED claims only)").pack(anchor="w")
        self.c_tree = ttk.Treeview(left, columns=("id", "title"), show="headings", height=8)
        self.c_tree.heading("id", text="id")
        self.c_tree.heading("title", text="title")
        self.c_tree.column("id", width=40)
        self.c_tree.column("title", width=300)
        self.c_tree.pack(fill="x")
        self.c_tree.bind("<<TreeviewSelect>>", self._concept_show)
        ttk.Label(left, text="Research reports").pack(anchor="w")
        self.r_tree = ttk.Treeview(left, columns=("f",), show="headings", height=6)
        self.r_tree.heading("f", text="report")
        self.r_tree.pack(fill="x")
        self.r_tree.bind("<<TreeviewSelect>>", self._report_show)
        ttk.Label(left, text="Mistakes learned from").pack(anchor="w")
        self.m_text = ScrolledText(left, height=8, font=("Courier", 8))
        self.m_text.pack(fill="both", expand=True)
        self.c_text = ScrolledText(pw, font=("TkDefaultFont", 10), wrap="word")
        pw.add(self.c_text, weight=2)

    def _concept_show(self, _=None):
        sel = self.c_tree.selection()
        if sel and self.orch:
            for c in self.orch.db.concepts():
                if str(c["id"]) == sel[0]:
                    self.c_text.delete("1.0", "end")
                    self.c_text.insert("end", f"{c['title']}\n\n{c['summary']}\n\nbuilt from verified claims: {c['claim_ids']}")

    def _report_show(self, _=None):
        sel = self.r_tree.selection()
        if sel and self.orch:
            p = self.orch.run_dir / "reports" / sel[0]
            if p.exists():
                self.c_text.delete("1.0", "end")
                self.c_text.insert("end", p.read_text(encoding="utf-8"))

    # =================================================================== BACKUPS
    def _tab_backups(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Backups")
        bar = ttk.Frame(t)
        bar.pack(fill="x", padx=8, pady=6)
        for txt, cmd in (("Backup now", self._bk_now), ("Verify selected", self._bk_verify), ("Restore selected into a NEW run...", self._bk_restore),
                         ("Roll back to selected generation", self._bk_rollback)):
            ttk.Button(bar, text=txt, command=cmd).pack(side="left", padx=3)
        self.bk_tree = ttk.Treeview(t, columns=("name", "label", "gen", "score", "weights", "files"), show="headings", height=8)
        for c, wd in zip(("name", "label", "gen", "score", "weights", "files"), (200, 90, 60, 70, 70, 60)):
            self.bk_tree.heading(c, text=c)
            self.bk_tree.column(c, width=wd)
        self.bk_tree.pack(fill="x", padx=8)
        ttk.Label(t, text="Generations (select one for rollback; only generations whose weights still exist can be rolled back to)").pack(anchor="w", padx=8, pady=(8, 0))
        self.bg_tree = ttk.Treeview(t, columns=("id", "method", "score", "accepted", "weights"), show="headings", height=8)
        for c, wd in zip(("id", "method", "score", "accepted", "weights"), (50, 120, 70, 70, 300)):
            self.bg_tree.heading(c, text=c)
            self.bg_tree.column(c, width=wd)
        self.bg_tree.pack(fill="x", padx=8)
        self.bk_log = ScrolledText(t, height=6, font=("Courier", 9))
        self.bk_log.pack(fill="both", expand=True, padx=8, pady=6)

    def _bk_say(self, msg):
        self.bk_log.insert("end", msg + "\n")
        self.bk_log.see("end")

    def _bk_now(self):
        if self._need_run():
            self._bg(lambda: self.orch.backup.create("manual"), "manual backup")

    def _bk_sel(self):
        sel = self.bk_tree.selection()
        return str(self.orch.backup.root / sel[0]) if sel and self.orch else None

    def _bk_verify(self):
        p = self._bk_sel()
        if p:
            ok, bad = BackupManager.verify(p)
            self._bk_say(f"verify {Path(p).name}: {'OK - all hashes match' if ok else 'CORRUPTED: ' + str(bad[:5])}")

    def _bk_restore(self):
        p = self._bk_sel()
        if not p:
            return
        folder = filedialog.askdirectory(title="Choose an EMPTY folder for the restored run")
        if folder:
            try:
                new = BackupManager.restore_to_new_run(p, Path(folder) / "restored_run")
                self._bk_say(f"restored into {new}. Start it with:  python -m sira --resume \"{new}\"")
            except Exception as e:
                messagebox.showerror("Restore failed", str(e))

    def _bk_rollback(self):
        sel = self.bg_tree.selection()
        if not sel or not self._need_run():
            return
        if messagebox.askyesno("Roll back", f"Make generation {sel[0]} the best model again? Later generations are marked not accepted."):
            try:
                self.orch.backup.rollback_to(int(sel[0]))
            except ValueError as e:
                messagebox.showerror("Rollback", str(e))

    # =================================================================== NETWORK & POLICY
    def _tab_policy(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Network & Policy")
        self.pol = {}
        box = ttk.LabelFrame(t, text="Internet for the agents' web tools (applies immediately)")
        box.pack(fill="x", padx=8, pady=8)
        self.pol["web_mode"] = tk.StringVar(value="open")
        for i, (v, lab) in enumerate((("open", "OPEN - the whole public internet"), ("allowlist", "ALLOWLIST - only these domains"),
                                      ("blocklist", "BLOCKLIST - everything except these domains"))):
            ttk.Radiobutton(box, text=lab, variable=self.pol["web_mode"], value=v).grid(row=i, column=0, sticky="w", padx=8)
        ttk.Label(box, text="Allowlist (one domain per line)").grid(row=0, column=1, padx=8)
        ttk.Label(box, text="Blocklist (one domain per line)").grid(row=0, column=2, padx=8)
        self.allow_txt = tk.Text(box, height=6, width=34)
        self.block_txt = tk.Text(box, height=6, width=34)
        self.allow_txt.grid(row=1, column=1, rowspan=3, padx=8, pady=4)
        self.block_txt.grid(row=1, column=2, rowspan=3, padx=8, pady=4)
        box2 = ttk.LabelFrame(t, text="Sandbox and self-extension (applies immediately)")
        box2.pack(fill="x", padx=8, pady=4)
        self.pol["sandbox_network"] = tk.StringVar(value="none")
        ttk.Label(box2, text="Network inside the sandbox:").grid(row=0, column=0, sticky="w", padx=8, pady=3)
        ttk.Combobox(box2, textvariable=self.pol["sandbox_network"], values=("none", "open"), state="readonly", width=8).grid(row=0, column=1, sticky="w")
        for r, (k, lab) in enumerate((("allow_agent_installs", "Agents may pip-install libraries (needs sandbox network = open)"),
                                      ("approve_new_tools", "I must approve every tool/world the agents create"),
                                      ("require_approval", "I must approve every training round"),
                                      ("try_merge_candidates", "Also test weight-interpolation (soup) candidates"),
                                      ("backup_weights", "Include model weights in backups")), start=1):
            self.pol[k] = tk.BooleanVar(value=getattr(self.cfg, k))
            ttk.Checkbutton(box2, text=lab, variable=self.pol[k]).grid(row=r, column=0, columnspan=3, sticky="w", padx=8)
        ttk.Button(t, text="Apply", command=self._pol_apply).pack(anchor="w", padx=12, pady=8)
        ttk.Label(t, text="Always blocked regardless of mode: your local network / loopback / cloud-metadata addresses (so a web page can never reach your machine).\n"
                          "The immutable core (evaluation, verification, safety, control, sandbox code) can never be edited by the agents.",
                  foreground="#555").pack(anchor="w", padx=12)

    def _pol_apply(self):
        if not self.orch:
            return messagebox.showinfo("Policy", "Start a run first.")
        c = self.orch.cfg
        c.web_mode = self.pol["web_mode"].get()
        c.web_allow = [x.strip() for x in self.allow_txt.get("1.0", "end").splitlines() if x.strip()]
        c.web_block = [x.strip() for x in self.block_txt.get("1.0", "end").splitlines() if x.strip()]
        c.sandbox_network = self.pol["sandbox_network"].get()
        for k in ("allow_agent_installs", "approve_new_tools", "require_approval", "try_merge_candidates", "backup_weights"):
            setattr(c, k, bool(self.pol[k].get()))
        self.orch.bus.emit("human", "human", f"policy: web={c.web_mode} allow={c.web_allow} block={c.web_block} sandbox_net={c.sandbox_network} "
                                             f"installs={c.allow_agent_installs} approve_tools={c.approve_new_tools}", level="warn")

    # =================================================================== STATS
    def _tab_stats(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Statistics")
        self.st_text = tk.Label(t, justify="left", anchor="nw", font=("Courier", 10))
        self.st_text.pack(fill="x", padx=10, pady=8)
        self.st_cv = tk.Canvas(t, bg="white", height=300)
        self.st_cv.pack(fill="both", expand=True, padx=10, pady=6)

    def _refresh_stats(self):
        o = self.orch
        db = o.db
        exps = db.experiments(100000)
        ok = sum(1 for e in exps if e["exit_code"] == 0 and not e["timed_out"])
        claims = {s: len(db.claims(s, 100000)) for s in ("verified", "refuted", "inconclusive", "lead", "vetoed")}
        gens = db.generations()
        acc = [g for g in gens if g["accepted"]]
        b = o.control.budget_status()
        up = o.control.wall_s
        self.st_text.config(text=(
            f"run time {up / 3600:.2f} h   cycles {o.cycle}   tokens in/out {b['tokens'][0]:,}/{b['tokens'][1]:,}   LLM spend ${b['usd'][0]:.2f}   GPU {b['gpu_h'][0]:.2f} h\n"
            f"experiments {len(exps)} (clean {ok}, failed {len(exps) - ok})   claims: " + "  ".join(f"{k} {v}" for k, v in claims.items()) +
            f"\ngenerations {len(gens)} (accepted {len(acc)})   best score {max([g['score'] for g in acc] or [0]):.3f}   "
            f"tools {len(db.tools('active'))}   worlds {len(db.worlds('active'))}   concepts {len(db.concepts())}   mistakes learned {len(db.mistakes(100000))}"))
        cv = self.st_cv
        cv.delete("all")
        w, h = max(cv.winfo_width(), 400), max(cv.winfo_height(), 200)
        l, r, tp, bt = 40, 160, 15, 30
        for i in range(0, 11, 2):
            y = tp + (h - tp - bt) * (1 - i / 10)
            cv.create_line(l, y, w - r, y, fill="#eee")
            cv.create_text(l - 6, y, text=f"{i / 10:.1f}", anchor="e", fill="#777")
        pts = [g for g in gens if g["method"] != "failed" and g["detail"].get("by_domain")]
        palette = {"math": "#c0392b", "logic": "#2980b9", "physics": "#27ae60", "chemistry": "#8e44ad", "code": "#d35400"}
        if len(pts) >= 1:
            for k, col in palette.items():
                xy = []
                for i, g in enumerate(pts):
                    v = g["detail"]["by_domain"].get(k)
                    if v is None:
                        continue
                    x = l + (w - l - r) * (i / max(1, len(pts) - 1)) if len(pts) > 1 else (l + w - r) / 2
                    xy.append((x, tp + (h - tp - bt) * (1 - v["rate"])))
                if len(xy) > 1:
                    cv.create_line(*[c for p in xy for c in p], fill=col, width=2)
                for x, y in xy:
                    cv.create_oval(x - 3, y - 3, x + 3, y + 3, fill=col, outline=col)
        for i, (k, col) in enumerate(palette.items()):
            cv.create_rectangle(w - r + 20, 20 + i * 20, w - r + 34, 32 + i * 20, fill=col, outline=col)
            cv.create_text(w - r + 40, 26 + i * 20, text=k, anchor="w")
        cv.create_text(l, h - 10, anchor="w", fill="#555", text="per-domain pass rate on fresh hidden problems, by generation")

    # =================================================================== QUEST
    def _tab_quest(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Quest")
        self.q_head = ttk.Label(t, text="No quest running. Set MODE = quest/auto and a goal in Setup.", font=("TkDefaultFont", 11, "bold"))
        self.q_head.pack(anchor="w", padx=8, pady=6)
        pw = ttk.PanedWindow(t, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=8, pady=4)
        left = ttk.Frame(pw)
        pw.add(left, weight=2)
        ttk.Label(left, text="Rounds: exam score, blank student (no notes), learning delta, criteria coverage, life score, challenge skill").pack(anchor="w")
        cols = ("round", "exam", "blank", "delta", "coverage", "life", "skill", "brier")
        self.qr_tree = ttk.Treeview(left, columns=cols, show="headings", height=7)
        for c in cols:
            self.qr_tree.heading(c, text=c)
            self.qr_tree.column(c, width=66)
        self.qr_tree.pack(fill="x")
        self.qr_tree.bind("<<TreeviewSelect>>", self._quest_round)
        ttk.Label(left, text="Exam answers of the selected round (ground truth is mechanical; wrong answers get an autopsy)").pack(anchor="w", pady=(6, 0))
        self.qe_tree = ttk.Treeview(left, columns=("kind", "variant", "score", "reason"), show="headings", height=9)
        for c, wd in zip(("kind", "variant", "score", "reason"), (70, 60, 50, 380)):
            self.qe_tree.heading(c, text=c)
            self.qe_tree.column(c, width=wd)
        self.qe_tree.pack(fill="both", expand=True)
        self.qe_tree.bind("<<TreeviewSelect>>", self._quest_exam_item)
        self.q_text = ScrolledText(pw, wrap="word", font=("TkDefaultFont", 10))
        pw.add(self.q_text, weight=3)
        self._qe_rows = {}

    def _quest_round(self, _=None):
        sel = self.qr_tree.selection()
        if not sel or not self.orch:
            return
        rows = self.orch.db.exams(self.orch.db.get_meta("quest_id"), int(sel[0]))
        self._qe_rows = {str(r["id"]): r for r in rows}
        self._refill(self.qe_tree, [(str(r["id"]), (r["kind"], r["variant"], f"{r['score']:.2f}", r["reason"][:80])) for r in rows])
        rp = self.orch.run_dir / "reports" / f"position_{int(sel[0]):03d}.md"
        self.q_text.delete("1.0", "end")
        if rp.exists():
            self.q_text.insert("end", rp.read_text(encoding="utf-8"))

    def _quest_exam_item(self, _=None):
        sel = self.qe_tree.selection()
        if sel and sel[0] in self._qe_rows:
            r = self._qe_rows[sel[0]]
            self.q_text.delete("1.0", "end")
            self.q_text.insert("end", f"QUESTION ({r['kind']}, {r['variant']} student)\n{r['question']}\n\nGROUND TRUTH\n{r['truth']}\n\nANSWER (confidence {r['confidence']})\n{r['answer']}\n\nSCORE {r['score']}  -  {r['reason']}")

    # =================================================================== CHALLENGES & REWARDS
    def _tab_challenges(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Challenges & Rewards")
        self.ch_head = ttk.Label(t, text="", font=("TkDefaultFont", 10, "bold"))
        self.ch_head.pack(anchor="w", padx=8, pady=6)
        pw = ttk.PanedWindow(t, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=8, pady=4)
        left = ttk.Frame(pw)
        pw.add(left, weight=2)
        cols = ("id", "title", "domain", "kind", "lvl", "status", "best", "tries")
        self.ch_tree = ttk.Treeview(left, columns=cols, show="headings", height=9)
        for c, wd in zip(cols, (35, 230, 90, 80, 30, 60, 50, 40)):
            self.ch_tree.heading(c, text=c)
            self.ch_tree.column(c, width=wd)
        self.ch_tree.pack(fill="x")
        self.ch_tree.bind("<<TreeviewSelect>>", self._challenge_show)
        ttk.Label(left, text="Reward / penalty ledger (closer = more reward; gaming, bluffing, repeating = penalty)").pack(anchor="w", pady=(6, 0))
        self.rw_tree = ttk.Treeview(left, columns=("src", "item", "reward", "penalty", "note"), show="headings", height=9)
        for c, wd in zip(("src", "item", "reward", "penalty", "note"), (60, 200, 55, 55, 250)):
            self.rw_tree.heading(c, text=c)
            self.rw_tree.column(c, width=wd)
        self.rw_tree.pack(fill="both", expand=True)
        right = ttk.Frame(pw)
        pw.add(right, weight=3)
        self.at_tree = ttk.Treeview(right, columns=("n", "score", "reward", "penalty", "approach", "note"), show="headings", height=7)
        for c, wd in zip(("n", "score", "reward", "penalty", "approach", "note"), (30, 50, 55, 55, 200, 250)):
            self.at_tree.heading(c, text=c)
            self.at_tree.column(c, width=wd)
        self.at_tree.pack(fill="x")
        self.at_tree.bind("<<TreeviewSelect>>", self._attempt_show)
        self.ch_text = ScrolledText(right, wrap="word", font=("Courier", 9))
        self.ch_text.pack(fill="both", expand=True)
        self._at_rows = {}

    def _challenge_show(self, _=None):
        sel = self.ch_tree.selection()
        if not sel or not self.orch:
            return
        ch = self.orch.db.challenge(int(sel[0]))
        at = self.orch.db.attempts(int(sel[0]))
        self._at_rows = {str(a["id"]): a for a in at}
        self._refill(self.at_tree, [(str(a["id"]), (a["n"], f"{a['score']:.3f}", f"{a['reward']:.1f}", f"{a['penalty']:.1f}", a["approach"][:60], a["note"][:80])) for a in at])
        self.ch_text.delete("1.0", "end")
        self.ch_text.insert("end", f"# {ch['title']}  [{ch['domain']}, {ch['kind']}, level {ch['difficulty']}]  status={ch['status']}  best={ch['best_score']:.3f}\n\n{ch['statement']}\n\n"
                                   f"--- hidden verifier (the solver never sees this) ---\n{ch['verifier_code']}")

    def _attempt_show(self, _=None):
        sel = self.at_tree.selection()
        if sel and sel[0] in self._at_rows:
            a = self._at_rows[sel[0]]
            self.ch_text.delete("1.0", "end")
            self.ch_text.insert("end", f"attempt {a['n']}: score {a['score']:.3f} holdout {a['holdout']}\napproach: {a['approach']}\nfeedback: {a['note']}\nanswer: {a['answer'][:400]}\n\n{a['code']}")

    # =================================================================== SOCIETY
    def _tab_society(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Society")
        bar = ttk.Frame(t)
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Label(bar, text="The agent lives among simulated humans; a deterministic engine knows the truth (beliefs, needs, what really happened).").pack(side="left")
        ttk.Button(bar, text="Run a society round now", command=self._society_now).pack(side="left", padx=10)
        pw = ttk.PanedWindow(t, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=8, pady=4)
        cols = ("id", "round", "life", "false_belief", "hidden_need", "promise", "honesty", "deception")
        self.sc_tree = ttk.Treeview(pw, columns=cols, show="headings")
        for c in cols:
            self.sc_tree.heading(c, text=c)
            self.sc_tree.column(c, width=70)
        self.sc_tree.bind("<<TreeviewSelect>>", self._society_show)
        pw.add(self.sc_tree, weight=2)
        right = ttk.Frame(pw)
        pw.add(right, weight=2)
        self.sc_text = ScrolledText(right, wrap="word", font=("Courier", 9))
        self.sc_text.pack(fill="both", expand=True)
        row = ttk.Frame(right)
        row.pack(fill="x")
        ttk.Label(row, text="Join the society as a human (the agent hears you in its next episode):").pack(anchor="w")
        self.sc_entry = ttk.Entry(row)
        self.sc_entry.pack(side="left", fill="x", expand=True)
        self.sc_entry.bind("<Return>", lambda e: self._society_say())
        ttk.Button(row, text="Say", command=self._society_say).pack(side="left", padx=4)
        self._sc_rows = {}

    def _society_now(self):
        if self._need_run():
            self._bg(lambda: self.orch.society.run(self.orch.cfg.society_episodes, seed=int(__import__("time").time()) % 10 ** 6,
                                                   quest_id=self.orch.db.get_meta("quest_id"), rnd=0, label="manual"), "society round")

    def _society_say(self):
        t = self.sc_entry.get().strip()
        if t and self.orch and self.orch.ctx:
            self.orch.society.human_inbox.append(t)
            self.sc_entry.delete(0, "end")
            self.sc_text.insert("end", f"\n[you] {t}\n")

    def _society_show(self, _=None):
        sel = self.sc_tree.selection()
        if sel and sel[0] in self._sc_rows:
            r = self._sc_rows[sel[0]]
            self.sc_text.delete("1.0", "end")
            for ep in r["log"]:
                self.sc_text.insert("end", f"=== {ep['kind']}  primary score {ep['primary']}\n")
                for step in ep["log"]:
                    self.sc_text.insert("end", "  " + json.dumps(step, ensure_ascii=False)[:300] + "\n")

    # =================================================================== THINKING
    def _tab_thinking(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Thinking")
        bar = ttk.Frame(t)
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Label(bar, text="Layer:").pack(side="left")
        self.th_filter = tk.StringVar(value="all")
        ttk.Combobox(bar, textvariable=self.th_filter, state="readonly", width=20,
                     values=("all", "evaluation_core", "researcher", "sandbox", "verifier", "failure_analysis", "memory", "improvement_engine", "red_team")).pack(side="left", padx=4)
        ttk.Label(bar, text="Every layer deliberates (analysis, risks, alternatives, decision, confidence) before it acts. Thoughts are advice; the checks stay mechanical.").pack(side="left", padx=8)
        pw = ttk.PanedWindow(t, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=8, pady=4)
        self.th_tree = ttk.Treeview(pw, columns=("id", "layer", "decision", "conf"), show="headings")
        for c, wd in zip(("id", "layer", "decision", "conf"), (40, 130, 380, 50)):
            self.th_tree.heading(c, text=c)
            self.th_tree.column(c, width=wd)
        self.th_tree.bind("<<TreeviewSelect>>", self._thought_show)
        pw.add(self.th_tree, weight=1)
        self.th_text = ScrolledText(pw, wrap="word", font=("TkDefaultFont", 10))
        pw.add(self.th_text, weight=1)
        self._th_rows = {}

    def _thought_show(self, _=None):
        sel = self.th_tree.selection()
        if sel and sel[0] in self._th_rows:
            r = self._th_rows[sel[0]]["result"]
            self.th_text.delete("1.0", "end")
            self.th_text.insert("end", f"LAYER: {r.get('layer')}\nQUESTION: {r.get('question')}\n\nANALYSIS\n{r.get('analysis')}\n\nRISKS\n- " + "\n- ".join(r.get("risks", [])) +
                                "\n\nALTERNATIVES\n- " + "\n- ".join(r.get("alternatives", [])) + f"\n\nDECISION\n{r.get('decision')}\n\nCONFIDENCE {r.get('confidence')}\n\nNEXT CHECKS\n- " +
                                "\n- ".join(r.get("next_checks", [])))

    # =================================================================== ENVIRONMENT
    def _tab_env(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Environment")
        bar = ttk.Frame(t)
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Button(bar, text="Re-scan machine", command=lambda: self._env_run(False)).pack(side="left")
        ttk.Button(bar, text="Install missing packages", command=lambda: self._env_run(True)).pack(side="left", padx=6)
        ttk.Label(bar, text="SIRA probes the hardware, installs the training stack (PyTorch, transformers, PEFT, ...) and adapts batch sizes / parallelism at start-up.").pack(side="left")
        self.env_text = ScrolledText(t, height=14, font=("Courier", 9))
        self.env_text.pack(fill="x", padx=8)
        ttk.Label(t, text="Learning algorithms known to SIRA (usable = all required packages and hardware present)").pack(anchor="w", padx=8, pady=(8, 0))
        self.alg_tree = ttk.Treeview(t, columns=("name", "kind", "usable", "needs", "summary"), show="headings", height=14)
        for c, wd in zip(("name", "kind", "usable", "needs", "summary"), (90, 120, 60, 130, 600)):
            self.alg_tree.heading(c, text=c)
            self.alg_tree.column(c, width=wd)
        self.alg_tree.pack(fill="both", expand=True, padx=8, pady=4)
        self.alg_tree.bind("<<TreeviewSelect>>", self._alg_show)
        self.alg_eq = ttk.Label(t, text="", font=("Courier", 10), foreground="#124")
        self.alg_eq.pack(anchor="w", padx=8, pady=4)
        self._alg_rows = {}

    def _env_run(self, install):
        if not self._need_run():
            return

        def job():
            from .bootstrap import Bootstrap
            cfg = self.orch.cfg
            old = cfg.auto_install
            cfg.auto_install = bool(install)
            try:
                self.orch.env_report = Bootstrap(cfg, self.orch.bus).run()
                self.orch.env_report["sandbox_libs"] = self.orch.bootstrap.sandbox_libs(self.orch.ctx.sandbox)
            finally:
                cfg.auto_install = old
        self._bg(job, "environment scan" + (" + install" if install else ""))

    def _alg_show(self, _=None):
        sel = self.alg_tree.selection()
        if sel and sel[0] in self._alg_rows:
            self.alg_eq.config(text=f"{sel[0]}:  {self._alg_rows[sel[0]]['equation']}")

    def _refresh_env(self):
        r = getattr(self.orch, "env_report", None)
        if not r:
            return
        hw = r["hw"]
        lines = [f"OS: {r['os']} ({r['machine']})   Python {r['python']}   tkinter: {r['tkinter']}   bwrap: {r['bwrap']}   docker: {r['docker']}",
                 f"CPU cores: {hw.get('cpu_count')}   RAM: {hw.get('ram_gb')} GB   free disk: {r['disk_free_gb']} GB   CUDA driver: {r['cuda_driver'] or '-'}",
                 "GPU: " + (", ".join(f"{g['name']} {g['vram_gb']} GB (bf16={g['bf16']})" for g in hw["gpus"]) or "none (CPU only)"), "", "Python packages:"]
        lines += [f"  {k:16s} {v or 'MISSING'}" for k, v in r["packages"].items()]
        lines += ["", f"Sandbox science libraries: {r.get('sandbox_libs')}", f"Installed now: {r.get('installed_now')}   failed: {list((r.get('failed') or {}))}",
                  "Adapted settings: " + (", ".join(f"{k}: {o}->{n}" for k, (o, n) in (r.get("adapted") or {}).items()) or "none")]
        txt = "\n".join(lines)
        if self.env_text.get("1.0", "end").strip() != txt.strip():
            self.env_text.delete("1.0", "end")
            self.env_text.insert("end", txt)
        self._alg_rows = {a["name"]: a for a in r["algorithms"]}
        self._refill(self.alg_tree, [(a["name"], (a["name"], a["kind"], "yes" if a["usable"] else "no", ", ".join(a["missing"]) or "-", a["summary"]))
                                     for a in r["algorithms"]])

    # =================================================================== periodic refresh (called by App._refresh)
    def _extra_refresh(self):
        o = self.orch
        if not o.ctx:
            return
        db = o.db
        now = time.time()
        if self.brain_struct is None and not getattr(self, "_brain_drawn_empty", False):
            self._brain_drawn_empty = True
            self._brain_draw()
        if getattr(self, "_brain_sig", None) != id(self.brain_struct):
            self._brain_sig = id(self.brain_struct)
            self._brain_draw()
        if now - self._extra_last < 3.0:
            return
        self._extra_last = now
        self._refill(self.w_tree, [(w["name"], (w["name"], w["domain"], w["kind"], w.get("status", "built-in")))
                                   for w in o.ctx.workshop.list_worlds()["worlds"]])
        self._refill(self.t_tree, [(t["name"], (t["name"], t["status"], t["uses"], t["created_by"])) for t in db.tools()])
        self._refill(self.ev_tree, [(str(e["id"]), (e["id"], e["label"][:40], f"{e['total']:.3f}", e["n"])) for e in db.evals(150)])
        self._refill(self.c_tree, [(str(c["id"]), (c["id"], c["title"][:60])) for c in db.concepts()])
        rep = sorted((o.run_dir / "reports").glob("*.md"), reverse=True)
        self._refill(self.r_tree, [(p.name, (p.name,)) for p in rep])
        self.m_text.delete("1.0", "end")
        for m in db.mistakes(30):
            self.m_text.insert("end", f"[{m['area']}] {m['description'][:160]}" + (f"  FIX: {m['fix'][:100]}" if m["fix"] else "") + "\n")
        self._refill(self.bk_tree, [(b["name"], (b["name"], b["label"], b["best_generation"], f"{b['best_score']:.3f}" if b["best_score"] is not None else "-",
                                                  "yes" if b["weights"] else "no", b["files"])) for b in o.backup.list()])
        # quest / challenges / society
        qid = db.get_meta("quest_id")
        if qid:
            q = db.quest(qid)
            st = db.get_meta("quest_state", {}) or {}
            self.q_head.config(text=f"QUEST: {q['goal']}   |   status: {q['status']}   |   rounds {st.get('rounds', 0)}   streak {st.get('streak', 0)}")
            self._refill(self.qr_tree, [(str(h["round"]), (h["round"], h["exam"], h["blank"], f"{h['delta']:+.2f}", h["coverage"], h["life"], h["skill"],
                                                           None if h.get("brier") is None else round(h["brier"], 3))) for h in st.get("history", [])])
            if not self.qr_tree.selection() and not self.q_text.get("1.0", "end").strip():
                self.q_text.insert("end", f"{q['definition']}\n\nCRITERIA\n" + "\n".join(f"{c.get('id')}: {c['text']}  [test: {c.get('test')}]" for c in q["criteria"]) +
                                   f"\n\nTOPICS QUEUE: {db.get_meta('quest_topics', [])}")
        self._refill(self.ch_tree, [(str(c["id"]), (c["id"], c["title"][:50], c["domain"], c["kind"], c["difficulty"], c["status"], f"{c['best_score']:.2f}", c["attempts"])) for c in db.challenges(None, 200)])
        tot = o.challenges.ledger.totals()
        sk = db.get_meta("skills", {})
        self.ch_head.config(text=f"reward {tot['reward']:.1f}   penalty {tot['penalty']:.1f}   NET {tot['net']:.1f}   |   skills (EMA, level): " + ", ".join(f"{k}: {v['ema']:.2f}/L{v['level']}" for k, v in sk.items()))
        self._refill(self.rw_tree, [(str(r["id"]), (r["source"], r["item"][:40], f"{r['reward']:.1f}", f"{r['penalty']:.1f}", r["note"][:60])) for r in db.rewards(150)])
        runs = db.society_runs(60)
        self._sc_rows = {str(r["id"]): r for r in runs}
        self._refill(self.sc_tree, [(str(r["id"]), (r["id"], r["round"], r["metrics"].get("life_score"), r["metrics"].get("false_belief"), r["metrics"].get("hidden_need"), r["metrics"].get("promise"),
                                                    r["metrics"].get("honesty"), r["metrics"].get("deception"))) for r in runs])
        if o.society.human_outbox:
            self.sc_text.insert("end", "\n[agent] " + o.society.human_outbox.pop(0) + "\n")
        self._refill(self.bg_tree, [(str(g["id"]), (g["id"], g["method"], f"{g['score']:.3f}", "yes" if g["accepted"] else "no", (g["path"] or "-")[-60:]))
                                    for g in db.generations()])
        self._refresh_files()
        self._refresh_stats()
        self._refresh_env()
        f = self.th_filter.get()
        rows = db.thoughts(150, None if f == "all" else f)
        self._th_rows = {str(r["id"]): r for r in rows}
        self._refill(self.th_tree, [(str(r["id"]), (r["id"], r["layer"], r["result"].get("decision", "")[:90], r["result"].get("confidence"))) for r in rows])
