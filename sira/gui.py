"""Tkinter GUI (pure Python, no web). Full visibility + human control:
Pause/Resume, STOP, send directives, change budget limits live, approve training rounds, veto claims."""
from __future__ import annotations

import json
import queue
import threading
import time
import tkinter as tk
from dataclasses import fields
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .config import Config
from .control import StopReason
from .gui_extra import ExtraTabs
from .llm import LLM, PROVIDERS, ollama_base, ollama_models, ollama_pull
from .orchestrator import Orchestrator
from .setup_layout import layout
from .ui_kit import ScrollFrame, apply_theme

LEVEL_COLORS = {"info": "#222222", "warn": "#b36b00", "error": "#c00000"}
KIND_COLORS = {"tool": "#2060a0", "experiment": "#006666", "verify": "#7a2fa0", "thought": "#777777",
               "plan": "#0a7a0a", "eval": "#0a7a0a", "generation": "#0a7a0a", "human": "#c00000", "control": "#c00000"}


class App(tk.Tk, ExtraTabs):
    def __init__(self, cfg: Config | None = None):
        super().__init__()
        apply_theme(self)
        self.title("SIRA - Self-Improving Research Agent")
        self.geometry("1280x820")
        self.minsize(960, 600)
        self.cfg = cfg or Config()
        self.orch: Orchestrator | None = None
        self.vars: dict[str, tk.Variable] = {}
        self._uiq: queue.Queue = queue.Queue()
        self._last_refresh = 0.0
        self._shown_alert = None
        self._build_topbar()
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=6, pady=4)
        self._tab_setup()
        self._tab_dashboard()
        self._tab_agents()
        self._tab_experiments()
        self._tab_knowledge()
        self._tab_log()
        self._tab_sandbox()
        self._tab_model()
        self._extra_tabs()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(200, self._tick)

    # ------------------------------------------------------------------ top bar
    def _build_topbar(self):
        f = ttk.Frame(self)
        f.pack(fill="x", padx=6, pady=4)
        self.status_lbl = ttk.Label(f, text="NOT STARTED", font=("TkDefaultFont", 12, "bold"), width=30)
        self.status_lbl.pack(side="left")
        self.pause_btn = ttk.Button(f, text="Pause", command=self._pause_toggle, state="disabled")
        self.pause_btn.pack(side="left", padx=4)
        self.stop_btn = tk.Button(f, text="STOP", bg="#c00000", fg="white", font=("TkDefaultFont", 11, "bold"),
                                  command=self._stop, state="disabled", width=8)
        self.stop_btn.pack(side="left", padx=8)
        ttk.Label(f, text="Directive to Grand Maestro:").pack(side="left", padx=(20, 4))
        self.dir_entry = ttk.Entry(f, width=55)
        self.dir_entry.pack(side="left")
        self.dir_entry.bind("<Return>", lambda e: self._send_directive())
        ttk.Button(f, text="Send", command=self._send_directive).pack(side="left", padx=4)
        self.alert_lbl = tk.Label(self, text="", fg="white", bg="#c00000")

    # ------------------------------------------------------------------ setup tab
    def _tab_setup(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Setup")
        spec = [("mode", "MODE: model | quest | both | auto (auto = give only a goal, it does everything)", ("model", "quest", "both", "auto")),
                ("quest_goal", "Quest goal (e.g. 'reach AGI', 'reach artificial consciousness') - no headings needed", "str"),
                ("quest_seeds", "Optional hints (comma separated; may be empty)", "list"),
                ("examiner_provider", "Examiner model provider (blank = same brain; may be a smaller model)", "str"), ("examiner_model", "Examiner model name", "str"),
                ("society_enabled", "Virtual society of humans (life test)", "bool"),
                ("base_model", "Base model (Hugging Face id or local folder)", "str"),
                ("goal_text", "Goal", "str"), ("target_score", "Target score on Fresh Set (0-1)", "float"),
                ("confirm_rounds", "Confirmation rounds", "int"),
                ("budget_usd", "Budget: LLM spend (USD)", "float"), ("budget_gpu_hours", "Budget: GPU hours", "float"),
                ("budget_wall_hours", "Budget: wall-clock hours", "float"),
                ("llm_provider", "Brain provider (the LLM that drives the agents)", tuple(PROVIDERS)), ("llm_model", "Brain model (e.g. llama3.1:8b for Ollama)", "str"),
                ("llm_base_url", "Brain base URL (Ollama: http://localhost:11434, OpenAI-compatible: https://host/v1)", "str"),
                ("llm_api_key_env", "Environment variable holding the API key (blank = provider default)", "str"),
                ("llm_num_ctx", "Ollama context window (tokens)", "int"), ("llm_json_mode", "Ask the server for JSON output", "bool"),
                ("thinking", "Thinking in every layer (deliberation effort)", ("off", "low", "medium", "high")),
                ("algorithm", "Learning algorithm", ("auto", "rft", "grpo", "dpo")), ("lora_variant", "Adapter variant", ("lora", "rslora", "dora")),
                ("lora_plus_ratio", "LoRA+ ratio (1 = off, 16 = typical)", "float"), ("optimizer", "Optimizer", ("adamw", "adamw8bit")),
                ("accept_p", "Acceptance p-value for new generations (1 = off)", "float"),
                ("auto_install", "Install missing Python packages at start-up", "bool"), ("auto_adapt", "Adapt settings to this machine at start-up", "bool"),
                ("sandbox_backend", "Sandbox backend", ("auto", "docker", "bwrap", "subprocess")),
                ("sandbox_network", "Experiment network", ("none", "open")),
                ("allow_weak_sandbox", "Allow WEAK sandbox (no host isolation!)", "bool"),
                ("require_approval", "Human approves every training round", "bool"),
                ("allow_cpu_training", "Allow CPU training (tiny models)", "bool"),
                ("eval_problems", "Eval problems per fresh set", "int"), ("train_problems", "Training problems per round", "int"),
                ("parallel_experiments", "Parallel experiments", "int"),
                ("scratch_language", "From scratch: language (Wikipedia code: en, ar, ...)", "str"),
                ("scratch_size", "From scratch: model size", ("auto", "tiny", "small", "base")),
                ("scratch_arch", "From scratch: architecture (cla2/cla3 = Cross-Layer Attention)", ("gqa", "llama", "mha", "mqa", "cla2", "cla3", "custom")),
                ("arch_kv_heads", "Custom architecture: KV heads (0 = default)", "int"), ("arch_cla_factor", "Custom architecture: layers sharing K/V (1 = off)", "int"),
                ("scratch_corpus_dir", "From scratch: folder with your .txt corpus (optional)", "str"),
                ("scratch_wiki_mb", "From scratch: MB of Wikipedia text to fetch", "int"),
                ("scratch_steps", "From scratch: pretraining steps", "int")]
        self.spec = spec
        by_key = {k: (lbl, typ) for k, lbl, typ in spec}

        head = ttk.Frame(t, padding=(14, 10, 14, 0))
        head.pack(side="top", fill="x")
        ttk.Label(head, text="Configure a run", font=("TkDefaultFont", 15, "bold")).pack(anchor="w")
        ttk.Label(head, text="Fill in the sections, then press START at the bottom. Scroll with the mouse wheel or jump to a section:",
                  style="Hint.TLabel").pack(anchor="w", pady=(2, 6))
        chips = ttk.Frame(head)
        chips.pack(anchor="w")

        bar = ttk.Frame(t, style="Bar.TFrame", padding=(12, 8))       # packed before the scroller so it is always visible
        bar.pack(side="bottom", fill="x")
        sf = ScrollFrame(t)
        sf.pack(side="top", fill="both", expand=True, pady=(8, 0))

        for title, short, hint, keys in layout([k for k, _, _ in spec]):
            card = ttk.LabelFrame(sf.body, text=title, padding=(14, 8))
            card.pack(fill="x", padx=12, pady=6)
            card.columnconfigure(1, weight=1)
            r = 0
            if hint:
                ttk.Label(card, text=hint, style="Hint.TLabel", wraplength=900, justify="left").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
                r = 1
            for key in keys:
                label, typ = by_key[key]
                ttk.Label(card, text=label, wraplength=430, justify="left").grid(row=r, column=0, sticky="nw", padx=(0, 14), pady=4)
                w = self._make_field(card, sf, key, typ)
                w.grid(row=r, column=1, sticky="ew" if typ in ("str", "list") else "w", pady=4)
                r += 1
            ttk.Button(chips, text=short, style="Chip.TButton", command=lambda c=card: sf.scroll_to(c)).pack(side="left", padx=(0, 6))

        row1 = ttk.Frame(bar, style="Bar.TFrame")
        row1.pack(fill="x")
        btn = dict(fg="white", relief="flat", font=("TkDefaultFont", 11, "bold"), padx=14, pady=6, cursor="hand2", activeforeground="white")
        self.start_btn = tk.Button(row1, text="START RUN (existing model)", bg="#0a7a0a", activebackground="#0c9a0c", command=self._start, **btn)
        self.start_btn.pack(side="left")
        self.scratch_btn = tk.Button(row1, text="CREATE MODEL FROM SCRATCH + START", bg="#1a4fa0", activebackground="#2362c4",
                                     command=lambda: self._start(scratch=True), **btn)
        self.scratch_btn.pack(side="left", padx=10)
        row2 = ttk.Frame(bar, style="Bar.TFrame")
        row2.pack(fill="x", pady=(8, 0))
        for text, cmd in (("Load config...", self._load), ("Save config...", self._save), ("Test brain connection", self._test_brain),
                          ("Ollama: list models", self._ollama_list), ("Ollama: pull model", self._ollama_pull)):
            ttk.Button(row2, text=text, command=cmd).pack(side="left", padx=(0, 6))
        ttk.Label(bar, style="BarHint.TLabel",
                  text="API keys are read from environment variables (ANTHROPIC_API_KEY / OPENAI_API_KEY, HF_TOKEN). "
                       "The run can only end by: goal reached, budget exhausted, or your STOP.").pack(anchor="w", pady=(8, 0))

    def _make_field(self, parent, sf, key, typ):
        val = getattr(self.cfg, key)
        if typ == "bool":
            v = tk.BooleanVar(value=val)
            w = ttk.Checkbutton(parent, variable=v)
        elif typ == "list":
            v = tk.StringVar(value=", ".join(val))
            w = ttk.Entry(parent, textvariable=v)
        elif isinstance(typ, tuple):
            v = tk.StringVar(value=val)
            w = ttk.Combobox(parent, textvariable=v, values=typ, state="readonly", width=24)
            sf.guard_wheel(w)
        else:
            v = tk.StringVar(value=str(val))
            w = ttk.Entry(parent, textvariable=v, width=18 if typ in ("int", "float") else 40)
        self.vars[key] = v
        return w

    def _collect(self) -> Config:
        kw = {}
        for key, _, typ in self.spec:
            raw = self.vars[key].get()
            kw[key] = bool(raw) if typ == "bool" else int(float(raw)) if typ == "int" else float(raw) if typ == "float" else \
                [x.strip() for x in str(raw).split(",") if x.strip()] if typ == "list" else str(raw)
        base = {f.name: getattr(self.cfg, f.name) for f in fields(Config)}
        base.update(kw)
        cfg = Config(**base)
        cfg.validate()
        return cfg

    def _load(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if p:
            self.cfg = Config.load(p)
            for key, _, typ in self.spec:
                v = getattr(self.cfg, key)
                self.vars[key].set(bool(v) if typ == "bool" else ", ".join(v) if typ == "list" else str(v))

    def _save(self):
        try:
            cfg = self._collect()
        except Exception as e:
            return messagebox.showerror("Invalid settings", str(e))
        p = filedialog.asksaveasfilename(defaultextension=".json")
        if p:
            cfg.save(p)

    def _brain_cfg(self):
        return self._collect()

    def _test_brain(self):
        try:
            cfg = self._brain_cfg()
        except Exception as e:
            return messagebox.showerror("Invalid settings", str(e))

        def job():
            from .bus import EventBus
            from .control import RunControl
            from .db import DB
            import tempfile
            ctl = RunControl(cfg, DB(tempfile.mkdtemp() + "/t.db"), EventBus())
            ok, msg = LLM(cfg, ctl).check()
            self.ui(lambda: (messagebox.showinfo if ok else messagebox.showerror)("Brain connection", msg))
        threading.Thread(target=job, daemon=True).start()

    def _ollama_list(self):
        try:
            base = ollama_base(self._collect())
            names = ollama_models(base)
            messagebox.showinfo("Ollama", f"{base}\n\n" + ("\n".join(names) if names else "no models installed"))
        except Exception as e:
            messagebox.showerror("Ollama", f"Cannot reach Ollama: {e}\nStart it with: ollama serve")

    def _ollama_pull(self):
        try:
            cfg = self._collect()
        except Exception as e:
            return messagebox.showerror("Invalid settings", str(e))
        base, model = ollama_base(cfg), cfg.llm_model

        def job():
            try:
                ollama_pull(base, model, lambda ev: self.ui(lambda: self.status_lbl.config(text=f"ollama: {ev.get('status', '')[:40]}")))
                self.ui(lambda: messagebox.showinfo("Ollama", f"{model} is ready"))
            except Exception as e:
                self.ui(lambda: messagebox.showerror("Ollama", str(e)))
        threading.Thread(target=job, daemon=True).start()

    def _start(self, scratch=False, run_dir=None):
        try:
            cfg = self._collect()
            cfg.scratch_mode = bool(scratch)
            if scratch:
                cfg.base_model = ""
            cfg.validate()
        except Exception as e:
            return messagebox.showerror("Invalid settings", str(e))
        if cfg.allow_weak_sandbox and not messagebox.askyesno(
                "Weak sandbox", "The weak sandbox does NOT isolate your files or network from model-written code.\nContinue anyway?"):
            return
        self.cfg = cfg
        self.orch = Orchestrator(cfg, run_dir=run_dir)
        self.orch.start()
        self.start_btn.config(state="disabled")
        self.scratch_btn.config(state="disabled")
        self.pause_btn.config(state="normal")
        self.stop_btn.config(state="normal")
        self.nb.select(1)

    # ------------------------------------------------------------------ controls
    def _pause_toggle(self):
        if not self.orch:
            return
        c = self.orch.control
        if c.paused:
            c.resume()
            self.pause_btn.config(text="Pause")
        else:
            c.pause()
            self.pause_btn.config(text="Resume")

    def _stop(self):
        if self.orch and messagebox.askyesno("STOP", "Stop the whole run now? Running experiments and training are killed."):
            self.orch.stop()

    def _send_directive(self):
        t = self.dir_entry.get().strip()
        if t and self.orch:
            self.orch.control.send_directive(t)
            self.dir_entry.delete(0, "end")

    def _on_close(self):
        if self.orch and not self.orch.control.stopped:
            if not messagebox.askyesno("Quit", "A run is active. Stop it and quit?"):
                return
            self.orch.stop()
            if self.orch.thread:
                self.orch.thread.join(timeout=10)
        self.destroy()

    # ------------------------------------------------------------------ dashboard
    def _tab_dashboard(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Dashboard")
        top = ttk.LabelFrame(t, text="Budget (limits are editable live - only YOU can change them)")
        top.pack(fill="x", padx=8, pady=6)
        self.bars, self.limit_vars, self.bar_lbls = {}, {}, {}
        for i, (k, label) in enumerate((("usd", "LLM spend (USD)"), ("gpu_h", "GPU hours"), ("wall_h", "Wall-clock hours"))):
            ttk.Label(top, text=label, width=18).grid(row=i, column=0, sticky="w", padx=4, pady=3)
            pb = ttk.Progressbar(top, length=420, maximum=100)
            pb.grid(row=i, column=1, padx=4)
            lbl = ttk.Label(top, text="-", width=26)
            lbl.grid(row=i, column=2)
            v = tk.StringVar()
            ttk.Entry(top, textvariable=v, width=9).grid(row=i, column=3, padx=4)
            self.bars[k], self.bar_lbls[k], self.limit_vars[k] = pb, lbl, v
        ttk.Button(top, text="Apply limits", command=self._apply_limits).grid(row=0, column=4, padx=8)
        self.tok_lbl = ttk.Label(top, text="")
        self.tok_lbl.grid(row=3, column=0, columnspan=4, sticky="w", padx=4)
        mid = ttk.Frame(t)
        mid.pack(fill="both", expand=True, padx=8)
        left = ttk.LabelFrame(mid, text="Fresh-Set score per generation (accepted = filled)")
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, bg="white", height=260)
        self.canvas.pack(fill="both", expand=True)
        right = ttk.LabelFrame(mid, text="Generations")
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        cols = ("id", "parent", "method", "score", "parent_score", "accepted", "notes")
        self.gen_tree = ttk.Treeview(right, columns=cols, show="headings", height=12)
        for c in cols:
            self.gen_tree.heading(c, text=c)
            self.gen_tree.column(c, width=70 if c != "notes" else 260)
        self.gen_tree.pack(fill="both", expand=True)
        self.dom_lbl = ttk.Label(t, text="", font=("TkDefaultFont", 10))
        self.dom_lbl.pack(fill="x", padx=8, pady=6)

    def _apply_limits(self):
        if not self.orch:
            return
        try:
            self.orch.control.set_limits(*(float(self.limit_vars[k].get()) for k in ("usd", "gpu_h", "wall_h")))
        except ValueError:
            messagebox.showerror("Limits", "Enter numbers")

    def _draw_chart(self, gens):
        cv = self.canvas
        cv.delete("all")
        w, h = max(cv.winfo_width(), 300), max(cv.winfo_height(), 200)
        l, r, tp, b = 45, 15, 15, 30
        for i in range(0, 11, 2):
            y = tp + (h - tp - b) * (1 - i / 10)
            cv.create_line(l, y, w - r, y, fill="#e5e5e5")
            cv.create_text(l - 6, y, text=f"{i / 10:.1f}", anchor="e", fill="#777")
        tgt = self.cfg.target_score
        y = tp + (h - tp - b) * (1 - tgt)
        cv.create_line(l, y, w - r, y, fill="#0a7a0a", dash=(4, 3))
        cv.create_text(w - r - 4, y - 8, text="target", anchor="e", fill="#0a7a0a")
        pts = [g for g in gens if g["method"] != "failed"]
        if not pts:
            return
        n = len(pts)
        xs = [l + (w - l - r) * (i / max(1, n - 1)) if n > 1 else (l + w - r) / 2 for i in range(n)]
        ys = [tp + (h - tp - b) * (1 - g["score"]) for g in pts]
        acc = [(x, y) for x, y, g in zip(xs, ys, pts) if g["accepted"]]
        if len(acc) > 1:
            cv.create_line(*[c for p in acc for c in p], fill="#2060a0", width=2)
        for x, y, g in zip(xs, ys, pts):
            col = "#2060a0" if g["accepted"] else "#c00000"
            cv.create_oval(x - 5, y - 5, x + 5, y + 5, outline=col, fill=col if g["accepted"] else "white", width=2)
            cv.create_text(x, y - 12, text=f"{g['score']:.2f}", fill=col)
            cv.create_text(x, h - b + 12, text=f"#{g['id']}", fill="#555")

    # ------------------------------------------------------------------ agents
    def _tab_agents(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Agents")
        self.agent_tree = ttk.Treeview(t, columns=("role", "status", "age"), show="tree headings")
        self.agent_tree.heading("#0", text="agent (hierarchy)")
        self.agent_tree.column("#0", width=300)
        for c, wd in (("role", 160), ("status", 520), ("age", 80)):
            self.agent_tree.heading(c, text=c)
            self.agent_tree.column(c, width=wd)
        self.agent_tree.pack(fill="both", expand=True, padx=8, pady=8)

    def _refresh_agents(self):
        st = dict(self.orch.bus.agent_status)
        now = time.time()
        self.agent_tree.delete(*self.agent_tree.get_children())

        def add(name, parent_iid=""):
            s = st[name]
            iid = self.agent_tree.insert(parent_iid, "end", text=name, open=True,
                                         values=(s.get("role", ""), s.get("status", ""), f"{int(now - s['t'])}s"))
            for k, v in sorted(st.items()):
                if v.get("parent") == name and k != name:
                    add(k, iid)

        for n, v in sorted(st.items()):
            if not v.get("parent") or v["parent"] not in st:
                add(n)

    # ------------------------------------------------------------------ experiments
    def _tab_experiments(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Experiments")
        pw = ttk.PanedWindow(t, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=8, pady=8)
        cols = ("id", "agent", "hypothesis", "exit", "sec")
        self.exp_tree = ttk.Treeview(pw, columns=cols, show="headings")
        for c, wd in zip(cols, (50, 110, 330, 45, 50)):
            self.exp_tree.heading(c, text=c)
            self.exp_tree.column(c, width=wd)
        self.exp_tree.bind("<<TreeviewSelect>>", self._show_exp)
        pw.add(self.exp_tree, weight=1)
        self.exp_text = ScrolledText(pw, wrap="none", font=("Courier", 9))
        pw.add(self.exp_text, weight=1)

    @staticmethod
    def _refill(tree, rows):
        keep = set(tree.selection())
        tree.delete(*tree.get_children())
        for iid, vals in rows:
            tree.insert("", "end", iid=iid, values=vals)
        present = [i for i in keep if tree.exists(i)]
        if present:
            tree.selection_set(present)

    def _show_exp(self, _=None):
        sel = self.exp_tree.selection()
        if not sel or not self.orch:
            return
        e = self.orch.db.experiment(int(sel[0]))
        if e:
            self.exp_text.delete("1.0", "end")
            self.exp_text.insert("end", f"# experiment {e['id']} by {e['agent']} seed={e['seed']} backend={e['backend']} "
                                        f"exit={e['exit_code']} timeout={bool(e['timed_out'])}\n# {e['hypothesis']}\n\n{e['code']}\n\n"
                                        f"--- result ---\n{e['result']}\n--- stdout ---\n{e['stdout']}\n--- stderr ---\n{e['stderr']}")

    # ------------------------------------------------------------------ knowledge
    def _tab_knowledge(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Knowledge")
        bar = ttk.Frame(t)
        bar.pack(fill="x", padx=8, pady=4)
        self.kfilter = tk.StringVar(value="all")
        ttk.Label(bar, text="Show:").pack(side="left")
        ttk.Combobox(bar, textvariable=self.kfilter, values=("all", "verified", "refuted", "inconclusive", "lead", "vetoed"),
                     state="readonly", width=14).pack(side="left", padx=4)
        ttk.Button(bar, text="Veto selected claim", command=self._veto).pack(side="left", padx=10)
        ttk.Label(bar, text="(only 'verified' claims passed experiment + replication + audit; 'lead' = unverified)").pack(side="left")
        pw = ttk.PanedWindow(t, orient="vertical")
        pw.pack(fill="both", expand=True, padx=8, pady=4)
        self.k_tree = ttk.Treeview(pw, columns=("id", "status", "text", "source"), show="headings")
        for c, wd in zip(("id", "status", "text", "source"), (50, 90, 700, 250)):
            self.k_tree.heading(c, text=c)
            self.k_tree.column(c, width=wd)
        self.k_tree.bind("<<TreeviewSelect>>", self._show_claim)
        pw.add(self.k_tree, weight=3)
        self.k_text = ScrolledText(pw, height=8, font=("Courier", 9))
        pw.add(self.k_text, weight=1)

    def _veto(self):
        sel = self.k_tree.selection()
        if sel and self.orch:
            self.orch.db.set_claim_status(int(sel[0]), "vetoed")
            self.orch.bus.emit("human", "human", f"claim #{sel[0]} vetoed", level="warn")

    def _show_claim(self, _=None):
        sel = self.k_tree.selection()
        if sel and self.orch:
            for c in self.orch.db.claims(None, 2000):
                if c["id"] == int(sel[0]):
                    self.k_text.delete("1.0", "end")
                    self.k_text.insert("end", c["text"] + "\n\n" + json.dumps(json.loads(c["evidence"] or "{}"), indent=2)[:6000])
                    break

    # ------------------------------------------------------------------ log / sandbox / model
    def _tab_log(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Live log")
        bar = ttk.Frame(t)
        bar.pack(fill="x", padx=8, pady=4)
        self.lfilter = tk.StringVar(value="all")
        ttk.Label(bar, text="Filter:").pack(side="left")
        ttk.Combobox(bar, textvariable=self.lfilter, state="readonly", width=14,
                     values=("all", "important", "tool", "experiment", "verify", "thought", "train", "error")).pack(side="left", padx=4)
        self.log = ScrolledText(t, wrap="word", font=("Courier", 9), state="disabled")
        self.log.pack(fill="both", expand=True, padx=8, pady=4)
        for k, col in {**{f"lvl_{k}": v for k, v in LEVEL_COLORS.items()}, **{f"kind_{k}": v for k, v in KIND_COLORS.items()}}.items():
            self.log.tag_config(k, foreground=col)

    def _log_event(self, ev):
        f = self.lfilter.get()
        kind, lvl = ev["kind"], ev["level"]
        if kind == "agent_status":
            return
        if f == "important" and lvl == "info" and kind not in ("plan", "eval", "generation", "verify", "run", "goal"):
            return
        if f == "error" and lvl not in ("warn", "error"):
            return
        if f not in ("all", "important", "error") and not kind.startswith(f):
            return
        tag = f"lvl_{lvl}" if lvl != "info" else (f"kind_{kind}" if kind in KIND_COLORS else "lvl_info")
        self.log.config(state="normal")
        self.log.insert("end", f"{time.strftime('%H:%M:%S', time.localtime(ev['t']))} [{ev['agent']}] {ev['msg']}\n", tag)
        if int(self.log.index("end-1c").split(".")[0]) > 6000:
            self.log.delete("1.0", "1000.0")
        self.log.see("end")
        self.log.config(state="disabled")

    def _tab_sandbox(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Sandbox")
        self.sb_text = ScrolledText(t, font=("Courier", 10))
        self.sb_text.pack(fill="both", expand=True, padx=8, pady=8)

    def _tab_model(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Model intake")
        self.model_text = ScrolledText(t, font=("Courier", 10))
        self.model_text.pack(fill="both", expand=True, padx=8, pady=8)

    # ------------------------------------------------------------------ periodic update
    def ui(self, fn):
        """Thread-safe: run fn on the GUI thread (background threads must never touch Tk directly)."""
        self._uiq.put(fn)

    def _tick(self):
        try:
            while not self._uiq.empty():
                self._uiq.get_nowait()()
            if self.orch:
                for ev in self.orch.bus.drain():
                    self._log_event(ev)
                    if ev["kind"] == "selftest" and "report" in ev["data"]:
                        self._show_selftest(ev["data"]["report"])
                    if ev["kind"] == "intake" and "report" in ev["data"]:
                        self.model_text.delete("1.0", "end")
                        self.model_text.insert("end", json.dumps(ev["data"], indent=2, default=str))
                self._poll_approvals()
                if time.time() - self._last_refresh > 1.5:
                    self._last_refresh = time.time()
                    self._refresh()
        except Exception as e:                      # the GUI must never die because of a refresh glitch
            print("GUI refresh error:", e)
        self.after(250, self._tick)

    def _poll_approvals(self):
        c = self.orch.control
        while not c.approvals.empty():
            box = c.approvals.get()
            box["answer"] = messagebox.askyesno(box["title"], box["detail"])
            box["ev"].set()

    def _show_selftest(self, rep):
        self.sb_text.delete("1.0", "end")
        self.sb_text.insert("end", f"Backend: {rep['backend']}   strong isolation: {rep['strong']}   self-test passed: {rep['passed']}\n\n")
        for c in rep["checks"]:
            self.sb_text.insert("end", f"[{'PASS' if c['ok'] else 'FAIL'}] {c['name']}{'' if c['critical'] else ' (non-critical)'} {c['detail']}\n")
        self.sb_text.insert("end", "\nExperiment code runs ONLY here. Web tools are open to the whole public internet; "
                                   "downloads land in a read-only data folder and are never executed outside the sandbox.\n")

    def _refresh(self):
        o = self.orch
        c = o.control
        if c.stopped:
            self.status_lbl.config(text=f"ENDED: {c.stop_reason.value.upper()}", foreground="#c00000")
            self.pause_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
        elif c.paused:
            self.status_lbl.config(text="PAUSED", foreground="#b36b00")
        else:
            self.status_lbl.config(text=f"RUNNING (cycle {o.cycle})", foreground="#0a7a0a")
        if c.integrity_alert and c.integrity_alert != self._shown_alert:
            self._shown_alert = c.integrity_alert
            self.alert_lbl.config(text="  " + c.integrity_alert + "  ")
            self.alert_lbl.pack(fill="x", before=self.nb)
            messagebox.showerror("Safety alert", c.integrity_alert + "\n\nReview, then press Resume (or STOP).")
        elif not c.integrity_alert:
            self.alert_lbl.pack_forget()
            self._shown_alert = None
        b = c.budget_status()
        for k, unit in (("usd", "$"), ("gpu_h", "h"), ("wall_h", "h")):
            spent, lim = b[k]
            self.bars[k]["value"] = min(100, 100 * spent / lim)
            self.bar_lbls[k].config(text=f"{spent:.2f} / {lim:.2f} {unit}")
            if not self.focus_get() or str(self.focus_get().winfo_class()) != "TEntry":
                self.limit_vars[k].set(f"{lim:g}")
        self.tok_lbl.config(text=f"tokens in/out: {b['tokens'][0]:,} / {b['tokens'][1]:,}")
        gens = o.db.generations()
        self._draw_chart(gens)
        self.gen_tree.delete(*self.gen_tree.get_children())
        for g in gens:
            ps = g["detail"].get("parent_score")
            self.gen_tree.insert("", "end", values=(g["id"], g["parent"] or "-", g["method"], f"{g['score']:.3f}",
                                                     f"{ps:.3f}" if ps is not None else "-", "yes" if g["accepted"] else "no", g["notes"]))
        if gens and gens[-1]["detail"].get("by_domain"):
            self.dom_lbl.config(text="Latest per-domain: " + "   ".join(
                f"{k}: {v['rate']:.0%}" for k, v in gens[-1]["detail"]["by_domain"].items()))
        self._refill(self.exp_tree, [(str(e["id"]), (e["id"], e["agent"], (e["hypothesis"] or "")[:90], e["exit_code"],
                                                       f"{e['duration']:.1f}")) for e in o.db.experiments(200)])
        flt = self.kfilter.get()
        self._refill(self.k_tree, [(str(cl["id"]), (cl["id"], cl["status"], cl["text"][:160], (cl["source"] or "")[:60]))
                                   for cl in o.db.claims(None if flt == "all" else flt, 400)])
        self._refresh_agents()
        self._extra_refresh()


def run_gui(cfg: Config | None = None, resume_dir=None):
    app = App(cfg)
    if resume_dir:
        app.after(300, lambda: app._start(scratch=cfg.scratch_mode, run_dir=resume_dir))
    app.mainloop()
