"""Small tkinter helpers: a vertically scrolling frame (mouse wheel on Windows, macOS and Linux) and a calmer ttk theme."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

BAR_BG = "#e8edf3"


def apply_theme(root) -> None:
    st = ttk.Style(root)
    try:
        if "clam" in st.theme_names():
            st.theme_use("clam")
    except tk.TclError:
        pass
    st.configure("TNotebook.Tab", padding=(16, 7), font=("TkDefaultFont", 10, "bold"))
    st.configure("TLabelframe.Label", font=("TkDefaultFont", 10, "bold"), foreground="#0b4a99")
    st.configure("TButton", padding=(10, 5))
    st.configure("Chip.TButton", padding=(10, 3))
    st.configure("Hint.TLabel", foreground="#5b6770")
    st.configure("Bar.TFrame", background=BAR_BG)
    st.configure("BarHint.TLabel", background=BAR_BG, foreground="#5b6770")


class ScrollFrame(ttk.Frame):
    """Vertical scrolling container. Put your widgets into `.body`."""

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        bg = ttk.Style(self).lookup("TFrame", "background") or "#f0f0f0"
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background=bg)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set, yscrollincrement=24)
        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.body = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):      # Windows/macOS/Tk9 | Linux with Tk 8.6
            self.bind_all(seq, self._on_wheel, add="+")

    def _inside(self, e) -> bool:
        try:
            w = self.winfo_containing(e.x_root, e.y_root)
        except (KeyError, tk.TclError):
            return False
        while w is not None:
            if w is self:
                return True
            w = w.master
        return False

    def _on_wheel(self, e):
        try:
            if not self._inside(e) or self.body.winfo_reqheight() <= self.canvas.winfo_height():
                return None
            if e.num == 4:
                units = -3
            elif e.num == 5:
                units = 3
            else:
                d = e.delta
                units = -3 * (d // 120 if abs(d) >= 120 else (1 if d > 0 else -1 if d < 0 else 0))
            if units:
                self.canvas.yview_scroll(units, "units")
        except tk.TclError:
            pass
        return None

    def guard_wheel(self, widget) -> None:
        """Stop a combobox from changing its value when the wheel is used to scroll the page."""
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(seq, lambda e: (self._on_wheel(e), "break")[1])

    def scroll_to(self, widget) -> None:
        self.update_idletasks()
        total = max(1, self.body.winfo_reqheight())
        self.canvas.yview_moveto(max(0.0, (widget.winfo_y() - 6) / total))
