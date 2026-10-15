import ast
import unittest
from pathlib import Path

from sira.setup_layout import SETUP_GROUPS, layout

GUI = Path(__file__).resolve().parent.parent / "sira" / "gui.py"


def gui_spec_keys():
    tree = ast.parse(GUI.read_text(encoding="utf-8"))
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "_tab_setup":
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "spec" for t in node.targets) and isinstance(node.value, ast.List):
                    return [e.elts[0].value for e in node.value.elts if isinstance(e, ast.Tuple)]
    raise AssertionError("spec not found in gui.py")


class SetupLayout(unittest.TestCase):
    def test_every_gui_field_is_placed_exactly_once(self):
        keys = gui_spec_keys()
        placed = [k for _, _, _, ks in layout(keys) for k in ks]
        self.assertEqual(sorted(placed), sorted(keys))
        self.assertEqual(len(placed), len(set(placed)))

    def test_no_typos_in_groups(self):
        keys = set(gui_spec_keys())
        for title, _, _, ks in SETUP_GROUPS:
            self.assertFalse(set(ks) - keys, f"unknown keys in group {title}: {set(ks) - keys}")
        self.assertEqual(len([k for g in SETUP_GROUPS for k in g[3]]), len({k for g in SETUP_GROUPS for k in g[3]}))

    def test_scratch_group_is_present(self):
        self.assertTrue(any(t == "Create a model from scratch" and "scratch_arch" in ks for t, _, _, ks in layout(gui_spec_keys())))

    def test_unplaced_keys_go_to_more_settings(self):
        self.assertEqual(layout(["mode", "brand_new_key"])[-1][3], ["brand_new_key"])


class ScrollFrameTk(unittest.TestCase):
    def test_scrolls(self):
        try:
            import tkinter as tk
            root = tk.Tk()
        except Exception:
            self.skipTest("no tkinter or no display")
        try:
            from tkinter import ttk
            from sira.ui_kit import ScrollFrame
            root.geometry("400x200")
            sf = ScrollFrame(root)
            sf.pack(fill="both", expand=True)
            for i in range(60):
                ttk.Label(sf.body, text=f"row {i}").pack()
            root.update()
            self.assertGreater(sf.body.winfo_reqheight(), sf.canvas.winfo_height())
            sf.canvas.yview_scroll(5, "units")
            root.update()
            self.assertGreater(sf.canvas.yview()[0], 0.0)
            sf.scroll_to(sf.body.winfo_children()[30])
            root.update()
            self.assertGreater(sf.canvas.yview()[0], 0.2)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
