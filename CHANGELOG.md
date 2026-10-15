# Changelog

## 2.2.0
* **GUI:** the Setup tab is now organised in cards (Goal, Brain, Model, From scratch, Budgets, Safety, System), scrolls with the mouse wheel (Windows, macOS, Linux), has quick-jump buttons, and keeps the START / CREATE FROM SCRATCH buttons in an always-visible bar. Calmer theme for all tabs.
* The layout lives in `sira/setup_layout.py` (tested: every GUI field must be placed exactly once).

## 2.1.1
* **Fix:** bwrap sandbox failed with `Creating new namespace failed: Resource temporarily unavailable` on busy desktops. The process limit counted processes but the kernel counts threads, so the limit was below the user's real task count. It now counts threads and keeps a larger margin.
* `--doctor` now prints the first error message of a failed sandbox self-test.

## 2.1.0
* **Security:** `worldlab.math_verify` no longer calls `eval()`/passes raw text to `sympy.sympify`; expressions go through a whitelist AST evaluator (`safe_eval`/`check_expr`) with size, node, exponent and factorial limits.
* **New `--doctor`:** one command checks install integrity, Python, tkinter, training stack, sandbox backend + isolation self-test, GPU, disk, config and brain connectivity (`--fast`, `--json`).
* **New `--check-config`:** catches typos (with suggestions), wrong types, out-of-range values, missing API keys, inconsistent mode settings.
* **New `--report`, `--list-runs`, `--serve`:** HTML/Markdown/JSON run reports with score chart, and a read-only local dashboard (loopback by default, token mandatory otherwise, strict CSP).
* **Repo:** `tests/__init__.py` (discover works), CI workflow (3 Python versions, manifest freshness), LICENSE, `.gitignore`, Makefile, manifest tool reads the version.
* Docs: fixed duplicated README heading, documented the new tooling.

## 2.0.0
Initial public release.
