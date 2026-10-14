"""Static config checker: catches typos, wrong types and out-of-range values before a long (expensive) run starts."""
from __future__ import annotations

import difflib
import json
import os
from dataclasses import fields
from pathlib import Path

from .config import Config

# name -> (min, max); None = unbounded
_RANGES = {
    "target_score": (0.0, 1.0), "accept_p": (0.0, 1.0), "examiner_weight": (0.0, 1.0), "quest_pass_score": (0.0, 1.0),
    "quest_skill_pass": (0.0, 1.0), "budget_usd": (0, None), "budget_gpu_hours": (0, None), "budget_wall_hours": (0, None),
    "price_in_per_mtok": (0, None), "price_out_per_mtok": (0, None), "llm_timeout_s": (1, None), "llm_num_ctx": (512, None),
    "llm_max_tokens": (1, None), "agent_max_steps": (1, None), "parallel_experiments": (1, None), "confirm_rounds": (1, None),
}


def _type_ok(val, default) -> bool:
    if isinstance(default, bool):
        return isinstance(val, bool)
    if isinstance(default, int):
        return isinstance(val, int) and not isinstance(val, bool)
    if isinstance(default, float):
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if isinstance(default, str):
        return isinstance(val, str)
    if isinstance(default, (list, tuple)):
        return isinstance(val, list)
    if isinstance(default, dict):
        return isinstance(val, dict)
    return True


def check_config(source) -> dict:
    """source: path to a JSON file or an already-loaded dict. Returns {ok, errors, warnings}."""
    errors, warnings = [], []
    if isinstance(source, dict):
        data = source
    else:
        try:
            data = json.loads(Path(source).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"ok": False, "errors": [f"file not found: {source}"], "warnings": []}
        except (OSError, ValueError) as e:
            return {"ok": False, "errors": [f"cannot read JSON: {e}"], "warnings": []}
    if not isinstance(data, dict):
        return {"ok": False, "errors": ["top level must be a JSON object"], "warnings": []}

    defaults = Config()
    names = [f.name for f in fields(Config)]
    known = {}
    for k, v in data.items():
        if k not in names:
            hint = difflib.get_close_matches(k, names, n=1)
            warnings.append(f"unknown key '{k}' is ignored" + (f" - did you mean '{hint[0]}'?" if hint else ""))
            continue
        if not _type_ok(v, getattr(defaults, k)):
            errors.append(f"'{k}': expected {type(getattr(defaults, k)).__name__}, got {type(v).__name__} ({v!r})")
            continue
        known[k] = v
        if k in _RANGES and isinstance(v, (int, float)):
            lo, hi = _RANGES[k]
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                errors.append(f"'{k}'={v} is outside the allowed range [{lo}, {hi if hi is not None else 'inf'}]")

    g = lambda k: known.get(k, getattr(defaults, k))
    from .llm import PROVIDERS, LOCAL_PROVIDERS, PRESETS
    prov = g("llm_provider")
    if prov not in PROVIDERS:
        errors.append(f"llm_provider '{prov}' is not one of: {', '.join(PROVIDERS)}")
    else:
        if prov == "openai_compatible" and not g("llm_base_url"):
            errors.append("llm_provider 'openai_compatible' needs llm_base_url")
        if prov != "mock" and prov not in LOCAL_PROVIDERS:
            env = g("llm_api_key_env") or {"anthropic": "ANTHROPIC_API_KEY"}.get(prov) or PRESETS.get(prov, ("", "OPENAI_API_KEY", False))[1]
            if not os.environ.get(env):
                warnings.append(f"environment variable {env} is not set (needed by provider '{prov}')")
        if not g("llm_model") and prov != "mock":
            errors.append("llm_model is empty")
    mode = g("mode")
    if mode in ("model", "both") and not g("base_model") and not g("scratch_mode"):
        warnings.append(f"mode '{mode}' without base_model and scratch_mode=false: there is no model to improve")
    if mode == "quest" and not str(g("quest_goal")).strip():
        errors.append("mode 'quest' needs quest_goal")
    if g("sandbox_network") == "open":
        warnings.append("sandbox_network='open': experiment code can reach the network")
    if g("sandbox_backend") == "subprocess" or g("allow_weak_sandbox"):
        warnings.append("weak sandbox accepted: experiment code is NOT strongly isolated")
    try:
        Config(**known).validate()
    except Exception as e:                      # Config.validate raises ValueError with a clear message
        errors.append(f"validate(): {e}")
    return {"ok": not errors, "errors": errors, "warnings": warnings}
