"""Layout of the GUI Setup tab as plain data (no tkinter import, so it can be unit-tested)."""
from __future__ import annotations

# (card title, short label for the quick-jump chip, one-line hint, [config keys in display order])
SETUP_GROUPS = [
    ("Mode & goal", "Goal", "What should SIRA work on, and when is it done?",
     ["mode", "quest_goal", "quest_seeds", "goal_text", "target_score", "confirm_rounds", "society_enabled"]),
    ("Brain: the LLM that drives the agents", "Brain", "Pick a provider (Ollama is local and free; 'mock' runs offline with no intelligence).",
     ["llm_provider", "llm_model", "llm_base_url", "llm_api_key_env", "llm_num_ctx", "llm_json_mode", "thinking", "examiner_provider", "examiner_model"]),
    ("Model to improve", "Model", "Used by START RUN. Ignored when you create a model from scratch.",
     ["base_model", "algorithm", "lora_variant", "lora_plus_ratio", "optimizer", "accept_p", "train_problems", "eval_problems",
      "parallel_experiments", "allow_cpu_training"]),
    ("Create a model from scratch", "From scratch",
     "Starts from nothing: learns a language from raw text, gets a first lesson, then continues as usual. Use the CREATE button below.",
     ["scratch_language", "scratch_size", "scratch_arch", "arch_kv_heads", "arch_cla_factor", "scratch_corpus_dir", "scratch_wiki_mb", "scratch_steps"]),
    ("Budgets", "Budgets", "The run ends when any budget is used up (or when you press STOP).",
     ["budget_usd", "budget_gpu_hours", "budget_wall_hours"]),
    ("Safety & sandbox", "Safety", "Experiments run in an isolated sandbox. Keep the weak sandbox off unless you know why.",
     ["sandbox_backend", "sandbox_network", "allow_weak_sandbox", "require_approval"]),
    ("System", "System", "", ["auto_install", "auto_adapt"]),
]


def layout(spec_keys):
    """Return SETUP_GROUPS restricted to keys that exist, plus a 'More settings' card for any key not placed in a group."""
    have = list(spec_keys)
    out, placed = [], set()
    for title, short, hint, keys in SETUP_GROUPS:
        ks = [k for k in keys if k in have]
        placed.update(ks)
        if ks:
            out.append((title, short, hint, ks))
    rest = [k for k in have if k not in placed]
    if rest:
        out.append(("More settings", "More", "", rest))
    return out
