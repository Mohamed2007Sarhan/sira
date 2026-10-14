"""Catalog of learning / adaptation / merging algorithms SIRA knows, what each needs, and whether it is implemented.
The Environment Bootstrapper uses it to install exactly what the chosen algorithms require."""
from __future__ import annotations

# pip name -> (purpose, platforms where it makes sense)
PACKAGES = {
    "torch": ("tensor engine (training, generation, merging)", "any"),
    "transformers": ("model definitions, tokenizers, generation", "any"),
    "peft": ("LoRA / rsLoRA / DoRA adapters", "any"),
    "accelerate": ("device placement, low-memory loading", "any"),
    "safetensors": ("safe weight files (merge, transplant, brain map)", "any"),
    "tokenizers": ("fast BPE tokenizers (from-scratch models)", "any"),
    "huggingface_hub": ("download models from Hugging Face", "any"),
    "sentencepiece": ("tokenizer backend of many Llama/Mistral-family models", "any"),
    "numpy": ("numerics", "any"),
    "bitsandbytes": ("4-bit NF4 (QLoRA) and 8-bit optimizers", "linux+cuda"),
}
SANDBOX_LIBS = ["numpy", "scipy", "sympy"]          # for experiments (installed into the agents' /station/pylib if missing)
CORE_TRAINING = ["torch", "transformers", "peft", "accelerate", "safetensors", "tokenizers", "huggingface_hub", "numpy"]

# name: dict(kind, implemented, needs(pip), hardware, summary, equation)
ALGORITHMS = {
    "sft_full":  dict(kind="adaptation", needs=["torch", "transformers"], hw="any", eq="min_θ  -Σ log p_θ(y|x)",
                      summary="Full-weight supervised fine-tuning (used for models born inside SIRA)."),
    "lora":      dict(kind="adaptation", needs=["peft"], hw="any", eq="W' = W + (α/r)·B·A,  B∈R^{d×r}, A∈R^{r×k}",
                      summary="Low-rank adapters; trains r(d+k) numbers per matrix."),
    "rslora":    dict(kind="adaptation", needs=["peft"], hw="any", eq="W' = W + (α/√r)·B·A",
                      summary="Rank-stabilised scaling so larger ranks keep learning."),
    "dora":      dict(kind="adaptation", needs=["peft"], hw="any", eq="W' = m · (W + BA)/‖W + BA‖_c",
                      summary="Weight-decomposed LoRA: separate magnitude m and direction."),
    "lora_plus": dict(kind="adaptation", needs=["peft"], hw="any", eq="lr_B = λ·lr_A  (λ≈16)",
                      summary="Different learning rates for A and B (faster feature learning)."),
    "qlora":     dict(kind="adaptation", needs=["peft", "bitsandbytes"], hw="cuda", eq="W_nf4 (4-bit) frozen, LoRA in bf16",
                      summary="LoRA on a 4-bit quantised base: ~0.6 byte/param instead of 2."),
    "adamw8bit": dict(kind="optimizer", needs=["bitsandbytes"], hw="cuda", eq="Adam moments stored in 8 bit",
                      summary="Cuts optimizer memory from 8 to 2 bytes/param."),
    "rft":       dict(kind="self-improvement", needs=[], hw="any", eq="θ ← argmax Σ_{R(y)=1} log p_θ(y|x)",
                      summary="Rejection-sampling fine-tuning: learn only from verified-correct own samples."),
    "grpo":      dict(kind="self-improvement", needs=[], hw="any",
                      eq="A_i=(r_i-mean(r))/std(r);  L=-E[min(ρA, clip(ρ,1±ε)A)],  ρ=π_θ/π_old",
                      summary="Group-relative policy optimisation with verifiable rewards (no value network)."),
    "dpo":       dict(kind="self-improvement", needs=[], hw="any",
                      eq="L=-log σ(β[(log π(y_w)-log π(y_l)) - (log π_ref(y_w)-log π_ref(y_l))])",
                      summary="Preference optimisation from verified-correct vs wrong samples."),
    "soup":      dict(kind="merge", needs=["safetensors"], hw="any", eq="θ = (1-λ)θ_parent + λθ_child",
                      summary="Weight interpolation (model soup / WiSE-FT)."),
    "slerp":     dict(kind="merge", needs=["safetensors"], hw="any", eq="slerp(θ1,θ2;t)=sin((1-t)Ω)/sinΩ·θ1+sin(tΩ)/sinΩ·θ2",
                      summary="Spherical interpolation."),
    "task":      dict(kind="merge", needs=["safetensors"], hw="any", eq="θ = θ_base + Σ λ_i(θ_i - θ_base)",
                      summary="Task arithmetic."),
    "ties":      dict(kind="merge", needs=["safetensors"], hw="any", eq="trim → elect sign → disjoint mean of task vectors",
                      summary="TIES merging."),
    "dare":      dict(kind="merge", needs=["safetensors"], hw="any", eq="τ' = (m⊙τ)/(1-p), m~Bernoulli(1-p)",
                      summary="Drop-and-rescale task vectors before adding."),
    "transplant": dict(kind="surgery", needs=["safetensors"], hw="any", eq="W_target[l] ← (1-α)W_target[l] + α·R(W_donor[l'])",
                       summary="Copy/blend/resize chosen tensors from a donor model."),
}


def algorithm_status(installed: dict, has_cuda: bool) -> list:
    out = []
    for name, a in ALGORITHMS.items():
        missing = [p for p in a["needs"] if not installed.get(p)]
        usable = not missing and (a["hw"] == "any" or has_cuda)
        out.append({"name": name, "kind": a["kind"], "usable": usable, "missing": missing, "hw": a["hw"],
                    "summary": a["summary"], "equation": a["eq"]})
    return out
