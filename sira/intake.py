"""Model intake: analyse a Hugging Face (or local) causal LM and choose a safe training recipe.

Never executes code from a model repository (trust_remote_code is refused).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

OK_LICENSES = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "cc-by-4.0", "openrail", "openrail++", "unlicense"}
CAUTION_LICENSES_HINT = ("llama", "gemma", "other", "unknown", "nc", "research", "gpl", "agpl")


class IntakeError(RuntimeError):
    pass


def _http_json(url: str, token: str | None = None, timeout: int = 30):
    headers = {"User-Agent": "sira-intake/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise IntakeError(f"HTTP {e.code} for {url}")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise IntakeError(f"cannot fetch {url}: {e}")


def estimate_params(c: dict) -> int | None:
    """Rough parameter count of a decoder-only transformer from its config."""
    try:
        h, L, v = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"]
        i = c.get("intermediate_size", 4 * h)
        heads = c.get("num_attention_heads", 1)
        kv = c.get("num_key_value_heads", heads)
        hd = c.get("head_dim", h // heads)
        attn = h * heads * hd + 2 * h * kv * hd + heads * hd * h
        experts = c.get("num_local_experts") or c.get("num_experts") or c.get("n_routed_experts") or 1
        mlp = 3 * h * i * experts
        emb = v * h * (1 if c.get("tie_word_embeddings") else 2)
        return int(L * (attn + mlp) + emb)
    except (KeyError, TypeError, ZeroDivisionError):
        return None


def detect_hardware() -> dict:
    """Hardware via nvidia-smi only - the main process never imports torch (slow, memory hungry)."""
    hw = {"gpus": [], "cuda": False, "ram_gb": None, "cpu_count": os.cpu_count()}
    try:
        hw["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=15).stdout
            for line in out.strip().splitlines():
                name, mem, cc = [x.strip() for x in line.rsplit(",", 2)]
                hw["gpus"].append({"name": name, "vram_gb": round(float(mem) * 1.048576 / 1000, 1),
                                   "bf16": float(cc) >= 8.0})
            hw["cuda"] = bool(hw["gpus"])
        except Exception:
            pass
    return hw


def analyze_model(ref: str, token: str | None = None) -> dict:
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    rep = {"ref": ref, "source": None, "license": "unknown", "warnings": [], "supported": True, "errors": []}
    cfg, tok_cfg, siblings, total_params, gated, local_params = {}, {}, [], None, False, None
    if os.path.isdir(ref):
        rep["source"] = "local"
        p = Path(ref)
        if not (p / "config.json").exists():
            raise IntakeError(f"{ref} has no config.json")
        cfg = json.loads((p / "config.json").read_text())
        if (p / "tokenizer_config.json").exists():
            tok_cfg = json.loads((p / "tokenizer_config.json").read_text())
        siblings = [f.name for f in p.iterdir() if f.is_file()]
        try:
            from .brainmap import read_headers
            local_params = sum(int(__import__("math").prod(v["shape"])) for v in read_headers(ref).values()) or None
        except Exception:
            local_params = None
        rep["size_on_disk_gb"] = round(sum(f.stat().st_size for f in p.iterdir() if f.is_file()) / 1e9, 2)
        rep["license"] = "unknown (local directory)"
    else:
        rep["source"] = "huggingface"
        if "/" not in ref and not ref.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise IntakeError("model reference looks invalid")
        info = _http_json(f"https://huggingface.co/api/models/{ref}", token)
        gated = bool(info.get("gated"))
        siblings = [s["rfilename"] for s in info.get("siblings", [])]
        tags = info.get("tags", [])
        lic = (info.get("cardData") or {}).get("license") or next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), "unknown")
        rep["license"] = str(lic)
        st = info.get("safetensors") or {}
        total_params = st.get("total")
        rep["pipeline_tag"], rep["library"], rep["downloads"] = info.get("pipeline_tag"), info.get("library_name"), info.get("downloads")
        try:
            cfg = _http_json(f"https://huggingface.co/{ref}/resolve/main/config.json", token)
        except IntakeError as e:
            raise IntakeError(f"{e} (gated/private model? set HF_TOKEN and accept the terms on the model page)")
        try:
            tok_cfg = _http_json(f"https://huggingface.co/{ref}/resolve/main/tokenizer_config.json", token)
        except IntakeError:
            tok_cfg = {}
    arch = (cfg.get("architectures") or ["?"])[0]
    rep.update({
        "architecture": arch, "model_type": cfg.get("model_type"), "hidden_size": cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model"),
        "layers": cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers"), "heads": cfg.get("num_attention_heads") or cfg.get("n_head"),
        "kv_heads": cfg.get("num_key_value_heads"), "vocab": cfg.get("vocab_size"),
        "max_context": cfg.get("max_position_embeddings"), "torch_dtype": cfg.get("torch_dtype"),
        "has_chat_template": bool(tok_cfg.get("chat_template")) or any("chat_template" in s for s in siblings),
        "files": len(siblings), "gated": gated,
        "is_moe": bool(cfg.get("num_local_experts") or cfg.get("num_experts") or cfg.get("n_routed_experts")),
        "quantized_already": bool(cfg.get("quantization_config")),
    })
    params = total_params or local_params or estimate_params(cfg)
    rep["params"] = params
    rep["params_b"] = round(params / 1e9, 3) if params else None
    # ---- support checks ------------------------------------------------------------------------------
    if not (arch.endswith("ForCausalLM") or arch.endswith("LMHeadModel")):
        rep["supported"] = False
        rep["errors"].append(f"architecture {arch} is not a decoder-only causal LM; this pipeline cannot fine-tune it")
    if cfg.get("auto_map"):
        rep["supported"] = False
        rep["errors"].append("model requires trust_remote_code (custom code from the repo). Refused for safety: "
                             "SIRA never executes code shipped inside model repositories.")
    if rep["quantized_already"]:
        rep["warnings"].append("model is already quantized (GPTQ/AWQ/bnb); LoRA on quantized weights may not merge cleanly")
    if rep["is_moe"]:
        rep["warnings"].append("Mixture-of-Experts model: LoRA applies to linear layers only; expect smaller gains")
    if not rep["has_chat_template"]:
        rep["warnings"].append("no chat template found; a plain Q/A prompt format will be used")
    lic = rep["license"].lower()
    if lic in OK_LICENSES:
        rep["license_note"] = "permissive license"
    else:
        rep["license_note"] = "CHECK THE LICENSE before publishing or commercial use of derived weights"
        rep["warnings"].append(f"license '{rep['license']}' may restrict redistribution/commercial use of fine-tuned weights")
    if gated:
        rep["warnings"].append("gated model: HF_TOKEN required for download")
    return rep


def choose_recipe(rep: dict, hw: dict, cfg, prefer_full: bool = False) -> dict:
    """Pick LoRA / QLoRA / CPU-LoRA from model size and available memory (single-GPU)."""
    P = (rep.get("params") or 0) / 1e9
    if not rep.get("supported", True):
        return {"method": "none", "reason": "; ".join(rep["errors"])}
    if not P:
        return {"method": "none", "reason": "cannot determine parameter count"}
    if prefer_full or getattr(cfg, "training_method", "auto") == "full":       # models born inside SIRA (from scratch) or an explicit choice: train ALL weights
        need_full = 16.0 * P + 2.0
        if hw.get("cuda") and hw["gpus"] and need_full <= hw["gpus"][0]["vram_gb"] * 0.92:
            g = hw["gpus"][0]
            return {"method": "full", "dtype": "bfloat16" if g.get("bf16") else "float16", "device": "cuda", "grad_checkpointing": False,
                    "est_vram_gb": round(need_full, 1), "gpu": g["name"], "reason": f"{P:.3f}B params: full fine-tuning (all weights) on {g['name']}"}
        if (not hw.get("cuda")) and P <= 0.5 and (hw.get("ram_gb") or 8) >= 16 * P + 2:
            return {"method": "full", "dtype": "float32", "device": "cpu", "grad_checkpointing": False, "est_vram_gb": 0,
                    "reason": f"no GPU; {P:.3f}B model trained on CPU (all weights, slow)"}
    if getattr(cfg, "training_method", "auto") == "full":
        prefer_full = True
    act = 3.0   # GB for activations / optimizer states of the adapters (conservative)
    need_lora, need_qlora = 2.0 * P * 1.15 + act, 0.62 * P + act
    if hw.get("cuda") and hw["gpus"]:
        g = hw["gpus"][0]
        vram = g["vram_gb"]
        dtype = "bfloat16" if g.get("bf16") else "float16"
        if need_lora <= vram * 0.92:
            m, need = "lora", need_lora
        elif need_qlora <= vram * 0.92:
            m, need = "qlora", need_qlora
        else:
            return {"method": "none", "reason": f"{P:.1f}B params need >= {need_qlora:.0f} GB (QLoRA); GPU has {vram} GB. "
                                                 f"Choose a smaller model or a bigger GPU."}
        return {"method": m, "dtype": dtype, "device": "cuda", "grad_checkpointing": vram < 40,
                "est_vram_gb": round(need, 1), "gpu": g["name"],
                "reason": f"{P:.2f}B params, {vram} GB VRAM -> {m.upper()} (~{need:.0f} GB)"}
    if cfg.allow_cpu_training and P <= 0.8 and (hw.get("ram_gb") or 0) >= 4 * P + 4:
        return {"method": "lora", "dtype": "float32", "device": "cpu", "grad_checkpointing": False,
                "est_vram_gb": 0, "reason": f"no GPU; {P:.2f}B model small enough for CPU LoRA (slow)"}
    return {"method": "none", "reason": "no CUDA GPU detected (set allow_cpu_training for models <= 0.8B)"}
