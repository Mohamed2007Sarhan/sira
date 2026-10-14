"""Brain map: the architecture of the target model as a navigable structure (layers, components, tensors),
plus conversion of a human/agent selection ("layer 3 of the donor -> layer 5 of mine") into transplant operations.
Reads safetensors headers directly - no torch needed for the structure."""
from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path

LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks|block|layer)\.(\d+)\.(.+)$")
# exact leaf-module names -> canonical component (families: Llama/Mistral/Qwen/Gemma/OLMo/StableLM/StarCoder2, GPT-2/J/NeoX/OPT/Bloom/Falcon/Phi)
LEAF = {"q_proj": "attn.q", "k_proj": "attn.k", "v_proj": "attn.v", "o_proj": "attn.o", "out_proj": "attn.o", "dense": "attn.o",
        "qkv_proj": "attn.qkv", "c_attn": "attn.qkv", "query_key_value": "attn.qkv", "Wqkv": "attn.qkv",
        "gate_proj": "mlp.gate", "up_proj": "mlp.up", "down_proj": "mlp.down", "gate_up_proj": "mlp.gate_up",
        "c_fc": "mlp.up", "fc1": "mlp.up", "fc_in": "mlp.up", "dense_h_to_4h": "mlp.up", "fc_out": "mlp.down", "fc2": "mlp.down", "dense_4h_to_h": "mlp.down",
        "input_layernorm": "norm.in", "post_attention_layernorm": "norm.post", "ln_1": "norm.in", "ln_2": "norm.post", "ln_attn": "norm.in", "ln_mlp": "norm.post",
        "self_attn_layer_norm": "norm.in", "final_layer_norm": "norm.post"}
SPECIAL_LEAF = {"embed_tokens": "embed", "wte": "embed", "embed_in": "embed", "word_embeddings": "embed", "wpe": "pos_embed", "embed_positions": "pos_embed",
                "lm_head": "lm_head", "embed_out": "lm_head", "ln_f": "final_norm", "norm": "final_norm", "final_layer_norm": "final_norm",
                "word_embeddings_layernorm": "embed_norm"}
COMPONENT_ORDER = ["attn.q", "attn.k", "attn.v", "attn.qkv", "attn.o", "mlp.gate", "mlp.up", "mlp.gate_up", "mlp.down", "norm.in", "norm.post"]


def _component(rest: str) -> str:
    """rest = name below the layer index, e.g. 'self_attn.q_proj.weight' or 'attn.c_attn.bias'."""
    parts = [x for x in rest.split(".") if x not in ("weight", "bias")]
    if parts and parts[-1] == "c_proj":                                       # GPT-2 / StarCoder2 share the leaf name between attention and MLP
        return "attn.o" if any(x in ("attn", "attention", "self_attn") for x in parts[:-1]) else "mlp.down"
    for leaf in reversed(parts):
        if leaf in LEAF:
            if leaf == "dense" and not any(p in ("attention", "self_attention", "attn") for p in parts[:-1]):
                continue                                                   # a bare 'dense' outside attention is not the attention output
            return LEAF[leaf]
    return ".".join(parts[-2:]) if len(parts) > 1 else (parts[0] if parts else rest)


def read_headers(model_dir: str) -> dict:
    out = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                out[k] = {"shape": v["shape"], "dtype": v["dtype"], "file": f.name}
    return out


def synth_from_config(cfg: dict) -> dict:
    h, L, V = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    I = cfg.get("intermediate_size", 4 * h)
    heads = cfg.get("num_attention_heads", 1)
    kv = cfg.get("num_key_value_heads", heads)
    hd = cfg.get("head_dim", h // heads)
    t = {"model.embed_tokens.weight": {"shape": [V, h]}, "model.norm.weight": {"shape": [h]}}
    if not cfg.get("tie_word_embeddings"):
        t["lm_head.weight"] = {"shape": [V, h]}
    f = max(1, int(cfg.get("cla_factor", 1)))
    for i in range(L):
        p = f"model.layers.{i}."
        t[p + "self_attn.q_proj.weight"] = {"shape": [heads * hd, h]}
        if i % f == 0:                                          # Cross-Layer Attention: only source layers own K/V
            t[p + "self_attn.k_proj.weight"] = {"shape": [kv * hd, h]}
            t[p + "self_attn.v_proj.weight"] = {"shape": [kv * hd, h]}
        t[p + "self_attn.o_proj.weight"] = {"shape": [h, heads * hd]}
        t[p + "mlp.gate_proj.weight"] = {"shape": [I, h]}
        t[p + "mlp.up_proj.weight"] = {"shape": [I, h]}
        t[p + "mlp.down_proj.weight"] = {"shape": [h, I]}
        t[p + "input_layernorm.weight"] = {"shape": [h]}
        t[p + "post_attention_layernorm.weight"] = {"shape": [h]}
    return t


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def read_structure(model_ref: str) -> dict:
    """Structure of a local model dir (real tensors) or, for a not-yet-downloaded model, synthesised from its config."""
    source = "headers"
    cfg = {}
    if os.path.isdir(model_ref):
        cp = Path(model_ref) / "config.json"
        cfg = json.loads(cp.read_text()) if cp.exists() else {}
        tensors = read_headers(model_ref)
        if not tensors:
            if not cfg:
                raise ValueError("no safetensors and no config.json")
            tensors, source = synth_from_config(cfg), "config"
    else:
        raise ValueError("pass a local directory (the trainer downloads Hugging Face models first)")
    layers: dict[int, dict] = {}
    special: dict[str, dict] = {}
    other = {}
    for name, info in tensors.items():
        entry = {"name": name, "shape": info["shape"], "numel": _numel(info["shape"])}
        m = LAYER_RE.search(name)
        if m:
            idx, rest = int(m.group(1)), m.group(2)
            comp = _component(rest)
            cur = layers.setdefault(idx, {})
            if comp in cur and not name.endswith(".weight"):                       # bias: count its numbers, keep the weight as the node
                cur[comp]["numel"] += entry["numel"]
            elif comp in cur and cur[comp]["name"].endswith(".bias"):
                entry["numel"] += cur[comp]["numel"]
                cur[comp] = entry
            else:
                cur[comp] = entry
            continue
        leaf = [x for x in name.split(".") if x not in ("weight", "bias")]
        key = SPECIAL_LEAF.get(leaf[-1]) if leaf else None
        if key and key in special and not name.endswith(".weight"):
            special[key]["numel"] += entry["numel"]
        else:
            (special if key else other).__setitem__(key or name, entry)
    ordered = [{"index": i, "components": {c: layers[i][c] for c in sorted(layers[i], key=lambda c: (COMPONENT_ORDER.index(c) if c in COMPONENT_ORDER else 99, c))}}
               for i in sorted(layers)]
    total = sum(e["numel"] for e in special.values()) + sum(e["numel"] for e in other.values()) + \
        sum(c["numel"] for L in ordered for c in L["components"].values())
    return {"source": source, "model": model_ref, "config": {k: cfg.get(k) for k in (
        "architectures", "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "intermediate_size", "vocab_size", "max_position_embeddings")}, "layers": ordered, "special": special, "other": other,
        "total_params": total}


def attach_stats(structure: dict, scan: dict) -> dict:
    """Merge per-tensor statistics (from the worker 'scan' command) into the structure; adds layer-level summaries."""
    def put(e):
        s = scan.get(e["name"])
        if s:
            e["stats"] = {k: s[k] for k in ("mean", "std", "absmax", "norm", "frac_small") if k in s}
            if "delta_rel" in s:
                e["stats"]["delta_rel"] = s["delta_rel"]
    for e in list(structure["special"].values()) + list(structure["other"].values()):
        put(e)
    for L in structure["layers"]:
        for e in L["components"].values():
            put(e)
        norms = [e["stats"]["norm"] for e in L["components"].values() if "stats" in e]
        deltas = [e["stats"]["delta_rel"] for e in L["components"].values() if "stats" in e and "delta_rel" in e["stats"]]
        L["summary"] = {"norm": sum(norms) / len(norms) if norms else None, "delta_rel": sum(deltas) / len(deltas) if deltas else None}
    return structure


def build_ops(target: dict, donor: dict, selection: list) -> list:
    """selection items:
         {"target_layer": 5, "donor_layer": 3, "components": "all" | ["mlp.down", ...], "mode": "copy|blend|resize", "alpha": 0.5}
         {"special": "embed|lm_head|final_norm", "mode":..., "alpha":...}
       Returns worker transplant ops. Raises ValueError for anything that doesn't exist."""
    ops = []
    tl = {L["index"]: L["components"] for L in target["layers"]}
    dl = {L["index"]: L["components"] for L in donor["layers"]}
    for s in selection:
        mode, alpha = s.get("mode", "blend"), float(s.get("alpha", 0.5))
        if mode not in ("copy", "blend", "resize") or not 0 <= alpha <= 1:
            raise ValueError("mode must be copy|blend|resize and alpha in [0,1]")
        if "special" in s:
            t, d = target["special"].get(s["special"]), donor["special"].get(s["special"])
            if not t or not d:
                raise ValueError(f"special tensor {s['special']} missing in target or donor")
            ops.append({"target": t["name"], "donor": d["name"], "mode": mode, "alpha": alpha})
            continue
        ti, di = int(s["target_layer"]), int(s["donor_layer"])
        if ti not in tl or di not in dl:
            raise ValueError(f"layer {ti} (target) or {di} (donor) does not exist")
        comps = list(tl[ti]) if s.get("components", "all") == "all" else list(s["components"])
        for c in comps:
            if c not in tl[ti] or c not in dl[di]:
                if s.get("components", "all") == "all":
                    continue
                raise ValueError(f"component {c} missing in target layer {ti} or donor layer {di}")
            same = tl[ti][c]["shape"] == dl[di][c]["shape"]
            if not same and mode != "resize":
                raise ValueError(f"{c}: shape {tl[ti][c]['shape']} (target) vs {dl[di][c]['shape']} (donor) - use mode=resize")
            ops.append({"target": tl[ti][c]["name"], "donor": dl[di][c]["name"], "mode": mode, "alpha": alpha})
    if not ops:
        raise ValueError("selection produced no operations")
    return ops
