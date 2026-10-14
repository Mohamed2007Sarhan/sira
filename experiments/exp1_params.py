"""Exp 1 - verify the closed-form parameter / memory / FLOP equations against (a) real PyTorch models, (b) published sizes."""
import json, sys
sys.path.insert(0, "..")

def n_params(c):
    """Exact parameter count of a Llama-style decoder (RMSNorm, GQA, SwiGLU, optional tied embeddings)."""
    d, L, V, I, H = c["hidden"], c["layers"], c["vocab"], c["inter"], c["heads"]
    kv = c.get("kv", H); hd = c.get("head_dim", d // H)
    attn = d * H * hd + 2 * d * kv * hd + H * hd * d          # q, k, v, o
    mlp = 3 * d * I                                           # gate, up, down
    norms = 2 * d                                             # two RMSNorms per layer
    emb = V * d * (1 if c.get("tie", False) else 2)           # embedding (+ lm_head if untied)
    return L * (attn + mlp + norms) + emb + d                 # + final norm
def n_nonemb(c):
    return n_params(c) - c["vocab"] * c["hidden"] * (1 if c.get("tie", False) else 2)
def lora_params(c, r):
    d, L, I, H = c["hidden"], c["layers"], c["inter"], c["heads"]; kv = c.get("kv", H); hd = c.get("head_dim", d // H)
    per_layer = r * ((d + H * hd) + 2 * (d + kv * hd) + (H * hd + d) + 2 * (d + I) + (I + d))
    return L * per_layer
def kv_bytes_per_token(c, bytes_=2, cla=1):
    kv = c.get("kv", c["heads"]); hd = c.get("head_dim", c["hidden"] // c["heads"])
    return 2 * c["layers"] / cla * kv * hd * bytes_

out = {"verify_vs_torch": [], "real_models": []}
import torch
from transformers import LlamaConfig, LlamaForCausalLM
for c in [dict(hidden=64, layers=2, vocab=400, inter=128, heads=4, kv=2), dict(hidden=96, layers=3, vocab=1000, inter=256, heads=4, kv=4, tie=True),
          dict(hidden=128, layers=4, vocab=2048, inter=344, heads=8, kv=2), dict(hidden=256, layers=6, vocab=4096, inter=688, heads=8, kv=8, tie=True),
          dict(hidden=48, layers=5, vocab=300, inter=100, heads=6, kv=3)]:
    m = LlamaForCausalLM(LlamaConfig(vocab_size=c["vocab"], hidden_size=c["hidden"], intermediate_size=c["inter"], num_hidden_layers=c["layers"],
                                     num_attention_heads=c["heads"], num_key_value_heads=c["kv"], tie_word_embeddings=c.get("tie", False)))
    actual = sum(p.numel() for p in m.parameters())
    out["verify_vs_torch"].append({**c, "formula": n_params(c), "torch": actual, "exact": n_params(c) == actual})
REAL = {  # architecture hyper-parameters from the public model configs
 "Llama-2-7B": dict(hidden=4096, layers=32, vocab=32000, inter=11008, heads=32, kv=32),
 "Llama-2-13B": dict(hidden=5120, layers=40, vocab=32000, inter=13824, heads=40, kv=40),
 "Llama-2-70B": dict(hidden=8192, layers=80, vocab=32000, inter=28672, heads=64, kv=8),
 "Mistral-7B": dict(hidden=4096, layers=32, vocab=32000, inter=14336, heads=32, kv=8),
 "Llama-3-8B": dict(hidden=4096, layers=32, vocab=128256, inter=14336, heads=32, kv=8),
 "Llama-3-70B": dict(hidden=8192, layers=80, vocab=128256, inter=28672, heads=64, kv=8),
 "Qwen2.5-0.5B": dict(hidden=896, layers=24, vocab=151936, inter=4864, heads=14, kv=2, tie=True),
}
for name, c in REAL.items():
    N = n_params(c)
    row = {"model": name, "params_B": round(N / 1e9, 3), "non_embedding_B": round(n_nonemb(c) / 1e9, 3),
           "lora_r16_params_M": round(lora_params(c, 16) / 1e6, 1), "lora_r16_pct": round(100 * lora_params(c, 16) / N, 3),
           "weights_bf16_GB": round(2 * N / 1e9, 1), "weights_nf4_GB": round(0.5 * N / 1e9 * 1.06, 1),
           "full_ft_adamw_GB": round(16 * N / 1e9, 1),                       # bf16 w + bf16 g + fp32 master + 2 fp32 moments = 2+2+4+4+4
           "kv_cache_KB_per_token_bf16": round(kv_bytes_per_token(c) / 1e3, 1),
           "kv_cache_GB_32k_ctx": round(kv_bytes_per_token(c) * 32768 / 1e9, 2),
           "kv_cache_GB_32k_ctx_CLA2": round(kv_bytes_per_token(c, cla=2) * 32768 / 1e9, 2),
           "train_flops_per_token": 6 * N, "decode_flops_per_token": 2 * N}
    out["real_models"].append(row)
json.dump(out, open("results/exp1.json", "w"), indent=1)
print("formula vs torch (exact?):", [(r["formula"], r["torch"], r["exact"]) for r in out["verify_vs_torch"]])
for r in out["real_models"]:
    print(f"{r['model']:14s} N={r['params_B']:7.3f}B  nonemb={r['non_embedding_B']:7.3f}B  LoRA r16={r['lora_r16_params_M']:7.1f}M ({r['lora_r16_pct']:.3f}%)  bf16={r['weights_bf16_GB']:6.1f}GB nf4={r['weights_nf4_GB']:5.1f}GB fullFT={r['full_ft_adamw_GB']:7.1f}GB  KV/tok={r['kv_cache_KB_per_token_bf16']:6.1f}KB  KV@32k={r['kv_cache_GB_32k_ctx']:5.2f}GB (CLA-2 {r['kv_cache_GB_32k_ctx_CLA2']:.2f})")
