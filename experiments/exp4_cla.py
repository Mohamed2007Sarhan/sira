"""Exp 4 - Cross-Layer Attention on real English text.
 A) trained from scratch with CLA factor f in {1,2,3,6} (6 layers, d=128, 2 KV heads): quality vs parameters vs KV-cache bytes.
 B) convert the trained standard model to CLA-2/CLA-3 by weight surgery (mean over the group) and measure loss before / after a short healing run."""
import json, math, os, sys, time, torch
sys.path.insert(0, ".."); torch.set_num_threads(1)
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import LlamaConfig, LlamaForCausalLM
from sira import cla_model as C
C.register()
text = open("/tmp/corpus_en.txt").read()
tk = Tokenizer(models.BPE(unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
tk.train_from_iterator([text], trainers.BpeTrainer(vocab_size=2048, special_tokens=["<unk>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
ids = torch.tensor(tk.encode(text).ids); V = tk.get_vocab_size(); cut = int(0.9 * len(ids)); train, val = ids[:cut], ids[cut:]
SEQ, BS = 128, 16
BASE = dict(vocab_size=V, hidden_size=128, intermediate_size=344, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=SEQ, tie_word_embeddings=True)
def mk(f): return LlamaForCausalLM(LlamaConfig(**BASE)) if f == 1 else C.ClaForCausalLM(C.make_config(BASE, cla_factor=f))
def batch(data, g):
    st = torch.randint(0, len(data) - SEQ - 1, (BS,), generator=g); return torch.stack([data[i:i + SEQ] for i in st])
def fit(m, steps, lr, seed=1, warm=20):
    g = torch.Generator().manual_seed(seed); opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    for s in range(steps):
        f = (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
        for gr in opt.param_groups: gr["lr"] = lr * f
        x = batch(train, g); m(input_ids=x, labels=x).loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
@torch.no_grad()
def val_loss(m):
    m.eval(); g = torch.Generator().manual_seed(7); v = sum(m(input_ids=(x := batch(val, g)), labels=x).loss.item() for _ in range(40)) / 40; m.train(); return v
res = json.load(open("results/exp4.json")) if os.path.exists("results/exp4.json") else {"A": [], "B": []}
mode = sys.argv[1]
if mode == "A":
    for f in [int(x) for x in sys.argv[2].split(",")]:
        SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0; torch.manual_seed(SEED); m = mk(f); N = sum(p.numel() for p in m.parameters()); t0 = time.time(); fit(m, 250, float(os.environ.get("LR", 3e-3)), seed=1 + SEED); secs = time.time() - t0
        if f == 1 and SEED == 0: torch.save(m.state_dict(), "/tmp/std6.pt")
        row = {"lr": float(os.environ.get("LR", 3e-3)), "seed": SEED, "cla_factor": f, "params": N, "val_loss": val_loss(m), "kv_bytes_per_token_bf16": C.kv_cache_bytes_per_token(6, 2, 32, f), "train_secs": secs}
        row["val_ppl"] = math.exp(row["val_loss"]); res["A"].append(row); print(row, flush=True)
else:
    sd = torch.load("/tmp/std6.pt"); std = mk(1); std.load_state_dict(sd); base_loss = val_loss(std); print("standard model val loss", base_loss, flush=True)
    for f in [int(x) for x in sys.argv[2].split(",")]:
        for init in ("mean", "first"):
            m = C.ClaForCausalLM(C.make_config(BASE, cla_factor=f)); new = {}
            for k, v in sd.items():
                import re
                mm = re.match(r"(model\.layers\.)(\d+)(\.self_attn\.)([kv])_proj\.weight", k)
                if not mm: new[k] = v; continue
                li = int(mm.group(2))
                if li % f: continue
                members = range(li, min(6, li + f)); new[k] = (sum(sd[f"{mm.group(1)}{i}{mm.group(3)}{mm.group(4)}_proj.weight"] for i in members) / len(members)) if init == "mean" else v
            missing = m.load_state_dict(new, strict=False); assert not missing.unexpected_keys
            before = val_loss(m); t0 = time.time(); fit(m, 80, 1e-3, seed=5, warm=5); after = val_loss(m)
            row = {"cla_factor": f, "init": init, "std_val_loss": base_loss, "converted_no_healing": before, "healed_80_steps": after, "heal_secs": time.time() - t0,
                   "params_removed_pct": 100 * (1 - sum(p.numel() for p in m.parameters()) / sum(p.numel() for p in std.parameters()))}
            res["B"].append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
json.dump(res, open("results/exp4.json", "w"), indent=1)
