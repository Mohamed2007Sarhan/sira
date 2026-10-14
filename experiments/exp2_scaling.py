"""Exp 2 - pre-training scaling on REAL English text (Python reference documentation, ~0.7M characters).
Fixed data & steps, model width varied. Measures held-out loss, tokens/s, peak memory. CPU, 1 core."""
import json, math, os, resource, sys, time, torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import LlamaConfig, LlamaForCausalLM
torch.set_num_threads(1)
text = open("/tmp/corpus_en.txt").read()
tk = Tokenizer(models.BPE(unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
tk.train_from_iterator([text], trainers.BpeTrainer(vocab_size=2048, special_tokens=["<unk>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
ids = torch.tensor(tk.encode(text).ids); V = tk.get_vocab_size()
cut = int(0.9 * len(ids)); train, val = ids[:cut], ids[cut:]
print("tokens total", len(ids), "train", len(train), "val", len(val), "vocab", V, flush=True)
SEQ, BS, STEPS, LR = 128, 16, int(sys.argv[2]) if len(sys.argv) > 2 else 250, 3e-3
def batch(data, g):
    st = torch.randint(0, len(data) - SEQ - 1, (BS,), generator=g)
    return torch.stack([data[i:i + SEQ] for i in st])
def run(d, L=4):
    torch.manual_seed(0); g = torch.Generator().manual_seed(1)
    m = LlamaForCausalLM(LlamaConfig(vocab_size=V, hidden_size=d, intermediate_size=int(8 * d / 3) // 8 * 8, num_hidden_layers=L, num_attention_heads=max(1, d // 32),
                                     num_key_value_heads=max(1, d // 64), max_position_embeddings=SEQ, tie_word_embeddings=True))
    N = sum(p.numel() for p in m.parameters()); Nne = N - V * d
    opt = torch.optim.AdamW(m.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1); warm = 20
    t0 = time.time(); losses = []
    for s in range(STEPS):
        f = (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / (STEPS - warm)))
        for gr in opt.param_groups: gr["lr"] = LR * f
        x = batch(train, g); loss = m(input_ids=x, labels=x).loss; loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad(); losses.append(loss.item())
    secs = time.time() - t0
    m.eval(); vl = []
    with torch.no_grad():
        gv = torch.Generator().manual_seed(7)
        for _ in range(40): x = batch(val, gv); vl.append(m(input_ids=x, labels=x).loss.item())
    v = sum(vl) / len(vl)
    return {"d": d, "layers": L, "params": N, "non_embedding": Nne, "train_loss_last20": sum(losses[-20:]) / 20, "val_loss": v, "val_ppl": math.exp(v),
            "val_bits_per_token": v / math.log(2), "tokens_seen": STEPS * BS * SEQ, "tok_per_s": STEPS * BS * SEQ / secs, "secs": secs,
            "peak_rss_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "measured_flops_per_s": 6 * N * STEPS * BS * SEQ / secs}
res = []
for d in [int(x) for x in sys.argv[1].split(",")]:
    r = run(d); res.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
prev = json.load(open("results/exp2.json")) if os.path.exists("results/exp2.json") else []
json.dump(prev + res, open("results/exp2.json", "w"), indent=1)
