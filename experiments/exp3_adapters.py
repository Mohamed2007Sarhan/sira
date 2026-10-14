"""Exp 3 - adaptation methods on one pretrained model (d=128, 4 layers, ~1M params, English text) learning 2-digit addition.
Same data, same steps. Reports trainable parameters, held-out answer loss, exact-match accuracy, wall time."""
import json, math, os, random, sys, time, torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model
torch.set_num_threads(1)
text = open("/tmp/corpus_en.txt").read()
tk = Tokenizer(models.BPE(unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
tk.train_from_iterator([text, "".join(f"Q: {a}+{b}=\nA: {a+b}\n" for a in range(10, 100, 7) for b in range(10, 100, 3))],
                       trainers.BpeTrainer(vocab_size=2048, special_tokens=["<unk>", "<eos>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
V, EOS = tk.get_vocab_size(), tk.token_to_id("<eos>")
enc = lambda s: tk.encode(s).ids
PRE = "/tmp/pre128.pt"
cfg = LlamaConfig(vocab_size=V, hidden_size=128, intermediate_size=344, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=True)
if not os.path.exists(PRE):
    ids = torch.tensor(enc(text)); torch.manual_seed(0); m = LlamaForCausalLM(cfg); opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    g = torch.Generator().manual_seed(1)
    for s in range(250):
        f = (s + 1) / 20 if s < 20 else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - 20) / 230))
        for gr in opt.param_groups: gr["lr"] = 3e-3 * f
        st = torch.randint(0, len(ids) - 65, (16,), generator=g); x = torch.stack([ids[i:i + 64] for i in st]); m(input_ids=x, labels=x).loss.backward(); opt.step(); opt.zero_grad()
    torch.save(m.state_dict(), PRE)
rng = random.Random(0); pairs = [(a, b) for a in range(10, 100) for b in range(10, 100)]; rng.shuffle(pairs)
train_p, test_p = pairs[:2000], pairs[2000:2300]
def example(a, b):
    p = enc(f"Q: {a}+{b}=\nA:"); r = enc(f" {a + b}") + [EOS]; return p, r
def collate(items):
    m = max(len(p) + len(r) for p, r in items); ids = torch.zeros(len(items), m, dtype=torch.long); lab = torch.full((len(items), m), -100); att = torch.zeros(len(items), m, dtype=torch.long)
    for k, (p, r) in enumerate(items):
        s = p + r; ids[k, :len(s)] = torch.tensor(s); lab[k, len(p):len(s)] = torch.tensor(r); att[k, :len(s)] = 1
    return ids, lab, att
@torch.no_grad()
def evaluate(m):
    m.eval(); items = [example(a, b) for a, b in test_p]; tot = n = 0
    for i in range(0, len(items), 100):
        ids, lab, att = collate(items[i:i + 100]); lg = m(input_ids=ids, attention_mask=att).logits[:, :-1]; l = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        tot += l.item(); n += (lab[:, 1:] != -100).sum().item()
    ok = 0
    for a, b in test_p:     # greedy decoding
        x = torch.tensor([enc(f"Q: {a}+{b}=\nA:")]); out = m.generate(input_ids=x, max_new_tokens=6, do_sample=False, eos_token_id=EOS, pad_token_id=EOS)
        ok += tk.decode(out[0, x.size(1):].tolist(), skip_special_tokens=True).strip() == str(a + b)
    m.train(); return tot / n, ok / len(test_p)
def finetune(name, kind, r=8, lr=None, steps=int(os.environ.get("STEPS", 150)), alpha=None, **kw):
    torch.manual_seed(0); m = LlamaForCausalLM(cfg); m.load_state_dict(torch.load(PRE))
    lr = lr or (1e-3 if kind.startswith("full") else float(os.environ.get("LRL", 2e-2)))
    if kind != "full":
        m = get_peft_model(m, LoraConfig(r=r, lora_alpha=alpha or 2 * r, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM",
                                          use_rslora=(kind == "rslora"), use_dora=(kind == "dora")))
    if kind == "full_noemb":
        for n_, p_ in m.named_parameters(): p_.requires_grad = "embed_tokens" not in n_
    tr = [p for p in m.parameters() if p.requires_grad]; ntr = sum(p.numel() for p in tr); ntot = sum(p.numel() for p in m.parameters())
    ratio = kw.get("plus", 1.0); named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    groups = [{"params": [p for n, p in named if "lora_B" in n], "lr": lr * ratio}, {"params": [p for n, p in named if "lora_B" not in n], "lr": lr}] if ratio > 1 else [{"params": tr, "lr": lr}]
    for g_ in groups: g_["base"] = g_["lr"]
    opt = torch.optim.AdamW([g_ for g_ in groups if g_["params"]], weight_decay=0.0); g = random.Random(1); items = [example(a, b) for a, b in train_p]; t0 = time.time(); last = []
    for s in range(steps):
        f = (s + 1) / 8 if s < 8 else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - 8) / (steps - 8)))
        for g_ in opt.param_groups: g_["lr"] = g_["base"] * f
        ids, lab, att = collate([items[g.randrange(len(items))] for _ in range(32)]); loss = m(input_ids=ids, attention_mask=att, labels=lab).loss; loss.backward()
        torch.nn.utils.clip_grad_norm_(tr, 1.0); opt.step(); opt.zero_grad(); last.append(loss.item())
    secs = time.time() - t0; vl, acc = evaluate(m)
    row = {"name": name, "lr": lr, "trainable": ntr, "trainable_pct": 100 * ntr / (ntot if kind.startswith("full") else ntot - ntr), "optimizer_state_KB": ntr * 8 / 1024, "train_loss": sum(last[-10:]) / 10,
           "heldout_loss": vl, "exact_match": acc, "secs": secs}
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True); return row
specs = {"base": ("zero-shot", None)}
which = sys.argv[1].split(",")
res = []
for w in which:
    if w == "zero":
        torch.manual_seed(0); m = LlamaForCausalLM(cfg); m.load_state_dict(torch.load(PRE)); vl, acc = evaluate(m); row = {"name": "no fine-tuning", "trainable": 0, "heldout_loss": vl, "exact_match": acc}; print(row, flush=True); res.append(row)
    elif w == "full": res.append(finetune("full fine-tuning", "full"))
    elif w == "fullnoemb": res.append(finetune("full FT, embeddings frozen", "full_noemb"))
    elif w.startswith("lora"): res.append(finetune(f"LoRA r={w[4:]}", "lora", r=int(w[4:])))
    elif w == "rslora16": res.append(finetune("rsLoRA r=16", "rslora", r=16))
    elif w == "dora8": res.append(finetune("DoRA r=8", "dora", r=8))
    elif w == "plus8": res.append(finetune("LoRA+ r=8 (ratio 16)", "lora", r=8, plus=16.0))
prev = json.load(open("results/exp3.json")) if os.path.exists("results/exp3.json") else []
json.dump(prev + res, open("results/exp3.json", "w"), indent=1)
