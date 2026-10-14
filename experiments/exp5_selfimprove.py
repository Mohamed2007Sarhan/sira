"""Exp 5 - self-improvement with a verifier on 2-digit addition (character-level model, ~0.8M params, trained from scratch).
gen0: supervised on a SMALL labelled set. Then rounds of: sample the model on UNLABELLED problems -> verify (exact check only) -> learn.
 RFT : fine-tune on verified-correct samples only                      L = -mean log p(y | x)  for y with R(y)=1
 GRPO: A_i=(r_i-mean)/(std+1e-4), clipped ratio objective (eps=0.2)      L = -mean_tokens min(rho*A, clip(rho,1-eps,1+eps)*A)
Metrics on a held-out test set: greedy accuracy, pass@4 (T=1), mean sampling reward."""
import json, math, os, random, sys, time, torch
torch.set_num_threads(1)
from transformers import LlamaConfig, LlamaForCausalLM
CH = "0123456789+=\n"; PAD, BOS = len(CH), len(CH) + 1; V = len(CH) + 2; NL = CH.index("\n")
enc = lambda s: [CH.index(c) for c in s]
cfg = LlamaConfig(vocab_size=V, hidden_size=128, intermediate_size=344, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
                  bos_token_id=BOS, eos_token_id=NL, pad_token_id=PAD, tie_word_embeddings=True)
rng = random.Random(0); pairs = [(a, b) for a in range(10, 100) for b in range(10, 100)]; rng.shuffle(pairs)
LAB, POOL, TEST = pairs[:int(os.environ.get("NLAB", 1500))], pairs[1000:5000], pairs[7700:8100]
prompt = lambda a, b: [BOS] + enc(f"{a}+{b}=")
def seqs(items):   # (a,b,answer_text) -> ids, mask
    out = []
    for a, b, ans in items: p = prompt(a, b); r = enc(ans) + [NL]; out.append((p + r, [0] * len(p) + [1] * len(r)))
    return out
def pad(batch):
    m = max(len(s) for s, _ in batch); ids = torch.full((len(batch), m), PAD); mk = torch.zeros(len(batch), m); att = torch.zeros(len(batch), m, dtype=torch.long)
    for k, (s, mm) in enumerate(batch): ids[k, :len(s)] = torch.tensor(s); mk[k, :len(mm)] = torch.tensor(mm, dtype=torch.float); att[k, :len(s)] = 1
    return ids, mk, att
def sgd(m, steps, lr, loss_fn, data, bs=64, seed=0):
    g = random.Random(seed); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0); m.train()
    for s in range(steps):
        f = (s + 1) / 5 if s < 5 else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - 5) / max(1, steps - 5)))
        for gr in opt.param_groups: gr["lr"] = lr * f
        loss = loss_fn(m, [data[g.randrange(len(data))] for _ in range(bs)]); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
def sft_loss(m, batch):
    ids, mk, att = pad(batch[:]); lg = m(input_ids=ids, attention_mask=att).logits[:, :-1]; lp = torch.log_softmax(lg.float(), -1).gather(-1, ids[:, 1:, None])[..., 0]
    return -((lp * mk[:, 1:]).sum() / mk[:, 1:].sum())
@torch.no_grad()
def sample(m, probs, n, temp):
    m.eval(); out = []
    for i in range(0, len(probs), 500):
        P = torch.tensor([prompt(a, b) for a, b in probs[i:i + 500]])
        kw = dict(do_sample=True, temperature=temp, top_k=0, num_return_sequences=n) if temp > 0 else dict(do_sample=False)
        g = m.generate(input_ids=P, attention_mask=torch.ones_like(P), max_new_tokens=5, eos_token_id=NL, pad_token_id=PAD, **kw)[:, P.size(1):]
        for row in g.tolist():
            s = "";  
            for t in row:
                if t == NL or t >= len(CH): break
                s += CH[t]
            out.append(s)
    m.train(); return out
def evaluate(m):
    g = sample(m, TEST, 1, 0); acc = sum(s == str(a + b) for s, (a, b) in zip(g, TEST)) / len(TEST)
    s4 = sample(m, TEST, 4, 1.0); pass4 = sum(any(s4[4 * i + j] == str(a + b) for j in range(4)) for i, (a, b) in enumerate(TEST)) / len(TEST)
    mean_r = sum(s4[4 * i + j] == str(a + b) for i, (a, b) in enumerate(TEST) for j in range(4)) / (4 * len(TEST)); return {"acc": acc, "pass@4": pass4, "mean_reward_T1": mean_r}
def new_model(seed): torch.manual_seed(seed); return LlamaForCausalLM(cfg)
def gen0(seed=0, steps=int(os.environ.get("STEPS0", 250))):
    path = f"/tmp/g0_{seed}_{len(LAB)}_{steps}.pt"; m = new_model(seed)
    if os.path.exists(path): m.load_state_dict(torch.load(path)); return m
    sgd(m, steps, 3e-3, sft_loss, seqs([(a, b, str(a + b)) for a, b in LAB]), seed=seed); torch.save(m.state_dict(), path); return m
def rft_round(m, k, r_):
    probs = POOL[r_ * 1000:(r_ + 1) * 1000]; s = sample(m, probs, k, 1.0); kept = {}
    for i, (a, b) in enumerate(probs):
        for j in range(k):
            if s[k * i + j] == str(a + b): kept[(a, b, s[k * i + j])] = 1
    data = seqs(list(kept)); sgd(m, 100, 5e-4, sft_loss, data, seed=r_); return len(data), sum(s[k * i + j] == str(a + b) for i, (a, b) in enumerate(probs) for j in range(k)) / (k * len(probs))
def nofilter_round(m, k, r_):
    """CONTROL: self-training on the model's own samples WITHOUT the verifier (wrong answers are kept as targets)."""
    probs = POOL[r_ * 1000:(r_ + 1) * 1000]; s = sample(m, probs, k, 1.0); data = seqs([(a, b, s[k * i + j]) for i, (a, b) in enumerate(probs) for j in range(k) if s[k * i + j]])
    sgd(m, 100, 5e-4, sft_loss, data, seed=r_); return len(data), sum(s[k * i + j] == str(a + b) for i, (a, b) in enumerate(probs) for j in range(k)) / (k * len(probs))
def labelled_round(m, k, r_):
    """CONTROL: the same number of extra optimisation steps on the ORIGINAL labelled data only."""
    sgd(m, 100, 5e-4, sft_loss, seqs([(a, b, str(a + b)) for a, b in LAB]), seed=r_); return len(LAB), float("nan")
def grpo_round(m, k, r_, eps=0.2):
    probs = POOL[r_ * 1000:(r_ + 1) * 1000]; s = sample(m, probs, k, 1.0); rows = []; rew = 0
    for i, (a, b) in enumerate(probs):
        rs = [1.0 if s[k * i + j] == str(a + b) else 0.0 for j in range(k)]; rew += sum(rs)
        t = torch.tensor(rs)
        if t.std(unbiased=False) < 1e-6: continue
        adv = ((t - t.mean()) / (t.std(unbiased=False) + 1e-4)).tolist()
        for j in range(k): rows.append((seqs([(a, b, s[k * i + j])])[0], adv[j]))
    m.eval()
    with torch.no_grad():
        old = []
        for i in range(0, len(rows), 256):
            ids, mk, att = pad([r[0] for r in rows[i:i + 256]]); lg = m(input_ids=ids, attention_mask=att).logits[:, :-1]; old.append(torch.log_softmax(lg.float(), -1).gather(-1, ids[:, 1:, None])[..., 0])
        maxlen = max(o.size(1) for o in old); old = torch.cat([torch.nn.functional.pad(o, (0, maxlen - o.size(1))) for o in old])
    m.train(); GLR = float(os.environ.get("GLR", 5e-4)); opt = torch.optim.AdamW(m.parameters(), lr=GLR, weight_decay=0.0); g = random.Random(r_); steps = 100
    for st in range(steps):
        f = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * st / steps))
        for gr in opt.param_groups: gr["lr"] = GLR * f
        idx = [g.randrange(len(rows)) for _ in range(64)]; ids, mk, att = pad([rows[i][0] for i in idx]); adv = torch.tensor([rows[i][1] for i in idx])[:, None]
        lg = m(input_ids=ids, attention_mask=att).logits[:, :-1]; lp = torch.log_softmax(lg.float(), -1).gather(-1, ids[:, 1:, None])[..., 0]
        ol = old[idx][:, :lp.size(1)]; ratio = torch.exp(lp - ol); obj = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv); mm = mk[:, 1:]
        loss = (-(obj * mm).sum(1) / mm.sum(1)).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
    return len(rows), rew / (k * len(probs))
if __name__ == "__main__":
    mode, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
    res = json.load(open("results/exp5.json")) if os.path.exists("results/exp5.json") else {}
    t0 = time.time(); m = gen0(seed); e0 = evaluate(m); print("gen0", {k: round(v, 3) for k, v in e0.items()}, "%.0fs" % (time.time() - t0), flush=True)
    hist = [{"round": 0, **e0}]; rounds = int(os.environ.get("ROUNDS", 4))
    for r_ in range(rounds):
        t1 = time.time(); n, rew = {"rft": rft_round, "grpo": grpo_round, "nofilter": nofilter_round, "labelled": labelled_round}[mode](m, 4, r_); e = evaluate(m)
        hist.append({"round": r_ + 1, "train_rows": n, "sampling_reward": rew, **e}); print(mode, "round", r_ + 1, {k: round(v, 3) if isinstance(v, float) else v for k, v in hist[-1].items()}, "%.0fs" % (time.time() - t1), flush=True)
    torch.save(m.state_dict(), f"/tmp/{mode}_{seed}.pt"); res[f"{mode}{os.environ.get('GLR','') and '_lr'+os.environ['GLR']}_seed{seed}_nlab{len(LAB)}"] = hist; json.dump(res, open("results/exp5.json", "w"), indent=1)
