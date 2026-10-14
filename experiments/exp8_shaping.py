"""Exp 8 - 'closer = more reward' on a HARD task (cold start).
Generation 0 knows only 2-digit addition (the exp5 model). Target: 3-digit addition, learned from an UNLABELLED pool with a verifier only.
Rewards for the same samples:
  binary   : 1 if the answer is exactly right else 0
  digit    : 0.5 * (fraction of digits correct, aligned from the right) + 0.5 * exact          (graded closeness)
  numeric  : max(0, 1 - |answer - truth| / truth)                                              (numeric closeness)
Methods: GRPO with each reward; RFT keeping exact answers only; RFT keeping the best-of-k sample if its graded score >= 0.5."""
import json, math, os, random, sys, time, torch
sys.argv = ["x", "none"]; os.environ.setdefault("NLAB", "1000"); os.environ.setdefault("STEPS0", "1500")
exec(open("exp5_selfimprove.py").read().split('if __name__ == "__main__":')[0])
rr = random.Random(11); P3 = [(a, b) for a in range(100, 1000) for b in range(100, 1000)]; rr.shuffle(P3); POOL3, TEST3 = P3[:6000], P3[6000:6400]
def digit(s, t):
    t = str(t); k = sum(1 for i in range(1, len(t) + 1) if len(s) >= i and s[-i] == t[-i]); return 0.5 * k / len(t) + 0.5 * float(s == t)
def numeric(s, t):
    try: return max(0.0, 1 - abs(int(s) - t) / t)
    except ValueError: return 0.0
REW = {"binary": lambda s, t: float(s == str(t)), "digit": digit, "numeric": numeric}
def ev3(m):
    g = sample(m, TEST3, 1, 0); acc = sum(s == str(a + b) for s, (a, b) in zip(g, TEST3)) / len(TEST3)
    dg = sum(digit(s, a + b) for s, (a, b) in zip(g, TEST3)) / len(TEST3); s4 = sample(m, TEST3, 4, 1.0)
    p4 = sum(any(s4[4 * i + j] == str(a + b) for j in range(4)) for i, (a, b) in enumerate(TEST3)) / len(TEST3); return {"acc3": acc, "digit_score3": dg, "pass@4_3": p4}
def grpo(m, probs, k, rew, lr, seed, eps=0.2, steps=100):
    s = sample(m, probs, k, 1.0); rows = []; mean_r = 0.0; exact = 0
    for i, (a, b) in enumerate(probs):
        rs = [rew(s[k * i + j], a + b) for j in range(k)]; mean_r += sum(rs); exact += sum(s[k * i + j] == str(a + b) for j in range(k)); t = torch.tensor(rs)
        if t.std(unbiased=False) < 1e-6: continue
        adv = ((t - t.mean()) / (t.std(unbiased=False) + 1e-4)).tolist()
        for j in range(k): rows.append((seqs([(a, b, s[k * i + j])])[0], adv[j]))
    if not rows: return 0, exact / (k * len(probs))
    m.eval()
    with torch.no_grad():
        old = []
        for i in range(0, len(rows), 256):
            ids, mk, att = pad([r[0] for r in rows[i:i + 256]]); lg = m(input_ids=ids, attention_mask=att).logits[:, :-1]; old.append(torch.log_softmax(lg.float(), -1).gather(-1, ids[:, 1:, None])[..., 0])
        ml = max(o.size(1) for o in old); old = torch.cat([torch.nn.functional.pad(o, (0, ml - o.size(1))) for o in old])
    m.train(); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0); g = random.Random(seed)
    for st in range(steps):
        for gr in opt.param_groups: gr["lr"] = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * st / steps)))
        idx = [g.randrange(len(rows)) for _ in range(64)]; ids, mk, att = pad([rows[i][0] for i in idx]); adv = torch.tensor([rows[i][1] for i in idx])[:, None]
        lg = m(input_ids=ids, attention_mask=att).logits[:, :-1]; lp = torch.log_softmax(lg.float(), -1).gather(-1, ids[:, 1:, None])[..., 0]; ol = old[idx][:, :lp.size(1)]
        ratio = torch.exp(lp - ol); obj = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv); mm = mk[:, 1:]
        (( -(obj * mm).sum(1) / mm.sum(1)).mean()).backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
    return len(rows), exact / (k * len(probs))
def rft(m, probs, k, graded, seed):
    s = sample(m, probs, k, 1.0); kept = []
    for i, (a, b) in enumerate(probs):
        cand = [(digit(s[k * i + j], a + b), s[k * i + j]) for j in range(k)]
        if graded:
            sc, best = max(cand)
            if sc >= 0.5 and best: kept.append((a, b, best))
        else:
            kept += [(a, b, x) for sc, x in set(cand) if x == str(a + b)]
    if kept: sgd(m, 100, 5e-4, sft_loss, seqs(kept), seed=seed)
    return len(kept), sum(s[k * i + j] == str(a + b) for i, (a, b) in enumerate(probs) for j in range(k)) / (k * len(probs))
if __name__ == "__main__" or True:
    pass
