"""Exp 8c - curriculum (zone of proximal development) + rehearsal. Level 1: 3-digit + 2-digit; level 2: 3-digit + 3-digit.
Rounds 1-2 on level 1, rounds 3-4 on level 2; 25% of every round is 2-digit rehearsal. Same verifier-only learning as before."""
import json, os, sys, time, torch
method, seed = sys.argv[1], int(sys.argv[2]); sys.argv = ["x", "none"]
exec(open("exp8_shaping.py").read())
r2 = random.Random(21); L1 = [(a, b) for a in range(100, 1000) for b in range(10, 100)]; r2.shuffle(L1); POOL1, TESTL1 = L1[:4000], L1[4000:4400]
def evx(m, pairs):
    g = sample(m, pairs, 1, 0); return sum(s == str(a + b) for s, (a, b) in zip(g, pairs)) / len(pairs)
def ev_all(m): return {"acc_L1": evx(m, TESTL1), "acc3": evx(m, TEST3), "acc2": evx(m, TEST)}
m = gen0(seed); hist = [{"round": 0, **ev_all(m)}]; print("gen0", {k: round(v, 3) for k, v in hist[0].items()}, flush=True)
rew = REW["digit"] if method.endswith("digit") else REW["binary"]
for r_ in range(4):
    pool = POOL1 if r_ < 2 else POOL3; main = pool[r_ * 600:(r_ + 1) * 600]; replay = POOL[r_ * 200:(r_ + 1) * 200]; probs = main + replay; t0 = time.time()
    n, sr = rft(m, probs, 4, False, r_) if method == "cur_rft" else grpo(m, probs, 4, rew, 1e-4, r_)
    hist.append({"round": r_ + 1, "level": 1 if r_ < 2 else 2, "rows": n, "sample_exact": sr, **ev_all(m)}); print(method, "round", r_ + 1, {k: round(v, 3) if isinstance(v, float) else v for k, v in hist[-1].items()}, "%.0fs" % (time.time() - t0), flush=True)
res = json.load(open("results/exp8.json")) if os.path.exists("results/exp8.json") else {}; res[f"{method}_seed{seed}"] = hist; json.dump(res, open("results/exp8.json", "w"), indent=1)
