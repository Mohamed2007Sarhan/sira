"""Exp 8d - bootstrap then self-improve. After a capability cliff (acc = 0 at every level), give a SMALL external lesson
(labelled 3-digit examples from a trusted solver, mixed with 2-digit rehearsal), then continue with verifier-only RFT rounds.
Control: spend the same optimisation steps only on the labelled lesson (no self-improvement rounds)."""
import json, os, sys, time, torch
method, seed = sys.argv[1], int(sys.argv[2]); NB = int(os.environ.get("NBOOT", 400)); sys.argv = ["x", "none"]
exec(open("exp8_shaping.py").read())
def evx(m, pairs): g = sample(m, pairs, 1, 0); return sum(s == str(a + b) for s, (a, b) in zip(g, pairs)) / len(pairs)
def ev_all(m): return {"acc3": evx(m, TEST3), "acc2": evx(m, TEST)}
m = gen0(seed); hist = [{"round": 0, **ev_all(m)}]; print("gen0", hist[0], flush=True)
lesson = [(a, b, str(a + b)) for a, b in POOL3[-NB:]] + [(a, b, str(a + b)) for a, b in LAB[:NB]]       # external lesson + 2-digit rehearsal
sgd(m, int(os.environ.get("LSTEPS", 400)), 1e-3, sft_loss, seqs(lesson), seed=seed); hist.append({"round": "lesson", **ev_all(m)}); print("after lesson", hist[-1], flush=True)
for r_ in range(4):
    t0 = time.time()
    if method == "boot_rft":
        probs = POOL3[r_ * 700:(r_ + 1) * 700] + POOL[r_ * 250:(r_ + 1) * 250]; n, sr = rft(m, probs, 4, False, r_)
    else:
        sgd(m, 100, 5e-4, sft_loss, seqs(lesson), seed=r_); n, sr = len(lesson), float("nan")
    hist.append({"round": r_ + 1, "rows": n, "sample_exact": sr, **ev_all(m)}); print(method, "round", r_ + 1, {k: round(v, 3) if isinstance(v, float) else v for k, v in hist[-1].items()}, "%.0fs" % (time.time() - t0), flush=True)
res = json.load(open("results/exp8.json")) if os.path.exists("results/exp8.json") else {}; res[f"{method}_seed{seed}_nboot{NB}"] = hist; json.dump(res, open("results/exp8.json", "w"), indent=1)
