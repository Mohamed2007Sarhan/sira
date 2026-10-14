import json, os, sys, time, torch
sys.argv_real = sys.argv[:]; method, seed = sys.argv[1], int(sys.argv[2]); sys.argv = ["x", "none"]
exec(open("exp8_shaping.py").read())
res = json.load(open("results/exp8.json")) if os.path.exists("results/exp8.json") else {}
m = gen0(seed); e0 = ev3(m); e2 = evaluate(m); hist = [{"round": 0, **e0, "acc2": e2["acc"]}]; print("gen0", {k: round(v, 3) for k, v in hist[0].items()}, flush=True)
for r_ in range(4):
    probs = POOL3[r_ * 1000:(r_ + 1) * 1000]; t0 = time.time()
    if method.startswith("grpo_"): n, sr = grpo(m, probs, 4, REW[method[5:]], float(os.environ.get("GLR", 1e-4)), r_)
    elif method == "rft_exact": n, sr = rft(m, probs, 4, False, r_)
    elif method == "rft_graded": n, sr = rft(m, probs, 4, True, r_)
    e = ev3(m); hist.append({"round": r_ + 1, "rows": n, "sample_exact": sr, **e}); print(method, "round", r_ + 1, {k: round(v, 3) if isinstance(v, float) else v for k, v in hist[-1].items()}, "%.0fs" % (time.time() - t0), flush=True)
hist[-1]["acc2"] = evaluate(m)["acc"]; res[f"{method}_seed{seed}"] = hist; json.dump(res, open("results/exp8.json", "w"), indent=1); print("2-digit retention at end:", round(hist[-1]["acc2"], 3))
