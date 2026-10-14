import json, sys
from stats import summary, paired, fmt
r = json.load(open("results/exp5.json"))
seeds = sorted({int(k.split("_seed")[1].split("_")[0]) for k in r if k.startswith("rft_seed")})
def get(prefix):
    return {s: r[k] for k in r for s in [int(k.split("_seed")[1].split("_")[0])] if k.startswith(prefix + "_seed") and "nlab1000" in k}
rft, lab, nof, grpo = get("rft"), get("labelled"), get("nofilter"), get("grpo_lr1e-4")
S = [s for s in seeds if s in rft and s in lab and s in nof and s in grpo]
final = lambda d: [d[s][-1]["acc"] for s in S]
start = [rft[s][0]["acc"] for s in S]
gain = lambda d: [d[s][-1]["acc"] - d[s][0]["acc"] for s in S]
out = {"seeds": S, "gen0": summary(start), "final": {k: summary(final(v)) for k, v in dict(RFT=rft, GRPO_1e4=grpo, labelled_only=lab, no_verifier=nof).items()},
       "gain": {k: summary(gain(v)) for k, v in dict(RFT=rft, GRPO_1e4=grpo, labelled_only=lab, no_verifier=nof).items()},
       "paired_gain_vs_labelled": paired(gain(rft), gain(lab)), "paired_gain_vs_noverifier": paired(gain(rft), gain(nof)), "paired_gain_grpo_vs_labelled": paired(gain(grpo), gain(lab))}
json.dump(out, open("results/exp5_stats.json", "w"), indent=1, default=float)
print("seeds", S, "| gen0 accuracy:", fmt(out["gen0"]))
for k in out["gain"]: print(f"{k:14s} gain {fmt(out['gain'][k])}   final {fmt(out['final'][k])}")
for k in ("paired_gain_vs_labelled", "paired_gain_vs_noverifier", "paired_gain_grpo_vs_labelled"):
    p = out[k]; print(k, {a: (round(b * 100, 2) if a in ("mean_diff", "sd_diff") else (tuple(round(x * 100, 2) for x in b) if a == "ci95" else b)) for a, b in p.items()})
print("per-seed gen0:", [round(x, 3) for x in start]); print("per-seed RFT gain:", [round(x, 3) for x in gain(rft)]); print("per-seed labelled gain:", [round(x, 3) for x in gain(lab)])
