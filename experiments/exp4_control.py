"""Control for Exp 4B: the same 80 healing steps applied to the UNCONVERTED standard model (so 'healing' is not confused with 'more training')."""
import json, sys, torch
sys.argv = ["x", "B", ""]            # reuse the definitions of exp4_cla without running its modes
src = open("exp4_cla.py").read().split("res = json.load")[0]
exec(src)
sd = torch.load("/tmp/std6.pt"); m = mk(1); m.load_state_dict(sd); before = val_loss(m); fit(m, 80, 1e-3, seed=5, warm=5); after = val_loss(m)
r = json.load(open("results/exp4.json")); r["control_standard_plus_80_steps"] = {"before": before, "after": after}; json.dump(r, open("results/exp4.json", "w"), indent=1)
print("standard model:", round(before, 4), "-> +80 steps:", round(after, 4))
