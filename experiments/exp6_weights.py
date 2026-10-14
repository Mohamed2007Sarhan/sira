"""Exp 6 - weight-space operations. A0 = generation 0, A1 = A0 after 4 RFT rounds (same lineage), B = independently initialised model
trained on the same labelled data. (a) linear interpolation (soup), (b) layer transplant."""
import json, os, sys, torch
sys.argv = ["x", "none"]; os.environ.setdefault("NLAB", "1000"); os.environ.setdefault("STEPS0", "1500")
src = open("exp5_selfimprove.py").read().split('if __name__ == "__main__":')[0]; exec(src)
A0, A1, B = [torch.load(p) for p in ("/tmp/g0_0_1000_1500.pt", "/tmp/rft_0.pt", "/tmp/g0_1_1000_1500.pt")]
def acc_of(sd):
    m = new_model(0); m.load_state_dict(sd); return evaluate(m)["acc"]
mix = lambda X, Y, l: {k: ((1 - l) * X[k].float() + l * Y[k].float()).to(X[k].dtype) if X[k].is_floating_point() else X[k] for k in X}
out = {"A0": acc_of(A0), "A1": acc_of(A1), "B": acc_of(B)}; print("A0 %.3f  A1(RFT x4) %.3f  B(independent) %.3f" % (out["A0"], out["A1"], out["B"]))
out["interp_same_lineage_A0_to_A1"] = {str(l): acc_of(mix(A0, A1, l)) for l in (0, 0.25, 0.5, 0.75, 1.0, 1.25)}
out["interp_independent_A0_to_B"] = {str(l): acc_of(mix(A0, B, l)) for l in (0, 0.25, 0.5, 0.75, 1.0)}
out["interp_independent_A1_to_B"] = {str(l): acc_of(mix(A1, B, l)) for l in (0, 0.25, 0.5, 0.75, 1.0)}
def transplant(target, donor, layer):
    sd = dict(target)
    for k in target:
        if f"model.layers.{layer}." in k: sd[k] = donor[k]
    return sd
out["transplant_into_A1_from_A0_same_lineage"] = {str(i): acc_of(transplant(A1, A0, i)) for i in range(4)}
out["transplant_into_A1_from_B_independent"] = {str(i): acc_of(transplant(A1, B, i)) for i in range(4)}
for k, v in out.items(): print(k, {a: round(b, 3) for a, b in v.items()} if isinstance(v, dict) else round(v, 3))
json.dump(out, open("results/exp6.json", "w"), indent=1)
