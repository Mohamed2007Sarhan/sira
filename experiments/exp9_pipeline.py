"""Exp 9 - the REAL SIRA pipeline (TrainingMaestro: intake -> baseline -> rejection-sampling data (+ bootstrap lessons, replay) -> full-weight training ->
parent-vs-child judging with McNemar, retention guard, adaptive LR) on tiny models of several architecture families, domain 'arith' (levels 1-3).
After every round the CURRENT BEST model is measured on a fixed, separate test set (namespace 'confirm'). Usage: exp9_pipeline.py <arch> <seed> [guard=on|off]"""
import json, os, sys, tempfile, time, shutil, torch
sys.path.insert(0, "..")
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from sira.config import Config
from sira.orchestrator import Orchestrator
arch, seed, guard = sys.argv[1], int(sys.argv[2]), (sys.argv[3] if len(sys.argv) > 3 else "on")
ROUNDS = int(os.environ.get("ROUNDS", 5))
tok = AutoTokenizer.from_pretrained("/tmp/tiny"); V = len(tok); d, L, H = 64, 2, 4
KW = {"gpt2": ("gpt2", dict(n_embd=d, n_layer=L, n_head=H, n_positions=512)), "qwen2": ("qwen2", dict(hidden_size=d, num_hidden_layers=L, num_attention_heads=H, num_key_value_heads=2, intermediate_size=128, max_position_embeddings=512)),
      "gpt_neox": ("gpt_neox", dict(hidden_size=d, num_hidden_layers=L, num_attention_heads=H, intermediate_size=128, max_position_embeddings=512)),
      "llama": ("llama", dict(hidden_size=d, num_hidden_layers=L, num_attention_heads=H, num_key_value_heads=2, intermediate_size=128, max_position_embeddings=512))}
mt, kw = KW[arch]; torch.manual_seed(seed)
mdir = tempfile.mkdtemp(); m = AutoModelForCausalLM.from_config(AutoConfig.for_model(mt, vocab_size=V, bos_token_id=1, eos_token_id=2, pad_token_id=2, **kw)); m.save_pretrained(mdir); tok.save_pretrained(mdir)
cfg = Config(llm_provider="mock", auto_install=False, auto_adapt=False, thinking="off", sandbox_backend="auto", sandbox_timeout_s=30, run_root=tempfile.mkdtemp(), base_model=mdir,
             domains=["arith"], training_method="full", allow_cpu_training=True, worker_inprocess=True, eval_problems=48, train_problems=300, samples_per_problem=2, keep_per_problem=1,
             max_new_tokens=12, max_seq_len=300, micro_batch=16, grad_accum=1, epochs=8, learning_rate=2e-3, min_gain=0.0, accept_p=0.3, parallel_experiments=1,
             replay_fraction=0.3 if guard == "on" else 0.0, retention_tol=0.10 if guard == "on" else 1.0, algorithm="rft", try_merge_candidates=False)
o = Orchestrator(cfg); o.build(); tm = o.training; tm.do_intake(); print("recipe:", tm.recipe["method"], "| params:", tm.report["params"], flush=True)
TEST = o.ctx.evalcore.make_set("confirm", 90)           # fixed, never used for acceptance decisions
def measure(path):
    outs = tm.generate(path, TEST, 1, 0.0); res = o.ctx.evalcore.score(TEST, {k: v[0] if v else "" for k, v in outs.items()}, o.ctx.sandbox, 1)
    return {"acc": res.total, **{k: round(v["rate"], 3) for k, v in res.by_level.items()}}
b = tm.baseline(); hist = [{"round": 0, **measure(cfg.base_model), "accepted": True}]; print("round 0", hist[0], flush=True)
for r in range(1, ROUNDS + 1):
    t0 = time.time(); g = tm.improve([]); best = tm.best()
    row = {"round": r, "accepted": bool(g and g["accepted"]), "lr": tm.hyper["learning_rate"], **measure(best["path"])}
    if g: row.update(p=round(g["detail"].get("p_value", 1), 3), regress=len(g["detail"].get("retention_violations", [])), algo=(g["detail"].get("data") or {}).get("algorithm"), boot=(g["detail"].get("data") or {}).get("bootstrap_rows"),
                      replay=(g["detail"].get("data") or {}).get("replay_rows"), support=((g["detail"].get("data") or {}).get("signal_audit") or {}).get("support"))
    hist.append(row); print("round", r, row, "%.0fs" % (time.time() - t0), flush=True)
out = json.load(open("results/exp9.json")) if os.path.exists("results/exp9.json") else {}
out[f"{arch}_seed{seed}_guard{guard}"] = {"params": tm.report["params"], "history": hist}; json.dump(out, open("results/exp9.json", "w"), indent=1); o.stop()
