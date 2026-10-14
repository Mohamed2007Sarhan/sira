"""Exp 7 - computed (not measured) numbers for real-size models from the verified equations.
Assumptions are explicit and printed; change them at the top."""
import json, math
exec(open("exp1_params.py").read().split("out = {")[0])      # n_params, lora_params, kv_bytes_per_token
MODELS = {"Qwen2.5-0.5B": dict(hidden=896, layers=24, vocab=151936, inter=4864, heads=14, kv=2, tie=True),
          "Llama-3-8B": dict(hidden=4096, layers=32, vocab=128256, inter=14336, heads=32, kv=8),
          "Llama-2-70B": dict(hidden=8192, layers=80, vocab=32000, inter=28672, heads=64, kv=8)}
A = dict(gpu="A100-80GB", peak_tflops=312.0, mfu=0.40, hbm_GBps=2039.0, usd_per_gpu_hour=2.0)    # ASSUMPTIONS (bf16 dense)
H = dict(gpu="H100-80GB", peak_tflops=989.0, mfu=0.40, hbm_GBps=3350.0, usd_per_gpu_hour=3.5)
E_, A_, B_, al, be = 1.6934, 406.4, 410.7, 0.3392, 0.2849                                      # Hoffmann et al. 2022 (precise values as reported by Besiroglu et al. 2024)
EPOCH = (1.817, 482.01, 2085.43, 0.3478, 0.3658)                                                    # Epoch AI replication fit (for comparison)
chin = lambda N, D: E_ + A_ / N ** al + B_ / D ** be
rows = []
for name, c in MODELS.items():
    N = n_params(c); r = {"model": name, "N": N}
    r["chinchilla_optimal_tokens"] = 20 * N; r["chinchilla_loss_at_20N"] = chin(N, 20 * N); r["epoch_loss_at_20N"] = EPOCH[0] + EPOCH[1] / N ** EPOCH[3] + EPOCH[2] / (20 * N) ** EPOCH[4]
    r["pretrain_flops_20N"] = 6 * N * 20 * N
    for g in (A, H):
        sec = r["pretrain_flops_20N"] / (g["peak_tflops"] * 1e12 * g["mfu"]); r[f"pretrain_gpu_hours_{g['gpu']}"] = sec / 3600; r[f"pretrain_usd_{g['gpu']}"] = sec / 3600 * g["usd_per_gpu_hour"]
    # fine-tuning 10M tokens (one RFT/SFT round) on one GPU
    D_ft = 1e7; f_ft = 6 * N * D_ft; r["ft_10M_tokens_flops"] = f_ft
    r["ft_10M_tokens_minutes_A100"] = f_ft / (A["peak_tflops"] * 1e12 * A["mfu"]) / 60
    # sampling for one RFT round: 1000 problems x 4 samples x 300 new tokens, memory-bound decode: tokens/s ~ batch * HBM / weight_bytes
    toks = 1000 * 4 * 300; batch = 128; wbytes = 2 * N
    tps = batch * A["hbm_GBps"] * 1e9 / wbytes; r["rft_sampling_tokens"] = toks; r["rft_sampling_seconds_A100_batch128"] = toks / tps
    r["weights_bf16_GB"] = 2 * N / 1e9; r["lora_r16_trainable_M"] = lora_params(c, 16) / 1e6
    r["qlora_weights_GB"] = 0.5 * N * 1.06 / 1e9; r["full_ft_GB"] = 16 * N / 1e9
    r["kv_GB_per_32k_sequence"] = kv_bytes_per_token(c) * 32768 / 1e9; r["kv_GB_per_32k_sequence_CLA2"] = kv_bytes_per_token(c, cla=2) * 32768 / 1e9
    r["max_concurrent_32k_seqs_in_40GB_free_after_weights_80GB_gpu"] = max(0, (80 - 2 * N / 1e9 - 4) * 1e9 / (kv_bytes_per_token(c) * 32768))
    r["max_concurrent_32k_seqs_CLA2"] = max(0, (80 - 2 * N / 1e9 - 4) * 1e9 / (kv_bytes_per_token(c, cla=2) * 32768))
    rows.append(r)
# statistics of the acceptance test
import math as m
def se(p, n): return m.sqrt(p * (1 - p) / n)
stats = {"se_accuracy_p0.5_n80": se(0.5, 80), "se_accuracy_p0.9_n400": se(0.9, 400), "ci95_halfwidth_p0.9_n400": 1.96 * se(0.9, 400),
         "paired_example_9_vs_3_p": sum(m.comb(12, i) for i in range(9, 13)) / 2 ** 12}
# pass@k and expected yield of rejection sampling
passk = {p: {k: 1 - (1 - p) ** k for k in (1, 4, 16)} for p in (0.01, 0.05, 0.2, 0.5)}
json.dump({"assumptions": {"A100": A, "H100": H}, "models": rows, "stats": stats, "pass_at_k": passk}, open("results/exp7.json", "w"), indent=1)
for r in rows:
    print(f"{r['model']:13s} N={r['N']/1e9:6.2f}B  Chinchilla D*={r['chinchilla_optimal_tokens']/1e12:6.3f}T tokens, L(N,D*)={r['chinchilla_loss_at_20N']:.3f} (Epoch fit {r['epoch_loss_at_20N']:.3f}), FLOPs={r['pretrain_flops_20N']:.2e}, "
          f"A100h={r['pretrain_gpu_hours_A100-80GB']:,.0f} (${r['pretrain_usd_A100-80GB']:,.0f}), H100h={r['pretrain_gpu_hours_H100-80GB']:,.0f}")
    print(f"{'':13s} FT 10M tok: {r['ft_10M_tokens_minutes_A100']:.1f} min A100 | RFT sampling 1.2M tok: {r['rft_sampling_seconds_A100_batch128']:.0f}s | bf16 {r['weights_bf16_GB']:.1f}GB, QLoRA {r['qlora_weights_GB']:.1f}GB, fullFT {r['full_ft_GB']:.0f}GB, LoRA16 {r['lora_r16_trainable_M']:.1f}M | "
          f"KV@32k {r['kv_GB_per_32k_sequence']:.2f}GB -> CLA2 {r['kv_GB_per_32k_sequence_CLA2']:.2f}GB; concurrent seqs {r['max_concurrent_32k_seqs_in_40GB_free_after_weights_80GB_gpu']:.1f} -> {r['max_concurrent_32k_seqs_CLA2']:.1f}")
print(stats); print(passk)
