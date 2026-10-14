# SIRA - Experiments

Everything here is reproducible (CPU, one core, a few minutes each). Results are written to `results/*.json`.

| Script | What it measures |
|---|---|
| `exp1_params.py` | closed-form parameter / memory / FLOP / KV-cache equations vs real PyTorch models and published model sizes |
| `exp2_scaling.py` | pre-training on real English text, model width sweep: loss, tokens/s, memory |
| `exp3_adapters.py` | full fine-tuning vs LoRA (ranks) / rsLoRA / DoRA / LoRA+ on one pretrained model (`STEPS`, `LRL` env vars) |
| `exp4_cla.py` / `exp4_control.py` | Cross-Layer Attention: trained from scratch with factor 1/2/3/6, and conversion of a standard model to CLA with healing |
| `exp5_selfimprove.py` | self-improvement with a verifier: RFT vs GRPO vs controls (no verifier, labelled data only) |
| `exp6_weights.py` | weight interpolation (soup) and layer transplant: same lineage vs independent models |
| `exp8_shaping.py`, `exp8_run.py`, `exp8c_run.py`, `exp8d_run.py` | graded rewards (closer = more reward), curriculum and external bootstrap on a cold-start task: mostly NEGATIVE results (see `results/exp8_notes.json`) |
| `exp7_scale.py` | computed (not measured) cost / memory numbers for 0.5B, 8B, 70B models |

Prerequisites: `pip install torch transformers peft tokenizers scipy`; exp2-4 read a text corpus from `/tmp/corpus_en.txt`
(`python -c "import pydoc_data.topics as t; open('/tmp/corpus_en.txt','w').write('\n\n'.join(t.topics.values()))"`).
