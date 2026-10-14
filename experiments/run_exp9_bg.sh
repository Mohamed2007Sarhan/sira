#!/bin/bash
cd /home/claude/sira_project/experiments
for seed in 0 1; do
  for arch in gpt2 qwen2 gpt_neox llama; do python3 -u exp9_pipeline.py $arch $seed on > /tmp/bg_exp9_${arch}_${seed}_on.log 2>&1; done
  python3 -u exp9_pipeline.py llama $seed off > /tmp/bg_exp9_llama_${seed}_off.log 2>&1
  echo "seed $seed done" >> /tmp/bg_exp9_progress.txt
done
echo ALLDONE >> /tmp/bg_exp9_progress.txt
