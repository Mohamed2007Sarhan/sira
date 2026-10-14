#!/bin/bash
# Exp5 replication with seeds 2,3,4 (same protocol as seeds 0,1): RFT, labelled-only control, no-verifier control, GRPO lr 1e-4
cd /home/claude/sira_project/experiments
export NLAB=1000 STEPS0=1500 ROUNDS=4
for seed in 2 3 4; do
  for md in rft labelled nofilter; do python3 -u exp5_selfimprove.py $md $seed > /tmp/bg_exp5_${md}_${seed}.log 2>&1; done
  GLR=1e-4 python3 -u exp5_selfimprove.py grpo $seed > /tmp/bg_exp5_grpo_${seed}.log 2>&1
  echo "seed $seed done" >> /tmp/bg_exp5_progress.txt
done
echo ALLDONE >> /tmp/bg_exp5_progress.txt
