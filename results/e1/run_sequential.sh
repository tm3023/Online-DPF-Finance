#!/usr/bin/env bash
# E1: run the three methods one after another (parallel runs contend for the GPU and are ~2x slower overall).
cd "$(dirname "$0")/../../OL-DPF-gaussian" || exit 1
for t in pretrain online supervised; do
  echo "start $t $(date)" >> ../results/e1/raw_logs/sequence.log
  ../.venv/Scripts/python -u test_losses.py --resampler_type normal --device cuda --trainType $t \
    --NF-cond --measurement CRNVP --NF-dyn --num_dim 2 --num_experiments 10 \
    > ../results/e1/raw_logs/dim2_$t.log 2>&1
  echo "end $t exit=$? $(date)" >> ../results/e1/raw_logs/sequence.log
done
