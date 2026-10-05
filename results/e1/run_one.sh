#!/usr/bin/env bash
# E1: run one trainType (10 seeds, d=2), checkpointing each seed.
# Usage: bash results/e1/run_one.sh online|supervised [seed_start]
t=$1
s=${2:-0}
cd "$(dirname "$0")/../../OL-DPF-gaussian" || exit 1
echo "start $t seeds ${s}..9 $(date)" >> ../results/e1/raw_logs/sequence.log
../.venv/Scripts/python -u test_losses.py --resampler_type normal --device cuda --trainType $t \
  --NF-cond --measurement CRNVP --NF-dyn --num_dim 2 --num_experiments 10 --seed_start $s \
  >> ../results/e1/raw_logs/dim2_${t}.log 2>&1
echo "end $t exit=$? $(date)" >> ../results/e1/raw_logs/sequence.log
