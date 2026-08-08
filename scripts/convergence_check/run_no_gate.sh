#!/usr/bin/env bash
# Variant 4: no-gate always-on news fusion (train_dual_news.py). Resume each 20-ep seed +20 -> 40.
set -e
cd /c/luanvan/stock_vol_prediction01
export PYTHONUNBUFFERED=1
S="baselines/2026-07-25_dual_group_news_embedding_baseline/code/train_dual_news.py"

python $S --epochs 20 --seed 42 --resume_start_epoch 20 \
  --resume_checkpoint models/dual_group_news_2026-08-05_230040/best.pt

python $S --epochs 20 --seed 123 --resume_start_epoch 20 \
  --resume_checkpoint models/dual_group_news_2026-08-05_231746/best.pt

python $S --epochs 20 --seed 2026 --resume_start_epoch 20 \
  --resume_checkpoint models/dual_group_news_2026-08-05_233438/best.pt

echo "NO_GATE_ALL_DONE"
