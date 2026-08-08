#!/usr/bin/env bash
# Variant 2: FULL / per-ticker-gate (train_per_ticker_gate.py). MAX_EPOCHS=10 per invocation,
# so reach 40 from 20 via two chained +10 resumes (20->30, 30->40), mirroring how the original
# 20 was built as 10+10. Each seed resumes from its 20-ep checkpoint, then from the new 30-ep one.
set -e
cd /c/luanvan/stock_vol_prediction01
export PYTHONUNBUFFERED=1
S="baselines/2026-07-26_per_ticker_news_gate_baseline/code/train_per_ticker_gate.py"

newest_results() { ls -dt results/per_ticker_gate_2026-08-06_* | head -1; }
newest_models()  { ls -dt models/per_ticker_gate_2026-08-06_*  | head -1; }

resume_two_steps () {
  local seed=$1 ckpt20=$2 rdir20=$3
  echo "=== FULL seed $seed step 1: 20->30 ==="
  python $S --epochs 10 --seed $seed --resume_checkpoint "$ckpt20" --resume_results_dir "$rdir20"
  local rdir30; rdir30=$(newest_results)
  local mdir30; mdir30=$(newest_models)
  echo "=== FULL seed $seed step 2: 30->40 (from $rdir30) ==="
  python $S --epochs 10 --seed $seed --resume_checkpoint "$mdir30/best.pt" --resume_results_dir "$rdir30"
  echo "=== FULL seed $seed final results dir: $(newest_results) ==="
}

resume_two_steps 42   models/per_ticker_gate_2026-08-03_230821/best.pt results/per_ticker_gate_2026-08-03_230821
resume_two_steps 123  models/per_ticker_gate_2026-08-04_000448/best.pt results/per_ticker_gate_2026-08-04_000448
resume_two_steps 2026 models/per_ticker_gate_2026-08-04_002252/best.pt results/per_ticker_gate_2026-08-04_002252

echo "FULL_ALL_DONE"
