#!/usr/bin/env bash
# Re-run sp500 with the positive-output floor + a proper common-date window (min_common=3000 -> 457 nodes,
# ~299 test dates) and a large batch for speed. Overwrites the degenerate 34-test-date sp500 results.
set -u
cd /c/luanvan/stock_vol_prediction01
PY=".venv_gpu_encode/Scripts/python.exe"
RX="baselines/2026-08-21_har_anchored_residual/code/run_experiment.py"
LOG="results/har_anchored/logs"
mkdir -p "$LOG"
export PYTHONIOENCODING=utf-8
for h in 1 5 10 22; do
  echo "=== $(date +%H:%M:%S) sp500 h$h (min_common=3000, batch=32) ==="
  "$PY" "$RX" sp500 "$h" --data-root data/processed --batch 32 --min-common 3000 \
      > "$LOG/sp500_h${h}_fixed.log" 2>&1
  echo "   exit=$? $(grep -h '\[result\]' "$LOG/sp500_h${h}_fixed.log" | tail -1)"
done
echo "=== SP500 FIXED DONE $(date +%H:%M:%S) ==="
