#!/usr/bin/env bash
# Full HAR-anchored grid: E0-E10 x 5 seeds x {h1,h5,h10,h22} for vn30, vn100, sp500.
# Sequential (no GPU self-contention). Logs per config under results/har_anchored/logs/.
set -u
cd /c/luanvan/stock_vol_prediction01
PY=".venv_gpu_encode/Scripts/python.exe"
RX="baselines/2026-08-21_har_anchored_residual/code/run_experiment.py"
LOG="results/har_anchored/logs"
mkdir -p "$LOG"
export PYTHONIOENCODING=utf-8

run() {  # dataset horizon extra...
  local ds=$1 h=$2; shift 2
  echo "=== $(date +%H:%M:%S) $ds h$h $* ==="
  "$PY" "$RX" "$ds" "$h" "$@" > "$LOG/${ds}_h${h}.log" 2>&1
  echo "   exit=$? $(grep -h '\[result\]' "$LOG/${ds}_h${h}.log" | tail -1)"
}

for h in 1 5 10 22; do run vn30  "$h"; done
for h in 1 5 10 22; do run vn100 "$h"; done
for h in 1 5 10 22; do run sp500 "$h" --data-root data/processed --batch 8; done
echo "=== GRID DONE $(date +%H:%M:%S) ==="
