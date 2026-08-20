#!/usr/bin/env bash
# Run the full SOICT experiment suite as concurrency-capped background processes.
set -uo pipefail
cd /c/luanvan/stock_vol_prediction01
GPUPY=.venv_gpu_encode/Scripts/python.exe
MAX=3
mkdir -p submission/soict_lstm_gat/logs
# config = "<dataset> <lookback> <horizon>"
CONFIGS=(
  "vn30 10 1" "vn30 10 5"       # main
  "vn30 22 1" "vn30 22 5"       # variation: lookback 22
  "vn100 10 1" "vn100 10 5"     # variation: VN100
  "sp500 10 1" "sp500 10 5"     # variation: S&P500
)
echo "=== suite start $(date) ==="
declare -A PID
for c in "${CONFIGS[@]}"; do
  set -- $c; ds=$1; lb=$2; h=$3
  name="${ds}_lb${lb}_h${h}"
  log="submission/soict_lstm_gat/logs/${name}.log"
  ( echo ">>> $name START $(date)"
    PYTHONIOENCODING=utf-8 "$GPUPY" submission/soict_lstm_gat/run_all.py "$ds" "$lb" "$h" \
      --data-root data/processed
    echo ">>> $name DONE rc=$? $(date)"
  ) > "$log" 2>&1 &
  PID[$name]=$!
  while [ "$(jobs -r | wc -l)" -ge "$MAX" ]; do sleep 10; done
done
FAILS=0
for name in "${!PID[@]}"; do
  if ! wait "${PID[$name]}"; then echo "!!! FAILED: $name" >&2; FAILS=$((FAILS+1)); fi
done
echo "=== suite done $(date) fails=$FAILS ==="
