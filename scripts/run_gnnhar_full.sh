#!/usr/bin/env bash
# Sequential (single-GPU) full run for GNNHAR P1/P2: MSE-2hop (ref), QLIKE-2hop (P1), QLIKE-1hop (P2)
# across horizons {1,5,10,22} x seeds {42,123,2026}, 15-epoch cap (early-stop patience 3, min 6).
# One TS base; results dirs: volatility_ablation_h{h}_seed{s}_{TS}[_qlike][_gat1].
set -uo pipefail
ROOT="C:/luanvan/stock_vol_prediction01"
cd "$ROOT" || exit 1
PY="$ROOT/.venv_gpu_encode/Scripts/python.exe"
RUN="$ROOT/baselines/2026-08-15_volatility/code/run_ablation.py"
export PYTHONIOENCODING=utf-8
TS="$1"                       # base timestamp (passed in so it is stable)
SEEDS=(42 123 2026)
HZ="1 5 10 22"
LOG="/tmp/gnnhar_full_progress.log"
echo "START $(date) TS=$TS" > "$LOG"

run_one() {   # $1=loss $2=gat_layers $3=seed $4=tag
  local loss="$1" gat="$2" seed="$3" tag="$4"
  echo "[$(date +%H:%M:%S)] BEGIN $tag seed=$seed loss=$loss gat=$gat" >> "$LOG"
  ABLATION_LOSS="$loss" ABLATION_GAT_LAYERS="$gat" "$PY" "$RUN" "$TS" cuda "$seed" 15 $HZ \
      > "/tmp/gnnhar_${tag}_seed${seed}.log" 2>&1
  echo "[$(date +%H:%M:%S)] END   $tag seed=$seed rc=$?" >> "$LOG"
}

for s in "${SEEDS[@]}"; do run_one mse   2 "$s" mse2hop  ; done   # reference
for s in "${SEEDS[@]}"; do run_one qlike 2 "$s" qlike2hop; done   # P1
for s in "${SEEDS[@]}"; do run_one qlike 1 "$s" qlike1hop; done   # P2 depth
echo "ALL DONE $(date)" >> "$LOG"
