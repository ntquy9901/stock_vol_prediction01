#!/usr/bin/env bash
# 15-epoch x 3-seed seq-lookback run for seq in {5,10}, all horizons, concurrency-capped.
# Single GPU is batch=1 / ~idle, so several concurrent processes overlap on the GPU while using
# separate CPU cores for the CPU-heavy basis builds. Each (seq,seed) writes its own result dirs
# (seed is in the dir name), so there is no cross-process shared state.
set -uo pipefail
cd /c/luanvan/stock_vol_prediction01
GPUPY=.venv_gpu_encode/Scripts/python.exe
TS="${1:?usage: run_15ep_3seed.sh <TS> [MAX]}"   # base stamp, e.g. 2026-08-18_2310
MAX="${2:-4}"                                     # max concurrent processes
case "$MAX" in ''|*[!0-9]*) echo "MAX must be a positive integer, got '$MAX'" >&2; exit 2;; esac
[ "$MAX" -ge 1 ] || { echo "MAX must be >= 1, got '$MAX'" >&2; exit 2; }
SEQS=(5 10)
SEEDS=(42 123 2026)
mkdir -p scripts/seq_lookback/logs15
echo "=== 15ep/3seed start $(date) TS=$TS MAX=$MAX ==="
declare -A JOB_PID   # name -> pid, so we can collect each job's REAL exit status
for seq in "${SEQS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    name="seq${seq}_seed${seed}"
    log="scripts/seq_lookback/logs15/${name}.log"
    (
      echo ">>> $name START $(date)"
      PYTHONIOENCODING=utf-8 "$GPUPY" scripts/seq_lookback/run_seq.py \
        "$seq" "${TS}_15ep_seq${seq}" cuda "$seed" 15 1 5 10 22
      rc=$?
      echo ">>> $name DONE rc=$rc $(date)"
      exit "$rc"        # propagate python's status as the subshell status (not the trailing echo)
    ) > "$log" 2>&1 &
    JOB_PID["$name"]=$!
    # concurrency gate: cap peak running jobs at MAX
    while [ "$(jobs -r | wc -l)" -ge "$MAX" ]; do sleep 5; done
  done
done
# collect real per-job statuses instead of a blanket `wait`
FAILS=0
for name in "${!JOB_PID[@]}"; do
  if ! wait "${JOB_PID[$name]}"; then
    echo "!!! JOB FAILED: $name (see logs15/${name}.log)" >&2
    FAILS=$((FAILS + 1))
  fi
done
if [ "$FAILS" -gt 0 ]; then
  echo "=== 15ep/3seed FINISHED WITH $FAILS FAILED JOB(S) $(date) ===" >&2
  exit 1
fi
echo "=== 15ep/3seed ALL DONE (all jobs rc=0) $(date) ==="
