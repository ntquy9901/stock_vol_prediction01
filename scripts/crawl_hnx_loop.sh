#!/usr/bin/env bash
# Parallel HNX-only crawl (resume-safe re-launch loop; vnstock rate-guard hard-kills the process).
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv_vnstock/Scripts/python.exe
HNX=data/raw/prices/hnx_vnstock
stall=0
for i in $(seq 1 80); do
  before=$(ls "$HNX"/*_ohlcv.csv 2>/dev/null | wc -l)
  echo "[hnx-loop] iter $i start: $before HNX files ($(date))"
  PYTHONIOENCODING=utf-8 "$PY" scripts/crawl_hose_hnx.py --exchange HNX
  after=$(ls "$HNX"/*_ohlcv.csv 2>/dev/null | wc -l)
  echo "[hnx-loop] iter $i end: $after HNX files"
  if [ "$after" -le "$before" ]; then stall=$((stall+1)); else stall=0; fi
  [ "$stall" -ge 4 ] && { echo "[hnx-loop] stalled 4x; stopping."; break; }
  sleep 65
done
echo "[hnx-loop] DONE: $(ls "$HNX"/*_ohlcv.csv 2>/dev/null | wc -l) HNX files"
