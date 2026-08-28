#!/usr/bin/env bash
# Resilient wrapper for the HOSE/HNX crawl. vnstock's guest-tier rate guard HARD-TERMINATES the
# process on breach (uncatchable), so we re-launch until done. The crawl is resume-safe (skips
# existing CSVs), so each restart continues where the last was killed. Between restarts we wait 65s
# to let the 60s rate window reset. Stops when all tickers are present or progress stalls 3x.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv_vnstock/Scripts/python.exe
HOSE=data/raw/prices/hose_vnstock
EXPECTED=405   # HOSE equities only (from Listing 2026-08-22). HNX crawled by a separate process.

count() { ls "$HOSE"/*_ohlcv.csv 2>/dev/null | wc -l; }

stall=0
for i in $(seq 1 60); do
  before=$(count)
  echo "[loop] iter $i start: $before/$EXPECTED files present ($(date))"
  PYTHONIOENCODING=utf-8 "$PY" scripts/crawl_hose_hnx.py --exchange HOSE
  after=$(count)
  echo "[loop] iter $i end: $after/$EXPECTED files present"
  if [ "$after" -ge "$EXPECTED" ]; then
    echo "[loop] all tickers present; done."
    break
  fi
  if [ "$after" -le "$before" ]; then
    stall=$((stall+1))
    echo "[loop] no progress this iter (stall=$stall)"
    if [ "$stall" -ge 3 ]; then
      echo "[loop] progress stalled 3x; remaining $((EXPECTED-after)) are likely genuine failures. Stopping."
      break
    fi
  else
    stall=0
  fi
  echo "[loop] sleeping 65s to reset rate window..."
  sleep 65
done
# Final manifest refresh (script writes manifests on each run; this ensures a clean last pass).
echo "[loop] finished at $(date)"
