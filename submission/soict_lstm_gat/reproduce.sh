#!/usr/bin/env bash
# One-command reproduce for reviewers: install deps, run tests, then the experiment suites.
#   MAIN model  = per-observation LSTM  (run_lstm.py)  -> results/soict_perobs/*/result.json
#   Ablation    = HAR-LSTM-GAT graph    (run_all.py)   -> results/soict/*/result.json
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
echo "== install =="; $PY -m pip install -r requirements.txt
echo "== tests ==";   $PY -m pytest tests -q

echo "== MAIN model: per-observation LSTM vs HAR + GARCH (run_lstm.py) =="
for cfg in "vn30 10 1" "vn30 10 5" "vn100 10 1" "vn100 10 5"; do
  set -- $cfg
  echo ">>> LSTM $1 lb$2 h$3"
  $PY run_lstm.py "$1" "$2" "$3" --data-root data
done

echo "== Ablation: HAR-LSTM-GAT graph vs LSTM(w/o GAT) (run_all.py) =="
for cfg in "vn30 10 1" "vn30 10 5" "vn30 22 1" "vn30 22 5" "vn100 10 1" "vn100 10 5"; do
  set -- $cfg
  echo ">>> GAT $1 lb$2 h$3"
  $PY run_all.py "$1" "$2" "$3" --data-root data
done

echo "== done =="
echo "  MAIN model results:  results/soict_perobs/*/result.json"
echo "  Graph ablation:      results/soict/*/result.json"
echo "  (S&P500 is not shipped — see REPRODUCE.md to regenerate, then rerun both scripts on sp500.)"
