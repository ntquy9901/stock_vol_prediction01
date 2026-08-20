#!/usr/bin/env bash
# One-command reproduce for reviewers: install deps, run tests, then the full experiment suite.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
echo "== install =="; $PY -m pip install -r requirements.txt
echo "== tests ==";   $PY -m pytest tests -q
echo "== suite =="
for cfg in "vn30 10 1" "vn30 10 5" "vn30 22 1" "vn30 22 5" "vn100 10 1" "vn100 10 5"; do
  set -- $cfg
  echo ">>> $1 lb$2 h$3"
  $PY run_all.py "$1" "$2" "$3" --data-root data
done
echo "== done: see results/soict/*/result.json =="
