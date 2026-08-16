#!/usr/bin/env bash
cd "$(dirname "$0")"
PY=python3; command -v python3 >/dev/null 2>&1 || PY=python; [ -x .venv/bin/python ] && PY=.venv/bin/python
echo "== RUN INFERENCE - final model G1 on test split =="
"$PY" reproduce.py infer
