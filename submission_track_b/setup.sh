#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m venv .venv || python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo "Setup complete. Run ./START_HERE.sh"
