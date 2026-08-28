"""Smoke test for the volatility-proxy robustness driver (scripts/eda/run_yz_robustness.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_yz_robustness as YZ  # noqa: E402


def test_done_requires_metrics_per_seed(tmp_path):
    rp = tmp_path / "r.json"
    assert YZ._done(rp) is False                      # missing file
    rp.write_text(json.dumps({"metrics": {}}))
    assert YZ._done(rp) is False                      # no metrics_per_seed -> not done
    rp.write_text(json.dumps({"metrics_per_seed": {"LSTM": {}}}))
    assert YZ._done(rp) is True                       # has metrics_per_seed -> done


def test_done_survives_corrupt_json(tmp_path):
    rp = tmp_path / "bad.json"
    rp.write_text("{not json")
    assert YZ._done(rp) is False
