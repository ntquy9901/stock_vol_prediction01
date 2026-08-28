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
    # R-12: a per-seed block that lacks a model / a finite QLIKE is NOT complete
    rp.write_text(json.dumps({"metrics_per_seed": {"LSTM": {}}}))
    assert YZ._done(rp) is False                      # missing LSTM_wGAT_vol2pk + qlike -> not done
    rp.write_text(json.dumps({"metrics_per_seed": {"LSTM": {"qlike": None},
                                                   "LSTM_wGAT_vol2pk": {"qlike": 0.5}}}))
    assert YZ._done(rp) is False                      # null qlike -> not done
    rp.write_text(json.dumps({"metrics_per_seed": {"LSTM": "x",
                                                   "LSTM_wGAT_vol2pk": {"qlike": 0.5}}}))
    assert YZ._done(rp) is False                      # a model block that is not a dict -> not done
    rp.write_text(json.dumps({"metrics_per_seed": {"LSTM": {"qlike": 0.7},
                                                   "LSTM_wGAT_vol2pk": {"qlike": 0.5}}}))
    assert YZ._done(rp) is True                       # both learned models with finite qlike -> done


def test_done_survives_corrupt_json(tmp_path):
    rp = tmp_path / "bad.json"
    rp.write_text("{not json")
    assert YZ._done(rp) is False
