"""Tests for the over/under-fit fit-verdict logic (scripts/quality_gate/overfit_check.py)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overfit_check as OC  # noqa: E402


def test_classify_ok_when_val_and_test_agree():
    m = {"qlike": 0.70, "r2": 0.23}
    v = OC.classify_fit({"qlike": 0.68, "r2": 0.30}, m, {"qlike": 0.71, "r2": 0.22})
    assert v["status"] == "ok" and not v["reasons"]


def test_classify_overfit_on_val_test_qlike_gap():
    # test QLIKE 40% worse than val -> overfit
    v = OC.classify_fit({"qlike": 0.5, "r2": 0.6}, {"qlike": 0.5, "r2": 0.4}, {"qlike": 0.7, "r2": 0.2})
    assert v["status"] == "overfit"
    assert v["val_test_qlike_gap_rel"] > 0.25


def test_classify_overfit_on_r2_drop():
    # val->test qlike fine, but train R2 0.9 vs test 0.1 -> big generalization gap
    v = OC.classify_fit({"qlike": 0.5, "r2": 0.9}, {"qlike": 0.7, "r2": 0.15}, {"qlike": 0.72, "r2": 0.1})
    assert v["status"] == "overfit" and v["train_test_r2_drop"] > 0.2


def test_classify_underfit_when_train_r2_poor():
    # cannot fit train (and test also poor) -> underfit, takes precedence over any gap
    v = OC.classify_fit({"qlike": 2.0, "r2": -0.1}, {"qlike": 2.1, "r2": -0.2}, {"qlike": 2.2, "r2": -0.3})
    assert v["status"] == "underfit"


def test_classify_unknown_on_missing_metrics():
    v = OC.classify_fit({"qlike": 0.5}, {"qlike": 0.5, "r2": 0.4}, {"qlike": 0.5, "r2": 0.4})
    assert v["status"] == "unknown"


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_classify_nonfinite_metric_is_unknown_not_ok(split, bad):
    # F-03 (v3): a NaN/inf in any split must NOT pass as 'ok' (would be fail-open).
    m = {"train": {"qlike": 0.68, "r2": 0.30}, "val": {"qlike": 0.70, "r2": 0.24},
         "test": {"qlike": 0.71, "r2": 0.23}}
    m[split]["qlike"] = bad
    v = OC.classify_fit(m["train"], m["val"], m["test"])
    assert v["status"] == "unknown" and any("not finite" in r for r in v["reasons"])


def test_check_result_evidence_missing_blocks():
    ok, probs = OC.check_result_evidence({"metrics": {"LSTM": {"qlike": 0.7, "r2": 0.2}}})
    assert ok is False and any("missing train_metrics" in p for p in probs)


def test_check_result_evidence_flags_model_missing_from_a_block():
    # blocks present but a learned model is absent from train_metrics -> reported per-model, not a crash
    res = {
        "train_metrics": {"LSTM_wGAT_vol2pk": {"qlike": 0.66, "r2": 0.31}},   # LSTM missing here
        "val_metrics":   {"LSTM": {"qlike": 0.70, "r2": 0.24}, "LSTM_wGAT_vol2pk": {"qlike": 0.67, "r2": 0.25}},
        "metrics":       {"LSTM": {"qlike": 0.71, "r2": 0.23}, "LSTM_wGAT_vol2pk": {"qlike": 0.66, "r2": 0.24}},
    }
    ok, probs = OC.check_result_evidence(res)
    assert ok is False and any("LSTM: missing train/val/test metrics" in p for p in probs)


def test_check_result_evidence_passes_clean_and_flags_overfit():
    good = {
        "train_metrics": {"LSTM": {"qlike": 0.68, "r2": 0.30}, "LSTM_wGAT_vol2pk": {"qlike": 0.66, "r2": 0.31}},
        "val_metrics":   {"LSTM": {"qlike": 0.70, "r2": 0.24}, "LSTM_wGAT_vol2pk": {"qlike": 0.67, "r2": 0.25}},
        "metrics":       {"LSTM": {"qlike": 0.71, "r2": 0.23}, "LSTM_wGAT_vol2pk": {"qlike": 0.66, "r2": 0.24}},
    }
    ok, probs = OC.check_result_evidence(good)
    assert ok is True and probs == []

    bad = {k: {m: dict(v) for m, v in blk.items()} for k, blk in good.items()}
    bad["metrics"]["LSTM"] = {"qlike": 1.10, "r2": 0.05}   # test QLIKE 57% worse than val -> overfit
    ok2, probs2 = OC.check_result_evidence(bad)
    assert ok2 is False and any("LSTM: overfit" in p for p in probs2)
