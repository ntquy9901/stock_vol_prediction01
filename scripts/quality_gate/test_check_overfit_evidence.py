"""Tests for the pre-push over/under-fit evidence gate (scripts/quality_gate/check_overfit_evidence.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_overfit_evidence as CO  # noqa: E402


def _good():
    return {
        "metrics":       {"HAR": {"qlike": 0.6, "r2": 0.3}, "LSTM": {"qlike": 0.71, "r2": 0.23},
                          "LSTM_wGAT_vol2pk": {"qlike": 0.66, "r2": 0.24}},
        "train_metrics": {"LSTM": {"qlike": 0.68, "r2": 0.30}, "LSTM_wGAT_vol2pk": {"qlike": 0.66, "r2": 0.31}},
        "val_metrics":   {"LSTM": {"qlike": 0.70, "r2": 0.24}, "LSTM_wGAT_vol2pk": {"qlike": 0.67, "r2": 0.25}},
    }


def test_gate_passes_clean_result(tmp_path):
    p = tmp_path / "vn30_h1_result.json"; p.write_text(json.dumps(_good()), encoding="utf-8")
    assert CO.check_files([str(p)]) == {}
    assert CO.main([str(p)]) == 0


def test_gate_blocks_result_without_evidence(tmp_path):
    r = _good(); del r["train_metrics"]
    p = tmp_path / "r.json"; p.write_text(json.dumps(r), encoding="utf-8")
    probs = CO.check_files([str(p)])
    assert str(p) in probs and any("missing train_metrics" in x for x in probs[str(p)])
    assert CO.main([str(p)]) == 1


def test_gate_blocks_overfit_result(tmp_path):
    r = _good(); r["metrics"]["LSTM"] = {"qlike": 1.10, "r2": 0.05}   # test QLIKE 57% worse than val
    p = tmp_path / "r.json"; p.write_text(json.dumps(r), encoding="utf-8")
    probs = CO.check_files([str(p)])
    assert any("overfit" in x for x in probs[str(p)])


def test_gate_skips_non_training_result(tmp_path):
    # a result.json without learned-model metrics (e.g. a GARCH-only or unrelated artifact) is skipped
    p = tmp_path / "other.json"; p.write_text(json.dumps({"metrics": {"HAR-X": {"qlike": 0.5, "r2": 0.3}}}), encoding="utf-8")
    assert CO.check_files([str(p)]) == {}


def test_gate_blocks_partial_masked_rich_missing_a_learned_model(tmp_path):
    # F-04 (v3): a masked-rich result (identified by design/per-seed) that LOST its LSTM block must FAIL,
    # not be skipped as "not a training result".
    r = _good()
    r["design"] = "masked-union-panel-rich-5feat-weighted-gat"
    del r["metrics"]["LSTM"]; del r["train_metrics"]["LSTM"]; del r["val_metrics"]["LSTM"]
    p = tmp_path / "vn30_h1_result.json"; p.write_text(json.dumps(r), encoding="utf-8")
    probs = CO.check_files([str(p)])
    assert str(p) in probs and any("LSTM" in x for x in probs[str(p)])


def test_masked_rich_detector_recognises_design_and_per_seed():
    assert CO._is_masked_rich_result({"design": "masked-union-panel", "metrics": {}}) is True
    assert CO._is_masked_rich_result({"metrics_per_seed": {"LSTM": {}}}) is True
    assert CO._is_masked_rich_result({"metrics": {"LSTM_wGAT_vol2pk": {"qlike": 0.6}}}) is True
    assert CO._is_masked_rich_result({"metrics": {"HAR-X": {"qlike": 0.5}}}) is False   # deterministic-only
    assert CO._is_masked_rich_result("not-a-dict") is False                              # non-dict guard
    assert CO._is_masked_rich_result({}) is False                                        # empty artifact


def test_gate_flags_unreadable_file(tmp_path):
    p = tmp_path / "bad.json"; p.write_text("{not json", encoding="utf-8")
    probs = CO.check_files([str(p)])
    assert str(p) in probs and any("unreadable" in x for x in probs[str(p)])
