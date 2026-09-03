import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_pooled_ablation as rp  # noqa: E402


def test_paired_dm_and_did_wiring(monkeypatch, tmp_path):
    keys = [(0, "2020-01-01"), (1, "2020-01-02")]

    def fake_arm(panel, folds, wf, cfg, train_idx, score_idx):
        base = 0.5 if len(train_idx) > 5 else 0.6   # pooled (more train nodes) better
        preds = {m: {k: (1.0, base) for k in keys}
                 for m in ("HAR", "HAR-X", "LSTM", "LSTM_wGAT_vol2pk")}
        return {"metrics": {m: {"qlike": base, "n": 2} for m in preds}, "preds": preds,
                "seed_stats": {}, "per_fold": []}

    class Panel:
        N = 10
        target_dates = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")

    class WF:
        horizon = 1
        lookback = 22

    class Cfg:
        qlike_floor = 1e-8
        seeds = (0,)

    def fake_build(h, ft, epochs=16, lookback=22):
        return (Panel(), [0, 1], WF(), Cfg(), np.arange(10), np.arange(3))

    monkeypatch.setattr(rp.ra, "run_arm", fake_arm)
    monkeypatch.setattr(rp, "_build", fake_build)
    monkeypatch.setattr(rp, "assert_no_leakage", lambda *a, **k: None)
    out = rp.run_ablation(1, out=tmp_path / "h1.json")
    assert "LSTM" in out["paired_dm"] and "LSTM_wGAT_vol2pk" in out["paired_dm"]
    assert set(out["diff_in_diff"]) == {"LSTM", "LSTM_wGAT_vol2pk"}
    assert (tmp_path / "h1.json").exists()
