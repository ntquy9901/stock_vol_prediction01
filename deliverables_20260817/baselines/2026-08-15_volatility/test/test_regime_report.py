"""P3: regime-split (calm/turbulent) metrics + DM (Zhang et al. 2308.01419 market-regime analysis).

Paper: nonlinear-spillover / QLIKE advantages concentrate on high-volatility days; a pooled average
can hide where a model adds value. Split test observations by realized target volatility (top-frac =
turbulent), recompute metrics + DM per regime. Pure post-hoc analysis on the prediction dumps.
"""
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import regime_report as rr  # noqa: E402


def test_split_regime_top_fraction_by_target():
    targets = np.arange(1.0, 11.0)          # 1..10
    mask = rr.split_regime(targets, turbulent_frac=0.30)
    # top 30% by target volatility -> {8, 9, 10}
    assert mask.tolist() == [False] * 7 + [True] * 3


def test_regime_metrics_computed_on_subset_only():
    target = np.array([1.0, 1.0, 1.0, 10.0])
    pred = np.array([1.0, 1.0, 1.0, 5.0])    # perfect on calm, off on the turbulent point
    mask = np.array([False, False, False, True])
    calm = rr.regime_metrics(target[~mask], pred[~mask])
    turb = rr.regime_metrics(target[mask], pred[mask])
    assert calm["n"] == 3 and abs(calm["mse"]) < 1e-12 and abs(calm["mae"]) < 1e-12
    assert turb["n"] == 1 and abs(turb["mse"] - 25.0) < 1e-9   # (10-5)^2


def test_regime_dm_favors_better_model_in_that_regime():
    rng = np.random.default_rng(0)
    n = 60
    target = np.concatenate([np.full(40, 1.0), np.full(20, 5.0)])
    mask = np.concatenate([np.zeros(40, bool), np.ones(20, bool)])
    # model A ~ perfect on turbulent; model B systematically under-predicts turbulent
    pred_a = target + rng.normal(0, 0.01, n)
    pred_b = target.copy()
    pred_b[mask] = 2.0
    dm_turb = rr.regime_dm(target[mask], pred_a[mask], pred_b[mask], horizon=1, loss="qlike")
    assert dm_turb["favors"] == "A"          # A (lower loss) wins on turbulent days
    assert dm_turb["n"] == 20


def test_split_regime_frac_bounds():
    targets = np.arange(1.0, 11.0)
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            rr.split_regime(targets, turbulent_frac=bad)
            raise AssertionError(f"expected ValueError for frac={bad}")
        except ValueError:
            pass


def _write_dump(base: Path, ts, horizon, seed, dump_dir, rows):
    d = base / f"volatility_ablation_h{horizon}_seed{seed}_{ts}" / dump_dir
    d.mkdir(parents=True, exist_ok=True)
    import json
    (d / "predictions_test.json").write_text(json.dumps(rows), encoding="utf-8")


def test_run_regime_end_to_end(tmp_path, monkeypatch):
    import dm_report
    monkeypatch.setattr(dm_report, "RESULTS", tmp_path)
    rng = np.random.default_rng(0)
    ts, h, seed = "TESTTS", 1, 42
    # 40 obs: 30 calm (target 1) + 10 turbulent (target 10) -> frac 0.25 selects the 10 turbulent.
    # Small noise gives the DM loss-differential nonzero variance (a constant diff is undefined).
    targets = np.array([1.0] * 30 + [10.0] * 10)
    keys = [(t % 3, f"2020-01-{t + 1:02d}") for t in range(40)]
    full_pred = targets + rng.normal(0, 0.05, 40)                       # FULL ~ accurate everywhere
    har_pred = targets + rng.normal(0, 0.05, 40)
    har_pred[targets >= 5] = 2.0 + rng.normal(0, 0.05, 10)             # HAR under-predicts turbulent

    def _rows(pred):
        return [{"ticker_id": k[0], "target_date": k[1], "target_raw": float(tg),
                 "prediction_raw": float(p)} for k, tg, p in zip(keys, targets, pred)]

    _write_dump(tmp_path, ts, h, seed, "FULL", _rows(full_pred))
    _write_dump(tmp_path, ts, h, seed, "P0", _rows(har_pred))

    report = rr.run_regime(ts, h, [seed], comparators=("HAR",), turbulent_frac=0.25)

    assert report["n_total"] == 40 and report["n_turbulent"] == 10
    turb = report["regimes"]["turbulent"]
    assert turb["metrics"]["FULL"]["n"] == 10
    assert turb["metrics"]["FULL"]["mse"] < 0.1                # FULL accurate on turbulent
    assert turb["metrics"]["HAR"]["mse"] > 1.0                 # HAR wrong on turbulent
    assert turb["dm_full_vs"]["HAR"]["favors"] == "A"          # FULL beats HAR on turbulent
    assert report["regimes"]["calm"]["metrics"]["FULL"]["n"] == 30
