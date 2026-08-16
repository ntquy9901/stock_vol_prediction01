"""Fairness baseline HAR-X: linear regression on ALL 5 node features (parkinson, har_weekly,
har_monthly, market_pk, volume_zscore) — the augmented-HAR (HAR-X) linear model. The classical HAR
(run_e0) uses only the first 3 (own-history) features. Comparing lstm_only (5 features, nonlinear)
against BOTH isolates the extra-feature contribution (HAR-X vs HAR) from the LSTM nonlinearity
(lstm_only vs HAR-X). Uses the same train-only target scaler and the same positivity floor as run_e0.

Run: python <.../code/harx_report.py> <TS_qlike_sweep> [seeds_csv] [horizons...]
Reports, per horizon: HAR / HAR-X / lstm_only / FULL test QLIKE + DM(HAR-X vs HAR),
DM(lstm_only vs HAR-X), DM(lstm_only vs HAR).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

CODE = Path(__file__).resolve().parent
_ROOT = CODE.resolve().parents[2]
for _p in (CODE, _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code",
           _ROOT / "baselines" / "2026-08-11_eda_gnn_baseline" / "code",
           _ROOT / "baselines" / "2026-08-14_pooled_news_edanode_gnn" / "code", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402


def _linear_norm_predictions(x_train: np.ndarray, y_train_norm: np.ndarray,
                             x_test: np.ndarray) -> np.ndarray:
    """Ordinary least-squares fit on the training features, predicted on the test features."""
    model = LinearRegression().fit(np.asarray(x_train, dtype=float), np.asarray(y_train_norm, dtype=float))
    return model.predict(np.asarray(x_test, dtype=float))


def harx_records(allowed, test_samples, store, width: int) -> list[dict[str, Any]]:
    """Fit a width-feature linear model (last-timestep features, normalized target) on `allowed`
    (train), predict on `test_samples`, apply the shared positivity floor. Returns eval records."""
    from eda_ladder import _floor_norm_records, _slice_samples  # read-only import

    train = _slice_samples(allowed, width)
    x_train = np.asarray([s.x_price_raw[-1] for s in train], dtype=float)
    y_train = np.asarray([
        store.get(s.key.ticker_id).target_scaler.transform(
            np.asarray([s.y_model_raw if s.y_model_raw is not None else s.y_raw]))[0]
        for s in train], dtype=float)
    sliced = _slice_samples(test_samples, width)
    x_test = np.asarray([s.x_price_raw[-1] for s in sliced], dtype=float)
    preds = _linear_norm_predictions(x_train, y_train, x_test)
    records = [{"ticker_id": s.key.ticker_id, "target_date": s.key.target_date,
                "prediction_norm": float(p),
                "target_raw": float(s.y_eval_raw if s.y_eval_raw is not None else s.y_raw)}
               for s, p in zip(sliced, preds, strict=True)]
    _floor_norm_records(records, store)
    return records


def _raw_by_key(records, store) -> dict[tuple[int, str], tuple[float, float]]:
    out = {}
    for r in records:
        scaler = store.get(int(r["ticker_id"])).target_scaler
        raw = r["prediction_norm"] * float(scaler.std[0]) + float(scaler.mean[0])
        out[(int(r["ticker_id"]), str(r["target_date"]))] = (float(r["target_raw"]), float(raw))
    return out


def main(ts: str, seeds, horizons=(1, 5, 10, 22)) -> None:  # pragma: no cover
    import combo_ladder
    import dm_report as dm
    from combo_ladder import build_basis
    from diebold_mariano import diebold_mariano
    from run_volatility import ROOT

    dm.RESULTS = ROOT / "results"
    print(f"HAR-X fairness (seeds={list(seeds)}). 3-seed-ensemble deep dumps; HAR/HAR-X deterministic.")
    print(f"{'h':>3} {'HAR':>7s} {'HAR-X':>7s} {'lstm_only':>9s} {'FULL':>7s} | "
          f"{'HARX vs HAR':>16s} {'lstm vs HARX':>16s}")
    for h in horizons:
        combo_ladder.HORIZON = h
        pooled, _graph, store, allowed, _ = build_basis(ROOT / "results" / f"_harx_tmp_h{h}")
        harx = _raw_by_key(harx_records(allowed, pooled.samples["test"], store, 5), store)
        keys = sorted(harx)
        tgt = np.array([harx[k][0] for k in keys])
        px = np.array([harx[k][1] for k in keys])
        def q(pred):
            return float(np.mean(dm._qlike(tgt, pred)))
        # deep + HAR raw preds, seed-ensembled, aligned to `keys`
        def ens(rung):
            kk, _tt, pp = dm._ensemble(ts, h, rung, seeds)
            m = dict(zip(kk, pp))
            return np.array([m[k] for k in keys])
        har = ens("HAR")
        lstm = ens("LSTM_only")
        full = ens("FULL")
        def dmp(a, b):
            r = diebold_mariano(dm._qlike(tgt, a), dm._qlike(tgt, b), h=h)
            s = "*" if r.p_value < 0.05 else ""
            fav = "A" if r.mean_diff < 0 else ("B" if r.mean_diff > 0 else "tie")
            return f"{r.dm_hln:+.2f}p{r.p_value:.2f}{s}({fav})"
        print(f"{h:>3} {q(har):7.4f} {q(px):7.4f} {q(lstm):9.4f} {q(full):7.4f} | "
              f"{dmp(px, har):>16s} {dmp(lstm, px):>16s}")
    print("A=first model (lower loss favours A). HARXvsHAR: A=HAR-X. lstmVsHARX: A=lstm_only.")


if __name__ == "__main__":  # pragma: no cover
    _ts = sys.argv[1]
    _seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [42, 123, 2026]
    _hz = tuple(int(x) for x in sys.argv[3:]) if len(sys.argv) > 3 else (1, 5, 10, 22)
    main(_ts, _seeds, _hz)
