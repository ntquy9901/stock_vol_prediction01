"""Data-mine the VN100 train/val/test splits to explain WHY the deep LSTM underperforms the
parsimonious linear HAR-X baseline on multi-horizon Parkinson-variance forecasting.

Scope: DEEP (LSTM, 5 node features, no graph) vs LINEAR (HAR-X, 5-feature OLS) across the three
chronological splits produced by the delivered ``masked_rich`` pipeline. Read-only: it re-uses
``masked_rich.build_masked_rich`` and ``run_masked_rich.train_masked_rich(..., return_splits=True)``
(the delivered runners) and never edits any live-training-path file. The ``parkinson_volatility``
column is a VARIANCE (sigma^2), the QLIKE positivity floor is identical across compared models, and
every scaler / OLS coefficient is fit on TRAIN rows only (no leakage).

The module is split into (a) small PURE analysis functions (unit-tested with synthetic fixtures) and
(b) a heavy driver (``main`` and helpers) that builds the panel, trains the LSTM ensemble on GPU and
renders the HTML/Markdown report. The driver is marked ``# pragma: no cover`` because it needs the
VN100 data and a GPU; the numeric primitives it relies on are the covered, tested functions below.

Hypotheses tested (ranked in the report by evidence strength):
  1. Over-smoothing / spike-miss (primary): the LSTM predicts near the central level (good MAE) but
     under-predicts volatility spikes -> large squared error (MSE/RMSE) and asymmetric QLIKE penalty.
  2. Overfitting (secondary): LSTM train->test generalisation gap vs the stable linear model.
  3. HAR inductive-bias near-optimality: strong target persistence + low signal-to-noise favour the
     high-bias linear model.
  4. Loss-metric mismatch (contributing): LSTM trains on MSE, is scored on QLIKE -- but HAR-X wins
     MSE/RMSE too, so the mismatch is contributing, not the whole story.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------------------------------
# Pure analysis primitives (unit-tested; no data / GPU dependency)
# --------------------------------------------------------------------------------------------------


def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to 1-D float arrays; require equal length, non-empty and finite (fail loud)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"arrays must have the same shape, got {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("arrays must be non-empty")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("arrays must be finite")
    return a, b


def per_obs_qlike(y: np.ndarray, pred: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Per-observation QLIKE ``r - log(r) - 1`` with ``r = clamp(y)/clamp(pred)`` on a single shared
    positivity ``floor``. This mirrors the delivered ``metrics.per_obs_qlike`` (identical formula and
    floor) so the analysis is self-contained yet consistent with the pipeline it mines. QLIKE
    penalises UNDER-prediction of high volatility asymmetrically (``r`` large -> loss grows ~linearly),
    which is why an over-smoothing forecaster is punished in the tail."""
    y, pred = _finite_pair(y, pred)
    if not (np.isfinite(floor) and floor > 0.0):
        raise ValueError(f"floor must be finite and positive, got {floor}")
    r = np.maximum(y, floor) / np.maximum(pred, floor)
    return r - np.log(r) - 1.0


def error_by_magnitude(y: np.ndarray, pred: np.ndarray, n_bins: int = 10,
                       floor: float = 1e-8) -> dict[str, np.ndarray]:
    """Equal-count bins of the observations ordered by TARGET magnitude; per-bin error stats.

    Sorting by ``y`` and splitting into ``n_bins`` equal-count groups isolates the tail: the top
    group is the high-volatility decile where an over-smoothing forecaster incurs its large errors.
    Returns arrays (length ``n_bins``): ``count``, ``mean_target``, ``mean_pred``, ``mse``, ``mae``,
    ``qlike`` (mean per-bin QLIKE) and ``mean_signed_error`` (``mean(pred - y)``; negative =
    under-prediction).
    """
    y, pred = _finite_pair(y, pred)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if n_bins > y.size:
        raise ValueError(f"n_bins ({n_bins}) exceeds number of observations ({y.size})")
    order = np.argsort(y, kind="mergesort")
    ql = per_obs_qlike(y, pred, floor)
    out: dict[str, list[float]] = {"count": [], "mean_target": [], "mean_pred": [], "mse": [],
                                   "mae": [], "qlike": [], "mean_signed_error": []}
    for g in np.array_split(order, n_bins):
        yy, pp = y[g], pred[g]
        out["count"].append(float(len(g)))
        out["mean_target"].append(float(yy.mean()))
        out["mean_pred"].append(float(pp.mean()))
        out["mse"].append(float(np.mean((yy - pp) ** 2)))
        out["mae"].append(float(np.mean(np.abs(yy - pp))))
        out["qlike"].append(float(np.mean(ql[g])))
        out["mean_signed_error"].append(float(np.mean(pp - yy)))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def variance_ratio(pred: np.ndarray, y: np.ndarray) -> float:
    """``var(pred) / var(y)`` (population variance). A value well below 1 is the compression signature
    of an over-smoothing forecaster (its predictions vary less than the target)."""
    pred, y = _finite_pair(pred, y)
    vy = float(np.var(y))
    if vy == 0.0:
        raise ValueError("target variance is zero; variance ratio undefined")
    return float(np.var(pred) / vy)


def signed_bias_top_decile(y: np.ndarray, pred: np.ndarray, q: float = 0.9) -> float:
    """Mean signed error ``mean(pred - y)`` over the observations with ``y >= quantile(y, q)``.

    Negative => the model systematically UNDER-predicts the high-volatility tail (spike-miss).
    """
    y, pred = _finite_pair(y, pred)
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    thr = float(np.quantile(y, q))
    mask = y >= thr
    if not mask.any():  # pragma: no cover - quantile guarantees >=1 obs at/above threshold
        raise ValueError("no observations in the top quantile")
    return float(np.mean(pred[mask] - y[mask]))


def generalization_gap(train_val: float, test_val: float) -> dict[str, float]:
    """Compare a metric between the TRAIN and TEST splits: ``diff = test - train`` and the
    ``ratio = test / train``. A large positive diff/ratio for the deep model (with a near-flat
    linear model) is the overfitting signature."""
    if not (np.isfinite(train_val) and np.isfinite(test_val)):
        raise ValueError("train_val and test_val must be finite")
    ratio = float(test_val / train_val) if train_val != 0.0 else float("inf")
    return {"train": float(train_val), "test": float(test_val),
            "diff": float(test_val - train_val), "ratio": ratio}


def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS with intercept: returns ``[intercept, b_1, ..., b_k]`` via least squares."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"X and y row mismatch: {X.shape[0]} vs {y.shape[0]}")
    design = np.column_stack([np.ones(X.shape[0]), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def _ols_predict(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return float(coef[0]) + np.asarray(X, dtype=float) @ np.asarray(coef, dtype=float)[1:]


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def har_ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    """In-sample R^2 of an OLS fit of ``y`` on ``[1, X]`` (how much of the target the linear HAR
    basis explains where it is fit)."""
    coef = _ols_fit(X, y)
    return _r2(y, _ols_predict(X, coef))


def ols_oos_r2(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray) -> float:
    """Out-of-sample R^2: fit OLS on (X_tr, y_tr), score on (X_te, y_te). Stable in <-> out-of-sample
    R^2 is evidence the linear basis generalises (low variance)."""
    coef = _ols_fit(X_tr, y_tr)
    return _r2(y_te, _ols_predict(X_te, coef))


def signal_to_noise_har(X: np.ndarray, y: np.ndarray) -> float:
    """In-sample HAR signal-to-noise ``R^2 / (1 - R^2)`` (explained / residual variance). A small
    value means most of the target is unforecastable noise -- a regime that favours the high-bias
    linear model over a flexible deep one."""
    r2 = har_ols_r2(X, y)
    if r2 >= 1.0:
        raise ValueError("R^2 >= 1 (perfect fit); signal-to-noise undefined")
    return float(r2 / (1.0 - r2))


def autocorr(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation at lags ``1..max_lag`` of a 1-D series (persistence of the target)."""
    s = np.asarray(series, dtype=float).ravel()
    if s.ndim != 1:  # pragma: no cover - ravel guarantees 1-D; defensive
        raise ValueError("series must be 1-D")
    if not np.isfinite(s).all():
        raise ValueError("series must be finite")
    if max_lag < 1 or max_lag >= s.size:
        raise ValueError(f"max_lag must be in [1, len(series)-1], got {max_lag} for n={s.size}")
    s = s - s.mean()
    denom = float(np.dot(s, s))
    if denom == 0.0:
        raise ValueError("series has zero variance; autocorrelation undefined")
    return np.asarray([float(np.dot(s[k:], s[:-k]) / denom) for k in range(1, max_lag + 1)],
                      dtype=float)


# --------------------------------------------------------------------------------------------------
# Heavy driver: build panel, train LSTM ensemble, render report (needs VN100 data + GPU)
# --------------------------------------------------------------------------------------------------
# The functions below are excluded from coverage: they require the VN100 processed panel and a CUDA
# GPU to train the LSTM, neither of which is available in the unit-test sandbox. Every numeric result
# they compute flows through the tested primitives above.

import base64  # noqa: E402
import io  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
_SUB = REPO / "submission" / "soict_lstm_gat"
_QG = REPO / "scripts" / "quality_gate"
_CODE = REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"
HORIZONS = (1, 5, 10, 22)
LOOKBACK = 10
QLIKE_FLOOR = 1e-8
MODEL_COLORS = {"HAR-X": "#1f77b4", "LSTM": "#d62728", "actual": "#2ca02c"}


def _bootstrap_paths():  # pragma: no cover - import-path shim for the delivered runners
    for p in (str(_SUB), str(_QG), str(_CODE)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _har_predict_split(har3, coef, t_mean, B):
    """HAR prediction on a split, floored at the SHARED per-node positivity floor ``1e-2*t_mean+1e-12``
    (identical to the delivered ``run_masked_rich`` -- floor-mismatch would bias the QLIKE comparison)."""
    hp = B.har_predict(har3.reshape(-1, 3), coef, floor=QLIKE_FLOOR).reshape(har3.shape[:2])
    return np.maximum(hp, 1e-2 * t_mean + 1e-12)


def _harx_predict_split(har5, cx, t_mean):
    """5-feature linear (HAR-X) prediction on a split, floored at the SAME ``1e-2*t_mean+1e-12``."""
    n, N, _ = har5.shape
    flat = np.column_stack([np.ones(n * N), har5.reshape(-1, 5)]) @ cx
    return np.maximum(flat.reshape(n, N), 1e-2 * t_mean + 1e-12)


def _flat(pred2d, y2d, tmask2d):
    """Flatten a split's (pred, target) to the masked (valid-target) observations -> ``(y, pred)``."""
    m = tmask2d.astype(bool)
    return y2d[m], pred2d[m]


def build_horizon(files, price_dir, horizon, cfg, MR, RUN, B):  # pragma: no cover
    """Build one horizon: HAR / HAR-X / LSTM-ensemble predictions on train/val/test (2-D + masks)."""
    D = MR.build_masked_rich(files, price_dir, LOOKBACK, horizon,
                             edge_min_overlap=MR.EDGE_MIN_OVERLAP, top_k=MR.EDGE_TOP_K)
    mtr = D.tmask_tr.astype(bool)
    coef = B.har_fit(D.har_tr[mtr], D.y_tr[mtr])
    xtr = np.column_stack([np.ones(int(mtr.sum())), D.har5_tr[mtr]])
    cx = np.linalg.lstsq(xtr, D.y_tr[mtr], rcond=None)[0]
    har = {s: _har_predict_split(h, coef, D.t_mean, B)
           for s, h in (("train", D.har_tr), ("val", D.har_va), ("test", D.har_te))}
    harx = {s: _harx_predict_split(h, cx, D.t_mean)
            for s, h in (("train", D.har5_tr), ("val", D.har5_va), ("test", D.har5_te))}
    outs = [RUN.train_masked_rich(D, cfg, s, False, D.adj_vol2pk, "zscore_floor", return_splits=True)
            for s in cfg.seeds]
    lstm = {s: np.mean([o[s] for o in outs], axis=0) for s in ("train", "val", "test")}
    curves = {"train": [o["train_curve"] for o in outs], "val": [o["val_curve"] for o in outs],
              "best_epoch": [o["best_epoch"] for o in outs]}
    y2d = {"train": D.y_tr, "val": D.y_va, "test": D.y_te}
    tm = {"train": D.tmask_tr, "val": D.tmask_va, "test": D.tmask_te}
    return {"D": D, "har": har, "harx": harx, "lstm": lstm, "curves": curves, "y2d": y2d,
            "tmask": tm, "coef": coef, "cx": cx}


def _split_metrics(y, p):
    """MSE/RMSE/MAE/QLIKE/R2 on a masked split via the delivered ``metrics`` module (shared QLIKE floor)."""
    from metrics import mae, mse, qlike, r2, rmse  # delivered metrics (submission/soict_lstm_gat)
    return {"mse": mse(y, p), "rmse": rmse(y, p), "mae": mae(y, p),
            "qlike": qlike(y, p, QLIKE_FLOOR), "r2": r2(y, p), "n": int(len(y))}


def analyse_horizon(bundle):
    """All numeric evidence for one horizon, computed via the tested pure primitives."""
    models = {"HAR-X": bundle["harx"], "LSTM": bundle["lstm"]}
    flat = {m: {s: _flat(models[m][s], bundle["y2d"][s], bundle["tmask"][s]) for s in ("train", "val", "test")}
            for m in models}
    metrics = {m: {s: _split_metrics(*flat[m][s]) for s in ("train", "val", "test")} for m in models}
    yte, _ = flat["HAR-X"]["test"]
    ebm = {m: error_by_magnitude(*flat[m]["test"], n_bins=10) for m in models}
    vratio = {m: variance_ratio(flat[m]["test"][1], yte) for m in models}
    vratio["actual"] = 1.0
    tail_bias = {m: signed_bias_top_decile(*flat[m]["test"], q=0.9) for m in models}
    gap = {m: {k: generalization_gap(metrics[m]["train"][k], metrics[m]["test"][k]) for k in ("qlike", "mse")}
           for m in models}
    # HAR inductive-bias evidence uses the 3 raw HAR features (daily/weekly/monthly) at the anchor.
    D = bundle["D"]
    mtr, mte = D.tmask_tr.astype(bool), D.tmask_te.astype(bool)
    har_is = har_ols_r2(D.har_tr[mtr], D.y_tr[mtr])
    har_oos = ols_oos_r2(D.har_tr[mtr], D.y_tr[mtr], D.har_te[mte], D.y_te[mte])
    snr = signal_to_noise_har(D.har_tr[mtr], D.y_tr[mtr])
    return {"metrics": metrics, "flat": flat, "ebm": ebm, "vratio": vratio, "tail_bias": tail_bias,
            "gap": gap, "har_is_r2": har_is, "har_oos_r2": har_oos, "snr": snr}


def _fig_to_b64(fig):  # pragma: no cover - matplotlib IO
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def target_persistence(files, max_lag=22):  # pragma: no cover - reads the raw panel
    """Average lag-1..max_lag autocorrelation of the Parkinson-variance target across tickers."""
    import pandas as pd
    acs = []
    for f in files:
        s = pd.read_csv(f, parse_dates=["date"]).sort_values("date")["parkinson_volatility"]
        s = s.dropna().to_numpy(dtype=float)
        if s.size > max_lag + 5 and np.var(s) > 0:
            acs.append(autocorr(s, max_lag))
    return np.mean(acs, axis=0) if acs else np.full(max_lag, np.nan)


def _fig_error_by_magnitude(results):  # pragma: no cover - matplotlib IO
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    a = results[1]["analysis"]["ebm"]
    x = np.arange(1, 11)
    for m in ("HAR-X", "LSTM"):
        axes[0].plot(x, a[m]["qlike"], "-o", color=MODEL_COLORS[m], label=m)
    axes[0].set_title("h1 test QLIKE by target-magnitude decile")
    axes[0].set_xlabel("target decile (10 = highest volatility)"); axes[0].set_ylabel("mean QLIKE")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    qdiff = a["LSTM"]["qlike"] - a["HAR-X"]["qlike"]
    axes[1].bar(x, qdiff, color=["#d62728" if v > 0 else "#2ca02c" for v in qdiff])
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].set_title("h1 QLIKE penalty of LSTM over HAR-X (LSTM - HAR-X)")
    axes[1].set_xlabel("target decile (10 = highest volatility)")
    axes[1].set_ylabel("QLIKE(LSTM) - QLIKE(HAR-X)"); axes[1].grid(alpha=0.3)
    return _fig_to_b64(fig)


def _fig_mse_by_magnitude(results):  # pragma: no cover - matplotlib IO
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    a = results[1]["analysis"]["ebm"]
    x = np.arange(1, 11)
    for m in ("HAR-X", "LSTM"):
        axes[0].plot(x, a[m]["mse"], "-o", color=MODEL_COLORS[m], label=m)
        axes[1].plot(x, a[m]["mean_signed_error"], "-o", color=MODEL_COLORS[m], label=m)
    axes[0].set_title("h1 test MSE by target-magnitude decile (log scale)")
    axes[0].set_xlabel("target decile (10 = highest volatility)"); axes[0].set_ylabel("MSE")
    axes[0].set_yscale("log"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].set_title("h1 mean signed error (pred - actual) by decile")
    axes[1].set_xlabel("target decile (10 = highest volatility)")
    axes[1].set_ylabel("mean(pred - actual)"); axes[1].legend(); axes[1].grid(alpha=0.3)
    return _fig_to_b64(fig)


def _fig_variance_and_tail(results):  # pragma: no cover - matplotlib IO
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    hs = list(HORIZONS); xw = np.arange(len(hs))
    for i, m in enumerate(("HAR-X", "LSTM")):
        vr = [results[h]["analysis"]["vratio"][m] for h in hs]
        tb = [results[h]["analysis"]["tail_bias"][m] for h in hs]
        axes[0].bar(xw + (i - 0.5) * 0.4, vr, 0.4, color=MODEL_COLORS[m], label=m)
        axes[1].bar(xw + (i - 0.5) * 0.4, tb, 0.4, color=MODEL_COLORS[m], label=m)
    axes[0].axhline(1.0, color=MODEL_COLORS["actual"], ls="--", label="actual = 1.0")
    axes[0].set_title("prediction-variance ratio  var(pred)/var(actual)")
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set_title("top-decile signed bias  mean(pred - actual)")
    for ax in axes:
        ax.set_xticks(xw); ax.set_xticklabels([f"h{h}" for h in hs]); ax.legend(); ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


def _fig_generalization_gap(results):  # pragma: no cover - matplotlib IO
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    hs = list(HORIZONS); xw = np.arange(len(hs))
    for col, metric in enumerate(("qlike", "mse")):
        for i, m in enumerate(("HAR-X", "LSTM")):
            tr = [results[h]["analysis"]["gap"][m][metric]["train"] for h in hs]
            te = [results[h]["analysis"]["gap"][m][metric]["test"] for h in hs]
            axes[col].bar(xw + (i - 0.5) * 0.4, tr, 0.4, color=MODEL_COLORS[m], alpha=0.45,
                          label=f"{m} train")
            axes[col].bar(xw + (i - 0.5) * 0.4, te, 0.4, color=MODEL_COLORS[m], alpha=1.0,
                          fill=False, edgecolor=MODEL_COLORS[m], lw=1.8, label=f"{m} test")
        axes[col].set_title(f"{metric.upper()} train (solid) vs test (outline)")
        axes[col].set_xticks(xw); axes[col].set_xticklabels([f"h{h}" for h in hs])
        axes[col].legend(fontsize=8); axes[col].grid(alpha=0.3)
        if metric == "mse":
            axes[col].set_yscale("log")
    return _fig_to_b64(fig)


def _fig_persistence(persistence):  # pragma: no cover - matplotlib IO
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(persistence) + 1), persistence, "-o", color="#6a3d9a")
    ax.set_title("VN100 target persistence: mean autocorrelation of Parkinson variance")
    ax.set_xlabel("lag (trading days)"); ax.set_ylabel("autocorrelation"); ax.grid(alpha=0.3)
    ax.axhline(0, color="0.5", lw=0.8)
    return _fig_to_b64(fig)


def _fig_spike_series(results, top_n=2):  # pragma: no cover - matplotlib IO
    """Per-ticker h1 test time series showing the LSTM flattening the volatility peaks."""
    import matplotlib.pyplot as plt
    b = results[1]["bundle"]; D = b["D"]
    yte, tm = b["y2d"]["test"], b["tmask"]["test"]
    harx, lstm = b["harx"]["test"], b["lstm"]["test"]
    peak = np.array([yte[tm[:, j].astype(bool), j].max() if tm[:, j].any() else 0.0
                     for j in range(D.N)])
    picks = np.argsort(-peak)[:top_n]
    fig, axes = plt.subplots(top_n, 1, figsize=(11, 3.2 * top_n), squeeze=False)
    for r, j in enumerate(picks):
        m = tm[:, j].astype(bool)
        ax = axes[r][0]
        ax.plot(yte[m, j], color=MODEL_COLORS["actual"], lw=1.4, label="actual")
        ax.plot(harx[m, j], color=MODEL_COLORS["HAR-X"], lw=1.0, alpha=0.9, label="HAR-X")
        ax.plot(lstm[m, j], color=MODEL_COLORS["LSTM"], lw=1.0, alpha=0.9, label="LSTM")
        ax.set_title(f"ticker {D.tickers[j]} — h1 test (actual vs HAR-X vs LSTM)")
        ax.set_ylabel("Parkinson variance"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[-1][0].set_xlabel("test observation index (time-ordered)")
    return _fig_to_b64(fig)


def _metric_table_html(results):  # pragma: no cover - string assembly
    rows = []
    for h in HORIZONS:
        mt = results[h]["analysis"]["metrics"]
        for m in ("HAR-X", "LSTM"):
            d = mt[m]
            rows.append(
                f"<tr><td>h{h}</td><td>{m}</td>"
                + "".join(f"<td>{d[s]['mse']:.3e}</td><td>{d[s]['mae']:.3e}</td>"
                          f"<td>{d[s]['qlike']:.4f}</td>" for s in ("train", "val", "test"))
                + "</tr>")
    head = ("<tr><th>h</th><th>model</th>"
            + "".join(f"<th>{s} MSE</th><th>{s} MAE</th><th>{s} QLIKE</th>"
                      for s in ("train", "val", "test")) + "</tr>")
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _qlike_gap_shares(m1):  # pragma: no cover - arithmetic over the h1 decile QLIKE arrays
    """h1 QLIKE-gap concentration: per-decile (LSTM - HAR-X), and the share of the decile-mean-diff
    sum that the top-2 and top-4 deciles contribute (this sum ~= 10 x the overall mean gap for
    equal-count deciles; np.array_split counts differ by <=1 for a non-multiple-of-10 n)."""
    qd = m1["ebm"]["LSTM"]["qlike"] - m1["ebm"]["HAR-X"]["qlike"]
    tot = float(qd.sum())
    top2 = float(qd[-2:].sum()) / tot if tot != 0 else float("nan")
    top4 = float(qd[-4:].sum()) / tot if tot != 0 else float("nan")
    return qd, top2, top4


def render(results, persistence, out_html, out_md):  # pragma: no cover - report IO
    figs = {"ebm": _fig_error_by_magnitude(results),
            "mse": _fig_mse_by_magnitude(results),
            "vartail": _fig_variance_and_tail(results),
            "gap": _fig_generalization_gap(results),
            "persist": _fig_persistence(persistence),
            "spike": _fig_spike_series(results)}
    m1 = results[1]["analysis"]
    vr_lstm, vr_harx = m1["vratio"]["LSTM"], m1["vratio"]["HAR-X"]
    tb_lstm, tb_harx = m1["tail_bias"]["LSTM"], m1["tail_bias"]["HAR-X"]
    ql_lstm = m1["metrics"]["LSTM"]["test"]["qlike"]; ql_harx = m1["metrics"]["HAR-X"]["test"]["qlike"]
    mse_lstm = m1["metrics"]["LSTM"]["test"]["mse"]; mse_harx = m1["metrics"]["HAR-X"]["test"]["mse"]
    gap_lstm = m1["gap"]["LSTM"]["qlike"]; gap_harx = m1["gap"]["HAR-X"]["qlike"]
    _qd, top2, top4 = _qlike_gap_shares(m1)

    def _emb(key, cap):
        return (f'<figure><img src="data:image/png;base64,{figs[key]}" style="max-width:100%">'
                f'<figcaption>{cap}</figcaption></figure>')

    summary = (
        "On VN100 the deep LSTM's one robust deficit against the parsimonious linear HAR-X is on QLIKE "
        f"(h1 test {ql_lstm:.4f} vs {ql_harx:.4f}; delivered date-clustered DM p=1.1e-3): on squared "
        f"error the two are near-parity (h1 test MSE {mse_lstm:.3e} vs {mse_harx:.3e}). Decomposing "
        "QLIKE by target magnitude localises the bulk of the h1 gap to the high-volatility deciles — the "
        f"top four deciles carry ~{top4:.0%} of it and the top two alone ~{top2:.0%} — where the LSTM "
        "under-predicts the volatility spikes slightly more than HAR-X and QLIKE penalises tail "
        "under-prediction asymmetrically. Ranked by evidence: (1) tail spike-miss under an asymmetric "
        f"QLIKE (primary, but a MODEST relative effect — both models smooth heavily, var(pred)/var(actual)"
        f"={vr_lstm:.2f} for the LSTM vs {vr_harx:.2f} for HAR-X); (2) loss-metric mismatch — the LSTM is "
        "MSE-competitive yet QLIKE-deficient, the MSE-trained / QLIKE-scored signature; (3) HAR's "
        "parsimonious basis is near-optimal for a low signal-to-noise, strongly persistent target; "
        "(4) only a mild overfitting signal — both models' test loss sits BELOW their train loss (the "
        "test regime is lower-variance), the LSTM merely generalises marginally worse than HAR-X.")

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>VN100 LSTM vs HAR-X data-mining</title>
<style>body{{font-family:system-ui,Arial,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#1a1a1a;line-height:1.5}}
h1,h2{{border-bottom:2px solid #ddd;padding-bottom:.3rem}}table{{border-collapse:collapse;font-size:.82rem;margin:1rem 0}}
th,td{{border:1px solid #ccc;padding:3px 7px;text-align:right}}th{{background:#f2f2f2}}td:nth-child(-n+2){{text-align:left}}
figure{{margin:1.4rem 0}}figcaption{{font-size:.85rem;color:#555;margin-top:.3rem}}
.box{{background:#f7f7f9;border-left:4px solid #444;padding:.8rem 1rem;margin:1rem 0}}code{{background:#eee;padding:0 3px}}</style>
</head><body>
<h1>VN100 volatility forecasting: why the deep LSTM underperforms the linear HAR-X</h1>
<p><em>Read-only data-mining of the delivered <code>masked_rich</code> pipeline across the TRAIN, VALIDATION
and TEST splits. Deep = LSTM (5 node features, no graph); Linear = HAR-X (5-feature OLS). Target is
Parkinson VARIANCE; QLIKE floor {QLIKE_FLOOR:g} identical across models; all fits on train rows only.</em></p>
<div class="box"><strong>Executive summary.</strong> {summary}</div>

<h2>0. Established result (delivered, cited)</h2>
<p>On VN100 (delivered <code>masked_rich_floor1e2</code>, 5 seeds, date-clustered Diebold-Mariano) HAR-X
beats the LSTM at every horizon on MSE, RMSE and QLIKE; the LSTM wins only MAE at h1/h5/h10, and the
QLIKE gap is significant at h1 (DM <code>LSTM_vs_HARX</code> p=1.1e-3, favouring HAR-X). This report
re-trains the same LSTM (5 seeds) to obtain per-observation train/val/test predictions and mine the
mechanism. In the re-trained pipeline the QLIKE ranking reproduces exactly at every horizon; the MSE
gap is near-parity (and marginally reverses at h1), consistent with the deficit being concentrated in
QLIKE rather than squared error.</p>
{_metric_table_html(results)}

<h2>1. Tail spike-miss under an asymmetric QLIKE (PRIMARY)</h2>
<p>QLIKE penalises under-prediction of high volatility asymmetrically. Decomposing the h1 QLIKE by
target-magnitude decile localises the bulk of the LSTM's deficit to the high-volatility deciles: it is
neutral-to-favourable in the low deciles and pays its penalty in deciles 7-10, where it under-predicts
the volatility spikes slightly more than HAR-X. The top four deciles carry ~{top4:.0%} of the h1 QLIKE
gap and the top two alone ~{top2:.0%}.</p>
{_emb("ebm", "Lead figure. Left: h1 test QLIKE by target-magnitude decile — the two models coincide in "
     "the bulk; the LSTM's QLIKE rises above HAR-X's in the top deciles. Right: per-decile QLIKE penalty "
     "of the LSTM over HAR-X (LSTM - HAR-X) — positive and largest in deciles 9-10 (the volatility tail).")}
{_emb("spike", "Per-ticker h1 test series for the two highest-peak tickers: both models flatten the "
     "volatility peaks, and the LSTM (red) sits marginally below HAR-X (blue) at the spikes.")}
<p>The effect is a MODEST, RELATIVE one: both models smooth heavily. At h1 var(pred)/var(actual)
={vr_lstm:.3f} (LSTM) vs {vr_harx:.3f} (HAR-X) — HAR-X is a smoother too — and the top-decile signed
bias is {tb_lstm:.3e} (LSTM) vs {tb_harx:.3e} (HAR-X): both under-predict spikes, the LSTM slightly
more. So the over-smoothing is not a large aggregate variance-compression difference; it is a small,
tail-localised extra under-prediction that QLIKE (unlike MSE) magnifies.</p>
{_emb("vartail", "Left: prediction-variance ratio var(pred)/var(actual) by horizon — well below 1 for "
     "BOTH models (both compress the target), the LSTM marginally more. Right: top-decile signed bias "
     "mean(pred-actual) — both negative (spike under-prediction), the LSTM slightly more negative.")}
{_emb("mse", "Squared-error view: h1 MSE by decile (log scale, left) is dominated by the shared top-decile "
     "error and the models nearly coincide — the LSTM's marginal tail under-prediction (right, signed "
     "error) costs little MSE but meaningful QLIKE, which is why the deficit is QLIKE-specific.")}

<h2>2. Loss-metric mismatch (CONTRIBUTING)</h2>
<p>The LSTM is trained on masked MSE but scored on QLIKE. The two disagree here: the LSTM is
MSE-competitive with HAR-X (near-parity, and it matches/edges HAR-X at h1) yet clearly worse on QLIKE.
That is exactly the MSE-trained / QLIKE-scored signature — an MSE-optimal forecast need not be
QLIKE-optimal, and the divergence surfaces precisely in the tail region QLIKE up-weights. This is
tightly coupled to cause 1 (the tail under-prediction is cheap in MSE, expensive in QLIKE) and is a
genuine contributor rather than a mere artifact, since the delivered 5-seed pipeline also gives HAR-X
the MSE/RMSE edge at every horizon.</p>

<h2>3. HAR inductive-bias near-optimality (CONTRIBUTING)</h2>
<p>The target is strongly persistent; HAR's three fixed lags are a parsimonious basis, and the
in-sample vs out-of-sample R^2 of a plain HAR OLS is stable (the linear basis generalises). Forecast-
ability is low: the in-sample R^2 is about 0.2 (only ~one fifth of the target is explainable), so the
signal-to-noise ratio R^2/(1-R^2) is small (see the table) — a regime that favours a high-bias linear
model over a flexible deep one that must learn persistence from a noisy sample and buys variance it
cannot recoup.</p>
{_emb("persist", "Mean autocorrelation of the Parkinson-variance target across VN100 tickers — slow "
     "decay = persistence, which the fixed HAR lags capture directly.")}
<table><thead><tr><th>horizon</th><th>HAR in-sample R^2</th><th>HAR OOS R^2</th><th>signal-to-noise R^2/(1-R^2)</th></tr></thead>
<tbody>{''.join(f"<tr><td>h{h}</td><td>{results[h]['analysis']['har_is_r2']:.4f}</td>"
    f"<td>{results[h]['analysis']['har_oos_r2']:.4f}</td><td>{results[h]['analysis']['snr']:.4f}</td></tr>"
    for h in HORIZONS)}</tbody></table>

<h2>4. Overfitting (MILD, not dominant)</h2>
<p>A memorising LSTM would show a large train->test degradation absent in the linear model. That is
NOT the picture here: for BOTH models the test loss sits below the train loss (the test period is
lower-variance), so there is no gross overfitting. The LSTM does fit train marginally better while
testing marginally worse than HAR-X, i.e. a mild relative generalisation gap — a contributing, not
dominant, factor.</p>
{_emb("gap", "QLIKE and MSE on train (solid) vs test (outline) per horizon. Test bars sit below train "
     "for both models (no gross overfitting); the LSTM's train->test relationship is marginally worse "
     "than HAR-X's.")}
<p>h1 QLIKE train->test: LSTM {gap_lstm['train']:.4f}->{gap_lstm['test']:.4f}
(ratio {gap_lstm['ratio']:.2f}); HAR-X {gap_harx['train']:.4f}->{gap_harx['test']:.4f}
(ratio {gap_harx['ratio']:.2f}).</p>

<h2>Honesty / caveats</h2>
<ul>
<li>The deep model genuinely loses to the parsimonious HAR-X on QLIKE (significant at h1); it is
near-parity on squared error and wins only MAE at the short horizons. This is not a pure metric
artifact — the delivered pipeline gives HAR-X the MSE/RMSE edge at every horizon too.</li>
<li>The over-smoothing is a MODEST, RELATIVE effect: both models compress the target heavily
(var-ratio ~0.24); the LSTM's extra tail under-prediction is small in magnitude and visible mainly in
the QLIKE decile decomposition, not in aggregate variance.</li>
<li>The LSTM numbers here are metrics of the seed-AVERAGED (ensemble) prediction — the same quantity
as the delivered <code>metrics</code> field and the DM basis, so it is self-consistent — not the
paper's per-seed-mean headline (<code>metrics_per_seed</code>), which is generally slightly higher.</li>
<li>The evidence is correlational — the decile-QLIKE, variance and tail-bias patterns are consistent
with the stated mechanism but are measured, not causally proven.</li>
<li>Single delivered configuration (lookback 10, 5 seeds, 20-epoch early-stopped LSTM, per-node
scaler). A different capacity / loss (e.g. training on QLIKE) / lookback could shift the ranking; this
report characterises the delivered setup.</li>
<li>Graph / horizon-decay is out of scope here (studied separately).</li>
</ul>
</body></html>"""
    Path(out_html).write_text(html, encoding="utf-8")

    md = _render_md(results, persistence, summary)
    Path(out_md).write_text(md, encoding="utf-8")


def _render_md(results, persistence, summary):  # pragma: no cover - report IO
    m1 = results[1]["analysis"]
    _qd, top2, top4 = _qlike_gap_shares(m1)
    lines = ["# VN100: why the deep LSTM underperforms the linear HAR-X (data-mining)", "",
             "_Read-only mining of the delivered `masked_rich` pipeline across train/val/test. "
             "Deep = LSTM (5 feats, no graph); Linear = HAR-X (5-feat OLS). Target = Parkinson "
             f"variance; QLIKE floor {QLIKE_FLOOR:g} shared; fits on train only._", "",
             "## Executive summary", "", summary, "",
             "## Ranked conclusion", "",
             "1. **Tail spike-miss under an asymmetric QLIKE (primary).** The LSTM's QLIKE deficit is "
             f"localised to the high-volatility deciles (top-4 ~{top4:.0%}, top-2 ~{top2:.0%} of the h1 "
             "gap): it under-predicts spikes slightly more than HAR-X, and QLIKE punishes tail "
             "under-prediction asymmetrically. The effect is modest and relative — both models smooth "
             "heavily (var(pred)/var(actual) ~ 0.24 for both).",
             "2. **Loss-metric mismatch (contributing).** MSE-trained, QLIKE-scored: the LSTM is "
             "MSE-competitive but QLIKE-deficient — the mismatch surfaces in the tail region QLIKE "
             "up-weights. HAR-X keeps the MSE/RMSE edge in the delivered 5-seed pipeline.",
             "3. **HAR inductive-bias near-optimality (contributing).** Strong persistence + low "
             "signal-to-noise (~0.2 of the target is forecastable) favour the high-bias linear basis; "
             "HAR in-sample vs OOS R^2 is stable.",
             "4. **Overfitting (mild, not dominant).** Both models' TEST loss is below their TRAIN loss "
             "(lower-variance test regime); the LSTM only generalises marginally worse than HAR-X.", "",
             "## Quantitative evidence (h1 test)", "",
             f"- test QLIKE: LSTM {m1['metrics']['LSTM']['test']['qlike']:.4f} vs "
             f"HAR-X {m1['metrics']['HAR-X']['test']['qlike']:.4f} (the robust, significant gap)",
             f"- test MSE: LSTM {m1['metrics']['LSTM']['test']['mse']:.3e} vs "
             f"HAR-X {m1['metrics']['HAR-X']['test']['mse']:.3e} (near-parity — deficit is QLIKE-specific)",
             f"- QLIKE-gap concentration: top-4 deciles ~{top4:.0%}, top-2 deciles ~{top2:.0%} of the gap",
             f"- prediction-variance ratio var(pred)/var(actual): LSTM {m1['vratio']['LSTM']:.3f}, "
             f"HAR-X {m1['vratio']['HAR-X']:.3f} (both << 1 — both compress; actual = 1.0)",
             f"- top-decile signed bias mean(pred-actual): LSTM {m1['tail_bias']['LSTM']:.3e}, "
             f"HAR-X {m1['tail_bias']['HAR-X']:.3e} (both negative; LSTM slightly more = worse spike-miss)",
             f"- h1 HAR in-sample R^2 {m1['har_is_r2']:.4f} / OOS R^2 {m1['har_oos_r2']:.4f} / "
             f"signal-to-noise {m1['snr']:.4f}",
             f"- h1 lag-1 target autocorrelation {persistence[0]:.3f} (slow decay = persistence)", "",
             "## Per-split metrics (retrained pipeline)", "",
             "| h | model | train MSE | train QLIKE | test MSE | test MAE | test QLIKE | QLIKE gap (test-train) |",
             "|---|---|---|---|---|---|---|---|"]
    for h in HORIZONS:
        mt = results[h]["analysis"]["metrics"]; gp = results[h]["analysis"]["gap"]
        for m in ("HAR-X", "LSTM"):
            d = mt[m]
            lines.append(f"| h{h} | {m} | {d['train']['mse']:.3e} | {d['train']['qlike']:.4f} | "
                         f"{d['test']['mse']:.3e} | {d['test']['mae']:.3e} | {d['test']['qlike']:.4f} | "
                         f"{gp[m]['qlike']['diff']:+.4f} |")
    lines += ["", "_QLIKE gap (test-train) is NEGATIVE for both models: the test regime is "
              "lower-variance, so neither model grossly overfits; the LSTM's train->test relationship "
              "is only marginally worse than HAR-X's._", "",
              "## Caveats", "",
              "- The deep model genuinely loses on QLIKE (significant at h1); near-parity on squared "
              "error; wins only MAE at short horizons. Delivered pipeline gives HAR-X the MSE/RMSE edge "
              "at every horizon.",
              "- Over-smoothing is a modest, RELATIVE effect (both models compress ~equally); the LSTM's "
              "extra tail under-prediction shows mainly in the QLIKE decile decomposition.",
              "- Evidence is correlational (measured patterns consistent with the mechanism, not a "
              "causal proof).",
              "- LSTM metrics are of the seed-averaged (ensemble) prediction (the delivered `metrics` / "
              "DM basis), not the per-seed-mean paper headline (`metrics_per_seed`, generally slightly "
              "higher).",
              "- Single delivered configuration (lookback 10, 5 seeds, 20-epoch early-stopped LSTM).",
              "- Graph/horizon-decay studied separately; out of scope here."]
    return "\n".join(lines) + "\n"


def main():  # pragma: no cover - end-to-end driver (needs VN100 data + GPU)
    import glob
    _bootstrap_paths()
    import baselines as B
    import masked_rich as MR
    import run_masked_rich as RUN
    from config import Config
    files = glob.glob(str(_SUB / "data" / "vn100" / "*_processed.csv"))
    if not files:
        raise SystemExit("VN100 processed panel not found; nothing to analyse.")
    price_dir = str(REPO / "data" / "raw" / "prices" / "vn100_vnstock")
    cfg = Config()
    t0 = time.time()
    results = {}
    for h in HORIZONS:
        print(f"[datamining] building + training h{h} ...", flush=True)
        bundle = build_horizon(files, price_dir, h, cfg, MR, RUN, B)
        results[h] = {"bundle": bundle, "analysis": analyse_horizon(bundle)}
        a = results[h]["analysis"]["metrics"]
        print(f"[datamining] h{h} done: test QLIKE HAR-X={a['HAR-X']['test']['qlike']:.4f} "
              f"LSTM={a['LSTM']['test']['qlike']:.4f} | MAE HAR-X={a['HAR-X']['test']['mae']:.3e} "
              f"LSTM={a['LSTM']['test']['mae']:.3e} ({time.time() - t0:.0f}s)", flush=True)
    persistence = target_persistence(files, max_lag=22)
    out_html = REPO / "docs" / "reports" / "2026-08-30_vn100_lstm_vs_harx_datamining.html"
    out_md = REPO / "docs" / "reports" / "2026-08-30_vn100_lstm_vs_harx_datamining.md"
    render(results, persistence, out_html, out_md)
    print(f"[datamining] wrote {out_html} and {out_md} ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
