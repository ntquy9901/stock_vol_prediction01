"""Model-free extended screening (diagnosis doc section 7 S5/S6 + section 15 richer node features).

The prior screen (``screen_graph.py``, S0-S4) found neighbour signals from a Parkinson-variance
graph add ~0 incremental OOS R2 beyond HAR, but only on HAR-only node features and without volume.
This module extends the same leakage-safe OLS screen to:

  S5  volume-shock GRAPH: build a standardized causal log-volume shock ``vshock`` per ticker, a
      TRAIN-only vshock-correlation Top-5 adjacency, and test whether the neighbour volume-shock
      signal ``sum_j A_ij vshock[j,t]`` (and its equal-weight mean) predicts own pk[t+h] beyond HAR.
      Shuffled-edge placebo (density/degree preserved) included.

  RICHER node features (doc section 15), each appended to the 3 HAR features individually AND all
  together: own return & |return| (close-to-close), market return (cross-sectional mean return),
  market Parkinson variance (cross-sectional mean pk), stock/market vol ratio, vol-of-vol (rolling
  std of pk), cross-sectional dispersion of pk, own volume shock. Tests whether HAR-only node
  features were too thin (the null graph result was measured on HAR-only features).

  S6  sector graph: BLOCKED — no historically-valid sector/GICS/constituent metadata exists in the
      repo (verified by search); not fabricated / not fetched online.

Method (identical to screen_graph.py for comparability): append the candidate column(s) to the 3 HAR
features, fit OLS with intercept on TRAIN only, incremental value = 1 - MSE(HAR+feat)/MSE(HAR-only)
on TEST (and VAL). Anchors, 80/10/10 split, and the ``h``-purge at split boundaries mirror
screen_graph exactly. All rolling stats are CAUSAL (per-row, past-only) and all cross-sectional stats
use only day-t observations, so no train-only global standardization is required; the vshock
adjacency and the HAR fit are TRAIN-only.

CLI: python screen_features.py <dataset> [--data-root DIR] [--raw-root DIR] [--min-common N]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
_SUB = _ROOT / "submission" / "soict_lstm_gat"
sys.path.insert(0, str(_SUB)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import baselines as B  # noqa: E402
import data_utils as du  # noqa: E402
from snapshots import _load_panel  # noqa: E402

HORIZONS = [1, 5, 10, 22]
FIRST_VALID = 21
VOL_WIN = 22       # causal rolling window for the log-volume shock (monthly)
VOV_WIN = 22       # causal rolling window for vol-of-vol (rolling std of pk)
_EPS = 1e-12

# Raw OHLCV directory candidates per dataset (first that has files wins).
RAW_DIRS = {
    "vn30": ["data/raw/prices"],
    "vn100": ["data/raw/prices/vn100_vnstock", "data/raw/prices/vn100"],
    "sp500": ["data/raw/prices/sp500"],
}


def _ols_fit(X, y):
    Xb = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return coef


def _ols_pred(X, coef):
    return np.column_stack([np.ones(len(X)), X]) @ coef


def _mse(y, p):
    return float(np.mean((y - p) ** 2))


def _causal_zscore(x: np.ndarray, win: int) -> np.ndarray:
    """Per-column trailing (causal) z-score: (x - rollmean) / rollstd over ``win`` past obs.

    ``x`` is [T, N]. Uses pandas rolling with ``min_periods=win`` so the first ``win-1`` rows are NaN
    (dropped by the anchor start well past warmup). Purely causal — row t uses rows [t-win+1..t].
    """
    s = pd.DataFrame(x)
    mean = s.rolling(win, min_periods=win).mean()
    std = s.rolling(win, min_periods=win).std(ddof=0)
    z = (s - mean) / (std + _EPS)
    return z.to_numpy(dtype=float)


def _load_raw_aligned(tickers, dates, raw_root: Path, dataset: str):
    """Volume [T,N] and close [T,N] for ``tickers`` aligned (inner-reindex) to panel ``dates``.

    Raw files share the processed source, so raw dates are a superset of panel dates; reindexing to
    ``dates`` yields an exact inner alignment. A missing ticker file / missing date maps to NaN.
    """
    raw_dir = None
    for cand in RAW_DIRS.get(dataset, []):
        p = raw_root / cand
        if p.exists() and any(p.glob("*_ohlcv.csv")):
            raw_dir = p
            break
    if raw_dir is None:
        raise FileNotFoundError(f"no raw OHLCV dir with *_ohlcv.csv for dataset={dataset} under {raw_root}")

    T, N = len(dates), len(tickers)
    vol = np.full((T, N), np.nan)
    close = np.full((T, N), np.nan)
    idx = pd.DatetimeIndex(dates)
    for j, tk in enumerate(tickers):
        f = raw_dir / f"{tk}_ohlcv.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date").set_index("date")
        df = df[~df.index.duplicated(keep="last")].reindex(idx)
        vol[:, j] = df["volume"].to_numpy(dtype=float)
        close[:, j] = df["close"].to_numpy(dtype=float)
    try:
        raw_label = str(raw_dir.relative_to(_ROOT))
    except ValueError:
        raw_label = str(raw_dir)
    return vol, close, raw_label


def _build_node_features(pk: np.ndarray, vol: np.ndarray, close: np.ndarray, vshock: np.ndarray) -> dict:
    """Return dict name -> [T, N] causal candidate node feature."""
    T, N = pk.shape
    # own close-to-close return (row 0 NaN), |return|
    ret = np.full((T, N), np.nan)
    ret[1:] = close[1:] / (close[:-1] + _EPS) - 1.0
    abs_ret = np.abs(ret)
    # cross-sectional (day-t) market aggregates, broadcast to [T, N]
    mkt_ret = np.nanmean(ret, axis=1, keepdims=True) * np.ones((1, N))
    mkt_pk = np.nanmean(pk, axis=1, keepdims=True) * np.ones((1, N))
    disp = np.nanstd(pk, axis=1, keepdims=True) * np.ones((1, N))
    vol_ratio = pk / (np.nanmean(pk, axis=1, keepdims=True) + _EPS)
    # vol-of-vol: causal rolling std of pk
    vov = pd.DataFrame(pk).rolling(VOV_WIN, min_periods=VOV_WIN).std(ddof=0).to_numpy(dtype=float)
    return {
        "own_return": ret,
        "abs_return": abs_ret,
        "market_return": mkt_ret,
        "market_pk": mkt_pk,
        "vol_ratio": vol_ratio,
        "vol_of_vol": vov,
        "xsec_dispersion": disp,
        "own_vshock": vshock,
    }


def _vshock_adjacency(vshock_tr: np.ndarray, top_k: int = 5) -> np.ndarray:
    """Signed Top-K node adjacency from the TRAIN vshock correlation matrix (self excluded)."""
    N = vshock_tr.shape[1]
    x = np.nan_to_num(vshock_tr)
    x = x - x.mean(0)
    std = x.std(0) + _EPS
    xs = x / std
    corr = (xs.T @ xs) / len(xs)
    np.fill_diagonal(corr, 0.0)
    k = min(top_k, N - 1)
    adj = np.zeros((N, N), dtype=float)
    for i in range(N):
        order = np.argsort(-np.abs(corr[i]), kind="stable")[:k]
        for j in order:
            if np.abs(corr[i, j]) > 0.0:
                adj[i, j] = corr[i, j]
    return adj


def screen(files, dataset, raw_root, min_common=300, seed=0):
    panel = _load_panel(files, min_common=min_common)
    tickers = list(panel.columns)
    dates = panel.index
    pk = panel.to_numpy(float)                       # [T, N]
    T, N = pk.shape
    feats = np.stack([du.har_features(pk[:, j]) for j in range(N)], axis=1)   # [T,N,3]

    vol, close, raw_dir_used = _load_raw_aligned(tickers, dates, raw_root, dataset)
    log_vol = np.log(np.maximum(vol, 1.0))           # floor at 1 (0-volume halts) before log
    vshock = _causal_zscore(log_vol, VOL_WIN)        # [T,N] causal standardized log-volume shock
    vshock_f = np.nan_to_num(vshock)                 # for matmul neighbour sums
    node_feats = _build_node_features(pk, vol, close, vshock)

    rng = np.random.default_rng(seed)
    out = {
        "dataset": dataset, "num_nodes": N, "common_dates": T, "min_common": min_common,
        "raw_dir": raw_dir_used, "vol_win": VOL_WIN, "vov_win": VOV_WIN,
        "n_tickers_with_volume": int(np.sum(np.any(np.isfinite(vol), axis=0))),
        "horizons": {},
    }

    for h in HORIZONS:
        anchors = np.arange(FIRST_VALID + 9, T - h)   # match screen_graph anchor start
        n = len(anchors)
        i_tr, i_va = int(n * 0.8), int(n * 0.9)
        a_tr = anchors[:i_tr - h]                      # purge h at boundaries
        a_va = anchors[i_tr:i_va - h]
        a_te = anchors[i_va:]
        if len(a_tr) < 50 or len(a_va) < 10 or len(a_te) < 10:
            continue

        X_tr = feats[a_tr].reshape(-1, 3); y_tr = pk[a_tr + h].reshape(-1)
        X_va = feats[a_va].reshape(-1, 3); y_va = pk[a_va + h].reshape(-1)
        X_te = feats[a_te].reshape(-1, 3); y_te = pk[a_te + h].reshape(-1)

        base = _ols_fit(X_tr, y_tr)
        mse_va_base = _mse(y_va, _ols_pred(X_va, base))
        mse_te_base = _mse(y_te, _ols_pred(X_te, base))

        def incr(col_tr, col_va, col_te):
            """col_* are lists of [len(a_*), N] arrays; pooled (date-major, node-minor) like HAR."""
            c_tr = [np.nan_to_num(c).reshape(-1, 1) for c in col_tr]
            c_va = [np.nan_to_num(c).reshape(-1, 1) for c in col_va]
            c_te = [np.nan_to_num(c).reshape(-1, 1) for c in col_te]
            coef = _ols_fit(np.column_stack([X_tr] + c_tr), y_tr)
            mva = _mse(y_va, _ols_pred(np.column_stack([X_va] + c_va), coef))
            mte = _mse(y_te, _ols_pred(np.column_stack([X_te] + c_te), coef))
            return {"val_incr_R2": 1 - mva / mse_va_base, "test_incr_R2": 1 - mte / mse_te_base}

        # ---- S5 volume-shock graph ----
        adj = _vshock_adjacency(vshock_f[a_tr], top_k=5)
        perm = rng.permutation(N)
        adj_shuf = adj[perm][:, perm]                  # density/degree-preserving placebo
        deg = np.maximum((adj != 0).sum(1), 1)
        deg_shuf = np.maximum((adj_shuf != 0).sum(1), 1)

        def v_weighted(A, t_idx):
            return vshock_f[t_idx] @ A.T               # [len, N] sum_j A_ij vshock[j,t]

        def v_mean(A, dg, t_idx):
            return (vshock_f[t_idx] @ (A != 0).T) / dg

        s5 = {
            "S5_vshock_weighted": incr([v_weighted(adj, a_tr)], [v_weighted(adj, a_va)], [v_weighted(adj, a_te)]),
            "S5_vshock_mean": incr([v_mean(adj, deg, a_tr)], [v_mean(adj, deg, a_va)], [v_mean(adj, deg, a_te)]),
            "PLACEBO_S5": incr([v_weighted(adj_shuf, a_tr)], [v_weighted(adj_shuf, a_va)], [v_weighted(adj_shuf, a_te)]),
        }

        # ---- richer node features (individual + combined) ----
        rich = {}
        for name, arr in node_feats.items():
            rich[name] = incr([arr[a_tr]], [arr[a_va]], [arr[a_te]])
        all_tr = [node_feats[k][a_tr] for k in node_feats]
        all_va = [node_feats[k][a_va] for k in node_feats]
        all_te = [node_feats[k][a_te] for k in node_feats]
        rich["ALL_richer"] = incr(all_tr, all_va, all_te)

        out["horizons"][str(h)] = {
            "n_test_dates": int(len(a_te)), "n_test_obs": int(len(y_te)),
            "S5": s5, "richer": rich,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["vn30", "vn100", "sp500"])
    ap.add_argument("--data-root", default=str(_SUB / "data"))
    ap.add_argument("--raw-root", default=str(_ROOT))
    ap.add_argument("--min-common", type=int, default=300)
    a = ap.parse_args()
    root = Path(a.data_root)
    mp = {"vn30": [root / "vn30" / "*_processed.csv", root / "*_processed.csv"],
          "vn100": [root / "vn100" / "*_processed.csv", root / "vn100_vnstock" / "*_processed.csv"],
          "sp500": [root / "sp500" / "*_processed.csv"]}
    files = next((glob.glob(str(p)) for p in mp[a.dataset] if glob.glob(str(p))), [])
    if not files:
        raise FileNotFoundError(f"no processed CSVs found for {a.dataset} under {root}")
    res = screen(files, a.dataset, Path(a.raw_root), min_common=a.min_common)
    outp = _ROOT / "results" / "graph_screen" / f"{a.dataset}_features.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[screen_features] {a.dataset} N={res['num_nodes']} dates={res['common_dates']} "
          f"vol_tickers={res['n_tickers_with_volume']} raw={res['raw_dir']}")
    for h, r in res["horizons"].items():
        s5 = " ".join(f"{k}={v['test_incr_R2']:+.4f}" for k, v in r["S5"].items())
        best = max(r["richer"].items(), key=lambda kv: kv[1]["test_incr_R2"])
        print(f"  h{h} (test_dates={r['n_test_dates']}): {s5} | best_richer {best[0]}={best[1]['test_incr_R2']:+.4f} "
              f"ALL={r['richer']['ALL_richer']['test_incr_R2']:+.4f}")


if __name__ == "__main__":
    main()
