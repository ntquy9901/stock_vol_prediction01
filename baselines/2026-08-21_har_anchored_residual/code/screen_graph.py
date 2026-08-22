"""Model-free graph screening (diagnosis doc section 7): does a WEIGHTED / SIGNED / INNOVATION neighbour
signal add out-of-sample predictive value beyond HAR, before building any corrected GAT?

The prior model-free test was only S0 (equal-weight neighbour mean) — which matches the binary graph the
GATLayer actually consumes (V2). This screens the richer families the binary graph cannot represent:
  S0  equal-weight neighbour mean of PK^2 at t
  S1  signed-weighted neighbour sum   sum_j A_ij PK^2_{j,t}
  S2  separate positive / negative neighbour sums (two features)
  S3  signed-weighted neighbour HAR-RESIDUAL (innovation) sum   sum_j A_ij r^HAR_{j,t}
  PLACEBO  S1 with row-shuffled edges (density preserved) — must be ~0
Each signal is appended to the 3 HAR features in an OLS fit on TRAIN only; incremental value = relative
reduction in TEST (and VAL) MSE vs the HAR-only OLS. Leakage-safe: glasso edges + HAR coef + residual HAR
fit are all TRAIN-only; the neighbour signal at date t uses only day-t observations.

CLI: python screen_graph.py <dataset> [--data-root DIR] [--min-common N]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

_SUB = Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"
sys.path.insert(0, str(_SUB)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import baselines as B  # noqa: E402
import data_utils as du  # noqa: E402
import edges as EG  # noqa: E402
from snapshots import _load_panel  # noqa: E402

HORIZONS = [1, 5, 10, 22]
FIRST_VALID = 21


def _ols_fit(X, y):
    Xb = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return coef


def _ols_pred(X, coef):
    return np.column_stack([np.ones(len(X)), X]) @ coef


def _mse(y, p):
    return float(np.mean((y - p) ** 2))


def screen(files, dataset, min_common=300, seed=0):
    panel = _load_panel(files, min_common=min_common)
    tickers = list(panel.columns)
    pk = panel.to_numpy(float)                       # [T, N]
    T, N = pk.shape
    feats = np.stack([du.har_features(pk[:, j]) for j in range(N)], axis=1)   # [T,N,3]
    rng = np.random.default_rng(seed)

    out = {"dataset": dataset, "num_nodes": N, "common_dates": T, "min_common": min_common, "horizons": {}}
    for h in HORIZONS:
        anchors = np.arange(FIRST_VALID + 9, T - h)   # lookback 10 -> window start valid
        n = len(anchors)
        i_tr, i_va = int(n * 0.8), int(n * 0.9)
        a_tr = anchors[:i_tr - h]                      # purge h at boundaries
        a_va = anchors[i_tr:i_va - h]
        a_te = anchors[i_va:]
        if len(a_tr) < 50 or len(a_va) < 10 or len(a_te) < 10:
            continue
        # train-only signed glasso adjacency (Top-5)
        adj = np.asarray(EG.glasso_adjacency(panel.iloc[a_tr], top_k=5), dtype=float)
        np.fill_diagonal(adj, 0.0)                      # neighbour signal excludes self
        # train-fit HAR (pooled) to build residual innovations r[j,t] = pk[j,t]-HAR(feat[j,t]) leakage-safe
        har_coef = B.har_fit(feats[a_tr].reshape(-1, 3), pk[a_tr + 0].reshape(-1))  # contemporaneous HAR
        r_all = np.empty((T, N))                        # residual innovation per (t, node)
        valid = np.arange(FIRST_VALID, T)
        r_all[:] = np.nan
        r_all[valid] = pk[valid] - B.har_predict(feats[valid].reshape(-1, 3), har_coef).reshape(len(valid), N)

        adj_shuf = adj.copy()
        perm = rng.permutation(N)
        adj_shuf = adj_shuf[perm][:, perm]             # density/degree-preserving placebo

        # S4 directed lead-lag edges (train-only): LL[i,j] = corr(r_j[t], r_i[t+h]) over train dates;
        # keep the top-5 |corr| predictors j per target i, signed. Tests PREDICTIVE (not contemporaneous)
        # spillover with edges selected to be predictive rather than symmetric correlations.
        rt = np.nan_to_num(r_all)
        r_src = rt[a_tr]                                # [ntr, N] neighbour innovation at t
        r_dst = rt[a_tr + h]                            # [ntr, N] own innovation at t+h
        rs = (r_src - r_src.mean(0)) / (r_src.std(0) + 1e-12)
        rd = (r_dst - r_dst.mean(0)) / (r_dst.std(0) + 1e-12)
        ll = (rd.T @ rs) / len(a_tr)                    # [i, j] corr(r_i[t+h], r_j[t])
        np.fill_diagonal(ll, 0.0)
        ll_top = np.zeros_like(ll)
        for i in range(N):
            k = np.argsort(-np.abs(ll[i]))[:5]
            ll_top[i, k] = ll[i, k]

        def neigh(A, mat, t_idx):                       # sum_j A_ij mat[t,j] -> [len(t_idx), N]
            return mat[t_idx] @ A.T

        def signals(A, t_idx):
            pkt = pk                                    # [T,N]
            s0 = (pkt[t_idx] @ (A != 0).T) / np.maximum((A != 0).sum(1), 1)   # equal-weight mean
            s1 = neigh(A, pkt, t_idx)                                          # signed-weighted PK
            sp = neigh(np.where(A > 0, A, 0.0), pkt, t_idx)
            sn = neigh(np.where(A < 0, -A, 0.0), pkt, t_idx)
            s3 = neigh(A, np.nan_to_num(r_all), t_idx)                         # signed-weighted innovation
            s4 = neigh(ll_top, np.nan_to_num(r_all), t_idx)                    # directed lead-lag innovation
            return {"S0_mean": s0, "S1_weighted": s1, "S2_pos": sp, "S2_neg": sn, "S3_resid": s3, "S4_leadlag": s4}

        def build(t_idx):
            X_har = feats[t_idx].reshape(-1, 3)
            y = pk[t_idx + h].reshape(-1)
            return X_har, y

        Xtr, ytr = build(a_tr); Xva, yva = build(a_va); Xte, yte = build(a_te)
        sig_tr = signals(adj, a_tr); sig_va = signals(adj, a_va); sig_te = signals(adj, a_te)
        sig_tr_p = signals(adj_shuf, a_tr); sig_te_p = signals(adj_shuf, a_te); sig_va_p = signals(adj_shuf, a_va)

        base = _ols_fit(Xtr, ytr)
        mse_va_base, mse_te_base = _mse(yva, _ols_pred(Xva, base)), _mse(yte, _ols_pred(Xte, base))

        def incr(cols_tr, cols_va, cols_te):
            Xtr2 = np.column_stack([Xtr] + cols_tr)
            c = _ols_fit(Xtr2, ytr)
            mva = _mse(yva, _ols_pred(np.column_stack([Xva] + cols_va), c))
            mte = _mse(yte, _ols_pred(np.column_stack([Xte] + cols_te), c))
            return {"val_incr_R2": 1 - mva / mse_va_base, "test_incr_R2": 1 - mte / mse_te_base}

        res = {
            "n_test_dates": len(a_te), "n_test_obs": len(yte),
            "S0_mean": incr([sig_tr["S0_mean"].reshape(-1, 1)], [sig_va["S0_mean"].reshape(-1, 1)], [sig_te["S0_mean"].reshape(-1, 1)]),
            "S1_weighted": incr([sig_tr["S1_weighted"].reshape(-1, 1)], [sig_va["S1_weighted"].reshape(-1, 1)], [sig_te["S1_weighted"].reshape(-1, 1)]),
            "S2_signed": incr([sig_tr["S2_pos"].reshape(-1, 1), sig_tr["S2_neg"].reshape(-1, 1)],
                              [sig_va["S2_pos"].reshape(-1, 1), sig_va["S2_neg"].reshape(-1, 1)],
                              [sig_te["S2_pos"].reshape(-1, 1), sig_te["S2_neg"].reshape(-1, 1)]),
            "S3_resid": incr([sig_tr["S3_resid"].reshape(-1, 1)], [sig_va["S3_resid"].reshape(-1, 1)], [sig_te["S3_resid"].reshape(-1, 1)]),
            "S4_leadlag": incr([sig_tr["S4_leadlag"].reshape(-1, 1)], [sig_va["S4_leadlag"].reshape(-1, 1)], [sig_te["S4_leadlag"].reshape(-1, 1)]),
            "PLACEBO_S1": incr([sig_tr_p["S1_weighted"].reshape(-1, 1)], [sig_va_p["S1_weighted"].reshape(-1, 1)], [sig_te_p["S1_weighted"].reshape(-1, 1)]),
        }
        out["horizons"][str(h)] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["vn30", "vn100", "sp500"])
    ap.add_argument("--data-root", default=str(_SUB / "data"))
    ap.add_argument("--min-common", type=int, default=300)
    a = ap.parse_args()
    root = Path(a.data_root)
    mp = {"vn30": [root / "vn30" / "*_processed.csv", root / "*_processed.csv"],
          "vn100": [root / "vn100" / "*_processed.csv", root / "vn100_vnstock" / "*_processed.csv"],
          "sp500": [root / "sp500" / "*_processed.csv"]}
    files = next((glob.glob(str(p)) for p in mp[a.dataset] if glob.glob(str(p))), [])
    res = screen(files, a.dataset, min_common=a.min_common)
    outp = Path(__file__).resolve().parents[3] / "results" / "graph_screen" / f"{a.dataset}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[screen] {a.dataset} N={res['num_nodes']} dates={res['common_dates']}")
    for h, r in res["horizons"].items():
        print(f"  h{h} (test_dates={r['n_test_dates']}): "
              + " ".join(f"{k}={v['test_incr_R2']:+.4f}" for k, v in r.items() if isinstance(v, dict)))


if __name__ == "__main__":
    main()
