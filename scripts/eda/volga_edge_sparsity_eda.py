"""EDA (analysis-only, CPU): VolGA horizon-matched spillover-edge SPARSITY decomposition.

Reproduces the delivered ``directed_vol2pk_hmatched`` edge (baselines/2026-09-05_edge_horizon_matched/
code/run_edge_hmatched.py) READ-ONLY on each walk-forward TRAIN window and quantifies:
  1. sparsity decomposition -- Bonferroni floor vs Top-K cap;
  2. correlation structure -- is the significance screen pruning signal or noise (analytic z / BH-FDR);
  3. density vs performance across the 12 (market x horizon) result JSONs;
  4. denser-edge candidate edge counts (FDR, relaxed alpha, higher Top-K, vol->vol, |return|->vol);
  5. thin-market self-loop-only fraction.

NOT a baseline: trains nothing, writes no result JSON, touches no baseline code. Run:
  .venv_gpu_encode/Scripts/python.exe scripts/eda/volga_edge_sparsity_eda.py --out docs/reports/2026-09-06_volga_edge_sparsity_eda
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _p in ("baselines/2026-08-31_walkforward_volga/code",
           "baselines/2026-08-30_walkforward_harx_lstm/code",
           "baselines/2026-08-21_har_anchored_residual/code",
           "submission/soict_lstm_gat",
           "baselines/2026-09-05_edge_horizon_matched/code"):
    sys.path.insert(0, str(REPO / _p))

import masked_rich as MR          # noqa: E402
from wf_folds import make_folds   # noqa: E402
from wf_enriched_panel import build_enriched_panel, frozen_universe  # noqa: E402
from run_volga_walkforward import VolgaWFConfig, enriched_glob        # noqa: E402
from run_edge_hmatched import EDGE_SIG_ALPHA                          # noqa: E402

ND = NormalDist()
MARKETS = ["vn30", "vn100", "sp500_clean"]
HORIZONS = [1, 5, 10, 22]
LOOKBACK = 10          # canonical lb10 used by the delivered edge_hmatched runs (config.lookback)
FOLDS_TARGET = 7       # matches the delivered runs


def _corr_matrices(src, tgt, min_pairs=MR._MIN_PAIRS):
    """Pairwise-complete Pearson r[i,j]=corr(src_i, tgt_j) + jointly-finite pair count, exactly as the
    delivered edge (BLAS matmuls over NaN-masked [T,N] matrices). Returns (R, npair)."""
    Sf = np.isfinite(src); Ff = np.isfinite(tgt)
    S = np.where(Sf, src, 0.0).astype(np.float64); F = np.where(Ff, tgt, 0.0).astype(np.float64)
    Sm = Sf.astype(np.float64); Fm = Ff.astype(np.float64)
    npair = Sm.T @ Fm
    sa = S.T @ Fm; sb = Sm.T @ F
    saa = (S * S).T @ Fm; sbb = Sm.T @ (F * F); sab = S.T @ F
    with np.errstate(invalid="ignore", divide="ignore"):
        num = npair * sab - sa * sb
        den = np.sqrt(npair * saa - sa * sa) * np.sqrt(npair * sbb - sb * sb)
        R = num / den
    R[~np.isfinite(R)] = np.nan
    R[npair < min_pairs] = np.nan
    np.fill_diagonal(R, np.nan)
    return R, npair


def _bonf_topk(R, npair, n, top_k, alpha):
    """Reproduce the delivered screen: Bonferroni sig mask (before Top-K) then Top-K per target column.
    Returns (sig, kept) boolean [source,target] masks. ``kept`` = final edges in A (off-diagonal)."""
    z_bonf = ND.inv_cdf(1.0 - alpha / (2.0 * max(n - 1, 1))) if alpha else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        thr = (z_bonf / np.sqrt(npair)) if alpha else 0.0
    sig = np.isfinite(R) & (np.abs(R) > thr)
    scores = np.where(sig, np.abs(R), -np.inf)
    topk = np.argsort(-scores, axis=0)[:top_k]
    tgts = np.arange(n)
    kept = np.zeros_like(sig)
    for rank in range(topk.shape[0]):
        srcs = topk[rank]; k = sig[srcs, tgts]
        kept[srcs[k], tgts[k]] = True
    return sig, kept


def _bh_fdr_count(R, npair, n, q):
    """Benjamini-Hochberg count of significant off-diagonal pairs at FDR level q (per target column),
    a LESS conservative alternative to Bonferroni. Returns total kept-before-TopK count."""
    total = 0
    z = np.abs(R) * np.sqrt(npair)
    with np.errstate(invalid="ignore"):
        p = 2.0 * (1.0 - np.array([[ND.cdf(v) if np.isfinite(v) else np.nan for v in row] for row in z]))
    for j in range(n):
        col = p[:, j]
        pv = np.sort(col[np.isfinite(col)])
        m = len(pv)
        if m == 0:
            continue
        thresh = q * (np.arange(1, m + 1) / m)
        below = np.where(pv <= thresh)[0]
        if len(below):
            total += below[-1] + 1
    return total


def _density(kept, n):
    return float(kept.sum()) / (n * (n - 1))


def _returns_matrix(files, keep, dates):  # pragma: no cover - data/JSON I/O glue (pure stats helpers are tested)
    """|daily_return| aligned to panel dates for the |return|->vol candidate edge."""
    idx = pd.DatetimeIndex(dates)
    M = np.full((len(idx), len(keep)), np.nan)
    fmap = {Path(f).stem: f for f in files}
    for j, tk in enumerate(keep):
        df = pd.read_csv(fmap[tk], parse_dates=["date"]).set_index("date")
        M[:, j] = df["daily_return"].reindex(idx).to_numpy(dtype=float)
    return np.abs(M)


def analyse_config(market, horizon):  # pragma: no cover - data/JSON I/O glue (pure stats helpers are tested)
    files = glob.glob(enriched_glob(market))
    keep = frozen_universe(files, LOOKBACK, horizon)
    panel = build_enriched_panel(files, LOOKBACK, horizon, keep)
    wf = VolgaWFConfig(lookback=LOOKBACK, horizon=horizon, folds_target=FOLDS_TARGET)
    nA = len(panel.anchors); ts = int(nA * wf.test_frac)
    K = max(1, math.ceil((nA - ts) / wf.folds_target))
    folds = make_folds(nA, ts, K, wf.val, wf.horizon)
    n = panel.N
    vshock = panel.feats[:, :, 4]
    sqrt_pk = np.sqrt(panel.pk)
    absret = _returns_matrix(files, panel.tickers, panel.dates)

    per_fold = []
    for fold in folds:
        last_tr = int(panel.anchors[fold.train][-1]) + horizon
        v = vshock[:last_tr + 1]; p = sqrt_pk[:last_tr + 1]; ar = absret[:last_tr + 1]
        src = v[:-horizon]; tgt = p[horizon:]
        R, npair = _corr_matrices(src, tgt)
        finite = np.isfinite(R)
        # -- delivered screen (alpha=0.05 Bonferroni + Top-K=5) --
        sig, kept = _bonf_topk(R, npair, n, MR.EDGE_TOP_K, EDGE_SIG_ALPHA)
        sig_per_tgt = sig.sum(0)          # sources passing Bonferroni per target (before Top-K)
        kept_per_tgt = kept.sum(0)        # sources kept after Top-K
        # -- correlation structure: analytic z and uncorrected significance --
        z = np.abs(R[finite]) * np.sqrt(npair[finite])
        n_pairs_valid = int(finite.sum())
        n_uncorr = int((z > ND.inv_cdf(0.975)).sum())          # |z|>1.96 (uncorrected p<0.05, two-sided)
        exp_uncorr = 0.05 * n_pairs_valid                       # expected false positives under null
        # -- denser candidates (counts of edges BEFORE Top-K unless noted) --
        _, kept_a10 = _bonf_topk(R, npair, n, MR.EDGE_TOP_K, 0.10)
        _, kept_a20 = _bonf_topk(R, npair, n, MR.EDGE_TOP_K, 0.20)
        _, kept_k10 = _bonf_topk(R, npair, n, 10, EDGE_SIG_ALPHA)
        _, kept_k20 = _bonf_topk(R, npair, n, 20, EDGE_SIG_ALPHA)
        bh05 = _bh_fdr_count(R, npair, n, 0.05)
        bh10 = _bh_fdr_count(R, npair, n, 0.10)
        # vol->vol edge (sqrt_pk_i(t) -> sqrt_pk_j(t+h))
        Rvv, npvv = _corr_matrices(p[:-horizon], tgt)
        _, kept_vv = _bonf_topk(Rvv, npvv, n, MR.EDGE_TOP_K, EDGE_SIG_ALPHA)
        # |return|->vol edge
        Rrv, nprv = _corr_matrices(ar[:-horizon], tgt)
        _, kept_rv = _bonf_topk(Rrv, nprv, n, MR.EDGE_TOP_K, EDGE_SIG_ALPHA)

        per_fold.append({
            "n": n,
            "density_delivered": _density(kept, n),
            "n_pairs_valid": n_pairs_valid,
            "sig_bonf_total": int(sig.sum()),
            "kept_topk_total": int(kept.sum()),
            "self_loop_only_frac": float((kept_per_tgt == 0).mean()),
            "sig_per_tgt_mean": float(sig_per_tgt.mean()),
            "sig_per_tgt_median": float(np.median(sig_per_tgt)),
            "kept_per_tgt_mean": float(kept_per_tgt.mean()),
            "frac_tgt_sig_gt_k": float((sig_per_tgt > MR.EDGE_TOP_K).mean()),   # Top-K is binding here
            "frac_tgt_sig_le_k_gt0": float(((sig_per_tgt <= MR.EDGE_TOP_K) & (sig_per_tgt > 0)).mean()),
            "n_uncorr_p05": n_uncorr,
            "exp_uncorr_null": exp_uncorr,
            "signal_excess_ratio": (n_uncorr / exp_uncorr) if exp_uncorr > 0 else float("nan"),
            "median_abs_r": float(np.nanmedian(np.abs(R))),
            "p95_abs_r": float(np.nanpercentile(np.abs(R), 95)),
            "max_abs_r": float(np.nanmax(np.abs(R))),
            "dens_a10": _density(kept_a10, n), "dens_a20": _density(kept_a20, n),
            "dens_k10": _density(kept_k10, n), "dens_k20": _density(kept_k20, n),
            "dens_bh05": bh05 / (n * (n - 1)), "dens_bh10": bh10 / (n * (n - 1)),
            "dens_vol2vol": _density(kept_vv, n), "dens_absret2vol": _density(kept_rv, n),
        })
    keys = per_fold[0].keys()
    agg = {k: float(np.mean([f[k] for f in per_fold])) for k in keys}
    agg["n_folds"] = len(folds)
    agg["market"] = market
    agg["horizon"] = horizon
    return agg


def load_perf():  # pragma: no cover - data/JSON I/O glue (pure stats helpers are tested)
    """Density + QLIKE + DM p from the 12 delivered result JSONs."""
    rows = []
    for f in sorted(glob.glob(str(REPO / "results/edge_hmatched/edgehm_*.json"))):
        d = json.load(open(f))
        m = d["metrics"]; dm = d["dm_date_clustered"]
        rows.append({
            "market": d["market"], "horizon": d["horizon"], "n": d["num_nodes"],
            "density": d["edge_density_mean"],
            "qlike_harx": m["HAR-X"]["qlike"], "qlike_lstm": m["LSTM"]["qlike"],
            "qlike_volga": m["VolGA"]["qlike"],
            "benefit_vs_harx": m["HAR-X"]["qlike"] - m["VolGA"]["qlike"],
            "benefit_vs_lstm": m["LSTM"]["qlike"] - m["VolGA"]["qlike"],
            "dm_p_lstm": dm["VolGA_vs_LSTM"]["qlike"]["p_value"],
            "dm_fav_lstm": dm["VolGA_vs_LSTM"]["qlike"]["favors"],
            "dm_p_harx": dm["VolGA_vs_HAR-X"]["qlike"]["p_value"],
        })
    return rows


def main():  # pragma: no cover - data/JSON I/O glue (pure stats helpers are tested)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "docs/reports/2026-09-06_volga_edge_sparsity_eda"))
    a = ap.parse_args()
    out = {}
    for mkt in MARKETS:
        for h in HORIZONS:
            print(f"[eda] {mkt} h{h} ...", flush=True)
            out[f"{mkt}_h{h}"] = analyse_config(mkt, h)
    perf = load_perf()
    payload = {"sparsity": out, "perf": perf,
               "params": {"lookback": LOOKBACK, "folds_target": FOLDS_TARGET,
                          "alpha": EDGE_SIG_ALPHA, "top_k": MR.EDGE_TOP_K, "min_pairs": MR._MIN_PAIRS}}
    jp = Path(str(a.out) + ".json")
    jp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[eda] wrote {jp}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
