"""Render the VolGA edge-sparsity EDA plots to a single self-contained HTML (base64 PNGs).

Reads the summary JSON from volga_edge_sparsity_eda.py and recomputes one representative fold's
correlation matrix for the |r|-distribution panels. Analysis-only; writes no result JSON.
Run: .venv_gpu_encode/Scripts/python.exe scripts/eda/volga_edge_sparsity_plots.py
"""
from __future__ import annotations

import base64
import glob
import io
import json
import math
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

import volga_edge_sparsity_eda as E  # noqa: E402  (sibling module, same dir on sys.path)

ND = NormalDist()
REPO = E.REPO
OUT_HTML = REPO / "docs/reports/2026-09-06_volga_edge_sparsity_eda.html"
JSON = REPO / "docs/reports/2026-09-06_volga_edge_sparsity_eda.json"


def _b64(fig):  # pragma: no cover - matplotlib/data I/O
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()


def _last_fold_R(market, horizon):  # pragma: no cover - matplotlib/data I/O
    """Recompute corr matrix + Bonferroni threshold on the last (largest) train fold for the |r| panel."""
    files = glob.glob(E.enriched_glob(market))
    keep = E.frozen_universe(files, E.LOOKBACK, horizon)
    panel = E.build_enriched_panel(files, E.LOOKBACK, horizon, keep)
    wf = E.VolgaWFConfig(lookback=E.LOOKBACK, horizon=horizon, folds_target=E.FOLDS_TARGET)
    nA = len(panel.anchors); ts = int(nA * wf.test_frac)
    K = max(1, math.ceil((nA - ts) / wf.folds_target))
    folds = E.make_folds(nA, ts, K, wf.val, wf.horizon)
    fold = folds[-1]
    last_tr = int(panel.anchors[fold.train][-1]) + horizon
    v = panel.feats[:, :, 4][:last_tr + 1]; p = np.sqrt(panel.pk)[:last_tr + 1]
    R, npair = E._corr_matrices(v[:-horizon], p[horizon:])
    n = panel.N
    z_bonf = ND.inv_cdf(1.0 - E.EDGE_SIG_ALPHA / (2.0 * max(n - 1, 1)))
    thr = z_bonf / np.sqrt(np.nanmedian(npair[np.isfinite(R)]))   # representative threshold
    return R[np.isfinite(R)], thr


def main():  # pragma: no cover - matplotlib/data I/O
    d = json.load(open(JSON)); S = d["sparsity"]; P = d["perf"]
    key = lambda p: f"{p['market']}_h{p['horizon']}"
    imgs = []

    # Panel 1: density vs benefit_vs_LSTM (colored by horizon) --------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cmap = {1: "C0", 5: "C1", 10: "C2", 22: "C3"}
    for p in P:
        c = cmap[p["horizon"]]
        ax[0].scatter(p["density"] * 100, p["benefit_vs_lstm"], color=c, s=60)
        ax[0].annotate(f"{p['market'][:4]}h{p['horizon']}", (p["density"] * 100, p["benefit_vs_lstm"]),
                       fontsize=7, alpha=.7)
        ax[1].scatter(S[key(p)]["signal_excess_ratio"], p["benefit_vs_lstm"], color=c, s=60)
        ax[1].annotate(f"{p['market'][:4]}h{p['horizon']}", (S[key(p)]["signal_excess_ratio"], p["benefit_vs_lstm"]),
                       fontsize=7, alpha=.7)
    for a in ax:
        a.axhline(0, color="k", lw=.6)
        a.set_ylabel("QLIKE(LSTM) - QLIKE(VolGA)  (>0 = graph helps)")
    ax[0].set_xlabel("edge density (%)"); ax[0].set_title("Density vs graph benefit (Spearman +0.19)")
    ax[1].set_xlabel("signal excess ratio (n|z|>1.96 / null)")
    ax[1].set_title("Signal strength vs graph benefit (Spearman +0.73)")
    imgs.append(("Density is NOT the lever; signal strength IS", _b64(fig)))

    # Panel 2: sparsity decomposition (Bonferroni-bound vs Top-K-bound) -----------------------
    fig, ax = plt.subplots(figsize=(11, 4))
    labels = list(S.keys())
    selfonly = [S[k]["self_loop_only_frac"] * 100 for k in labels]
    sig_gt_k = [S[k]["frac_tgt_sig_gt_k"] * 100 for k in labels]
    mid = [100 - a - b for a, b in zip(selfonly, sig_gt_k)]
    x = np.arange(len(labels))
    ax.bar(x, sig_gt_k, label="Top-K binding (sig sources > K=5)", color="C0")
    ax.bar(x, mid, bottom=sig_gt_k, label="Bonferroni binding (0 < sig <= K)", color="C1")
    ax.bar(x, selfonly, bottom=[a + b for a, b in zip(sig_gt_k, mid)],
           label="self-loop only (no spillover survives)", color="C3")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of targets"); ax.legend(fontsize=8)
    ax.set_title("Sparsity decomposition: what limits each target's edges")
    imgs.append(("What binds the edge count: Top-K (h1) vs Bonferroni/self-loop (long h)", _b64(fig)))

    # Panel 3: |r| distribution, strong-signal vs weak-signal config -------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, (mkt, h, ttl) in zip(ax, [("sp500_clean", 1, "SP500 h1 (excess 18.9x)"),
                                     ("vn30", 22, "VN30 h22 (excess 1.7x)")]):
        r, thr = _last_fold_R(mkt, h)
        a.hist(np.abs(r), bins=60, color="C0", alpha=.7)
        a.axvline(thr, color="C3", lw=1.5, label=f"Bonferroni thr ~{thr:.3f}")
        a.set_xlabel("|Pearson r|  volume_i(t) -> sqrt_pk_j(t+h)")
        a.set_ylabel("pair count"); a.set_title(ttl); a.legend(fontsize=8)
    imgs.append(("Correlation magnitudes are tiny (|r|<0.25); screen keeps the right tail", _b64(fig)))

    html = ["<html><head><meta charset='utf-8'><title>VolGA edge-sparsity EDA</title>",
            "<style>body{font-family:sans-serif;max-width:1000px;margin:2em auto}"
            "h2{color:#333}img{width:100%;border:1px solid #ccc}</style></head><body>",
            "<h1>VolGA horizon-matched spillover-edge: sparsity EDA</h1>",
            "<p>Generated 2026-09-06. Analysis-only (no training). See "
            "<code>docs/reports/2026-09-06_volga_edge_sparsity_eda.md</code>.</p>"]
    for title, b in imgs:
        html.append(f"<h2>{title}</h2><img src='data:image/png;base64,{b}'/>")
    html.append("</body></html>")
    OUT_HTML.write_text("\n".join(html), encoding="utf-8")
    print(f"[plots] wrote {OUT_HTML}")


if __name__ == "__main__":  # pragma: no cover
    main()
