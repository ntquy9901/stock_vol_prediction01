"""Reusable test-set error-analysis EDA: WHY do the pooled metrics vary across stocks?

For each panel/horizon it rebuilds the masked-rich test set, computes PER-TICKER data characteristics
(variance level, vol-of-vol, low-range/floor fraction, persistence, skew, volume coverage) and PER-TICKER,
PER-MODEL errors (MSE/MAE/QLIKE/R2 + share of the pooled SSE/QLIKE), correlates error against the
characteristics (Spearman), and renders a self-contained HTML report (tables + matplotlib plots) so the
concentration and drivers of the pooled metrics are visible.

Models: HAR-X and GARCH are deterministic and cheap (no training) and are computed here directly. The deep
models are not re-trained by this tool; run the training pipeline with per-node prediction dumps to add them.

Usage:
  python scripts/eda/test_diagnostics.py --panels vn30 vn100 hose hnx sp500 --horizon 1 \
      --out docs/reports/eda/<stamp>_test_diagnostics.html
Reuses the delivered build_masked_rich + the HAR-X/GARCH predictors; every number is recomputed from data.
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
_CODE = REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"
_SUB = REPO / "submission" / "soict_lstm_gat"
_GAR = REPO / "scripts" / "garch_masked"
for _p in (_SUB, _CODE, _GAR):
    sys.path.insert(0, str(_p))

import masked_rich as MR          # noqa: E402
import compute_garch_masked as CG  # noqa: E402
import metrics as M               # noqa: E402
from config import Config         # noqa: E402
from scipy.stats import skew as _skew, spearmanr  # noqa: E402

# panel -> (processed glob candidates, raw price dir), mirroring the delivered runners
_SUBDATA = _SUB / "data"
FILES = {
    "vn30": [_SUBDATA / "vn30" / "*_processed.csv", _SUBDATA / "*_processed.csv"],
    "vn100": [_SUBDATA / "vn100" / "*_processed.csv", _SUBDATA / "vn100_vnstock" / "*_processed.csv"],
    "hose": [REPO / "data" / "processed" / "hose" / "*_processed.csv"],
    "hnx": [REPO / "data" / "processed" / "hnx" / "*_processed.csv"],
    "sp500": [REPO / "data" / "processed" / "sp500" / "*_processed.csv"],
}
PRICE = {
    "vn30": REPO / "data" / "raw" / "prices",
    "vn100": REPO / "data" / "raw" / "prices" / "vn100_vnstock",
    "hose": REPO / "data" / "raw" / "prices" / "hose_vnstock",
    "hnx": REPO / "data" / "raw" / "prices" / "hnx_vnstock",
    "sp500": REPO / "data" / "raw" / "prices" / "sp500",
}
SCREEN = {"hose", "hnx", "sp500"}


def _resolve_files(panel):
    files = next((glob.glob(str(p)) for p in FILES[panel] if glob.glob(str(p))), [])
    files = sorted(files)
    if panel in SCREEN:
        import floor_sensitivity as FS
        files = FS.screen_files(files)
    return files, str(PRICE[panel])


def _series_by_node(pred_dict, N):
    """Group a {(node, date): (y, pred)} dict into per-node (y, pred) arrays."""
    ys = [[] for _ in range(N)]
    ps = [[] for _ in range(N)]
    for (j, _d), (y, p) in pred_dict.items():
        ys[j].append(y)
        ps[j].append(p)
    return [np.asarray(a) for a in ys], [np.asarray(a) for a in ps]


def _volume_missing_frac(ticker, price_dir):
    """Fraction of a ticker's own trading days whose raw volume is missing/NaN or zero (F4 coverage)."""
    path = Path(price_dir) / f"{ticker}_ohlcv.csv"
    if not path.exists():
        return np.nan
    try:
        raw = pd.read_csv(path)
        v = pd.to_numeric(raw.get("volume"), errors="coerce").to_numpy(dtype=float)
        if v.size == 0:
            return np.nan
        bad = ~np.isfinite(v) | (v <= 0.0)
        return float(bad.mean())
    except Exception:
        return np.nan


def per_ticker_frame(D, cfg, price_dir, floor, horizon):
    """Return a per-ticker DataFrame: data characteristics + HAR-X and GARCH errors and pooled-share."""
    harx = CG._harx_pred(D, cfg)
    garch = CG._garch_pred(D, horizon, cfg)
    hy, hp = _series_by_node(harx, D.N)
    gy, gp = _series_by_node(garch, D.N)
    tmask = D.tmask_te.astype(bool)
    node_floor = 1e-2 * D.t_mean + 1e-12
    tot_sse_h = sum(float(np.sum((hy[j] - hp[j]) ** 2)) for j in range(D.N) if hy[j].size)
    tot_ql_h = sum(float(np.sum(M.per_obs_qlike(hy[j], hp[j], floor))) for j in range(D.N) if hy[j].size)
    rows = []
    for j in range(D.N):
        y = D.y_te[tmask[:, j], j]
        if y.size < 5 or hy[j].size == 0:
            continue
        mean = float(y.mean())
        std = float(y.std())
        ac1 = float(np.corrcoef(y[:-1], y[1:])[0, 1]) if y.size > 2 and y.std() > 0 else np.nan
        row = {
            "ticker": D.tickers[j],
            "n_test": int(y.size),
            "mean_var": mean,
            "median_var": float(np.median(y)),
            "vol_of_vol": std / mean if mean > 0 else np.nan,   # dispersion of the variance series
            "low_range_frac": float(np.mean(y < node_floor[j])),  # days at/under the per-node floor
            "autocorr1": ac1,
            "skew": float(_skew(y)) if y.size > 2 else np.nan,
            "vol_missing_frac": _volume_missing_frac(D.tickers[j], price_dir),
        }
        for name, yy, pp, tot_sse, tot_ql in (("harx", hy[j], hp[j], tot_sse_h, tot_ql_h),
                                              ("garch", gy[j], gp[j], tot_sse_h, tot_ql_h)):
            sse = float(np.sum((yy - pp) ** 2))
            row[f"{name}_mse"] = M.mse(yy, pp)
            row[f"{name}_mae"] = M.mae(yy, pp)
            row[f"{name}_qlike"] = M.qlike(yy, pp, floor)
            row[f"{name}_r2"] = M.r2(yy, pp)
            row[f"{name}_clip_rate"] = float(np.mean(pp <= node_floor[j] * (1 + 1e-9)))
            row[f"{name}_sse_share"] = sse / tot_sse_h if tot_sse_h > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


CHARS = ["mean_var", "vol_of_vol", "low_range_frac", "autocorr1", "skew", "vol_missing_frac", "n_test"]


def spearman_table(df, model="harx"):
    """Spearman correlation of a model's per-ticker QLIKE / MSE against each characteristic."""
    out = []
    for err in (f"{model}_qlike", f"{model}_mse"):
        r = {"error": err}
        for c in CHARS:
            sub = df[[err, c]].dropna()
            r[c] = round(float(spearmanr(sub[err], sub[c]).statistic), 3) if len(sub) > 4 else np.nan
        out.append(r)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------------------------------
def _fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _plots_for_panel(df, panel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    imgs = []
    # 1) QLIKE vs low-range fraction (floor activation) and vs vol-of-vol
    for xcol, xlabel in (("low_range_frac", "fraction of days at/under floor"),
                         ("vol_of_vol", "vol-of-vol (std/mean of variance)")):
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.scatter(df[xcol], df["harx_qlike"], s=12, alpha=0.6)
        ax.set_xlabel(xlabel); ax.set_ylabel("HAR-X QLIKE (per ticker)")
        ax.set_title(f"{panel}: QLIKE vs {xcol}")
        imgs.append(_fig_b64(fig))
    # 2) log-log MSE vs mean variance level
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.scatter(df["mean_var"], df["harx_mse"], s=12, alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mean Parkinson variance"); ax.set_ylabel("HAR-X MSE (per ticker)")
    ax.set_title(f"{panel}: MSE vs variance level"); imgs.append(_fig_b64(fig))
    # 3) Lorenz curve of SSE concentration
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    s = np.sort(df["harx_sse_share"].dropna().to_numpy())[::-1]
    cum = np.concatenate([[0], np.cumsum(s)])
    ax.plot(np.arange(len(cum)) / max(len(s), 1), cum, marker=".", ms=3)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("fraction of tickers (worst first)"); ax.set_ylabel("cumulative share of pooled SSE")
    ax.set_title(f"{panel}: SSE concentration"); imgs.append(_fig_b64(fig))
    return imgs


def _df_html(df, cols, floatfmt=4, n=None):
    d = df[cols].copy()
    if n:
        d = d.head(n)
    return d.to_html(index=False, float_format=lambda x: f"{x:.{floatfmt}g}", border=0,
                     classes="tbl", justify="center")


def render_html(panels, out_path, horizon):
    parts = ["<html><head><meta charset='utf-8'><title>Test EDA — error analysis</title>",
             "<style>body{font-family:system-ui,Arial,sans-serif;margin:24px;color:#1a1a1a;max-width:1100px}"
             "h1{font-size:22px}h2{font-size:18px;margin-top:34px;border-bottom:2px solid #ddd}"
             "h3{font-size:15px;color:#333}table.tbl{border-collapse:collapse;font-size:12px;margin:8px 0}"
             ".tbl td,.tbl th{border:1px solid #ccc;padding:3px 7px;text-align:center}"
             ".tbl th{background:#f2f2f2}img{margin:6px 10px 6px 0;border:1px solid #eee}"
             ".note{color:#555;font-size:13px}.k{color:#b00}</style></head><body>"]
    parts.append(f"<h1>Test-set error analysis (EDA) — horizon h{horizon}</h1>")
    parts.append("<p class='note'>Per-ticker diagnostics on the masked test set. Models shown: HAR-X (linear "
                 "baseline) and GARCH, both deterministic. QLIKE blows up where predictions sit at the per-node "
                 "floor on many low-range days; MSE is dominated by high-variance tickers. Every number is "
                 "recomputed from the processed data + raw OHLCV.</p>")
    # cross-panel summary
    summ = []
    for panel, df in panels.items():
        top1 = df.nlargest(1, "harx_sse_share")
        summ.append({
            "panel": panel, "tickers": len(df),
            "HAR-X QLIKE (mean)": round(df["harx_qlike"].mean(), 4),
            "worst-ticker SSE share": round(float(top1["harx_sse_share"].iloc[0]), 3) if len(top1) else np.nan,
            "top-5 SSE share": round(float(df.nlargest(5, "harx_sse_share")["harx_sse_share"].sum()), 3),
            "median low-range frac": round(float(df["low_range_frac"].median()), 3),
            "median vol-missing frac": round(float(df["vol_missing_frac"].median()), 3),
        })
    parts.append("<h2>Cross-panel summary</h2>")
    parts.append(pd.DataFrame(summ).to_html(index=False, border=0, classes="tbl", justify="center"))
    parts.append("<p class='note'>“top-5 SSE share” = fraction of the pooled squared error carried by the 5 "
                 "worst tickers — a high value means the pooled MSE is driven by a few stocks.</p>")
    for panel, df in panels.items():
        parts.append(f"<h2>{panel}</h2>")
        parts.append("<h3>Correlation (Spearman) of per-ticker error vs data characteristics</h3>")
        parts.append(spearman_table(df).to_html(index=False, border=0, classes="tbl", justify="center",
                                                 float_format=lambda x: f"{x:.3f}"))
        parts.append("<h3>10 worst tickers by QLIKE</h3>")
        parts.append(_df_html(df.nlargest(10, "harx_qlike"),
                              ["ticker", "n_test", "mean_var", "vol_of_vol", "low_range_frac",
                               "harx_clip_rate", "vol_missing_frac", "harx_qlike", "harx_mse"], n=10))
        parts.append("<h3>10 worst tickers by share of pooled SSE</h3>")
        parts.append(_df_html(df.nlargest(10, "harx_sse_share"),
                              ["ticker", "n_test", "mean_var", "vol_of_vol", "harx_sse_share",
                               "harx_mse", "harx_qlike"], n=10))
        parts.append("<div>")
        for b64 in _plots_for_panel(df, panel):
            parts.append(f"<img src='data:image/png;base64,{b64}'/>")
        parts.append("</div>")
    parts.append("</body></html>")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+", default=["vn30", "vn100", "hose", "hnx", "sp500"])
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = Config()
    panels = {}
    for panel in a.panels:
        files, price_dir = _resolve_files(panel)
        if not files:
            print(f"[eda] {panel}: no processed files, skip", flush=True)
            continue
        D = MR.build_masked_rich(files, price_dir, cfg.lookback, a.horizon)
        df = per_ticker_frame(D, cfg, price_dir, cfg.qlike_floor, a.horizon)
        panels[panel] = df
        print(f"[eda] {panel} h{a.horizon}: {len(df)} tickers | HAR-X QLIKE mean={df['harx_qlike'].mean():.4f} "
              f"| top-5 SSE share={df.nlargest(5, 'harx_sse_share')['harx_sse_share'].sum():.3f}", flush=True)
    out = a.out or str(REPO / "docs" / "reports" / "eda" / f"h{a.horizon}_test_diagnostics.html")
    p = render_html(panels, out, a.horizon)
    print(f"[eda] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
