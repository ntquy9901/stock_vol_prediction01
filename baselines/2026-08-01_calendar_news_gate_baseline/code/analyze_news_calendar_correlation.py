"""EDA: does the news-fusion model's advantage over HAR-only vary by calendar period?

Answers the user's question (2026-08-01) with statistics instead of another training run: reuses
the TWO ALREADY-TRAINED, epoch-matched checkpoints from `2026-07-25_news_usefulness_ablation`
(Model A = HAR-only `ParallelLSTMGNN`, Model B = all-ON `DualGroupNewsBaseline`, both 10 epoch,
SAME test windows) instead of training anything new. That ablation baseline only ever computed
delta_QLIKE aggregated PER TICKER (`compute_ablation_deltas.py`); this script computes it PER
TEST WINDOW instead, joins each window's target DATE with `calendar_features.compute_calendar_vector`,
and looks for a relationship between "when" and "how much news helps".

delta_qlike = qlike(Model B, that point) - qlike(Model A, that point)
    negative -> news model is BETTER at that point (lower QLIKE = better)
    positive -> news model is WORSE at that point

Method:
  1. Rebuild the exact same test set (`create_dual_news_dataloaders`, same panel/pipeline the two
     checkpoints were trained on) -- shuffle=False guarantees window order is reproducible.
  2. Run both checkpoints on it, per-point QLIKE (not the averaged metric).
  3. Recover each point's target calendar date directly from `dataset.stock_data_with_har`
     (read-only; the dataset class itself discards dates after building tensors, so this
     replicates its own indexing logic externally rather than modifying the sibling dataset).
  4. Bucket by month / Tet-window / earnings-window; Welch's t-test + Pearson correlation against
     the continuous tet_proximity/earnings_proximity signals.

Caveats (stated up front, not just in the report): single-seed models, ~164 test windows x 30
tickers = ~4900 points but NOT independent (32 tickers share market-wide co-movement, and 22-day
overlapping windows are highly autocorrelated) -- p-values here are indicative, not rigorous
(effective N is far smaller than raw point count). Treat this as a screening signal, not proof.

Run: python analyze_news_calendar_correlation.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
_DUAL_SIBLING_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_CODE), str(_DUAL_SIBLING_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy import stats

from src.lstm_gat_hybrid.config import LSTMGATConfig
from src.lstm_gat_hybrid.model_parallel import ParallelLSTMGNN

from dataset_dual_news import create_dual_news_dataloaders, _norm_date  # noqa: E402 (sibling, read-only)
from model_dual_news import DualGroupNewsBaseline  # noqa: E402 (sibling, read-only)

from calendar_features import compute_calendar_vector, CALENDAR_FEATURE_NAMES

MODEL_A_CHECKPOINT = _ROOT / "models" / "har_only_ablation_ref_2026-07-25_110813" / "best.pt"
MODEL_B_CHECKPOINT = _ROOT / "models" / "dual_group_news_2026-07-25_011719" / "best.pt"
NEWS_PANEL_PATH = _ROOT / "data" / "features" / "dual_group_news_panel.parquet"


def qlike_pointwise(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Per-point QLIKE term (mean of this == `src.common.evaluation.qlike_loss`'s aggregate)."""
    y_pred = np.maximum(y_pred, epsilon)
    y_true = np.maximum(y_true, epsilon)
    ratio = y_true / y_pred
    return ratio - np.log(ratio) - 1.0


def _denormalize(preds_n: np.ndarray, targs_n: np.ndarray, dataset) -> tuple[np.ndarray, np.ndarray]:
    n_stocks = len(dataset.stock_names)
    preds_d = np.zeros_like(preds_n)
    targs_d = np.zeros_like(targs_n)
    for i in range(len(preds_n)):
        sn = dataset.stock_names[i % n_stocks]
        if sn in dataset.target_normalizers:
            preds_d[i] = dataset.target_normalizers[sn].inverse_transform(
                preds_n[i:i + 1].reshape(1, -1)).flatten()[0]
            targs_d[i] = dataset.target_normalizers[sn].inverse_transform(
                targs_n[i:i + 1].reshape(1, -1)).flatten()[0]
        else:
            preds_d[i] = preds_n[i]
            targs_d[i] = targs_n[i]
    return preds_d, targs_d


def extract_target_dates(dataset) -> list[str]:
    """Per (window, ticker) target date, in the SAME flatten order as `preds.reshape(-1)`
    (window-major, then ticker-minor within `dataset.stock_names` order) -- mirrors exactly how
    `_create_sequences` builds `y_all` per window (see dataset_dual_news.py's
    `target_idx = i + seq_length + forecast_horizon - 1`), but computed externally (read-only,
    no edit to the sibling dataset class)."""
    seq_length = dataset.seq_length
    horizon = dataset.forecast_horizon
    n_windows = len(dataset)
    dates = []
    for i in range(n_windows):
        target_idx = i + seq_length + horizon - 1
        for stock in dataset.stock_names:
            d = dataset.stock_data_with_har[stock]['date'].iloc[target_idx]
            dates.append(_norm_date(str(d)))
    return dates


def run_model_a(model, loader, device):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for x_har, adj, _x_news, y in loader:
            x_har, adj, y = x_har.to(device), adj.to(device), y.to(device)
            pred = model(x_har, adj)
            preds.append(pred.cpu().numpy().reshape(-1))
            targs.append(y.cpu().numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(targs)


def run_model_b(model, loader, device):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for x_har, adj, x_news, y in loader:
            x_har, adj, x_news, y = x_har.to(device), adj.to(device), x_news.to(device), y.to(device)
            pred = model(x_har, adj, x_news)
            preds.append(pred.cpu().numpy().reshape(-1))
            targs.append(y.cpu().numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(targs)


def build_dataframe(df_dates, tickers, delta_qlike, y_true) -> pd.DataFrame:
    unique_dates = sorted(set(df_dates))
    cal_cache = {d: compute_calendar_vector(d) for d in unique_dates}
    cal_matrix = np.stack([cal_cache[d] for d in df_dates])
    idx = {name: i for i, name in enumerate(CALENDAR_FEATURE_NAMES)}

    months = [int(d[5:7]) for d in df_dates]
    df = pd.DataFrame({
        "ticker": tickers,
        "date": df_dates,
        "month": months,
        "y_true": y_true,
        "delta_qlike": delta_qlike,
        "tet_proximity": cal_matrix[:, idx["tet_proximity"]],
        "in_tet_window": cal_matrix[:, idx["in_tet_window"]].astype(bool),
        "earnings_proximity": cal_matrix[:, idx["earnings_proximity"]],
        "in_earnings_window": cal_matrix[:, idx["in_earnings_window"]].astype(bool),
    })
    return df


def analyze(df: pd.DataFrame) -> dict:
    out = {}

    by_month = df.groupby("month")["delta_qlike"].agg(["mean", "count"]).reindex(range(1, 13))
    out["by_month"] = {int(m): {"mean_delta_qlike": None if pd.isna(r["mean"]) else float(r["mean"]),
                                 "n": int(r["count"]) if not pd.isna(r["count"]) else 0}
                       for m, r in by_month.iterrows()}

    for flag_col, label in (("in_tet_window", "tet_window"), ("in_earnings_window", "earnings_window")):
        inside = df.loc[df[flag_col], "delta_qlike"]
        outside = df.loc[~df[flag_col], "delta_qlike"]
        if len(inside) >= 2 and len(outside) >= 2:
            t_stat, p_val = stats.ttest_ind(inside, outside, equal_var=False)
        else:
            t_stat, p_val = float("nan"), float("nan")
        out[label] = {
            "n_inside": int(len(inside)), "mean_delta_inside": float(inside.mean()) if len(inside) else None,
            "n_outside": int(len(outside)), "mean_delta_outside": float(outside.mean()) if len(outside) else None,
            "welch_t_stat": float(t_stat), "welch_p_value": float(p_val),
        }

    for prox_col, label in (("tet_proximity", "tet_proximity_corr"), ("earnings_proximity", "earnings_proximity_corr")):
        r, p = stats.pearsonr(df[prox_col], df["delta_qlike"])
        out[label] = {"pearson_r": float(r), "p_value": float(p), "n": int(len(df))}

    return out


def plot_by_month(analysis: dict, out_path: Path):
    months = list(range(1, 13))
    means = [analysis["by_month"][m]["mean_delta_qlike"] for m in months]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#d62728" if (m is not None and m > 0) else "#2ca02c" for m in means]
    ax.bar(months, [0 if m is None else m for m in means], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("month")
    ax.set_ylabel("mean delta_QLIKE (news model - HAR-only)")
    ax.set_title("News-model advantage by month (negative = news helps)")
    ax.set_xticks(months)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] plot saved: {out_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[analyze] device={device}")

    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    _, _, test_loader, (train_ds, val_ds, test_ds) = create_dual_news_dataloaders(
        data_dir=str(_ROOT / "data" / "processed"), news_panel_path=str(NEWS_PANEL_PATH),
        graph_method="knn", batch_size=32, config=config)

    n_feat = train_ds._n_feat
    model_a = ParallelLSTMGNN(config).to(device)
    model_a.load_state_dict(torch.load(MODEL_A_CHECKPOINT, map_location=device))
    model_b = DualGroupNewsBaseline(config, n_feat=n_feat, d_news=64, dropout=0.5).to(device)
    model_b.load_state_dict(torch.load(MODEL_B_CHECKPOINT, map_location=device))
    print(f"[analyze] loaded Model A={MODEL_A_CHECKPOINT.name}, Model B={MODEL_B_CHECKPOINT.name}")

    preds_a_n, targs_n = run_model_a(model_a, test_loader, device)
    preds_b_n, targs_b_n = run_model_b(model_b, test_loader, device)
    np.testing.assert_array_almost_equal(targs_n, targs_b_n, decimal=5)  # same windows, sanity check

    preds_a_d, targs_d = _denormalize(preds_a_n, targs_n, test_ds)
    preds_b_d, _ = _denormalize(preds_b_n, targs_b_n, test_ds)

    qlike_a = qlike_pointwise(targs_d, preds_a_d)
    qlike_b = qlike_pointwise(targs_d, preds_b_d)
    delta_qlike = qlike_b - qlike_a

    dates = extract_target_dates(test_ds)
    n_stocks = len(test_ds.stock_names)
    tickers = [test_ds.stock_names[i % n_stocks] for i in range(len(dates))]
    assert len(dates) == len(delta_qlike), (
        f"date/prediction length mismatch: {len(dates)} dates vs {len(delta_qlike)} predictions")

    df = build_dataframe(dates, tickers, delta_qlike, targs_d)
    analysis = analyze(df)

    print("\n=== Mean delta_QLIKE by month (negative = news helps) ===")
    for m in range(1, 13):
        info = analysis["by_month"][m]
        mdq = info["mean_delta_qlike"]
        print(f"  month {m:>2}: n={info['n']:>4}  mean_delta_qlike="
              f"{'n/a' if mdq is None else f'{mdq:+.4f}'}")

    for label in ("tet_window", "earnings_window"):
        r = analysis[label]
        print(f"\n=== {label}: inside vs outside ===")
        print(f"  inside  (n={r['n_inside']:>4}): mean_delta_qlike={r['mean_delta_inside']:+.4f}")
        print(f"  outside (n={r['n_outside']:>4}): mean_delta_qlike={r['mean_delta_outside']:+.4f}")
        print(f"  Welch t={r['welch_t_stat']:.3f}, p={r['welch_p_value']:.4f}")

    for label in ("tet_proximity_corr", "earnings_proximity_corr"):
        r = analysis[label]
        print(f"\n=== {label} ===  pearson_r={r['pearson_r']:+.4f}, p={r['p_value']:.4f}, n={r['n']}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = _ROOT / "results" / f"news_calendar_correlation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_parquet(out_dir / "per_point_delta_qlike.parquet", index=False)
    plot_by_month(analysis, out_dir / "delta_qlike_by_month.png")
    print(f"\n[done] -> {out_dir}")


if __name__ == "__main__":
    main()
