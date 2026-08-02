"""
Market-level sentiment analysis (deepening of the sentiment<->price EDA).

Pools all 30 VN30 stocks into DAILY market-wide sentiment/attention series, then
correlates with (a) market return and (b) market volatility. Reuses helpers from
sentiment_price_eda (same code/ folder).

Key questions:
  - Does market news ATTENTION (news volume/day) predict next-day volatility?
  - Does mean daily sentiment predict market return direction (T -> T+k)?
  - Does sentiment predict return/vol MAGNITUDE?
  - Are results robust to detrending (both sentiment and market trended up 2010-2025)?

Run:
    python baselines/2026-07-11_sentiment_price_eda/code/sentiment_market_eda.py

Outputs -> results/2026-07-11_sentiment_price_eda/market/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
import sentiment_price_eda as eda  # reuse helpers, constants, paths

OUT_DIR = eda.OUT_DIR / "market"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 3, 5]          # forward-return horizons (market)
VOL_LAGS = [0, 1, 3, 5]          # lags for vol / magnitude targets
ROLL_VOL_WINDOW = 22             # ~1 trading month realized-vol window
MIN_PAIRS = 30                   # min non-null pairs to report a correlation


# ---------------------------------------------------------------------------
# Build daily market series
# ---------------------------------------------------------------------------
def build_market_sentiment() -> pd.DataFrame:
    """Pool per-stock events -> daily market sentiment measures (only news days)."""
    frames = []
    for t in eda.load_tickers():
        sp = eda.SENT_DIR / f"{t}_sentiment.csv"
        if not sp.exists():
            continue
        df = pd.read_csv(sp, parse_dates=["date"])
        ev = df[df["news_count_1d"].astype(float) > 0][["date", "sentiment_1d"]].copy()
        ev["ticker"] = t
        frames.append(ev)
    ev = pd.concat(frames, ignore_index=True)
    ev["sentiment_1d"] = ev["sentiment_1d"].astype(float)

    g = ev.groupby("date").agg(
        mkt_sent_mean=("sentiment_1d", "mean"),
        mkt_news_count=("sentiment_1d", "size"),
        mkt_sent_std=("sentiment_1d", "std"),
        mkt_neg_count=("sentiment_1d", lambda x: int((x < eda.NEG_THR).sum())),
    ).reset_index()
    g["mkt_neg_ratio"] = g["mkt_neg_count"] / g["mkt_news_count"]
    return g


def build_market_panel(horizons: list[int]) -> pd.DataFrame:
    """Equal-weighted cross-sectional mean of per-stock forward returns + Parkinson vol,
    plus a daily market return series and its rolling realized volatility."""
    # --- per-stock forward returns, cross-sectionally averaged per date ---
    long_ret, long_vol = [], []
    daily_ret_long = []
    for t in eda.load_tickers():
        pp = eda.PRICE_DIR / f"{t}_ohlcv.csv"
        if not pp.exists():
            continue
        price = pd.read_csv(pp, parse_dates=["date"]).sort_values("date")
        fr = eda.compute_forward_returns(price, horizons)
        for k in horizons:
            sub = fr[["date", f"ret_{k}d"]].dropna().rename(columns={f"ret_{k}d": "val"})
            sub["horizon"] = k
            long_ret.append(sub)
        # same-day realized daily return (for realized-vol series)
        dr = price[["date", "close"]].copy()
        dr["val"] = dr["close"].astype(float).pct_change()
        daily_ret_long.append(dr[["date", "val"]].dropna())

        vp = eda.VOL_DIR / f"{t}_processed.csv"
        if vp.exists():
            vol = pd.read_csv(vp, parse_dates=["date"])
            v = vol[["date", "parkinson_volatility"]].dropna()
            v = v[v["parkinson_volatility"].astype(float) > 0]
            long_vol.append(v.rename(columns={"parkinson_volatility": "val"}))

    lr = pd.concat(long_ret, ignore_index=True)
    mkt_ret = lr.groupby(["date", "horizon"])["val"].mean().unstack("horizon")
    mkt_ret.columns = [f"mkt_ret_{int(k)}d" for k in mkt_ret.columns]
    # winsorize each forward-return column (robust to split artifacts)
    for c in mkt_ret.columns:
        mkt_ret[c] = eda.winsorize(mkt_ret[c].to_numpy(), 1, 99)
    mkt_abs = mkt_ret.abs().rename(columns={c: c.replace("mkt_ret_", "mkt_abs_ret_") for c in mkt_ret.columns})

    lv = pd.concat(long_vol, ignore_index=True)
    mkt_vol = lv.groupby("date")["val"].mean().to_frame("mkt_vol_avg")

    drl = pd.concat(daily_ret_long, ignore_index=True)
    mkt_daily_ret = drl.groupby("date")["val"].mean()
    mkt_daily_ret = pd.Series(eda.winsorize(mkt_daily_ret.to_numpy(), 1, 99), index=mkt_daily_ret.index)
    mkt_daily = mkt_daily_ret.to_frame("mkt_ret_1d_realized")
    # trailing realized vol: window [T-21 .. T] (contemporaneous, NOT predictive)
    mkt_daily["mkt_realized_vol"] = mkt_daily["mkt_ret_1d_realized"].rolling(ROLL_VOL_WINDOW).std()
    # FORWARD realized vol: window [T+1 .. T+22] = trailing vol evaluated at T+22, shifted back.
    # This is the predictive quantity (vol realized AFTER the sentiment signal at T).
    mkt_daily["mkt_fwd_vol_22d"] = mkt_daily["mkt_realized_vol"].shift(-ROLL_VOL_WINDOW)

    panel = (mkt_ret.join(mkt_abs).join(mkt_vol).join(mkt_daily).reset_index()
             .sort_values("date").reset_index(drop=True))
    return panel


def build_full_panel(horizons: list[int]) -> pd.DataFrame:
    """Full trading-day panel: market targets + sentiment (news_count=0, sent=NaN on no-news)."""
    panel = build_market_panel(horizons)
    sent = build_market_sentiment()
    panel = panel.merge(sent, on="date", how="left")
    panel["mkt_news_count"] = panel["mkt_news_count"].fillna(0).astype(int)
    panel["has_news"] = (panel["mkt_news_count"] > 0).astype(int)
    return panel


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------
def spearman_lag(x: pd.Series, y: pd.Series, lag: int) -> float:
    """Spearman corr(x[T], y[T+lag]). y shifted to T+lag."""
    df = pd.concat([x, y.shift(-lag)], axis=1).dropna()
    if len(df) < MIN_PAIRS or df.iloc[:, 0].nunique() <= 1 or df.iloc[:, 1].nunique() <= 1:
        return np.nan
    rho, _ = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return float(rho) if not np.isnan(rho) else np.nan


def corr_table(panel: pd.DataFrame, measures: list[str], targets_lag: list[tuple]) -> pd.DataFrame:
    """targets_lag: list of (target_col, lag). Returns DataFrame measures x targets."""
    rows = []
    for m in measures:
        row = {"measure": m}
        for (tcol, lag) in targets_lag:
            row[f"{tcol}+{lag}"] = spearman_lag(panel[m], panel[tcol], lag)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Event study (market level)
# ---------------------------------------------------------------------------
def event_study_news_vs_nonews(panel: pd.DataFrame) -> dict:
    """Compare SUBSEQUENT (forward) volatility & |return| on news vs no-news days.
    All targets are forward-looking (realized AFTER day T), so this is predictive.
    A contemporaneous parkinson reference is included separately, clearly labeled."""
    out = {}
    fwd_targets = [
        ("mkt_fwd_vol_22d", "fwd_vol_22d"),     # vol over [T+1..T+22]
        ("mkt_abs_ret_1d", "fwd_abs_ret_1d"),    # |return| T -> T+1
        ("mkt_vol_avg", "contemporaneous_parkinson"),  # spot vol at T (NOT predictive; reference)
    ]
    for target, label in fwd_targets:
        news = panel.loc[panel["has_news"] == 1, target].dropna()
        none = panel.loc[panel["has_news"] == 0, target].dropna()
        if len(news) < 2 or len(none) < 2:
            out[label] = None
            continue
        u, p = stats.mannwhitneyu(news, none, alternative="two-sided")
        denom = float(np.mean(none))
        out[label] = {
            "n_news_days": int(len(news)), "n_no_news_days": int(len(none)),
            "mean_news": float(np.mean(news)), "mean_no_news": denom,
            "ratio_news_over_none": float(np.mean(news) / denom) if denom > 0 else np.nan,
            "p_value": float(p),
        }
    return out


def event_study_sentiment_direction(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean market forward return on positive vs negative sentiment days."""
    rows = []
    pos = panel[panel["mkt_sent_mean"] > eda.POS_THR]
    neg = panel[panel["mkt_sent_mean"] < eda.NEG_THR]
    for k in HORIZONS:
        col = f"mkt_ret_{k}d"
        pr, nr = pos[col].dropna(), neg[col].dropna()
        _, p = (stats.mannwhitneyu(pr, nr, alternative="two-sided")
                if len(pr) >= 2 and len(nr) >= 2 else (np.nan, np.nan))
        rows.append({
            "horizon": k, "n_pos": len(pr), "n_neg": len(nr),
            "mean_pos_bp": float(np.mean(pr) * 1e4) if len(pr) else np.nan,
            "mean_neg_bp": float(np.mean(nr) * 1e4) if len(nr) else np.nan,
            "spread_bp": float((np.mean(pr) - np.mean(nr)) * 1e4) if len(pr) and len(nr) else np.nan,
            "p_value": float(p) if not np.isnan(p) else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_market(panel: pd.DataFrame, att: pd.DataFrame, sent_dir: pd.DataFrame,
                news_study: dict) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. attention -> forward vol / |return| (lag curve)
    fig, ax = plt.subplots(figsize=(7, 4))
    mrow = att[att["measure"] == "mkt_news_count"].iloc[0]
    vol_ys = [mrow.get(f"mkt_fwd_vol_22d+{l}", np.nan) for l in VOL_LAGS]
    abs_ys = [mrow.get(f"mkt_abs_ret_1d+{l}", np.nan) for l in VOL_LAGS]
    ax.plot(VOL_LAGS, vol_ys, "-o", label="news_count -> fwd_vol_22d", color="#dd8452")
    ax.plot(VOL_LAGS, abs_ys, "-s", label="news_count -> |return_1d|", color="#4c72b0")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel("lag (days)"); ax.set_ylabel("Spearman rho")
    ax.set_title("Market news ATTENTION vs forward volatility / |return|")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "mkt_fig_attention_vol.png", dpi=120); plt.close(fig)

    # 2. sentiment_mean -> market forward return (by horizon)
    fig, ax = plt.subplots(figsize=(7, 4))
    sr = att[att["measure"] == "mkt_sent_mean"]
    xs, ys = [], []
    for k in HORIZONS:
        col = f"mkt_ret_{k}d+0"
        if col in sr.columns:
            xs.append(k); ys.append(sr[col].iloc[0])
    ax.bar(xs, ys, color="#4c72b0")
    ax.axhline(0, c="k", lw=0.8); ax.set_xlabel("forward horizon (days)")
    ax.set_ylabel("Spearman rho"); ax.set_title("Mean daily sentiment -> market forward return")
    fig.tight_layout(); fig.savefig(OUT_DIR / "mkt_fig_sentiment_return.png", dpi=120); plt.close(fig)

    # 3. news vs no-news forward-vol comparison
    if news_study.get("fwd_vol_22d"):
        d = news_study["fwd_vol_22d"]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["no-news days", "news days"], [d["mean_no_news"], d["mean_news"]],
               color=["#7f7f7f", "#d62728"])
        ax.set_ylabel("Mean FORWARD vol (next 22d)")
        ax.set_title(f"Forward vol: news vs no-news (ratio={d['ratio_news_over_none']:.2f}, p={d['p_value']:.2e})")
        fig.tight_layout(); fig.savefig(OUT_DIR / "mkt_fig_news_vs_nonews_vol.png", dpi=120); plt.close(fig)

    # 4. dual-axis time series: news_count vs realized vol (2018+ where coverage is dense)
    sub = panel[panel["date"] >= "2018-01-01"].copy()
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    ax1.plot(sub["date"], sub["mkt_realized_vol"], color="#dd8452", lw=0.8, label="realized vol")
    ax2.plot(sub["date"], sub["mkt_news_count"], color="#4c72b0", lw=0.6, alpha=0.6, label="news count")
    ax1.set_ylabel("realized vol", color="#dd8452")
    ax2.set_ylabel("news count / day", color="#4c72b0")
    ax1.set_title("Market realized volatility vs daily news count (2018+)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "mkt_fig_timeseries.png", dpi=120); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    panel = build_full_panel(HORIZONS)
    panel.to_csv(OUT_DIR / "market_panel.csv", index=False)
    print(f"[info] Panel: {len(panel)} trading days, "
          f"{int(panel['has_news'].sum())} news days ({panel['has_news'].mean()*100:.1f}%).")

    # --- attention + sentiment measures ---
    measures = ["mkt_news_count", "mkt_sent_mean", "mkt_neg_ratio", "mkt_sent_std"]
    # targets: forward returns (lag 0, since ret_kd already encodes T->T+k)
    ret_targets = [(f"mkt_ret_{k}d", 0) for k in HORIZONS]
    abs_targets = [(f"mkt_abs_ret_{k}d", 0) for k in HORIZONS]
    # FORWARD vol (predictive): fwd_vol window starts after T. Parkinson avg = spot vol at T.
    vol_targets = [("mkt_fwd_vol_22d", lag) for lag in VOL_LAGS] + \
                  [("mkt_vol_avg", lag) for lag in VOL_LAGS]
    abs1_targets = [("mkt_abs_ret_1d", lag) for lag in VOL_LAGS]

    att_ret = corr_table(panel, measures, ret_targets)
    att_abs = corr_table(panel, measures, abs_targets)
    att_vol = corr_table(panel, measures, vol_targets)
    att_abs1 = corr_table(panel, measures, abs1_targets)
    full_att = att_ret.merge(att_abs, on="measure").merge(att_vol, on="measure").merge(att_abs1, on="measure")
    full_att.to_csv(OUT_DIR / "market_corr_table.csv", index=False)

    # --- detrended check: first-difference to remove common 2010-2025 trend ---
    detrend = {
        "sent_mean_diff_vs_mkt_ret_1d": spearman_lag(
            panel["mkt_sent_mean"].diff(), panel["mkt_ret_1d"], 0),
        "sent_mean_diff_vs_mkt_ret_5d": spearman_lag(
            panel["mkt_sent_mean"].diff(), panel["mkt_ret_5d"], 0),
        "sent_mean_diff_vs_fwd_vol_22d": spearman_lag(
            panel["mkt_sent_mean"].diff(), panel["mkt_fwd_vol_22d"], 0),
        "news_count_diff_vs_fwd_vol_22d_lag1": spearman_lag(
            panel["mkt_news_count"].diff(), panel["mkt_fwd_vol_22d"], 1),
    }

    news_study = event_study_news_vs_nonews(panel)
    sent_dir = event_study_sentiment_direction(panel)
    sent_dir.to_csv(OUT_DIR / "market_sentiment_direction.csv", index=False)

    # console
    pd.set_option("display.width", 200)
    print("\n=== Correlation: measures -> market forward RETURN (Spearman) ===")
    print(att_ret.round(3).to_string(index=False))
    print("\n=== Correlation: measures -> market VOLATILITY (Spearman) ===")
    print(att_vol.round(3).to_string(index=False))
    print("\n=== Correlation: measures -> |market return| MAGNITUDE ===")
    print(att_abs1.round(3).to_string(index=False))
    print("\n=== News vs no-news day: FORWARD vol / |return| (predictive) ===")
    print(json.dumps(news_study, indent=2, default=str))
    print("\n=== Sentiment direction -> market forward return (bp) ===")
    print(sent_dir.round({"mean_pos_bp": 1, "mean_neg_bp": 1, "spread_bp": 1, "p_value": 4}).to_string(index=False))
    print("\n=== Detrended (first-difference) checks ===")
    print(json.dumps({k: (None if np.isnan(v) else round(v, 3)) for k, v in detrend.items()}, indent=2))

    summary = {
        "units": "returns decimal; *_bp = basis points. Correlations are Spearman rho.",
        "n_trading_days": len(panel),
        "n_news_days": int(panel["has_news"].sum()),
        "pct_news_days": float(round(panel["has_news"].mean() * 100, 1)),
        "corr_measures_to_return": att_ret.to_dict(orient="records"),
        "corr_measures_to_volatility": att_vol.to_dict(orient="records"),
        "corr_measures_to_abs_return": att_abs1.to_dict(orient="records"),
        "news_vs_no_news_next_day": news_study,
        "sentiment_direction_return_bp": sent_dir.to_dict(orient="records"),
        "detrended_checks": {k: (None if np.isnan(v) else float(v)) for k, v in detrend.items()},
        "caveats": [
            "Market sentiment most days based on only 1-2 stocks' news (low breadth).",
            "Sentiment mean and market both trended up 2010-2025; see detrended_checks.",
            "Equal-weighted basket of 30 stocks (no cap data); not cap-weighted VN30 index; "
            "basket composition varies across the sample (listings/delistings).",
            "Multiple testing: ~80 correlations across 4 corr tables reported WITHOUT correction; "
            "expect ~4 |rho| with nominal p<0.05 by chance under the null. Treat as exploratory.",
            "Sample mismatch: mkt_news_count corr uses all days (75% tied at 0) while other measures "
            "use ~1197 news days only; rows are NOT directly comparable.",
            "mkt_fwd_vol_22d is a forward-looking target (vol realized over [T+1..T+22]); "
            "mkt_vol_avg (Parkinson) is contemporaneous spot vol.",
        ],
    }
    (OUT_DIR / "market_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    plot_market(panel, full_att, sent_dir, news_study)
    print(f"\n[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
