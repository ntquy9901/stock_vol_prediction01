"""EDA: does market-wide news VOLUME correlate with market-wide volatility movement?

Answers the question "tin tuc chung chung/thi truong co nhieu khong, no anh huong gia ra sao"
with a model-free statistical correlation instead of training anything new. Two data sources
already exist and only need to be joined by date:

  - `data/processed/{TICKER}_processed.csv` (32 files) -> per-ticker daily Parkinson volatility,
    already computed, no new pipeline needed.
  - `data/features/dual_group_news_panel.parquet` -> per-(ticker, date) topic-count columns
    (`kq_topic_*_count`, `th_topic_*_count`), already ticker-tagged. Summed across all 32 tickers
    and both source groups for a given date, this is a proxy for "how much VN30-relevant news
    volume existed market-wide that day" (NOT the full untagged raw crawl -- see caveat below).

Method:
  1. Build `market_avg_change[date]` = cross-sectional mean, across the 32 tickers, of each
     ticker's day-over-day change in Parkinson volatility (skipna; some tickers may be missing on
     some trading days). This is a simple market-wide "did volatility broadly rise or fall today"
     signal, independent of any model.
  2. Build `news_volume[date]` = sum of every `*_topic_*_count` column across all 32 tickers on
     that date (market-wide topic-tagged article count that day).
  3. Correlate `news_volume[date]` against `market_avg_change[date]` (same day, contemporaneous)
     and `market_avg_change[date+1 trading day]` (next day) via Pearson r.
  4. Split trading days into high-volume (top quartile of `news_volume`) vs low-volume (bottom
     quartile, mostly zero) and Welch t-test the next-day `|market_avg_change|` between the two
     groups (absolute value, since the question is "does more news precede bigger moves in either
     direction", not "does it push volatility up specifically").

Caveat (stated up front): `news_volume` here is the ALREADY ticker-tagged, topic-flagged subset
(only articles that (a) came from the 12 sources in `GROUP_SOURCES` and (b) explicitly mention a
VN30 ticker and (c) matched one of the 7 topic keyword categories) -- NOT the full raw crawl
(`crawl_data/data/`, ~9.3M rows across dozens of untagged outlets). This is a much narrower, cheaper
proxy; a true "total market news volume" measure would require parsing the full untagged crawl by
date, which this script does not do. Single-seed cross-sectional data, days are not independent
(volatility is autocorrelated) -- treat p-values as screening signals, not rigorous proof.

Run: python analyze_market_news_volume_correlation.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

NEWS_PANEL_PATH = _ROOT / "data" / "features" / "dual_group_news_panel.parquet"
PROCESSED_DIR = _ROOT / "data" / "processed"


def load_market_volatility(processed_dir: Path) -> pd.DataFrame:
    """[date, ticker, parkinson_volatility] long frame from every `{TICKER}_processed.csv`."""
    frames = []
    for path in sorted(processed_dir.glob("*_processed.csv")):
        ticker = path.stem.replace("_processed", "")
        df = pd.read_csv(path, usecols=["date", "parkinson_volatility"])
        parsed = pd.to_datetime(df["date"])
        if parsed.dt.tz is not None:
            # VPB/VRE processed CSVs carry a +07:00 offset unlike the other 30 tickers (tz-naive)
            # -- strip it so all 32 tickers pivot on a common tz-naive date index.
            parsed = parsed.dt.tz_localize(None)
        df["date"] = parsed.dt.normalize()
        df["ticker"] = ticker
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "parkinson_volatility"])
    return pd.concat(frames, ignore_index=True)


def market_avg_change(vol_long: pd.DataFrame) -> pd.Series:
    """Cross-sectional mean of each ticker's day-over-day volatility change, indexed by date."""
    wide = vol_long.pivot(index="date", columns="ticker", values="parkinson_volatility").sort_index()
    change = wide.diff()
    return change.mean(axis=1, skipna=True)


def load_news_volume(panel_path: Path) -> pd.Series:
    """Market-wide topic-tagged news volume per date (sum across all tickers + topic columns)."""
    df = pd.read_parquet(panel_path)
    topic_cols = [c for c in df.columns if c.endswith("_count") and ("topic" in c)]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    daily = df.groupby("date")[topic_cols].sum(min_count=1).fillna(0.0)
    return daily.sum(axis=1)


def build_joined_frame(market_change: pd.Series, news_volume: pd.Series) -> pd.DataFrame:
    joined = pd.DataFrame({
        "market_avg_change": market_change,
        "news_volume": news_volume.reindex(market_change.index).fillna(0.0),
    }).sort_index()
    joined["market_avg_change_next"] = joined["market_avg_change"].shift(-1)
    joined["market_avg_change_abs_next"] = joined["market_avg_change_next"].abs()
    return joined.dropna(subset=["market_avg_change", "market_avg_change_next"])


def correlate(joined: pd.DataFrame) -> dict:
    r_same, p_same = stats.pearsonr(joined["news_volume"], joined["market_avg_change"])
    r_next, p_next = stats.pearsonr(joined["news_volume"], joined["market_avg_change_next"])
    r_next_abs, p_next_abs = stats.pearsonr(joined["news_volume"], joined["market_avg_change_abs_next"])

    q_high = joined["news_volume"].quantile(0.75)
    q_low = joined["news_volume"].quantile(0.25)
    high = joined[joined["news_volume"] >= q_high]
    low = joined[joined["news_volume"] <= q_low]
    t_stat, p_ttest = stats.ttest_ind(
        high["market_avg_change_abs_next"], low["market_avg_change_abs_next"], equal_var=False
    )

    return {
        "n_days": int(len(joined)),
        "pearson_same_day": {"r": float(r_same), "p": float(p_same)},
        "pearson_next_day": {"r": float(r_next), "p": float(p_next)},
        "pearson_next_day_abs_change": {"r": float(r_next_abs), "p": float(p_next_abs)},
        "high_vs_low_volume_next_day_abs_change": {
            "high_volume_threshold": float(q_high),
            "low_volume_threshold": float(q_low),
            "n_high": int(len(high)),
            "n_low": int(len(low)),
            "mean_abs_change_high": float(high["market_avg_change_abs_next"].mean()),
            "mean_abs_change_low": float(low["market_avg_change_abs_next"].mean()),
            "welch_t": float(t_stat),
            "p_value": float(p_ttest),
        },
    }


def plot_scatter(joined: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(joined["news_volume"], joined["market_avg_change_abs_next"], alpha=0.4, s=12)
    ax.set_xlabel("News volume (topic-tagged mentions, market-wide, that day)")
    ax.set_ylabel("|market_avg_change| next trading day")
    ax.set_title("Market-wide news volume vs next-day volatility move (VN30)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run(processed_dir: Path = PROCESSED_DIR, panel_path: Path = NEWS_PANEL_PATH,
        out_dir: Path | None = None) -> dict:
    vol_long = load_market_volatility(processed_dir)
    if vol_long.empty:
        raise ValueError(f"No `*_processed.csv` files found under {processed_dir}")
    change = market_avg_change(vol_long)
    news_volume = load_news_volume(panel_path)
    joined = build_joined_frame(change, news_volume)
    result = correlate(joined)

    if out_dir is None:
        out_dir = _ROOT / "results" / f"market_news_volume_correlation_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    plot_scatter(joined, out_dir / "scatter.png")
    joined.to_parquet(out_dir / "joined_daily_series.parquet")
    return result


if __name__ == "__main__":
    res = run()
    print(json.dumps(res, indent=2, ensure_ascii=False))
