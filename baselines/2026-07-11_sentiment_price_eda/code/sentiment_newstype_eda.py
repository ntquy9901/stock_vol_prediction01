"""
News-TYPE refinement of the sentiment<->market analysis.

The corpus is ~89% periodic analyst/earnings news and ~0.5% event news, so an
event-vs-periodic vol comparison is infeasible (too few events). Instead this
script tests the cleanest directional signal available in a periodic corpus:
**analyst RATING direction** (BUY/OUTPERFORM vs SELL/HOLD) -> forward return,
plus a composition audit and a per-category forward-vol check.

Reuses helpers from sentiment_price_eda (per-stock forward returns) and
sentiment_market_eda (market panel with forward vol).

Run:
    python baselines/2026-07-11_sentiment_price_eda/code/sentiment_newstype_eda.py
Outputs -> results/2026-07-11_sentiment_price_eda/newstype/
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
import sentiment_price_eda as eda  # noqa: E402
import sentiment_market_eda as mkt  # noqa: E402

OUT_DIR = eda.OUT_DIR / "newstype"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- keyword lexicons (lowercase substring match on Vietnamese/English titles) ---
RATING_POS = ["mua", "khả quan", "kha quan", "tích cực", "tich cuc", "ưu vượt",
              "uu vuot", "outperform", "overweight", "accumulate", "strong buy"]
RATING_NEG = ["bán", "kém quan", "kem quan", "tiêu cực", "tieu cuc", "underperform",
              "underweight", "tránh", "né tránh", "reduce", "bán khống", "sell"]
RATING_NEU = ["giữ", "nắm giữ", "trung lập", "trung lap", "neutral", "hold",
              "phù hợp thị trường", "market perform", "bằng xét"]
EARN_KW = ["kqkd", "kết quả kinh doanh", "ket qua kinh doanh", "báo cáo nhanh",
           "báo cáo ngắn", "kết quả", "quatrin", "q1", "q2", "q3", "q4",
           "1q", "2q", "3q", "4q", "quý"]
EVENT_KW = ["sáp nhập", "sáp nhâp", "m&a", "mua lại", "tăng vốn", "phát hành",
            "tố kiện", "vụ kiện", "phạt", "cách chức", "downgrade", "hạ cấp",
            "cháy", "sự cố", "vi phạm", "đình chỉ", "phá sản", "trúng thầu", "chuyển nhượng"]
# Compounds where 'bán'/'mua' mean business activity, not a rating. Removed before
# rating extraction so e.g. "Chi phí bán hàng" no longer flips a BUY title to SELL.
NONRATING_COMPOUNDS = [
    "bán lẻ", "bán buôn", "bán hàng", "bán nhà", "bán vốn", "bán cổ phiếu",
    "chào bán", "doanh số bán", "giá bán", "bán cho", "bán phá", "bán thanh lý",
    "bán đấu giá", "đại bán", "bán máu", "thu mua", "mua sắm", "chào mua",
]


def _clean_for_rating(title: str) -> str:
    t = title.lower()
    for c in NONRATING_COMPOUNDS:
        t = t.replace(c, " ")
    return t


def extract_rating(title: str, is_event: bool) -> str | None:
    """Direction of an analyst rating. Skipped for event/M&A titles (where
    'mua'/'bán' mean transactions). Non-rating compounds are stripped first so
    'bán hàng'/'bán lẻ' cannot flip a BUY title to SELL."""
    if is_event:
        return None
    t = _clean_for_rating(title)
    if any(k in t for k in RATING_NEG):
        return "NEG"
    if any(k in t for k in RATING_POS):
        return "POS"
    if any(k in t for k in RATING_NEU):
        return "NEU"
    return None


def classify(title: str) -> dict:
    t = title.lower()
    is_event = any(k in t for k in EVENT_KW)
    return {
        "rating": extract_rating(title, is_event),
        "is_earnings": any(k in t for k in EARN_KW),
        "is_event": is_event,
    }


# ---------------------------------------------------------------------------
# Load + classify all events
# ---------------------------------------------------------------------------
def load_classified_events() -> pd.DataFrame:
    rows = []
    for t in eda.load_tickers():
        sp = eda.SENT_DIR / f"{t}_sentiment.csv"
        if not sp.exists():
            continue
        df = pd.read_csv(sp, parse_dates=["date"])
        ev = df[df["news_count_1d"].astype(float) > 0].copy()
        for _, r in ev.iterrows():
            title = str(r.get("news_titles", ""))
            c = classify(title)
            rows.append({
                "ticker": t, "date": r["date"], "sentiment_1d": float(r["sentiment_1d"]),
                "title": title, **c,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-stock: rating direction -> forward return
# ---------------------------------------------------------------------------
def rating_return_study(events: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Merge rating events with per-stock forward returns; compare POS vs NEG returns."""
    recs = []
    for ticker, sub in events.groupby("ticker"):
        pp = eda.PRICE_DIR / f"{ticker}_ohlcv.csv"
        if not pp.exists():
            continue
        price = pd.read_csv(pp, parse_dates=["date"])
        fr = eda.compute_forward_returns(price, horizons)
        m = sub.merge(fr, on="date", how="inner")
        recs.append(m)
    if not recs:
        return pd.DataFrame()
    pooled = pd.concat(recs, ignore_index=True)
    rows = []
    for k in horizons:
        col = f"ret_{k}d"
        pos = pooled.loc[pooled["rating"] == "POS", col].dropna().to_numpy()
        neg = pooled.loc[pooled["rating"] == "NEG", col].dropna().to_numpy()
        neu = pooled.loc[pooled["rating"] == "NEU", col].dropna().to_numpy()
        _, p = (stats.mannwhitneyu(pos, neg, alternative="two-sided")
                if len(pos) >= 2 and len(neg) >= 2 else (np.nan, np.nan))
        rows.append({
            "horizon": k,
            "n_pos": len(pos), "n_neg": len(neg), "n_neu": len(neu),
            "mean_pos_bp": float(np.mean(pos) * 1e4) if len(pos) else np.nan,
            "mean_neg_bp": float(np.mean(neg) * 1e4) if len(neg) else np.nan,
            "mean_neu_bp": float(np.mean(neu) * 1e4) if len(neu) else np.nan,
            "spread_pos_neg_bp": float((np.mean(pos) - np.mean(neg)) * 1e4)
            if len(pos) and len(neg) else np.nan,
            "p_value": float(p) if not np.isnan(p) else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Market: news category -> forward volatility (pooled AND year-matched)
# ---------------------------------------------------------------------------
NEWS_CATS = {
    "all_news": lambda e: e,
    "rating_POS": lambda e: e[e["rating"] == "POS"],
    "rating_NEG": lambda e: e[e["rating"] == "NEG"],
    "earnings_update": lambda e: e[e["is_earnings"]],
    "event_shock": lambda e: e[e["is_event"]],
}


def category_vol_study(events: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """For each news category: pooled ratio vs no-news days AND year-matched ratio
    (news-day vol / no-news-day vol WITHIN the same year) to control for the secular
    coverage trend (news is sparse pre-2018, dense post-2018 = different vol regimes)."""
    pan = panel.copy()
    pan["year"] = pan["date"].dt.year
    none_by_year = (pan[pan["has_news"] == 0].groupby("year")["mkt_fwd_vol_22d"]
                    .apply(lambda s: s.dropna().to_numpy()))

    base_none = pan.loc[pan["has_news"] == 0, "mkt_fwd_vol_22d"].dropna().to_numpy()
    base_mean = float(np.mean(base_none)) if len(base_none) else np.nan

    out = []
    for name, fn in NEWS_CATS.items():
        sub = fn(events)
        days = sub["date"].drop_duplicates()
        news_rows = pan.loc[pan["date"].isin(days), ["year", "mkt_fwd_vol_22d"]].dropna()

        # year-matched: per-year ratio of (mean news-day vol) / (mean no-news-day vol)
        per_year = []
        for y, grp in news_rows.groupby("year"):
            base = none_by_year.get(y, np.array([]))
            if len(grp) >= 2 and len(base) >= 2 and np.mean(base) > 0:
                per_year.append(float(np.mean(grp["mkt_fwd_vol_22d"]) / np.mean(base)))
        ratios = np.array(per_year) if per_year else np.array([])

        pooled = news_rows["mkt_fwd_vol_22d"].to_numpy()
        _, p = (stats.mannwhitneyu(pooled, base_none, alternative="two-sided")
                if len(pooled) >= 2 and len(base_none) >= 2 else (np.nan, np.nan))
        out.append({
            "category": name, "n_days": int(len(pooled)),
            "pooled_ratio_vs_no_news": float(np.mean(pooled) / base_mean) if base_mean > 0 and len(pooled) else np.nan,
            "yearmatched_median_ratio": float(np.median(ratios)) if len(ratios) else np.nan,
            "yearmatched_mean_ratio": float(np.mean(ratios)) if len(ratios) else np.nan,
            "n_years_matched": int(len(ratios)),
            "p_value_pooled": float(p) if not np.isnan(p) else np.nan,
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    events = load_classified_events()
    events.to_csv(OUT_DIR / "classified_events.csv", index=False)
    print(f"[info] {len(events)} events classified.")

    # composition
    comp = {
        "total": len(events),
        "with_rating": int(events["rating"].notna().sum()),
        "rating_breakdown": {k: int(v) for k, v in events["rating"].value_counts(dropna=False).items()},
        "earnings_update": int(events["is_earnings"].sum()),
        "event_shock": int(events["is_event"].sum()),
        "unclassified": int((~events["is_earnings"] & ~events["is_event"]
                             & events["rating"].isna()).sum()),
    }
    print("[info] Composition:")
    print(json.dumps(comp, indent=2, default=str))

    rr = rating_return_study(events, [1, 3, 5])
    rr.to_csv(OUT_DIR / "rating_return.csv", index=False)
    panel = mkt.build_full_panel([1, 3, 5])
    cv = category_vol_study(events, panel)
    cv.to_csv(OUT_DIR / "category_vol.csv", index=False)

    print("\n=== Rating direction -> per-stock forward return (bp) ===")
    print(rr.round({"mean_pos_bp": 1, "mean_neg_bp": 1, "mean_neu_bp": 1,
                    "spread_pos_neg_bp": 1, "p_value": 4}).to_string(index=False))
    print("\n=== News category -> market FORWARD vol (pooled vs year-matched) ===")
    print(cv.round({"pooled_ratio_vs_no_news": 3, "yearmatched_median_ratio": 3,
                     "yearmatched_mean_ratio": 3, "p_value_pooled": 4}).to_string(index=False))

    summary = {
        "units": "returns decimal (*_bp = basis points); vol is forward 22-day realized.",
        "composition": comp,
        "rating_return_per_stock_bp": rr.to_dict(orient="records"),
        "category_forward_vol_vs_no_news": cv.to_dict(orient="records"),
        "interpretation_notes": [
            "Corpus is ~89% periodic (analyst ratings + earnings updates); event/shock news is ~0.5%.",
            "An event-vs-periodic vol comparison is infeasible (too few event news).",
            "NEG ratings are very rare (analyst optimism bias) AND typically issued after price "
            "drops, so any higher forward return reflects mean reversion, not predictive value.",
            "yearmatched_median_ratio controls for the coverage trend (news sparse pre-2018, dense "
            "post-2018 = different vol regimes); compare to pooled_ratio to see the confound.",
            "Multiple testing: ~8 Mann-Whitney tests across categories/horizons, uncorrected; "
            "treat p-values as exploratory.",
        ],
    }
    (OUT_DIR / "newstype_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # --- plots ---
    plt.style.use("seaborn-v0_8-whitegrid")
    rb = comp["rating_breakdown"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    cats = ["POS", "NEU", "NEG", "UNK"]
    vals = [rb.get(c, 0) for c in cats]
    axs[0].bar(cats, vals, color=["#2ca02c", "#7f7f7f", "#d62728", "#c7c7c7"])
    axs[0].set_title("Analyst rating distribution (rated events)")
    axs[0].set_ylabel("event count")
    if not rr.empty:
        x = np.arange(len(rr))
        axs[1].bar(x - 0.2, rr["mean_pos_bp"], 0.4, label="POS rating", color="#2ca02c")
        axs[1].bar(x + 0.2, rr["mean_neg_bp"], 0.4, label="NEG rating", color="#d62728")
        axs[1].set_xticks(x); axs[1].set_xticklabels([f"T+{int(h)}" for h in rr["horizon"]])
        axs[1].axhline(0, c="k", lw=0.8)
        axs[1].set_title("Per-stock forward return by rating direction")
        axs[1].set_ylabel("mean forward return (bp)"); axs[1].legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "newstype_fig.png", dpi=120); plt.close(fig)
    print(f"\n[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
