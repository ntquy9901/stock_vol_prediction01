"""
Full-corpus sentiment re-test (the real NO-GO challenge).

Scores the external crawl project's unified news corpus (~21k articles, 2001-2026)
with the project's XLM-R sentiment scorer, splits REAL NEWS (cafef/ssi/vndirect)
from ANALYST REPORTS (brokerages), aggregates daily, and re-runs the market-level
relationship analysis (year-matched) to see whether richer / genuinely-different
news overturns the prior NO-GO verdict.

Reads (read-only) from the SEPARATE crawl project:
    D:/bmad-projects/crawl_data/aggregated/unified_articles.csv
Reuses helpers from sentiment_price_eda + sentiment_market_eda.

Scoring is cached to results/.../fullcorpus/scored_articles.csv (re-score only with
--force). Run:
    python baselines/2026-07-11_sentiment_price_eda/code/sentiment_fullcorpus_eda.py [--force]
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
# src/sentiment_baseline on path so we can import the scorer directly
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "sentiment_baseline"))
import sentiment_price_eda as eda  # noqa: E402
import sentiment_market_eda as mkt  # noqa: E402
import phobert_scorer  # noqa: E402  (project's XLM-R scorer, cached)

OUT_DIR = eda.OUT_DIR / "fullcorpus"
OUT_DIR.mkdir(parents=True, exist_ok=True)
UNIFIED = Path("D:/bmad-projects/crawl_data/aggregated/unified_articles.csv")
SCORED_CACHE = OUT_DIR / "scored_articles.csv"

REAL_NEWS_SOURCES = {"cafef", "ssi", "vndirect"}   # exact (lowercase) = journalism w/ body
HORIZONS = [1, 3, 5]
MIN_PAIRS = 30


# ---------------------------------------------------------------------------
# Load + score
# ---------------------------------------------------------------------------
def _score_text(r) -> str:
    """Score text = title + lead ONLY (short -> fast CPU inference; title+lead carry
    the sentiment signal). Body is intentionally NOT used: full-body articles would
    hit the 256-token cap and ~10x the scoring time on CPU for marginal signal gain."""
    title = str(r.get("title", "") or "")
    lead = r.get("lead")
    if pd.notna(lead) and len(str(lead)) > 5:
        return (title + " " + str(lead)).strip()
    return title.strip()


def score_articles(force: bool = False, analyst_sample: int = 4000,
                   max_length: int = 128, batch_size: int = 64) -> pd.DataFrame:
    """Score with a direct HF pipeline (max_length=128 -> fast on CPU).
    Scores ALL real-news (cafef/ssi/vndirect) fully + a reproducible sample of the
    analyst/brokerage titles, which is enough for a representative daily signal."""
    if SCORED_CACHE.exists() and not force:
        print(f"[info] loading cached scores: {SCORED_CACHE}", flush=True)
        return pd.read_csv(SCORED_CACHE, parse_dates=["date"])

    df = pd.read_csv(UNIFIED)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df["score_text"] = df.apply(_score_text, axis=1)
    df = df[df["score_text"].str.len() > 0].reset_index(drop=True)
    df["is_real_news"] = df["source"].astype(str).isin(REAL_NEWS_SOURCES)

    real = df[df["is_real_news"]]
    other = df[~df["is_real_news"]].sample(n=min(analyst_sample, len(df[~df["is_real_news"]])),
                                           random_state=42)
    to_score = pd.concat([real, other]).sort_values("date").reset_index(drop=True)
    print(f"[info] scoring {len(to_score)} articles (real_news={len(real)} + "
          f"analyst_sample={len(other)}) with XLM-R max_length={max_length}...", flush=True)

    from transformers import pipeline, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(phobert_scorer.DEFAULT_MODEL, use_fast=False)
    pipe = pipeline("sentiment-analysis", model=phobert_scorer.DEFAULT_MODEL,
                    tokenizer=tok, truncation=True, max_length=max_length)
    texts = to_score["score_text"].tolist()
    scores = []
    for i in range(0, len(texts), batch_size):
        res = pipe(texts[i:i + batch_size], batch_size=batch_size,
                   truncation=True, max_length=max_length)
        scores.extend(phobert_scorer._label_to_score(r.get("label", ""), float(r.get("score", 0.0)))
                      for r in res)
        if i % 1000 == 0:
            print(f"  ...{i}/{len(texts)}", flush=True)
    to_score["sentiment"] = [max(-1.0, min(1.0, s)) for s in scores]

    keep = ["unified_id", "source", "date", "title", "sentiment", "is_real_news"]
    to_score[keep].to_csv(SCORED_CACHE, index=False)
    print(f"[info] cached {len(to_score)} scores -> {SCORED_CACHE}", flush=True)
    return to_score[keep]


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------
def daily_sentiment(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    scored["date"] = pd.to_datetime(scored["date"]).dt.normalize()
    real = scored[scored["is_real_news"]]

    g_all = scored.groupby("date")["sentiment"].agg(["mean", "size"])
    g_real = real.groupby("date")["sentiment"].agg(["mean", "size"])

    out = g_all.rename(columns={"mean": "sent_all", "size": "n_all"})
    out["sent_realnews"] = g_real["mean"]
    out["n_realnews"] = g_real["size"].fillna(0).astype(int)
    return out.reset_index()


def _spearman(x, y):
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < MIN_PAIRS or df.iloc[:, 0].nunique() <= 1 or df.iloc[:, 1].nunique() <= 1:
        return np.nan
    rho, _ = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return float(rho) if not np.isnan(rho) else np.nan


# ---------------------------------------------------------------------------
# Year-matched vol comparison (news vs no-news WITHIN same year)
# ---------------------------------------------------------------------------
def yearmatched_vol(panel: pd.DataFrame, sent_col: str, newsflag: str) -> dict:
    pan = panel.copy()
    pan["year"] = pan["date"].dt.year
    none_by_year = (pan[pan[newsflag] == 0].groupby("year")["mkt_fwd_vol_22d"]
                    .apply(lambda s: s.dropna().to_numpy()))
    news_rows = pan.loc[pan[newsflag] == 1, ["year", "mkt_fwd_vol_22d"]].dropna()
    per_year = []
    for y, grp in news_rows.groupby("year"):
        base = none_by_year.get(y, np.array([]))
        if len(grp) >= 2 and len(base) >= 2 and np.mean(base) > 0:
            per_year.append(float(grp["mkt_fwd_vol_22d"].mean() / np.mean(base)))
    return {
        "n_years": len(per_year),
        "median_ratio": float(np.median(per_year)) if per_year else np.nan,
        "mean_ratio": float(np.mean(per_year)) if per_year else np.nan,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    force = "--force" in sys.argv
    scored = score_articles(force=force)
    print(f"[info] scored: {len(scored)} articles | "
          f"real_news={int(scored['is_real_news'].sum())} "
          f"({scored['is_real_news'].mean()*100:.1f}%)")
    print(f"[info] sentiment: mean={scored['sentiment'].mean():.3f} "
          f"median={scored['sentiment'].median():.3f} "
          f"neg(<-0.2)={int((scored['sentiment']<-0.2).sum())} "
          f"pos(>0.2)={int((scored['sentiment']>0.2).sum())}")

    daily = daily_sentiment(scored)
    daily.to_csv(OUT_DIR / "daily_sentiment.csv", index=False)

    # market price/vol targets (independent of news source)
    panel = mkt.build_market_panel(HORIZONS)
    panel = panel.merge(daily, on="date", how="left")
    panel["n_all"] = panel["n_all"].fillna(0).astype(int)
    panel["n_realnews"] = panel["n_realnews"].fillna(0).astype(int)
    panel["has_news_all"] = (panel["n_all"] > 0).astype(int)
    panel["has_news_real"] = (panel["n_realnews"] > 0).astype(int)
    panel.to_csv(OUT_DIR / "fullcorpus_panel.csv", index=False)

    # --- correlation: daily sentiment -> market forward return / vol ---
    measures = {
        "sent_all (all news)": (panel["sent_all"], panel["has_news_all"]),
        "sent_realnews (cafef/ssi/vndirect)": (panel["sent_realnews"], panel["has_news_real"]),
    }
    targets = [(f"mkt_ret_{k}d", 0) for k in HORIZONS] + \
              [("mkt_fwd_vol_22d", 0), ("mkt_abs_ret_1d", 0)]
    corr_rows = []
    for name, (scol, _) in measures.items():
        row = {"measure": name, "n_days_with_sent": int(scol.notna().sum())}
        for tcol, lag in targets:
            y = panel[tcol]
            row[f"{tcol}+{lag}"] = _spearman(scol, y)
        corr_rows.append(row)
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT_DIR / "corr_table.csv", index=False)

    # --- year-matched vol: news days vs no-news days, all vs realnews ---
    ym = {
        "all_news": yearmatched_vol(panel, "sent_all", "has_news_all"),
        "real_news": yearmatched_vol(panel, "sent_realnews", "has_news_real"),
    }

    # --- sentiment-direction event study (market forward return) ---
    def dir_study(scol, flag):
        out = []
        for k in HORIZONS:
            col = f"mkt_ret_{k}d"
            pos = panel.loc[panel[scol] > 0.2, col].dropna()
            neg = panel.loc[panel[scol] < -0.2, col].dropna()
            _, p = (stats.mannwhitneyu(pos, neg, alternative="two-sided")
                    if len(pos) >= 2 and len(neg) >= 2 else (np.nan, np.nan))
            out.append({
                "horizon": k, "n_pos": len(pos), "n_neg": len(neg),
                "mean_pos_bp": float(np.mean(pos) * 1e4) if len(pos) else np.nan,
                "mean_neg_bp": float(np.mean(neg) * 1e4) if len(neg) else np.nan,
                "spread_bp": float((np.mean(pos) - np.mean(neg)) * 1e4) if len(pos) and len(neg) else np.nan,
                "p_value": float(p) if not np.isnan(p) else np.nan,
            })
        return pd.DataFrame(out)

    dir_all = dir_study("sent_all", "has_news_all")
    dir_real = dir_study("sent_realnews", "has_news_real")

    # console
    pd.set_option("display.width", 200)
    print("\n=== Daily sentiment -> market forward return / vol (Spearman, year-aware via panel) ===")
    print(corr.round(3).to_string(index=False))
    print("\n=== Year-matched forward-vol ratio (news vs no-news WITHIN year; <1 = news calmer) ===")
    print(json.dumps(ym, indent=2))
    print("\n=== Sentiment DIRECTION -> market forward return: ALL news ===")
    print(dir_all.round({"mean_pos_bp": 1, "mean_neg_bp": 1, "spread_bp": 1, "p_value": 4}).to_string(index=False))
    print("\n=== Sentiment DIRECTION -> market forward return: REAL news only ===")
    print(dir_real.round({"mean_pos_bp": 1, "mean_neg_bp": 1, "spread_bp": 1, "p_value": 4}).to_string(index=False))

    summary = {
        "n_articles_scored": len(scored),
        "n_real_news": int(scored["is_real_news"].sum()),
        "sentiment_distribution": {
            "mean": float(scored["sentiment"].mean()),
            "neg_lt_-0.2": int((scored["sentiment"] < -0.2).sum()),
            "pos_gt_0.2": int((scored["sentiment"] > 0.2).sum()),
        },
        "corr_sentiment_to_market": corr.to_dict(orient="records"),
        "yearmatched_vol_ratio": ym,
        "direction_return_all_bp": dir_all.to_dict(orient="records"),
        "direction_return_realnews_bp": dir_real.to_dict(orient="records"),
        "caveats": [
            "XLM-R scores title (+lead/body-chunk), truncated to 256 tokens.",
            "sent_realnews uses only cafef/ssi/vndirect (~6.5k journalism articles w/ body).",
            "Year-matched ratio controls the coverage-trend confound found in the prior market analysis.",
        ],
    }
    (OUT_DIR / "fullcorpus_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # plot: sentiment distribution all vs realnews
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    axs[0].hist(scored["sentiment"], bins=40, color="#4c72b0", alpha=0.7, label="all")
    axs[0].hist(scored.loc[scored["is_real_news"], "sentiment"], bins=40, color="#dd8452", alpha=0.7, label="real news")
    axs[0].set_title("Sentiment score distribution"); axs[0].set_xlabel("sentiment"); axs[0].legend()
    # corr bar: realnews vs all for return/vol targets
    c = corr.set_index("measure")
    cols = [f"mkt_ret_{k}d+0" for k in HORIZONS] + ["mkt_fwd_vol_22d+0"]
    x = np.arange(len(cols))
    axs[1].bar(x - 0.2, [c.loc["sent_all (all news)", col] for col in cols], 0.4, label="all news", color="#4c72b0")
    axs[1].bar(x + 0.2, [c.loc["sent_realnews (cafef/ssi/vndirect)", col] for col in cols], 0.4, label="real news", color="#dd8452")
    axs[1].axhline(0, c="k", lw=0.8); axs[1].set_xticks(x)
    axs[1].set_xticklabels([f"ret{k}d" for k in HORIZONS] + ["fwd_vol"], rotation=20)
    axs[1].set_ylabel("Spearman rho"); axs[1].set_title("Daily sentiment -> market (year-aware panel)")
    axs[1].legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "fullcorpus_fig.png", dpi=120); plt.close(fig)
    print(f"\n[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
