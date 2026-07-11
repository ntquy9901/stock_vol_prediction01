"""
Sentiment <-> Price relationship EDA for VN30 (event-study + lead-lag).

Self-contained analysis script. Bootstraps sys.path because the baseline folder
name contains dashes and is not importable via `python -m`.

Run:
    python baselines/2026-07-11_sentiment_price_eda/code/sentiment_price_eda.py

Outputs -> results/2026-07-11_sentiment_price_eda/

Units convention: all returns are DECIMAL (0.03 means 3%). Multiplying by 1e4
gives basis points (bp). Reported *_bp fields are in basis points.
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

# ---------------------------------------------------------------------------
# Paths (project_root = 3 levels up from this file: code/ -> baseline/ -> baselines/ -> root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "data"
PRICE_DIR = DATA / "raw" / "prices"            # <TICKER>_ohlcv.csv
SENT_DIR = DATA / "sentiment_baseline"         # <TICKER>_sentiment.csv
VOL_DIR = DATA / "processed" / "vn30_only"     # <TICKER>_processed.csv
OUT_DIR = PROJECT_ROOT / "results" / "2026-07-11_sentiment_price_eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HORIZONS = [1, 2, 3, 5, 10]
PRIMARY_HORIZON = 5          # pre-registered primary horizon (project target = 5-day)
POS_THR = 0.2                # sentiment_1d > POS_THR  -> positive
NEG_THR = -0.2               # sentiment_1d < NEG_THR  -> negative
MIN_EVENTS = 5               # skip ticker if fewer event days
WINS_LOW, WINS_HIGH = 1, 99  # percentile winsorize for forward returns (event-level)
ALPHA = 0.05                 # family-wise significance level (Bonferroni-corrected below)
MIN_SPREAD_BP = 30           # minimum |pos-neg| mean spread for GO (heuristic ~0.3%/period)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def load_tickers() -> list[str]:
    return sorted(p.stem.replace("_ohlcv", "") for p in PRICE_DIR.glob("*_ohlcv.csv"))


def winsorize(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clip array to [low, high] percentiles (by value, over non-NaN). NaNs preserved."""
    a = arr.astype(float).copy()
    mask = ~np.isnan(a)
    if mask.sum() == 0:
        return a
    lo, hi = np.percentile(a[mask], [low, high])
    a[a < lo] = lo
    a[a > hi] = hi
    return a


def compute_forward_returns(price: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Pure shift: ret_kd[T] = close[T+k]/close[T]-1. No winsorize here (done at event level).
    Guards zero/negative closes -> NaN."""
    df = price.sort_values("date").reset_index(drop=True).copy()
    close = df["close"].astype(float).to_numpy()
    close = np.where(close <= 0, np.nan, close)
    n = len(close)
    out = {"date": df["date"].to_numpy()}
    for k in horizons:
        fr = np.full(n, np.nan)
        if k < n:
            with np.errstate(divide="ignore", invalid="ignore"):
                fr[: n - k] = close[k:] / close[: n - k] - 1.0
        out[f"ret_{k}d"] = fr
    return pd.DataFrame(out)


def winsorize_events(events: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Winsorize forward returns per horizon over the POOLED event days (not the full
    trading series), so non-event-day distribution cannot bias event statistics."""
    ev = events.copy()
    for k in horizons:
        col = f"ret_{k}d"
        ev[col] = winsorize(ev[col].to_numpy(), WINS_LOW, WINS_HIGH)
    return ev


def classify(s: float) -> str:
    if pd.isna(s):
        return "neu"
    if s > POS_THR:
        return "pos"
    if s < NEG_THR:
        return "neg"
    return "neu"


def build_ticker_events(ticker: str) -> pd.DataFrame | None:
    """Merge price forward-returns with sentiment, keep event days (news_count>0)."""
    price_path = PRICE_DIR / f"{ticker}_ohlcv.csv"
    sent_path = SENT_DIR / f"{ticker}_sentiment.csv"
    if not (price_path.exists() and sent_path.exists()):
        return None

    price = pd.read_csv(price_path, parse_dates=["date"])
    sent = pd.read_csv(sent_path, parse_dates=["date"])
    fr = compute_forward_returns(price, HORIZONS)

    m = sent.merge(fr, on="date", how="inner")
    events = m[m["news_count_1d"].astype(float) > 0].copy()
    if len(events) < MIN_EVENTS:
        return None

    events["ticker"] = ticker
    events["grp"] = events["sentiment_1d"].astype(float).apply(classify)
    keep = ["ticker", "date", "sentiment_1d", "news_count_1d", "grp"] + [
        f"ret_{k}d" for k in HORIZONS
    ]
    return events[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistical tests (pooled, with per-ticker demeaning to remove ticker confounding)
# ---------------------------------------------------------------------------
def _mw(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return (float(u), float(p))


def mann_whitney_pos_neg(events: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Mann-Whitney pos vs neg, both raw and per-ticker-demeaned (removes ticker-level
    return-level confounding from the pooled test)."""
    rows = []
    for k in horizons:
        col = f"ret_{k}d"
        sub = events[["ticker", "grp", col]].copy()
        # per-ticker demean
        sub["demeaned"] = sub.groupby("ticker")[col].transform(lambda x: x - x.mean())

        pos_raw = sub.loc[sub["grp"] == "pos", col].dropna().to_numpy()
        neg_raw = sub.loc[sub["grp"] == "neg", col].dropna().to_numpy()
        pos_dm = sub.loc[sub["grp"] == "pos", "demeaned"].dropna().to_numpy()
        neg_dm = sub.loc[sub["grp"] == "neg", "demeaned"].dropna().to_numpy()

        _, p_raw = _mw(pos_raw, neg_raw)
        _, p_dm = _mw(pos_dm, neg_dm)
        rows.append({
            "horizon": k, "n_pos": len(pos_raw), "n_neg": len(neg_raw),
            "mean_pos_bp": float(np.mean(pos_raw) * 1e4) if len(pos_raw) else np.nan,
            "mean_neg_bp": float(np.mean(neg_raw) * 1e4) if len(neg_raw) else np.nan,
            "spread_bp": float((np.mean(pos_raw) - np.mean(neg_raw)) * 1e4)
            if len(pos_raw) and len(neg_raw) else np.nan,
            "p_raw": p_raw, "p_demeaned": p_dm,
        })
    return pd.DataFrame(rows)


def group_summary(events: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows = []
    for grp in ["pos", "neu", "neg"]:
        sub = events[events["grp"] == grp]
        for k in horizons:
            r = sub[f"ret_{k}d"].dropna().to_numpy()
            rows.append({
                "grp": grp, "horizon": k, "n": len(r),
                "mean_bp": float(np.mean(r) * 1e4) if len(r) else np.nan,
                "median_bp": float(np.median(r) * 1e4) if len(r) else np.nan,
                "sentiment_median": float(sub["sentiment_1d"].astype(float).median())
                if len(sub) else np.nan,
            })
    return pd.DataFrame(rows)


def per_ticker_corr(events: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Spearman corr(sentiment_1d, ret_kd) on event days, per ticker.
    NOTE: computed over ALL event days (pos+neu+neg), so it is dominated by the large
    neutral mass; corr_posneg uses only pos+neg days for a cleaner monotonic signal."""
    rows = []
    for ticker, sub in events.groupby("ticker"):
        s_all = sub["sentiment_1d"].astype(float)
        rec = {"ticker": ticker, "n_events": len(sub),
               "n_pos": int((sub["grp"] == "pos").sum()),
               "n_neg": int((sub["grp"] == "neg").sum())}
        for k in horizons:
            r = sub[f"ret_{k}d"].astype(float)
            rec[f"corr_ret{k}d"] = _spearman_safe(s_all, r)
        # pos+neg only at primary horizon
        pn = sub[sub["grp"].isin(["pos", "neg"])]
        rec["corr_posneg_ret5d"] = _spearman_safe(pn["sentiment_1d"].astype(float),
                                                  pn["ret_5d"].astype(float))
        rows.append(rec)
    return pd.DataFrame(rows)


def _spearman_safe(s: pd.Series, r: pd.Series) -> float:
    valid = ~(s.isna() | r.isna())
    if valid.sum() < MIN_EVENTS or s[valid].nunique() <= 1 or r[valid].nunique() <= 1:
        return np.nan
    rho, _ = stats.spearmanr(s[valid], r[valid])
    return float(rho) if not np.isnan(rho) else np.nan


def neg_composition(events: pd.DataFrame) -> dict:
    """Report which tickers contribute negative-sentiment events (confound audit)."""
    neg = events[events["grp"] == "neg"]
    comp = neg.groupby("ticker").size().sort_values(ascending=False)
    return {
        "n_neg_events_total": int(len(neg)),
        "n_tickers_with_neg": int(comp.shape[0]),
        "n_tickers_with_ge2_neg": int((comp >= 2).sum()),
        "top_contributors": {t: int(c) for t, c in comp.head(8).items()},
    }


def volatility_comparison(horizons: list[int]) -> dict:
    """Per-ticker (ret_rho, vol_rho) PAIRS over the intersection of tickers that yield
    both correlations, so the ret-vs-vol mean comparison is apples-to-apples."""
    tickers = load_tickers()
    pairs = []  # (ticker, k, |ret_rho|, |vol_rho|)
    for t in tickers:
        sp = SENT_DIR / f"{t}_sentiment.csv"
        pp = PRICE_DIR / f"{t}_ohlcv.csv"
        vp = VOL_DIR / f"{t}_processed.csv"
        if not (sp.exists() and pp.exists() and vp.exists()):
            continue
        sent = pd.read_csv(sp, parse_dates=["date"])
        price = pd.read_csv(pp, parse_dates=["date"])
        vol = pd.read_csv(vp, parse_dates=["date"])
        ev = sent[sent["news_count_1d"].astype(float) > 0]
        fr = compute_forward_returns(price, horizons)

        vol_df = vol.sort_values("date").reset_index(drop=True)
        pv = vol_df["parkinson_volatility"].astype(float).to_numpy()
        pv = np.where(pv <= 0, np.nan, pv)

        for k in horizons:
            m1 = ev.merge(fr[["date", f"ret_{k}d"]], on="date", how="inner")
            ret_rho = _spearman_safe(m1["sentiment_1d"].astype(float),
                                     m1[f"ret_{k}d"].astype(float))
            n = len(pv)
            fvc = np.full(n, np.nan)
            if k < n:
                with np.errstate(divide="ignore", invalid="ignore"):
                    fvc[: n - k] = pv[k:] / pv[: n - k] - 1.0
            volchg = pd.DataFrame({"date": vol_df["date"].to_numpy(), f"vol_{k}d": fvc})
            m2 = ev.merge(volchg, on="date", how="inner")
            vol_rho = _spearman_safe(m2["sentiment_1d"].astype(float),
                                     m2[f"vol_{k}d"].astype(float))
            if not (np.isnan(ret_rho) or np.isnan(vol_rho)):
                pairs.append((t, k, abs(ret_rho), abs(vol_rho)))

    out = {"ret_vs_vol_by_horizon": {}, "n_tickers_paired": {}}
    for k in horizons:
        ks = [(r, v) for (_, kk, r, v) in pairs if kk == k]
        if ks:
            rs, vs = zip(*ks)
            out["ret_vs_vol_by_horizon"][int(k)] = {
                "mean_abs_corr_return": float(np.mean(rs)),
                "mean_abs_corr_volatility": float(np.mean(vs)),
            }
            out["n_tickers_paired"][int(k)] = len(ks)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_all(events: pd.DataFrame, mw: pd.DataFrame, gs: pd.DataFrame,
             tc: pd.DataFrame, vol_cmp: dict) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"pos": "#2ca02c", "neu": "#7f7f7f", "neg": "#d62728"}

    # 1. sentiment distribution on event days
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(events["sentiment_1d"].astype(float).dropna(), bins=30, color="#4c72b0", edgecolor="white")
    ax.axvline(POS_THR, ls="--", c="green", label=f"pos thr {POS_THR}")
    ax.axvline(NEG_THR, ls="--", c="red", label=f"neg thr {NEG_THR}")
    ax.set_title("Sentiment distribution on news-event days (pooled)")
    ax.set_xlabel("sentiment_1d"); ax.set_ylabel("event count"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_sentiment_distribution.png", dpi=120); plt.close(fig)

    # 2. mean forward return by group x horizon
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    x = np.arange(len(HORIZONS))
    for i, grp in enumerate(["pos", "neu", "neg"]):
        vals = []
        for k in HORIZONS:
            row = gs[(gs["grp"] == grp) & (gs["horizon"] == k)]
            vals.append(float(row["mean_bp"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + (i - 1) * width, vals, width, label=f"{grp} (n={int((events['grp']==grp).sum())})",
               color=colors[grp])
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"T+{k}" for k in HORIZONS])
    ax.set_ylabel("Mean forward return (bp)")
    ax.set_title("Mean forward return by sentiment group (1 bp = 0.01%)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_mean_return_by_group.png", dpi=120); plt.close(fig)

    # 3. per-ticker Spearman corr(sentiment, ret_5d) on event days
    if "corr_ret5d" in tc.columns:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        s = tc.sort_values("corr_ret5d", na_position="last")
        ax.barh(s["ticker"], s["corr_ret5d"].fillna(0), color="#4c72b0")
        ax.axvline(0, c="k", lw=0.8)
        ax.set_title("Per-ticker Spearman corr(sentiment, ret_5d) on event days")
        ax.set_xlabel("Spearman rho (all event days)")
        fig.tight_layout(); fig.savefig(OUT_DIR / "fig_per_ticker_corr_ret5d.png", dpi=120); plt.close(fig)

    # 4. lag correlation box across horizons
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [tc[f"corr_ret{k}d"].dropna().to_numpy() for k in HORIZONS]
    ax.boxplot(data, tick_labels=[f"T+{k}" for k in HORIZONS])
    ax.axhline(0, c="k", lw=0.8)
    ax.set_ylabel("Spearman rho (per ticker)")
    ax.set_title("Distribution of per-ticker sentiment/return correlation by horizon")
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_lag_correlation.png", dpi=120); plt.close(fig)

    # 5. ret vs vol correlation comparison (paired, same tickers)
    rv = vol_cmp.get("ret_vs_vol_by_horizon", {})
    if rv:
        ks = sorted(int(k) for k in rv)
        ret_v = [rv[k]["mean_abs_corr_return"] for k in ks]
        vol_v = [rv[k]["mean_abs_corr_volatility"] for k in ks]
        x = np.arange(len(ks))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - 0.15, ret_v, 0.3, label="-> return", color="#4c72b0")
        ax.bar(x + 0.15, vol_v, 0.3, label="-> volatility", color="#dd8452")
        ax.set_xticks(x); ax.set_xticklabels([f"T+{k}" for k in ks])
        ax.set_ylabel("Mean |Spearman rho| (paired tickers)")
        ax.set_title("Sentiment predictive value: return vs volatility")
        ax.legend()
        fig.tight_layout(); fig.savefig(OUT_DIR / "fig_vol_vs_ret_corr.png", dpi=120); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    tickers = load_tickers()
    print(f"[info] {len(tickers)} price tickers found.")

    frames, skipped = [], []
    for t in tickers:
        ev = build_ticker_events(t)
        (frames.append(ev) if ev is not None else skipped.append(t))
    if not frames:
        print("[error] No ticker with enough events. Aborting.")
        return
    events_raw = pd.concat(frames, ignore_index=True)

    # sanity: returns must be decimal (guard against pipeline unit changes)
    all_ret = pd.concat([events_raw[f"ret_{k}d"] for k in HORIZONS]).dropna()
    if all_ret.abs().max() > 2.0:
        raise ValueError(f"Forward return max abs = {all_ret.abs().max():.3f} (>2.0). "
                         "Expected decimal returns; data unit may have changed.")

    events = winsorize_events(events_raw, HORIZONS)        # event-level winsorize (per horizon)
    events.to_csv(OUT_DIR / "events_all.csv", index=False)
    print(f"[info] Pooled events: {len(events)} across {events['ticker'].nunique()} tickers. "
          f"Skipped (too few events): {skipped}")

    grp_counts = events["grp"].value_counts().to_dict()
    print(f"[info] Group counts: {grp_counts}")

    gs = group_summary(events, HORIZONS)
    mw = mann_whitney_pos_neg(events, HORIZONS)
    tc = per_ticker_corr(events, HORIZONS)
    tc.to_csv(OUT_DIR / "per_ticker_stats.csv", index=False)
    vol_cmp = volatility_comparison([1, 5])
    neg_comp = neg_composition(events)

    # Go/No-Go: Bonferroni across horizons; primary = 5-day; require spread too
    alpha_corr = ALPHA / len(HORIZONS)
    primary = mw[mw["horizon"] == PRIMARY_HORIZON].iloc[0]
    p_primary = primary["p_demeaned"] if not np.isnan(primary["p_demeaned"]) else primary["p_raw"]
    sig_any = mw[(mw["p_raw"].notna()) & (mw["p_raw"] < alpha_corr)]
    spread_ok = sig_any[sig_any["spread_bp"].abs() >= MIN_SPREAD_BP]
    go = (not spread_ok.empty) and (not np.isnan(p_primary)) and (p_primary < alpha_corr)

    summary = {
        "units": "returns are decimal; *_bp fields are basis points (1 bp = 0.01%).",
        "n_tickers_price": len(tickers),
        "n_tickers_with_events": int(events["ticker"].nunique()),
        "skipped_tickers": skipped,
        "n_pooled_events": len(events),
        "group_counts": {k: int(v) for k, v in grp_counts.items()},
        "thresholds": {"pos": POS_THR, "neg": NEG_THR},
        "note_class_imbalance": "Sentiment is severely positive-skewed; the 'neg' group is "
                                "an extreme-tail, rare sample (see neg_composition).",
        "neg_composition": neg_comp,
        "mann_whitney_pos_vs_neg": mw.to_dict(orient="records"),
        "ret_vs_vol_correlation_paired": vol_cmp,
        "per_ticker_corr_note": "corr_ret*Kd computed over ALL event days (pos+neu+neg), "
                                "dominated by the large neutral mass; corr_posneg_ret5d uses "
                                "pos+neg days only.",
        "go_no_go": {
            "verdict": "GO" if go else "NO-GO",
            "alpha_bonferroni_corrected": alpha_corr,
            "primary_horizon": PRIMARY_HORIZON,
            "p_value_primary_demeaned_or_raw": None if np.isnan(p_primary) else float(p_primary),
            "significant_horizons_raw_p": sig_any["horizon"].tolist(),
            "horizons_with_spread_ge_30bp": spread_ok["horizon"].tolist(),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    plot_all(events, mw, gs, tc, vol_cmp)

    # Console report
    print(f"\n=== Group mean forward return (bp) ===")
    print(gs.pivot(index="horizon", columns="grp", values="mean_bp").round(1).to_string())
    print(f"\n=== Mann-Whitney pos vs neg (raw & per-ticker-demeaned) ===")
    print(mw.round({"mean_pos_bp": 1, "mean_neg_bp": 1, "spread_bp": 1,
                    "p_raw": 4, "p_demeaned": 4}).to_string(index=False))
    print(f"\n=== Negative-event composition (confound audit) ===")
    print(json.dumps(neg_comp, indent=2, default=str))
    print(f"\n=== Sentiment -> return vs -> volatility (paired |rho|) ===")
    print(json.dumps(vol_cmp, indent=2, default=str))
    print(f"\n=== VERDICT: {summary['go_no_go']['verdict']} "
          f"(Bonferroni alpha={alpha_corr:.4f}, primary T+{PRIMARY_HORIZON}) ===")
    print(f"significant horizons (raw p<{alpha_corr:.4f}): {summary['go_no_go']['significant_horizons_raw_p']}")
    print(f"horizons with |spread|>={MIN_SPREAD_BP}bp: {summary['go_no_go']['horizons_with_spread_ge_30bp']}")
    print(f"\n[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
