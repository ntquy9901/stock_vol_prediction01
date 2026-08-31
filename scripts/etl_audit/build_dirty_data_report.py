"""Driver: per-market dirty-data audit HTML + consolidated ETL-cleaning spec.

CPU/pandas only (GPU committed to a training job). Reads raw OHLCV + the processed Parkinson-VARIANCE target
READ-ONLY, runs the per-(ticker,date) detectors (``dirty_data_detectors``), and renders one self-contained
HTML per market (base64 charts, no CDN) + one consolidated spec markdown.

Usage:
    python scripts/etl_audit/build_dirty_data_report.py                 # all 5 markets, full
    python scripts/etl_audit/build_dirty_data_report.py --limit 12      # smoke on <=12 tickers/market
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "etl_audit"))
sys.path.insert(0, str(REPO / "scripts" / "eda"))
sys.path.insert(0, str(REPO / "scripts" / "garch_masked"))

import dirty_data_detectors as DD  # noqa: E402

PANELS = ["hnx", "hose", "vn30", "vn100", "sp500"]
PROCESSED = {
    "vn30": REPO / "submission" / "soict_lstm_gat" / "data" / "vn30",
    "vn100": REPO / "submission" / "soict_lstm_gat" / "data" / "vn100",
    "hose": REPO / "data" / "processed" / "hose",
    "hnx": REPO / "data" / "processed" / "hnx",
    "sp500": REPO / "data" / "processed" / "sp500",
}
RAW = {
    "vn30": REPO / "data" / "raw" / "prices",
    "vn100": REPO / "data" / "raw" / "prices" / "vn100_vnstock",
    "hose": REPO / "data" / "raw" / "prices" / "hose_vnstock",
    "hnx": REPO / "data" / "raw" / "prices" / "hnx_vnstock",
    "sp500": REPO / "data" / "raw" / "prices" / "sp500",
}
OUT_DIR = REPO / "docs" / "reports"
_LN2 = float(np.log(2.0))
CLIP_CAP = 0.1                       # the processed upper-clip value to test for

CLASSES = ["high_lt_low", "open_close_outside", "nonpositive", "zero_range", "split_jumps",
           "stale_runs", "naninf", "zero_volume", "leading_backfill"]

# class -> (recommended ETL rule, estimators affected, TARGET-affecting? for the Parkinson target)
ETL_RULE = {
    "high_lt_low": ("swap H<->L if a transposition, else drop the bar", "Parkinson, GK, RS, YZ", True),
    "open_close_outside": ("WIDEN range H=max(H,O,C), L=min(L,O,C) (recommended); or clip O/C into [L,H]",
                           "GK, RS, YZ (Parkinson immune: H/L only)", False),
    "nonpositive": ("reconstruct H/L from positive OHLC (max/min), clamp O/C; else drop",
                    "Parkinson, GK, RS, YZ", True),
    "zero_range": ("KEEP + FLAG (liquidity screen / vol floor); do NOT delete", "Parkinson, GK, RS, YZ", True),
    # Parkinson ln(H/L)^2 (and GK/RS within-day ratios) are SCALE-INVARIANT: an unadjusted split rescales H
    # and L by the same factor, so ln(H/L) -- and the Parkinson target -- is UNCHANGED. Only Yang-Zhang's
    # overnight/close-to-close term ln(O_t/C_{t-1}) crosses the split boundary and is affected, on the single
    # boundary day. So split jumps are COSMETIC for the delivered Parkinson target (code review 2026-08-30).
    "split_jumps": ("back-adjust prior prices by the split factor; else flag+winsorize "
                    "(does NOT move the Parkinson target)", "YZ overnight-boundary only "
                    "(Parkinson/GK/RS scale-invariant)", False),
    "leading_backfill": ("cut to the true first-trade date", "Parkinson, GK, RS, YZ (leading rows only)", False),
    "stale_runs": ("flag; optionally drop the run", "GK, RS, YZ (close-based); cosmetic for Parkinson", False),
    "naninf": ("drop / impute; must never reach the model", "Parkinson, GK, RS, YZ", True),
    "zero_volume": ("flag illiquidity (keep)", "none directly (liquidity flag)", False),
}


# --------------------------------------------------------------------------------------------------
# Pure aggregation / measurement (unit-tested)
# --------------------------------------------------------------------------------------------------
def raw_parkinson(df: pd.DataFrame) -> np.ndarray:
    """Parkinson VARIANCE ln(H/L)^2/(4 ln2) from raw OHLCV; NaN on invalid geometry (H<L or nonpositive)."""
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    ok = np.isfinite(h) & np.isfinite(lo) & (h > 0) & (lo > 0) & (h >= lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        pk = np.where(ok, np.log(h / lo) ** 2 / (4 * _LN2), np.nan)
    return pk


def clip_evidence(raw_df: pd.DataFrame, proc_df: pd.DataFrame | None, cap: float = CLIP_CAP) -> dict:
    """Measure (no assumption) whether the processed Parkinson target is upper-clipped at ``cap``: align the
    processed value to the raw Parkinson by date and count rows where processed is at the cap while raw
    Parkinson exceeded it. Returns processed max + clip counts. Empty dict when no processed frame."""
    if proc_df is None or "parkinson_variance" not in proc_df.columns \
            or "date" not in proc_df.columns or "date" not in raw_df.columns:
        return {"has_processed": False}
    pv = pd.to_numeric(proc_df["parkinson_variance"], errors="coerce")
    proc = pd.DataFrame({"date": pd.to_datetime(proc_df["date"], errors="coerce"), "proc": pv}).dropna()
    raw = pd.DataFrame({"date": pd.to_datetime(raw_df["date"], errors="coerce"),
                        "raw_pk": raw_parkinson(raw_df)}).dropna()
    merged = proc.merge(raw, on="date", how="inner")
    at_cap = np.isclose(merged["proc"].to_numpy(float), cap)
    raw_over = merged["raw_pk"].to_numpy(float) > cap
    proc_vals = proc["proc"].to_numpy(float)
    return {
        "has_processed": True,
        "n_processed": int(len(proc)),
        "n_aligned": int(len(merged)),
        "proc_max": float(np.max(proc_vals)) if proc_vals.size else float("nan"),
        "n_at_cap": int(np.sum(at_cap)),
        "n_clipped_from_raw": int(np.sum(at_cap & raw_over)),   # processed==cap AND raw exceeded cap
        "n_zero_processed": int(np.sum(proc_vals == 0.0)),
    }


def aggregate_frames(items: list) -> dict:
    """Aggregate detectors over a market. items = list of (ticker, raw_df, processed_df_or_None).
    Returns totals per class, per-ticker rows, worst-offender ticker lists per class, and a pooled
    raw-vs-processed clip summary. Pure (no IO)."""
    totals = {k: 0 for k in CLASSES}
    per_ticker = []
    clip_pool = {"has_processed": False, "n_processed": 0, "n_aligned": 0, "proc_max": 0.0,
                 "n_at_cap": 0, "n_clipped_from_raw": 0, "n_zero_processed": 0}
    total_rows = 0
    for ticker, raw_df, proc_df in items:
        row = DD.per_ticker_summary(ticker, raw_df)
        per_ticker.append(row)
        total_rows += row["rows"]
        for k in CLASSES:
            totals[k] += row[k]
        ce = clip_evidence(raw_df, proc_df)
        if ce.get("has_processed"):
            clip_pool["has_processed"] = True
            for k in ("n_processed", "n_aligned", "n_at_cap", "n_clipped_from_raw", "n_zero_processed"):
                clip_pool[k] += ce[k]
            clip_pool["proc_max"] = max(clip_pool["proc_max"], ce["proc_max"])
    worst = {k: sorted([r for r in per_ticker if r[k] > 0], key=lambda r, kk=k: r[kk], reverse=True)
             for k in CLASSES}
    return {"totals": totals, "per_ticker": per_ticker, "total_rows": total_rows,
            "worst": worst, "clip": clip_pool, "n_tickers": len(per_ticker)}


def build_executive_table(summary: dict) -> list:
    """Executive dirty-data summary rows: class, count, % of ticker-days, recommended ETL, estimators,
    target-affecting flag. ``count`` for leading_backfill is total leading rows across tickers."""
    rows = []
    tot_rows = summary["total_rows"] or 1
    for k in CLASSES:
        cnt = summary["totals"][k]
        rule, est, target = ETL_RULE[k]
        rows.append({
            "issue": k, "count": cnt, "pct_of_rows": 100.0 * cnt / tot_rows,
            "recommended_etl": rule, "estimators": est,
            "target_affecting": "REAL (Parkinson)" if target else "cosmetic for Parkinson",
        })
    return rows


def build_spec_md(all_summaries: dict) -> str:
    """Consolidated ETL-cleaning spec markdown: per class = detection + cleaning + estimators + priority,
    plus a cross-market prevalence table. ``all_summaries`` maps panel -> summary dict."""
    lines = ["# Consolidated ETL-cleaning specification — dirty-data classes across VN + US markets", "",
             f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}. Parkinson target = sigma^2 (VARIANCE), "
             "H/L only. CPU/pandas only. Every count traces to the audited data.", "",
             "## Priority: TARGET-affecting vs cosmetic", "",
             "REAL (moves the delivered Parkinson vol target / results): high<low, nonpositive OHLC, "
             "zero-range, NaN/inf. Cosmetic for the Parkinson target (only touches the open/close-using "
             "estimators GK/RS/YZ, or leading rows / liquidity flags): open/close-outside, split jumps, "
             "stale closes, leading backfill, zero-volume.", "",
             "Note: Parkinson ln(H/L)^2 and the GK/RS within-day ratios are SCALE-INVARIANT, so an unadjusted "
             "split (a uniform price rescale) does NOT change their value on any day -- only Yang-Zhang's "
             "overnight term is affected, and only on the split boundary day. Back-adjusting a split therefore "
             "does not move the delivered Parkinson target; it matters for close-to-close / overnight "
             "estimators and for any level-based (non-ratio) feature.", "",
             "Prioritised action order (fix first -> last): (1) NaN/inf drop, (2) nonpositive reconstruct, "
             "(3) high<low swap/drop, (4) zero-range flag + liquidity screen / vol floor [the dominant "
             "target driver], (5) leading-backfill cut, then cosmetic: (6) open/close-outside widen "
             "[GK/RS/YZ only], (7) split back-adjust [overnight estimators only], (8) stale/zero-volume flags.",
             ""]
    lines.append("## Per-class detection + cleaning rules")
    lines.append("")
    lines.append("| # | issue | detection rule | cleaning rule | estimators affected | priority |")
    lines.append("|---|---|---|---|---|---|")
    detect_txt = {
        "high_lt_low": "high < low (finite)",
        "open_close_outside": "high < max(O,C)*(1-1e-5) or low > min(O,C)*(1+1e-5)",
        "nonpositive": "any O/H/L/C <= 0",
        "zero_range": "finite positive high == low",
        "split_jumps": "|1-day simple return| > 50%",
        "stale_runs": ">= 5 identical consecutive closes",
        "naninf": "non-finite O/H/L/C/volume",
        "zero_volume": "finite volume == 0",
        "leading_backfill": "leading run: constant close + (zero volume or zero range)",
    }
    # stale_runs is reported in stale DAYS (sum of run lengths), comparable to the other day-based counts.
    count_unit = {k: "ticker-days" for k in CLASSES}
    count_unit["stale_runs"] = "stale days"
    count_unit["leading_backfill"] = "leading rows"
    for i, k in enumerate(CLASSES, 1):
        rule, est, target = ETL_RULE[k]
        pr = "REAL" if target else "cosmetic (Parkinson)"
        lines.append(f"| {i} | {k} | {detect_txt[k]} | {rule} | {est} | {pr} |")
    lines.append("")
    lines.append("## Cross-market dirty-data prevalence (raw ticker-day counts)")
    lines.append("")
    hdr = "| market | tickers | ticker-days | " + " | ".join(CLASSES) + " |"
    lines.append(hdr)
    lines.append("|" + "---|" * (3 + len(CLASSES)))
    for panel, s in all_summaries.items():
        cells = " | ".join(str(s["totals"][k]) for k in CLASSES)
        lines.append(f"| {panel} | {s['n_tickers']} | {s['total_rows']:,} | {cells} |")
    lines.append("")
    lines.append(f"Count units: most classes = {count_unit['high_lt_low']}; "
                 f"stale_runs = {count_unit['stale_runs']}; leading_backfill = {count_unit['leading_backfill']}.")
    lines.append("")
    lines.append("## Raw-vs-processed (does the current ETL already clean it?)")
    lines.append("")
    lines.append("| market | processed rows | processed max | at 0.1 cap | clipped-from-raw (>0.1) | "
                 "zero processed |")
    lines.append("|---|---|---|---|---|---|")
    for panel, s in all_summaries.items():
        c = s["clip"]
        if not c.get("has_processed"):
            lines.append(f"| {panel} | (no processed) | - | - | - | - |")
            continue
        lines.append(f"| {panel} | {c['n_processed']:,} | {c['proc_max']:.4g} | {c['n_at_cap']:,} | "
                     f"{c['n_clipped_from_raw']:,} | {c['n_zero_processed']:,} |")
    lines.append("")
    lines.append("Reading: a nonzero `clipped-from-raw` count is direct evidence the ETL upper-clips the "
                 "Parkinson target at 0.1 (raw Parkinson exceeded 0.1 but the processed value is 0.1). "
                 "`zero processed` = zero-range/limit days that pass through to the target and get floored "
                 "in QLIKE scoring (the target-affecting driver).")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------------
# IO loading (thin wrappers)
# --------------------------------------------------------------------------------------------------
def _load_raw(path: str) -> pd.DataFrame:  # pragma: no cover - file IO
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", kind="stable").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


def _load_processed(panel: str, ticker: str):  # pragma: no cover - file IO
    pf = PROCESSED[panel] / f"{ticker}_processed.csv"
    if not pf.exists():
        return None
    try:
        return pd.read_csv(pf)
    except Exception:
        return None


def analyze_market(panel: str, limit: int | None = None) -> dict:  # pragma: no cover - IO wrapper
    """Load one market's raw + processed frames and aggregate. IO wrapper around the pure aggregate."""
    files = sorted(glob.glob(str(RAW[panel] / "*_ohlcv.csv")))
    if limit is not None:
        files = files[:limit]
    items = []
    for f in files:
        ticker = Path(f).stem.replace("_ohlcv", "")
        raw = _load_raw(f)
        if not {"open", "high", "low", "close"} <= set(raw.columns) or len(raw) < 2:
            continue
        items.append((ticker, raw, _load_processed(panel, ticker)))
    s = aggregate_frames(items)
    s["panel"] = panel
    s["_raw_frames"] = {t: r for t, r, _p in items}      # kept for drill-down charts; released by caller
    return s


# --------------------------------------------------------------------------------------------------
# Charts + HTML (presentation; base64 PNG, no CDN)
# --------------------------------------------------------------------------------------------------
def _fig_b64(fig) -> str:  # pragma: no cover - matplotlib IO
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def drilldown_chart(ticker: str, df: pd.DataFrame, issue: str, dates: list) -> str:  # pragma: no cover
    """Close line with the dirty dates for ``issue`` highlighted, so a reviewer sees exactly what is wrong."""
    d = pd.to_datetime(df["date"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(d, c, color="#3b6ea5", lw=0.8, label="close")
    ax.fill_between(d, lo, h, color="#3b6ea5", alpha=0.12, label="H-L range")
    flagged = pd.to_datetime(pd.Series(dates), errors="coerce")
    sel = d.isin(flagged).to_numpy()
    if sel.any():
        ax.scatter(d[sel], c[sel], color="#b00", s=18, zorder=5, label=f"{issue} ({int(sel.sum())})")
    ax.set_title(f"{ticker} — {issue}", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7)
    return f"<img alt='{ticker} {issue}' style='max-width:100%' src='data:image/png;base64,{_fig_b64(fig)}'>"


_SORT_JS = """<script>
function sortTable(tbl, col, numeric){var t=document.getElementById(tbl),b=t.tBodies[0];
var rows=Array.prototype.slice.call(b.rows);var dir=t.getAttribute('data-dir-'+col)==='asc'?-1:1;
rows.sort(function(x,y){var a=x.cells[col].innerText,c=y.cells[col].innerText;
if(numeric){a=parseFloat(a)||0;c=parseFloat(c)||0;return (a-c)*dir;}return a.localeCompare(c)*dir;});
rows.forEach(function(r){b.appendChild(r);});t.setAttribute('data-dir-'+col,dir===1?'asc':'desc');}
</script>"""

_CSS = ("body{font-family:system-ui,Arial,sans-serif;margin:24px;max-width:1180px;color:#222}"
        "table{border-collapse:collapse;font-size:12.5px;margin:8px 0}td,th{border:1px solid #ccc;"
        "padding:3px 7px;text-align:center}th{background:#f2f2f2;cursor:pointer}h2{border-bottom:2px solid "
        "#ddd;margin-top:28px}.note{color:#555;font-size:13px}.warn{color:#b00;font-weight:bold}"
        ".grid{display:flex;flex-wrap:wrap;gap:10px}.card{flex:1 1 420px}code{background:#f4f4f4;padding:1px 4px}")


def render_market_html(summary: dict, top_n: int = 8) -> str:  # pragma: no cover - presentation
    p = summary["panel"]
    frames = summary.get("_raw_frames", {})
    parts = [f"<html><head><meta charset='utf-8'><title>{p.upper()} dirty-data ETL</title>"
             f"<style>{_CSS}</style>{_SORT_JS}</head><body>"]
    parts.append(f"<h1>{p.upper()} — dirty-data audit &amp; ETL-cleaning</h1>")
    parts.append(f"<p class='note'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | raw "
                 f"<code>{RAW[p].name}</code> | processed <code>{PROCESSED[p].name}</code>. Parkinson = "
                 "sigma^2 (VARIANCE), H/L only. CPU/pandas only. Counts trace to the data.</p>")

    # (i) executive summary table
    parts.append("<h2>1. Executive dirty-data summary</h2>")
    parts.append(f"<p class='note'>{summary['n_tickers']} tickers, {summary['total_rows']:,} ticker-days.</p>")
    ex = build_executive_table(summary)
    parts.append("<table><tr><th>issue</th><th>count</th><th>% rows</th><th>recommended ETL</th>"
                 "<th>estimators affected</th><th>priority</th></tr>")
    for r in ex:
        cls = "warn" if r["target_affecting"].startswith("REAL") else ""
        parts.append(f"<tr><td class='{cls}'>{r['issue']}</td><td>{r['count']:,}</td>"
                     f"<td>{r['pct_of_rows']:.3f}%</td><td style='text-align:left'>{r['recommended_etl']}</td>"
                     f"<td style='text-align:left'>{r['estimators']}</td><td>{r['target_affecting']}</td></tr>")
    parts.append("</table>")

    # raw-vs-processed
    c = summary["clip"]
    if c.get("has_processed"):
        parts.append("<h2>2. Raw-vs-processed (current ETL)</h2>")
        parts.append(f"<p class='note'>Processed rows {c['n_processed']:,}; processed max "
                     f"<b>{c['proc_max']:.4g}</b>; rows at the 0.1 cap {c['n_at_cap']:,}, of which "
                     f"<b>{c['n_clipped_from_raw']:,}</b> had raw Parkinson &gt; 0.1 (evidence of an upper "
                     f"clip); zero-valued processed rows (floored target) {c['n_zero_processed']:,}.</p>")

    # (ii) sortable per-ticker table
    parts.append("<h2>3. Per-ticker dirty-data table (click a header to sort)</h2>")
    cols = ["ticker", "rows", "first", "last"] + CLASSES + ["zero_range_frac", "zero_volume_frac",
                                                            "oc_median_rel_violation"]
    head = "".join(f"<th onclick=\"sortTable('pt',{i},{str(cn not in ('ticker','first','last')).lower()})\">"
                   f"{cn}</th>" for i, cn in enumerate(cols))
    parts.append(f"<table id='pt' data-dir-0='desc'><tr>{head}</tr>")
    for r in sorted(summary["per_ticker"], key=lambda r: sum(r[k] for k in CLASSES), reverse=True):
        cells = []
        for cn in cols:
            v = r[cn]
            cells.append(f"<td>{v:.4f}</td>" if isinstance(v, float) else f"<td>{v}</td>")
        parts.append("<tr>" + "".join(cells) + "</tr>")
    parts.append("</table>")

    # (iii) per-stock drill-down for the worst offenders per class
    parts.append("<h2>4. Per-stock drill-down (worst offenders; dirty dates highlighted)</h2>")
    for k in CLASSES:
        worst = summary["worst"].get(k, [])[:top_n]
        if not worst or k in ("leading_backfill",):
            continue
        parts.append(f"<h3>{k} — top {len(worst)}</h3><div class='grid'>")
        for r in worst:
            t = r["ticker"]
            df = frames.get(t)
            if df is None:
                continue
            res = DD.detect_all(df)
            ex_dates = _example_dates(res, k)
            parts.append("<div class='card'>" + drilldown_chart(t, df, k, ex_dates) + "</div>")
        parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)


def _example_dates(detect_res: dict, issue: str) -> list:
    """Extract the date strings for an issue from a detect_all() result (handles tuple-valued classes)."""
    ex = detect_res["examples"].get(issue, [])
    if issue in ("open_close_outside", "split_jumps"):
        return [d for d, _m in ex]
    if issue == "stale_runs":
        return [s for s, _e, _l in ex]
    if issue == "leading_backfill":
        return []
    return list(ex)


# --------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------
def run(panels=tuple(PANELS), limit=None, out_dir=OUT_DIR):  # pragma: no cover - IO driver
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    compact = {}
    for panel in panels:
        s = analyze_market(panel, limit=limit)
        html = render_market_html(s)
        path = out_dir / f"2026-08-31_{panel}_dirty_data_etl.html"
        path.write_text(html, encoding="utf-8")
        written.append(str(path))
        compact[panel] = {k: s[k] for k in ("totals", "n_tickers", "total_rows", "clip")}
        del s
    spec = build_spec_md(compact)
    spec_path = out_dir / "2026-08-31_etl_cleaning_spec.md"
    spec_path.write_text(spec, encoding="utf-8")
    written.append(str(spec_path))
    return {"written": written, "compact": compact}


def main():  # pragma: no cover - CLI entry
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels", nargs="+", default=PANELS)
    ap.add_argument("--limit", type=int, default=None, help="cap tickers/market (smoke)")
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    res = run(panels=a.panels, limit=a.limit, out_dir=a.out)
    for w in res["written"]:
        print("wrote", w)


if __name__ == "__main__":  # pragma: no cover
    main()
