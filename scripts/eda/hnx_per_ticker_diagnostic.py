"""Per-ticker HNX diagnostic -> interactive HTML (one row per HNX stock).

Purpose: a DATA-quality + MODEL diagnostic to FIND per-stock problems on the HNX panel, for EVERY
HNX ticker -- both the ones kept by the liquidity+history screen AND the ones excluded (so the reason
for exclusion is visible). Surfaces issues plainly; it does not hide them.

Part A (DATA, no GPU): for every HNX ticker compute valid-row count, first/last date, calendar
coverage + weekday gaps, exact-zero Parkinson fraction (the Parkinson column is a VARIANCE sigma^2, so
an exact zero == high==low == a limit/illiquid day), the sigma^2 distribution (min/median/p95/max +
NaN/inf/nonpositive counts), per-ticker OHLC sanity (high<low, open/close outside [low,high],
nonpositive prices, suspicious pre-listing backfill), the screen decision + reason + threshold, and --
for tickers that enter the masked panel -- train/val/test valid-cell counts, per-split zero-target
fraction, and a test floor-activation proxy (fraction of test targets at/under the per-node relative
floor 1e-2*mean, the QLIKE driver).

Part B (MODEL, GPU-polite, best-effort): per-ticker test QLIKE/MSE/R2 for the three delivered HNX-h1
models -- no-graph LSTM, HAR-X, and VolGA (LSTM+wGAT vol->PK) -- from ONE evaluation run that reuses
the delivered runner/config, grouped by ticker. The SAME QLIKE positivity floor is used across all
three (identical basis). If no clean GPU slot appears, the HTML ships DATA-only with model columns
marked "pending -- GPU busy".

Reuses (READ-ONLY): floor_sensitivity.screen_files (screen), masked_rich.build_masked_rich (panel),
run_masked_rich.train_masked_rich (models), volatility_estimators.PRICE / estimators_from_ohlcv, and
the delivered processed HNX data. Does NOT edit any live-training-path file.

Usage:
    python scripts/eda/hnx_per_ticker_diagnostic.py [--no-model] [--seeds 42 123 ...]
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "submission" / "soict_lstm_gat",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "scripts" / "garch_masked",
           REPO / "scripts" / "eda"):
    sys.path.insert(0, str(_p))

# Screen threshold constants (single source of truth = floor_sensitivity.screen_files defaults).
SCREEN_MIN_ROWS = 250
SCREEN_MAX_ZERO_FRAC = 0.5
SCREEN_MAX_NAN_FRAC = 0.5

# Issue-flag thresholds (diagnostic; distinct from, and stricter than, the screen).
FLAG_ZERO_FRAC = 0.30        # red: exact-zero Parkinson fraction above this
FLAG_MIN_ROWS = 252          # red: fewer valid rows than this (< one trading year)
FLAG_FLOOR_ACT = 0.20        # amber: test floor-activation above this
FLAG_QLIKE_MULT = 2.0        # amber: per-ticker model QLIKE above this multiple of the panel median
NODE_FLOOR_REL = 1e-2        # per-node relative positivity floor (1e-2 * train mean) -- the QLIKE driver
OHLC_RTOL = 1e-5             # OHLC geometry tolerance (matches tests/test_raw_prices_quality.py + volatility_estimators): absorb float32 storage noise (~1e-7), keep only real violations

PROCESSED_DIR = REPO / "data" / "processed" / "hnx"
PRICE_DIR = REPO / "data" / "raw" / "prices" / "hnx_vnstock"


# --------------------------------------------------------------------------- Part A: pure DATA helpers
def weekday_gaps(dates) -> int:
    """Count business days (Mon-Fri) between the first and last trade date with NO row present.

    NOTE: VN market holidays are legitimately missing weekdays, so this OVER-counts true data gaps --
    it is an upper bound / coarse liquidity signal, not a defect count.
    """
    d = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dropna().sort_values().unique()
    if len(d) < 2:
        return 0
    present = {pd.Timestamp(x).normalize() for x in d}
    bdays = pd.bdate_range(pd.Timestamp(d[0]).normalize(), pd.Timestamp(d[-1]).normalize())
    return int(sum(1 for b in bdays if b not in present))


def parkinson_dist(values) -> dict:
    """Distribution + validity counts of a Parkinson-variance column (before dropping invalids)."""
    v = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(float)
    n_nan = int(np.sum(np.isnan(v)))
    n_inf = int(np.sum(np.isinf(v)))
    finite = v[np.isfinite(v)]
    n_nonpos = int(np.sum(finite <= 0.0))
    pos = finite[finite > 0.0]
    if pos.size:
        stats = {"min": float(pos.min()), "median": float(np.median(pos)),
                 "p95": float(np.percentile(pos, 95)), "max": float(pos.max())}
    else:
        stats = {"min": float("nan"), "median": float("nan"),
                 "p95": float("nan"), "max": float("nan")}
    return {"n_nan": n_nan, "n_inf": n_inf, "n_nonpos": n_nonpos, **stats}


def ohlc_sanity(raw: pd.DataFrame) -> dict:
    """Per-ticker OHLC violation counts from raw OHLCV (high<low, open/close outside [low,high],
    nonpositive prices) + a leading pre-listing backfill proxy (leading rows with zero volume)."""
    cols = {"open", "high", "low", "close"}
    if not cols <= set(raw.columns):
        return {"n_high_lt_low": 0, "n_oc_outside": 0, "n_nonpos_price": 0,
                "backfill_lead_zerovol": 0, "n_raw_rows": len(raw)}
    o = pd.to_numeric(raw["open"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(raw["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(raw["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(raw["close"], errors="coerce").to_numpy(float)
    finite = np.isfinite([o, h, lo, c]).all(0)
    n_high_lt_low = int(np.sum(finite & (h < lo)))
    # open/close outside [low, high] beyond the shared float32 tolerance (real violations only): high must
    # be >= max(open,close) and low <= min(open,close), so a violation is high < max(o,c) or low > min(o,c).
    hi_oc = np.maximum(o, c)
    lo_oc = np.minimum(o, c)
    n_oc_outside = int(np.sum(finite & ((h < hi_oc * (1 - OHLC_RTOL)) | (lo > lo_oc * (1 + OHLC_RTOL)))))
    n_nonpos_price = int(np.sum(finite & ((o <= 0) | (h <= 0) | (lo <= 0) | (c <= 0))))
    if "volume" in raw.columns:
        vol = pd.to_numeric(raw["volume"], errors="coerce").to_numpy(float)
        lead = 0
        for x in vol:
            if np.isfinite(x) and x == 0.0:
                lead += 1
            else:
                break
    else:
        lead = 0
    return {"n_high_lt_low": n_high_lt_low, "n_oc_outside": n_oc_outside,
            "n_nonpos_price": n_nonpos_price, "backfill_lead_zerovol": int(lead),
            "n_raw_rows": len(raw)}


def data_stats_for_ticker(ticker: str, proc: pd.DataFrame, raw: pd.DataFrame,
                          screen_reason: str) -> dict:
    """Assemble the Part-A per-ticker row (pure): validity, coverage, zero fraction, sigma^2
    distribution, OHLC sanity, and screen decision. ``screen_reason`` is None/'' when kept."""
    v_all = pd.to_numeric(proc["parkinson_volatility"], errors="coerce")
    valid = proc.loc[v_all.notna() & np.isfinite(v_all)]
    n_valid = int(len(valid))
    dts = pd.to_datetime(valid["date"], errors="coerce").dropna().sort_values()
    if len(dts):
        first, last = dts.iloc[0], dts.iloc[-1]
        cal_days = int((last - first).days) + 1
        n_bdays = int(len(pd.bdate_range(first.normalize(), last.normalize())))
        coverage = float(n_valid / n_bdays) if n_bdays else float("nan")
    else:
        first = last = pd.NaT
        cal_days, coverage = 0, float("nan")
    vv = valid["parkinson_volatility"].to_numpy(float) if n_valid else np.array([])
    zero_frac = float(np.mean(vv == 0.0)) if n_valid else float("nan")
    dist = parkinson_dist(proc["parkinson_volatility"])
    ohlc = ohlc_sanity(raw)
    kept = not screen_reason
    return {
        "ticker": ticker,
        "n_valid": n_valid,
        "first_date": first.strftime("%Y-%m-%d") if pd.notna(first) else "",
        "last_date": last.strftime("%Y-%m-%d") if pd.notna(last) else "",
        "calendar_days": cal_days,
        "coverage": coverage,
        "weekday_gaps": weekday_gaps(dts),
        "zero_park_frac": zero_frac,
        "pk_min": dist["min"], "pk_median": dist["median"], "pk_p95": dist["p95"], "pk_max": dist["max"],
        "pk_n_nan": dist["n_nan"], "pk_n_inf": dist["n_inf"], "pk_n_nonpos": dist["n_nonpos"],
        "n_high_lt_low": ohlc["n_high_lt_low"], "n_oc_outside": ohlc["n_oc_outside"],
        "n_nonpos_price": ohlc["n_nonpos_price"], "backfill_lead_zerovol": ohlc["backfill_lead_zerovol"],
        "n_raw_rows": ohlc["n_raw_rows"],
        "kept": kept,
        "screen_decision": "kept" if kept else "excluded",
        "screen_reason": screen_reason or "kept",
    }


def panel_split_stats(tickers, splits, t_mean, floor_rel: float = NODE_FLOOR_REL) -> dict:
    """Per-node train/val/test valid-cell counts, per-split zero-target fraction, and a test
    floor-activation proxy (fraction of test targets at/under the per-node relative floor).

    ``splits`` = {'train': (y, tmask), 'val': (...), 'test': (...)} with y/tmask shaped [n_anchor, N].
    """
    node_floor = floor_rel * np.asarray(t_mean, float) + 1e-12
    out = {}
    for j, tk in enumerate(tickers):
        res = {}
        for name, (y, tm) in splits.items():
            m = tm[:, j].astype(bool)
            nn = int(m.sum())
            yv = y[m, j]
            res[name] = {"n": nn, "zero_frac": float(np.mean(yv == 0.0)) if nn else float("nan")}
        y_te, tm_te = splits["test"]
        mte = tm_te[:, j].astype(bool)
        res["test"]["floor_activation"] = (float(np.mean(y_te[mte, j] <= node_floor[j]))
                                           if int(mte.sum()) else float("nan"))
        out[tk] = res
    return out


# ----------------------------------------------------------------------- Part B: pure MODEL aggregation
def per_ticker_model_metrics(y_te, tmask_te, preds_by_model, tickers, floor) -> dict:
    """Group the test predictions by ticker and compute per-ticker QLIKE/MSE/R2 for each model, using
    the IDENTICAL ``floor`` for every model (shared QLIKE basis). ``preds_by_model`` maps model name ->
    seed-ensembled test prediction array [n_anchor, N]. Skips tickers with no valid test cell."""
    import metrics as M
    out = {}
    tm = np.asarray(tmask_te).astype(bool)
    for j, tk in enumerate(tickers):
        m = tm[:, j]
        if int(m.sum()) == 0:
            continue
        y = np.asarray(y_te)[m, j]
        row = {}
        for name, parr in preds_by_model.items():
            p = np.asarray(parr)[m, j]
            row[name] = {"qlike": M.qlike(y, p, floor), "mse": M.mse(y, p),
                         "r2": M.r2(y, p), "n": int(m.sum())}
        out[tk] = row
    return out


# --------------------------------------------------------------------------------- flags + summary
def flag_row(row: dict, qlike_median: float | None) -> dict:
    """Attach ISSUE FLAGS + severity to a per-ticker row (pure). Red: zero-frac>0.3, any OHLC
    violation, or n_valid<252. Amber: test floor-activation>0.2 or an extreme per-ticker model QLIKE
    (> FLAG_QLIKE_MULT x panel median). ``qlike_median`` is None when model metrics are pending."""
    red, amber = [], []
    zf = row.get("zero_park_frac")
    if zf is not None and np.isfinite(zf) and zf > FLAG_ZERO_FRAC:
        red.append(f"zero-Parkinson frac {zf:.2f}>{FLAG_ZERO_FRAC}")
    ohlc_bad = (row.get("n_high_lt_low", 0) + row.get("n_oc_outside", 0)
                + row.get("n_nonpos_price", 0))
    if ohlc_bad > 0:
        red.append(f"OHLC violations={ohlc_bad}")
    if row.get("n_valid", 0) < FLAG_MIN_ROWS:
        red.append(f"n_valid {row.get('n_valid', 0)}<{FLAG_MIN_ROWS}")
    fa = row.get("floor_activation")
    if fa is not None and np.isfinite(fa) and fa > FLAG_FLOOR_ACT:
        amber.append(f"floor-activation {fa:.2f}>{FLAG_FLOOR_ACT}")
    q = row.get("qlike_max")
    if (q is not None and np.isfinite(q) and qlike_median is not None
            and np.isfinite(qlike_median) and q > FLAG_QLIKE_MULT * qlike_median):
        amber.append(f"QLIKE {q:.2f}>{FLAG_QLIKE_MULT:.0f}x median")
    severity = "red" if red else ("amber" if amber else "ok")
    return {**row, "flags": red + amber, "severity": severity}


def build_summary(rows: list) -> dict:
    """Top-of-page summary: counts flagged, worst-10 lists per issue, aggregate zero-Parkinson."""
    n = len(rows)
    n_red = sum(1 for r in rows if r["severity"] == "red")
    n_amber = sum(1 for r in rows if r["severity"] == "amber")
    total_valid = sum(int(r.get("n_valid", 0)) for r in rows)
    weighted_zero = sum(float(r["zero_park_frac"]) * int(r.get("n_valid", 0))
                        for r in rows if np.isfinite(r.get("zero_park_frac", np.nan)))
    agg_zero = float(weighted_zero / total_valid) if total_valid else float("nan")
    per_ticker_zero = [r["zero_park_frac"] for r in rows if np.isfinite(r.get("zero_park_frac", np.nan))]
    mean_zero = float(np.mean(per_ticker_zero)) if per_ticker_zero else float("nan")

    def worst(key, reverse=True, finite_only=True, limit=10):
        vals = [(r["ticker"], r.get(key)) for r in rows
                if not finite_only or (r.get(key) is not None and np.isfinite(r.get(key, np.nan)))]
        vals.sort(key=lambda kv: kv[1], reverse=reverse)
        return vals[:limit]

    return {
        "n_tickers": n, "n_red": n_red, "n_amber": n_amber, "n_ok": n - n_red - n_amber,
        "agg_zero_park_frac": agg_zero, "mean_ticker_zero_frac": mean_zero, "total_valid_rows": total_valid,
        "worst_zero_frac": worst("zero_park_frac"),
        "worst_fewest_rows": worst("n_valid", reverse=False),
        "worst_ohlc": worst("ohlc_total"),
        "worst_floor_activation": worst("floor_activation"),
        "worst_qlike": worst("qlike_max"),
    }


# ------------------------------------------------------------------------------------- rendering
def _cell(val, fmt="{:.4f}"):
    if val is None:
        return "<td class='na'>pending</td>"
    if isinstance(val, float) and not np.isfinite(val):
        return "<td class='na'>-</td>"
    if isinstance(val, float):
        return f"<td>{fmt.format(val)}</td>"
    return f"<td>{val}</td>"


_COLUMNS = [
    ("ticker", "ticker", None), ("severity", "flag", None), ("screen_decision", "screen", None),
    ("screen_reason", "reason", None), ("n_valid", "n_valid", "{:d}"),
    ("first_date", "first", None), ("last_date", "last", None), ("weekday_gaps", "wkday_gaps", "{:d}"),
    ("coverage", "coverage", "{:.3f}"), ("zero_park_frac", "zero_frac", "{:.3f}"),
    ("pk_median", "pk_median", "{:.2e}"), ("pk_p95", "pk_p95", "{:.2e}"), ("pk_max", "pk_max", "{:.2e}"),
    ("pk_n_nan", "pk_nan", "{:d}"), ("pk_n_inf", "pk_inf", "{:d}"), ("pk_n_nonpos", "pk_nonpos", "{:d}"),
    ("n_high_lt_low", "high<low", "{:d}"), ("n_oc_outside", "oc_out", "{:d}"),
    ("n_nonpos_price", "nonpos_px", "{:d}"), ("backfill_lead_zerovol", "backfill", "{:d}"),
    ("train_n", "train_n", "{:d}"), ("val_n", "val_n", "{:d}"), ("test_n", "test_n", "{:d}"),
    ("floor_activation", "floor_act", "{:.3f}"),
    ("qlike_LSTM", "QLIKE LSTM", "{:.3f}"), ("qlike_HARX", "QLIKE HAR-X", "{:.3f}"),
    ("qlike_VolGA", "QLIKE VolGA", "{:.3f}"),
    ("r2_LSTM", "R2 LSTM", "{:.3f}"),
]


def render_html(rows: list, summary: dict, meta: dict) -> str:
    """Build a self-contained interactive HTML (vanilla JS sortable/filterable table, no CDN)."""
    head = ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            "<title>HNX per-ticker diagnostic</title><style>"
            "body{font-family:system-ui,Arial,sans-serif;margin:20px;color:#1a1a1a}"
            "h1{font-size:20px}h2{font-size:16px;border-bottom:2px solid #ddd;margin-top:24px}"
            ".note{color:#555;font-size:13px;max-width:1100px}"
            "table{border-collapse:collapse;font-size:12px;margin-top:8px}"
            "th,td{border:1px solid #ccc;padding:3px 7px;text-align:right}"
            "th{background:#f2f2f2;cursor:pointer;position:sticky;top:0}"
            "td.na{color:#999}td:first-child,th:first-child{text-align:left}"
            "tr.red td{background:#fde0e0}tr.amber td{background:#fff4d6}"
            ".pill{padding:1px 6px;border-radius:4px;font-weight:600}"
            ".pill.red{background:#c0392b;color:#fff}.pill.amber{background:#e0a800;color:#111}"
            ".pill.ok{background:#e6e6e6;color:#333}"
            "input,button{font-size:13px;padding:3px 6px;margin-right:8px}"
            "ul{font-size:13px}</style></head><body>")
    parts = [head, "<h1>HNX per-ticker diagnostic</h1>"]
    parts.append(f"<p class='note'>Generated {meta.get('generated','')}. One row per HNX ticker "
                 f"({summary['n_tickers']} total). This is a diagnostic to FIND per-stock data/model "
                 "issues. The Parkinson column is a VARIANCE (sigma^2); an exact zero means high==low "
                 "(limit/illiquid day). Screen = floor_sensitivity.screen_files: keep if valid rows &ge; "
                 f"{SCREEN_MIN_ROWS} AND exact-zero-Parkinson fraction &le; {SCREEN_MAX_ZERO_FRAC} AND "
                 f"NaN fraction &le; {SCREEN_MAX_NAN_FRAC}. Model columns: {meta.get('model_status','')}.</p>")
    # summary
    parts.append("<h2>Summary</h2><p class='note'>"
                 f"<b>Flagged:</b> {summary['n_red']} red, {summary['n_amber']} amber, "
                 f"{summary['n_ok']} ok (of {summary['n_tickers']}). "
                 f"<b>Aggregate exact-zero Parkinson</b> (row-weighted across all tickers) = "
                 f"{summary['agg_zero_park_frac']:.3f}; mean per-ticker zero-fraction = "
                 f"{summary['mean_ticker_zero_frac']:.3f}. "
                 "Red = zero-frac&gt;0.30 or any OHLC violation or n_valid&lt;252; "
                 "amber = floor-activation&gt;0.20 or per-ticker QLIKE&gt;2x panel median.</p>")

    def worst_ul(title, items, fmt):
        li = "".join(f"<li>{t}: {fmt.format(v)}</li>" for t, v in items)
        return f"<b>{title}</b><ul>{li}</ul>"

    parts.append("<div style='display:flex;gap:30px;flex-wrap:wrap'>")
    parts.append(worst_ul("Worst-10 zero-Parkinson fraction", summary["worst_zero_frac"], "{:.3f}"))
    parts.append(worst_ul("Worst-10 fewest valid rows", summary["worst_fewest_rows"], "{:d}"))
    parts.append(worst_ul("Worst-10 OHLC violations", summary["worst_ohlc"], "{:d}"))
    parts.append(worst_ul("Worst-10 floor-activation", summary["worst_floor_activation"], "{:.3f}"))
    parts.append(worst_ul("Worst-10 per-ticker QLIKE (max of 3 models)", summary["worst_qlike"], "{:.3f}"))
    parts.append("</div>")
    # controls + table
    parts.append("<h2>Per-ticker table</h2>")
    parts.append("<p class='note'>Click a header to sort. Filter by ticker text; toggle to show only "
                 "flagged rows.</p>")
    parts.append("<input id='q' placeholder='filter ticker...' oninput='flt()'>"
                 "<label><input type='checkbox' id='only' onchange='flt()'> only flagged</label>")
    parts.append("<table id='t'><thead><tr>")
    for i, (_, label, _) in enumerate(_COLUMNS):
        parts.append(f"<th onclick='srt({i})'>{label}</th>")
    parts.append("</tr></thead><tbody>")
    for r in rows:
        parts.append(f"<tr class='{r['severity']}' data-t='{r['ticker'].lower()}' "
                     f"data-flag='{0 if r['severity'] == 'ok' else 1}'>")
        for key, _, fmt in _COLUMNS:
            if key == "severity":
                parts.append(f"<td><span class='pill {r['severity']}'>{r['severity']}</span></td>")
            elif key in ("ticker", "screen_decision", "screen_reason", "first_date", "last_date"):
                parts.append(f"<td>{r.get(key, '')}</td>")
            else:
                parts.append(_cell(r.get(key), fmt or "{:.4f}"))
        parts.append("</tr>")
    parts.append("</tbody></table>")
    # vanilla JS: sort + filter
    parts.append("""<script>
function flt(){var q=document.getElementById('q').value.toLowerCase();
var only=document.getElementById('only').checked;
document.querySelectorAll('#t tbody tr').forEach(function(tr){
var okt=tr.getAttribute('data-t').indexOf(q)>-1;
var okf=!only||tr.getAttribute('data-flag')==='1';
tr.style.display=(okt&&okf)?'':'none';});}
function srt(n){var tb=document.querySelector('#t tbody');
var rows=Array.prototype.slice.call(tb.querySelectorAll('tr'));
var asc=tb.getAttribute('data-sc')==String(n)?tb.getAttribute('data-asc')!=='1':true;
rows.sort(function(a,b){var x=a.cells[n].innerText,y=b.cells[n].innerText;
var nx=parseFloat(x.replace(/,/g,'')),ny=parseFloat(y.replace(/,/g,''));
if(!isNaN(nx)&&!isNaN(ny)){return asc?nx-ny:ny-nx;}
return asc?x.localeCompare(y):y.localeCompare(x);});
rows.forEach(function(r){tb.appendChild(r);});
tb.setAttribute('data-sc',n);tb.setAttribute('data-asc',asc?'1':'0');}
</script>""")
    parts.append("</body></html>")
    return "\n".join(parts)


def render_md(rows: list, summary: dict, meta: dict) -> str:
    """Short markdown: worst tickers + concrete issues + recommendation (objective wording)."""
    lines = [f"# HNX per-ticker diagnostic — {meta.get('generated','')}", "",
             f"One row per HNX ticker ({summary['n_tickers']} total). Diagnostic to find per-stock "
             "data/model issues. Interactive table: "
             "`docs/reports/2026-08-30_hnx_per_ticker_diagnostic.html`.", "",
             f"Model columns: {meta.get('model_status','')}.", "",
             "## Aggregate", "",
             f"- Flagged: {summary['n_red']} red, {summary['n_amber']} amber, {summary['n_ok']} ok.",
             f"- Aggregate exact-zero Parkinson (row-weighted) = {summary['agg_zero_park_frac']:.3f}; "
             f"mean per-ticker zero-fraction = {summary['mean_ticker_zero_frac']:.3f}.",
             f"- Total valid rows across tickers = {summary['total_valid_rows']}.", "",
             "Flag rules — red: exact-zero Parkinson fraction > 0.30, any OHLC violation, or "
             "n_valid < 252. Amber: test floor-activation > 0.20, or per-ticker model QLIKE > 2x the "
             "panel median.", ""]

    def block(title, items, fmt):
        out = [f"### {title}", ""]
        out += [f"- {t}: {fmt.format(v)}" for t, v in items] or ["- (none)"]
        out.append("")
        return out

    lines += ["## Worst tickers", ""]
    lines += block("Highest zero-Parkinson fraction", summary["worst_zero_frac"], "{:.3f}")
    lines += block("Fewest valid rows", summary["worst_fewest_rows"], "{:d}")
    lines += block("Most OHLC violations", summary["worst_ohlc"], "{:d}")
    lines += block("Highest floor-activation", summary["worst_floor_activation"], "{:.3f}")
    lines += block("Highest per-ticker QLIKE (max of 3 models)", summary["worst_qlike"], "{:.3f}")
    lines += ["## Scope note", "",
              "- The delivered target ``parkinson_volatility`` is computed from the intraday range only "
              "(high/low), so an open/close-outside-[low,high] bar does NOT corrupt the Parkinson target "
              "directly; it corrupts the overnight-augmented estimators (Garman-Klass / Rogers-Satchell / "
              "Yang-Zhang) that read open/close. It is still flagged as a raw-data-geometry defect.",
              "- The exact-zero Parkinson fraction (high==low days) is the defect that DOES drive the "
              "Parkinson target and its QLIKE floor.", "",
              "## Recommendation", "",
              "- Tickers flagged red on zero-Parkinson fraction are unreliable: their Parkinson targets are "
              "dominated by limit/illiquid days and their point/QLIKE metrics are floor-driven, not "
              "forecast-driven. Exclude them from headline tables or report separately.",
              "- Tickers with high test floor-activation are floor-driven even inside the screened panel; "
              "note this in the paper's data section as a QLIKE caveat rather than a model result.",
              "- The screen already drops the worst illiquid names; the residual red/amber rows inside the "
              "kept universe are the ones to disclose as data-quality limitations.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------------- drivers (real IO)
def collect_data_rows(processed_dir, price_dir, lookback=10, horizon=1,
                      build_panel=True, min_valid=8, min_train_rows=252):  # pragma: no cover - real IO driver
    """Part A driver: screen all HNX tickers, compute per-ticker data stats for EVERY ticker (kept +
    excluded), and (optionally) attach masked-panel per-split stats for the kept universe. Returns
    (rows, panel_data_or_None, meta_dict)."""
    import floor_sensitivity as FS
    import masked_rich as MR
    all_files = sorted(glob.glob(str(Path(processed_dir) / "*_processed.csv")))
    report: dict = {}
    kept = FS.screen_files(all_files, min_rows=SCREEN_MIN_ROWS, max_zero_frac=SCREEN_MAX_ZERO_FRAC,
                           max_nan_frac=SCREEN_MAX_NAN_FRAC, report=report)
    excluded = report["excluded"]
    rows = []
    for f in all_files:
        tk = Path(f).name.replace("_processed.csv", "")
        proc = pd.read_csv(f)
        rf = Path(price_dir) / f"{tk}_ohlcv.csv"
        raw = pd.read_csv(rf) if rf.exists() else pd.DataFrame()
        reason = excluded.get(f, "")
        row = data_stats_for_ticker(tk, proc, raw, reason)
        row["ohlc_total"] = row["n_high_lt_low"] + row["n_oc_outside"] + row["n_nonpos_price"]
        rows.append(row)
    panel = None
    if build_panel and kept:
        panel = MR.build_masked_rich(kept, str(price_dir), lookback, horizon,
                                     min_valid=min_valid, min_train_rows=min_train_rows)
        splits = {"train": (panel.y_tr, panel.tmask_tr), "val": (panel.y_va, panel.tmask_va),
                  "test": (panel.y_te, panel.tmask_te)}
        pstats = panel_split_stats(panel.tickers, splits, panel.t_mean)
        by_tk = {r["ticker"]: r for r in rows}
        for tk, st in pstats.items():
            r = by_tk[tk]
            r["train_n"] = st["train"]["n"]
            r["val_n"] = st["val"]["n"]
            r["test_n"] = st["test"]["n"]
            r["floor_activation"] = st["test"]["floor_activation"]
            r["test_zero_frac"] = st["test"]["zero_frac"]
            r["in_panel"] = True
    meta = {"n_all": len(all_files), "n_kept": len(kept), "n_excluded": len(excluded)}
    return rows, panel, meta


def run_model_eval(panel, seeds):  # pragma: no cover - GPU training driver
    """Part B driver: reproduce the delivered HNX-h1 eval (HAR-X + no-graph LSTM + VolGA) and return
    seed-ensembled test prediction arrays keyed by model. Uses the delivered config + shared floor."""
    from dataclasses import replace

    import run_masked_rich as RM
    from config import Config
    cfg = replace(Config(), batch_size=32, seeds=tuple(seeds))
    D = panel
    mtr = D.tmask_tr.astype(bool)
    xtr = np.column_stack([np.ones(int(mtr.sum())), D.har5_tr[mtr]])
    cx = np.linalg.lstsq(xtr, D.y_tr[mtr], rcond=None)[0]
    hx = (np.column_stack([np.ones(len(D.har5_te.reshape(-1, 5))), D.har5_te.reshape(-1, 5)]) @ cx
          ).reshape(D.y_te.shape)
    harx = np.maximum(hx, NODE_FLOOR_REL * D.t_mean + 1e-12)
    lstm = np.mean([RM.train_masked_rich(D, cfg, s, False, D.adj_vol2pk, "zscore_floor") for s in seeds], axis=0)
    volga = np.mean([RM.train_masked_rich(D, cfg, s, True, D.adj_vol2pk, "zscore_floor") for s in seeds], axis=0)
    return {"HARX": harx, "LSTM": lstm, "VolGA": volga}, cfg.qlike_floor


def assemble(rows, panel, model_metrics):  # pragma: no cover - thin glue over tested helpers
    """Merge Part-B per-ticker model metrics into the rows, compute panel median QLIKE, and flag."""
    by_tk = {r["ticker"]: r for r in rows}
    qmax_vals = []
    if model_metrics:
        for tk, mm in model_metrics.items():
            r = by_tk[tk]
            r["qlike_LSTM"] = mm["LSTM"]["qlike"]
            r["qlike_HARX"] = mm["HARX"]["qlike"]
            r["qlike_VolGA"] = mm["VolGA"]["qlike"]
            r["r2_LSTM"] = mm["LSTM"]["r2"]
            r["mse_LSTM"] = mm["LSTM"]["mse"]
            r["qlike_max"] = max(mm["LSTM"]["qlike"], mm["HARX"]["qlike"], mm["VolGA"]["qlike"])
            qmax_vals.append(r["qlike_max"])
    qlike_median = float(np.median(qmax_vals)) if qmax_vals else None
    flagged = [flag_row(r, qlike_median) for r in rows]
    summary = build_summary(flagged)
    return flagged, summary


def main():  # pragma: no cover - CLI entry driver
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-model", action="store_true", help="DATA-only (skip GPU model eval)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026, 7, 2024])
    ap.add_argument("--out-html", default=str(REPO / "docs" / "reports" / "2026-08-30_hnx_per_ticker_diagnostic.html"))
    ap.add_argument("--out-md", default=str(REPO / "docs" / "reports" / "2026-08-30_hnx_per_ticker_diagnostic.md"))
    a = ap.parse_args()
    t0 = time.time()
    rows, panel, meta = collect_data_rows(PROCESSED_DIR, PRICE_DIR)
    print(f"[hnx-diag] Part A done: {meta['n_all']} tickers ({meta['n_kept']} kept, "
          f"{meta['n_excluded']} excluded), {time.time()-t0:.0f}s", flush=True)
    model_metrics = None
    model_status = "pending — GPU busy / skipped (DATA-only)"
    if not a.no_model and panel is not None:
        import metrics as M  # noqa: F401  (ensure importable before the heavy run)
        preds, floor = run_model_eval(panel, a.seeds)
        model_metrics = per_ticker_model_metrics(panel.y_te, panel.tmask_te, preds, panel.tickers, floor)
        model_status = (f"no-graph LSTM / HAR-X / VolGA, HNX h1, {len(a.seeds)} seeds, shared QLIKE "
                        f"floor={floor:g} (reproduces the delivered run_masked_rich config)")
        print(f"[hnx-diag] Part B done: {len(model_metrics)} tickers scored, {time.time()-t0:.0f}s", flush=True)
    flagged, summary = assemble(rows, panel, model_metrics)
    meta_out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "model_status": model_status}
    Path(a.out_html).write_text(render_html(flagged, summary, meta_out), encoding="utf-8")
    Path(a.out_md).write_text(render_md(flagged, summary, meta_out), encoding="utf-8")
    print(f"[hnx-diag] wrote {a.out_html} and {a.out_md} | flagged {summary['n_red']} red / "
          f"{summary['n_amber']} amber of {summary['n_tickers']}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
