"""Compare daily volatility-VARIANCE estimators from raw OHLCV, focused on the low-range (H approx L)
problem that inflates QLIKE on illiquid panels.

Per-day estimators (all are variances, comparable to the current Parkinson target):
  close2close   : r_t^2, r_t = ln(C_t / C_{t-1})                         (uses only closes; has overnight)
  parkinson     : ln(H/L)^2 / (4 ln2)                                    (CURRENT target; intraday range only)
  garman_klass  : 0.5 ln(H/L)^2 - (2 ln2 - 1) ln(C/O)^2                  (intraday; adds open-close)
  rogers_satchell: ln(H/C)ln(H/O) + ln(L/C)ln(L/O)                       (intraday; drift-independent)
  rs_overnight  : rogers_satchell + ln(O_t / C_{t-1})^2                  (RS + overnight gap; Yang-Zhang-like per-day)

Key question: on days with H approx L (limit / thin trading) every INTRADAY estimator collapses to ~0 and gets
floored (the QLIKE driver found by the EDA). Only estimators with an OVERNIGHT term (close2close, rs_overnight)
stay non-zero when the price gapped/limit-moved from the prior close. This tool quantifies that per panel.

Usage: python scripts/eda/volatility_estimators.py --panels vn30 vn100 hose hnx sp500 --out <file>.html
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "garch_masked"))

_LN2 = np.log(2.0)
_YZ_N = 20                                            # Yang-Zhang window (RMA period / k parameter)
_YZ_K = 0.34 / (1.34 + (_YZ_N + 1) / (_YZ_N - 1))     # classical YZ weight ~0.139 for n=20
EST = ["close2close", "parkinson", "garman_klass", "rogers_satchell", "rs_overnight"]

PRICE = {
    "vn30": REPO / "data" / "raw" / "prices",
    "vn100": REPO / "data" / "raw" / "prices" / "vn100_vnstock",
    "hose": REPO / "data" / "raw" / "prices" / "hose_vnstock",
    "hnx": REPO / "data" / "raw" / "prices" / "hnx_vnstock",
    "sp500": REPO / "data" / "raw" / "prices" / "sp500",
}


def estimators_from_ohlcv(df: pd.DataFrame, overnight_cap: float | None = 0.20) -> pd.DataFrame:
    """Return a DataFrame with the five variance estimators for each valid OHLCV row (positive prices,
    high>=max(open,close,low), low<=min(...)). NaN where inputs are invalid or no prior close.

    The OVERNIGHT return ln(O_t / C_{t-1}) requires a positive PRIOR close and is corrupted by unadjusted
    splits / zero or missing prior closes (which the intraday Parkinson estimator is immune to). We therefore
    (a) require ``prev_close > 0`` and (b) winsorize the overnight log-return to +/- ``overnight_cap`` (default
    0.20 ~ twice the +/-10% VN price limit) so a few unadjusted-split / bad-data days do not dominate the
    overnight-bearing estimators. Set ``overnight_cap=None`` to disable winsorization (raw behavior)."""
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    ok = np.isfinite([o, h, lo, c]).all(0) & (o > 0) & (h > 0) & (lo > 0) & (c > 0) & (h >= lo)
    prev_c = np.concatenate([[np.nan], c[:-1]])
    prev_ok = np.isfinite(prev_c) & (prev_c > 0)             # overnight needs a valid positive prior close
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_hl = np.log(h / lo)
        ln_co = np.log(c / o)
        ln_ho, ln_hc = np.log(h / o), np.log(h / c)
        ln_lo, ln_lc = np.log(lo / o), np.log(lo / c)
        r_cc = np.where(prev_ok, np.log(c / prev_c), np.nan)      # close-to-close return
        r_on = np.where(prev_ok, np.log(o / prev_c), np.nan)      # overnight (gap) return
        if overnight_cap is not None:                            # winsorize away unadjusted-split/bad-data spikes
            r_cc = np.clip(r_cc, -overnight_cap, overnight_cap)
            r_on = np.clip(r_on, -overnight_cap, overnight_cap)
        park = ln_hl ** 2 / (4 * _LN2)
        gk = 0.5 * ln_hl ** 2 - (2 * _LN2 - 1) * ln_co ** 2
        rs = np.clip(ln_hc * ln_ho + ln_lc * ln_lo, 0.0, None)   # Rogers-Satchell (clipped >=0)
        c2c = r_cc ** 2
        rs_ov = rs + r_on ** 2
        # Yang-Zhang per-day (indicator form, e.g. TradingView YZV): instantaneous daily variance using
        # single-day squared overnight + single-day squared open-close + Rogers-Satchell, blended with the
        # classical YZ weight k = 0.34/(1.34+(n+1)/(n-1)) (n = _YZ_N). This IS computable each day (unlike
        # the academic YZ, which uses windowed variances of the returns).
        yz_daily = r_on ** 2 + _YZ_K * ln_co ** 2 + (1.0 - _YZ_K) * rs
    out = pd.DataFrame({
        "close2close": c2c, "parkinson": park, "garman_klass": gk,
        "rogers_satchell": rs, "rs_overnight": rs_ov, "yz_daily": yz_daily,
        "is_zero_range": np.isclose(ln_hl, 0.0),      # H approx L day
        "ok": ok,
    })
    for e in ["garman_klass", "rogers_satchell", "rs_overnight", "yz_daily"]:
        out[e] = out[e].clip(lower=0.0)               # variance estimators are non-negative
    # yz_rma20: the TradingView YZV output = Wilder RMA smoothing of yz_daily (trailing, no look-ahead).
    # NOTE: a smoothed target is highly autocorrelated -> forecasts look artificially easy; it no longer
    # measures next-day variance. Kept for parity with the indicator, flagged in the report.
    out["yz_rma20"] = pd.Series(yz_daily).ewm(alpha=1.0 / _YZ_N, adjust=False, ignore_na=True).mean().to_numpy()
    out.loc[~ok, EST + ["yz_daily", "yz_rma20"]] = np.nan
    return out


def panel_summary(panel: str, price_dir: Path, rel_floor: float = 1e-2) -> dict:
    """Aggregate estimator diagnostics over all tickers of a panel."""
    files = sorted(glob.glob(str(price_dir / "*_ohlcv.csv")))
    rows = []
    zero_among_hl = {e: [0, 0] for e in EST}   # [rescued(non-floored), total H~L days]
    for f in files:
        df = pd.read_csv(f)
        if not {"open", "high", "low", "close"} <= set(df.columns):
            continue
        est = estimators_from_ohlcv(df)
        valid = est[est["ok"]]
        if len(valid) < 50:
            continue
        rec = {"ticker": Path(f).stem.replace("_ohlcv", ""), "n": len(valid)}
        for e in EST:
            v = valid[e].dropna().to_numpy()
            if v.size == 0:
                continue
            med = np.median(v[v > 0]) if (v > 0).any() else 0.0
            floor = rel_floor * med
            rec[f"{e}_zero_frac"] = float(np.mean(v <= floor))     # fraction floored (the QLIKE driver)
            rec[f"{e}_mean"] = float(np.mean(v))
        # on H~L days, is the estimator rescued (non-floored) by an overnight term?
        hl = est[est["is_zero_range"] & est["ok"]]
        for e in EST:
            v = hl[e].dropna().to_numpy()
            if v.size:
                med = np.median(valid[e].dropna()[valid[e].dropna() > 0]) if (valid[e].dropna() > 0).any() else 0.0
                floor = rel_floor * med
                zero_among_hl[e][0] += int(np.sum(v > floor))
                zero_among_hl[e][1] += int(v.size)
        rows.append(rec)
    df = pd.DataFrame(rows)
    resc = {e: (zero_among_hl[e][0] / zero_among_hl[e][1] if zero_among_hl[e][1] else np.nan) for e in EST}
    return {"panel": panel, "per_ticker": df, "n_hl_days": zero_among_hl["parkinson"][1],
            "rescued_on_hl": resc}


def _fmt(x, d=3):
    return f"{x:.{d}f}" if isinstance(x, float) and np.isfinite(x) else "-"


def render_html(summaries, out_path):
    parts = ["<html><head><meta charset='utf-8'><title>Volatility estimator comparison</title>",
             "<style>body{font-family:system-ui,Arial,sans-serif;margin:24px;max-width:1000px}"
             "table{border-collapse:collapse;font-size:13px;margin:10px 0}td,th{border:1px solid #ccc;"
             "padding:4px 9px;text-align:center}th{background:#f2f2f2}h2{border-bottom:2px solid #ddd}"
             ".note{color:#555;font-size:13px}b.k{color:#b00}</style></head><body>"]
    parts.append("<h1>Daily volatility-variance estimators — low-range (H&approx;L) diagnostic</h1>")
    parts.append("<p class='note'>All five are variance estimators comparable to the current Parkinson target. "
                 "<b>zero_frac</b> = fraction of days at/under a relative floor (1e-2&times;median) = days that would "
                 "be floored and inflate QLIKE. <b>rescued_on_HL</b> = of the H&approx;L (zero intraday range) days, "
                 "the fraction where the estimator is still non-floored — only estimators with an OVERNIGHT term "
                 "(close2close, rs_overnight) can rescue a limit/gap day.</p>")
    # Table 1: floored-day fraction per estimator per panel
    parts.append("<h2>Fraction of floored (near-zero) days, per estimator</h2>")
    parts.append("<table><tr><th>panel</th><th>tickers</th>" + "".join(f"<th>{e}</th>" for e in EST) + "</tr>")
    for s in summaries:
        d = s["per_ticker"]
        cells = "".join(f"<td>{_fmt(d[f'{e}_zero_frac'].mean())}</td>" for e in EST)
        parts.append(f"<tr><td><b>{s['panel']}</b></td><td>{len(d)}</td>{cells}</tr>")
    parts.append("</table>")
    # Table 2: rescue rate on H~L days
    parts.append("<h2>Rescue rate on H&approx;L days (fraction still non-floored)</h2>")
    parts.append("<table><tr><th>panel</th><th>H&approx;L days</th>" + "".join(f"<th>{e}</th>" for e in EST) + "</tr>")
    for s in summaries:
        cells = "".join(f"<td>{_fmt(s['rescued_on_hl'][e])}</td>" for e in EST)
        parts.append(f"<tr><td><b>{s['panel']}</b></td><td>{s['n_hl_days']}</td>{cells}</tr>")
    parts.append("</table>")
    parts.append("<p class='note'>Reading: if <b>parkinson</b> has a high floored-day fraction on HOSE/HNX and "
                 "<b>rs_overnight</b>/<b>close2close</b> rescue most H&approx;L days, then switching to an "
                 "overnight-augmented estimator would directly reduce the QLIKE inflation identified by the EDA.</p>")
    parts.append("</body></html>")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(parts), encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+", default=["vn30", "vn100", "hose", "hnx", "sp500"])
    ap.add_argument("--out", default=str(REPO / "docs" / "reports" / "eda" / "volatility_estimators.html"))
    a = ap.parse_args()
    summaries = []
    for panel in a.panels:
        s = panel_summary(panel, PRICE[panel])
        summaries.append(s)
        d = s["per_ticker"]
        print(f"[est] {panel}: {len(d)} tickers | floored-day frac: "
              + " ".join(f"{e}={d[f'{e}_zero_frac'].mean():.3f}" for e in EST), flush=True)
        print(f"       H~L rescue: " + " ".join(f"{e}={s['rescued_on_hl'][e]:.3f}" for e in EST), flush=True)
    p = render_html(summaries, a.out)
    print(f"[est] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
