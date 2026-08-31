"""Self-contained HTML build report for the enriched-processed build (baseline A3)."""
from __future__ import annotations

from pathlib import Path

_CSS = ("body{font-family:system-ui,Arial,sans-serif;margin:24px;max-width:1050px}"
        "table{border-collapse:collapse;font-size:13px;margin:10px 0}"
        "td,th{border:1px solid #ccc;padding:4px 9px;text-align:right}"
        "th{background:#f2f2f2;text-align:center}td.l{text-align:left}"
        "h2{border-bottom:2px solid #ddd;margin-top:28px}.note{color:#555;font-size:13px}")


def _fmt(x, d=6) -> str:
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f"{x:,}"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if xf != xf:  # NaN
        return "-"
    return f"{xf:.{d}g}"


def _table(headers, rows) -> str:
    head = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    body = "".join("<tr>" + "".join(f"<td class='l'>{c}</td>" if i == 0 else f"<td>{c}</td>"
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table>{head}{body}</table>"


def build_html_report(summaries: dict, out_path, regression: dict | None = None) -> Path:
    """Render per-market build stats to a single self-contained HTML file. Returns the path written."""
    out_path = Path(out_path)
    parts = ["<html><head><meta charset='utf-8'><title>Enriched processed build</title>",
             f"<style>{_CSS}</style></head><body>",
             "<h1>Enriched processed-data build report</h1>",
             "<p class='note'>ETL-cleaned + causal-enriched per-ticker files "
             "(<code>data/processed_enriched/&lt;market&gt;/</code>). All columns are backward-looking; "
             "no train/val/test-boundary statistic is baked in.</p>"]

    parts.append("<h2>Rows in / out and dirty bars</h2>")
    rows = [[s["market"], s["n_tickers"], _fmt(s["rows_in"]), _fmt(s["rows_out"]),
             _fmt(s["n_dropped"]), _fmt(s["n_dirty_bars"])] for s in summaries.values()]
    parts.append(_table(["market", "tickers", "rows_in", "rows_out", "dropped", "dirty_bars"], rows))

    parts.append("<h2>Dirty bars by class (raw-bar detectors)</h2>")
    classes = list(next(iter(summaries.values()))["dirty_by_class"]) if summaries else []
    rows = [[s["market"]] + [_fmt(s["dirty_by_class"][c]) for c in classes] for s in summaries.values()]
    parts.append(_table(["market"] + classes, rows))

    parts.append("<h2>cleaning_applied breakdown</h2>")
    labels = sorted({lab for s in summaries.values() for lab in s["cleaning_applied"]})
    rows = [[s["market"]] + [_fmt(s["cleaning_applied"].get(lab, 0)) for lab in labels]
            for s in summaries.values()]
    parts.append(_table(["market"] + labels, rows))

    parts.append("<h2>Estimator means (valid bars) — Parkinson vs GK / RS / YZ agreement</h2>")
    ecols = ["parkinson_variance", "garman_klass_variance", "rogers_satchell_variance", "yang_zhang_n20"]
    rows = [[s["market"]] + [_fmt(s["estimator_mean"][c]) for c in ecols] for s in summaries.values()]
    parts.append(_table(["market"] + ecols, rows))

    parts.append("<h2>market_pk sanity (cross-sectional mean of parkinson_variance)</h2>")
    rows = [[s["market"], _fmt(s["market_pk"]["n_days"]), _fmt(s["market_pk"]["min"]),
             _fmt(s["market_pk"]["mean"]), _fmt(s["market_pk"]["max"])] for s in summaries.values()]
    parts.append(_table(["market", "n_days", "min", "mean", "max"], rows))

    if regression is not None:
        parts.append("<h2>Clean-bar regression vs existing data/processed (VN30)</h2>")
        parts.append("<p class='note'>Enriched parkinson_variance vs the delivered value on NON-dirty, "
                     "non-capped bars. The 0.1 cap is a downstream modeling floor, not the causal estimator, "
                     "so capped bars are excluded (and preserved uncapped in the enriched file).</p>")
        parts.append(_table(["worst_noncapped_diff", "n_capped(excluded)", "n_compared"],
                            [[_fmt(regression["worst_noncapped_diff"], 3),
                              _fmt(regression["n_capped"]), _fmt(regression["n_compared"])]]))

    parts.append("</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
