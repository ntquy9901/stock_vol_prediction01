"""Render the Experiment-B diagnostic ("why adding the topology graph does not help") to a standalone HTML.

Consumes ``results/gamma_gbm/complex_network_<market>.json`` (headline QLIKE + DM) and
``results/gamma_gbm/complex_network_<market>_diag.json`` (:mod:`diag_gbm`) and writes a self-contained page:
per-horizon head-to-head QLIKE, train-vs-test QLIKE (overfitting signature), prediction/error correlation,
and QLIKE permutation-importance bars split into own-history vs topology features.

Run: ``python build_topo_diag_html.py [hose|sp500]``.  Output: docs/reports/<date>_complex_network_<market>_why_topo_hurts.html
"""
import json
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap
import config  # noqa: E402

TOPO = set(config.TOPO)
_CSS = """body{font:14px/1.5 system-ui,Segoe UI,Arial;margin:24px;color:#1a1a1a;max-width:1000px}
h1{font-size:22px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}
table{border-collapse:collapse;margin:8px 0}th,td{border:1px solid #ccc;padding:5px 10px;text-align:right}
th{background:#f2f2f2}td.l,th.l{text-align:left}.good{color:#0a7d2c}.bad{color:#c0271a}
.bar{height:15px;display:inline-block;vertical-align:middle;border-radius:2px}
.own{background:#2b7bba}.topo{background:#d98c1f}.k{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:middle}
.muted{color:#666}.card{background:#fafafa;border:1px solid #e5e5e5;border-radius:6px;padding:10px 16px;margin:8px 0}"""


def _sign(x, inv=False):
    """CSS class for a signed delta: green when helpful. ``inv=True`` flips (lower-is-better metric)."""
    good = (x < 0) if inv else (x > 0)
    return "good" if good else "bad"


def _headline_table(result):
    rows = ["<tr><th class=l>Horizon</th><th>GBM QLIKE</th><th>GBM+topo QLIKE</th>"
            "<th>gain %</th><th>DM p</th></tr>"]
    for h, r in result.items():
        cls = _sign(r["gain_pct"])
        rows.append(f"<tr><td class=l>{h}</td><td>{r['GBM']:.4f}</td><td>{r['GBM+topo']:.4f}</td>"
                    f"<td class={cls}>{r['gain_pct']:+.2f}%</td><td>{r['dm_p']:.3f}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def _trainvtest_table(diag):
    rows = ["<tr><th class=l>Horizon</th><th>train GBM</th><th>train +topo</th><th>train Δ</th>"
            "<th>test GBM</th><th>test +topo</th><th>test Δ</th></tr>"]
    for h, r in diag.items():
        trd = r["train_qlike"]["GBM"] - r["train_qlike"]["GBM+topo"]      # >0: topo lowers train QLIKE
        ted = r["test_qlike"]["GBM"] - r["test_qlike"]["GBM+topo"]        # >0: topo lowers test QLIKE
        rows.append(
            f"<tr><td class=l>{h}</td>"
            f"<td>{r['train_qlike']['GBM']:.4f}</td><td>{r['train_qlike']['GBM+topo']:.4f}</td>"
            f"<td class={_sign(trd)}>{trd:+.4f}</td>"
            f"<td>{r['test_qlike']['GBM']:.4f}</td><td>{r['test_qlike']['GBM+topo']:.4f}</td>"
            f"<td class={_sign(ted)}>{ted:+.4f}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def _corr_table(diag):
    rows = ["<tr><th class=l>Horizon</th><th>pred corr</th><th>QLIKE-err corr</th>"
            "<th>topo importance frac</th></tr>"]
    for h, r in diag.items():
        rows.append(f"<tr><td class=l>{h}</td><td>{r['pred_corr']:.4f}</td>"
                    f"<td>{r['qlike_err_corr']:.4f}</td><td>{r['topo_importance_frac']*100:.2f}%</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def _importance_bars(imp):
    """Horizontal bars for one horizon's permutation importance, sorted by magnitude, own vs topo coloured."""
    items = sorted(imp.items(), key=lambda kv: -abs(kv[1]))
    scale = max((abs(v) for v in imp.values()), default=0.0) or 1.0
    out = []
    for name, val in items:
        cls = "topo" if name in TOPO else "own"
        width = int(abs(val) / scale * 260)
        out.append(f"<div><span style='display:inline-block;width:110px' class=muted>{name}</span>"
                   f"<span class='bar {cls}' style='width:{width}px'></span> "
                   f"<span>{val:+.4f}</span></div>")
    return "".join(out)


def _importance_section(diag):
    blocks = []
    for h, r in diag.items():
        blocks.append(f"<div class=card><b>{h}</b> "
                      f"<span class=muted>(own Σ={r['own_importance_sum']:+.4f}, "
                      f"topo Σ={r['topo_importance_sum']:+.4f})</span>{_importance_bars(r['perm_importance'])}</div>")
    return "".join(blocks)


def build_html(result, diag, market):
    """Assemble the full standalone HTML string from the two result dicts."""
    legend = ("<span class='k own'></span> own-history (HAR + realized) &nbsp; "
              "<span class='k topo'></span> topology metric")
    return (
        f"<!doctype html><html><head><meta charset=utf-8><title>Why topology does not help "
        f"({market})</title><style>{_CSS}</style></head><body>"
        f"<h1>Experiment B — why adding the topology metrics does not help ({market})</h1>"
        f"<p class=muted>Per-stock gamma-GBM, own-history (9) vs own+topology (15). QLIKE lower is better.</p>"
        f"<h2>1. Head-to-head QLIKE (walk-forward, Diebold-Mariano)</h2>{_headline_table(result)}"
        f"<p class=muted>Negative gain and small DM p at h1 means the topology model is significantly worse; "
        f"the other horizons are statistical ties.</p>"
        f"<h2>2. Train vs test QLIKE — the overfitting signature</h2>{_trainvtest_table(diag)}"
        f"<p class=muted>Positive train Δ with a non-positive test Δ means topology lowers in-sample error "
        f"(extra capacity fits noise) without lowering out-of-sample error.</p>"
        f"<h2>3. How little topology changes the forecast</h2>{_corr_table(diag)}"
        f"<p class=muted>Prediction correlation near 1 means the topology features barely move the forecast; "
        f"topo importance fraction is the share of total QLIKE permutation importance held by the topology metrics.</p>"
        f"<h2>4. QLIKE permutation importance (final fold)</h2><p class=muted>{legend}</p>"
        f"{_importance_section(diag)}"
        f"<p class=muted>Bar length is the mean QLIKE increase when a feature is shuffled. Own-history features "
        f"carry the signal; the topology metrics sit near zero.</p>"
        f"</body></html>")


def main():  # pragma: no cover - entry driver: reads JSON, writes HTML
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    gj = REPO / "results" / "gamma_gbm"
    result = json.loads((gj / f"complex_network_{market}.json").read_text())
    diag = json.loads((gj / f"complex_network_{market}_diag.json").read_text())
    html = build_html(result, diag, market)
    outp = REPO / "docs" / "reports" / f"2026-09-12_complex_network_{market}_why_topo_hurts.html"
    outp.write_text(html, encoding="utf-8")
    print(f"wrote {outp.relative_to(REPO)} ({len(html)} bytes)", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
