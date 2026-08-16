"""Generate a self-contained HTML that visualises WHY graph and news add little to the model.

Panels: (1) QLIKE difference vs HAR per rung/horizon as diverging bars (bars hugging 0 = no
contribution); (2) leave-one-out DM contribution table, colour-coded (helps/hurts/none); (3) 3-seed
vs 5-seed robustness of the marginal cells; (4) root-cause notes. Offline (no CDN/JS libs).

Run: python scripts/viz/graph_news_contribution.py [out.html]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code",
           _ROOT / "baselines" / "2026-08-15_volatility" / "code", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

QLIKE_TS = "2026-08-16_141447_gnnhar_qlike"
SEEDS3 = [42, 123, 2026]
HORIZONS = [1, 5, 10, 22]
RUNGS = ["FULL", "minus_graph", "minus_gate", "minus_news", "lstm_only"]

# 5-seed robustness of the marginal (p ~ 0.05) cells checked this session.
ROBUSTNESS = [
    {"cell": "graph edge (graphical-LASSO) beats HAR @ h5", "three": "p=0.048 *", "five": "p=0.286",
     "outcome": "collapsed", "note": "graph edge does NOT beat HAR once multi-seed"},
    {"cell": "LSTM nonlinearity beats fair HAR-X @ h5", "three": "p=0.07", "five": "p=0.03 *",
     "outcome": "strengthened", "note": "nonlinearity edge is real at h5 too (not the graph)"},
    {"cell": "FULL vs HAR @ h10 (HAR better)", "three": "p=0.014 *", "five": "p=0.023 *",
     "outcome": "robust", "note": "HAR stays better at h10"},
    {"cell": "graph HURTS @ h10 (FULL vs minus_graph)", "three": "p=0.045 *", "five": "p=0.001 *",
     "outcome": "robust", "note": "removing the graph significantly improves QLIKE at h10"},
]


def verdict(component: str, dm: float, p: float) -> str:
    """LOO verdict from DM(FULL vs minus_X): p>=.05 -> 'none'; dm<0 -> 'helps'; else 'hurts'."""
    if p >= 0.05:
        return "none"
    return "helps" if dm < 0 else "hurts"


def bar_geometry(delta: float, scale: float) -> dict:
    """Diverging-bar geometry for a QLIKE delta vs HAR. Negative = better (left), positive = worse."""
    width = min(abs(delta) / scale * 100.0, 100.0) if scale > 0 else 0.0
    side = "none" if delta == 0 else ("better" if delta < 0 else "worse")
    return {"width_pct": width, "side": side}


def contribution_data() -> dict:  # pragma: no cover  (reads real result dumps)
    import numpy as np
    import dm_report as dm
    dm.RESULTS = _ROOT / "results"
    R = _ROOT / "results"

    def q(rung, h):
        return float(np.mean([
            json.load(open(R / f"volatility_ablation_h{h}_seed{s}_{QLIKE_TS}" / "ladder_metrics.json"))
            ["rungs"][rung]["test_metrics"]["qlike"] for s in SEEDS3]))

    qlike = {r: {h: q(r, h) for h in HORIZONS} for r in (["HAR"] + RUNGS)}
    loo = {}
    for comp, rung in (("graph", "minus_graph"), ("news", "minus_news"), ("gate", "minus_gate")):
        loo[comp] = {}
        for h in HORIZONS:
            res = dm.dm_pair(QLIKE_TS, h, "FULL", rung, SEEDS3, loss="qlike")
            loo[comp][h] = (res["dm_hln"], res["p_value"])
    return {"horizons": HORIZONS, "rungs": RUNGS, "qlike": qlike, "loo": loo,
            "robustness": ROBUSTNESS}


_CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;color:#1a1a1a;background:#fafafa}
h1{font-size:22px} h2{font-size:17px;margin-top:28px;border-bottom:2px solid #ddd;padding-bottom:4px}
table{border-collapse:collapse;margin:8px 0;font-size:13px}
td,th{border:1px solid #ccc;padding:6px 10px;text-align:center}
th{background:#eee}
.helps{background:#c8e6c9} .hurts{background:#ffcdd2} .none{background:#eceff1;color:#666}
.row{display:flex;align-items:center;margin:3px 0;font-size:12px}
.lbl{width:130px;text-align:right;padding-right:8px;color:#333}
.track{flex:1;position:relative;height:18px;background:#f0f0f0;border-left:2px solid #888}
.bar{position:absolute;height:18px;top:0}
.better{background:#43a047;right:50%} .worse{background:#e53935;left:50%}
.center{position:absolute;left:50%;top:-2px;bottom:-2px;border-left:1px dashed #888}
.val{width:70px;padding-left:8px;color:#333}
.collapsed{color:#e53935;font-weight:bold} .strengthened{color:#2e7d32;font-weight:bold} .robust{color:#555}
.note{color:#555;font-size:13px;max-width:820px}
.legend span{display:inline-block;padding:2px 8px;margin-right:6px;border-radius:3px;font-size:12px}
"""


def render_html(data: dict) -> str:
    hz = data["horizons"]
    # panel 1: diverging bars of QLIKE - HAR (per horizon, per rung)
    scale = 0.03  # |delta| of 0.03 fills the half-track
    bars_html = []
    for h in hz:
        bars_html.append(f'<h3 style="font-size:14px;margin:14px 0 4px">Horizon h{h} '
                         f'(HAR QLIKE = {data["qlike"]["HAR"][h]:.4f})</h3>')
        for r in data["rungs"]:
            delta = data["qlike"][r][h] - data["qlike"]["HAR"][h]
            g = bar_geometry(delta, scale)
            bar = (f'<div class="bar {g["side"]}" style="width:{g["width_pct"] / 2:.1f}%"></div>'
                   if g["side"] != "none" else "")
            bars_html.append(
                f'<div class="row"><div class="lbl">{r}</div>'
                f'<div class="track"><div class="center"></div>{bar}</div>'
                f'<div class="val">{delta:+.4f}</div></div>')
    # panel 2: LOO contribution table
    head = "".join(f"<th>h{h}</th>" for h in hz)
    rows = []
    for comp in ("graph", "news", "gate"):
        cells = []
        for h in hz:
            dmv, p = data["loo"][comp][h]
            v = verdict(comp, dmv, p)
            label = {"helps": "helps", "hurts": "no help", "none": "ns"}[v]
            cells.append(f'<td class="{v}">{label}<br><small>dm={dmv:+.2f}, p={p:.3f}</small></td>')
        rows.append(f"<tr><th>{comp}</th>{''.join(cells)}</tr>")
    loo_table = f"<table><tr><th>component</th>{head}</tr>{''.join(rows)}</table>"
    # panel 3: robustness
    rob = ["<table><tr><th>marginal cell</th><th>3-seed</th><th>5-seed</th><th>outcome</th><th>meaning</th></tr>"]
    for r in data["robustness"]:
        rob.append(f'<tr><td style="text-align:left">{r["cell"]}</td><td>{r["three"]}</td>'
                   f'<td>{r["five"]}</td><td class="{r["outcome"]}">{r["outcome"]}</td>'
                   f'<td style="text-align:left">{r["note"]}</td></tr>')
    rob.append("</table>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Graph &amp; News contribution — VN30 volatility</title><style>{_CSS}</style></head><body>
<h1>Do the graph and news branches contribute? — VN30 volatility (QLIKE, 3-seed unless noted)</h1>
<p class="note">Reading: the graph (GAT over the vol&rarr;PK edge) and news (gated PhoBERT) branches
were added to beat HAR. Below, near-zero bars and grey/red cells show they do not add reliable
out-of-sample value; the model's small edge comes from the node features + LSTM nonlinearity at the
short horizons, not from the graph or news.</p>

<h2>1. QLIKE difference vs HAR (bars near the centre line = no contribution)</h2>
<p class="note"><span class="legend"><span class="better" style="color:#fff">left / green = better than HAR</span>
<span class="worse" style="color:#fff">right / red = worse than HAR</span></span>
Half-track = a QLIKE gap of {scale:.02f}. Most bars hug the centre: FULL, minus_graph, minus_gate are
within ~0.01 of HAR at every horizon; the only visibly-left bars are lstm_only / minus_news at h1&ndash;h5.</p>
{''.join(bars_html)}

<h2>2. Leave-one-out contribution (Diebold-Mariano, FULL vs FULL&minus;component)</h2>
<p class="note">"helps" (green) = removing the component significantly worsens QLIKE. "no help" (red) =
removing it significantly <b>improves</b> QLIKE (the component hurts). "ns" (grey) = no significant
difference. The graph is never green; news/gate are red at the short horizons.</p>
{loo_table}

<h2>3. Robustness of the marginal cells (3-seed &rarr; 5-seed)</h2>
<p class="note">Cells near p=0.05 were re-estimated at 5 seeds. The one apparent "graph beats HAR"
result (graphical-LASSO @ h5) collapses; the LSTM-nonlinearity result strengthens &mdash; confirming
the value is nonlinearity over the node features, not the graph edge.</p>
{''.join(rob)}

<h2>4. Root cause</h2>
<ul class="note">
<li><b>Graph edge:</b> the vol&rarr;PK / correlation / graphical-LASSO edges mostly re-encode a common
market factor that is already a node feature (<code>market_pk</code>); conditional on it, no edge beats
HAR or the graph-removed model out-of-sample (it even significantly hurts at h10).</li>
<li><b>News:</b> removing the news branch significantly lowers QLIKE at h1&ndash;h5 (news does not help
under this architecture/data); its signal is weak/noisy at daily frequency and overlaps price history.</li>
<li><b>Where value is:</b> node features (market factor + volume z-score) read through a non-linear LSTM
give a small, significant edge over a same-feature linear model (HAR-X) at h1 and h5; the linear HAR
family remains the standard to beat at h10&ndash;h22.</li>
</ul>
<p class="note">Source: <code>docs/reports/2026-08-16_1600_gnnhar_p1p2p3_results_report.md</code>,
<code>2026-08-16_2330_p5_glasso_edge_vs_vol2pk_report.md</code>. Regenerate:
<code>python scripts/viz/graph_news_contribution.py</code>.</p>
</body></html>"""


def write_html(data: dict, out: Path) -> None:
    Path(out).write_text(render_html(data), encoding="utf-8")


def main(out: str | None = None) -> None:  # pragma: no cover
    data = contribution_data()
    target = Path(out) if out else _ROOT / "docs" / "reports" / "graph_news_contribution.html"
    write_html(data, target)
    print(f"wrote {target}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1] if len(sys.argv) > 1 else None)
