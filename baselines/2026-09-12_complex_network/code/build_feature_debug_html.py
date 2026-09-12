"""Visual debug report: WHY the topology ("graph") features do not help, and which features to keep/drop.

Consumes the Experiment-B head-to-head (``complex_network_<market>.json``), the Experiment-A index result
(``complex_network_index_<market>.json``) and the causal feature screen (``complex_network_<market>_screen.json``,
:mod:`feature_screen`), and renders a standalone HTML with embedded charts:

1. Experiment-B per-horizon QLIKE (GBM vs GBM+topo) + train-vs-test (the over-fit signature of adding
   uninformative features).
2. Causal per-fold feature screen -- Mutual Information, Pearson/Spearman and VIF per feature, grouped by
   HAR / own-history / graph -- showing the topology metrics carry ~0 information about the target.
3. A keep/drop recommendation table built straight from the screen verdicts.
4. Experiment-A OOS R^2 of the topology metrics against the future index volatility.

Run: ``python build_feature_debug_html.py [hose|sp500]``.
Output: docs/reports/<date>_complex_network_<market>_feature_debug.html
"""
import base64
import io
import json
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap

RES = REPO / "results" / "gamma_gbm"
GROUP_COLOR = {"har": "#2b7bba", "own": "#5aa9e6", "graph": "#d98c1f"}


def _load(market):
    """Load the three result JSONs for ``market``; screen/index may be ``None`` if not yet produced."""
    hh = json.loads((RES / f"complex_network_{market}.json").read_text())
    scr_p = RES / f"complex_network_{market}_screen.json"
    idx_p = RES / f"complex_network_index_{market}.json"
    scr = json.loads(scr_p.read_text()) if scr_p.exists() else None
    idx = json.loads(idx_p.read_text()) if idx_p.exists() else None
    return hh, scr, idx


def keep_drop_summary(screen):
    """Collapse the per-horizon screen into one verdict per feature (worst-case across horizons).

    A feature is dropped only if it reads "drop (no signal)" at EVERY horizon; flagged redundant if it is
    redundant at any horizon (and never kept elsewhere); else kept. Returns a list of
    ``(feature, group, verdicts_by_h, final)`` sorted graph-last then by final verdict.
    """
    feats = list(next(iter(screen["horizons"].values()))["features"])
    rows = []
    for f in feats:
        vs = {h: screen["horizons"][h]["features"][f]["verdict"] for h in screen["horizons"]}
        grp = screen["horizons"][next(iter(screen["horizons"]))]["features"][f]["group"]
        if all("drop" in v for v in vs.values()):
            final = "DROP"
        elif any("review" in v for v in vs.values()) and not any(v == "keep" for v in vs.values()):
            final = "REVIEW"
        else:
            final = "KEEP"
        rows.append((f, grp, vs, final))
    order = {"KEEP": 0, "REVIEW": 1, "DROP": 2}
    return sorted(rows, key=lambda r: (r[1] == "graph", order[r[3]], r[0]))


def _png(fig):  # pragma: no cover - matplotlib rendering
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _chart_headline(hh):  # pragma: no cover - matplotlib rendering
    import matplotlib.pyplot as plt
    import numpy as np
    hs = list(hh); x = np.arange(len(hs)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].bar(x - w / 2, [hh[h]["GBM"] for h in hs], w, label="GBM (own)", color="#2b7bba")
    ax[0].bar(x + w / 2, [hh[h]["GBM+topo"] for h in hs], w, label="GBM+topo", color="#d98c1f")
    ax[0].set_xticks(x); ax[0].set_xticklabels(hs); ax[0].set_ylabel("test QLIKE (lower better)")
    ax[0].set_title("Test QLIKE: own-history vs +topology"); ax[0].legend()
    for i, h in enumerate(hs):
        ax[0].text(i, max(hh[h]["GBM"], hh[h]["GBM+topo"]), f"DM p={hh[h]['dm_p']:.2g}",
                   ha="center", va="bottom", fontsize=8)
    for m, c in [("GBM", "#2b7bba"), ("GBM+topo", "#d98c1f")]:
        ax[1].plot(hs, [hh[h]["train_metrics"][m] for h in hs], "--o", color=c, label=f"{m} train")
        ax[1].plot(hs, [hh[h]["test_metrics"][m] for h in hs], "-o", color=c, label=f"{m} test")
    ax[1].set_title("Train vs test QLIKE (over-fit signature)"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    return _png(fig)


def _chart_screen(screen, h):  # pragma: no cover - matplotlib rendering
    import matplotlib.pyplot as plt
    import numpy as np
    feats = screen["horizons"][h]["features"]
    order = sorted(feats, key=lambda f: feats[f]["mi"])
    colors = [GROUP_COLOR[feats[f]["group"]] for f in order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    ax[0].barh(y, [feats[f]["mi"] for f in order], color=colors)
    ax[0].axvline(screen["thresholds"]["mi_lo"], color="red", ls=":", lw=1)
    ax[0].set_yticks(y); ax[0].set_yticklabels(order); ax[0].set_title(f"Mutual information ({h})")
    # Pearson/Spearman coloured by STATISTIC (not group) so they don't clash with the group legend/MI+VIF bars
    bh = 0.4
    ax[1].barh(y - bh / 2, [feats[f]["pearson"] for f in order], bh, color="#555555", label="Pearson")
    ax[1].barh(y + bh / 2, [feats[f]["spearman"] for f in order], bh, color="#7b3f9e", label="Spearman")
    ax[1].axvline(0, color="#888", lw=0.8); ax[1].set_title("Pearson / Spearman vs y"); ax[1].legend(fontsize=8)
    ax[2].barh(y, [min(feats[f]["vif"], 50) for f in order], color=colors)
    ax[2].axvline(screen["thresholds"]["vif_hi"], color="red", ls=":", lw=1)
    ax[2].set_title("VIF (clipped at 50)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in GROUP_COLOR]
    fig.legend(handles, [f"{g} (MI & VIF bar colour)" for g in GROUP_COLOR],
               loc="upper center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _png(fig)


_CSS = """body{font:14px/1.5 system-ui,Segoe UI,Arial;margin:24px;color:#1a1a1a;max-width:1050px}
h1{font-size:23px}h2{font-size:17px;margin-top:30px;border-bottom:1px solid #ddd;padding-bottom:4px}
table{border-collapse:collapse;margin:10px 0}th,td{border:1px solid #ccc;padding:5px 10px;text-align:right}
th{background:#f2f2f2}td.l,th.l{text-align:left}.KEEP{color:#0a7d2c;font-weight:600}
.DROP{color:#c0271a;font-weight:600}.REVIEW{color:#b8860b;font-weight:600}
img{max-width:100%;margin:8px 0}.muted{color:#666}.har{background:#eaf2fb}.graph{background:#fdf3e3}
.card{background:#fafafa;border:1px solid #e5e5e5;border-radius:6px;padding:10px 16px;margin:10px 0}"""


def _keepdrop_table(rows, hs):  # pragma: no cover - HTML formatting
    head = "".join(f"<th>{h}</th>" for h in hs)
    out = [f"<tr><th class=l>feature</th><th class=l>group</th>{head}<th>decision</th></tr>"]
    for f, grp, vs, final in rows:
        cells = "".join(f"<td>{vs[h].replace(' (no signal)','').replace(' (redundant)','')}</td>" for h in hs)
        out.append(f"<tr class={grp}><td class=l>{f}</td><td class=l>{grp}</td>{cells}"
                   f"<td class={final}>{final}</td></tr>")
    return "<table>" + "".join(out) + "</table>"


def _headline_table(hh):  # pragma: no cover - HTML formatting
    out = ["<tr><th class=l>horizon</th><th>GBM</th><th>GBM+topo</th><th>gain %</th><th>DM p</th></tr>"]
    for h, r in hh.items():
        cls = "DROP" if r["gain_pct"] < 0 else "KEEP"
        out.append(f"<tr><td class=l>{h}</td><td>{r['GBM']:.4f}</td><td>{r['GBM+topo']:.4f}</td>"
                   f"<td class={cls}>{r['gain_pct']:+.2f}</td><td>{r['dm_p']:.2g}</td></tr>")
    return "<table>" + "".join(out) + "</table>"


INDEX_NAME = {"hose": "VNINDEX", "sp500": "^GSPC"}
_TGT_LABEL = {"idx_vol": "index volatility (std log-ret) — HEADLINE",
              "idx_ret": "index mean log-return", "idx_lnvol": "index mean log-volume"}


def _index_table(idx):  # pragma: no cover - HTML formatting
    out = ["<tr><th class=l>model</th><th class=l>target</th><th>R2 oos</th><th>RMSE oos</th>"
           "<th>train R2</th><th class=l>top metric</th></tr>"]
    for mdl in ("LinearRegression", "RandomForest"):
        for tgt, v in idx["headline"][mdl].items():
            cls = "KEEP" if v["r2_oos"] > 0 else "DROP"
            fi = v.get("feature_importance", {})
            top = max(fi, key=lambda k: abs(fi[k])) if fi else "-"
            out.append(f"<tr><td class=l>{mdl}</td><td class=l>{_TGT_LABEL.get(tgt, tgt)}</td>"
                       f"<td class={cls}>{v['r2_oos']:+.4f}</td><td>{v['rmse_oos']:.5f}</td>"
                       f"<td>{v['train_r2']:.4f}</td><td class=l>{top}</td></tr>")
    return "<table>" + "".join(out) + "</table>"


def _chart_index(idx, market):  # pragma: no cover - matplotlib rendering
    import matplotlib.pyplot as plt
    import numpy as np
    tgts = list(idx["headline"]["RandomForest"]); x = np.arange(len(tgts)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for off, mdl, c in [(-w / 2, "LinearRegression", "#2b7bba"), (w / 2, "RandomForest", "#d98c1f")]:
        ax.bar(x + off, [idx["headline"][mdl][t]["r2_oos"] for t in tgts], w, label=mdl, color=c)
    ax.axhline(0, color="#c0271a", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(tgts); ax.set_ylabel("OOS R² (>0 = predictive)")
    ax.set_title(f"Experiment A: topology metrics → future {INDEX_NAME.get(market, market)} (OOS R²)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _png(fig)


def build_html(market, date):  # pragma: no cover - orchestration + HTML assembly
    hh, scr, idx = _load(market)
    parts = [f"<h1>Feature debug &mdash; complex-network topology ({market.upper()})</h1>",
             f"<p class=muted>Generated {date}. Target = future Parkinson variance (per-stock). "
             "All screen statistics estimated causally, separately per walk-forward training fold.</p>"]
    parts.append("<h2>1. Does the graph help? Per-horizon QLIKE (Experiment B)</h2>")
    parts.append(_headline_table(hh))
    parts.append(f'<img src="{_chart_headline(hh)}" alt="headline QLIKE">')
    parts.append("<p class=muted>Adding the topology metrics lowers the <b>train</b> QLIKE but not the "
                 "<b>test</b> QLIKE &mdash; the classic signature of fitting in-sample noise with "
                 "uninformative features.</p>")
    if scr:
        hs = list(scr["horizons"])
        parts.append("<h2>2. Causal feature screen: Variance &rarr; Pearson/Spearman &rarr; MI &rarr; VIF</h2>")
        parts.append(f"<p class=muted>Folds train on [{scr['train_start']}, fold_start&minus;embargo); "
                     f"MI subsample cap {scr['mi_subsample_cap']:,} rows/fold.</p>")
        for h in hs:
            parts.append(f'<img src="{_chart_screen(scr, h)}" alt="screen {h}">')
        parts.append("<h2>3. Keep / drop recommendation</h2>")
        parts.append(_keepdrop_table(keep_drop_summary(scr), hs))
        parts.append("<p class=muted>DROP = negligible Pearson, Spearman AND mutual information at every "
                     "horizon. REVIEW = informative but VIF-redundant. Per-horizon cells show the raw "
                     "verdict.</p>")
    else:
        parts.append("<h2>2. Causal feature screen</h2><p class=muted>screen JSON not found &mdash; "
                     "run <code>feature_screen.py</code> first.</p>")
    if idx:
        iname = INDEX_NAME.get(market, market)
        nw = idx.get("n_windows", "?")
        L = idx.get("headline", {}).get("L", "L")
        parts.append(f"<h2>4. Experiment A: topology metrics &rarr; FUTURE {iname} volatility "
                     "(OOS R<sup>2</sup>)</h2>")
        parts.append(f"<p class=muted>Faithful paper replication (§2.4): features = the topology metrics of a "
                     f"rolling return+volume network; target = the future {L}-day {iname} volatility "
                     f"(std of log-returns, the HEADLINE target), plus index mean return / log-volume. "
                     f"Causal expanding walk-forward, {nw} monthly windows. OOS R²&le;0 means the topology "
                     f"metrics do not predict {iname} volatility out-of-sample.</p>")
        parts.append(f'<img src="{_chart_index(idx, market)}" alt="Experiment A {iname}">')
        parts.append(_index_table(idx))
    html = f"<!doctype html><html><head><meta charset=utf-8><style>{_CSS}</style></head><body>" \
           + "".join(parts) + "</body></html>"
    outp = REPO / "docs" / "reports" / f"{date}_complex_network_{market}_feature_debug.html"
    outp.write_text(html, encoding="utf-8")
    return outp


def main():  # pragma: no cover - entry driver
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    date = sys.argv[2] if len(sys.argv) > 2 else "2026-09-12"
    outp = build_html(market, date)
    print(f"saved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
