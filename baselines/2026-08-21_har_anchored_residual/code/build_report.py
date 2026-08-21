"""Assemble reports/experiment_results.md from all results/har_anchored/*/result.json.

Builds the plan section-22 tables (overall performance, expert contribution) and the section-23 H1-H6
accept/reject decision, per dataset x horizon. Objective wording only (no personal address, no self-praise).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "results" / "har_anchored"
LADDER = ["E0_HAR", "GARCH", "E1", "E2", "E3_blend", "E5", "E6", "E7", "E8", "E9_gate_static", "E10_gate_dyn"]
DATASETS = ["vn30", "vn100", "sp500"]
HORIZONS = [1, 5, 10, 22]


def _load(ds, h):
    p = RES / f"{ds}_h{h}" / "result.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _dmp(r, name):
    d = r["dm_vs_har"].get(name, {}).get("dm_qlike", {})
    return d.get("p_value") if isinstance(d, dict) and "p_value" in d else None


def _delta_qlike(r, name):
    h = r["metrics"]["E0_HAR"]["qlike"]; m = r["metrics"][name]["qlike"]
    return 100.0 * (h - m) / h if h else 0.0


def _fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _overall_table(r):
    rows = ["| Model | MAE | RMSE | QLIKE | R2_OOS_vs_HAR | dQLIKE% | DM p-value |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for name in LADDER:
        if name not in r["metrics"]:
            continue
        m = r["metrics"][name]
        dmp = "—" if name == "E0_HAR" else _fmt(_dmp(r, name), 4)
        dq = "—" if name == "E0_HAR" else f"{_delta_qlike(r, name):+.2f}"
        rows.append(f"| {name} | {_fmt(m['mae'],5)} | {_fmt(m['rmse'],5)} | {_fmt(m['qlike'])} | "
                    f"{_fmt(m['rel_r2_vs_har'],4)} | {dq} | {dmp} |")
    return "\n".join(rows)


def _contrib_row(r):
    d = r.get("diagnostics", {})
    rr = d.get("residual_r2_oos", {})
    ec = d.get("error_complementarity_har_vs_E2", {})
    return (f"| {r['horizon']} | {_fmt(r.get('alpha_E3'),3)} | {_fmt(r.get('lambda_E9_static'),3)} | "
            f"{_fmt(ec.get('pearson'),3)} | {_fmt(rr.get('E5'),4)} | {_fmt(rr.get('E6'),4)} | "
            f"{_fmt(rr.get('E7'),4)} |")


def _decisions(results):
    """Aggregate H1-H6 accept/reject across horizons for one dataset (per plan section 23)."""
    lines = []
    def sig_better(r, name):
        p = _dmp(r, name); dq = _delta_qlike(r, name)
        return (p is not None) and p < 0.05 and dq > 0
    h1 = any(sig_better(r, "E3_blend") for r in results)
    alphas = [r.get("alpha_E3") for r in results if r.get("alpha_E3") is not None]
    h2 = (max(alphas) - min(alphas) > 0.1) if len(alphas) > 1 else False
    h3 = any((r["diagnostics"]["residual_r2_oos"].get(e, -1) or -1) > 0
             for r in results for e in ("E5", "E6", "E7"))
    def graph_wins(r):
        q = r["metrics"]; return q.get("E7", {}).get("qlike", 9) < q.get("E5", {}).get("qlike", 9) and \
            sig_better(r, "E7")
    h4 = any(graph_wins(r) for r in results)
    h5 = any(r["metrics"].get("E10_gate_dyn", {}).get("qlike", 9) <
             r["metrics"].get("E9_gate_static", {}).get("qlike", 9) - 1e-6 and sig_better(r, "E10_gate_dyn")
             for r in results)
    def safe_anchor(r):
        q = r["metrics"]
        best_resid = min(q.get(e, {}).get("qlike", 9) for e in ("E5", "E7", "E8"))
        best_full = min(q.get(e, {}).get("qlike", 9) for e in ("E1", "E2"))
        return best_resid <= best_full + 1e-9
    h6 = all(safe_anchor(r) for r in results)
    verdicts = {
        "H1 static combo beats both experts": h1,
        "H2 horizon specialization (alpha differs)": h2,
        "H3 residual learnability (any resid R2_OOS>0)": h3,
        "H4 cross-sectional/graph incremental value": h4,
        "H5 state dependence (dynamic gate > static)": h5,
        "H6 safe anchoring (residual <= full neural)": h6,
    }
    lines.append("| Hypothesis | Verdict |")
    lines.append("|---|---|")
    for k, v in verdicts.items():
        lines.append(f"| {k} | {'ACCEPT' if v else 'REJECT'} |")
    return "\n".join(lines)


def build() -> str:
    out = ["# Experiment Results — HAR-Anchored LSTM–GAT Study (E0-E10)",
           "",
           "Source: `results/har_anchored/<dataset>_h<h>/result.json`. Primary metric QLIKE on Parkinson",
           "variance; snapshot common-date design with target-overlap purge; 5 seeds, per-horizon HAR anchor.",
           "DM = Diebold–Mariano (HLN) vs HAR on per-observation QLIKE; positive dQLIKE% = better than HAR.",
           ""]
    for ds in DATASETS:
        results = [r for h in HORIZONS if (r := _load(ds, h))]
        if not results:
            continue
        out.append(f"## {ds.upper()}")
        for r in results:
            out.append(f"\n### {ds} h{r['horizon']}  (nodes={r['num_nodes']}, n_test={r['n_test']}, "
                       f"MCS={r.get('mcs', {}).get('mcs_set')})")
            out.append(_overall_table(r))
        out.append("\n**Expert contribution**")
        out.append("| Horizon | alpha_HAR (E3) | lambda (E9) | HAR–E2 err corr | E5 resid R2 | E6 resid R2 | E7 resid R2 |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in results:
            out.append(_contrib_row(r))
        out.append(f"\n**Hypothesis decisions ({ds})**")
        out.append(_decisions(results))
        out.append("")
    return "\n".join(out)


def main():
    txt = build()
    p = REPO / "reports" / "experiment_results.md"
    p.parent.mkdir(exist_ok=True)
    p.write_text(txt, encoding="utf-8")
    print(f"wrote {p} ({len(txt)} chars)")


if __name__ == "__main__":
    main()
