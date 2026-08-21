"""Assemble reports/experiment_results.md from all results/har_anchored/*/result.json.

Builds the plan section-22 tables (overall performance, expert contribution) and the section-23 H1-H6
accept/reject decision, per dataset x horizon. Objective wording only (no personal address, no self-praise).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "submission" / "soict_lstm_gat"))
sys.path.insert(0, str(HERE))
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "results" / "har_anchored"
LADDER = ["E0_HAR", "GARCH", "E1", "E2", "E3_blend", "E5", "E6", "E7", "E8", "E9_gate_static", "E10_gate_dyn"]
DATASETS = ["vn30", "vn100", "sp500"]
HORIZONS = [1, 5, 10, 22]


def _load(ds, h):
    p = RES / f"{ds}_h{h}" / "result.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _dmp(r, name):
    """Date-clustered DM p-value vs HAR (the panel-correct test; row-level over-states ~sqrt(N)) — C-1 fix."""
    d = r["dm_vs_har"].get(name, {}).get("date_clustered", {})
    return d.get("p_value") if isinstance(d, dict) and "p_value" in d else None


def paired_dm_from_rows(ds, h, col_a, col_b, floor=1e-8):
    """Date-clustered paired DM of two model predictions from row_predictions.csv (M-1 fix).

    Returns {"p_value","mean_diff","favors"} for QLIKE(col_a) vs QLIKE(col_b); mean_diff<0 => col_a better.
    None if the CSV or a column is missing."""
    p = RES / f"{ds}_h{h}" / "row_predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    ca, cb = f"pred_{col_a}", f"pred_{col_b}"
    if ca not in df or cb not in df:
        return None
    y = df["y_true"].to_numpy(float)
    la = M.per_obs_qlike(y, df[ca].to_numpy(float), floor)
    lb = M.per_obs_qlike(y, df[cb].to_numpy(float), floor)
    dates = df["target_date"].to_numpy()
    try:
        r = ST.date_clustered_dm(la, lb, dates, h)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return {"p_value": r["p_value"], "mean_diff": r["mean_diff"],
            "favors": col_a if r["mean_diff"] < 0 else col_b}


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
    def _rr(r, e):
        v = r["diagnostics"]["residual_r2_oos"].get(e)
        return v if v is not None else -1.0
    h3 = any(_rr(r, e) > 0 for r in results for e in ("E5", "E6", "E7"))
    def graph_wins(r):
        # plan decision rule: a graph residual must beat the no-graph E5 (paired, date-clustered) AND HAR
        for e in ("E6", "E7"):
            pe5 = paired_dm_from_rows(r["dataset"], r["horizon"], e, "E5")
            ph = _dmp(r, e)
            if (pe5 and isinstance(pe5, dict) and pe5.get("p_value") is not None
                    and pe5["p_value"] < 0.05 and pe5["favors"] == e
                    and ph is not None and ph < 0.05 and _delta_qlike(r, e) > 0):
                return True
        return False
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
        out.append("\n**Graph attribution — date-clustered paired DM** (p; model favored). "
                   "Graph value requires E6/E7 to beat the no-graph E5, not only HAR.")
        out.append("| Horizon | E6 vs E5 | E7 vs E5 | E6 vs HAR (dc) | E3 vs HAR (dc) |")
        out.append("|---|---|---|---:|---:|")
        for r in results:
            h = r["horizon"]
            def _pd(a, b):
                d = paired_dm_from_rows(ds, h, a, b)
                return "n/a" if not d or "p_value" not in d else f"{d['p_value']:.4f} ({d['favors']})"
            out.append(f"| {h} | {_pd('E6','E5')} | {_pd('E7','E5')} | {_fmt(_dmp(r,'E6'),4)} | "
                       f"{_fmt(_dmp(r,'E3_blend'),4)} |")
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
