"""Summarise pooled_vn30_h*.json into a markdown report (Arm0 vs Arm1 + paired DM + diff-in-diff)."""
from __future__ import annotations

import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "results" / "pooled_transfer_vn30"
OUT = REPO / "docs" / "reports" / "2026-09-04_pooled_transfer_vn30_report.md"
_DEEP = (("LSTM", "LSTM"), ("VolGA", "LSTM_wGAT_vol2pk"))


def _fp(x):
    return "n/a" if x is None else f"{x:.3f}"


def main():  # pragma: no cover
    files = sorted(glob.glob(str(RES / "pooled_vn30_h*.json")),
                   key=lambda p: int(Path(p).stem.split("_h")[-1]))
    if not files:
        print("no pooled_vn30 results yet")
        return
    L = ["# Pooled/transfer ablation for VN30 — results", "",
         "Single VN100 panel; Arm 0 trains VN30 (31), Arm 1 trains VN100 (102); both score the 31 VN30",
         "stocks on the identical OOS grid. Headline = paired DM Arm1-vs-Arm0 (favours A = pooling helps).",
         "Prior Track B A1 (2026-08-08) found pooling did not help deep beat HAR; this is the clean-data,",
         "walk-forward, cross-universe re-test. Objective report — verdict stated regardless of sign.", "",
         "**Interpretation caveat:** Arm 0 here is VN30 trained on the shared VN100 panel (VN100 fold",
         "calendar, OOS grid and `market_pk` factor), NOT the previously delivered standalone-VN30 run",
         "(which used the VN30-only panel: fewer OOS obs, K=16, a VN30 `market_pk`). The two are on",
         "different OOS grids and a different market factor, so their absolute numbers are NOT directly",
         "comparable. The only valid comparison is Arm 0 vs Arm 1 (both on the VN100 grid, differing only",
         "in the training node set) — that is what the paired DM below measures.", ""]
    for f in files:
        d = json.load(open(f))
        h = d["horizon"]
        m0 = d["arm0"]["metrics"]
        m1 = d["arm1"]["metrics"]
        L.append(f"## Horizon h{h}  ({d['meta']['n_folds']} folds, {d['meta']['vn30_scored']} VN30 nodes, "
                 f"{d['meta']['panel_nodes']} panel nodes, {d['meta'].get('seconds', 0) / 3600:.1f}h)")
        L.append("")
        L.append("| Model | QLIKE Arm0 (VN30) | QLIKE Arm1 (VN100) | Δ (pooled−base) |")
        L.append("|---|---|---|---|")
        for mk in ("HAR", "HAR-X", "LSTM", "LSTM_wGAT_vol2pk"):
            q0, q1 = m0[mk]["qlike"], m1[mk]["qlike"]
            L.append(f"| {mk} | {q0:.4f} | {q1:.4f} | {q1 - q0:+.4f} |")
        L.append("")
        L.append("**Headline — paired DM (Arm1 vs Arm0), favours A = pooling helps the deep model:**")
        L.append("")
        L.append("| Deep model | QLIKE p (favors) | SE p (favors) | AE p (favors) |")
        L.append("|---|---|---|---|")
        for name, key in _DEEP:
            dm = d["paired_dm"][key]
            cells = []
            for basis in ("qlike", "se", "ae"):
                b = dm.get(basis, {})
                cells.append(f"{_fp(b.get('p_value'))} ({b.get('favors', '?')})")
            L.append(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} |")
        L.append("")
        L.append("**Secondary — diff-in-diff gap(deep − HAR):**  ")
        for name, key in _DEEP:
            dd = d["diff_in_diff"][key]
            L.append(f"- {name}: gap Arm0 = {dd['gap_arm0']:+.4f}, Arm1 = {dd['gap_arm1']:+.4f}, "
                     f"Δgap = {dd['delta_gap']:+.4f} (negative Δ = pooling narrows the deep−HAR gap)")
        L.append("")
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT} ({len(files)} horizon(s))")


if __name__ == "__main__":  # pragma: no cover
    main()
