"""HNX-primary paper tables: reuses the authoritative ``build_final_tables`` data (learned = per-seed mean,
deterministic = ensemble) and emits paper ``tabular`` bodies with a configurable MODEL subset and METRIC
subset (the submission drops $R^2$, keeping MSE/RMSE/MAE/QLIKE). Numbers are never hand-typed: every cell comes
from ``result.json`` via ``build_tables``.

Emits into docs/paper/generated/:
  * tab_main_hnx.tex           -- MAIN: HNX, {HAR-X, GARCH, LSTM+GAT}
  * tab_abl_graph_hnx.tex      -- graph ablation on HNX: {HAR-X, LSTM, LSTM+GAT} (LSTM-only vs LSTM+GAT)
  * tab_abl_vn100/vn30/sp500.tex -- market ablation: {HAR-X, GARCH, LSTM, LSTM+GAT}
  * tab_abl_yang_zhang.tex / tab_abl_rogers_satchell.tex -- estimator ablation (masked_rich_yz), HNX + panels
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_final_tables import build_tables, _scaled_value, _DEC  # noqa: E402

METRICS_NO_R2 = ("mse", "rmse", "mae", "qlike")
_HDR = {"mse": "MSE", "rmse": "RMSE", "mae": "MAE", "qlike": "QLIKE"}
MAIN_MODELS = (("HAR-X", "HAR-X"), ("GARCH", "GARCH"), ("LSTM_wGAT_vol2pk", "VolGA"))
GRAPH_ABL_MODELS = (("HAR-X", "HAR-X"), ("LSTM", "LSTM"), ("LSTM_wGAT_vol2pk", "VolGA"))
# estimator-comparison tables compare only the benchmark and the proposed model (no no-graph LSTM row)
EST_MODELS = (("HAR-X", "HAR-X"), ("LSTM_wGAT_vol2pk", "VolGA"))
FULL_MODELS = (("HAR-X", "HAR-X"), ("GARCH", "GARCH"), ("LSTM", "LSTM"), ("LSTM_wGAT_vol2pk", "VolGA"))
HORIZONS = (1, 5, 10, 22)


def render_panel(table: dict, panel: str, models, metrics=METRICS_NO_R2, horizons=HORIZONS) -> str:
    """Paper ``tabular`` for one panel with the given model rows and metric columns; best-in-column bolded
    (min, all four are losses), learned QLIKE carries its per-seed std."""
    by = {(r["horizon"], r["model"]): r["cells"] for r in table["rows"] if r["panel"] == panel}
    col = "ll" + "c" * len(metrics)
    head = "$h$ & Model & " + " & ".join(_HDR[m] for m in metrics) + r" \\"
    lines = [r"\begin{tabular}{" + col + "}", r"\toprule", head, r"\midrule"]
    first = True
    for h in horizons:
        present = [(k, d) for k, d in models if (h, k) in by]
        if not present:
            continue
        if not first:
            lines.append(r"\midrule")
        first = False
        best = {}
        for mt in metrics:
            vals = [v for v in (_scaled_value(mt, by[(h, k)].get(mt)) for k, _ in present) if v is not None]
            if vals:
                best[mt] = min(vals)
        for k, disp in present:
            cells = by[(h, k)]
            out = []
            for mt in metrics:
                sv = _scaled_value(mt, cells.get(mt))
                if sv is None:
                    out.append("-"); continue
                txt = f"{sv:.{_DEC[mt]}f}"
                if mt == "qlike" and cells[mt].get("std") is not None:
                    txt += r"\,$\pm$" + f"{cells[mt]['std']:.3f}".lstrip("0")
                if mt in best and abs(sv - best[mt]) < 10 ** (-_DEC[mt] - 1):
                    txt = r"\textbf{" + txt + "}"
                out.append(txt)
            lines.append(f"{h} & {disp} & " + " & ".join(out) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


_DM_PANELS = ("hnx", "vn100", "vn30", "sp500")


def _fmt_p(p: float | None) -> str:
    """DM p-value with a leading zero for clarity: '<0.001' for tiny, else '0.NNN'; '-' if missing."""
    if p is None:
        return "-"
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def render_dm_voltg(results_root: Path, panels=_DM_PANELS, horizons=HORIZONS) -> str:
    """Date-clustered DM table of VolGA vs HAR-X (QLIKE and MAE) per panel/horizon, HNX first. Reads the
    ``wGAT_vol2pk_vs_HARX`` block from each result.json (qlike + ae losses); favoured model in parentheses
    (V=VolGA, H=HAR-X), bold if p<0.05."""
    import json
    lines = [r"\begin{tabular}{ll cc}", r"\toprule",
             r"Panel & $h$ & QLIKE & MAE \\", r"\midrule"]
    first = True
    for panel in panels:
        rows = []
        for h in horizons:
            rp = results_root / f"{panel}_h{h}" / "result.json"
            if not rp.exists():
                continue
            dm = json.loads(rp.read_text()).get("dm_date_clustered", {}).get("wGAT_vol2pk_vs_HARX", {})
            cells = []
            for loss in ("qlike", "ae"):
                c = dm.get(loss, {})
                p, fav = c.get("p_value"), c.get("favors")
                if p is None:
                    cells.append("-"); continue
                tag = f"{_fmt_p(p)} ({'VolGA' if fav == 'A' else 'HAR-X'})"
                cells.append(r"\textbf{" + tag + "}" if p < 0.05 else tag)
            rows.append(f"{h} & " + " & ".join(cells))
        if not rows:
            continue
        if not first:
            lines.append(r"\midrule")
        first = False
        disp = panel.upper() if panel != "sp500" else r"S\&P 500"
        lines.append(f"{disp} & " + rows[0] + r" \\")
        for r in rows[1:]:
            lines.append(f" & {r}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def render_est_qlike(sources, panel, horizons=HORIZONS) -> str:
    """Per-market QLIKE comparison (HAR-X vs VolGA) across estimators. ``sources`` is a list of
    ``(label, table)`` (each table from build_tables of one estimator's results). QLIKE is scale-invariant so
    it is the metric comparable across estimators; the lower QLIKE per row is bolded."""
    lines = [r"\begin{tabular}{ll cc}", r"\toprule",
             r"Estimator & $h$ & HAR-X & VolGA \\", r"\midrule"]
    first = True
    for label, table in sources:
        by = {(r["horizon"], r["model"]): r["cells"] for r in table["rows"] if r["panel"] == panel}
        rows = []
        for h in horizons:
            har = by.get((h, "HAR-X"), {}).get("qlike")
            vga = by.get((h, "LSTM_wGAT_vol2pk"), {}).get("qlike")
            if har is None or vga is None:
                continue
            hv, vv = har["value"], vga["value"]
            ht, vt = f"{hv:.4f}", f"{vv:.4f}"
            if hv <= vv:
                ht = r"\textbf{" + ht + "}"
            else:
                vt = r"\textbf{" + vt + "}"
            rows.append((h, ht, vt))
        if not rows:
            continue
        if not first:
            lines.append(r"\midrule")
        first = False
        lines.append(f"{label} & {rows[0][0]} & {rows[0][1]} & {rows[0][2]}" + r" \\")
        for h, ht, vt in rows[1:]:
            lines.append(f" & {h} & {ht} & {vt}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():  # pragma: no cover - I/O entry driver
    repo = Path(__file__).resolve().parents[2]
    out = repo / "docs" / "paper" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    floor = build_tables(repo / "results" / "masked_rich_floor1e2")
    (out / "tab_main_hnx.tex").write_text(render_panel(floor, "hnx", MAIN_MODELS), encoding="utf-8")
    (out / "tab_abl_graph_hnx.tex").write_text(render_panel(floor, "hnx", GRAPH_ABL_MODELS), encoding="utf-8")
    for p in ("vn100", "vn30", "sp500"):
        (out / f"tab_abl_{p}.tex").write_text(render_panel(floor, p, FULL_MODELS), encoding="utf-8")
    written = ["tab_main_hnx", "tab_abl_graph_hnx", "tab_abl_vn100", "tab_abl_vn30", "tab_abl_sp500"]
    # Parkinson reference (all four horizons, from the delivered floor1e2 tree) so the estimator tables are
    # directly comparable in the same HAR-X vs LSTM+GAT format.
    (out / "tab_abl_parkinson.tex").write_text(render_panel(floor, "hnx", EST_MODELS), encoding="utf-8")
    written.append("tab_abl_parkinson")
    (out / "tab_dm_voltg.tex").write_text(
        render_dm_voltg(repo / "results" / "masked_rich_floor1e2"), encoding="utf-8")
    written.append("tab_dm_voltg")
    # per-market QLIKE across estimators (which model/estimator is best), HAR-X vs VolGA
    yz = repo / "results" / "masked_rich_yz"
    est_sources = [("Parkinson", floor)]
    if (yz / "yang_zhang").exists():
        est_sources.append(("Yang--Zhang", build_tables(yz / "yang_zhang")))
    if (yz / "rogers_satchell").exists():
        est_sources.append(("Rogers--Satchell", build_tables(yz / "rogers_satchell")))
    for p in ("vn100", "vn30", "sp500"):
        (out / f"tab_est_qlike_{p}.tex").write_text(render_est_qlike(est_sources, p), encoding="utf-8")
        written.append(f"tab_est_qlike_{p}")
    for est in ("yang_zhang", "rogers_satchell"):
        root = repo / "results" / "masked_rich_yz" / est
        if not root.exists():
            continue
        t = build_tables(root)
        if t["rows"]:
            (out / f"tab_abl_{est}.tex").write_text(render_panel(t, "hnx", EST_MODELS), encoding="utf-8")
            written.append(f"tab_abl_{est}")
    print("wrote:", ", ".join(written))


if __name__ == "__main__":  # pragma: no cover
    main()
