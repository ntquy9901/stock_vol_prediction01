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


_MKT_ORDER = ("hnx", "vn100", "vn30", "sp500")            # HNX first in every estimator table
_MKT_LABEL = {"hnx": "HNX", "vn100": "VN100", "vn30": "VN30", "sp500": r"S\&P 500"}


def render_est_allmarkets(table, panels=_MKT_ORDER, models=EST_MODELS,
                          metrics=METRICS_NO_R2, horizons=HORIZONS) -> str:
    """One estimator, all four metrics, every market (HNX first). Markets are row-blocks with a bold spanning
    label; within a market each horizon lists HAR-X and VolGA, lowest-per-metric bolded. Scales are comparable
    within a single estimator, so all four metrics are shown."""
    ncol = 2 + len(metrics)
    lines = [r"\begin{tabular}{ll" + "c" * len(metrics) + "}", r"\toprule",
             "$h$ & Model & " + " & ".join(_HDR[m] for m in metrics) + r" \\"]
    for panel in panels:
        by = {(r["horizon"], r["model"]): r["cells"] for r in table["rows"] if r["panel"] == panel}
        block = []
        for h in horizons:
            present = [(k, d) for k, d in models if (h, k) in by]
            if not present:
                continue
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
                block.append(f"{h} & {disp} & " + " & ".join(out) + r" \\")
        if not block:
            continue
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{" + str(ncol) + r"}{l}{\textbf{" + _MKT_LABEL[panel] + r"}} \\")
        lines.extend(block)
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
    (out / "tab_dm_voltg.tex").write_text(
        render_dm_voltg(repo / "results" / "masked_rich_floor1e2"), encoding="utf-8")
    written.append("tab_dm_voltg")
    # estimator tables: one per estimator, all markets (HNX first), all four metrics, HAR-X vs VolGA
    yz = repo / "results" / "masked_rich_yz"
    est_tables = [("parkinson", floor)]
    for est in ("yang_zhang", "rogers_satchell"):
        root = yz / est
        if root.exists():
            t = build_tables(root)
            if t["rows"]:
                est_tables.append((est, t))
    for name, tbl in est_tables:
        (out / f"tab_est_{name}.tex").write_text(render_est_allmarkets(tbl), encoding="utf-8")
        written.append(f"tab_est_{name}")
    print("wrote:", ", ".join(written))


if __name__ == "__main__":  # pragma: no cover
    main()
