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
MAIN_MODELS = (("HAR-X", "HAR-X"), ("GARCH", "GARCH"), ("LSTM_wGAT_vol2pk", "LSTM+GAT"))
GRAPH_ABL_MODELS = (("HAR-X", "HAR-X"), ("LSTM", "LSTM"), ("LSTM_wGAT_vol2pk", "LSTM+GAT"))
# estimator-comparison tables compare only the benchmark and the proposed model (no no-graph LSTM row)
EST_MODELS = (("HAR-X", "HAR-X"), ("LSTM_wGAT_vol2pk", "LSTM+GAT"))
FULL_MODELS = (("HAR-X", "HAR-X"), ("GARCH", "GARCH"), ("LSTM", "LSTM"), ("LSTM_wGAT_vol2pk", "LSTM+GAT"))
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
