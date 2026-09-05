"""Build LaTeX metric tables + fit/DM summaries for the new VolGA paper from edge_hmatched result JSONs.

Reads ``results/edge_hmatched/edgehm_<market>_h<h>.json`` and emits, per market, a booktabs table of each
metric (rows = models HAR/HAR-X/LSTM/VolGA, columns = horizons, best per column bold), a Diebold-Mariano
p-value summary, and the per-model over/under-fit verdict. Numbers come only from the JSONs (no hand-entry).

Run: .venv_gpu_encode/Scripts/python.exe scripts/paper/build_edgehm_tables.py [--markets vn100 vn30 sp500_clean]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "edge_hmatched"
OUT = REPO / "docs" / "paper" / "tables"
MODELS = ("HAR", "HAR-X", "LSTM", "VolGA")
METRICS = ("mse", "rmse", "mae", "qlike", "r2")
LOWER_BETTER = {"mse": True, "rmse": True, "mae": True, "qlike": True, "r2": False}


def load_results(market: str, horizons, results_dir: Path = RESULTS) -> dict:
    """Return {horizon: result_dict} for a market, skipping horizons whose JSON is absent."""
    out = {}
    for h in horizons:
        p = results_dir / f"edgehm_{market}_h{h}.json"
        if p.exists():
            out[h] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _fmt(v, decimals=4):
    return "--" if v is None else f"{v:.{decimals}f}"


def _best_value(by_h, h, metric, models):
    """The best (min or max) metric value across models present at horizon h, or None."""
    vals = [by_h[h]["metrics"][m][metric] for m in models if m in by_h[h].get("metrics", {})]
    if not vals:
        return None
    return (min if LOWER_BETTER[metric] else max)(vals)


def latex_metric_table(market: str, by_h: dict, horizons, models=MODELS, metric="qlike") -> str:
    """A booktabs table for one metric: rows = models, columns = horizons; best per column in bold."""
    hs = [h for h in horizons if h in by_h]
    header = " & ".join(["Model"] + [f"$h{h}$" for h in hs]) + " \\\\"
    lines = ["\\begin{tabular}{l" + "r" * len(hs) + "}", "\\toprule", header, "\\midrule"]
    for m in models:
        cells = [m]
        for h in hs:
            mt = by_h[h].get("metrics", {}).get(m)
            if mt is None:
                cells.append("--"); continue
            v = mt[metric]; best = _best_value(by_h, h, metric, models)
            s = _fmt(v)
            cells.append(f"\\textbf{{{s}}}" if best is not None and abs(v - best) < 1e-12 else s)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    cap = f"% {market}: {metric.upper()} by horizon (best per column bold). Source: edgehm_{market}_h*.json"
    return cap + "\n" + "\n".join(lines) + "\n"


def dm_summary(market: str, by_h: dict, horizons) -> str:
    """One line per horizon of the DM comparisons + p-values (favored model in parentheses)."""
    lines = [f"% {market} Diebold-Mariano (date-clustered) QLIKE:"]
    for h in horizons:
        if h not in by_h:
            continue
        dm = by_h[h].get("dm_date_clustered", {})
        parts = [f"{name}: p={dm[name]['qlike']['p_value']:.3f} ({dm[name]['qlike']['favors']})" for name in dm]
        lines.append(f"% h{h}: " + "; ".join(parts))
    return "\n".join(lines) + "\n"


def fit_summary(market: str, by_h: dict, horizons, models=MODELS) -> str:
    """Per-horizon over/under-fit verdict for each learned model (from fit_diagnostics)."""
    lines = [f"% {market} fit verdicts (train->val->test):"]
    for h in horizons:
        if h not in by_h:
            continue
        fd = by_h[h].get("fit_diagnostics", {})
        parts = [f"{m}={fd[m]['status']}" for m in models if m in fd]
        lines.append(f"% h{h}: " + ", ".join(parts))
    return "\n".join(lines) + "\n"


def main(argv=None):  # pragma: no cover - entry driver (file I/O)
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="*", default=["sp500_clean", "vn100", "vn30"])
    ap.add_argument("--horizons", nargs="*", type=int, default=[1, 5, 10, 22])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    found = sorted(Path(p).name for p in glob.glob(str(RESULTS / "edgehm_*.json")))
    print(f"[tables] result JSONs present: {found or 'none'}")
    for market in a.markets:
        by_h = load_results(market, a.horizons)
        if not by_h:
            print(f"[tables] {market}: no results yet, skip")
            continue
        blocks = [dm_summary(market, by_h, a.horizons), fit_summary(market, by_h, a.horizons)]
        blocks += [latex_metric_table(market, by_h, a.horizons, metric=mt) for mt in METRICS]
        (OUT / f"{market}_tables.tex").write_text("\n".join(blocks), encoding="utf-8")
        print(f"[tables] wrote {OUT / (market + '_tables.tex')} ({len(by_h)} horizons)")


if __name__ == "__main__":  # pragma: no cover
    main()
