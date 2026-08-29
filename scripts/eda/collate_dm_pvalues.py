"""Collate Diebold-Mariano (date-clustered) QLIKE p-values across every completed
volatility-proxy robustness cell into one paper-ready table.

Reads ``results/masked_rich_yz/<estimator>/<panel>_h<h>/result.json`` (each cell
already stores ``dm_date_clustered`` with per-comparison ``{qlike,se,ae}`` blocks),
and emits a Markdown table + CSV of the QLIKE levels and the DM p-values for the
comparisons against the HAR-X baseline (LSTM vs HAR-X, LSTM+GAT vs HAR-X) plus the
graph's marginal test (LSTM+GAT vs LSTM). p-values are formatted without scientific
notation (``<0.001``) for direct paper use.

Re-run after the overnight robustness suite finishes to regenerate the authoritative
table. Read-only w.r.t. results (never writes into the results tree).

Usage: python scripts/eda/collate_dm_pvalues.py [--results-dir ...] [--out-md ...] [--out-csv ...]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

# comparison key in dm_date_clustered -> (model_A, model_B) human labels
_COMPARISONS = {
    "LSTM_vs_HARX": ("LSTM", "HAR-X"),
    "wGAT_vol2pk_vs_HARX": ("LSTM+GAT", "HAR-X"),
    "wGAT_vol2pk_vs_LSTM": ("LSTM+GAT", "LSTM"),
}
_ESTIMATOR_ORDER = {"yang_zhang": 0, "rogers_satchell": 1, "garman_klass": 2, "parkinson": 3}
# Only these published estimators belong in the paper table. yz_daily and any other
# per-day proxy are excluded by default (CLAUDE.md no-proxy rule); --include-all overrides.
_PAPER_ESTIMATORS = frozenset(_ESTIMATOR_ORDER)
_PANEL_ORDER = {"vn30": 0, "vn100": 1, "hnx": 2, "hose": 3, "sp500": 4}


def format_p(p: float) -> str:
    """Paper-style p-value: '<0.001' for tiny values (no e-notation), else 3 decimals."""
    if p is None:
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _favored_label(comp: str, favors: str) -> str:
    """Translate DM 'favors' A/B into the winning model label for `comp`."""
    a, b = _COMPARISONS[comp]
    if favors == "A":
        return a
    if favors == "B":
        return b
    return "?"


def summarize_cell(res: dict) -> dict:
    """Extract QLIKE levels + DM-QLIKE p-values for one result.json dict.

    Returns keys: n_test_obs, n_test_dates, qlike_{HAR-X,LSTM,LSTM+GAT}, and for each
    comparison c in _COMPARISONS: p_<c> (float or None) and win_<c> (winning-model label).
    """
    out: dict = {}
    m = res.get("metrics", {})
    ps = res.get("metrics_per_seed", {})
    out["qlike_HAR-X"] = m.get("HAR-X", {}).get("qlike")
    out["qlike_LSTM"] = ps.get("LSTM", {}).get("qlike")
    out["qlike_LSTM+GAT"] = ps.get("LSTM_wGAT_vol2pk", {}).get("qlike")
    out["n_test_obs"] = res.get("n_test_obs")
    out["n_test_dates"] = res.get("n_test_dates")
    dm = res.get("dm_date_clustered", {})
    for comp in _COMPARISONS:
        block = dm.get(comp, {}).get("qlike") if isinstance(dm, dict) else None
        if isinstance(block, dict):
            out[f"p_{comp}"] = block.get("p_value")
            out[f"win_{comp}"] = _favored_label(comp, block.get("favors"))
        else:
            out[f"p_{comp}"] = None
            out[f"win_{comp}"] = None
    return out


def _cell_sort_key(row: dict):
    return (
        _ESTIMATOR_ORDER.get(row["estimator"], 9),
        row["horizon"],
        _PANEL_ORDER.get(row["panel"], 9),
    )


def collect_rows(results_dir: Path, include_all: bool = False) -> list[dict]:
    """Walk every result.json under results_dir and return sorted summary rows.

    By default only the published estimators in _PAPER_ESTIMATORS are returned
    (per-day proxies such as yz_daily are excluded); pass include_all=True to keep
    every estimator directory found.
    """
    rows = []
    for rp in glob.glob(str(results_dir / "*" / "*" / "result.json")):
        p = Path(rp)
        estimator = p.parent.parent.name
        if not include_all and estimator not in _PAPER_ESTIMATORS:
            continue
        panel, _, h = p.parent.name.rpartition("_h")
        try:
            res = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        row = {"estimator": estimator, "panel": panel, "horizon": int(h) if h.isdigit() else -1}
        row.update(summarize_cell(res))
        rows.append(row)
    rows.sort(key=_cell_sort_key)
    return rows


def _fmt_q(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"


def render_markdown(rows: list[dict]) -> str:
    """Render the collated rows as a Markdown table grouped by estimator."""
    lines = ["# DM (date-clustered) QLIKE p-values — volatility-proxy robustness", ""]
    lines.append("p-value against the HAR-X baseline; winner in parentheses. "
                 "'<0.001' denotes p below 0.001.")
    header = ("| estimator | panel | h | QLIKE HAR-X | QLIKE LSTM | QLIKE LSTM+GAT "
              "| LSTM vs HAR-X | LSTM+GAT vs HAR-X | LSTM+GAT vs LSTM |")
    sep = "|" + "|".join(["---"] * 9) + "|"
    cur = None
    for r in rows:
        if r["estimator"] != cur:
            lines += ["", f"## {r['estimator']}", "", header, sep]
            cur = r["estimator"]

        def cmp(c):
            p = r.get(f"p_{c}")
            w = r.get(f"win_{c}")
            return f"{format_p(p)} ({w})" if p is not None else "n/a"

        lines.append(
            f"| {r['estimator']} | {r['panel']} | {r['horizon']} "
            f"| {_fmt_q(r.get('qlike_HAR-X'))} | {_fmt_q(r.get('qlike_LSTM'))} "
            f"| {_fmt_q(r.get('qlike_LSTM+GAT'))} "
            f"| {cmp('LSTM_vs_HARX')} | {cmp('wGAT_vol2pk_vs_HARX')} "
            f"| {cmp('wGAT_vol2pk_vs_LSTM')} |"
        )
    lines.append("")
    lines.append(f"Total cells: {len(rows)}")
    return "\n".join(lines)


def write_csv(rows: list[dict], out_csv: Path) -> None:
    cols = ["estimator", "panel", "horizon", "n_test_obs", "n_test_dates",
            "qlike_HAR-X", "qlike_LSTM", "qlike_LSTM+GAT"]
    for c in _COMPARISONS:
        cols += [f"p_{c}", f"win_{c}"]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def main():  # pragma: no cover - I/O entry driver
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/masked_rich_yz")
    ap.add_argument("--out-md", default="docs/reports/2026-08-29_dm_pvalues_proxy_robustness.md")
    ap.add_argument("--out-csv", default="docs/reports/2026-08-29_dm_pvalues_proxy_robustness.csv")
    ap.add_argument("--include-all", action="store_true",
                    help="include non-published estimators (e.g. yz_daily proxy)")
    a = ap.parse_args()
    rows = collect_rows(Path(a.results_dir), include_all=a.include_all)
    Path(a.out_md).write_text(render_markdown(rows), encoding="utf-8")
    write_csv(rows, Path(a.out_csv))
    print(f"wrote {a.out_md} + {a.out_csv} ({len(rows)} cells)")


if __name__ == "__main__":  # pragma: no cover
    main()
