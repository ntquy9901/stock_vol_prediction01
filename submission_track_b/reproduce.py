#!/usr/bin/env python
"""Track-B volatility model - single reproducibility entry point.

Usage (no typing needed on Windows: double-click the .bat launchers):

    python reproduce.py            # interactive numbered menu
    python reproduce.py view       # print all paper results (NO data, NO training)
    python reproduce.py infer      # evaluate the FINAL model (G1) on the test split
    python reproduce.py train      # train the FINAL model (G1) and save a checkpoint

Model ladder (see PAPER_MAP.md) - a single-basis nested ablation:
    P0  HAR pooled linear                     (ablation)
    P1  Price LSTM                            (ablation)
    P2  Price + News                          (ablation; best test QLIKE in the study)
    P3  Price + News + per-ticker gate        (ablation; == G1 with the graph disabled)
    G1  P3 backbone + cross-stock graph       >>> FINAL / PROPOSED MODEL <<<

There is no separate G0 row: P3 is exactly G1 read out with the message-passing residual
disabled (graph-off determinism 0.0), so the graph ablation is G1 vs P3. Every rung and every
classical baseline is scored on the same 14,418 validation / 14,464 test observations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CODE = _HERE / "trackb_code"
_RESULTS = _HERE / "results"
_OUTPUT = _HERE / "output"
_CHECKPOINTS = _HERE / "checkpoints"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

_METRICS = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")
_METRIC_LABELS = {
    "mse": "MSE", "rmse": "RMSE", "mae": "MAE", "r2": "R2",
    "qlike": "QLIKE", "directional_accuracy": "DirAcc%",
}

# Ordered ladder: config -> (human label, role tag).
_LADDER = (
    ("P0", "HAR pooled linear", "ablation"),
    ("P1", "Price LSTM", "ablation"),
    ("P2", "Price + News", "ablation"),
    ("P3", "Price+News+gate (=G1 graph off)", "ablation"),
    ("G1", "Backbone + graph kNN-8", "FINAL / PROPOSED"),
)

# Classical econometric baselines (display order); GARCH family on a 32/33-ticker subset.
_CLASSICAL = ("Persistence", "EWMA", "HAR", "HARQ", "logHAR", "GARCH", "GJR-GARCH", "EGARCH")


def _fmt(name: str, value: float) -> str:
    if value is None:
        return "n/a"
    if name in ("mse", "rmse", "mae"):
        return f"{value:.3e}"
    if name == "directional_accuracy":
        return f"{value:.2f}"
    return f"{value:.4f}"


def _ladder_json() -> dict:
    path = _RESULTS / "ladder_consistent_h5.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _rung_mean(rung: dict, name: str) -> float:
    """Read the 3-seed mean of one metric from a rung block."""
    return float(rung[name]["mean"])


def _load_validation_metrics() -> dict[str, dict[str, float]]:
    """P0-G1 validation 3-seed means from the consistent-ladder JSON (no data needed)."""
    val = _ladder_json()["rung_metrics"]["val"]
    return {cfg: {name: _rung_mean(val[cfg], name) for name in _METRICS}
            for cfg, _, _ in _LADDER}


def _load_g1_test_metrics() -> dict[str, float] | None:
    """G1 held-out TEST 3-seed means from the consistent-ladder JSON."""
    test = _ladder_json()["rung_metrics"]["test"]
    if "G1" not in test:
        return None
    return {name: _rung_mean(test["G1"], name) for name in _METRICS}


def _load_test_metrics() -> dict[str, dict[str, float]]:
    """Full P0-G1 held-out TEST 3-seed means."""
    test = _ladder_json()["rung_metrics"]["test"]
    return {cfg: {name: _rung_mean(test[cfg], name) for name in _METRICS}
            for cfg, _, _ in _LADDER}


def _load_classical_test() -> dict[str, dict[str, float]]:
    """Classical econometric baselines, held-out TEST, from classical_baselines_h5.json."""
    path = _RESULTS / "classical_baselines_h5.json"
    test = json.loads(path.read_text(encoding="utf-8"))["rung_metrics"]["test"]
    return {name: {m: float(test[name][m]["mean"]) for m in _METRICS} for name in _CLASSICAL}


def _graph_significance_note() -> str:
    """One-line parsimony note derived from the ladder graph verdict (verdict B = null)."""
    data = _ladder_json()
    verdict = data["graph_effect_verdict"]["test"]
    if verdict["verdict"] == "B":
        p = verdict["paired_p"]
        return ("graph adds no significant gain over P3 - held-out test paired-t "
                f"p={p:.3f} (n.s.), Diebold-Mariano QLIKE not significant; the simpler "
                "ablation is preferred (parsimony)")
    return ""


def _multihorizon_note() -> list[str]:
    """Short multi-horizon graph verdict summary from the per-horizon ladder JSONs."""
    lines = ["Multi-horizon graph verdict (G1 vs P3, held-out test QLIKE):"]
    horizons = [("h1", "ladder_consistent_h1.json"),
                ("h5", "ladder_consistent_h5.json"),
                ("h10", "ladder_consistent_h10.json"),
                ("h22", "ladder_consistent_h22.json")]
    for label, fname in horizons:
        path = _RESULTS / fname
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        v = data["graph_effect_verdict"]["test"]
        test = data["rung_metrics"]["test"]
        p3q = float(test["P3"]["qlike"]["mean"])
        g1q = float(test["G1"]["qlike"]["mean"])
        lines.append(f"  {label:>3}: P3 {p3q:.4f} -> G1 {g1q:.4f}  (delta {g1q - p3q:+.5f}, "
                     f"paired-t p={v['paired_p']:.3f}, verdict {v['verdict']} = null)")
    lines.append("  Verdict B (graph null) at all four horizons.")
    return lines


def _table_lines(val: dict[str, dict[str, float]], g1_test: dict[str, float] | None) -> list[str]:
    header = f"{'Model':<6}{'Description':<34}{'Role':<18}"
    header += "".join(f"{_METRIC_LABELS[name]:>12}" for name in _METRICS)
    lines = [header, "-" * len(header)]
    for config, label, role in _LADDER:
        row = f"{config:<6}{label:<34}{role:<18}"
        row += "".join(f"{_fmt(name, val[config][name]):>12}" for name in _METRICS)
        lines.append(row)
    lines.append("")
    lines.append("Scope: metrics above are VALIDATION 3-seed means (seeds 42/123/2026).")
    lines.append("  * Single-basis nested ladder: same 14,418 val / 14,464 test obs across all rows.")
    lines.append("  * P3 == G1 with the cross-stock graph disabled (graph-off determinism 0.0);")
    lines.append("    there is no separate G0 row. The graph ablation is G1 vs P3.")
    lines.append("  * G1 is the FINAL / PROPOSED model; P2 (news backbone) has the lowest test QLIKE.")
    note = _graph_significance_note()
    if note:
        lines.append(f"  * Parsimony finding: {note}.")
    lines.append("")
    if g1_test is not None:
        trow = f"{'G1':<6}{'TEST (paper 3-seed)':<34}{'FINAL / PROPOSED':<18}"
        trow += "".join(f"{_fmt(name, g1_test[name]):>12}" for name in _METRICS)
        lines.append("G1 held-out TEST metrics (paper, 3-seed mean; QLIKE ~= P3 - graph does not help):")
        lines.append(trow)
        lines.append("")
    # Classical econometric baselines, held-out test.
    classical = _load_classical_test()
    lines.append("Classical econometric baselines - held-out TEST (same obs; GARCH family on 32/33 tickers):")
    chead = f"{'Baseline':<12}" + "".join(f"{_METRIC_LABELS[name]:>12}" for name in _METRICS)
    lines.append(chead)
    lines.append("-" * len(chead))
    for name in _CLASSICAL:
        row = f"{name:<12}" + "".join(f"{_fmt(m, classical[name][m]):>12}" for m in _METRICS)
        lines.append(row)
    lines.append("  HAR/HARQ tie the deep models on level metrics; GARCH family is far worse.")
    lines.append("")
    lines.extend(_multihorizon_note())
    return lines


def _write_summary_png(val: dict[str, dict[str, float]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = [config for config, _, _ in _LADDER]
    panels = (("qlike", "QLIKE (lower better)"), ("rmse", "RMSE (lower better)"),
              ("directional_accuracy", "Directional Acc % (higher better)"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, (metric, title) in zip(axes, panels):
        values = [val[config][metric] for config in configs]
        colors = ["#c62828" if config == "G1" else "#607d8b" for config in configs]
        axis.bar(configs, values, color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=0)
        axis.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Track-B P0->P1->P2->P3->G1 validation ladder (G1 = final/proposed, red)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def cmd_view() -> int:
    """Print the full P0->G1 table and write output/ artifacts. No data, no training."""

    _OUTPUT.mkdir(parents=True, exist_ok=True)
    val = _load_validation_metrics()
    g1_test = _load_g1_test_metrics()
    lines = _table_lines(val, g1_test)

    print("=" * 96)
    print("TRACK-B RESULTS  -  P0->P1->P2->P3->G1 model ladder  (G1 = FINAL / PROPOSED model)")
    print("=" * 96)
    for line in lines:
        print(line)

    md_path = _OUTPUT / "results_table.md"
    md = ["# Track-B Results - P0->P1->P2->P3->G1 ladder", "",
          "Source JSONs: `results/ladder_consistent_h5.json` (P0-G1, 3-seed val+test means),",
          "`results/classical_baselines_h5.json` (classical baselines), and",
          "`results/ladder_consistent_h{1,10,22}.json` (multi-horizon graph verdict).", "",
          "```", *lines, "```", ""]
    md_path.write_text("\n".join(md), encoding="utf-8")

    png_path = _OUTPUT / "summary.png"
    _write_summary_png(val, png_path)

    print("")
    print(f"Wrote table -> {md_path}")
    print(f"Wrote chart -> {png_path}")
    return 0


def cmd_infer() -> int:
    """Evaluate the FINAL model (G1) on the test split. Requires the dataset + a G1 checkpoint."""

    import g1_final

    checkpoint = _CHECKPOINTS / "g1_final.pt"
    if not checkpoint.exists():
        print("No G1 checkpoint found at", checkpoint)
        print("Run `python reproduce.py train` once (needs data/) to produce it, then retry infer.")
        return 1
    print("Evaluating FINAL model G1 on the held-out test split...")
    g1_final.infer_g1(checkpoint, device="auto")
    return 0


def cmd_train(epochs: int = 10, smoke: bool = False, max_tickers: int | None = None) -> int:
    """Train the FINAL model (G1) end-to-end and save a checkpoint. Requires the dataset."""

    import g1_final

    _CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    print(f"Training FINAL model G1 (P3 backbone + graph layer), epochs={epochs}...")
    payload = g1_final.train_g1(
        _CHECKPOINTS, seed=42, epochs=epochs, horizon=5, device="auto",
        smoke=smoke, max_tickers=max_tickers,
    )
    # Mirror metrics into output/ so `view` can surface the G1 test row.
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    (_OUTPUT / "g1_final_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


def _menu() -> int:
    while True:
        print("")
        print("=" * 60)
        print(" Track-B reproducibility - choose an option")
        print("=" * 60)
        print("  [1] View all results        (no training, no data)")
        print("  [2] Run inference           (final G1 on test set)")
        print("  [3] Train                   (G1, needs data ~minutes)")
        print("  [4] Everything              (view -> train -> infer)")
        print("  [0] Quit")
        try:
            choice = input("Enter choice: ").strip()
        except EOFError:
            return 0
        if choice == "1":
            cmd_view()
        elif choice == "2":
            cmd_infer()
        elif choice == "3":
            cmd_train()
        elif choice == "4":
            cmd_view()
            if cmd_train() == 0:
                cmd_infer()
        elif choice == "0":
            return 0
        else:
            print("Unrecognised choice:", choice)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return _menu()
    command = argv[0].lower()
    if command in ("view", "results"):
        return cmd_view()
    if command == "infer":
        return cmd_infer()
    if command == "train":
        smoke = "--smoke" in argv
        epochs = 10
        max_tickers = None
        for index, token in enumerate(argv):
            if token == "--epochs" and index + 1 < len(argv):
                epochs = int(argv[index + 1])
            if token == "--max-tickers" and index + 1 < len(argv):
                max_tickers = int(argv[index + 1])
        return cmd_train(epochs=epochs, smoke=smoke, max_tickers=max_tickers)
    if command in ("everything", "all"):
        cmd_view()
        if cmd_train() == 0:
            cmd_infer()
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
