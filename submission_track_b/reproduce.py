#!/usr/bin/env python
"""Track-B volatility model - single reproducibility entry point.

Usage (no typing needed on Windows: double-click the .bat launchers):

    python reproduce.py            # interactive numbered menu
    python reproduce.py view       # print all paper results (NO data, NO training)
    python reproduce.py infer      # evaluate the FINAL model (G1) on the test split
    python reproduce.py train      # train the FINAL model (G1) and save a checkpoint

Model ladder (see PAPER_MAP.md):
    P0  HAR pooled linear                 (ablation)
    P1  Price LSTM                        (ablation)
    P2  Price + News                      (ablation)
    P3  Price + News + per-ticker gate    (ablation; graph backbone)
    G0  Backbone, graph message-passing OFF   (ablation)
    G1  Backbone + graph message-passing ON   >>> FINAL / PROPOSED MODEL <<<
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
    ("P3", "Price + News + gate", "ablation"),
    ("G0", "Backbone, graph OFF", "ablation"),
    ("G1", "Backbone+graph kNN-8", "FINAL / PROPOSED"),
)


def _fmt(name: str, value: float) -> str:
    if value is None:
        return "n/a"
    if name in ("mse", "rmse", "mae"):
        return f"{value:.3e}"
    if name == "directional_accuracy":
        return f"{value:.2f}"
    return f"{value:.4f}"


def _load_validation_metrics() -> dict[str, dict[str, float]]:
    """Read P0-G1 validation metrics from the saved screening JSONs (no data needed)."""

    metrics: dict[str, dict[str, float]] = {}

    pooled_path = _RESULTS / "pooled_20ep_aggregate.json"
    pooled = json.loads(pooled_path.read_text(encoding="utf-8"))["aggregated"]
    for config in ("P0", "P1", "P2", "P3"):
        # each metric stored as [mean, std] across 3 seeds; take the mean.
        metrics[config] = {name: float(pooled[config][name][0]) for name in _METRICS}

    graph_path = _RESULTS / "g0g1_graph_validation_comparison.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))["results"]
    for config in ("G0", "G1"):
        vm = graph[config]["validation_metrics"]
        metrics[config] = {name: float(vm[name]) for name in _METRICS}
    return metrics


def _graph_significance_note() -> str:
    """One-line parsimony note from the canonical G0/G1 verdict JSON (empty if absent)."""

    graph_path = _RESULTS / "g0g1_graph_validation_comparison.json"
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    return str(payload.get("significance_note", "")).strip()


def _load_g1_test_metrics() -> dict[str, float] | None:
    """Return the paper's G1 held-out TEST metrics (3-seed mean) from the canonical JSON.

    This is shipped with the bundle, so the reviewer sees the paper's test numbers with no
    data and no training. (A reviewer's own `train`/`infer` run prints its own test metrics
    to the console; this row is the paper value.)
    """

    graph_path = _RESULTS / "g0g1_graph_validation_comparison.json"
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    test = payload.get("held_out_test_3seed_mean", {}).get("G1")
    if test:
        return {name: float(test[name]) for name in _METRICS}
    return None


def _table_lines(val: dict[str, dict[str, float]], g1_test: dict[str, float] | None) -> list[str]:
    header = f"{'Model':<6}{'Description':<24}{'Role':<18}"
    header += "".join(f"{_METRIC_LABELS[name]:>12}" for name in _METRICS)
    lines = [header, "-" * len(header)]
    for config, label, role in _LADDER:
        row = f"{config:<6}{label:<24}{role:<18}"
        row += "".join(f"{_fmt(name, val[config][name]):>12}" for name in _METRICS)
        lines.append(row)
    lines.append("")
    lines.append("Scope: metrics above are VALIDATION means.")
    lines.append("  * P0-P3 = pooled ablation family (pooled validation set, 3-seed mean).")
    lines.append("  * G0-G1 = graph ablation family (masked manifest, screening-P3 backbone,")
    lines.append("    k-NN-8 adjacency for G1, 3-seed mean over the same 14,418 val obs).")
    lines.append("    The P-family and G-family are two separate studies (different evaluation sets).")
    lines.append("  * G1 is the FINAL / PROPOSED model.")
    note = _graph_significance_note()
    if note:
        lines.append(f"  * Parsimony finding: {note}")
    lines.append("")
    if g1_test is not None:
        trow = f"{'G1':<6}{'TEST (paper 3-seed)':<24}{'FINAL / PROPOSED':<18}"
        trow += "".join(f"{_fmt(name, g1_test[name]):>12}" for name in _METRICS)
        lines.append("G1 held-out TEST metrics (paper, 3-seed mean; note QLIKE >= G0 - graph does not help):")
        lines.append(trow)
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
    fig.suptitle("Track-B P0->G1 validation ladder (G1 = final/proposed, red)", fontsize=13)
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
    print("TRACK-B RESULTS  -  P0->G1 model ladder  (G1 = FINAL / PROPOSED model)")
    print("=" * 96)
    for line in lines:
        print(line)

    md_path = _OUTPUT / "results_table.md"
    md = ["# Track-B Results - P0->G1 ladder", "",
          "Source JSONs: `results/pooled_20ep_aggregate.json` (P0-P3, 3-seed mean),",
          "`results/g0g1_graph_validation_comparison.json` (G0-G1, seed 42).", "",
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
