"""Generate the PROPOSED-UPDATES architecture SVG (current arch + overlaid changes).

Plain matplotlib box-and-arrow rendering (no Graphviz/Mermaid dependency - neither is
installed in this environment and Mermaid does not render in the target viewer). Mirrors
the style of docs/paper/diagrams/generate_final_combined.py.

Run: python docs/paper/diagrams/generate_proposed_updates.py

The CURRENT combined pipeline is drawn as a neutral spine (inputs -> encoders -> masked
GNN -> metrics, plus the external HAR lane). Each PROPOSED / DONE change is drawn as a
distinctly-coloured callout attached by a dashed connector to the exact stage it touches,
carrying a status tag so a reviewer can tell built-from-proposed at a glance:

  [DONE]        green, solid  - already implemented in the masked-gnn baseline
  [PENDING DM]  amber, solid  - implemented, but the result claim awaits multi-seed + DM
  [PROPOSED]    blue, dashed  - not yet built; a recommended change

Overlay -> motivating report map (each label also appears in the HTML review table):
  * k-NN top-8 sparse adjacency (single-seed hint Delta -0.00253) .....
        docs/reports/2026-08-09_0747_all_metrics_consolidated.md (cluster 5, seed42);
        docs/reports/2026-08-08_gnn_sparse_data_research.md (sparse adjacency family)
  * masked availability-aware message passing + presence mask (DONE novelty) .....
        docs/reports/2026-08-08_gnn_sparse_data_research.md (Rank 1)
  * cache frozen-encoder embeddings ~15x / encode present nodes only ~2.1x / epochs ~7 ...
        docs/reports/2026-08-09_g0g1_training_slowness.md (recs #1, #2, #7)
  * ReduceLROnPlateau LR scheduler .....
        docs/reports/2026-08-09_pooled_convergence_rootcause.md (rec 1)
  * HARQ + log-HAR + GARCH-family + EWMA/persistence baselines .....
        docs/reports/2026-08-08_har_benchmark_literature.md (secs 3-5)
  * Diebold-Mariano + Model Confidence Set .....
        docs/reports/2026-08-08_har_benchmark_literature.md (sec 4)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Keep label text as real <text> nodes (not outlined paths) so the SVG stays
# searchable/selectable in a reviewer's viewer.
matplotlib.rcParams["svg.fonttype"] = "none"
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
from pathlib import Path  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

COLOR_CURRENT = "#ececec"   # current pipeline stage (as built)
COLOR_INPUT = "#e0e0e0"     # inputs / preprocessing
COLOR_DONE = "#cfe8cf"      # DONE - already implemented
COLOR_PENDING = "#fbe2b0"   # PENDING DM - built, claim awaits confirmation
COLOR_PROPOSED = "#d9e8fb"  # PROPOSED - not yet built
COLOR_METRICS = "#f2e6f7"   # evaluation / metrics
EDGE = "#333333"
CONNECT = "#777777"

_STATUS_STYLE = {
    "DONE": (COLOR_DONE, False),
    "PENDING DM": (COLOR_PENDING, False),
    "PROPOSED": (COLOR_PROPOSED, True),
}


def box(ax, cx, cy, w, h, text, facecolor=COLOR_CURRENT, fontsize=8.0,
        edgecolor=EDGE, dashed=False, bold=False):
    b = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02", facecolor=facecolor, edgecolor=edgecolor,
        linewidth=1.6 if bold else 1.3, linestyle="--" if dashed else "-", zorder=2,
    )
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            zorder=3, wrap=True, fontweight="bold" if bold else "normal")
    return (cx, cy, w, h)


def overlay(ax, cx, cy, w, h, text, status, fontsize=7.0):
    """A proposed/done change callout, coloured + tagged by status."""
    facecolor, dashed = _STATUS_STYLE[status]
    return box(ax, cx, cy, w, h, text, facecolor=facecolor, fontsize=fontsize,
               dashed=dashed)


def arrow(ax, p_from, p_to, label=None, style="-|>", color=EDGE, lw=1.3,
          curve=0.0, fontsize=7.0, dashed=False, label_dy=0.0):
    a = FancyArrowPatch(
        p_from, p_to, arrowstyle=style, mutation_scale=13,
        connectionstyle=f"arc3,rad={curve}", color=color, linewidth=lw,
        linestyle="--" if dashed else "-", zorder=1,
    )
    ax.add_patch(a)
    if label:
        mx, my = (p_from[0] + p_to[0]) / 2, (p_from[1] + p_to[1]) / 2 + curve * 0.6 + label_dy
        ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize,
                color=color, zorder=3, backgroundcolor="white")


def connect(ax, p_from, p_to):
    """Dashed grey connector tying an overlay to the stage it modifies."""
    arrow(ax, p_from, p_to, style="-|>", color=CONNECT, lw=1.2, dashed=True)


def right(b):
    return (b[0] + b[2] / 2, b[1])


def left(b):
    return (b[0] - b[2] / 2, b[1])


def top(b):
    return (b[0], b[1] + b[3] / 2)


def bottom(b):
    return (b[0], b[1] - b[3] / 2)


def build():
    W, H = 18.5, 13.8
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ax.text(W / 2, H - 0.32,
            "Proposed updates over the current combined architecture "
            "(pooled news-GNN, Track B)",
            ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(W / 2, H - 0.78,
            "Neutral boxes = current pipeline as built.  Coloured callouts = changes, "
            "attached (dashed) to the stage they touch.  Tags:  [DONE]  [PENDING DM]  [PROPOSED]",
            ha="center", va="center", fontsize=8.5, color="#555555")

    # ---------------------------------------------------------------- CURRENT SPINE
    inputs = box(ax, 2.3, 7.5, 3.3, 1.9,
                 "Inputs + causal\npreprocessing\n\nprice OHLCV + news panel\n"
                 "train-only fit,\nleakage barrier,\nper-ticker 70/15/15 split",
                 COLOR_INPUT, fontsize=7.4)

    har = box(ax, 7.1, 10.6, 3.4, 1.35,
              "P0: HAR classical baseline\nLinear Reg. (Corsi 2009)\n"
              "per-ticker, external", COLOR_CURRENT, fontsize=7.6, bold=True)

    encoders = box(ax, 7.1, 7.5, 4.0, 1.9,
                   "Pooled shared-weight encoders\n(P1 / P2 / P3)\n\n"
                   "Price LSTM + News LSTM\n+ per-ticker gate\nlate-fusion concat + head",
                   COLOR_CURRENT, fontsize=7.4, bold=True)

    graph = box(ax, 12.4, 7.5, 4.2, 1.9,
                "Masked GNN (G0 / G1)\nfrozen P3 encoders\n\n"
                "per-day node set over 33-ticker\nvocabulary; correlation adjacency\n"
                "(dense default)", COLOR_CURRENT, fontsize=7.4, bold=True)

    metrics = box(ax, 16.1, 9.0, 3.6, 1.6,
                  "Evaluation\n6 metrics (raw scale)\nMSE / RMSE / MAE /\nR2 / QLIKE / DirAcc",
                  COLOR_METRICS, fontsize=7.6, bold=True)

    arrow(ax, top(inputs), left(har), curve=-0.15, label="HAR features", fontsize=6.6)
    arrow(ax, right(inputs), left(encoders))
    arrow(ax, right(encoders), left(graph))
    arrow(ax, top(har), (metrics[0] - metrics[2] / 2, metrics[1] + 0.2), curve=-0.1)
    arrow(ax, right(graph), (metrics[0] - metrics[2] / 2, metrics[1] - 0.3), curve=0.12)

    # ------------------------------------------------------------ TOP: BASELINE SUITE
    baseline_suite = overlay(
        ax, 6.7, 12.55, 5.6, 1.55,
        "Expanded baseline suite  [PROPOSED]\n"
        "+ HARQ  + log-HAR  + GARCH-family (GJR / EGARCH)\n"
        "+ EWMA / RiskMetrics  + random-walk persistence\n"
        "main reviewer-risk fix  (HAR-benchmark literature report)",
        "PROPOSED", fontsize=7.2)
    connect(ax, bottom(baseline_suite), top(har))

    # ------------------------------------------------- RIGHT: SIGNIFICANCE TESTS
    sig = overlay(
        ax, 16.1, 12.15, 3.7, 1.9,
        "Significance testing  [PROPOSED]\n"
        "+ Diebold-Mariano (pairwise)\n+ Model Confidence Set\n"
        "QLIKE + MSE primary\n(Patton 2011; HAR-benchmark report)",
        "PROPOSED", fontsize=7.0)
    connect(ax, bottom(sig), top(metrics))

    # ------------------------------------------------ LOWER LEFT: OPTIMIZER ROBUSTNESS
    optim = overlay(
        ax, 3.0, 4.55, 4.7, 1.75,
        "Optimizer robustness  [PROPOSED]\n"
        "add ReduceLROnPlateau LR scheduler\n"
        "(dropout 0.2 / weight-decay 1e-5 /\ngrad-clip 1.0 already present)\n"
        "(pooled-convergence root-cause report)",
        "PROPOSED", fontsize=7.0)
    connect(ax, top(optim), (encoders[0] - 1.0, bottom(encoders)[1]))

    # ------------------------------- LOWER CENTRE: MASKED AVAILABILITY-AWARE (DONE)
    masked = overlay(
        ax, 9.1, 4.55, 4.9, 1.9,
        "Masked availability-aware\nmessage passing + presence mask  [DONE]\n\n"
        "news-on-graph novelty; per-day present\nnode set; absent tickers masked,\n"
        "never imputed  (sparse-data GNN report)",
        "DONE", fontsize=7.0)
    connect(ax, top(masked), (graph[0] - 1.2, bottom(graph)[1]))

    # ------------------------------ LOWER RIGHT: k-NN TOP-8 SPARSE ADJACENCY (PENDING)
    knn = overlay(
        ax, 14.7, 4.5, 5.0, 2.0,
        "k-NN top-8 sparse adjacency  [PENDING DM]\n\n"
        "implemented on the masked graph.\n"
        "single-seed hint: Delta val-loss -0.00253,\nbest on 5/6 metrics (seed42).\n"
        "multi-seed + Diebold-Mariano in flight\n(consolidated-metrics report)",
        "PENDING DM", fontsize=7.0)
    connect(ax, top(knn), (graph[0] + 1.3, bottom(graph)[1]))

    # ------------------------------------------- BOTTOM: GRAPH-TRAINING SPEEDUPS
    speed = overlay(
        ax, 11.9, 2.0, 6.6, 1.55,
        "Graph-training speedups  [PROPOSED]\n"
        "cache frozen-encoder embeddings (~15x, reuse across epochs + G0/G1)   |   "
        "encode present nodes only (~2.1x)\n"
        "epochs 15 -> ~7 (val loss converged by epoch ~5)   "
        "(G0/G1 training-slowness report)",
        "PROPOSED", fontsize=7.0)
    connect(ax, top(speed), (masked[0] + 0.3, bottom(masked)[1]))
    connect(ax, (knn[0] - 0.6, bottom(knn)[1]), (speed[0] + 2.3, top(speed)[1]))

    # ------------------------------------------------------------------- LEGEND
    ly = 0.55
    swatches = [
        (COLOR_CURRENT, "current (as built)", 0.7, False),
        (COLOR_DONE, "[DONE] implemented", 4.0, False),
        (COLOR_PENDING, "[PENDING DM] awaits confirmation", 7.4, False),
        (COLOR_PROPOSED, "[PROPOSED] not yet built (dashed)", 12.6, True),
    ]
    for c, lab, x, dashed in swatches:
        r = FancyBboxPatch((x, ly - 0.14), 0.32, 0.28, boxstyle="round,pad=0.01",
                           facecolor=c, edgecolor=EDGE, linewidth=1.0,
                           linestyle="--" if dashed else "-", zorder=2)
        ax.add_patch(r)
        ax.text(x + 0.44, ly, lab, ha="left", va="center", fontsize=7.4, zorder=3)

    return fig


def main():
    fig = build()
    svg_path = OUT_DIR / "proposed_updates_architecture.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    print(f"wrote {svg_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
