"""Generate the HAR-LSTM-GAT architecture figure as SVG + PDF + PNG (matplotlib, no external converter)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "soict_harlstmgat"


def box(ax, x, y, w, h, title, sub=None, fc="#eef2f7"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec="#334155", lw=1.4))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, ha="center", va="center",
            fontsize=11, fontweight="bold", color="#0f172a")
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.5, color="#475569")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 lw=1.4, color="#334155"))


def main():
    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

    box(ax, 0.15, 1.35, 1.9, 1.3, "HAR features", "daily / weekly /\nmonthly (t-9..t)\nper node (ticker)")
    box(ax, 2.7, 2.35, 2.2, 1.0, "LSTM (temporal)", "2-layer, per node")
    box(ax, 2.7, 0.65, 2.2, 1.0, "GAT (spatial)", "node feats at t over\ngraphical-lasso graph")
    box(ax, 2.7, 0.06, 2.2, 0.42, "", "edges: partial-corr (train-only)", fc="#e2e8f0")
    box(ax, 5.55, 1.35, 1.5, 1.3, "concat", "temporal +\nspatial")
    box(ax, 7.7, 1.35, 2.15, 1.3, "MLP head", "Parkinson variance\nat t+h  (h=1, 5)")

    arrow(ax, 2.05, 2.1, 2.66, 2.75)
    arrow(ax, 2.05, 1.9, 2.66, 1.15)
    arrow(ax, 4.9, 2.75, 5.51, 2.1)
    arrow(ax, 4.9, 1.15, 5.51, 1.9)
    arrow(ax, 7.05, 2.0, 7.66, 2.0)

    ax.text(5.0, 3.85, "LSTM+GAT: LSTM temporal branch + GAT spatial branch (graphical-lasso graph) "
            "-> head;  ablation removes the GAT branch (main LSTM).",
            ha="center", va="center", fontsize=8.5, color="#334155")
    plt.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}.svg/.pdf/.png")


if __name__ == "__main__":
    main()
