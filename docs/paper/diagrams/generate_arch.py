"""Generate the LSTM+GAT architecture figure as SVG + PDF + PNG (matplotlib, no external converter).

Matches the delivered/paper model (masked-rich-5feat-weighted-GAT): 5 node features per ticker feed a
parallel two-branch model -- a temporal 2-layer LSTM over the lookback window, and a spatial GAT that reads
the 5 raw features at day t and attends over a directed volume->Parkinson Top-5, edge-weighted, two-hop
graph (train-only). The branches are concatenated into an MLP head predicting the Parkinson variance at t+h.
The ablation removes the GAT branch, leaving the same-feature LSTM.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

matplotlib.rcParams["svg.fonttype"] = "none"  # keep labels as real (searchable) text
OUT = Path(__file__).resolve().parent / "soict_harlstmgat"


def box(ax, x, y, w, h, title, sub=None, fc="#eef2f7", tfs=10.0, sfs=7.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec="#334155", lw=1.4))
    ax.text(x + w / 2, y + h * (0.68 if sub else 0.5), title, ha="center", va="center",
            fontsize=tfs, fontweight="bold", color="#0f172a")
    if sub:
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=sfs, color="#475569")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 lw=1.4, color="#334155"))


def main():
    fig, ax = plt.subplots(figsize=(11.4, 3.7))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4.3)
    ax.axis("off")

    box(ax, 0.2, 1.05, 3.0, 1.75, "5 node features",
        "PK, HAR-w, HAR-m,\nmarket PK, volume z\n(per ticker, at t)")
    box(ax, 3.9, 2.45, 2.85, 1.1, "LSTM (temporal)", "2-layer LSTM over\nlookback = 10")
    box(ax, 3.9, 0.85, 2.85, 1.1, "GAT (spatial)", "5 feats at t;\nweighted 2-hop attention")
    box(ax, 7.45, 1.35, 1.85, 1.35, "concat", "temporal +\nspatial")
    box(ax, 9.85, 1.35, 2.35, 1.35, "MLP head", "Parkinson variance\nat t+h (h=1,5,10,22)")

    arrow(ax, 3.2, 2.2, 3.85, 2.9)
    arrow(ax, 3.2, 1.6, 3.85, 1.35)
    arrow(ax, 6.75, 2.9, 7.4, 2.35)
    arrow(ax, 6.75, 1.35, 7.4, 1.9)
    arrow(ax, 9.3, 2.0, 9.81, 2.0)

    ax.text(5.325, 0.45, "edges: directed vol->Parkinson Top-5 (train-only)",
            ha="center", va="center", fontsize=7.6, style="italic", color="#64748b")
    ax.text(6.2, 4.05, "LSTM+GAT: parallel temporal (LSTM) and spatial (GAT: directed vol->Parkinson, "
            "weighted 2-hop) branches -> MLP head;  ablation removes the GAT branch (same-feature LSTM).",
            ha="center", va="center", fontsize=7.8, color="#334155")
    plt.tight_layout()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}.svg/.pdf/.png")


if __name__ == "__main__":
    main()
