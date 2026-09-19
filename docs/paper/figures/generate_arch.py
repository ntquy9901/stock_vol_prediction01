"""VolTree model-architecture figure (matplotlib -> PNG + PDF), authoritative source.

Redrawn 2026-09-19 per advisor: describe the FULL VolTree model only (the four ablation variants are
described in the text / Table 2, not in this figure), with larger fonts that stay legible at print size.
Writes the paper figure ``fig_architecture.{png,pdf}`` directly (the earlier draw.io export is retired
because no draw.io CLI is available in this environment; this script is now the single source).

Pipeline (top -> bottom): OHLCV (endogenous) and the earnings calendar (external) feed an endogenous
feature vector R^8 (3 HAR lags + 5 log-vol momentum) and an earnings block R^4; a gamma-loss
gradient-boosted tree ensemble (XGBoost) emits the per-stock multi-horizon Parkinson-variance forecast;
a post-prediction stage smooths those forecasts across each test day's cross-section over a
leaf-cooccurrence kNN graph built from the XGBoost trees' leaf indices. All stages are components of VolTree.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle  # noqa: E402

matplotlib.rcParams["svg.fonttype"] = "none"

# Larger, print-legible font sizes (advisor: previous figure fonts were too small).
FS_BOX = 13.0        # primary box labels
FS_SUB = 11.0        # secondary lines inside boxes
FS_ANNOT = 11.0      # arrow / annotation labels
FS_TITLE = 14.0      # panel titles
FS_FORMULA = 14.0    # blend formula


def box(ax, x, y, w, h, t, fc, fs=FS_BOX, ls="solid", ec="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4, ls=ls))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)


def arr(ax, x1, y1, x2, y2, txt=None, dy=0.20):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.6, color="0.2"))
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, txt, ha="center", fontsize=FS_ANNOT, color="0.25")


def graph_glyph(ax, cx, cy):
    """Small same-day cross-section: nodes wired by kNN leaf-cooccurrence edges (grayscale-safe)."""
    lo = [(cx - 0.60, cy + 0.12), (cx - 0.16, cy + 0.52), (cx + 0.22, cy + 0.08), (cx - 0.12, cy - 0.40)]
    hi = [(cx + 0.78, cy + 0.46), (cx + 0.98, cy - 0.10)]
    for a, b in [(0, 1), (0, 2), (1, 2), (0, 3)]:
        ax.plot([lo[a][0], lo[b][0]], [lo[a][1], lo[b][1]], color="0.45", lw=1.3, zorder=1)
    ax.plot([hi[0][0], hi[1][0]], [hi[0][1], hi[1][1]], color="0.45", lw=1.3, ls="--", zorder=1)
    for (x, y) in lo:
        ax.add_patch(Circle((x, y), 0.13, fc="0.85", ec="0.2", lw=1.2, zorder=2))
    for (x, y) in hi:
        ax.add_patch(Circle((x, y), 0.13, fc="white", ec="0.2", lw=1.2, zorder=2))


def main():
    fig, ax = plt.subplots(figsize=(8.2, 8.4))
    ax.axis("off"); ax.set_xlim(0, 9); ax.set_ylim(0, 9.7)

    # ---- title (VolTree only; variants live in the text / Table 2) ----
    ax.text(4.5, 9.45, "VolTree", ha="center", fontsize=17, fontweight="bold")

    # ---- two input sources at the top: OHLCV (endogenous) and the earnings calendar (external) ----
    box(ax, 0.9, 8.25, 3.2, 0.90,
        r"OHLCV (stock $i$)" + "\n" + r"$\to\ \mathrm{pk}_{i,t}$", "#eaf2fb")
    box(ax, 4.9, 8.25, 3.2, 0.90,
        "earnings calendar\n(external)", "#fdf3e3", ec="#b8791f")

    # ---- feature-vector boxes (endogenous R^8  +  earnings R^4) ----
    box(ax, 0.8, 6.45, 3.4, 1.05,
        r"endogenous $\mathbb{R}^{8}$" + "\n" + "HAR lags + momentum", "#e7f4ea", ec="#2f7d4a")
    box(ax, 4.8, 6.45, 3.4, 1.05,
        r"earnings $\mathbb{R}^{4}$" + "\n" + "distance to release", "#fdf3e3", ec="#b8791f")
    arr(ax, 2.5, 8.25, 2.5, 7.50)                      # OHLCV -> endogenous block
    arr(ax, 6.5, 8.25, 6.5, 7.50)                      # earnings calendar -> earnings block

    # ---- both feature blocks converge into the XGBoost box ----
    box(ax, 1.9, 4.75, 5.2, 1.10,
        "Gamma-loss XGBoost\n" +
        r"300 trees $\cdot$ 31 leaves $\cdot$ reg:gamma ($=$QLIKE)", "#e9e9f5", fs=12.5)
    arr(ax, 2.5, 6.45, 3.6, 5.85)                           # endogenous -> XGBoost
    arr(ax, 6.5, 6.45, 5.4, 5.85)                           # earnings -> XGBoost

    # ---- per-stock multi-horizon forecast ----
    box(ax, 2.55, 3.60, 3.9, 0.95,
        r"per-stock forecast  $\hat{\sigma}^2_{i,t+h}$" + "\n" + r"$h\in\{1,5,10,22\}$", "#eaf2fb")
    arr(ax, 4.5, 4.75, 4.5, 4.55)                           # XGBoost -> forecast

    # ---- leaf-cooccurrence graph smoothing stage (panel at the bottom) ----
    px, py, pw, ph = 0.45, 0.28, 8.1, 2.62
    ax.add_patch(FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0.03,rounding_size=0.10",
                                fc="#fbfbfd", ec="#7a6fae", lw=1.5))
    ax.text(px + pw / 2, py + ph - 0.32, "leaf-cooccurrence graph smoothing", ha="center",
            fontsize=FS_TITLE, color="#4a417a", fontweight="bold")
    graph_glyph(ax, px + 1.8, py + 1.55)
    ax.text(px + 1.8, py + 0.40, r"kNN on shared leaves ($k{=}10$)",
            ha="center", fontsize=FS_SUB, color="0.25")
    tx = px + 5.4
    ax.text(tx, py + 1.80,
            r"$\hat y^{\,sm}_i=(1-\alpha)\,\hat y_i+\alpha\,\overline{\hat y}_{\,\mathrm{kNN}(i)}$",
            ha="center", fontsize=FS_FORMULA)
    ax.text(tx, py + 1.05, r"$\alpha$ fit on validation",
            ha="center", fontsize=FS_SUB, color="0.25")
    arr(ax, 4.5, 3.60, 4.5, py + ph + 0.02)               # forecast -> panel
    ax.text(4.72, 3.45, r"$\hat y_i$ (all stocks)", ha="left", va="center",
            fontsize=FS_ANNOT, color="0.25")
    arr(ax, px + 3.0, py + 1.55, tx - 1.9, py + 1.70)      # glyph -> blend formula

    # Fallback preview only: the embedded fig_architecture.{pdf} is the hand-designed diagram
    # (docs/paper/figures/fig_architecture.drawio), so this writes fig_architecture_mpl.* and never clobbers it.
    for ext_ in ("png", "pdf"):
        fig.savefig(f"docs/paper/figures/fig_architecture_mpl.{ext_}", dpi=190, bbox_inches="tight")


if __name__ == "__main__":  # pragma: no cover - CLI entry (main() is covered by the test directly)
    main()
