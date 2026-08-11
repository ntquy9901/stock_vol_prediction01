"""Render architecture diagrams for the EDA-recommended GNN ablation ladder (E0 -> E3).

Produces two SVGs (+ PNG) in this folder:
  * eda_gnn_ladder.svg   -- the four rungs side by side (E0 HAR / E1 / E2 / E3).
  * eda_gnn_e3_detail.svg -- the E3 price-graph model in detail (per-node LSTM + directed
    volume->PK message passing + head + positivity floor).

Run: python baselines/2026-08-11_eda_gnn_baseline/design/diagrams/generate_eda_gnn_diagrams.py
No project imports; pure matplotlib so it renders anywhere.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent

# Palette (colour-blind-safe, light background).
C_FEAT = "#DCE7F5"      # input features
C_HAR = "#B7D3EF"       # HAR block
C_NEW = "#F6D9A0"       # new EDA features (MarketPK / volume_zscore_20)
C_ENC = "#CDE7CB"       # encoder (LSTM)
C_GRAPH = "#E9C6DE"     # graph / message passing
C_HEAD = "#D9D2EC"      # head
C_OUT = "#F4C6C1"       # output
C_EDGE = "#5A6B7B"
FONT = 9


def _box(ax, x, y, w, h, text, color, fontsize=FONT, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.1, edgecolor=C_EDGE, facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", color="#1c2530", wrap=True)


def _arrow(ax, x1, y1, x2, y2, style="-|>", color=C_EDGE, lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=1, shrinkB=1))


def _column(ax, x0, title, feats, encoder, extra, beats):
    """Draw one ladder rung as a vertical stack in a column of width 0.185 starting at x0."""
    w = 0.185
    cx = x0 + w / 2
    fh, pitch = 0.05, 0.062  # feature box height / vertical pitch
    ax.text(cx, 0.95, title, ha="center", va="center", fontsize=11, fontweight="bold",
            color="#12212e")
    # feature stack (top-down)
    top = 0.90
    fbot = top
    for index, (label, color) in enumerate(feats):
        y = top - fh - index * pitch
        _box(ax, x0, y, w, fh, label, color, fontsize=8)
        fbot = y
    # encoder box
    ey, eh = 0.46, 0.09
    _box(ax, x0, ey, w, eh, encoder, C_ENC, fontsize=8.4, bold=True)
    _arrow(ax, cx, fbot, cx, ey + eh)
    last_bottom = ey
    if extra is not None:
        gy, gh = 0.32, 0.08
        _box(ax, x0, gy, w, gh, extra, C_GRAPH, fontsize=8.0, bold=True)
        _arrow(ax, cx, ey, cx, gy + gh)
        last_bottom = gy
    # head
    hy, hh = 0.19, 0.06
    _box(ax, x0, hy, w, hh, "Linear head", C_HEAD, fontsize=8.2)
    _arrow(ax, cx, last_bottom, cx, hy + hh)
    # output
    oy, oh = 0.075, 0.055
    _box(ax, x0, oy, w, oh, "sigma_t+5\n(volatility)", C_OUT, fontsize=8)
    _arrow(ax, cx, hy, cx, oy + oh)
    # beats-HAR annotation (just below the output box)
    ax.text(cx, 0.03, beats, ha="center", va="center", fontsize=7.6, style="italic",
            color="#33475b")


def draw_ladder():
    fig, ax = plt.subplots(figsize=(14.0, 9.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.995, "EDA-recommended GNN ablation ladder — VN30 5-day Parkinson volatility (h=5)",
            ha="center", va="top", fontsize=13, fontweight="bold", color="#0f1c28")

    har = ("HAR: pk_daily", C_HAR), ("HAR: pk_weekly", C_HAR), ("HAR: pk_monthly", C_HAR)
    mkt = ("+ MarketPK (global)", C_NEW),
    vol = ("+ volume_zscore_20", C_NEW),

    _column(ax, 0.045, "E0  HAR", list(har),
            "Pooled linear\nregression", None,
            "reference (baseline)")
    _column(ax, 0.28, "E1  +MarketPK", list(har) + list(mkt),
            "2-layer LSTM\n(hidden 64)", None,
            "beats HAR QLIKE\nDM p=0.017")
    _column(ax, 0.515, "E2  +vol_z", list(har) + list(mkt) + list(vol),
            "2-layer LSTM\n(hidden 64)", None,
            "beats HAR QLIKE\nDM p=0.012")
    _column(ax, 0.75, "E3  +vol2pk graph", list(har) + list(mkt) + list(vol),
            "per-node 2-layer\nLSTM (hidden 64)",
            "GAT-style message passing\n(directed vol->PK Top-5)",
            "ties HAR (QLIKE p=0.116,\nerror p=0.19) — graph null")

    # legend
    handles = [
        mpatches.Patch(color=C_HAR, label="HAR features (pk daily/weekly/monthly)"),
        mpatches.Patch(color=C_NEW, label="EDA node features (MarketPK, volume_zscore_20)"),
        mpatches.Patch(color=C_ENC, label="Temporal encoder"),
        mpatches.Patch(color=C_GRAPH, label="Cross-stock message passing (E3 only)"),
        mpatches.Patch(color=C_HEAD, label="Linear head + per-ticker positivity floor"),
        mpatches.Patch(color=C_OUT, label="Volatility forecast"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3,
              fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(HERE / f"eda_gnn_ladder.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def draw_e3_detail():
    fig, ax = plt.subplots(figsize=(12.5, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.985, "E3 — price-graph model (recommended GNN)  |  33 VN30 nodes, per snapshot",
            ha="center", va="top", fontsize=12.5, fontweight="bold", color="#0f1c28")

    # left: one node's feature window
    _box(ax, 0.02, 0.60, 0.24, 0.24,
         "Node i window (22 days x 5 feat)\n"
         "pk_daily, pk_weekly, pk_monthly,\nMarketPK, volume_zscore_20", C_FEAT, fontsize=8.4)
    _box(ax, 0.05, 0.40, 0.18, 0.10, "2-layer LSTM\n(shared, hidden 64)", C_ENC, fontsize=8.6, bold=True)
    _arrow(ax, 0.14, 0.60, 0.14, 0.50)
    _box(ax, 0.05, 0.25, 0.18, 0.07, "h_i  (node embedding)", C_HEAD, fontsize=8.2)
    _arrow(ax, 0.14, 0.40, 0.14, 0.32)
    ax.text(0.14, 0.885, "per node i (all 33 in parallel)", ha="center", fontsize=8.4,
            style="italic", color="#33475b")

    # middle: directed vol2pk graph
    gx, gy, gr = 0.55, 0.55, 0.14
    _box(ax, 0.36, 0.30, 0.36, 0.44, "", C_GRAPH, fontsize=8)
    ax.text(0.54, 0.71, "Directed volume->PK graph\n(TRAIN-frozen Top-5 + self-loop)",
            ha="center", fontsize=8.8, fontweight="bold", color="#3a1f34")
    import math
    nodes = {}
    for k, ang in enumerate(range(0, 360, 60)):
        a = math.radians(ang)
        nx, ny = gx + gr * math.cos(a), gy + gr * math.sin(a) * 0.7
        nodes[k] = (nx, ny)
        ax.add_patch(plt.Circle((nx, ny), 0.021, color="#B96FA6", ec=C_EDGE, lw=1.0, zorder=3))
    # directed edges into node 0 (target j aggregates from Top-5 sources i)
    tgt = nodes[0]
    for k in (1, 2, 3, 4, 5):
        _arrow(ax, nodes[k][0], nodes[k][1], tgt[0], tgt[1], color="#7a4a6c", lw=1.1)
    ax.text(gx, 0.335, "message = softmax(A[j,i]) . h_i   (A[j,i] = vol->PK lead-lag)",
            ha="center", fontsize=7.8, color="#3a1f34")

    _arrow(ax, 0.23, 0.285, 0.37, 0.42, ls="--", color=C_EDGE)
    ax.text(0.30, 0.30, "h_i as\nnode state", ha="center", fontsize=7.6, style="italic", color="#33475b")

    # right: residual + head + positivity
    _box(ax, 0.76, 0.52, 0.22, 0.10, "Residual update\nh_j <- h_j + MP(h_j)", C_GRAPH, fontsize=8.4, bold=True)
    _arrow(ax, 0.72, 0.55, 0.76, 0.57)
    _box(ax, 0.78, 0.37, 0.18, 0.08, "Linear head", C_HEAD, fontsize=8.4)
    _arrow(ax, 0.87, 0.52, 0.87, 0.45)
    _box(ax, 0.76, 0.22, 0.22, 0.08, "Positivity floor\n(per-ticker denorm)", C_OUT, fontsize=8.2)
    _arrow(ax, 0.87, 0.37, 0.87, 0.30)
    _box(ax, 0.78, 0.09, 0.18, 0.07, "sigma_j,t+5", C_OUT, fontsize=8.6, bold=True)
    _arrow(ax, 0.87, 0.22, 0.87, 0.16)

    ax.text(0.5, 0.03,
            "Absent nodes are masked (no message in/out); neighbour identity + weights estimated on "
            "each ticker's TRAIN split only and frozen across snapshots (leakage-safe).",
            ha="center", fontsize=7.8, style="italic", color="#33475b")
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(HERE / f"eda_gnn_e3_detail.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    draw_ladder()
    draw_e3_detail()
    print(f"wrote diagrams to {HERE}")
