"""Generate the E0->E1->E2->E3 EDA-GNN ablation architecture SVG.

Plain matplotlib box-and-arrow rendering (no Graphviz/Mermaid dependency - neither
is installed here and Mermaid does not render in the target viewer). Mirrors the
style of docs/paper/diagrams/generate_current_architecture.py and
generate_proposed_updates.py: Agg backend + svg.fonttype='none' so every label
stays a searchable <text> node in a reviewer's viewer.

Run: python docs/paper/diagrams/generate_eda_ablation.py

Ground truth (every value below traces to these sources - do not edit numbers
without re-checking them):
  * docs/reports/2026-08-11_1631_eda_gnn_results.md  (the EDA-GNN ablation:
        E0/E1/E2/E3 test QLIKE + Diebold-Mariano-vs-HAR verdicts, plus the
        E3off / G1corr comparison rungs)
  * docs/eda/graph_recommendation.json  (leakage-safe EDA that recommended the
        node features [pk_daily, pk_weekly, pk_monthly, volume_zscore_20] and the
        directed vol->PK lead-lag edge, top_k=5, dynamic; market factor dominates)

All configs are pooled per-ticker-day, leakage-safe, share the same val/test
observations as the consistent ladder, and are evaluated over 3 seeds on TEST.

The honest reading the diagram is built to make visible: the accumulating NODE
FEATURES (MarketPK, then volume_zscore_20) drive the DM-significant HAR-beat
(E1 p=0.017; E2 p=0.012, best); adding the directed vol->PK GRAPH at E3 does NOT
beat HAR under DM (p=0.116) and erodes the win (QLIKE 0.5681 -> 0.5709), also
losing to a plain correlation edge (E3 vs G1corr p=0.044).
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

COLOR_BASE = "#ececec"     # shared HAR node features (present in every rung)
COLOR_MARKET = "#dfeadf"   # + MarketPK node feature (added at E1)
COLOR_VOL = "#cfe8cf"      # + volume_zscore_20 node feature (added at E2)
COLOR_GRAPH = "#cfe2f3"    # directed vol->PK graph layer (added only at E3)
COLOR_LSTM = "#eef2f7"     # pooled LSTM backbone stage
COLOR_HEAD = "#f2e6f7"     # prediction head
COLOR_OUT = "#fbe2b0"      # final output
COLOR_WIN = "#cfe8cf"      # DM-significant HAR-beat badge (E1, E2)
COLOR_FAIL = "#f6d0d0"     # no HAR-beat badge (E3)
COLOR_REF = "#e0e0e0"      # HAR reference badge (E0)
EDGE = "#333333"
EDGE_GRAPH = "#1f5c99"     # emphasised border for the graph layer
EDGE_WIN = "#2e7d32"
EDGE_FAIL = "#b23b3b"


def box(ax, cx, cy, w, h, text, facecolor=COLOR_BASE, fontsize=8.0,
        edgecolor=EDGE, dashed=False, bold=False, lw=1.3):
    b = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02", facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, linestyle="--" if dashed else "-", zorder=2,
    )
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            zorder=3, wrap=True, fontweight="bold" if bold else "normal")
    return (cx, cy, w, h)


def arrow(ax, p_from, p_to, label=None, style="-|>", color=EDGE, lw=1.3,
          curve=0.0, fontsize=7.0, dashed=False, label_dy=0.0):
    a = FancyArrowPatch(
        p_from, p_to, arrowstyle=style, mutation_scale=13,
        connectionstyle=f"arc3,rad={curve}", color=color, linewidth=lw,
        linestyle="--" if dashed else "-", zorder=1,
    )
    ax.add_patch(a)
    if label:
        mx = (p_from[0] + p_to[0]) / 2
        my = (p_from[1] + p_to[1]) / 2 + curve * 0.6 + label_dy
        ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize,
                color=color, zorder=3, backgroundcolor="white")


def right(b):
    return (b[0] + b[2] / 2, b[1])


def left(b):
    return (b[0] - b[2] / 2, b[1])


def build():
    W, H = 20.0, 13.8
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ax.text(W / 2, H - 0.35,
            "EDA-GNN ablation  E0 -> E1 -> E2 -> E3:  node features accumulate; "
            "the graph is added only at E3",
            ha="center", va="center", fontsize=14.5, fontweight="bold")
    ax.text(W / 2, H - 0.80,
            "One node = one VN30 ticker on one day; pooled per-ticker-day, "
            "leakage-safe, shared val/test observations, 3-seed TEST.   "
            "Reported metric: test QLIKE (lower is better); verdict: "
            "Diebold-Mariano vs HAR.",
            ha="center", va="center", fontsize=9.0, color="#555555")

    # ==================================================== SHARED FLOW (SPINE)
    sy = 11.55
    ax.text(1.35, sy + 1.02, "Shared flow (E1-E3 backbone)",
            ha="left", va="center", fontsize=9.2, fontweight="bold",
            color="#333333")
    nf = box(ax, 2.55, sy, 3.0, 1.35,
             "Node features\n(accumulate E0->E3)\nHAR + MarketPK\n+ volume_zscore_20",
             COLOR_BASE, fontsize=7.4)
    lstm = box(ax, 6.15, sy, 2.5, 1.35,
               "Pooled LSTM\n(shared weights,\nall ticker-days)", COLOR_LSTM,
               fontsize=7.6)
    base = box(ax, 9.35, sy, 2.2, 1.35, "Base\nembedding", COLOR_LSTM,
               fontsize=8.0, bold=True)
    graph = box(ax, 12.75, sy, 2.9, 1.35,
                "Graph message-\npassing  (E3 only)\ndirected vol->PK,\nmasked residual",
                COLOR_GRAPH, fontsize=7.2, dashed=True, edgecolor=EDGE_GRAPH,
                lw=1.8, bold=True)
    head = box(ax, 15.9, sy, 2.1, 1.35,
               "Head\nLinear +\ndenorm + floor", COLOR_HEAD, fontsize=7.6)
    out = box(ax, 18.55, sy, 2.2, 1.35,
              "sigma_hat^2\n5-day Parkinson\nvariance", COLOR_OUT, fontsize=7.6,
              bold=True)
    arrow(ax, right(nf), left(lstm))
    arrow(ax, right(lstm), left(base))
    arrow(ax, right(base), left(graph), label="E1/E2:\nskip", fontsize=6.4,
          curve=0.0)
    arrow(ax, right(graph), left(head))
    arrow(ax, right(head), left(out))

    # ==================================================== ABLATION STRIP
    cols = {"E0": 3.05, "E1": 7.55, "E2": 12.05, "E3": 16.55}
    cw = 4.0
    # Feature/component rows, top -> bottom, with a fixed y per row so presence
    # reads as a staircase across the four columns.
    row_graph = 8.35
    row_vol = 7.20
    row_market = 6.05
    row_har = 4.90
    row_back = 3.55
    hdr_y = 9.35

    ax.text(W / 2, 9.98,
            "Ablation strip - what each rung adds (filled chip = present; the "
            "newly-added component is outlined)",
            ha="center", va="center", fontsize=9.2, fontweight="bold",
            color="#333333")

    # Column frames + headers.
    for name, cx in cols.items():
        frame = FancyBboxPatch((cx - cw / 2, 2.55), cw, 7.05,
                               boxstyle="round,pad=0.02", facecolor="none",
                               edgecolor="#c2c2c2", linewidth=1.0,
                               linestyle="-", zorder=0)
        ax.add_patch(frame)
    hdr = {}
    for name, cx in cols.items():
        hdr[name] = box(ax, cx, hdr_y, cw - 0.4, 0.72, name, "#f4f4f4",
                        fontsize=11.0, bold=True)
    # Ladder progression arrows across the header row.
    for a, b in (("E0", "E1"), ("E1", "E2"), ("E2", "E3")):
        arrow(ax, right(hdr[a]), left(hdr[b]), lw=1.5)

    def chip(cx, cy, text, facecolor, new=False):
        box(ax, cx, cy, cw - 0.55, 0.86, text, facecolor, fontsize=7.0,
            edgecolor=EDGE_GRAPH if new else EDGE,
            lw=2.2 if new else 1.1, bold=new)

    # --- 3 HAR scales: present in EVERY rung (the shared base) ---
    for name, cx in cols.items():
        chip(cx, row_har,
             "3 HAR scales\nhar_daily / har_weekly / har_monthly\n(Parkinson variance)",
             COLOR_BASE)

    # --- + MarketPK: added at E1, carried through E2, E3 ---
    for name, cx in cols.items():
        if name == "E0":
            continue
        chip(cx, row_market,
             "+ MarketPK\ncross-sectional median PK\n(global node feature)",
             COLOR_MARKET, new=(name == "E1"))

    # --- + volume_zscore_20: added at E2, carried through E3 ---
    for name, cx in cols.items():
        if name in ("E0", "E1"):
            continue
        chip(cx, row_vol,
             "+ volume_zscore_20\ntrailing 20-day log-volume z-score\n(node feature)",
             COLOR_VOL, new=(name == "E2"))

    # --- directed vol->PK graph layer: added ONLY at E3 ---
    chip(cols["E3"], row_graph,
         "+ vol->PK graph\ndirected lead-lag, Top-5, dynamic\n"
         "TRAIN-frozen adj., masked residual MP",
         COLOR_GRAPH, new=True)

    # --- backbone row ---
    box(ax, cols["E0"], row_back, cw - 0.55, 0.78,
        "linear HAR\n(reference)", COLOR_REF, fontsize=7.4)
    for name in ("E1", "E2", "E3"):
        box(ax, cols[name], row_back, cw - 0.55, 0.78,
            "pooled LSTM ->\nbase embedding", COLOR_LSTM, fontsize=7.4)

    # ==================================================== RESULTS FOOTER
    def result(cx, qlike, verdict, badge_color, edge_color):
        ax.text(cx, 2.15, f"test QLIKE  {qlike}", ha="center", va="center",
                fontsize=9.4, fontweight="bold")
        box(ax, cx, 1.45, cw - 0.55, 0.66, verdict, badge_color, fontsize=7.6,
            edgecolor=edge_color, lw=1.6, bold=True)

    result(cols["E0"], "0.5735", "reference (HAR)", COLOR_REF, EDGE)
    result(cols["E1"], "0.5686", "beats HAR  DM p=0.017  OK", COLOR_WIN, EDGE_WIN)
    result(cols["E2"], "0.5681", "beats HAR  DM p=0.012  OK  BEST", COLOR_WIN,
           EDGE_WIN)
    result(cols["E3"], "0.5709", "no HAR-beat  DM p=0.116  x", COLOR_FAIL,
           EDGE_FAIL)

    # ==================================================== SIDE COMPARISON + TAKEAWAY
    ax.text(0.35, 0.92,
            "Graph-variant checks:  E3off (graph disabled) QLIKE 0.5760   |   "
            "G1corr (correlation edge) QLIKE 0.5708   |   E3 vs G1corr  p=0.044 "
            "(the vol->PK edge loses to a plain correlation edge).",
            ha="left", va="center", fontsize=8.2, color="#555555")
    ax.text(0.35, 0.42,
            "Takeaway:  the node features (MarketPK, then volume_zscore_20) drive "
            "the DM-significant HAR-beat (E1, E2);  the graph (E3) does not help "
            "and erodes the win (0.5681 -> 0.5709).",
            ha="left", va="center", fontsize=8.6, fontweight="bold",
            color="#333333")

    return fig


def main():
    fig = build()
    svg_path = OUT_DIR / "eda_ablation_E0_E3.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    print(f"wrote {svg_path}")
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    main()
