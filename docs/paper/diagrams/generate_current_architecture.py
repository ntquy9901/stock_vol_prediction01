"""Generate the CURRENT proposed-model architecture SVG (G1, Track B).

Plain matplotlib box-and-arrow rendering (no Graphviz/Mermaid dependency - neither
is installed here and Mermaid does not render in the target viewer). Mirrors the
style of docs/paper/diagrams/generate_proposed_updates.py and generate_final_combined.py:
Agg backend + svg.fonttype='none' so every label stays a searchable <text> node.

Run: python docs/paper/diagrams/generate_current_architecture.py

The drawn model is G1, the proposed Track-B pooled news-GNN, exactly as built.
Ground truth:
  * docs/reports/ladder_consistent_h5_2026-08-09_154402.md  (ladder, adjacency knn top_k=8,
        seeds [42,123,2026], horizon 5, P3 = G1 with message passing disabled)
  * docs/reports/2026-08-09_2148_old_gat_vs_new_g1_diagnosis.md  (3 HAR node features,
        k-NN-8 Pearson volatility-correlation edges, present-node masked, news channel added)
  * git show feature/masked-gnn:baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/models.py
        (PooledPriceNewsLSTM: price-LSTM + news-LSTM + per-ticker sigmoid gate, late-fusion
        concat + head; GraphAblationModel: frozen P3 encoders + residual message passing over
        the masked correlation graph; head + denormalize + softplus positivity floor)

Every element matches the implementation: node features are the three HAR scales (Parkinson
variance), the news branch is PhoBERT dual-group (146-dim) through a second LSTM, fusion is
base = h_price (+) g * h_news with a per-ticker gate, the graph layer is a single residual
message-passing step over the masked k-NN-8 correlation adjacency (correlation-weighted
aggregation - NOT multi-head attention), and the head floors a denormalized linear output.
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

COLOR_INPUT = "#e0e0e0"       # raw inputs
COLOR_ENC = "#eef2f7"         # encoder stages
COLOR_FUSION = "#dfeadf"      # node embedding / fusion
COLOR_G1 = "#cfe2f3"          # G1 graph layer - the headline
COLOR_HEAD = "#f2e6f7"        # prediction head
COLOR_OUT = "#fbe2b0"         # final output
COLOR_LADDER = "#ececec"      # ablation-ladder rungs
EDGE = "#333333"
EDGE_G1 = "#1f5c99"           # emphasised border for the headline model


def box(ax, cx, cy, w, h, text, facecolor=COLOR_ENC, fontsize=8.0,
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


def top(b):
    return (b[0], b[1] + b[3] / 2)


def bottom(b):
    return (b[0], b[1] - b[3] / 2)


def build():
    W, H = 19.0, 13.4
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ax.text(W / 2, H - 0.35,
            "Proposed model - G1 (Track B): pooled per-ticker news-GNN over a masked "
            "k-NN volatility-correlation graph",
            ha="center", va="center", fontsize=14.5, fontweight="bold")
    ax.text(W / 2, H - 0.82,
            "One node = one VN30 ticker on one day; pooled ~73k ticker-day samples.  "
            "Flow: per-node encoder  ->  graph layer (G1)  ->  head.",
            ha="center", va="center", fontsize=9.0, color="#555555")

    # ============================================================ PER-NODE ENCODER
    # Outer label for the encoder region.
    ax.text(3.55, 11.55, "Per-node encoder  (shared weights, pooled over all ticker-days)",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color="#333333")
    enc_frame = FancyBboxPatch((0.35, 3.55), 6.6, 7.85, boxstyle="round,pad=0.02",
                               facecolor="none", edgecolor="#b8b8b8", linewidth=1.1,
                               linestyle="--", zorder=0)
    ax.add_patch(enc_frame)

    # --- price branch ---
    p_seq = box(ax, 2.05, 10.35, 3.0, 1.5,
                "Price sequence  (seq_len 22)\n3 HAR scales:\nhar_daily / har_weekly / har_monthly\n"
                "(Parkinson variance sigma^2)",
                COLOR_INPUT, fontsize=7.4)
    p_lstm = box(ax, 2.05, 8.55, 3.0, 1.0,
                 "Price-LSTM\n2-layer, hidden 64", COLOR_ENC, fontsize=8.0)
    arrow(ax, bottom(p_seq), top(p_lstm))

    # --- news branch ---
    n_in = box(ax, 2.05, 6.9, 3.0, 1.2,
               "News\nPhoBERT dual-group (146-dim)\npublication-cutoff safe",
               COLOR_INPUT, fontsize=7.4)
    n_lstm = box(ax, 2.05, 5.35, 3.0, 1.0,
                 "News-LSTM\n2-layer, hidden 64", COLOR_ENC, fontsize=8.0)
    arrow(ax, bottom(n_in), top(n_lstm))

    # --- per-ticker gate ---
    gate = box(ax, 2.05, 4.15, 3.0, 0.85,
               "Per-ticker gate\ng = sigma(gate_logits[ticker])", COLOR_ENC, fontsize=7.6)
    arrow(ax, bottom(n_lstm), top(gate))

    # --- fusion / node embedding ---
    fusion = box(ax, 5.55, 7.4, 2.2, 1.7,
                 "Node embedding\n\nbase =\nh_price  (+)  g * h_news",
                 COLOR_FUSION, fontsize=7.8, bold=True)
    arrow(ax, right(p_lstm), (left(fusion)[0], fusion[1] + 0.45),
          label="h_price", fontsize=6.8, curve=-0.08)
    arrow(ax, right(gate), (left(fusion)[0], fusion[1] - 0.45),
          label="g * h_news", fontsize=6.8, curve=0.10)

    # ============================================================ GRAPH LAYER (G1)
    g1 = box(ax, 10.35, 7.4, 4.35, 2.55,
             "Graph layer  (G1)  -  message passing\n\n"
             "A = k-NN top-8 over Pearson\nvolatility-correlation\n"
             "(undirected, static per split,\npresent-node MASKED)\n\n"
             "base' = base + MP(base, A)\n"
             "correlation-weighted aggregation\n(single residual step)",
             COLOR_G1, fontsize=8.0, bold=True, edgecolor=EDGE_G1, lw=2.2)
    arrow(ax, right(fusion), left(g1), lw=1.6)
    ax.text(g1[0], bottom(g1)[1] - 0.32, "remove this layer  =>  P3",
            ha="center", va="center", fontsize=8.2, style="italic", color=EDGE_G1)

    # ============================================================ HEAD + OUTPUT
    head = box(ax, 14.9, 7.4, 2.7, 2.05,
               "Prediction head\n\n"
               "y_hat = floor(Linear(base'))\n"
               "denormalize +\npositivity (eps * softplus)",
               COLOR_HEAD, fontsize=7.8, bold=True)
    arrow(ax, right(g1), left(head), lw=1.6)

    out = box(ax, 17.85, 7.4, 2.0, 1.7,
              "Output\n5-day-ahead\nParkinson\nvariance  sigma_hat^2",
              COLOR_OUT, fontsize=8.0, bold=True)
    arrow(ax, right(head), left(out), lw=1.6)

    # ============================================================ ABLATION LADDER
    ax.text(W / 2, 2.95,
            "Ablation ladder  (one shared basis: masked manifest, leakage-safe graph-bound "
            "train, shared per-ticker scalers, identical val/test observations; seeds 42/123/2026)",
            ha="center", va="center", fontsize=9.0, fontweight="bold", color="#333333")

    rungs = [
        (2.15, "P0\nHAR classical\n(per-ticker OLS)", COLOR_LADDER, False),
        (5.55, "P1\n+ price-LSTM", COLOR_LADDER, False),
        (8.95, "P2\n+ news branch", COLOR_LADDER, False),
        (12.35, "P3\n+ per-ticker gate\n(graph OFF)", COLOR_LADDER, False),
        (16.05, "G1\n+ GAT / message\npassing", COLOR_G1, True),
    ]
    prev = None
    for cx, label, color, headline in rungs:
        b = box(ax, cx, 1.85, 2.9, 1.05, label, color, fontsize=7.8,
                bold=headline, edgecolor=EDGE_G1 if headline else EDGE,
                lw=2.2 if headline else 1.3)
        if prev is not None:
            arrow(ax, right(prev), left(b), lw=1.3)
        prev = b

    ax.text(W / 2, 0.85,
            "P3 = G1 with message passing disabled (graph-off determinism 0.0).   "
            "Graph effect G1 vs P3 = null at all horizons (DM n.s.)  ->  parsimony.",
            ha="center", va="center", fontsize=8.4, color="#555555")

    return fig


def main():
    fig = build()
    svg_path = OUT_DIR / "current_g1_architecture.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    print(f"wrote {svg_path}")
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    main()
