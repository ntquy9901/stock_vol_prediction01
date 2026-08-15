"""Generate publication-quality SVG architecture diagrams for the Track-A GAT model.

Plain matplotlib box-and-arrow rendering saved as .svg (no Graphviz/Mermaid).
Style matches docs/paper/diagrams/generate_diagrams.py (FancyBboxPatch + FancyArrowPatch).

Authoritative architecture source:
  baselines/2026-08-15_trackA_gat_edge/design/ARCHITECTURE_DETAILED.md

Run (venv):
  C:/luanvan/stock_vol_prediction01/.venv_gpu_encode/Scripts/python.exe \
      docs/paper/diagrams/generate_trackA_diagrams.py

Outputs:
  docs/paper/diagrams/trackA_gat_architecture.svg   (FIG 1: full model)
  docs/paper/diagrams/trackA_gat_ablation.svg       (FIG 2: leave-one-out ablation)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# Colour palette (matched to sibling generators)
COLOR_INPUT = "#e6e6e6"     # inputs / external data (grey)
COLOR_LSTM = "#dbe9f6"      # price-LSTM branch (blue)
COLOR_GAT = "#f6dfce"       # GAT graph branch (orange)
COLOR_RAW = "#f4c9a3"       # RAW-features-at-t node (darker orange, distinct)
COLOR_NEWS = "#e2d6f0"      # news branch (purple)
COLOR_GATE = "#ffe6b3"      # per-ticker gate (amber)
COLOR_HEAD = "#cfe8cf"      # concat + head (green)
COLOR_OUT = "#cdeccd"       # final output (green)
COLOR_HAR = "#e6e6e6"       # external HAR baseline (grey)
COLOR_REF = "#d9d9d9"       # external reference (grey)
COLOR_FULL = "#cfe8cf"      # FULL model (green)
COLOR_ABLATION = "#fdf3d0"  # retrained ablation variant (yellow)
COLOR_REMOVED = "#fbdcdc"   # removed component (red)
EDGE = "#333333"


def box(ax, cx, cy, w, h, text, facecolor="#ffffff", fontsize=9,
        edgecolor=EDGE, style="round,pad=0.02", dashed=False, fontweight="normal"):
    b = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=style, facecolor=facecolor, edgecolor=edgecolor,
        linewidth=1.4, linestyle="--" if dashed else "-", zorder=2,
    )
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            zorder=3, fontweight=fontweight)
    return (cx, cy, w, h)


def arrow(ax, p_from, p_to, label=None, style="-|>", color=EDGE, lw=1.3,
          curve=0.0, fontsize=8.0, dashed=False, label_dy=0.0):
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


def bottom(b):
    return (b[0], b[1] - b[3] / 2)


def top(b):
    return (b[0], b[1] + b[3] / 2)


def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    size = path.stat().st_size
    print(f"wrote {path}  ({size} bytes)")
    return path


# ---------------------------------------------------------------------------
# FIG 1 - full Track-A model
# ---------------------------------------------------------------------------
def diagram_full_model():
    fig, ax = new_fig(16, 10)

    # ---- INPUTS (left column) ----
    price_in = box(ax, 1.9, 8.4, 2.4, 1.0,
                   "price\n[B, N, seq=22, 5]", COLOR_INPUT)
    feats = box(ax, 1.9, 6.15, 2.9, 1.9,
                "5 node features (shared):\n"
                "1  pk_daily\n"
                "2  har_weekly\n"
                "3  har_monthly\n"
                "4  market_pk\n"
                "5  volume_zscore (22d)",
                "#f4f4f4", fontsize=8.3)
    adj_in = box(ax, 1.9, 4.15, 2.4, 1.0,
                 "adjacency [B, N, N]\ndirected vol->PK Top-5\n(TRAIN-only, frozen)",
                 COLOR_INPUT, fontsize=8.3)
    news_in = box(ax, 1.9, 2.5, 2.4, 1.0,
                  "news [B, N, 22, 146]\nnews_mask [B, N, 22]", COLOR_INPUT, fontsize=8.5)
    ticker_in = box(ax, 1.9, 0.9, 2.4, 0.8,
                    "ticker_ids [B, N]", COLOR_INPUT)

    # ---- BRANCH 1: Price-LSTM ----
    ax.text(5.0, 9.35, "BRANCH 1  -  Price-LSTM (temporal, per node)",
            ha="left", fontsize=9.5, fontweight="bold", color="#1f4e79")
    lstm = box(ax, 5.6, 8.4, 2.5, 1.1,
               "LSTM\n5 -> 64, 2 layers\ndropout 0.2, last step", COLOR_LSTM)
    hlstm = box(ax, 8.9, 8.4, 1.7, 0.9, "h_lstm\n[B, N, 64]", COLOR_LSTM)
    arrow(ax, right(price_in), left(lstm),
          label="reshape\n[B*N, 22, 5]", label_dy=0.15)
    arrow(ax, right(lstm), left(hlstm), label="[B,N,64]", label_dy=0.18)

    # ---- BRANCH 2: GAT graph ----
    ax.text(5.0, 6.9, "BRANCH 2  -  GAT graph (cross-sectional, RAW features @ t)",
            ha="left", fontsize=9.5, fontweight="bold", color="#8a4b1f")
    node_raw = box(ax, 5.6, 5.9, 2.6, 1.1,
                   "node_raw = price[:, :, -1, :]\nRAW features @ t\n(NOT h_lstm)   [B, N, 5]",
                   COLOR_RAW, fontsize=8.3, fontweight="bold")
    gat1 = box(ax, 8.6, 5.9, 1.75, 1.05,
               "GATLayer\n5 -> 64 x4 heads\nconcat [B,N,256]", COLOR_GAT, fontsize=8.0)
    gat2 = box(ax, 10.85, 5.9, 1.85, 1.05,
               "GATLayer\n256 -> 64 x4 heads\nh_gnn [B,N,256]", COLOR_GAT, fontsize=8.0)
    # price -> node_raw (RAW at last timestep), distinct routing
    arrow(ax, bottom(price_in), (node_raw[0] - 0.4, top(node_raw)[1]),
          label="last timestep", curve=-0.15, color="#8a4b1f", label_dy=0.1)
    arrow(ax, right(node_raw), left(gat1))
    arrow(ax, right(gat1), left(gat2))
    # adjacency masks both GAT layers
    arrow(ax, right(adj_in), (gat1[0] - 0.35, bottom(gat1)[1]),
          label="attention mask\n(adjacency>0)\nsoftmax over sources", curve=0.12,
          color="#8a4b1f", fontsize=7.6, label_dy=-0.05)
    arrow(ax, (adj_in[0] + 0.4, top(adj_in)[1]), (gat2[0] - 0.2, bottom(gat2)[1]),
          curve=0.28, color="#8a4b1f", dashed=True)

    # ---- BRANCH 3: News + per-ticker gate ----
    ax.text(5.0, 3.55, "BRANCH 3  -  News (per node) + per-ticker gate",
            ha="left", fontsize=9.5, fontweight="bold", color="#5a3b8a")
    news_lin = box(ax, 5.3, 2.5, 1.9, 1.0, "Linear\n146 -> 64\n+ ReLU", COLOR_NEWS, fontsize=8.3)
    news_lstm = box(ax, 7.5, 2.5, 1.9, 1.0, "News-LSTM\n64 -> 64, 2 layers\nlast step", COLOR_NEWS, fontsize=8.0)
    news_hidden = box(ax, 9.6, 2.5, 1.6, 0.9, "news_hidden\n[B, N, 64]", COLOR_NEWS, fontsize=8.3)
    gate = box(ax, 7.5, 0.85, 3.0, 0.9,
               "gate = sigmoid(gate_logits[ticker_ids])\nper-ticker scalar in (0, 1)",
               COLOR_GATE, fontsize=8.0)
    gated = box(ax, 11.7, 2.5, 1.7, 0.9,
                "gated_news\n= gate * news_hidden\n[B, N, 64]", COLOR_NEWS, fontsize=8.0)
    arrow(ax, right(news_in), left(news_lin), label="x news_mask", label_dy=0.15)
    arrow(ax, right(news_lin), left(news_lstm))
    arrow(ax, right(news_lstm), left(news_hidden))
    arrow(ax, right(news_hidden), left(gated))
    arrow(ax, right(ticker_in), (gate[0] - gate[2] / 2, gate[1]), curve=0.05)
    arrow(ax, top(gate), (gated[0], bottom(gated)[1]), curve=-0.25, color="#5a3b8a")

    # ---- HEAD (right column) ----
    concat = box(ax, 14.1, 5.9, 2.0, 2.0,
                 "Concat\n[ h_lstm 64,\n  h_gnn 256,\n  gated_news 64 ]\n[B, N, 384]",
                 COLOR_HEAD, fontsize=8.3, fontweight="bold")
    arrow(ax, right(hlstm), (concat[0], concat[1] + 0.7), curve=-0.18, color="#1f4e79")
    arrow(ax, right(gat2), left(concat), curve=0.0, color="#8a4b1f")
    arrow(ax, right(gated), (concat[0], concat[1] - 0.7), curve=0.18, color="#5a3b8a")

    head1 = box(ax, 14.1, 3.7, 2.4, 0.9,
                "Linear 384 -> 64\n+ ReLU + Dropout(0.2)", COLOR_HEAD, fontsize=8.3)
    head2 = box(ax, 14.1, 2.45, 2.4, 0.7, "Linear 64 -> 1  ->  raw [B, N]", COLOR_HEAD, fontsize=8.3)
    floor = box(ax, 14.1, 1.25, 2.7, 0.9,
                "per-ticker StandardScaler inverse\n+ softplus floor (eps=1e-6, >=0)",
                COLOR_HEAD, fontsize=7.8)
    out = box(ax, 14.1, 0.25, 2.7, 0.55,
              "Parkinson-variance forecast at t+h  [B, N]", COLOR_OUT, fontsize=8.3, fontweight="bold")
    arrow(ax, bottom(concat), top(head1))
    arrow(ax, bottom(head1), top(head2))
    arrow(ax, bottom(head2), top(floor))
    arrow(ax, bottom(floor), top(out))

    ax.set_title("Parallel Price-LSTM + directed vol->PK GAT + gated news -> fused head",
                 fontsize=12.5, pad=12)
    save(fig, "trackA_gat_architecture.svg")


# ---------------------------------------------------------------------------
# FIG 2 - leave-one-out ablation
# ---------------------------------------------------------------------------
def diagram_ablation():
    fig, ax = new_fig(15, 8.5)

    full = box(ax, 6.0, 7.2, 4.4, 1.1,
               "FULL\nLSTM + vol->PK GAT graph + news + per-ticker gate\n(retrained)",
               COLOR_FULL, fontsize=9.5, fontweight="bold")

    # Retrained leave-one-out variants (each = FULL minus exactly one component)
    y_var = 4.4
    minus_graph = box(ax, 2.0, y_var, 3.0, 1.5,
                      "minus_graph\nremove the WHOLE\nGAT branch\n(no node / edge / GAT)\nretrained",
                      COLOR_ABLATION, fontsize=8.5)
    minus_gate = box(ax, 5.5, y_var, 3.0, 1.5,
                     "minus_gate\ngate turned OFF\n(news always on)\ngraph + news kept\nretrained",
                     COLOR_ABLATION, fontsize=8.5)
    minus_news = box(ax, 9.0, y_var, 3.0, 1.5,
                     "minus_news\ndrop the news branch\n(gate is a no-op)\ngraph kept\nretrained",
                     COLOR_ABLATION, fontsize=8.5)

    # External references (not a leave-one-out of FULL)
    har = box(ax, 12.7, 5.55, 3.0, 1.15,
              "HAR (external)\npooled linear regression\nnot a variant of FULL",
              COLOR_HAR, fontsize=8.5, dashed=True)
    lstm_only = box(ax, 12.7, 3.55, 3.0, 1.15,
                    "LSTM-only (price-only)\nno graph, no news\nreference",
                    COLOR_REF, fontsize=8.5, dashed=True)

    # arrows FULL -> each retrained variant (solid: leave-one-out)
    arrow(ax, (full[0] - 1.4, bottom(full)[1]), top(minus_graph), curve=0.05)
    arrow(ax, bottom(full), top(minus_gate))
    arrow(ax, (full[0] + 1.4, bottom(full)[1]), top(minus_news), curve=-0.05)

    # dashed arrows FULL -> external references (comparison, not ablation)
    arrow(ax, right(full), left(har), curve=-0.1, dashed=True,
          label="external comparison", label_dy=0.25, fontsize=7.8)
    arrow(ax, (full[0] + 1.9, bottom(full)[1] + 0.1), left(lstm_only),
          curve=-0.25, dashed=True)

    # effect / significance annotation
    note = ("effect(X) = QLIKE(FULL) - QLIKE(minus_X)      (same test set, same positivity floor)\n"
            "negative  =>  removing X raised QLIKE  =>  X HELPED\n"
            "positive  =>  removing X lowered QLIKE  =>  X did not help\n"
            "Significance: Diebold-Mariano (HLN correction) on seed-ensembled test predictions,\n"
            "for FULL vs each minus_X and FULL vs HAR;  horizons h in {1, 5, 10, 22}.")
    box(ax, 6.2, 1.3, 11.4, 1.9, note, "#f4f4f4", fontsize=8.8)

    ax.set_title("Leave-one-out ablation: retrain FULL with exactly one component removed",
                 fontsize=12.5, pad=12)
    save(fig, "trackA_gat_ablation.svg")


if __name__ == "__main__":
    diagram_full_model()
    diagram_ablation()
    print("All Track-A diagrams generated.")
