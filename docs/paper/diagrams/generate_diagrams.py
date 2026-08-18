"""Generate SVG architecture diagrams for docs/paper/architecture_diagrams_review.md.

Plain matplotlib box-and-arrow rendering (no Graphviz/Mermaid dependency - neither
is installed in this environment, and Mermaid code blocks don't render in the
target viewer). Run: python docs/paper/diagrams/generate_diagrams.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

COLOR_FULL = "#cfe8cf"
COLOR_ABLATION = "#fdf3d0"
COLOR_HAR = "#e6e6e6"
COLOR_REMOVED = "#fbdcdc"
EDGE = "#333333"


def box(ax, cx, cy, w, h, text, facecolor="#ffffff", fontsize=10, edgecolor=EDGE, style="round,pad=0.02", dashed=False):
    b = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=style, facecolor=facecolor, edgecolor=edgecolor,
        linewidth=1.4, linestyle="--" if dashed else "-", zorder=2,
    )
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, zorder=3, wrap=True)
    return (cx, cy, w, h)


def arrow(ax, p_from, p_to, label=None, style="-|>", color=EDGE, lw=1.3, curve=0.0, fontsize=8.5, dashed=False):
    a = FancyArrowPatch(
        p_from, p_to, arrowstyle=style, mutation_scale=14,
        connectionstyle=f"arc3,rad={curve}", color=color, linewidth=lw,
        linestyle="--" if dashed else "-", zorder=1,
    )
    ax.add_patch(a)
    if label:
        mx, my = (p_from[0] + p_to[0]) / 2, (p_from[1] + p_to[1]) / 2 + curve * 0.6
        ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize,
                color=color, zorder=3, backgroundcolor="white")


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
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# 0. Overview
# ---------------------------------------------------------------------------
def diagram_overview():
    fig, ax = new_fig(11, 6.5)

    har = box(ax, 1.6, 5.2, 2.6, 1.0, "(0) HAR co dien\nLinear Regression", COLOR_HAR)
    bb = box(ax, 1.6, 1.2, 2.8, 1.1, "(1) Price-only backbone\nLSTM + GAT, KHONG tin tuc", COLOR_ABLATION)
    full = box(ax, 6.0, 3.2, 3.0, 1.3, "(FULL) News-fusion\n+ Graph + Gate\n(model chinh cua paper)", COLOR_FULL)
    ng = box(ax, 9.8, 5.2, 2.6, 1.1, "(2) No-Graph ablation\ndo thi = identity", COLOR_ABLATION)
    ngate = box(ax, 9.8, 1.2, 2.6, 1.1, "(3) No-Gate ablation\nbo gate per-ticker", COLOR_ABLATION)

    arrow(ax, (2.9, 4.7), (4.6, 3.6), label="So voi (1)/FULL:\ndeep learning co thang\nHAR khong?", curve=0.15)
    arrow(ax, (2.9, 1.6), (4.6, 2.8), label="Ablation 1: them tin tuc\nco giup gi khong?", curve=-0.1)
    arrow(ax, (7.4, 3.7), (8.6, 4.8), label="Ablation 2: bo do thi", curve=0.15)
    arrow(ax, (7.4, 2.7), (8.6, 1.6), label="Ablation 3: bo gate", curve=-0.15)

    ax.set_title("Ban do tong quan: 5 model, moi model tra loi 1 cau hoi", fontsize=12, pad=14)
    save(fig, "00_overview.svg")


# ---------------------------------------------------------------------------
# 1. HAR baseline
# ---------------------------------------------------------------------------
def diagram_har():
    fig, ax = new_fig(10, 3.4)

    v = box(ax, 1.1, 1.7, 1.8, 1.1, "parkinson_\nvolatility[t]\n(1 ma, 1 chuoi)", COLOR_HAR)
    d = box(ax, 3.4, 2.7, 1.7, 0.8, "har_daily\n(rolling 1 ngay)", "#ffffff")
    w = box(ax, 3.4, 1.7, 1.7, 0.8, "har_weekly\n(rolling 5 ngay)", "#ffffff")
    m = box(ax, 3.4, 0.7, 1.7, 0.8, "har_monthly\n(rolling 22 ngay)", "#ffffff")
    lr = box(ax, 6.2, 1.7, 2.4, 1.2, "Linear Regression\ny = b0+b1*d+b2*w+b3*m", COLOR_HAR)
    out = box(ax, 8.9, 1.7, 1.6, 1.0, "target_5d\n(du bao)", "#ffffff")

    for src in (d, w, m):
        arrow(ax, (v[0] + v[2] / 2, v[1]), (src[0] - src[2] / 2, src[1]))
    for src in (d, w, m):
        arrow(ax, (src[0] + src[2] / 2, src[1]), (lr[0] - lr[2] / 2, lr[1]))
    arrow(ax, (lr[0] + lr[2] / 2, lr[1]), (out[0] - out[2] / 2, out[1]))

    ax.set_title("(0) HAR co dien - baseline chinh, doi chung ngoai kien truc", fontsize=11, pad=10)
    save(fig, "01_har_baseline.svg")


# ---------------------------------------------------------------------------
# 2. Price-only backbone (ParallelLSTMGNN)
# ---------------------------------------------------------------------------
def diagram_backbone():
    fig, ax = new_fig(10, 5.2)

    x = box(ax, 1.3, 3.9, 2.0, 1.0, "x: HAR feature\n[B,22,N,3]", COLOR_HAR)
    adj = box(ax, 1.3, 1.6, 2.0, 1.0, "adj_matrix\ndo thi k-NN", COLOR_HAR)

    lstm = box(ax, 4.3, 3.9, 2.2, 1.1, "LSTM stream\n(moi ma 1 LSTM rieng)", "#ffffff")
    gat = box(ax, 4.3, 1.6, 2.2, 1.2, "GAT stream\n(2 lop Graph Attention,\ntron qua adj_matrix)", "#ffffff")

    hlstm = box(ax, 6.9, 3.9, 1.6, 0.8, "h_lstm\n[B,N,64]", "#ffffff")
    hgnn = box(ax, 6.9, 1.6, 1.6, 0.8, "h_gnn\n[B,N,256]", "#ffffff")

    cat = box(ax, 8.9, 2.75, 1.4, 1.4, "Concat\n[320]", COLOR_FULL)
    mlp = box(ax, 8.9, 0.7, 1.4, 0.8, "MLP\n320->64->32->1", "#ffffff")

    arrow(ax, (x[0] + x[2] / 2, x[1]), (lstm[0] - lstm[2] / 2, lstm[1]))
    arrow(ax, (adj[0] + adj[2] / 2, adj[1]), (gat[0] - gat[2] / 2, gat[1]))
    arrow(ax, (x[0] + 1.0, x[1] - 1.9), (gat[0] - gat[2] / 2, gat[1] + 0.3), curve=0.1)
    arrow(ax, (lstm[0] + lstm[2] / 2, lstm[1]), (hlstm[0] - hlstm[2] / 2, hlstm[1]))
    arrow(ax, (gat[0] + gat[2] / 2, gat[1]), (hgnn[0] - hgnn[2] / 2, hgnn[1]))
    arrow(ax, (hlstm[0] + hlstm[2] / 2, hlstm[1]), (cat[0], cat[1] + 0.4), curve=0.1)
    arrow(ax, (hgnn[0] + hgnn[2] / 2, hgnn[1]), (cat[0], cat[1] - 0.4), curve=-0.1)
    arrow(ax, (cat[0], cat[1] - cat[3] / 2), (mlp[0], mlp[1] + mlp[3] / 2))

    ax.text(8.9, -0.35, "Du bao volatility [B,N]", ha="center", fontsize=10)
    arrow(ax, (mlp[0], mlp[1] - mlp[3] / 2), (8.9, -0.15))

    ax.set_title("(1) Price-only backbone (ParallelLSTMGNN) - diem neo cho Ablation 1", fontsize=11, pad=10)
    save(fig, "02_backbone.svg")


# ---------------------------------------------------------------------------
# 3. FULL model
# ---------------------------------------------------------------------------
def diagram_full(no_gate=False):
    fig, ax = new_fig(11, 6.0)

    xh = box(ax, 1.1, 5.0, 1.7, 0.9, "x_har\n[B,22,N,3]", COLOR_HAR)
    adj = box(ax, 1.1, 3.6, 1.7, 0.9, "adj\ndo thi k-NN", COLOR_HAR)
    xn = box(ax, 1.1, 1.6, 1.9, 1.0, "x_news\n[B,22,N,146]\nPCAx2+EWMA(30d)", COLOR_HAR)

    bb = box(ax, 3.9, 4.3, 2.3, 1.3, "ParallelLSTMGNN\n.get_embeddings()\n(tai su dung backbone)", "#ffffff")
    newsb = box(ax, 3.9, 1.6, 2.2, 1.1, "NewsFeatureLSTM\nLinear(146->64)+LSTM", "#ffffff")

    hlstm = box(ax, 6.4, 4.9, 1.4, 0.7, "h_lstm[64]", "#ffffff", fontsize=9)
    hgnn = box(ax, 6.4, 3.9, 1.4, 0.7, "h_gnn[256]", "#ffffff", fontsize=9)
    nrep = box(ax, 6.4, 1.6, 1.5, 0.7, "news_rep[64]", "#ffffff", fontsize=9)

    arrow(ax, (xh[0] + xh[2] / 2, xh[1]), (bb[0] - bb[2] / 2, bb[1] + 0.3))
    arrow(ax, (adj[0] + adj[2] / 2, adj[1]), (bb[0] - bb[2] / 2, bb[1] - 0.3))
    arrow(ax, (xn[0] + xn[2] / 2, xn[1]), (newsb[0] - newsb[2] / 2, newsb[1]))
    arrow(ax, (bb[0] + bb[2] / 2, bb[1] + 0.3), (hlstm[0] - hlstm[2] / 2, hlstm[1]))
    arrow(ax, (bb[0] + bb[2] / 2, bb[1] - 0.3), (hgnn[0] - hgnn[2] / 2, hgnn[1]))
    arrow(ax, (newsb[0] + newsb[2] / 2, newsb[1]), (nrep[0] - nrep[2] / 2, nrep[1]))

    if not no_gate:
        gate = box(ax, 3.9, 0.4, 2.3, 0.7, "gate_logits[N]\n1 so hoc duoc / ma", "#ffe6b3", fontsize=9)
        arrow(ax, (gate[0] + 0.6, gate[1] + gate[3] / 2), (nrep[0] - 0.3, nrep[1] - nrep[3] / 2),
              label="sigmoid(.) nhan\ntheo tung ma", curve=0.15)
        gnews = box(ax, 8.6, 1.6, 1.6, 0.7, "gated_news\n= gate*news_rep", "#ffffff", fontsize=8.5)
        arrow(ax, (nrep[0] + nrep[2] / 2, nrep[1]), (gnews[0] - gnews[2] / 2, gnews[1]))
        cat_input3 = gnews
    else:
        removed = box(ax, 3.9, 0.4, 2.5, 0.7, "gate_logits / sigmoid\nDA BI BO so voi FULL", COLOR_REMOVED,
                       fontsize=9, dashed=True)
        gnews = box(ax, 8.6, 1.6, 1.7, 0.7, "news_rep DUNG THANG\n(khong nhan gate)", "#ffffff", fontsize=8.5)
        arrow(ax, (nrep[0] + nrep[2] / 2, nrep[1]), (gnews[0] - gnews[2] / 2, gnews[1]))
        cat_input3 = gnews

    cat = box(ax, 9.9, 3.6, 1.3, 1.6, "Concat\n[384]", COLOR_FULL, fontsize=9)
    arrow(ax, (hlstm[0] + hlstm[2] / 2, hlstm[1]), (cat[0] - 0.3, cat[1] + 0.6), curve=0.15)
    arrow(ax, (hgnn[0] + hgnn[2] / 2, hgnn[1]), (cat[0] - 0.3, cat[1]), curve=0.05)
    arrow(ax, (cat_input3[0] + cat_input3[2] / 2, cat_input3[1]), (cat[0] - 0.3, cat[1] - 0.6), curve=-0.15)

    mlp = box(ax, 9.9, 1.4, 1.3, 0.8, "MLP\n384->64->32->1", "#ffffff", fontsize=8.5)
    arrow(ax, (cat[0], cat[1] - cat[3] / 2), (mlp[0], mlp[1] + mlp[3] / 2))
    ax.text(9.9, 0.15, "Du bao [B,N]", ha="center", fontsize=9)
    arrow(ax, (mlp[0], mlp[1] - mlp[3] / 2), (9.9, 0.35))

    title = "(3) Ablation 3: No-Gate (tin tuc luon bat 100%)" if no_gate else \
        "(FULL) News-fusion + Graph + Gate - model chinh cua paper"
    ax.set_title(title, fontsize=11, pad=10)
    save(fig, "05_ablation3_nogate.svg" if no_gate else "03_full_model.svg")


# ---------------------------------------------------------------------------
# 4. Ablation 2 - no graph
# ---------------------------------------------------------------------------
def diagram_no_graph():
    fig, ax = new_fig(10, 4.6)

    ax.text(2.4, 4.15, "Do thi THAT (k-NN)", ha="center", fontsize=11, fontweight="bold")
    a = box(ax, 1.2, 2.9, 1.1, 0.7, "ma A", "#ffffff")
    b = box(ax, 3.2, 3.4, 1.1, 0.7, "ma B", "#ffffff")
    c = box(ax, 3.2, 2.2, 1.1, 0.7, "ma C", "#ffffff")
    arrow(ax, (a[0]+0.4, a[1]+0.2), (b[0]-0.4, b[1]-0.1), style="-")
    arrow(ax, (a[0]+0.4, a[1]-0.2), (c[0]-0.4, c[1]+0.1), style="-")
    arrow(ax, (b[0], b[1]-0.35), (c[0], c[1]+0.35), style="-")
    ax.text(2.4, 1.15, "Moi ma 'nhin thay' va tron thong tin\ntu ma khac (qua GAT attention)",
            ha="center", fontsize=8.5)

    ax.plot([5.0, 5.0], [0.5, 4.4], color=EDGE, linestyle=":", linewidth=1)
    ax.annotate("", xy=(5.0, 2.45), xytext=(4.5, 2.45),
                arrowprops=dict(arrowstyle="-|>", color=EDGE))
    ax.text(4.75, 2.75, "Ablation:\nadj -> eye(N)", ha="center", fontsize=8, rotation=0)

    ax.text(7.6, 4.15, "Do thi THAY THE (identity)", ha="center", fontsize=11, fontweight="bold")
    a2 = box(ax, 6.4, 2.9, 1.1, 0.7, "ma A", COLOR_ABLATION)
    b2 = box(ax, 8.4, 3.4, 1.1, 0.7, "ma B", COLOR_ABLATION)
    c2 = box(ax, 8.4, 2.2, 1.1, 0.7, "ma C", COLOR_ABLATION)
    # self loops only - small circular arrows
    for node in (a2, b2, c2):
        ax.annotate("", xy=(node[0]-0.35, node[1]+0.35), xytext=(node[0]+0.35, node[1]+0.35),
                    arrowprops=dict(arrowstyle="-|>", color=EDGE, connectionstyle="arc3,rad=1.4"))
    ax.text(7.6, 1.15, "Moi ma CHI tu nhin chinh no (self-loop) -\nGAT suy bien thanh bien doi rieng tung ma,\nKHONG con truyen thong tin cheo-ma",
            ha="center", fontsize=8.5)

    ax.set_title("(2) Ablation 2: No-Graph - thay adj_matrix bang identity, giu nguyen kien truc/so tham so",
                 fontsize=10.5, pad=10)
    save(fig, "04_ablation2_nograph.svg")


if __name__ == "__main__":
    diagram_overview()
    diagram_har()
    diagram_backbone()
    diagram_full(no_gate=False)
    diagram_no_graph()
    diagram_full(no_gate=True)
    print("All diagrams generated.")
