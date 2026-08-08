"""Generate the CURRENT final combined Track B (pooled news-GNN) architecture SVG.

Plain matplotlib box-and-arrow rendering (no Graphviz/Mermaid dependency - neither is
installed in this environment and Mermaid does not render in the target viewer). This
mirrors docs/paper/diagrams/generate_diagrams.py.

Run: python docs/paper/diagrams/generate_final_combined.py

Every box/edge traces to the committed masked-gnn baseline source. Citations below use
    baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/<file>
read from git object 3665936 (the frozen graph-safe baseline); the knn/|corr|>0.7 sparse
adjacency option is a LATER commit (e319307) and is drawn dashed as "being added" because
the frozen baseline default is dense correlation.

Component -> source map (verified):
  * Per-ticker chronological 70/15/15 split ....... data.py chronological_split()
  * train-only scalers / winsor, Parkinson+HAR .... data.py build_ticker_samples(),
                                                    build_pooled_manifest() (transform_frame)
  * causal news, 15:00 Asia/Ho_Chi_Minh cutoff .... data.py effective_trading_date(),
                                                    attach_news() (news_mask)
  * pooled ASYNC manifest, sort (target_date,id) .. data.py build_pooled_manifest()
  * P1 Price LSTM 2-layer h=64 + head ............. models.py PooledPriceLSTM
  * P2 + News LSTM, late-fusion concat ............ models.py PooledPriceNewsLSTM.forward()
  * P3 gate sigmoid(gate_logits[ticker_id])*h_news  models.py PooledPriceNewsLSTM.forward()
  * masked per-day graph manifest, presence_mask .. data.py build_masked_graph_manifest()
  * correlation adjacency over PRESENT nodes ...... data.py _masked_correlation_adjacency()
  * frozen P3 encoders (no-grad, eval) ............ models.py GraphAblationModel.__init__/train
  * G0 (MP off) / G1 (masked message passing) ..... models.py GraphAblationModel.forward(),
                                                    _ResidualMessagePassing.forward()
  * graph-safe P3 checkpoint boundary/provenance .. models.py from_p3_checkpoint(),
                                                    run_pilot.py build_graph_safe_p3_checkpoint()
  * positivity floor eps*softplus(raw/eps)+eps .... models.py _apply_positivity (POSITIVITY_EPSILON=1e-6)
  * inverse target transform per ticker + 6 metrics train.py evaluate_records()
  * P0 HAR external baseline (per ticker) ......... run_pilot.py run_har_reference()
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Keep label text as real <text> nodes (not outlined paths) so the SVG stays
# searchable/selectable in a reviewer's viewer.
matplotlib.rcParams["svg.fonttype"] = "none"
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# Shared palette with generate_diagrams.py plus a few roles for this combined view.
COLOR_DATA = "#e6e6e6"        # raw data / preprocessing / external HAR baseline
COLOR_NEUTRAL = "#ffffff"     # neutral / shared encoder internals
COLOR_NEWS = "#cfe8cf"        # news branch (empirically helps QLIKE/RMSE)
COLOR_NULL = "#fdf3d0"        # ablated components with no measured lift (marked NULL)
COLOR_FROZEN = "#d9e8fb"      # frozen (no-grad) P3 encoders
COLOR_OUT = "#f2e6f7"         # output / metrics
COLOR_BARRIER = "#c0392b"     # leakage barrier (red dashed)
EDGE = "#333333"


def box(ax, cx, cy, w, h, text, facecolor=COLOR_NEUTRAL, fontsize=8.0,
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


def right(b):
    return (b[0] + b[2] / 2, b[1])


def left(b):
    return (b[0] - b[2] / 2, b[1])


def top(b):
    return (b[0], b[1] + b[3] / 2)


def bottom(b):
    return (b[0], b[1] - b[3] / 2)


def build():
    W, H = 18.0, 15.4
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ax.text(W / 2, H - 0.35,
            "Combined pooled news-GNN (Track B) - current architecture, as built",
            ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(W / 2, H - 0.85,
            "baselines/2026-08-08_pooled_news_gnn_ablation_baseline  (P0 HAR external "
            "+ pooled P1/P2/P3 + masked G0/G1)",
            ha="center", va="center", fontsize=8.5, color="#555555")

    # -- Raw inputs -------------------------------------------------------
    raw_price = box(ax, 3.3, 13.55, 3.6, 0.95,
                    "Raw per-ticker OHLCV\n(*_processed.csv, 33 tickers)", COLOR_DATA)
    raw_news = box(ax, 7.6, 13.55, 3.6, 0.95,
                   "Effective-date news panel\n(PCA features, .parquet + provenance)", COLOR_DATA)

    # -- Per-ticker causal preprocessing ---------------------------------
    prep = box(ax, 5.6, 11.9, 10.6, 1.55,
               "Per-ticker causal preprocessing  (train-only fit)\n"
               "chronological 70/15/15 split (per ticker, rows not reordered)  |  "
               "train-only scalers + winsor bounds\n"
               "Parkinson volatility + HAR (daily / weekly / monthly)  |  "
               "causal news: effective trading date,\n"
               "15:00 Asia/Ho_Chi_Minh close cutoff, missing-news mask",
               COLOR_DATA, fontsize=7.8)
    arrow(ax, bottom(raw_price), (3.3, 12.68))
    arrow(ax, bottom(raw_news), (7.6, 12.68))

    # -- Leakage barrier --------------------------------------------------
    yb = 10.95
    ax.plot([0.4, W - 0.4], [yb, yb], color=COLOR_BARRIER, lw=1.8, ls="--", zorder=1)
    ax.text(W / 2, yb + 0.17,
            "LEAKAGE BARRIER: split precedes HAR/scaler/winsor/window fit; "
            "news cut at the forecast origin; val/test are read-only transforms",
            ha="center", va="center", fontsize=7.6, color=COLOR_BARRIER,
            backgroundcolor="white", zorder=3)
    arrow(ax, bottom(prep), (5.6, yb + 0.02), lw=1.2)

    # -- Two manifests (DIFFERENT axes) ----------------------------------
    pooled_m = box(ax, 4.6, 10.05, 6.8, 1.25,
                   "Pooled per-ticker ASYNC manifest\n"
                   "sample = (ticker_id, target_date), each ticker's OWN trading days\n"
                   "~73k train samples, sorted by (target_date, ticker_id)",
                   COLOR_NEUTRAL, fontsize=7.8, bold=True)
    graph_m = box(ax, 13.3, 10.05, 8.2, 1.25,
                  "Masked common-date GRAPH manifest\n"
                  "per-DAY node set over fixed 33-ticker vocabulary; absent ticker\n"
                  "presence_mask=0 (never imputed).  4,941 dates / 4,523 train snapshots",
                  COLOR_NEUTRAL, fontsize=7.8, bold=True)
    arrow(ax, (5.6, yb - 0.02), top(pooled_m), lw=1.2)
    arrow(ax, (5.6, yb - 0.02), (13.3, top(graph_m)[1]), curve=-0.05, lw=1.2)
    ax.text(9.0, 10.05, "DIFFERENT\nsample axes\n(no split mixing)",
            ha="center", va="center", fontsize=7.2, color=COLOR_BARRIER,
            fontstyle="italic", zorder=3)

    # -- P0 HAR external baseline (own left lane) ------------------------
    har = box(ax, 1.05, 6.9, 1.8, 2.3,
              "P0: HAR\nLinear Reg.\n(Corsi 2009)\n\nEXTERNAL\nbaseline,\nper ticker\n"
              "(no DL / no\nnews / no\ncross-ticker)",
              COLOR_DATA, fontsize=7.0)
    arrow(ax, left(pooled_m), top(har), curve=0.15,
          label="last-step HAR\nfeatures", fontsize=6.6, label_dy=0.2)

    # -- LEFT: pooled shared-weight encoders (P1/P2/P3) ------------------
    box(ax, 5.95, 6.65, 7.0, 4.35, "", COLOR_NEUTRAL, edgecolor="#888888")
    ax.text(5.95, 8.55,
            "Pooled shared-weight encoders  (ONE weight set for all tickers)  ->  "
            "P1 / P2 / P3",
            ha="center", va="center", fontsize=8.0, fontweight="bold")

    xprice = box(ax, 3.35, 7.75, 1.7, 0.85, "x_price\n[22, Fp]\nHAR+Parkinson", COLOR_DATA, 7.2)
    price_lstm = box(ax, 5.5, 7.75, 2.1, 0.9, "Price LSTM\n(shared, 2-layer,\nh=64)", COLOR_NEUTRAL, 7.4)
    xnews = box(ax, 3.35, 6.35, 1.7, 0.9, "x_news[22,Fn]\n+ news_mask\n(causal)", COLOR_NEWS, 7.2)
    news_lstm = box(ax, 5.5, 6.35, 2.1, 0.9, "News LSTM\n(shared, 2-layer,\nh=64)", COLOR_NEWS, 7.4)
    gate = box(ax, 7.65, 6.35, 1.5, 0.95,
               "gate x h_news\nsigmoid(gate_\nlogits[ticker])\n[NULL]", COLOR_NULL, 6.7)
    concat = box(ax, 7.95, 7.6, 1.3, 0.8, "concat\n(h_price,\nh_news)", COLOR_NEUTRAL, 7.2)
    head = box(ax, 7.95, 5.35, 1.55, 0.8, "Head\nLinear-ReLU-\nDropout-Linear", COLOR_NEUTRAL, 7.0)

    arrow(ax, right(xprice), left(price_lstm))
    arrow(ax, (right(price_lstm)[0], 7.75), (concat[0] - concat[2] / 2, concat[1] + 0.02),
          label="h_price", fontsize=6.6, curve=0.05)
    arrow(ax, right(xnews), left(news_lstm))
    arrow(ax, right(news_lstm), left(gate), label="h_news", fontsize=6.4, label_dy=0.18)
    arrow(ax, top(gate), (concat[0] - 0.1, concat[1] - concat[3] / 2), curve=-0.15,
          label="P3 only", fontsize=6.3, label_dy=0.05)
    arrow(ax, bottom(concat), top(head))

    ax.text(5.95, 4.72,
            "P1 = price LSTM + head (no news)   |   P2 = + News LSTM, late-fusion concat   |   "
            "P3 = + per-ticker gate",
            ha="center", va="center", fontsize=6.9, color="#444444", fontstyle="italic")

    # -- RIGHT: masked availability-aware GNN (G0/G1) --------------------
    box(ax, 13.6, 6.65, 8.5, 4.35, "", COLOR_NEUTRAL, edgecolor="#888888")
    ax.text(13.6, 8.55,
            "Masked availability-aware GNN  (frozen P3 encoders)  ->  G0 / G1",
            ha="center", va="center", fontsize=8.0, fontweight="bold")

    frozen = box(ax, 10.9, 7.75, 2.4, 1.1,
                 "Frozen P3 encoders\n(no-grad, eval)\nprice + news + gate", COLOR_FROZEN, 7.4)
    hbase = box(ax, 13.5, 7.75, 1.35, 0.8, "H_base\nper node", COLOR_NEUTRAL, 7.4)
    adj = box(ax, 11.3, 5.95, 3.15, 1.25,
              "Correlation adjacency\nover PRESENT nodes only\n"
              "(dense; sparse knn top-8 /\n|corr|>0.7 being added)", COLOR_NULL, 7.0)
    presence = box(ax, 13.95, 6.25, 1.95, 0.9,
                   "presence_mask\nabsent = masked,\nnever imputed", COLOR_NEUTRAL, 7.0)
    mp = box(ax, 16.45, 7.35, 2.05, 1.2,
             "G1: masked\nmessage passing\n(absent nodes give/\nget nothing)\n[NULL]", COLOR_NULL, 6.8)
    ghead = box(ax, 16.1, 5.35, 1.8, 0.8, "Graph head\n(shared P3 head)", COLOR_NEUTRAL, 7.2)

    arrow(ax, right(frozen), left(hbase))
    arrow(ax, bottom(hbase), top(ghead), curve=-0.15)
    ax.text(15.35, 6.72, "G0: MP OFF\n(bypass)", ha="center", va="center",
            fontsize=6.3, color=EDGE, backgroundcolor="white", zorder=3)
    arrow(ax, right(hbase), left(mp), curve=0.1, label="G1", fontsize=6.5)
    arrow(ax, top(adj), (13.5, hbase[1] - hbase[3] / 2 - 0.02), curve=-0.1)
    arrow(ax, right(adj), (mp[0] - mp[2] / 2, mp[1] - 0.25), curve=0.1)
    arrow(ax, top(presence), (mp[0] - mp[2] / 2 + 0.15, mp[1] - 0.4), curve=0.1)
    arrow(ax, bottom(mp), (ghead[0] + 0.3, ghead[1] + ghead[3] / 2), curve=0.2,
          label="+ residual", fontsize=6.4)

    # graph-safe P3 checkpoint edge (left P3 head -> frozen encoders on right)
    arrow(ax, (head[0] + head[2] / 2, head[1] + 0.15), left(frozen), curve=0.25,
          color="#1f6fb2", lw=1.6)
    ax.text(10.9, 6.95,
            "graph-safe P3 checkpoint (only permitted init;\n"
            "provenance + train targets <= train_end_date)",
            ha="center", va="center", fontsize=6.4, color="#1f6fb2",
            backgroundcolor="white", zorder=3)

    # -- Output band ------------------------------------------------------
    posfloor = box(ax, 12.9, 3.35, 3.5, 1.0,
                   "Positivity floor  (G0/G1)\neps*softplus(raw/eps)+eps\n(eps=1e-6, denorm scale)",
                   COLOR_OUT, 7.2)
    inv = box(ax, 9.0, 3.35, 3.1, 1.0,
              "Inverse target transform\nper ticker (z*std + mean)", COLOR_OUT, 7.4)
    metrics = box(ax, 9.0, 1.95, 7.4, 0.9,
                  "6 metrics (raw scale): MSE, RMSE, MAE, R2, QLIKE, DirAcc\n"
                  "(DirAcc grouped per ticker, sorted by target_date)", COLOR_OUT, 7.6, bold=True)

    arrow(ax, bottom(head), (inv[0] + 0.6, inv[1] + inv[3] / 2), curve=-0.08,
          label="pooled y_hat_norm", fontsize=6.5)
    arrow(ax, bottom(ghead), (posfloor[0] + 0.4, posfloor[1] + posfloor[3] / 2), curve=0.1,
          label="graph y_hat_norm", fontsize=6.5)
    arrow(ax, left(posfloor), right(inv))
    arrow(ax, bottom(inv), top(metrics))
    arrow(ax, bottom(har), (2.2, 2.15), curve=-0.05)
    arrow(ax, (2.2, 2.15), (metrics[0] - metrics[2] / 2, 2.0),
          label="P0 -> inverse transform -> metrics", fontsize=6.3)

    # -- Legend (single strip) -------------------------------------------
    ly = 1.05
    swatches = [
        (COLOR_DATA, "data / HAR baseline", 0.7),
        (COLOR_NEWS, "news (helps QLIKE/RMSE)", 4.0),
        (COLOR_NULL, "ablated [NULL]", 8.1),
        (COLOR_FROZEN, "frozen P3", 10.9),
        (COLOR_OUT, "output / metrics", 13.2),
    ]
    for c, lab, x in swatches:
        r = FancyBboxPatch((x, ly - 0.13), 0.3, 0.26, boxstyle="round,pad=0.01",
                           facecolor=c, edgecolor=EDGE, linewidth=1.0, zorder=2)
        ax.add_patch(r)
        ax.text(x + 0.4, ly, lab, ha="left", va="center", fontsize=7.0, zorder=3)
    ax.plot([15.8, 16.1], [ly, ly], color=COLOR_BARRIER, lw=1.8, ls="--", zorder=2)
    ax.text(16.2, ly, "leakage barrier", ha="left", va="center", fontsize=7.0, zorder=3)

    # -- Caption (manual line breaks to avoid overlap) -------------------
    cap_lines = [
        "HAR (P0) is an external classical baseline (per ticker; no deep learning, no news, no "
        "cross-ticker information).  Ablation-suite findings (n=3 seeds):",
        "the news branch improves QLIKE/RMSE with statistical significance; cross-stock graph "
        "message passing (G1 vs G0), the per-ticker gate (P3 vs P2), and pooling vs common-date "
        "training show no significant lift [NULL].",
        "The masked availability-aware graph (4,941 dates vs 1,296 intersection) removes the "
        "data-scarcity confound and the graph-null persists.  Sparse knn / |corr|>0.7 adjacency "
        "is a later option (dashed).",
    ]
    for i, line in enumerate(cap_lines):
        ax.text(9.0, 0.62 - i * 0.22, line, ha="center", va="center",
                fontsize=6.7, color="#333333")

    return fig


def main():
    fig = build()
    svg_path = OUT_DIR / "final_combined_architecture.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    print(f"wrote {svg_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
