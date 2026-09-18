"""Technical model-architecture figure for the multi-market volatility model (matplotlib -> PNG + PDF).
Shows the per-stock-per-day feature vector x_{i,t} = own-history R^8 (3 HAR lags + 5 log-vol momentum)
plus an OPTIONAL earnings block R^4, fed to a gamma-loss gradient-boosted tree ensemble (XGBoost) that
emits the per-stock multi-horizon Parkinson-variance forecast y_hat_{i,t+h}. A SEPARATE, optional
post-prediction stage then smooths those forecasts across each test day's cross-section over a
leaf-cooccurrence kNN graph built from the XGBoost trees' leaf indices; this is prediction smoothing,
not a GNN and not an input feature.

Layout: a VERTICAL top-to-bottom pipeline (OHLCV / earnings-calendar inputs at top -> feature-vector
boxes -> XGBoost -> per-stock forecast) with a dashed panel at the bottom holding the optional
leaf-graph smoothing stage (a small same-day cross-section glyph + the blend formula + final smoothed
forecast). The stacked-block orientation mirrors the reference design HTML
(baselines/2026-09-18_leaf_graph_paper/design/architecture_4variants.html)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle  # noqa: E402

matplotlib.rcParams["svg.fonttype"] = "none"


def box(ax, x, y, w, h, t, fc, fs=8.2, ls="solid", ec="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.1, ls=ls))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)


def arr(ax, x1, y1, x2, y2, txt=None, dy=0.18):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.2, color="0.25"))
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, txt, ha="center", fontsize=7.4, color="0.3")


def feature_rows():
    """Feature-vector rows shown in the stack, bottom-to-top: 4 optional external (earnings) then 8
    own-history. The cross-firm graph is NOT a feature row (the graph is a post-prediction step)."""
    own = ["har_daily", "har_weekly", "har_monthly",
           "mr_change", "mr_slope5", "mr_slope10", "mr_dev5", "mr_z22"]
    ext = ["earn_prox", "earn_soon", "earn_pre", "earn_post"]
    return ext[::-1] + own[::-1]


def graph_glyph(ax, cx, cy):
    """Small same-day cross-section: low-vol nodes wired by kNN leaf-cooccurrence edges plus a separate
    high-vol pair. Illustrates that the graph connects stocks the XGBoost trees route to shared leaves."""
    lo = [(cx - 0.60, cy + 0.12), (cx - 0.16, cy + 0.52), (cx + 0.22, cy + 0.08), (cx - 0.12, cy - 0.40)]
    hi = [(cx + 0.78, cy + 0.46), (cx + 0.98, cy - 0.10)]
    for a, b in [(0, 1), (0, 2), (1, 2), (0, 3)]:
        ax.plot([lo[a][0], lo[b][0]], [lo[a][1], lo[b][1]], color="#4aa3a0", lw=1.0, zorder=1)
    ax.plot([hi[0][0], hi[1][0]], [hi[0][1], hi[1][1]], color="#c98b3a", lw=1.0, zorder=1)
    for (x, y) in lo:
        ax.add_patch(Circle((x, y), 0.11, fc="#dff1ef", ec="#2f7d7a", lw=1.0, zorder=2))
    for (x, y) in hi:
        ax.add_patch(Circle((x, y), 0.11, fc="#fde7d2", ec="#b8791f", lw=1.0, zorder=2))


def _vjoin(labels):
    """Two lines of comma-separated feature names for a compact feature box."""
    half = (len(labels) + 1) // 2
    return ", ".join(labels[:half]) + ",\n" + ", ".join(labels[half:])


def main():
    # Vertical (portrait) canvas: flow runs top -> bottom. Equal unit scaling (10 wide / 12.5 tall
    # matches figsize 7.2 x 9.0) keeps the graph-glyph circles round.
    fig, ax = plt.subplots(figsize=(7.2, 9.0))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 12.5)

    rows = feature_rows()                              # earnings (reversed) then own-history (reversed)
    own = [r for r in rows if not r.startswith("earn_")]
    ext = [r for r in rows if r.startswith("earn_")]

    # ---- variant summary line (top) ----
    ax.text(5.0, 12.15,
            r"XGB: $x\in\mathbb{R}^{8}$ (own)   $\cdot$   XGB+E: $\mathbb{R}^{12}$ (own$+$E)"
            r"   $\cdot$   XGB+E+LG: $+$ leaf-graph smoothing",
            ha="center", fontsize=8.6, style="italic")

    # ---- two input sources at the top: OHLCV (own-history) and the earnings calendar (external) ----
    box(ax, 1.55, 10.55, 3.1, 1.2,
        r"OHLCV$_{i,\cdot}$ (stock $i$)" + "\n" + r"$\mathrm{pk}_{i,t}=\dfrac{\ln(H/L)^2}{4\ln 2}$",
        "#eaf2fb", 8.0)
    box(ax, 5.5, 10.62, 3.0, 1.06,
        "scheduled earnings\ncalendar (external)", "#fdf3e3", 7.8, ls=(0, (3, 2)), ec="#b8791f")

    # ---- feature-vector boxes (own-history R^8  +  optional earnings R^4) ----
    box(ax, 1.3, 8.45, 3.4, 1.65,
        r"own-history  $\mathbb{R}^{8}$" + "\n" + _vjoin(own), "#e7f4ea", 7.2, ec="#2f7d4a")
    box(ax, 5.3, 8.45, 3.4, 1.65,
        r"earnings  $\mathbb{R}^{4}$ (optional)" + "\n" + _vjoin(ext), "#fdf3e3", 7.4,
        ls=(0, (3, 2)), ec="#b8791f")
    arr(ax, 3.1, 10.55, 3.0, 10.12)                    # OHLCV -> own-history block
    arr(ax, 7.0, 10.62, 7.0, 10.12)                    # earnings calendar -> earnings block

    # ---- both feature blocks converge into the XGBoost box ----
    box(ax, 2.7, 6.15, 4.6, 1.9,
        "Gamma-loss gradient boosting  (XGBoost)\n\n" +
        r"$F(x)=\sum_{m=1}^{300}\eta\,f_m(x)$,   $f_m$: regression tree" + "\n" +
        r"(31 leaves, $\eta{=}0.05$)   $\cdot$   reg:gamma loss $=$ QLIKE$+$const" + "\n" +
        "3-seed ensemble   $\\cdot$   trained per market: S&P 500 $\\cdot$ HOSE", "#e9e9f5", 7.8)
    arr(ax, 3.0, 8.45, 4.35, 8.08, r"$x_{i,t}$", dy=0.20)   # own-history -> XGBoost
    arr(ax, 7.0, 8.45, 5.65, 8.08)                          # earnings -> XGBoost

    # ---- per-stock multi-horizon forecast ----
    box(ax, 3.15, 4.35, 3.7, 1.25,
        r"per-stock forecast  $\hat{\sigma}^2_{i,t+h}$" + "\n" + r"$h\in\{1,5,10,22\}$", "#eaf2fb", 8.0)
    arr(ax, 5.0, 6.15, 5.0, 5.60)                           # XGBoost -> forecast

    # ---- optional leaf-cooccurrence graph smoothing stage (dashed panel at the bottom) ----
    px, py, pw, ph = 0.7, 0.35, 8.6, 3.05
    ax.add_patch(FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0.03,rounding_size=0.10",
                                fc="#fbfbfd", ec="#7a6fae", lw=1.2, ls=(0, (4, 2))))
    ax.text(px + pw / 2, py + ph - 0.26, "leaf-cooccurrence graph smoothing", ha="center",
            fontsize=8.4, color="#4a417a", fontweight="bold")
    ax.text(px + pw / 2, py + ph - 0.55, "(optional post-prediction step)", ha="center",
            fontsize=7.2, color="#4a417a", style="italic")
    graph_glyph(ax, px + 1.85, py + 1.55)
    ax.text(px + 1.85, py + 0.55, "same-day cross-section;\nkNN on shared\nXGBoost leaves ($k{=}10$)",
            ha="center", fontsize=7.0, color="0.30")
    tx = px + 5.55
    ax.text(tx, py + 1.98,
            r"$\hat y^{\,sm}_i=(1-\alpha)\,\hat y_i+\alpha\,\overline{\hat y}_{\,\mathrm{kNN}(i)}$",
            ha="center", fontsize=9.2)
    ax.text(tx, py + 1.40, r"$\alpha\in[0,1]$ fit on validation ($\alpha{=}0\Rightarrow$ off)",
            ha="center", fontsize=7.2, color="0.30")
    ax.text(tx, py + 0.92, "prediction smoothing — not a GNN, not a feature",
            ha="center", fontsize=7.0, color="#8a2f2f", style="italic")
    ax.text(tx, py + 0.44, r"$\to\ \hat y^{\,sm}_{i,t+h}$: final smoothed forecast", ha="center",
            fontsize=7.8, color="0.25")
    arr(ax, 5.0, 4.35, 5.0, py + ph + 0.02, r"$\hat y_i$ (all stocks)", dy=0.18)   # forecast -> panel
    arr(ax, px + 3.15, py + 1.55, tx - 1.9, py + 1.75)      # glyph -> blend formula

    for ext_ in ("png", "pdf"):
        fig.savefig(f"docs/paper/figures/fig_architecture.{ext_}", dpi=190, bbox_inches="tight")


if __name__ == "__main__":  # pragma: no cover - CLI entry (main() is covered by the test directly)
    main()
