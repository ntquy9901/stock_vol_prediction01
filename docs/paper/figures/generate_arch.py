"""Technical model-architecture figure for the multi-market volatility model (matplotlib -> PNG + PDF).
Shows the per-stock-per-day feature vector x_{i,t} with explicit dimensions (own-history R^8, external R^4,
graph R^1), the cross-firm graph aggregation, and the gamma-loss gradient-boosting ensemble. GBM uses the R^8
own block; GBME adds the R^4 external (earnings) block; +graph adds the R^1 neighbour aggregate.

Layout is laid out on a wide canvas with non-overlapping bounding boxes: left column of input boxes (OHLCV,
graph W), a central feature-vector stack, dimension braces to its right, then the GBM box and the forecast box."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle  # noqa: E402

matplotlib.rcParams["svg.fonttype"] = "none"


def box(ax, x, y, w, h, t, fc, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                fc=fc, ec="black", lw=1.1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)


def arr(ax, x1, y1, x2, y2, txt=None, dy=0.18):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.2, color="0.25"))
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, txt, ha="center", fontsize=7.4, color="0.3")


def main():
    fig, ax = plt.subplots(figsize=(12.8, 7.0))
    ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 9)

    # ---- central feature-vector stack (13 rows: 8 own + 4 external + 1 graph) ----
    vx, vy, cw, ch = 4.5, 1.2, 3.5, 0.46
    own = ["har_daily", "har_weekly", "har_monthly",
           "mr_change", "mr_slope5", "mr_slope10", "mr_dev5", "mr_z22"]
    ext = ["earn_prox", "earn_soon", "earn_pre", "earn_post"]
    gr = [r"$g_{i,t}=\sum_j W_{ij}\,\mathrm{pk}_{j,t}$"]
    rows = gr + ext[::-1] + own[::-1]                       # graph at bottom, own on top
    for k, lab in enumerate(rows):
        fc = "#f6e9f2" if k < 1 else ("#fdf3e3" if k < 5 else "#e7f4ea")
        ax.add_patch(Rectangle((vx, vy + k * ch), cw, ch, fc=fc, ec="0.4", lw=0.7))
        ax.text(vx + 0.15, vy + k * ch + ch / 2, lab, ha="left", va="center", fontsize=7.0)
    top = vy + len(rows) * ch
    ax.text(vx + cw / 2, top + 0.28, r"feature vector  $x_{i,t}$", ha="center", fontsize=9.0)

    # ---- left input boxes (upper: OHLCV->pk ; lower: cross-firm graph W) ----
    box(ax, 0.3, 5.55, 2.5, 1.55,
        r"OHLCV$_{i,\cdot}$" + "\n(stock $i$)\n" + r"$\to\ \mathrm{pk}_{i,t}=\frac{\ln(H/L)^2}{4\ln 2}$",
        "#eaf2fb", 8.0)
    arr(ax, 2.8, 6.3, vx, top - 1.2)                       # OHLCV -> own-history rows (top of stack)

    box(ax, 0.3, 2.35, 3.15, 1.6,
        r"cross-firm graph  $W\!\in\!\mathbb{R}^{N\times N}$" + "\n" +
        r"top-$k$ corr$(\log\mathrm{pk})$," + "\nrow-normalised (train only)", "#f6e9f2", 7.6)
    arr(ax, 3.45, 2.75, vx, vy + ch / 2)                   # W -> graph feature row (bottom of stack)

    # ---- dimension braces to the right of the stack ----
    def brace(y0, y1, txt, col):
        xb = vx + cw + 0.15
        ax.annotate("", xy=(xb, y1), xytext=(xb, y0), arrowprops=dict(arrowstyle="-", color=col, lw=1.6))
        ax.text(xb + 0.15, (y0 + y1) / 2, txt, ha="left", va="center", fontsize=7.8, color=col)
    brace(vy, vy + ch, r"$\mathbb{R}^{1}$ graph", "#a03080")
    brace(vy + ch, vy + 5 * ch, r"$\mathbb{R}^{4}$ external (E)", "#b8791f")
    brace(vy + 5 * ch, vy + 13 * ch, r"$\mathbb{R}^{8}$ own-history", "#2f7d4a")

    # ---- GBM box ----
    box(ax, 9.7, 2.75, 3.2, 3.3,
        "Gamma-loss gradient\nboosting  (GBM)\n\n" + r"$F(x)=\sum_{m=1}^{300}\eta\,f_m(x)$" + "\n" +
        r"$f_m$: regression tree" + "\n(31 leaves, " + r"$\eta{=}0.05$)" + "\n\nloss: gamma deviance\n" +
        r"$=$ QLIKE $+$ const" + "\n(3-seed ensemble)", "#e9e9f5", 8.0)
    arr(ax, vx + cw + 1.45, vy + 6.5 * ch, 9.7, 4.4, r"$x_{i,t}$", dy=0.22)

    # ---- forecast box ----
    box(ax, 13.4, 3.6, 2.3, 1.5,
        r"$\hat{\sigma}^2_{i,t+h}$" + "\nParkinson\nvariance forecast\n" + r"$h\in\{1,5,10,22\}$", "#eaf2fb", 8.0)
    arr(ax, 12.9, 4.4, 13.4, 4.35)
    ax.text(14.55, 3.15, "trained per market:\nS&P 500  $\\cdot$  HOSE", ha="center", fontsize=7.4, color="0.35")

    # ---- top summary line ----
    ax.text(8.0, 8.5,
            r"GBM: $x\in\mathbb{R}^{8}$ (own)   $\cdot$   GBME: $\mathbb{R}^{12}$ (own$+$E)   $\cdot$   "
            r"+graph: $\mathbb{R}^{13}$ (own$+$E$+g$)", ha="center", fontsize=8.6, style="italic")

    for ext_ in ("png", "pdf"):
        fig.savefig(f"docs/paper/figures/fig_architecture.{ext_}", dpi=190, bbox_inches="tight")


if __name__ == "__main__":  # pragma: no cover - CLI entry (main() is covered by the test directly)
    main()
