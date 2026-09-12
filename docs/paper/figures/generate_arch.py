"""Technical model-architecture figure for the multi-market volatility model (matplotlib -> PNG + PDF).
Shows the per-stock-per-day feature vector x_{i,t} with explicit dimensions (own-history R^9, external R^4,
graph R^1), the cross-firm graph aggregation, and the gamma-loss gradient-boosting ensemble. GBM uses the R^9
own block; GBME adds the R^4 external (earnings) block; +graph adds the R^1 neighbour aggregate."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle  # noqa: E402

matplotlib.rcParams["svg.fonttype"] = "none"


def box(ax, x, y, w, h, t, fc, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec="black", lw=1.1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)


def arr(ax, x1, y1, x2, y2, txt=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.2, color="0.25"))
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, txt, ha="center", fontsize=7.4, color="0.3")


def main():
    fig, ax = plt.subplots(figsize=(11.4, 6.1))
    ax.axis("off"); ax.set_xlim(0, 15); ax.set_ylim(0, 8)
    box(ax, 0.15, 3.2, 1.85, 1.7,
        r"OHLCV$_{i,\cdot}$" + "\n(stock $i$)\n" + r"$\to\ \mathrm{pk}_{i,t}$" + "\n" + r"$=\frac{\ln(H/L)^2}{4\ln 2}$",
        "#eaf2fb", 8.0)
    vx, vy, cw, ch = 2.7, 0.7, 3.35, 0.5
    own = ["har_daily", "har_weekly", "har_monthly", "rq  (realized quarticity)",
           "mr_change", "mr_slope5", "mr_slope10", "mr_dev5", "mr_z22"]
    ext = ["earn_prox", "earn_soon", "earn_pre", "earn_post"]
    gr = [r"$g_{i,t}=\sum_j W_{ij}\,\mathrm{pk}_{j,t}$"]
    rows = gr + ext[::-1] + own[::-1]
    for k, lab in enumerate(rows):
        fc = "#f6e9f2" if k < 1 else ("#fdf3e3" if k < 5 else "#e7f4ea")
        ax.add_patch(Rectangle((vx, vy + k * ch), cw, ch, fc=fc, ec="0.4", lw=0.7))
        ax.text(vx + 0.12, vy + k * ch + ch / 2, lab, ha="left", va="center", fontsize=6.9)
    ax.text(vx + cw / 2, vy + len(rows) * ch + 0.3, r"feature vector  $x_{i,t}$", ha="center", fontsize=8.8)

    def brace(y0, y1, txt, col):
        xb = vx + cw + 0.12
        ax.annotate("", xy=(xb, y1), xytext=(xb, y0), arrowprops=dict(arrowstyle="-", color=col, lw=1.5))
        ax.text(xb + 0.12, (y0 + y1) / 2, txt, ha="left", va="center", fontsize=7.6, color=col)
    brace(vy, vy + ch, r"$\mathbb{R}^{1}$ graph", "#a03080")
    brace(vy + ch, vy + 5 * ch, r"$\mathbb{R}^{4}$ external (E)", "#b8791f")
    brace(vy + 5 * ch, vy + 10 * ch, r"$\mathbb{R}^{9}$ own-history", "#2f7d4a")

    box(ax, 2.55, 6.25, 3.65, 1.35,
        r"cross-firm graph  $W\in\mathbb{R}^{N\times N}$" + "\n" +
        r"$W=\mathrm{top}\text{-}k\,(\mathrm{corr}(\log\mathrm{pk}))$, row-norm (train)", "#f6e9f2", 7.6)
    arr(ax, 4.3, 6.25, 4.3, vy + ch + 0.35)

    box(ax, 6.85, 2.25, 3.15, 3.5,
        "Gamma-loss gradient\nboosting  (GBM)\n\n" + r"$F(x)=\sum_{m=1}^{300}\eta\,f_m(x)$" + "\n" +
        r"$f_m$: regression tree" + "\n(31 leaves, " + r"$\eta{=}0.05$)" + "\n\nloss: gamma deviance\n" +
        r"$=$ QLIKE $+$ const" + "\n(3-seed ensemble)", "#e9e9f5", 8.0)
    arr(ax, vx + cw + 1.35, vy + 5 * ch, 6.85, 4.0, r"$x_{i,t}$")

    box(ax, 10.7, 3.35, 2.15, 1.35,
        r"$\hat{\sigma}^2_{i,t+h}$" + "\nParkinson\nvariance forecast\n" + r"$h\in\{1,5,10,22\}$", "#eaf2fb", 8.0)
    arr(ax, 10.0, 4.0, 10.7, 4.02)

    ax.text(7.6, 7.35,
            r"GBM: $x\in\mathbb{R}^{9}$ (own)   $\cdot$   GBME: $\mathbb{R}^{13}$ (own$+$E)   $\cdot$   "
            r"+graph: $\mathbb{R}^{14}$ (own$+$E$+g$)", ha="center", fontsize=8.4, style="italic")
    ax.text(11.77, 2.85, "trained per market:\nS&P 500  $\\cdot$  HOSE", ha="center", fontsize=7.4, color="0.35")
    plt.tight_layout()
    for ext_ in ("png", "pdf"):
        fig.savefig(f"docs/paper/figures/fig_architecture.{ext_}", dpi=190, bbox_inches="tight")


if __name__ == "__main__":
    main()
