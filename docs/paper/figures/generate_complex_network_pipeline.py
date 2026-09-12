"""Technical pipeline figure for applying the Complexity-2026 (Nguyen et al., 5670093) complex-network method
as market-level features for our per-stock gamma-GBM (matplotlib -> PNG + PDF). Shows the full flow: trailing
window of N-stock returns and volume -> two correlation matrices -> alpha=0.7 blend -> Threshold network (tau=0.5)
-> 7 global topology metrics -> forward-fill daily and broadcast to every stock -> concatenated with the per-stock
OWN(9) feature vector (16-dim total) -> gamma-GBM -> Parkinson-variance forecast -> QLIKE + date-clustered DM vs
the OWN-only GBM. Design-before-run: this figure is for advisor review, matching scripts/eda/complex_network_gbm.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

matplotlib.rcParams["svg.fonttype"] = "none"


def box(ax, x, y, w, h, t, fc, fs=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                fc=fc, ec="black", lw=1.1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)


def arr(ax, x1, y1, x2, y2, txt=None, dy=0.16):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.2, color="0.25"))
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, txt, ha="center", fontsize=7.2, color="0.3")


def main():
    fig, ax = plt.subplots(figsize=(13.4, 7.4))
    ax.axis("off"); ax.set_xlim(0, 20); ax.set_ylim(0, 11)

    ax.text(10, 10.55, "Complex-network topology features for the gamma-GBM  (design, matching "
            "scripts/eda/complex_network_gbm.py)", ha="center", fontsize=10.2, weight="bold")
    ax.text(6.0, 9.95, "PART A  -  market-level topology (paper method, causal / trailing)",
            ha="center", fontsize=8.6, style="italic", color="#25518a")
    ax.text(15.6, 9.95, "PART B  -  per-stock daily gamma-GBM (our task)",
            ha="center", fontsize=8.6, style="italic", color="#2f7d4a")

    # ---- PART A : topology construction ----
    box(ax, 0.2, 7.7, 3.5, 1.8,
        r"N-stock panel, day $t$" + "\n" + r"daily_return$_{j}$" + "\n" +
        r"volume_zscore_22$_{j}$" + "\n" + r"trailing $\Delta T{=}66$ d (3 mo)", "#eaf2fb", 8.0)

    box(ax, 4.7, 8.55, 3.4, 1.05, r"return corr $\rho^{R}$" + "\n(common tickers)", "#dfeafb", 8.0)
    box(ax, 4.7, 7.05, 3.4, 1.05, r"volume corr $\rho^{V}$" + "\n(common tickers)", "#dfeafb", 8.0)
    arr(ax, 3.7, 8.9, 4.7, 9.05)
    arr(ax, 3.7, 8.3, 4.7, 7.55)

    box(ax, 9.1, 7.7, 3.3, 1.8,
        r"blend  $\rho_{\mathrm{mix}}$" + "\n" + r"$=\alpha\rho^{R}+(1{-}\alpha)\rho^{V}$" + "\n" +
        r"$\alpha=0.7$", "#e7eefb", 8.2)
    arr(ax, 8.1, 9.05, 9.1, 8.9, r"$\alpha$")
    arr(ax, 8.1, 7.55, 9.1, 8.3, r"$1{-}\alpha$")

    box(ax, 13.3, 7.7, 3.4, 1.8,
        r"Threshold network" + "\n" + r"edge if $|\rho_{\mathrm{mix}}|\geq\tau$" + "\n" +
        r"$\tau=0.5$  (undirected)", "#dbe9f6", 8.0)
    arr(ax, 12.4, 8.6, 13.3, 8.6)

    box(ax, 13.15, 4.9, 3.7, 2.2,
        r"7 GLOBAL metrics" + "\n" + "density | avg degree\nclustering | avg weight\n"
        "diameter | betweenness\neigenvector centrality\n" + r"one row per day $t$", "#d3e2f0", 7.8)
    arr(ax, 15.0, 7.7, 15.0, 7.1)

    box(ax, 8.7, 4.9, 3.6, 2.2,
        r"forward-fill to daily" + "\n" + r"recompute every" + "\n" + r"STEP$=22$ d" + "\n\n" +
        r"broadcast identically" + "\n" + r"to ALL stocks", "#eef4dd", 7.9)
    arr(ax, 13.15, 6.0, 12.3, 6.0)

    # ---- PART B : per-stock GBM ----
    own = ["har_daily", "har_weekly", "har_monthly", "rq", "mr_change",
           "mr_slope5", "mr_slope10", "mr_dev5", "mr_z22"]
    box(ax, 8.5, 1.4, 4.0, 2.6,
        r"per-stock OWN vector  $x_{i,t}\in\mathbb{R}^{9}$" + "\n" + "\n".join(own[:5]) + "\n" +
        "\n".join(own[5:]) + "\n" + r"(own history only)", "#e7f4ea", 7.4)
    arr(ax, 10.5, 4.9, 10.5, 4.0, "topo (7)")
    ax.text(11.35, 4.45, r"broadcast", ha="left", va="center", fontsize=7.0, color="0.35")

    box(ax, 13.6, 1.7, 3.4, 2.0,
        r"concat: OWN(9)$\oplus$topo(7)" + "\n" + r"$\Rightarrow\ \mathbb{R}^{16}$" + "\n\n" +
        r"gamma HistGBM" + "\n" + r"3-seed ensemble, floor FL", "#e9e9f5", 7.9)
    arr(ax, 12.5, 2.7, 13.6, 2.7, r"$x_{i,t}$")

    box(ax, 17.4, 4.7, 2.4, 1.9,
        r"$\hat{\mathrm{pk}}_{i,t+h}$" + "\n" + "Parkinson\nvariance\n" + r"$h\in\{1,5,10,22\}$", "#eaf2fb", 8.0)
    arr(ax, 17.0, 3.1, 18.6, 4.7)

    box(ax, 17.2, 1.7, 2.8, 2.0,
        "QLIKE (per obs)\n+\ndate-clustered DM\n" + r"GBM+topo vs GBM", "#f6e9f2", 7.9)
    arr(ax, 17.0, 2.7, 17.2, 2.7)

    # dimension legend
    ax.text(10.0, 0.55,
            r"GBM: $x\in\mathbb{R}^{9}$ (OWN)      GBM+topo: $x\in\mathbb{R}^{16}$ (OWN $\oplus$ 7 topology)   "
            r"     markets: HOSE  $\cdot$  S&P 500", ha="center", fontsize=8.4, style="italic")

    # separating guide line between Part A and Part B
    ax.plot([0.2, 19.8], [4.55, 4.55], color="0.8", lw=0.8, ls="--")

    plt.tight_layout()
    for ext_ in ("png", "pdf"):
        fig.savefig(f"docs/paper/figures/fig_complex_network_pipeline.{ext_}", dpi=180, bbox_inches="tight")


if __name__ == "__main__":  # pragma: no cover
    main()
