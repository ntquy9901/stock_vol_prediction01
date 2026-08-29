# GAT Branch: Design, Cause EDA, and Depth Analysis — Synthesis for the Discussion Section (2026-08-29)

## Purpose

Consolidate the verified graph-branch design, the exploratory analysis of why the graph's incremental value
is market- and horizon-dependent, and an assessment of network depth, as source material for the paper's
Discussion. Combines an independent code verification, a read-only data EDA, and a parallel review
(`2026-08-29_gat_signal_eda_and_harm_analysis.md`). All estimator numbers are from stored results; EDA numbers
are reproducible with the scripts in §6.

## 1. Verified graph-branch design (robustness run `results/masked_rich_yz`)

The model trained by `run_masked_rich.py` is `MaskedRichNet`, not the single-layer submission model.

- **Two message-passing hops.** `MaskedRichNet` stacks two `WeightedGATLayer` hops
  (`gat1: in_dim→hidden*heads` → ELU → `gat2: hidden*heads→hidden*heads`) over the **same** masked adjacency
  (`run_masked_rich.py:2-3, 83-107`; `gat_layers=2` default at `:92`; instantiated at `:135`). Only the second
  hop's output feeds the head.
- **Weight- and sign-aware messages.** `WeightedGATLayer` (`:40-`) consumes the signed, weighted adjacency in
  both the attention and the message (unlike the binary-mask `GATLayer`), so a negative edge flips the sign of
  the transmitted message.
- **Node inputs are the raw features at the last day `t`** (not the full neighbour history); the LSTM branch is
  parallel and its output is not consumed by the graph.
- **Two graph variants** are compared, both estimated on TRAIN dates only and frozen: a symmetric Top-K
  correlation edge (`masked_rich.py:224-231`) and a directed volume-shock(t) → volatility(t+1) Top-K edge
  (`masked_rich.py:234, _directed_vol2pk`).
- The two hops are **message-passing hops, not two market/industry tiers**. There are no sector labels, sector
  nodes, or a hierarchical market graph.

## 2. Estimator-swap consistency (robustness study)

When the target estimator is changed (Parkinson → Yang–Zhang / Rogers–Satchell / Garman–Klass), the estimator
value is written into the `parkinson_volatility` column (`estimator_forecast_ablation.py:85`) and every derived
quantity is recomputed on the new series: the HAR daily/weekly/monthly features (`masked_rich.py:175`), the
market factor `market_pk = median(sqrt(estimator))` (`:171, 176`), and BOTH graph edges (`:224, 234`). Only
`volume_zscore` is estimator-independent (it is a function of raw volume, `:177`), which is correct. The swap is
therefore a full change of the volatility measure across features, target, and graph, holding volume and the
architecture fixed.

## 3. EDA — why the graph's incremental value is limited (read-only)

Per-panel measures on the Parkinson series (`sqrt` scale), train/test split 80/20.

| Measure | VN30 (N=33) | HNX (N=162) | Reading |
|---|---|---|---|
| Own-history R² (persistence) | 0.208 | 0.132 | Own volatility dominates; weaker on illiquid HNX |
| Market-factor R² (alone) | 0.120 | 0.064 | Cross-sectional signal exists |
| Market **incremental** R² over own-history | 0.019 | 0.018 | Market adds ~1.8% beyond own-history — and is **already a node feature** (`market_pk`) |
| Edge stability train→test (Top-5 overlap) | 30.3% | 8.9% | Frozen edges largely do not persist out-of-sample |
| Neighbour partial-corr lift beyond own-history | 0.149 | 0.153 | A modest residual neighbour signal exists |

A parallel review additionally reports a median node–market correlation of about 0.656 (VN30) and 0.571
(VN100): the cross-sectional co-movement is real and large, but it is captured by the `market_pk` feature that
is fed to every node at every timestep, so the graph can only add value beyond that already-present factor.

## 4. Candidate reasons the graph does not robustly improve on own-history + market features

Consistent across the EDA and the parallel review:

1. The correlation graph overlaps with `market_pk` and the HAR features; the incremental cross-sectional signal
   is small (§3, ~1.8%).
2. The volume→volatility edge is a **one-day** lead-lag (shock at `t` → variance at `t+1`) but is applied
   unchanged to the `h∈{5,10,22}` horizons — a horizon mismatch.
3. The graph is estimated on TRAIN dates and frozen; the Top-5 neighbours persist to the test period only ~9–30%
   of the time (§3), so at test time the graph often connects nodes to stale neighbours.
4. Correlation level is driven by the market regime and by persistence, not necessarily by sector spillover.
5. With two hops, a node also aggregates indirect (neighbour-of-neighbour) signal, which can transmit noise; on
   VN30 (N=33, out-degree 5) two hops already reach most of the graph.
6. HNX has ~45.3% exactly-zero targets (illiquidity), which makes the correlation edge less stable (edge overlap
   8.9%, the lowest measured).
7. Signed negative edges flip the transmitted message; the economic meaning of a sign-flipped volatility message
   is not established.

The net effect is market- and horizon-dependent (from the stored DM results): the graph attains a significantly
lower QLIKE than the no-graph LSTM at some short-horizon liquid-panel cells and is neutral-to-worse at several
longer-horizon and developed-market cells.

## 5. Network depth assessment

The model is already at two hops. Increasing depth further is unlikely to help:

- Repeated neighbour averaging drives node representations toward a common value (over-smoothing); on the small
  panels two hops already cover most of the graph, so three or more hops would erode the own-history signal that
  dominates the target (§3).
- Depth propagates over an unstable adjacency (§3, item 3): a longer path over frozen, mostly-non-persistent
  edges compounds rather than reduces edge error.
- Additional layers add parameters, which the small liquid panels (VN30 N=33) do not support without overfitting.
- The GNNHAR study (arXiv:2308.01419) reports a one-hop configuration as its recommended setting, whereas this
  project's earlier internal report kept two hops on the Vietnamese panels — an unresolved comparison.

The actionable levers, in priority order, are therefore (i) the **edge** (a stable or sector-defined graph, and
a horizon-matched edge for multi-day forecasts), (ii) a **one-hop vs two-hop** re-test, and (iii) removing the
correlation-vs-`market_pk` redundancy — not additional depth.

## 6. Recommended paper wording (Discussion)

> The graph branch models train-period cross-sectional dependence and one-day volume-to-volatility associations.
> It is not an explicit industry hierarchy. Its incremental value beyond own-history and market features is
> market- and horizon-dependent.

The paper should not state that the current graph has learned an "industry" or "sector" signal, because no
sector graph or sector baseline is included.

## 7. Proposed follow-up ablation (after the current robustness run completes)

A leave-one-out style study over graph depth and edge construction:
`hops ∈ {1, 2, 3} × edge ∈ {volume→volatility, correlation, sector, horizon-matched}` on a small liquid panel
(VN30) and a large panel (S&P 500), reporting QLIKE and the date-clustered DM test, plus an over-smoothing
diagnostic (cross-node representation similarity by layer). Predeclared expectation from §3–§5: one hop performs
at least as well as two on the current statistical edges; a sector or horizon-matched edge is the most promising
change; three hops perform worst.

## 8. Reproduction (EDA, read-only)

The EDA in §3 recomputes the Parkinson series per ticker (`ln(high/low)^2 / (4 ln 2)`), builds the sqrt-scale
panel, and measures own-history vs market-factor R² (OLS), Top-5 correlation-edge overlap between the first 80%
and last 20% of dates, and the neighbour partial correlation beyond own-history. Price directories:
`data/raw/prices` (VN30 top-level) and `data/raw/prices/hnx_vnstock` (HNX). The measurement does not touch the
running robustness suite.
