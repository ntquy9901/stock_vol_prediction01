# Addendum — QLIKE-estimation, GAT depth, and market-regime results

> Paper-ready draft section (objective style) for the three experiments motivated by Zhang et al.
> (2023), *Graph Neural Networks for Forecasting Multivariate Realized Volatility with Spillover
> Effects* (arXiv:2308.01419). Numbers from a three-seed run (seeds 42, 123, 2026) across horizons
> h ∈ {1, 5, 10, 22} on the VN30 case study, batched training (15-epoch cap with early stopping),
> current data through 2026-08-14. Diebold-Mariano (DM) uses the HLN small-sample correction with a
> HAC lag of h−1, seed-ensembled. All compared models use one shared positivity floor. NOT yet
> reconciled with the single-seed numbers in the main draft; integrate after deciding the framing.

## A. QLIKE as the estimation criterion

Training all learned models by minimising QLIKE (rather than mean squared error) is evaluated
alongside the MSE-trained reference. Table A reports the three-seed mean held-out test QLIKE for the
reference HAR, the full model trained by MSE, and the full model trained by QLIKE, with the DM test of
the QLIKE-trained full model against HAR.

| Horizon | HAR | Full (MSE) | Full (QLIKE) | DM: Full(QLIKE) vs HAR |
|---|---|---|---|---|
| h1 | 0.4633 | 0.4589 | 0.4599 | −3.01, p=0.003 (favours full) |
| h5 | 0.5503 | 0.5484 | 0.5486 | −1.21, p=0.225 (no difference) |
| h10 | 0.5933 | 0.5993 | 0.6008 | +2.45, p=0.014 (favours HAR) |
| h22 | 0.6474 | 0.6735 | 0.6657 | +4.42, p<0.001 (favours HAR) |

QLIKE estimation and MSE estimation give comparable test QLIKE at h1 and h5; at h22 QLIKE estimation
lowers the full model's QLIKE from 0.6735 to 0.6657. The full model attains a lower QLIKE than HAR at
h1 (significant); HAR attains a lower QLIKE at h10 and h22 (significant).

## B. Which configuration attains the lowest QLIKE

Table B reports, per horizon, the trained variant with the lowest three-seed mean test QLIKE under
QLIKE estimation, and its DM test against HAR. The variants follow the leave-one-out ablation
(full; minus-graph; minus-gate; minus-news; and a price-only backbone with graph, news, and gate all
removed).

| Horizon | HAR | Full | Lowest-QLIKE variant | DM: variant vs HAR |
|---|---|---|---|---|
| h1 | 0.4633 | 0.4599 | price-only backbone (0.4553) | −5.52, p<0.001 (favours variant) |
| h5 | 0.5503 | 0.5486 | minus-news, i.e. price + graph (0.5430) | −3.08, p=0.002 (favours variant) |
| h10 | 0.5933 | 0.6008 | price-only backbone (0.5953) | +0.51, p=0.608 (no difference) |
| h22 | 0.6474 | 0.6657 | price-only backbone (0.6614) | +4.48, p<0.001 (favours HAR) |

A learned model attains a lower QLIKE than HAR at h1 and h5 (significant). At both horizons the
lowest-QLIKE configuration is a parsimonious one: the price-only LSTM backbone at h1, and the price
LSTM with the graph branch (news removed) at h5. In the leave-one-out ablation on this run the news
branch and the per-ticker gate do not lower QLIKE at the short horizons, and their removal raises the
full model's rank; the price LSTM over five node features (a market factor and a volume z-score in
addition to the three HAR terms) is the component that carries the short-horizon result.

## C. GAT depth: one hop versus two hops

Table C compares the full model with a single GAT layer (one hop: a ticker aggregates its direct
graph neighbours) against two stacked GAT layers (two hops), both QLIKE-trained, and reports the DM
test between depths.

| Horizon | Full, 2 hops | Full, 1 hop | DM: 2 hops vs 1 hop |
|---|---|---|---|
| h1 | 0.4599 | 0.4619 | −2.81, p=0.005 (favours 2 hops) |
| h5 | 0.5486 | 0.5507 | −0.27, p=0.787 (no difference) |
| h10 | 0.6008 | 0.6088 | −5.04, p<0.001 (favours 2 hops) |
| h22 | 0.6657 | 0.6663 | −0.02, p=0.981 (no difference) |

On the VN30 case study two hops attain a lower QLIKE than one hop at h1 and h10 and are indistinguishable
at h5 and h22; one hop is not lower at any horizon. Reducing the graph branch to one hop also removes
the significant h1 advantage of the model over HAR (the DM p-value moves from 0.003 to 0.098). A
mean-average-distance (MAD) diagnostic on the two-hop model at h1 (0.3063 after the first hop, 0.1915
after the second) shows the second hop increases the similarity of node embeddings, yet the added
receptive field of the second hop still lowers QLIKE. This differs from Zhang et al., who report one
hop is sufficient on Dow-Jones constituents.

## D. Market-regime split (calm and turbulent days)

Test observations are split by realised target volatility into a calm regime (the lower 90%) and a
turbulent regime (the upper 10%). Table D reports the three-seed mean QLIKE for HAR and the full
(QLIKE-trained, two-hop) model in each regime, with the DM test of the full model against HAR.

| Horizon | Calm HAR | Calm Full | DM calm | Turbulent HAR | Turbulent Full | DM turbulent |
|---|---|---|---|---|---|---|
| h1 | 0.3828 | 0.3791 | −3.45, p<0.001 (full) | 1.1875 | 1.1727 | −1.16, p=0.24 (no diff.) |
| h5 | 0.4406 | 0.4238 | −13.86, p<0.001 (full) | 1.5350 | 1.6553 | +5.23, p<0.001 (HAR) |
| h10 | 0.4686 | 0.4603 | −6.61, p<0.001 (full) | 1.7152 | 1.8441 | +6.12, p<0.001 (HAR) |
| h22 | 0.5004 | 0.5036 | +1.29, p=0.20 (no diff.) | 1.9696 | 2.1091 | +3.92, p<0.001 (HAR) |

On calm days the full model attains a lower QLIKE than HAR at h1, h5, and h10 (significant); on
turbulent days HAR attains a lower QLIKE at h5, h10, and h22 (significant). Turbulent-day QLIKE is
about three to four times the calm-day value, so the turbulent decile dominates the pooled average and
turns the pooled h5 and h10 comparisons into a tie or a HAR result even though the model attains a
lower QLIKE on the 90% of days that are calm. Both models under-predict the turbulent decile, and HAR
under-predicts it less. This regime dependence is consistent with the observation that a pooled
average can obscure where a model attains its advantage.

## E. Leakage audit

The result that a price-only backbone attains a lower QLIKE than HAR at h1 was audited for data
leakage. The per-ticker target and feature scalers are fitted on the training split only; the market
factor is a contemporaneous cross-sectional statistic; the volume z-score is a trailing rolling
statistic; each input window ends at the forecast origin and the target is exactly h steps later, and
windows are built within a split; and HAR is evaluated on the identical held-out observations and
targets with the identical positivity floor as the learned models. The advantage of the five-feature
price LSTM over HAR reproduces an earlier leakage-safe finding on this project that cross-sectional
node features (a market factor and a volume z-score) lower QLIKE relative to HAR.
