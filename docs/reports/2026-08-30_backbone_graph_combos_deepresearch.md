# Backbone × Graph combinations — deep-research synthesis (2026-08-30)

Deep-research workflow (97 agents, 3-vote adversarial verification). **Verification caveat:** the safety
classifier was unavailable during the verify pass — re-open every arXiv link before citing in the paper.

## Headline (the contribution-space finding)

**None of the four named backbones (iTransformer, PatchTST, xLSTM, DLinear) has a VERIFIED published
variant applied to stock VOLATILITY forecasting.** Every verified backbone+graph hybrid targets generic
multivariate TS, stock price/return/movement, traffic/energy, or molecules — not volatility. So swapping
our LSTM in the LSTM+GAT volatility model for any of these backbones (iTransformer/DLinear/xLSTM/PatchTST +
graph) would be **at least partially NOVEL** — there is a genuine gap. (Positive for contribution; but
combine with the prior finding that deep backbones rarely beat HAR on daily volatility — novelty ≠ it works.)

## Maturity of each backbone+graph family (verified)

| Backbone + graph | Maturity | Verified precedent | Task | Citation |
|---|---|---|---|---|
| **DLinear / linear-decomposition + graph** | **most mature** | **StemGNN** (spectral-temporal GNN), **FourierGNN** (Fourier graph operator) | generic MTS forecasting | 2103.07719 ; 2311.06190 |
| **iTransformer-neighborhood + graph** | mature | **MSGNet** (multi-scale FFT + adaptive MixHop learned graph per scale) | generic MTS | Cai+ AAAI'24, 2401.00423 |
| **cross-stock transformer (iTransformer-like)** | mature (but attention, not GNN) | **MASTER** (intra/inter-stock attention) | stock **PRICE/return** | Li+ AAAI'24, 2312.15235 |
| **PatchTST + graph** | weak/tangential only | KG-embedding bolt-on (2411.11046); TiWeaver (graph only in tokenizer, 2606.03121) | generic MTS | 2411.11046 ; 2606.03121 |
| **xLSTM + graph** | **essentially NONE for TS** | only MolGraph-xLSTM (drug discovery) | molecules, NOT TS | (no TS precedent) |
| **GNN + volatility (any temporal encoder)** | exists — but HAR, not these backbones | **GNNHAR** (spillover GNN on realized vol) | **VOLATILITY** ✓ | Zhang+ IJF'25, 2308.01419 |

Note: iTransformer, xLSTM, DLinear papers themselves contain NO graph component (verified) — any hybrid is a
separate downstream work. iTransformer's cross-variate attention is graph-LIKE (implicit fully-connected).

## Answers

- **(a) Most mature:** DLinear/linear-decomposition + graph (StemGNN, FourierGNN) and the iTransformer
  neighborhood (MSGNet's adaptive learned graph). **Least / none:** xLSTM + graph has no time-series
  precedent (novel territory); PatchTST + graph exists only as weak couplings.
- **(b) Any backbone+graph on stock VOLATILITY?** **No — none exists in the verified evidence.** The only
  strong GNN-for-volatility precedent is GNNHAR (2308.01419), whose temporal encoder is HAR, not any of the
  four backbones.
- **(c) Strongest precedent to adopt for our LSTM+GAT volatility model:** an **iTransformer-style
  variate-token** or **DLinear-decomposition** temporal encoder feeding a **learned inter-series graph**
  (MSGNet / StemGNN / FourierGNN architectural precedent). **xLSTM+graph and PatchTST+graph for volatility
  are unproven / novel / risky.**

## Ranked recommendation (backbone+graph to consider for the volatility model)

1. **iTransformer(variate-token) + learned graph** — closest mature precedent (MSGNet), cross-variate design
   naturally matches a cross-stock graph; adopting it for volatility is a modest, defensible novelty.
2. **DLinear/linear-decomposition + graph** — most mature graph precedent (StemGNN/FourierGNN), strong
   parsimony prior for a small noisy panel; volatility application would be novel.
3. **(Causal) Mamba + graph** — from the prior deep-research: FinMamba/SAMBA precedents (price/movement),
   near-linear cost; volatility application novel; must use unidirectional/causal Mamba (bidirectional leaks).
4. **PatchTST + graph** — only tangential precedent; would be largely novel and unproven.
5. **xLSTM + graph** — NO time-series precedent; fully novel and highest-risk.

## Bottom line for the project
The literature gap means a modern-backbone + graph model for daily stock VOLATILITY is a legitimate NOVEL
contribution regardless of which backbone is chosen. The lowest-risk, best-precedented direction is an
iTransformer- or DLinear-decomposition temporal encoder with a learned inter-series graph (MSGNet/StemGNN
lineage). But the empirical prior stands: on daily-input, small-noisy emerging-market volatility, HAR is
hard to beat — so any such model must be benchmarked honestly against HAR-X with DM, exactly as this project
already does.
