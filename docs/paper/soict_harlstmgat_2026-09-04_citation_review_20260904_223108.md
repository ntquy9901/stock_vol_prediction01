# Citation and reviewer-risk review

## Overall assessment

**Assessment: Share with caveats; citation revision recommended before submission.**

The manuscript already cites the main methodological foundations: HAR, Parkinson, Rogers--Satchell, Yang--Zhang, LSTM, GAT, QLIKE, Diebold--Mariano, and the Harvey--Leybourne--Newbold correction. The main remaining risk is not a lack of citations for the core model equations. It is that several explanatory or empirical claims are broader than the citation immediately attached to them, or are presented as data facts without a visible source.

The review below uses `docs/paper/soict_harlstmgat_2026-09-04_v2.tex` as the canonical manuscript structure and checks the corresponding rewritten English PDF. No source file was modified.

## Priority findings

### 1. High priority: date-clustered inference and the `sqrt(N)` claim

**Location:** `v2.tex` lines 97--107 and 456--458.

The manuscript states that co-moving stocks are not independent, that date-clustered DM inference is needed, and that a naive per-observation test inflates the statistic “roughly in proportion to the square root of the number of stocks.” The first part is plausible and consistent with panel-dependence literature, but the square-root statement is a quantitative asymptotic claim. It currently has no direct citation or derivation.

**Why a reviewer may ask:** The exact inflation depends on the dependence structure, effective cross-sectional correlation, weighting, missingness, horizon overlap, and the test implementation. `sqrt(N)` should not be presented as a general rule without a derivation or a source that matches this setting.

**Recommended fix:**

- Add a citation on inference with cross-sectional dependence and cluster-robust inference.
- Replace the universal wording with: “can substantially overstate precision when cross-sectional dependence is ignored.”
- If the square-root statement is retained, derive it for the paper’s loss-differential aggregation or provide a simulation/sensitivity appendix.
- State precisely whether the DM statistic is applied after date-level cross-sectional averaging, with HAC lag `h-1`, and whether the five-seed ensemble is fixed before testing.

### 2. High priority: the motivation for the directed `volume -> Parkinson` graph

**Location:** `v2.tex` lines 139--141 and 178--180.

The graph is described as being “motivated by volume leading volatility,” but no citation is attached to that motivation. The graph construction itself is an original design choice, but the economic motivation is a literature claim.

**Why a reviewer may ask:** A reviewer may ask why volume is a directed source and Parkinson variance is a directed response, rather than using contemporaneous correlation, returns, order imbalance, or an undirected graph. The paper should distinguish documented volume--volatility association from the stronger causal/lead-lag interpretation implied by an arrow.

**Recommended fix:**

- Cite a primary or authoritative volume--volatility study near the motivation sentence.
- Avoid causal wording unless the lead-lag relation is estimated and tested. Prefer “motivated by documented volume--volatility dependence” or “uses volume shocks as graph sources by design.”
- Report whether edge direction is a prior design choice or learned from training data, and add an ablation against an undirected or correlation-based graph if the causal interpretation is important.

### 3. High priority: VN100/VN30 size, liquidity, and correlation claims

**Locations:** `v2.tex` lines 84--86, 217--224, 338--347, and 431--447.

The manuscript calls VN100 “larger and more liquid,” calls VN30 smaller, and argues that the graph effect tracks node breadth and liquidity rather than correlation magnitude. It also states that VN30 has the higher mean pairwise correlation.

**Why a reviewer may ask:** These are empirical properties of the sample, not self-evident properties of the model. The current text gives node counts and masked observations, but it does not show a source or a table for liquidity, turnover, index membership date, or the exact pairwise-correlation calculation.

**Recommended fix:**

- Cite the official index methodology or exchange/index provider for VN30/VN100 composition and membership.
- Add a compact data-scope table with median/mean turnover, coverage dates, number of active names, mean/median pairwise correlation, and the exact calculation window.
- Rephrase the interpretation as an association: “In this sample, the graph effect coincides with the broader VN100 panel; this does not identify liquidity or node count as a causal mechanism.”
- Avoid using “more liquid” unless liquidity is measured and reported.

### 4. Medium priority: opening motivation about options, capital, and risk budgets

**Location:** `v2.tex` lines 58--61.

The claim that volatility affects option pricing, capital tied up in positions, and portfolio risk budgeting is reasonable, but the single citation to Corsi (2009) is primarily a realized-volatility/HAR citation and may not support every application claim.

**Recommended fix:** Either narrow the sentence to “volatility is a central input to financial risk measurement and derivative valuation” or add one finance/risk-management reference that directly supports the applications.

### 5. Medium priority: Parkinson estimator uses more information than close-to-close

**Location:** `v2.tex` lines 62--65.

The statement is supported later by the Parkinson reference at line 147, but the first occurrence has no citation immediately attached.

**Recommended fix:** Add `\cite{parkinson1980}` directly after the high--low estimator claim in the Introduction. This is a small but worthwhile citation-placement improvement.

### 6. Medium priority: QLIKE description is too strong in the terminology paragraph

**Location:** `v2.tex` lines 115--117.

The text says QLIKE “penalises under-forecasting of risk more heavily than over-forecasting.” Patton (2011) supports QLIKE as a robust loss for imperfect volatility proxies, but the asymmetry depends on the exact argument convention and loss definition.

**Why a reviewer may ask:** The paper later uses a ratio-based QLIKE with a positivity floor. The direction and size of asymmetry should be demonstrated for that exact implementation rather than asserted generally.

**Recommended fix:** Use a safer definition: “QLIKE is a proxy-robust loss for volatility forecasts; in the implementation used here, its asymmetric shape makes forecast under- and over-shoots matter differently.” Then give the exact formula and cite Patton (2011). If the under-forecasting statement is retained, include a one-line algebraic explanation or a citation that explicitly discusses the chosen convention.

### 7. Medium priority: GARCH definition and exclusion

**Location:** `v2.tex` lines 115 and 133--134.

GARCH is defined as a model that carries variance forward from recent return shocks, and the paper says GARCH was not run. The definition is standard, but the bibliography has no canonical GARCH citation.

**Recommended fix:** Add the canonical ARCH/GARCH reference if GARCH remains in the terminology and related-work discussion, or remove the definition and simply state that GARCH is outside the experiment scope.

### 8. Medium priority: “HAR is famously hard to beat” and optimality-style wording

**Locations:** `v2.tex` lines 70 and 446--447.

The manuscript cites Corsi and Audrino et al. for the strength of HAR, which is directionally appropriate. However, “famously hard to beat” and “already close to optimal” are rhetorical or near-optimality claims.

**Recommended fix:** Replace with “provides a strong low-complexity benchmark” and “may leave limited incremental room for more complex models in this sample.” The latter should remain explicitly an interpretation, not a theorem.

### 9. Low priority: GAT/spillover terminology

**Locations:** `v2.tex` lines 74--81 and 120--122.

GAT is cited, but “cross-sectional spillover” is used as if the attention edge itself establishes economic spillover. A GAT attention weight is a learned model quantity; it is not automatically a structural spillover measure.

**Recommended fix:** Use “cross-sectional dependence” or “cross-asset information sharing” for the model mechanism. Reserve “spillover” for the economic interpretation and cite Diebold--Yilmaz or a directly relevant spillover study. State that the graph is predictive, not causal.

### 10. Low priority: “split-invariant” and corporate-action statements

**Locations:** `v2.tex` lines 411--419 and 462--465.

The high--low ratio is invariant to a common multiplicative split adjustment within a day, but the manuscript should be careful not to imply that Parkinson solves all corporate-action problems. Dividends, bad OHLC records, and adjustments that alter intraday fields can still matter.

**Recommended fix:** Say “invariant to a common multiplicative split factor applied to the day’s high and low” and retain the adjusted-price limitation. A citation to the Parkinson estimator is sufficient for the formula; the data-cleaning claim should be documented by the project’s data specification or a source-data audit.

## Citation coverage that is currently adequate

- HAR and long-memory volatility: Corsi (2009).
- Parkinson estimator: Parkinson (1980), although the Introduction should cite it at first use.
- Rogers--Satchell and Yang--Zhang estimators: citations are attached to the formula section.
- LSTM: Hochreiter and Schmidhuber (1997).
- GAT: Veličković et al. (2018).
- QLIKE robustness to imperfect volatility proxies: Patton (2011), provided the terminology is made less asymmetric/absolute.
- DM test and HLN correction: citations are present in Related Work; repeat them in the Protocol sentence if the venue expects local citation placement.
- HAR-X and related extensions: citations are present, but the bibliographic metadata for Clements et al. (2024) and GNAR-HARX (2025) should be completed.

## Bibliographic risks reviewers may notice

1. `clements2024` has no journal, conference, working-paper, DOI, or arXiv identifier in the bibliography entry.
2. `gnarharx2025` has no author list and only a title plus arXiv identifier. This looks provisional and should be completed from the authoritative record.
3. The manuscript says “network-augmented HAR-X analogues have been proposed”; this claim should use a fully identifiable reference.
4. If GARCH remains discussed, add its canonical source rather than leaving the model class uncited.

## Suggested minimal pre-submission citation patch

If the authors want the smallest defensible revision, add or revise the following:

1. `\cite{parkinson1980}` at the first Parkinson claim in the Introduction.
2. A volume--volatility citation immediately after the graph-motivation sentence.
3. A cross-sectional-dependence/cluster-robust inference citation immediately after the date-clustered DM rationale.
4. A data/index source and a table reference for VN30/VN100 liquidity and correlation claims.
5. A canonical GARCH citation, or remove the GARCH definition.
6. Replace the `sqrt(N)` statement with a qualified statement unless it is derived or simulated.
7. Replace “causal” or “spillover” wording where the graph is only a predictive architecture.
8. Complete the metadata for `clements2024` and `gnarharx2025`.

## Reviewer-question forecast

The most likely reviewer questions are:

- Why is the graph directed from volume to Parkinson variance, and what evidence supports that direction?
- Is the graph a predictive device or an economic spillover model?
- How exactly is date-clustered DM implemented, and why is the `sqrt(N)` inflation statement valid here?
- Where are the liquidity and pairwise-correlation statistics supporting the VN100/VN30 interpretation?
- Does the positivity floor alter QLIKE rankings, especially for low-liquidity or zero-range days?
- Are the “significant” findings adjusted for the number of horizons, metrics, panels, and model comparisons?
- Can the cited HAR-X, GNAR-HARX, and alternative-estimator references be independently verified from complete bibliographic records?

