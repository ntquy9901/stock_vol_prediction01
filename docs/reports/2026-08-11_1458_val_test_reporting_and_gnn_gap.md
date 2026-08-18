# Validation-vs-test reporting convention and the GNN validation-good / test-bad gap

Read-only research report. Two questions: (Q1) do international time-series / volatility / GNN-for-finance
papers report validation AND test numbers separately, and what should our paper do; (Q2) why do metrics
look good on validation but not on test, especially with GNNs, mapped to our measured results. No code or
paper file was modified. Every convention/empirical claim is cited (author/year/venue or URL); claims not
directly verified against a primary source are marked `[unverified]`.

Project files read for grounding: `docs/paper/track_b_paper_draft.md` (Tables 1-4, §6),
`docs/reports/ladder_consistent_multihorizon_2026-08-09_180326.md`,
`docs/reports/2026-08-10_0412_beat_har_sweep_results.md`, `docs/reports/2026-08-10_0130_gat_price_har_quick.md`,
`docs/eda/reports/EDA_GRAPH_REPORT.md`.

---

## Q1 — Do international papers report validation AND test separately?

### 1.1 The general ML convention

The three-way split has one canonical role assignment: **train fits parameters, validation selects the
model (hyperparameters, early-stopping, checkpoint), test is touched once at the end to report an unbiased
estimate of generalization error.** The reason validation is *not* the headline number is that the moment
you tune against a set you begin to overfit to it; validation error is an optimistically biased estimate of
generalization precisely because it drove selection (Raschka 2016, "Model evaluation, model selection, and
algorithm selection in machine learning",
https://sebastianraschka.com/blog/2016/model-evaluation-selection-part3.html; Géron / standard three-way
split summary as aggregated at https://iq.opengenus.org/testing-training-validation-and-holdout-set/). The
"the more you look at the test set, the less useful it becomes" adaptive-overfitting caution is the same
principle applied to the test set itself (discussion in "On the Value of Out-of-Distribution Testing",
https://arxiv.org/pdf/2005.09241).

**Headline = test.** Validation exists to justify the tuning protocol, not to be reported as a result. When
data is scarce, nested / walk-forward CV plays validation's role, and the outer held-out fold is still the
reported number.

### 1.2 The convention specifically in RV / volatility / GNN-for-finance papers

Econometric realized-volatility forecasting reports **out-of-sample (test / forecast-sample) losses only.**
The standard design splits into an estimation (train) sample and a forecast (test) sample, re-estimates on a
rolling/expanding window, and reports one-step / h-step losses (MSE, MAE, QLIKE) on the forecast sample,
with Diebold-Mariano (Diebold & Mariano 1995) and the Model Confidence Set (Hansen, Lunde & Nason 2011) as
the significance layer — MCS being the preferred multi-model procedure (survey of the design in "Realized
range volatility forecasting", https://www.sciencedirect.com/science/article/abs/pii/S105905601500043X; MCS
usage e.g. https://arxiv.org/html/2503.00851v2). Classical HAR has a fixed daily/weekly/monthly lag
structure estimated by OLS, so there is often *no* separate validation set at all; a validation split only
appears for the ML variants, and even then only the final test-sample losses (with DM/MCS) are tabled — the
validation split is described in the protocol text, not shown as a results column.

- **GNNHAR — Zhang, Pu, Cucuringu & Dong, IJF 41(1) 2025, 377-397**, "Forecasting realized volatility with
  spillover effects: perspectives from graph neural networks" (https://web.media.mit.edu/~xdong/paper/ijf25.pdf;
  DOI 10.1016/j.ijforecast.2024.09.002; code https://github.com/chaozhang-ox/GNNHAR). The headline tables are
  **out-of-sample MSE / QLIKE with MCS** on DJIA-30 / S&P-100. Validation data is used inside the protocol —
  QL-trained models (HARQ, GHARQ, GNNHAR) are optimized with Adam "using both the training and validation
  data", and an ensemble over random initializations is used for robustness — i.e. validation feeds
  selection/optimization, it is not a separate reported results block. `[Verified via search summary of the
  IJF25 text; the specific "training and validation data" wording is quoted from that summary, not
  independently re-read from the PDF — treat the exact phrasing as [unverified] but the out-of-sample-only
  reporting is the field norm.]`
- **Audrino & Chassot, IJF 2025**, "HARd to Beat: The Overlooked Impact of Rolling Windows in the Era of
  Machine Learning" (arXiv:2406.08041). Headline is out-of-sample forecast-sample performance; the paper's
  own thesis is that rolling-window / evaluation-protocol choices, not a val-vs-test display, drive the ML
  results, and that ML ties or underperforms HAR. Reinforces "report test, not validation."
- Broader ML-RV comparisons (e.g. Rahimikia & Poon, "Machine Learning for Realised Volatility Forecasting",
  SSRN 3707796) follow the same shape: tune on a validation slice of the training window, report only
  held-out losses with DM/MCS.

**Conclusion for Q1.** The near-universal convention — in both mainstream ML and RV/GNN-finance venues — is:
**the headline reported result is the TEST set; validation is used only for model selection (hyperparameters,
early-stopping, checkpoint) and is described in the protocol, not presented as a co-equal results table.**
Papers avoid the "two sets of numbers" confusion by (a) leading with a single held-out out-of-sample table,
and (b) confining validation to one sentence in the evaluation-protocol paragraph (and to significance via
DM/MCS on the test losses). Showing a full validation results table beside the test table is *not* the norm.

### 1.3 What our paper currently does, and the recommendation

Our draft (`docs/paper/track_b_paper_draft.md`) currently shows **both** validation and test as co-equal
blocks:
- **Table 1** = test only (proposed G1 vs classical baselines) — already correct, matches convention.
- **Table 2** = a `VAL` block (5 rows) *and* a `TEST` block (5 rows) for the P0→G1 ladder, stacked as equals.
- **Table 3** = paired t-tests reported for `VAL` and `TEST` on every contrast.
- §6.2 prose narrates validation and test numbers side by side ("lowers validation QLIKE from 0.506196 to
  0.503117 and test QLIKE from 0.564780 to 0.559854").

This is the exact "two sets of numbers" ambiguity the convention avoids, and in our case it actively invites
misreading, because our val and test **disagree on the graph and gate verdicts** (see Q2): a reader could
cite the favorable validation QLIKE for G1 as if it were the result.

**Recommendation (lead with test, demote/clarify validation) — for consideration; not applied here:**

1. **Make every headline claim a test claim.** Table 1 stays test-only. In Table 2, promote **TEST** to the
   primary block and either (a) drop the VAL block from the main table, moving it to an appendix, or (b) keep
   VAL but visually subordinate it (clearly labeled "validation, used for model selection only — not the
   reported result"). Option (a) is the cleaner match to GNNHAR / Audrino-Chassot; option (b) preserves the
   ladder's diagnostic value while removing the ambiguity.

2. **State the protocol once, in the evaluation section.** One sentence, e.g.:
   > "Model selection — early-stopping and best-checkpoint — uses the validation split only; all reported
   > results and significance tests are computed on the held-out test set, which is consulted once."

   This is the standard phrasing that lets you mention validation without it competing with the test result.

3. **Where validation is genuinely informative (the val/test divergence itself), frame it as a finding, not
   a result.** Our validation-favors-graph / test-null split is scientifically interesting and worth one
   explicit sentence — but as evidence *about robustness/overfitting*, explicitly labeled, e.g.:
   > "The graph lowers validation QLIKE on all three seeds yet yields no significant test improvement under
   > DM; we therefore report the graph as null on the held-out test set and read the validation gain as
   > selection-period optimism (Section [Q2])."

   This converts a confusing double-number into a defensible robustness observation.

4. **Keep DM/MCS on test as the primary significance evidence** (the draft already does this in §6.3-6.4).
   Consider adding an MCS across all baselines+rungs on test — it is the RV-field-preferred multi-model
   procedure and would let the paper make one clean "which models are indistinguishable from HAR" statement
   instead of pairwise val/test tables. `[MCS not currently in the draft — suggestion, not a claim it exists.]`

5. **Suggested one-line phrasing to defuse the two-number confusion in §6.2:**
   > "Throughout, validation metrics are reported only to expose the selection-period behavior of each
   > component; the held-out test column is the result. Where the two diverge (the gate and the graph), the
   > test column governs the verdict."

Net: our Table 1 is already convention-correct. The fix is Table 2 / Table 3 / §6.2 — lead with test, label
validation as selection-only, and reframe the val-vs-test gap as a robustness finding rather than a second
result.

---

## Q2 — Why good on validation, bad on test (especially GNNs), mapped to our results

Five mechanisms, each with a citation and each tied to a measured number in our reports.

### 2.1 Best-validation checkpoint / model selection → optimistic bias on validation

Selecting the epoch/checkpoint by lowest validation loss makes validation an optimistically biased estimate
of generalization: you are reporting the minimum of a noisy curve you searched over (Raschka 2016,
https://sebastianraschka.com/blog/2016/model-evaluation-selection-part3.html). Our training uses exactly this
— "best-validation checkpoint selection on the pooled" set (paper draft §5, line 275; "best-validation-loss
checkpoint selection", line 307). So any component whose only edge is on validation (the gate, the graph) is
a prime candidate for selection-induced optimism.

**Mapped to us:** In the multi-horizon ladder
(`ladder_consistent_multihorizon_2026-08-09_180326.md`), the graph (G1 vs P3) improves **validation** QLIKE
on 3/3 seeds at h5/h10/h22 (h22 val paired-t p=0.0002) but does **not** carry to test (h22 test 0/3 seeds,
+0.0049 QLIKE; h5 test paired-t p=0.79). That is the signature of selection-period optimism: the gain lives
where the checkpoint was chosen and evaporates on the untouched split.

### 2.2 Temporal distribution shift — test is a later, different regime

The chronological split places validation and test in potentially different market regimes; finance data is
non-stationary with regime shifts, so minimizing training/validation loss no longer guarantees test
performance (survey: "Test-Time Adaptation for Non-stationary Time Series", https://arxiv.org/html/2602.00073v1;
RevIN motivation, ICLR 2022, https://openreview.net/pdf?id=cGDAkQo1C0p; finance-specific degradation,
https://arxiv.org/html/2511.18578v1). This is *the* central reason in finance.

**Mapped to us:** Our own EDA flags regime dependence as "Strong" evidence and reports high-vol-regime mean
|PK corr| = 0.2686 vs low-vol 0.0789 (`EDA_GRAPH_REPORT.md`) — cross-stock structure is itself
regime-dependent, so a graph whose usefulness is estimated on the train/val window need not hold in a
later-period test window. The test QLIKE levels are uniformly higher than validation across all rungs (e.g.
h5 P0 val 0.5096 vs test 0.5676; Table 2), consistent with a harder/shifted test period rather than model
failure per se — the shift moves *all* rows, and the small graph/gate edges are what get erased by it.

### 2.3 Edge / graph-structure instability out-of-sample (the GNN-specific reason)

Two GNN-specific findings: (i) graph structure itself induces overfitting — neighborhood aggregation makes
the model easier to fit the training graph, and when train and test graph structure differ, that reliance
hurts generalization ("Rethinking Generalization in GNNs: A Structural Complexity Perspective",
https://arxiv.org/html/2605.13597); (ii) the GNN generalization gap is unusually **sensitive to which edges
are exposed** — changing only the edge structure substantially alters the performance gap
(https://arxiv.org/pdf/2601.17130). If the adjacency estimated on train/val does not persist into test, the
message-passing that helped on validation becomes noise on test.

**Mapped to us — this is the decisive, directly measured reason for our pattern.** `EDA_GRAPH_REPORT.md`
shows the exact edge our G1 uses (plain PK-correlation kNN) is the **worst** construction tested: −2.20% OOS
RMSE vs a HAR+market baseline (per-stock sign p=0.014), because ~77% of cross-stock PK correlation is a
single market factor (mean R² on MarketPK = 0.4241; only 23.1% of |corr| survives market adjustment) that
HAR already captures via each stock's own autocorrelated volatility. So the correlation GAT re-learns a
factor HAR has and adds estimation noise. Edge instability is also measured directly: consecutive-snapshot
Top-5 neighbor Jaccard = 0.39 with edge turnover 0.60 — the graph the model aggregates over reshuffles out of
sample. This is precisely why the GAT "beats HAR 6/6 on validation but 1/6 on test"
(`2026-08-10_0130_gat_price_har_quick.md`): the correlation edge fits the val window and does not persist.

### 2.4 Small universe / high variance + overfitting to validation via selection

Small effective sample and few assets inflate the variance of any measured edge, so a small validation win is
within noise and need not replicate; deeper/multi-hop aggregation makes this worse ("From Local Structures to
Size Generalization in GNNs", https://arxiv.org/pdf/2010.08853). Repeatedly selecting components/epochs
against validation compounds the adaptive-overfitting risk (https://arxiv.org/pdf/2005.09241).

**Mapped to us:** 33 tickers is a small universe (paper Limitations; `EDA_GRAPH_REPORT.md` "Out-of-sample
stability: Weak"). The beat-HAR sweep (`2026-08-10_0412_beat_har_sweep_results.md`) tried C1-C6 (QLIKE-loss,
HAR-residual, spillover, learned adjacency) and **none** DM-beats the P0 HAR anchor on test QLIKE — several
validation-plausible levers all collapse to a documented null on test, exactly what small-universe high
variance plus selection predicts. The quick-GAT report itself notes test seed-std (RMSE 6.6e-7) is tiny
relative to the test miss (Δ +1.4e-5), so the test *underperformance* is a robust sign, not seed noise — the
validation edge, not the test miss, is the fragile quantity.

### 2.5 Leakage that inflates validation but not test

Any information from the selection period bleeding into features/graph would lift validation specifically. We
found **no evidence of leakage** in our pipeline — this mechanism is present in the literature but does not
appear to apply to us, and that is itself informative (it means the gap is genuine shift/overfitting, not a
bug). Our reports document leakage-safe construction: chronological 70/15/15 with train strictly preceding
test (asserted), market betas / regime thresholds / neighbor selection / coefficients fit on train only,
trailing-window snapshots with per-snapshot `assert_snapshot_no_lookahead`, and automated leakage assertions
(`EDA_GRAPH_REPORT.md` "Leakage Audit"); the ladder uses a leakage-safe graph-bound train window and
train-only per-ticker scalers, and the sweep re-verifies "no val/test date enters the graph structure". So
our val-good/test-bad gap is attributable to §2.1-2.4 (selection optimism + temporal shift + edge
instability + small-sample variance), **not** to leakage.

### 2.6 Why our GNN specifically shows this pattern — synthesis

Our G1 stacks three of the mechanisms at once: (a) it is selected by best-validation checkpoint (§2.1); (b)
its only value-add is a cross-stock correlation edge that our EDA measures to be redundant with a market
factor HAR already holds and negative OOS vs HAR+market (§2.3); (c) that edge is unstable across snapshots
(Jaccard 0.39, turnover 0.60) and regime-dependent (§2.2), on a 33-asset universe with weak OOS stability
(§2.4). The result is a component that reliably lowers validation loss (where it was selected and where its
edge happens to fit) and reliably fails to move held-out test loss under DM — observed identically in the
multi-horizon ladder (graph null at h1/h5/h10/h22 on test), the quick price-only GAT (val 6/6, test 1/6), and
the C1-C6 sweep (no DM-significant test beat of HAR). The pattern is not a single bug; it is the expected
behavior of a correlation-edge GNN on market-factor-dominated, regime-shifting, small-universe daily
volatility.

---

## Sources

- Raschka, S. (2016), "Model evaluation, model selection, and algorithm selection in machine learning" — https://sebastianraschka.com/blog/2016/model-evaluation-selection-part3.html
- Three-way split / holdout convention — https://iq.opengenus.org/testing-training-validation-and-holdout-set/
- "On the Value of Out-of-Distribution Testing: An Example of Goodhart's Law" — https://arxiv.org/pdf/2005.09241
- Zhang, Pu, Cucuringu & Dong (2025), "Forecasting realized volatility with spillover effects: perspectives from graph neural networks", IJF 41(1):377-397 — https://web.media.mit.edu/~xdong/paper/ijf25.pdf ; code https://github.com/chaozhang-ox/GNNHAR ; https://www.sciencedirect.com/science/article/abs/pii/S0169207024000967
- Audrino & Chassot (2025), "HARd to Beat: The Overlooked Impact of Rolling Windows in the Era of Machine Learning", IJF (arXiv:2406.08041)
- Rahimikia & Poon, "Machine Learning for Realised Volatility Forecasting", SSRN 3707796 — https://dx.doi.org/10.2139/ssrn.3707796
- RV rolling-window out-of-sample design + DM/MCS — https://www.sciencedirect.com/science/article/abs/pii/S105905601500043X ; https://arxiv.org/html/2503.00851v2
- Non-stationarity / regime shift / distribution shift in financial time series — https://arxiv.org/html/2602.00073v1 ; https://arxiv.org/html/2511.18578v1 ; RevIN (ICLR 2022) https://openreview.net/pdf?id=cGDAkQo1C0p
- GNN structure-induced overfitting — https://arxiv.org/html/2605.13597
- GNN gap sensitivity to exposed edges — https://arxiv.org/pdf/2601.17130
- GNN size/structure generalization — https://arxiv.org/pdf/2010.08853

## Verification notes

- The out-of-sample-only reporting norm and the DM/MCS convention are consistent across the sources above and
  match our own draft's Table 1 approach.
- The exact GNNHAR wording "using both the training and validation data" is quoted from a search summary of
  the IJF25 text, not from an independent re-read of the PDF — the specific phrasing is `[unverified]`; the
  out-of-sample-only headline reporting is the well-established field norm.
- All project numbers (QLIKE/RMSE, DM p-values, seed counts, EDA correlation/Jaccard/market-R² figures) are
  quoted from the four project reports named at the top; no new experiment was run for this document.
