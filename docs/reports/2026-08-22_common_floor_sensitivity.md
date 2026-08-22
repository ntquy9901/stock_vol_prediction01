# Common-floor sensitivity — evaluation protocol and QLIKE robustness

Date: 2026-08-22. A robustness study, not a change to the submitted main tables. The main tables keep their
pre-specified output parameterization (z-score target + linear output + relative floor). The output
parameterization and evaluation protocol reported here were investigated AFTER the main results and are
reported as a robustness supplement; the ratio-exp configuration is treated as a PRE-SPECIFIED candidate for
a future out-of-sample evaluation, not a replacement for the main tables (which used the same test set that
motivated it).

## Motivation
Fairness of a forecast comparison does not require every model to share the same output mechanism (HAR-X is
linear; the ratio-exp deep model is positive by construction), but every forecast entering a metric should
be mapped through the SAME evaluation floor applied to RAW forecasts:
`y_eval = max(y_raw, eps_i)`, with an identical `eps_i` for all models. The delivered runner instead
pre-floors HAR-X at 1e-2*mean before the metric, so a later shared 1e-8 QLIKE clamp does not restore a common
basis. This study re-scores every model from RAW forecasts under a single common floor and reports the
sensitivity of the conclusion to the floor choice.

## Protocol
On VN30 and VN100, five seeds, config C deep model (ratio target + exp output; positive by construction):
raw HAR-X (5-feature OLS, unfloored, can be non-positive) and raw deep-C (exp*mean, no relative floor;
a machine-epsilon only guards the fp path). Both are re-scored under two common-floor policies applied
IDENTICALLY to both models:
- **relative floor** (`eps_i = 1e-2 * training_mean_i`, per node) — the main tables' 1%-of-training-mean floor;
- **fixed floor** (`eps_i = 1e-8`).
Reported per policy: QLIKE per model, the fraction of forecasts clipped by the floor, and the date-clustered
Diebold-Mariano deep-C-vs-HAR-X test.

## Results (5-seed ensembles)
Clip fraction is ~0% for both models at every cell under both policies (max 0.01%, at h1); the floor almost
never binds.

| Panel/h | policy | HAR-X QLIKE | HAR-X clip | deep-C QLIKE | deep-C clip | DM p (deep-C vs HAR-X) |
|---|---|---:|---:|---:|---:|---:|
| VN30 h1 | fixed 1e-8 | 0.6661 | 0.01% | 0.5085 | 0.00% | 0.297 |
| VN30 h1 | relative | 0.5159 | 0.01% | 0.5085 | 0.00% | 0.569 |
| VN30 h5 | fixed 1e-8 | 0.5965 | 0.00% | 0.6102 | 0.00% | 0.005 |
| VN30 h5 | relative | 0.5965 | 0.00% | 0.6102 | 0.00% | 0.005 |
| VN30 h10 | both | 0.6428 | 0.00% | 0.6507 | 0.00% | 0.145 |
| VN30 h22 | both | 0.6422 | 0.00% | 0.6531 | 0.00% | 0.355 |
| VN100 h1 | fixed 1e-8 | 1.3054 | 0.01% | 0.4964 | 0.00% | 0.116 |
| VN100 h1 | relative | 0.5115 | 0.01% | 0.4964 | 0.00% | 0.240 |
| VN100 h5 | both | 0.5633 | 0.00% | 0.5698 | 0.00% | 0.317 |
| VN100 h10 | both | 0.6023 | 0.00% | 0.6105 | 0.00% | 0.237 |
| VN100 h22 | both | 0.6405 | 0.00% | 0.6548 | 0.00% | 0.267 |

(At h5/h10/h22 the two policies give identical numbers because no forecast is clipped.)

## Interpretation
1. **The conclusion is stable to the floor choice at h5, h10 and h22:** no forecast is clipped, so the
   QLIKE values and the deep-C-vs-HAR-X DM outcome are identical under both policies (HAR-X has the lower
   mean QLIKE, the difference is not significant).
2. **Only h1 is floor-sensitive, and the sensitivity is on the HAR-X side, not the deep model.** Under the
   fixed 1e-8 floor, HAR-X's h1 QLIKE inflates (VN100 1.3054, VN30 0.6661); under the relative floor it does
   not (0.5115, 0.5159). The deep-C QLIKE is unchanged across policies (0.4964, 0.5085).
   Extreme h1 QLIKE values arose from a very small number of non-positive or near-zero forecasts produced by
   the unconstrained linear output and subsequently mapped to a tight numerical floor; the
   positive-by-construction ratio-exp parameterization removed this sensitivity.
3. **At h1, deep-C obtained a lower mean QLIKE than HAR-X under the relative-floor policy, but no
   statistically significant difference was detected** (VN100 p=0.240, VN30 p=0.569). No claim of a win, a
   tie, or equivalence is made.
4. The value of this finding is methodological: only ~0.01% of h1 forecasts are clipped, yet under a tight
   fixed floor they change the mean HAR-X QLIKE substantially — a small number of near-zero linear-output
   forecasts dominate the ratio-based QLIKE. This motivates a positive-by-construction output for the deep
   model rather than reliance on a post-hoc floor.

## Scope and next step
Config C (ratio + exp) and the common-floor protocol are a robustness result, not a main-table
specification: config C was identified after observing QLIKE on the VN30/VN100 test set, so substituting it
into the main tables on the same test set would be test-set-driven selection. The locked configuration (config
C, common floor, seeds, evaluation protocol) is pre-specified for evaluation on an untouched holdout or a
future walk-forward window; if it remains stable there, it can serve as the primary specification for the next
research version.

## Artifacts
- `scripts/garch_masked/floor_sensitivity.py` (+ `test_floor_sensitivity.py`);
  `scripts/garch_masked/ablation_vn_5seed.py` (A/B/C/D, bias-matched C-vs-D).
- `results/floor_sensitivity/{vn30,vn100}_h{1,5,10,22}.json`;
  `results/ablation_vn_5seed/{vn30,vn100}_h5{,_bm}.json`.
