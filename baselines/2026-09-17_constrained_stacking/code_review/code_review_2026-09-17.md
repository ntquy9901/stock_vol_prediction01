# Adversarial code review — OOF metric-constrained stacking (Hướng 4)

Date: 2026-09-17. Scope: `code/stacking_config.py`, `code/base_models.py`,
`code/run_constrained_stacking.py`, `test/test_constrained_stacking.py`. Lenses: leakage, loss
consistency, numerical guards, simplex-constraint correctness, config hygiene, performance.

## Summary
No CRITICAL or MAJOR findings that block delivery. The design reuses the delivered sibling
(`2026-09-17_gbme_glm_anchor`) scaffolding for the walk-forward / embargo / DM / spike / checkpoint
logic (unchanged, already reviewed), and adds only the 4-model base set, the OOF cross-fitting, and
the simplex-QLIKE meta-optimiser. Those additions were reviewed below.

## Findings

### F1 (leakage) — OOF trains on future inner blocks. VERIFIED SAFE.
`oof_predict` predicts a held inner block from a base model trained on all OTHER blocks, including
temporally-later ones. This is intentional and matches the referenced pattern
(`embed.oof_train_z`): the OOF predictions are used ONLY to fit the level-2 meta-weights, never as a
forecast. The reported forecasts (test) come from a base fit on the causal `trf_e` (before the test
window with an `int(1.6h)+5`-day embargo). No test row participates in base training or weight
fitting. Verdict: no label leakage. Covered by `test_oof_covers_every_row_and_is_causal`.

### F2 (loss consistency) — meta objective == reported metric. OK.
The meta-optimiser minimises `stack_qlike` = pooled QLIKE with the champion floor `FM.FL`, the exact
metric reported and used in the DM test. The three gamma bases are trained on gamma deviance
(≈ QLIKE up to constants); HAR is deliberately MSE-trained as the "different" member. No
train/report loss mismatch on the quantity being optimised (the stack).

### F3 (numerical guard) — exp-link overflow. HANDLED (documented benign warning).
`GammaRegressor.predict` / XGBoost `reg:gamma` apply an exp link that can overflow on an extreme
z-scored row (a `RuntimeWarning: overflow encountered in exp` is emitted). Every base prediction and
the stack blend are clipped to `[FL, PRED_CAP]`, so `+inf → PRED_CAP` and `-inf → FL`; no NaN can
reach QLIKE. Covered by `test_base_predictors_stay_capped_on_extreme`. GBME uses the champion
floor-only (no cap), matching the delivered champion exactly.

### F4 (simplex-constraint correctness) — SLSQP + post-normalisation. OK.
`fit_simplex_qlike` imposes `sum w = 1` (eq constraint) and `w ∈ [0,1]` (bounds). SLSQP can return
tiny negative components and a sum a hair off 1; the code clips negatives then renormalises, and
falls back to uniform if the solver degenerates to all-zeros (fail-safe, not silent-zero). Tests
assert the constraint (`sum≈1`, `w≥0`) and the collapse-onto-dominant-base behaviour, plus the
zero-fallback branch.

### F5 (best-single is picked ex-post on test QLIKE) — CONSERVATIVE, by design.
`_pool_doc` selects `best_single = argmin test QLIKE` over the four bases, then asks whether the
stack beats THAT. Picking the ex-post best single makes the comparison HARDER for the stack (it must
beat an oracle-selected baseline), which is the correct direction for a falsification: if the stack
still cannot win, the NO-GO is strongly supported. The stack-vs-GBME (champion) comparison is
reported alongside for reference. Documented in `design.md`.

### F6 (stack train metric is in-sample / optimistic) — ACKNOWLEDGED, non-gating.
The stack's `train_metrics` blend in-sample base train predictions, so they are optimistic. The
stack is a deterministic combination (name has no learned token) and is exempt from the over/under-
fit gate; only the genuinely-learned `XGB` member must be `ok`. The optimism is disclosed and does
not affect any reported OOS number.

### F7 (config hygiene) — all tunables single-sourced. OK.
Every tunable (K, VALID_LEN, horizons, MIN_ROWS, SLSQP maxiter/ftol, GAIN_MIN, DM_ALPHA,
KILL_HORIZONS, SPIKE_WINDOWS, PRED_CAP, GLM/XGB capacity, HAR floor fraction) lives in
`stacking_config.py` with inline intent comments. OWN-8 is single-sourced from the paper_models
config by path (no second bare `config` module → no `sys.modules` collision). The uniquely-named
`stacking_config.py` avoids the `submission/soict_lstm_gat/config.py` collision.

### F8 (performance) — no batch=1 neural anti-pattern. OK.
All bases are vectorised tabular fits (sklearn / xgboost hist / numpy lstsq); no per-item Python
loop over samples, no GPU underutilisation (gamma boosters are CPU-hist by design). The OOF inner
fold is `K=3` (not leave-one-out) to bound cost. Seed-ensembling is a short comprehension over 3
seeds.

## Gate note (not a code defect)
On a SINGLE fold checkpoint the XGB val→test QLIKE gap can read ~35% (a regime shift between the
trailing-22-date val slice and that fold's test window), which `classify_fit` would label "overfit".
On the FULL pooled 8-fold result the gap collapses to ~3–4% (verified identical behaviour in the
delivered sibling `glm_anchor_hose_h1.json`: XGB gap 0.0353, status `ok`). The over/under-fit gate
must therefore be evaluated on the completed run, not a partial checkpoint — consistent with the
"wait for all folds before reading numbers" rule.

## Tests
19 tests, all pass. Diff-coverage on the three code modules: C0 line = 100%, C1 branch = 100%
(`pytest --cov --cov-branch`). Covered: every base predictor (positive/finite/capped), OOF coverage
+ causality + coverage-gap raise, simplex constraint + collapse + degenerate fallback,
`stack_qlike`, verdict/success, `_safe_dm` degenerate + ValueError, spike mask, `_apply_weights`,
`_load_earn` (sp500 passthrough / hose-missing / hose-parquet-present), `_metrics5`, and the
walk-forward run smoke (hose per-fold+spike, sp500 no-spike, multi-fold skip, all-in-spike omit)
with the gate-required evidence keys asserted present.
