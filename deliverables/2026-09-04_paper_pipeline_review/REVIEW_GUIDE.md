# Review guide — what to check, and where

Four dimensions (all requested). For each finding, cite `file:line`. `archive/` is out of scope.
Report critical/major issues that would invalidate a paper claim first.

## A. Leakage & correctness
Goal: confirm no future information reaches any train/val artifact, and that named metrics/estimators
match their published formulas.
- **Per-fold train-only fitting:** in `baselines/2026-08-31_walkforward_volga/code/wf_enriched_panel.py`
  `pack_fold` / `_fit_scalers` — verify every per-node target/feature scaler AND the vol→PK adjacency
  (`_directed_vol2pk(..., last_tr_row, ...)`) use only `fold.train` rows; `last_tr_row = tr_anchor[-1]+horizon`.
- **No-lookahead test:** `.../tests/` perturb-every-post-train-row test must show train artifacts
  bit-identical; `assert_no_leakage(folds, target_dates, horizon)` (`wf_folds.py`) run each fold.
- **Target formation:** target = `parkinson_variance` at `t+horizon`, formed at train time (not a
  stored column) — confirm horizon offset is applied consistently to y and to the val/test masks.
- **HAR/HAR-X:** `run_walkforward._har_ols_preds` fits OLS on `D.tmask_tr` rows only, refit per fold;
  HAR = [daily, weekly, monthly], HAR-X = +[market_pk, volume_zscore]. Confirm the positivity floor
  (`nfloor`) is IDENTICAL across all compared models (a known past bug class).
- **DM / QLIKE:** `run_masked_rich._dm_all` (date-clustered on QLIKE/SE/AE), `metrics.qlike` — check
  QLIKE floor is the same floor used in every model's predictions, and DM clusters by date with the
  horizon-aware correction.
- **Estimators:** `scripts/eda/volatility_estimators.py` — the windowed Yang–Zhang must match the
  published windowed formula (there is a test-vs-formula); Parkinson/GK/RS drift-independent forms
  correct. Named estimators must NOT be per-day proxies mislabeled (CLAUDE.md rule).
- **Ablation isolation (pooled):** `baselines/2026-09-04_pooled_transfer_vn30/code/pooled_panel.py`
  `restrict_fold` + the `test_isolation.py` gate — Arm 0 VN30 predictions must be invariant to
  non-VN30 node features (proves the single-panel mask reproduces a 31-node system). Confirm
  `test_alignment.py` shows Arm0/Arm1 share identical VN30 OOS keys.
- **Ablation caveat:** confirm the report states Arm 0 ≠ standalone-VN30 (different grid + market_pk)
  and that only Arm0-vs-Arm1 is compared.

## B. Reproducibility
Goal: a reviewer can regenerate every number.
- Follow `REPRODUCE.md` — commands should reproduce the JSONs in `results/…` (seeds fixed in
  `pipeline_config`/`training_config`). Spot-check one horizon reproduces the reported QLIKE.
- **Single source of truth:** grep the pipeline for hardcoded tunables outside
  `submission/soict_lstm_gat/pipeline_config.py` — the config-hardcode gate
  (`scripts/quality_gate/check_config_hardcode.py`) should report 0 for changed pipeline files.
- **Config coupling:** `_PROCESSED`↔`_PRICE_DIR`, VOLUME_ZSCORE_WINDOW=22, LOOKBACK — confirm the
  experiment lookback (22) is CLI-exposed, not silently forked from canonical (pc.LOOKBACK=10).
- **Determinism caveat:** different GPU/cuDNN → tiny FP differences within seed variation; note if the
  paper claims bitwise reproducibility.

## C. Paper-readiness
Goal: the claims are honest, supported, and defensible before a committee.
- Cross-check every claim in `CLAIMS.md` against `RESULTS_SUMMARY.md` and the JSONs — no number in a
  claim should exceed what the DM test supports (n.s. must not be reported as a win).
- **Baseline naming:** HAR-X must be justified as published (Corsi 2009 + Clements-Preve-Tee 2024;
  `docs/papers/README.md`), with the two disclosed deviations (range-based Parkinson target;
  direct single-day t+h, not h-day-average).
- **Honest framing:** the headline is a *negative/parsimony* result (HAR hard to beat; graph helps
  only VN100 short-h, loss-dependent). Confirm the reports state this without overclaiming.
- **Limitations section coverage:** not split-adjusted (overnight-tail appendix), σ² vs σ, floor
  sensitivity, VN30 small-N, single-market scope. All should be disclosed.
- **Ablation type:** CLAUDE.md mandates leave-one-out ablations; confirm the VolGA−LSTM (graph)
  contrast and the pooled train-universe ablation are framed correctly (not incremental-ladder).

## D. Code + test quality
- **Coverage:** changed-line C0=100 / C1≥95 enforced by pre-push step 2. Confirm tests are not
  vacuous (skip-guards `# pragma: no cover`, not silently skipping real assertions).
- **I/O runner tests + real-data smoke:** each `run_*` has an integration test (not only pure
  helpers); at least one test reads a real enriched slice (encoding/date/schema drift).
- **No silent degradation:** feature code must fail loud (raise) on missing input, not return zeros
  (`_check_feature_coverage`); confirm.
- **Performance:** the trainer is batched ([B,N,seq,5] on GPU, mask-aware loss); flag any batch=1
  hot loop / per-step host↔device sync (CLAUDE.md perf mandate). Note the pooled Arm 0 processes the
  full 102-node tensor (masked) — documented tradeoff, confirm it is intentional.
- **Isolation:** each baseline imports siblings read-only; no cross-baseline edits.

## Output format
List findings by severity (critical → minor), each: `dimension · file:line · what · why it matters ·
suggested fix`. End with a go / no-go recommendation for drafting the paper.
