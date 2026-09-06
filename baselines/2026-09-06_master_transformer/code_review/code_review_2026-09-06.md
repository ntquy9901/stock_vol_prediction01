# Adversarial code review — MASTER baseline (2026-09-06)

Self-review before hand-off (3 lenses: Blind Hunter, Edge-Case Hunter, Acceptance Auditor). Scope:
`code/master_net.py`, `code/market_features.py`, `code/master_config.py`, `code/run_master.py`,
`test/test_master.py`. `archive/` and other baselines are out of scope. The coordinator runs the full
`/code-review` + pre-push gate.

## Findings

### Leakage / fairness (highest priority)
- **F1 (checked, OK):** per-node target + feature scalers come from `pack_fold` (TRAIN-only, delivered,
  read-only). Market-feature scaler is fit on `mkt_raw[0 : last_train_anchor+1]` — every train input
  window ends at or before the last train anchor, so no val/test date enters the scaler. Test
  `test_market_scaler_fit_on_train_only` pins the row range.
- **F2 (checked, OK):** market features are causal — cross-sectional aggregates use only day t; the
  rolling z uses a trailing window ending at t. `test_causal_rolling_z_no_look_ahead` truncates the
  series and asserts z[t] is unchanged.
- **F3 (checked, OK):** the fold construction + purge reuse the delivered `make_folds` /
  `assert_no_leakage` (called in `run`), identical to VolGA/edge_hmatched.
- **F4 (checked, OK):** the VolGA edge is the horizon-matched train-only edge
  (`directed_vol2pk_hmatched`, built with `last_tr_row = last_train_anchor + horizon`), same as
  edge_hmatched. The MASTER gate reads market features at the last timestep only (t), causal.
- **F5 (design choice, documented):** the dense SAttention masks invalid (non-trading) stocks as keys
  via `key_mask=nmask` so valid stocks never attend to zero-padded stocks — this is the fair analogue
  of VolGA's masked adjacency. Without it, zero-padded stocks would dilute the cross-sectional softmax.

### Correctness
- **F6 (checked, OK):** the batched attention does not mix anchors —
  `test_per_anchor_independence_of_batched_attention` perturbs one anchor and asserts the other's output
  is unchanged. The dense inter-stock attention only mixes within one anchor's cross-section.
- **F7 (checked, OK):** positivity is enforced by inverse-transform + a shared positivity floor (no
  Softplus/ReLU) — the proven pattern; `test_floor_is_positive_and_linear_inverse` and the end-to-end
  test assert strictly positive predictions. QLIKE uses the single shared `qlike_floor=1e-8`, identical
  to HAR/HAR-X/LSTM/VolGA (fairness).
- **F8 (edge case, handled):** an invalid target attending to an all-masked key set would produce a NaN
  softmax; `torch.nan_to_num` zeroes it and the loss/metrics discard invalid targets
  (`test_all_but_one_masked_key_is_finite`).
- **F9 (edge case, handled):** all-NaN cross-sectional rows -> neutral 0 market feature (not NaN)
  (`test_cross_sectional_handles_all_nan_row`); `standardize` maps any residual non-finite to 0.

### Performance (mandate)
- **F10 (fixed):** the initial loop rebuilt the concatenated `[A,N,T,D]` input every epoch/inference
  call (large host allocation) — the batch/GPU-underutilization anti-pattern. Fixed: the concatenated
  inputs + key masks are built ONCE per split as GPU tensors; the epoch loop and per-epoch train/val
  learning-curve inference index minibatch views. Anchors are batched (`batch_size` up to 256), tensors
  stay on GPU, no per-item batch=1. (Root cause of an observed slowdown was actually ZOMBIE GPU
  processes from killed chains contending — resolved operationally; the loop fix stands on its own.)

### Config / hygiene
- **F11 (checked, OK):** all MASTER tunables live in `master_config.py`; floors/windows/seeds/lookback
  come from canonical `pipeline_config`. `min_epochs` sources from `pc.MIN_EPOCHS`. No magic numbers in
  the net/driver. Post-generate config-hardcode gate passes.
- **F12 (checked, OK):** `run()`/`main()`/`_git_commit`/`_versions` are thin `# pragma: no cover`
  drivers; substantive logic (train_master, run_fold, helpers) is exercised by tests (no hidden logic).
- **F13 (checked, OK):** hard isolation — only read-only imports from other baselines / submission; no
  other baseline or `src/` file is modified. MASTER nn.Module is vendored into `master_net.py` with an
  attribution header (arXiv:2312.15235 + MIT SJTU-DMTai repo).

## Limitations noted for the report
- LSTM/VolGA are retrained in-run for exactly-paired DM; their numbers are cross-checked against the
  delivered `results/edge_hmatched/` (same panel/folds/seeds/lb10/floor).
- SP500 skipped locally (heavy; Colab A100) — noted.
- Market features are a compact causal set (4); a richer market panel is a possible extension but the
  prior (graph/attention fails on this target) makes further market-feature engineering low-priority.

## Verdict
No unresolved critical/major findings. Fairness/leakage/positivity guards are tested. Ready for the
coordinator's `/code-review` + pre-push gate.
