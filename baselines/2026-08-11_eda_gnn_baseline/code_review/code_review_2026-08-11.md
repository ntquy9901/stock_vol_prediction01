# Code review — EDA-recommended GNN baseline (2026-08-11)

Adversarial review (3-layer: Blind Hunter + Edge-Case Hunter + Acceptance Auditor) of the new
in-scope code: `features.py`, `edges.py`, `eda_model.py`, `eda_ladder.py`, `aggregate.py`. Pilot
files read for contract only (out of scope). `archive/` excluded per project rule. 2 HIGH, 2 MEDIUM,
6 LOW.

## HIGH — fixed

### H1 — Val/test leakage into the "TRAIN-frozen" vol2pk adjacency (global date boundary)
`edges._train_panels` restricted each ticker's FULL series to a single global `train_end_date`
(= max train target date across tickers). Splits are per-ticker chronological, so any ticker whose
own train ends earlier contributed its validation/test volume-shock + PK observations to the
`corr(vshock_i(t), sqrt(PK)_j(t+1))` estimate that defines the frozen Top-5 topology used for
per-sample message passing. Direct leakage; violates acceptance criterion 1.
- **Fix:** `_train_panels` now builds the panels from each ticker's own `["train"]` split (no global
  boundary). Added `test_adjacency_no_leak_with_heterogeneous_histories` (short- and long-history
  tickers) asserting the frozen topology is unchanged when every ticker's val/test rows are bumped.

### H2 — QLIKE E3-vs-HAR confounded by an asymmetric positivity floor
Graph rungs (E3/E3off/G1corr) applied the per-ticker denormalized floor (`raw >= ~1e-6`) inside the
model; E0 (HAR linreg) and E1/E2 (`PooledPriceLSTM`) did not. QLIKE is dominated by tiny-prediction
tail spikes, so E3's floor could beat HAR on QLIKE purely from flooring — exactly the primary
verdict `E3_vs_E0`.
- **Fix:** added `_floor_norm_records` (identical per-ticker denormalized floor) applied to E0, E1,
  and E2 predictions before evaluation, so every rung is floored identically. Also removes the E1/E2
  nonpositive-gate crash risk (M1).

## MEDIUM — fixed / addressed

### M1 — E1/E2 could crash mid-run on the strict nonpositive gate
`PooledPriceLSTM` had no floor; if >1% of raw predictions were nonpositive, `evaluate_records`
raised and aborted the seed. Resolved by the H2 floor (all predictions now `>= ~1e-6 > 0`).

### M2 — obs-set parity vs the pilot never guarded at runtime
Bit-identity of the eligible-window set silently depended on the extras never being non-finite on an
otherwise HAR-valid row.
- **Fix:** `ExtendedTickerPreprocessor.transform_frame` now asserts `valid_rows == HAR-only mask`
  (raises "observation drift" otherwise). Also verified offline: 5-feature pooled manifest is
  byte-identical to the pilot 3-feature manifest (train 73026 / val 14418 / test 14464; HAR feature
  columns bit-identical).

## LOW

- **L1 (noted, not a bug):** the residual message-passing weights Top-5 neighbours by
  `softmax(signed corr)` with a self-loop fixed at 1.0, which concentrates mass on self and can
  under-weight weak/negative directed edges. A null E3-vs-E0 result could partly reflect this
  weighting rather than pure absence of spillover — this is the SAME message-passing block the pilot
  G1 uses, so E3-vs-G1corr remains a fair edge-only comparison. Noted as a caveat in the results
  report; edge-weight tuning is out of scope.
- **L3 — fixed:** `aggregate._ensemble` now asserts raw targets agree across seeds (not just key
  coverage).
- **L2 (accepted):** the pooled DM HAC lag treats the (ticker_id, target_date)-ordered loss vector
  as one series; mild variance misspecification at ticker boundaries, inherited from the pilot DM
  pattern. Documented.
- **L4 (wording):** the frozen-topology invariant holds on the pre-mask matrix; `swap_adjacency`
  zeroes absent rows/cols per snapshot by design. No code change.
- **L5 / L6 (accepted):** `_local_date_key` strips tz without shifting (correct for the VN +07:00
  wall-midnight timestamps); `groupby-last` collapses duplicate raw OHLCV dates (benign; raw files
  are clean). Documented follow-ups.

## Confirmed correct (by the reviewer)
- `volume_zscore_20` trailing/causal; `market_pk` contemporaneous with train-only standardiser;
  adjacency direction `A[target, source]` matches the message-passing convention; presence masking
  isolates absent nodes; `apply_message_passing=False` equals the no-graph readout; DM per-obs
  alignment sound (records order preserved into targets/predictions, re-keyed by (id,date) with
  equality guards); DM sign convention correct; best-val checkpoint selection correct; no bare
  excepts / mutable defaults.

## Status
All HIGH and MEDIUM findings fixed; L3 fixed; L1/L2/L4/L5/L6 accepted with documentation. 15 pytest
tests pass; ruff clean. Cleared for the real 3-seed run.
