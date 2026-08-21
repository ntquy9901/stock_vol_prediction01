# Adversarial Code Review — HAR-Anchored LSTM–GAT Residual Study (E0–E10)

Date: 2026-08-22
Scope reviewed: `baselines/2026-08-21_har_anchored_residual/code/` — `folds.py`, `har_cv.py`,
`io_preds.py`, `models.py`, `experts.py`, `blend.py`, `gate.py`, `run_experiment.py`,
`build_report.py`. Light check only: `stats.py`, `diagnostics.py` (TDD'd in a prior pass).
Reused read-only (context): `submission/soict_lstm_gat/{snapshots,baselines,edges,metrics,model,config}.py`.
Method: 3-layer (Blind Hunter + Edge Case Hunter + Acceptance Auditor). No code modified; GPU training
not run (GPU busy with live runs).

Invariants judged against: `reports/leakage_audit.md`, the baseline `requirements/` + `design/`, and
`docs/experement_guide/HAR_Anchored_LSTM_GAT_Experiment_Plan.md`.

## Severity counts
- CRITICAL: 1
- MAJOR: 1
- MINOR: 4
- Positive verifications (no defect): leakage controls, reconstruction scaling, QLIKE-floor parity,
  gate observability.

---

## CRITICAL

### C-1 — Report and hypothesis decisions consume the NAIVE row-level DM, not the date-clustered DM (violates invariant 5)
Files: `run_experiment.py:127-128`, `build_report.py:23-24, 44, 63-66, 71-77`.

Both DM variants are computed and stored per model:
```
dm[name] = {"dm_qlike":     _dm_dict(M.diebold_mariano(lq, lh, h=horizon)),   # per-observation (row-level)
            "date_clustered": ST.date_clustered_dm(lq, lh, dates, horizon)}    # correct, one value/date
```
`build_report._dmp` reads only `dm_qlike` (the per-observation series over every `(node, date)` row),
and `sig_better` uses that p-value to decide the "DM p-value" column and the H1/H4/H5 ACCEPT/REJECT
verdicts. The panel is cross-sectionally dependent (all `N` tickers share each `target_date`), so the
per-observation DM treats `n = N × T_dates` as the sample size and understates the loss-differential
variance.

Concrete failure scenario: VN100 with `N ≈ 71` surviving nodes and `T_dates` test dates gives
`n ≈ 71 · T`. The HLN/HAC variance scales as `1/n`, so the DM statistic is inflated by roughly
`sqrt(71) ≈ 8.4×` relative to the date-clustered statistic. A QLIKE gap that is a coin-flip once
collapsed to one value per date will report `p < 0.001` at the row level. The headline "E6 residual GAT
beats HAR at h10/h22 (significant)" therefore rests on an over-stated p-value; the plan's mandated
date-clustered stat that would judge it fairly is computed and then discarded by the report.

Note the internal inconsistency this produces in a single `result.json`: the Model Confidence Set and
block-bootstrap CI (`stats.py`) correctly aggregate to one value per date, while the DM column does not.
A reader can see MCS keep/eliminate a model that the DM column simultaneously calls highly significant.

Fix: point `build_report._dmp` (and `sig_better`) at
`dm_vs_har[name]["date_clustered"]["p_value"]`; keep the row-level value, if at all, only as an
explicitly-labelled secondary column. Re-generate `reports/experiment_results.md` before any
significance is quoted.

---

## MAJOR

### M-1 — "Graph adds value" (E6/E7 vs E5) is asserted from raw QLIKE point gaps with no inferential test
Files: `run_experiment.py:119-130`, `build_report.py:71-74`.

DM, block-bootstrap and MCS are all run **only against E0_HAR** (`if name != "E0_HAR": dm[name] = …`).
There is no paired test between the graph and no-graph residual experts. `build_report.graph_wins`
declares H4 (cross-sectional/graph incremental value) by `E7.qlike < E5.qlike` combined with
`sig_better(E7)` — but `sig_better(E7)` tests E7 vs HAR, not E7 vs E5. So the marginal graph
contribution is judged by (a) an unpaired point-QLIKE inequality and (b) a significance test against the
wrong baseline.

Concrete failure scenario: E7 beats HAR because of its LSTM branch while the GAT branch contributes
nothing; `E7.qlike` edges below `E5.qlike` by sampling noise. H4 is ACCEPTed even though a paired
DM/bootstrap of `per_obs_qlike(E6/E7) − per_obs_qlike(E5)` (date-clustered) would be indistinguishable
from zero. The task's stated headline "E6 beats E5" has, in the current code, no significance test at
all — only a raw mean-QLIKE comparison.

Fix: compute date-clustered DM and block-bootstrap CI for the specific contrast the claim needs
(E6−E5 and E7−E5) on the shared test keys, and drive H4 from that paired stat rather than from a
vs-HAR test.

---

## MINOR

### m-1 — Residual experts (E5–E8) train on fewer rows than E1/E2 (first CV block dropped)
Files: `experts.py:79-80, 128-134`. Residual framings train on `rows = D["cf_mask"]`, which drops the
first cross-fit block (~1/`n_cvfolds` ≈ 20% of train dates, the seed block with NaN OOS HAR). The full
framings (E1/E2) train on all train rows (`rows = ones`). Test comparability is preserved (all experts
scored on the identical intersected test keys), and the asymmetry, if anything, *handicaps* the residual
experts, so it cannot manufacture E6 > HAR. Worth a one-line note in the results report for honest
same-fold framing.

### m-2 — Residual target base ≠ reconstruction base (cross-fit HAR vs full-train HAR)
Files: `experts.py:79-84` (`oos_add/oos_mul` built from cross-fit `oos`), `experts.py:109-115` /
`models.py:56-67` (reconstruction adds the correction onto `harp_*`, the full-train HAR). The net learns
`y − oos_har` but at inference the correction is added to `harp` (full-train HAR). The two HAR fits
differ only by the small expanding-window vs full-train coefficient gap, and the offset is identical
across E5–E8, so it cannot create a spurious graph win; it does introduce a small systematic base offset
into every residual reconstruction. Design-justified (honest OOS residual target) but should be
acknowledged.

### m-3 — `build_report` H3 treats an exactly-zero residual R² as −1 (falsy-`or` bug)
File: `build_report.py:69`. `(r["diagnostics"]["residual_r2_oos"].get(e, -1) or -1) > 0` — when the
stored value is exactly `0.0`, `0.0 or -1` evaluates to `-1`. Harmless in practice (an exact 0.0 is
measure-zero and the hypothesis is `> 0` anyway), but the idiom is wrong; use an explicit
`v is None` check.

### m-4 — `_train_ensemble` returns `nparam` from the last seed only; cosmetic seed-mean of normalized `c`
Files: `run_experiment.py:54-60`. `nparam` is overwritten each seed and only the last kept (all seeds
share architecture, so the value is correct but the code reads as accidental). Separately, `corr_mean`
averages the *normalized* per-node correction `c` across seeds before feeding E9/E10/diagnostics; this is
a defensible ensemble choice but should be stated (the gate/diagnostics see the seed-mean correction, not
a per-seed distribution).

---

## Positive verifications (checked, no defect)

- **Temporal purge is correct.** Anchors are consecutive trading days (`snapshots.py:79`); dropping the
  last `horizon` train and val snapshots (`experts._purge_snapshots`) makes each split's last target
  date (`anchor+horizon`) land strictly before the next split's first anchor. `folds.purged_split`
  encodes the same rule for the per-stock design. No train/val target interval crosses a boundary.
- **All fits are train-only.** Graphical-lasso adjacency (`edges.glasso_adjacency` on
  `snap.adj_pk_train`), per-node feature/target scalers (`snapshots.py:84-91`), and the pooled HAR anchor
  (`experts.py:73`) are estimated on train rows only. The pre-purge scaler statistics survive only for
  the *feature* normalization of train-range inputs (no val/test rows), and `build_data` recomputes the
  target mean/std on post-purge train — no val/test bleed.
- **Residual targets are honest OOS.** `_crossfit_har_pooled` predicts each train date-block from
  strictly-earlier blocks; the first (seed) block is NaN and excluded via `cf_mask`. No in-sample HAR is
  used to build residual targets.
- **α (E3/E4) and λ (E9) are fit on validation only** (`blend.blend`, `gate.fit_lambda_static`);
  E10's dynamic gate trains on train QLIKE and early-stops on validation. Test is read once for the final
  reconstruction. No α/λ path touches the test set.
- **HAR-anchored fallback holds.** `models.ResidualNet` zero-inits the final head weight and bias, so
  `c == 0` at init; additive → `har`, multiplicative → `(har+eps)·exp(0) = har+eps > 0`. Multiplicative
  reconstruction is strictly positive by construction.
- **Reconstruction scaling is dimensionally consistent.** Per-node `add_scale`/`mul_scale` `[N]`
  broadcast correctly over `[n, N]` corrections in numpy and torch paths (`experts._reconstruct`,
  `gate.reconstruct`, `gate.recon_torch` via `np.broadcast_to`).
- **QLIKE floor parity (invariant 4).** HAR predictions are floored at `cfg.qlike_floor` in
  `har_predict`, neural reconstructions are clamped to `≥ 0` then the QLIKE metric applies the same
  `floor = cfg.qlike_floor` to every model; all E1/E2/E5/E6/E7/E8/GARCH share it. Additive
  negative-clip → floor blow-up is an honest model limitation, not a scoring asymmetry.
- **Gate uses only observable-at-`t` state.** `gate._state_features` = market HAR level, cross-sectional
  HAR dispersion, node HAR level, |correction| — all derived from the day-`t` HAR forecast and the frozen
  net's day-`t` correction; no target or future information.
- **Windowing/target indexing.** Anchor `t` → target `pk[t+horizon]`, `date = dates[t+horizon]`, with
  `anchors = range(first_valid+lookback-1, T-horizon)` keeping `t+horizon ≤ T-1`. No off-by-one.
- **stats.py (light check).** `date_clustered_dm`, `block_bootstrap_ci`, and `model_confidence_set` all
  collapse to one value per unique date before inference; ISO `%Y-%m-%d` string dates sort chronologically
  under `np.unique`; RNG is local. These are the correct dependent-panel treatments — they are simply not
  the series the report table reads (see C-1).

---

## Trust verdict — "VN100 GAT residual E6 beats HAR at h10/h22, and beats E5"

Split the claim into point estimate and significance:

- **Point-estimate ranking is trustworthy.** No leakage or scaling defect was found that could inflate
  test performance or flip the QLIKE ordering. E5 and E6 share data, seeds, folds, HAR anchor, QLIKE
  floor and reconstruction path, differing only in the LSTM-vs-GAT branch, so a measured
  `QLIKE(E6) < QLIKE(E5) < QLIKE(HAR)` on the common test keys is a genuine measurement.

- **Significance as reported is NOT trustworthy.** The p-values surfaced in `reports/experiment_results.md`
  and the H1/H4/H5 verdicts use the naive row-level DM (C-1), which over-states significance by an
  `~sqrt(N)` factor on this cross-sectionally dependent panel; and the specific "beats E5" contrast has no
  paired test computed at all (M-1). A bug cannot manufacture the point win, but the *statistical* claim
  ("significantly beats HAR / beats E5") is unsupported by the code as wired.

Recommendation before quoting the headline: re-point the report at the already-computed
`date_clustered` DM (C-1), add a date-clustered paired DM/bootstrap for E6−E5 and E7−E5 (M-1), and check
that E6 remains in the MCS. If the date-clustered p-values and MCS membership survive, the claim stands;
if they do not, the graph win is within noise despite the favorable point QLIKE.
