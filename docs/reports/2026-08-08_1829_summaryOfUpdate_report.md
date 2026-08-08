# T1.2 — Parsimony story on the POOLED regime

Date: 2026-08-08. Branch: `feature/pooled-news-gnn-pilot` (worktree). Horizon 5, 3 seeds
(42/123/2026), 5 epochs, validation metrics.

## Scope

Characterise the parsimony verdict on the pooled architecture using the A1 (T1.1) pooled
cells. No GPU run was required: P0–P3 pooled results already exist for all 3 seeds under
`results/a1_pooled_seed{42,123,2026}/h5/{P0,P1,P2,P3}/results.json` (12/12 present, all with
non-empty `validation_metrics`). These were reused as instructed.

Cell semantics (from `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/train.py`
and `run_pilot.py`):

- P0 = HAR linear reference
- P1 = pooled price-only LSTM (`PooledPriceLSTM`)
- P2 = pooled price+news LSTM (`PooledPriceNewsLSTM`, `use_gate=False`)
- P3 = P2 + per-ticker news gate (`use_gate=True`)

Contrasts: news effect = P2 vs P1; gate effect = P3 vs P2; vs HAR = P1/P2/P3 vs P0.

## What changed (files)

| Path | Purpose |
|------|---------|
| `docs/reports/2026-08-08_t12_parsimony_pooled.py` | Reproducible aggregation/analysis script (mean±std, paired-t df=2, per-seed sign consistency). |
| `docs/reports/test_t12_parsimony_pooled.py` | Pytest: hand-computed checks for `mean_std`/`paired_t`/`sign_consistency` + integration test on the real JSON. |

No result data was generated or modified.

## 3-seed cell table (validation, mean ± std over seeds)

| metric | P0 (HAR) | P1 (price) | P2 (+news) | P3 (+gate) |
|--------|----------|------------|------------|------------|
| mse    | 0.000002±0.000000 | 0.000002±0.000000 | 0.000002±0.000000 | 0.000002±0.000000 |
| rmse   | 0.001485±0.000000 | 0.001502±0.000004 | 0.001487±0.000003 | 0.001489±0.000004 |
| mae    | 0.000480±0.000000 | 0.000490±0.000002 | 0.000480±0.000003 | 0.000482±0.000003 |
| r2     | 0.735146±0.000000 | 0.728734±0.001483 | 0.734357±0.001214 | 0.733641±0.001349 |
| qlike  | 0.516707±0.000000 | 0.511839±0.001102 | 0.508392±0.000242 | 0.508564±0.000217 |
| dir_acc (%) | 48.540±0.000 | 48.588±0.042 | 48.470±0.135 | 48.580±0.068 |

P0 (HAR) is identical across seeds because the linear reference is deterministic; its per-seed
std is 0 and the vs-HAR paired-t is a one-sample test of the deep values against a constant.

Two-tailed critical values for df=2: t(.05)=4.303, t(.10)=2.920.

## Contrast tables (mean_diff = treatment − reference; paired-t df=2; sign = seeds improving)

### News effect (P2 vs P1)

| metric | mean_diff | paired-t | sign | sig @.05 |
|--------|-----------|----------|------|----------|
| mse   | -4.68e-08 | -4.810 | 3/3 improve | yes |
| rmse  | -1.57e-05 | -4.814 | 3/3 improve | yes |
| mae   | -1.01e-05 | -5.694 | 3/3 improve | yes |
| r2    | +5.62e-03 | +4.810 | 3/3 improve | yes |
| qlike | -3.45e-03 | -6.938 | 3/3 improve | yes |
| dir_acc | -1.18e-01 | -1.184 | 1/3 improve | no |

### Gate effect (P3 vs P2)

| metric | mean_diff | paired-t | sign | sig @.05 |
|--------|-----------|----------|------|----------|
| mse   | +5.96e-09 | +3.075 | 0/3 improve | no |
| rmse  | +2.00e-06 | +3.081 | 0/3 improve | no |
| mae   | +1.62e-06 | +3.540 | 0/3 improve | no |
| r2    | -7.16e-04 | -3.075 | 0/3 improve | no |
| qlike | +1.71e-04 | +1.053 | 1/3 improve | no |
| dir_acc | +1.10e-01 | +2.450 | 3/3 improve | no |

### vs HAR (P1/P2/P3 − P0)

| contrast | metric | mean_diff | paired-t | sign | sig @.05 |
|----------|--------|-----------|----------|------|----------|
| P1−HAR | qlike | -4.87e-03 | -7.651 | 3/3 improve | yes (better) |
| P1−HAR | rmse  | +1.79e-05 | +7.533 | 0/3 improve | yes (worse) |
| P1−HAR | r2    | -6.41e-03 | -7.487 | 0/3 improve | yes (worse) |
| P1−HAR | mae   | +1.07e-05 | +8.785 | 0/3 improve | yes (worse) |
| P1−HAR | dir_acc | +4.75e-02 | +1.948 | 3/3 improve | no |
| P2−HAR | qlike | -8.31e-03 | -59.502 | 3/3 improve | yes (better) |
| P2−HAR | rmse  | +2.21e-06 | +1.125 | 1/3 improve | no (tie) |
| P2−HAR | r2    | -7.89e-04 | -1.126 | 1/3 improve | no (tie) |
| P2−HAR | dir_acc | -7.05e-02 | -0.905 | 1/3 improve | no |
| P3−HAR | qlike | -8.14e-03 | -65.081 | 3/3 improve | yes (better) |
| P3−HAR | rmse  | +4.21e-06 | +1.935 | 0/3 improve | no (tie) |
| P3−HAR | r2    | -1.51e-03 | -1.933 | 0/3 improve | no (tie) |
| P3−HAR | dir_acc | +4.00e-02 | +1.017 | 2/3 improve | no |

## Verdict on the POOLED architecture

**(a) News helps — YES.** Adding news (P2 vs P1) improves every error/fit metric — MSE, RMSE,
MAE, R², QLIKE — in all 3 seeds, with paired-t exceeding the df=2, α=.05 critical value
(4.303) on all five (|t| 4.81–6.94). There is no directional-accuracy benefit (1/3, t=-1.18,
ns). The paper's main positive (news improves QLIKE/RMSE) therefore survives on the pooled
architecture, as an error-metric result, not a DirAcc one.

**(b) The per-ticker gate is inert — YES (null confirmed).** P3 vs P2 shows no error-metric
improvement in any seed (0/3 on MSE/RMSE/MAE/R², signs marginally worse), QLIKE a tie (1/3,
t=1.05), and only a small non-significant DirAcc nudge (3/3 but t=2.45 < 4.303). The gate adds
nothing measurable.

**(c) Does any deep model beat HAR — only on QLIKE.** All three deep configs beat HAR's QLIKE
decisively (3/3, |t| 7.7–65). On RMSE/MSE/MAE/R², HAR remains competitive-to-better: the
price-only P1 is significantly worse than HAR (0/3, |t|≈7.5); adding news (P2, P3) closes that
gap to a statistical tie (no significant difference; HAR marginally ahead on the mean). DirAcc
sits at ~48.5% for every cell — below 50%, consistent with the documented anti-persistence
ceiling — with no significant differences.

Net parsimony reading: HAR is hard to beat on RMSE/R². The deep stack's news term does real
work — it recovers the RMSE/R² that the price-only deep model lost versus HAR and delivers a
genuine, large QLIKE win — but it does not establish deep-model dominance on the standard error
metrics.

## Verification (real evidence)

Pytest (CPU `python`):
```
$ python -m pytest docs/reports/test_t12_parsimony_pooled.py -q
......                                                                   [100%]
6 passed in 0.09s
```

Ruff:
```
$ ruff check docs/reports/2026-08-08_t12_parsimony_pooled.py docs/reports/test_t12_parsimony_pooled.py
All checks passed!
```

Analysis numbers above are the verbatim output of
`python docs/reports/2026-08-08_t12_parsimony_pooled.py`.

## Caveats

- n = 3 seeds, 5 epochs, horizon 5 — a screening signal, not a final result. df=2 paired-t is
  low-powered; treat significance flags as directional.
- Validation-split metrics (the A1 pooled screening reports `validation_metrics`); no test-split
  claim is made here.
- Coverage gate (`diff-cover --fail-under=100`) not run: the tooling gap noted in CLAUDE.md
  §Per-project setup persists; the two new files are fully exercised by the 6 tests above
  (helpers hand-checked, `build_report` integration-tested on real JSON).

## DoD checklist

- [x] Reused existing P3 pooled cells (no redundant GPU run); confirmed 12/12 present.
- [x] Parsimony table with paired-t and per-seed sign consistency.
- [x] Tests green (6 passed) + ruff clean — output pasted.
- [x] Reproducible script under `docs/reports/`.
- [x] Summary report (this file).
- [ ] Code review (`/code-review`) — deferred to parent per task instruction (parent verifies).
- [ ] Push — NOT done per task instruction (parent verifies + pushes).
