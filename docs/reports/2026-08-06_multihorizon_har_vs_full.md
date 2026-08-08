# Multi-horizon HAR vs news-fusion (per-ticker gate): re-run on the corrected pipeline

Scope: re-run the 1-day, 10-day, and 22-day horizon-extension baselines on the current,
bug-fixed pipeline (P1.1 normalizer leakage, P1.2 cross-stock date misalignment, P1.3 DirAcc
per-ticker formula — all fixed 2026-08-02/03; see
`docs/reports/2026-08-03_final_paper_readiness_report.md` §3). The prior horizon results
(2026-08-01) predate those fixes and were flagged non-citable. This report supersedes them and
adds the existing 5-day numbers to give a complete 1/5/10/22-day comparison.

## 1. Bugs found and fixed in the horizon scripts

The six horizon scripts (`train_har_only_reference_h{1,10,22}.py`,
`train_per_ticker_gate_h{1,10,22}.py` under `baselines/2026-08-01_horizon*_baseline/code/`) are
copy-modifications of the 5-day lineage. Per-script inspection against the proven 5-day equivalents
(`baselines/2026-07-26_per_ticker_news_gate_baseline/code/train_per_ticker_gate.py`) found:

| Check | Status in horizon scripts |
|---|---|
| P1.1 normalizer applied (`.transform()` in `__getitem__` + inverse-transform in `validate`) | Already correct — inherited via read-only import of the P1.1/P1.2-fixed `dataset_dual_news` → `dataset_presplit`/`dataset_with_graph_method` pipeline. |
| P1.2 cross-stock date alignment (`_reindex_to_common_dates`, winsorize-not-drop) | Already correct — same shared pipeline; data is the current 33-ticker `data/processed/`. |
| P1.3 DirAcc per-ticker (`evaluate_predictions(..., n_stocks=n_stocks)`) | Already correct — all six pass `n_stocks=`, so `directional_accuracy` is the per-ticker value (flat-biased kept separately). |
| HAR-baseline per-ticker split bug (`src/har_baseline`) | Not applicable — the "HAR-only reference" here is the neural `ParallelLSTMGNN` backbone, not the linear `src/har_baseline/train.py`. |
| `--seed` CLI + multi-seed reproducibility | **Missing (defect).** All six hardcoded `torch.manual_seed(42)`/`np.random.seed(42)` with no `--seed` arg, so 3-seed runs were impossible. |
| `seed` / `provenance` in `results.json` | **Missing (defect).** No seed field; gate scripts lacked `get_provenance()`. |

Fix applied to all six scripts (mirroring the proven 5-day pattern exactly): added `--seed` argparse
(default 42), routed it into `torch.manual_seed`/`np.random.seed`, and recorded `seed` +
`provenance: get_provenance()` in `results.json`. The pre-fix scripts were otherwise sound; no
P1.1/P1.2/P1.3-class bug remained in them (those were already fixed upstream in the shared,
read-only dataset code).

One protocol adjustment: both script families capped `MAX_EPOCHS = 10`. To run the authorized
20-epoch budget (matching the 5-day protocol), the cap was raised to 20 in all six scripts and each
run was executed as a single clean 20-epoch run (early stopping `min_epochs=20` keeps it running the
full 20). This is symmetric for HAR and FULL (both clean 20-epoch), rather than the 5-day gate's
10+10 resume chain — a cleaner, fully symmetric fair-budget comparison. The `ParallelLSTMGNN`
HAR-only reference is a stochastic neural model (random init + shuffled loader), so it was run with
3 seeds too (not once), so both sides of every horizon carry mean±std over the same seeds
(42/123/2026).

## 2. Protocol

- Data: current `data/processed/` (33 tickers, P1.2-fixed), news panel
  `data/features/dual_group_news_panel.parquet` for the FULL model; `news_panel_path=None` for HAR.
- Both models: `ParallelLSTMGNN` HAR backbone (3 HAR features), `knn` graph, 20 epochs, lr 5e-3,
  batch 32, dropout 0.5, MSE loss; FULL adds the per-ticker news gate.
- Seeds: 42, 123, 2026 (paired across HAR and FULL). 3 horizons × 2 models × 3 seeds = 18 runs, CPU.
- Metrics on denormalized (physical) scale; DirAcc is the corrected per-ticker value.
- Provenance (`git_sha`, seed) recorded in every `results.json`.

## 3. Four-horizon comparison (test set, mean ± std over 3 seeds)

5-day column reproduced from `docs/reports/2026-08-03_final_paper_readiness_report.md` §1 (same
3-seed protocol, same corrected pipeline). Lower is better for QLIKE/RMSE/MAE; higher for R²/DirAcc.
Bold marks the better mean at each horizon.

### QLIKE
| Horizon | HAR-only | FULL (news gate) | FULL − HAR (paired) |
|---|---|---|---|
| 1-day  | **0.3778 ± 0.0100** | 0.3837 ± 0.0225 | +0.0059 (t=0.33, ns) |
| 5-day  | 0.4603 ± 0.0205 | **0.4430 ± 0.0185** | −0.0173 (t=−6.22, sig) |
| 10-day | 0.5240 ± 0.0168 | **0.5186 ± 0.0168** | −0.0054 (t=−0.96, ns) |
| 22-day | **0.5262 ± 0.0068** | 0.5675 ± 0.0370 | +0.0412 (t=1.64, ns) |

### RMSE
| Horizon | HAR-only | FULL (news gate) | FULL − HAR (paired) |
|---|---|---|---|
| 1-day  | **0.002501 ± 0.000089** | 0.002506 ± 0.000182 | +0.000005 (t=0.03, ns) |
| 5-day  | 0.002923 ± 0.000090 | **0.002734 ± 0.000096** | −0.000189 (t=−9.38, sig) |
| 10-day | 0.003093 ± 0.000048 | **0.003063 ± 0.000036** | −0.000030 (t=−3.69, consistent) |
| 22-day | **0.002955 ± 0.000021** | 0.003066 ± 0.000087 | +0.000111 (t=2.05, borderline) |

### R²
| Horizon | HAR-only | FULL (news gate) |
|---|---|---|
| 1-day  | **0.8345 ± 0.0117** | 0.8334 ± 0.0241 |
| 5-day  | 0.7749 ± 0.0140 | **0.8031 ± 0.0139** |
| 10-day | 0.7470 ± 0.0080 | **0.7519 ± 0.0058** |
| 22-day | **0.7540 ± 0.0035** | 0.7351 ± 0.0150 |

### MAE
| Horizon | HAR-only | FULL (news gate) |
|---|---|---|
| 1-day  | **0.000724 ± 0.000010** | 0.000728 ± 0.000030 |
| 5-day  | 0.000811 ± 0.000014 | **0.000793 ± 0.000012** |
| 10-day | 0.000849 ± 0.000007 | **0.000845 ± 0.000005** |
| 22-day | **0.000846 ± 0.000006** | 0.000854 ± 0.000004 |

### Directional accuracy (per-ticker, corrected)
| Horizon | HAR-only | FULL (news gate) |
|---|---|---|
| 1-day  | 32.47 ± 0.30 | **33.32 ± 0.62** |
| 5-day  | **48.47 ± 0.35** | 47.77 ± 0.52 |
| 10-day | 47.23 ± 1.88 | **47.72 ± 1.44** |
| 22-day | 42.05 ± 5.62 | **43.56 ± 6.50** |

DirAcc sits near or below the 50% random baseline at every horizon, with large seed variance at
10/22-day (±1.4–6.5 pp). This matches the project's documented DirAcc caveat
(`docs/reports/2026-08-04_diracc_low_accuracy_analysis.md`): per-ticker day-over-day sign is close
to unpredictable here, so DirAcc is not a reliable discriminator between HAR and FULL and no DirAcc
difference in the table reaches significance at n=3.

## 4. Does "news helps at short horizons, not long horizons" hold on corrected data?

**Partially — the long-horizon half is confirmed; the short-horizon half is refined, not confirmed.**
The news benefit on continuous-error metrics (QLIKE/RMSE/R²/MAE) is **non-monotonic in horizon and
peaks at 5-day**, not at the shortest horizon:

- **1-day: no benefit (tied/inconclusive).** QLIKE, RMSE, R², MAE differences are all within one
  standard deviation and change sign across seeds (QLIKE t=0.33, RMSE t=0.03). News neither helps
  nor hurts magnitude forecasting at 1-day. Only DirAcc nudges up (+0.85 pp, t=2.74, still below the
  n=3 significance threshold t≈4.30).
- **5-day: clear, statistically significant benefit.** FULL improves QLIKE (t=−6.22) and RMSE
  (t=−9.38) — the only horizon where the improvement is significant. This is the project's primary
  target and the strongest news effect.
- **10-day: marginal benefit.** FULL is better on RMSE and R² in all three seeds (t=−3.69 / +3.59,
  consistent direction, just under the n=3 significance bar) but the magnitudes are small
  (ΔRMSE ≈ 3e-5) and QLIKE is not significant (t=−0.96). News still helps a little.
- **22-day: net harmful.** FULL is worse on QLIKE (+0.041), RMSE (+0.000111, t=2.05), R² (−0.019),
  and MAE. The effect is not strictly significant at n=3 (seed 123 is a near-tie outlier) but the
  direction — news degrading accuracy — is consistent on 2 of 3 seeds and across four different
  metrics.

So the broad shape "news value decays as the horizon lengthens and turns negative by 22-day" is
supported on the corrected pipeline: benefit goes clear (5d) → marginal (10d) → harmful (22d). What
is **not** supported is the naive reading that the shortest horizon benefits most — at 1-day news
adds nothing. The honest summary is that news fusion helps most at a short-to-medium horizon
(5-day sweet spot), provides a small residual help at 10-day, and is counterproductive at 22-day;
the 1-day horizon is a wash. Given n=3 seeds, only the 5-day result carries statistical
significance; the 1/10/22-day conclusions are directional trends (consistent across seeds/metrics
but below the strict significance threshold), and would benefit from ≥5 seeds before being stated
as hard claims in the paper.

## 5. Provenance

18 runs, all with `git_sha` + `seed` in `results.json`. Result directories:

- HAR: `results/har_only_h{1,10,22}_2026-08-0{6,8}_*` (seeds 42/123/2026).
- FULL: `results/per_ticker_gate_h{1,10,22}_2026-08-08_*` (seeds 42/123/2026).

Per-seed values are in each directory's `results.json` (`test_metrics`). Aggregation and paired
t-statistics were computed directly from those files. Model checkpoints under `models/` are not
committed (repo convention); the small `results.json` artifacts are.

## 6. Definition of Done

- [x] Root-cause per-script check (P1.1/P1.2/P1.3 already fixed upstream; only `--seed`/provenance
      missing) — fixed by copying the proven 5-day pattern, no new approach invented.
- [x] Data pipeline confirmed current (33-ticker P1.2-fixed data, read-only shared dataset).
- [x] 18 real runs completed (3 horizons × 2 models × 3 seeds), 20 epochs each, metrics captured.
- [x] Report with 4-horizon table + honest hypothesis assessment (objective tone, no personal
      address, facts/numbers only).
- [ ] Diff-coverage gate (`diff-cover --fail-under=100`): Not run — tooling not installed in this
      environment (documented gap in CLAUDE.md). Change is training-script plumbing verified by
      real 20-epoch runs on real data (18/18 produced valid `results.json`), not unit-testable
      behavior; smoke was the earlier 2-epoch gate run + all 18 full runs.
