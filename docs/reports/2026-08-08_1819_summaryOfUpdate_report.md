# T1.1 — A1 ablation: pooled vs common-date training-sample set

Branch: `feature/pooled-news-gnn-pilot`. Baseline:
`baselines/2026-08-08_pooled_news_gnn_ablation_baseline`. Horizon 5, 5 epochs,
seeds 42/123/2026, GPU (RTX 4060, torch 2.6.0+cu124).

## Scope

Track B headline experiment A1: train the SAME P0–P3 models on the pooled per-sample
manifest vs a common-date-restricted training set, changing ONLY the training-sample set.
Tests whether pooling all per-ticker samples over full history gives a data advantage that
lets the deep models reach/beat classical HAR, whereas common-date-only training starved them.

## What changed (code)

| File | Purpose |
|------|---------|
| `code/data.py` | `_transform_full_frames` (shared common-date axis, extracted from `build_graph_manifest` with no behavior change); `common_trading_dates` (public axis = intersection of per-ticker post-HAR trading dates); `restrict_train_to_common_dates` (keeps only TRAIN windows whose input dates and target date all lie on the common axis; val/test pass through as the full pooled held-out sets). |
| `code/run_pilot.py` | `--regime {pooled,common-date}` (default `pooled`, unchanged behavior). The common-date branch runs AFTER scaler fitting and news attachment, so per-ticker scalers/winsor bounds and the news panel are byte-identical across regimes; only the training-sample SET differs. Regime recorded in `screening_metadata.json`. Guard: `--phase graph --regime common-date` is rejected (graph is common-date by construction). |
| `test/test_regime.py` (new) | Axis intersection, train-only restriction, eval-split preservation, cross-regime val/test content-hash identity, scaler-identity leakage guard, P2 attach-then-restrict news byte-identity, arg parsing/guards, metadata recording. |
| `test/test_train_smoke.py` | Two runner stubs updated for the new `regime` kwarg. |

## Leakage controls (verified)

- Split-before-HAR/scaler/window is inherited unchanged from `build_pooled_manifest`
  (per-ticker chronological; scaler/winsor fit on train rows only).
- The common-date regime REUSES the pooled per-ticker train-fitted scalers/winsor bounds —
  no refit on the smaller subset. `common_trading_dates` only transforms with the already-fit
  store; `restrict_train_to_common_dates` only subsets samples and passes
  `preprocessing_hash`/`ticker_to_id`/`exclusions` through. Test
  `test_common_date_regime_reuses_scalers_and_reduces_samples` asserts `store.to_dict()`
  identical across regimes on the real data path.
- Identical evaluation set: val/test are the full pooled held-out sets in BOTH regimes.
  Confirmed on the 33-ticker run — val/test `manifest_hashes` are byte-identical across
  regimes, train differs:
  `pooled val = common val = dce321ad08647f18`; `pooled train = f5d885e3…`,
  `common train = 0f36828d…`.
- Same seq_length (22), horizon (5), ticker-ID inverse transform, raw eval target untouched,
  `shuffle=False`. Seeds recorded in each `results.json`.

## Data reduction (33 tickers, horizon 5)

| Regime | Train samples | Val samples |
|--------|--------------:|------------:|
| pooled | 73,026 | 14,418 |
| common-date | 9,606 | 14,418 |

Pooling supplies ~7.6× more training samples (common-date = 13.2% of pooled train;
63,420 fewer train samples). Val/test are identical.

## A1 comparison (validation, denormalized; 3-seed mean ± std; paired-t = pooled − common-date, df = 2)

P0 (HAR) is a deterministic linear fit — identical across seeds (std = 0), so its paired-t is
undefined (`nan`); its deltas are exact constants. P0 is trained on each regime's train set and
scored on the shared val set.

### P0 — HAR reference
| metric | pooled | common-date | Δ(pool−comm) |
|--------|--------:|------------:|-------------:|
| rmse | 0.001485 | 0.001491 | −0.000006 |
| mae | 0.000480 | 0.000485 | −0.000005 |
| r² | 0.735146 | 0.732910 | +0.002237 |
| qlike | 0.516707 | 0.514721 | +0.001986 |
| dir_acc (%) | 48.5402 | 48.4783 | +0.0619 |

### P1 — price-only LSTM
| metric | pooled | common-date | Δ(pool−comm) | paired-t |
|--------|--------:|------------:|-------------:|---------:|
| rmse | 0.001502 ± 0.000004 | 0.001493 ± 0.000004 | +0.000009 | +4.48 |
| mae | 0.000490 ± 0.000002 | 0.000477 ± 0.000002 | +0.000014 | +15.36 |
| r² | 0.728734 ± 0.001483 | 0.731996 ± 0.001411 | −0.003262 | −4.47 |
| qlike | 0.511839 ± 0.001102 | 0.512773 ± 0.002356 | −0.000934 | −0.82 |
| dir_acc (%) | 48.5877 ± 0.0422 | 48.7911 ± 0.0218 | −0.2034 | −5.50 |

### P2 — price + news LSTM
| metric | pooled | common-date | Δ(pool−comm) | paired-t |
|--------|--------:|------------:|-------------:|---------:|
| rmse | 0.001487 ± 0.000003 | 0.001503 ± 0.000001 | −0.000017 | −6.39 |
| mae | 0.000480 ± 0.000003 | 0.000476 ± 0.000001 | +0.000004 | +1.87 |
| r² | 0.734357 ± 0.001214 | 0.728340 ± 0.000412 | +0.006017 | +6.41 |
| qlike | 0.508392 ± 0.000242 | 0.517774 ± 0.005525 | −0.009382 | −2.96 |
| dir_acc (%) | 48.4697 ± 0.1349 | 48.3632 ± 0.1567 | +0.1066 | +0.64 |

### P3 — price + news + gate (bonus; not required for A1)
| metric | pooled | common-date | Δ(pool−comm) | paired-t |
|--------|--------:|------------:|-------------:|---------:|
| rmse | 0.001489 ± 0.000004 | 0.001507 ± 0.000008 | −0.000018 | −3.58 |
| r² | 0.733641 ± 0.001349 | 0.727033 ± 0.002820 | +0.006608 | +3.57 |
| qlike | 0.508564 ± 0.000217 | 0.516278 ± 0.003487 | −0.007714 | −3.67 |
| dir_acc (%) | 48.5802 ± 0.0681 | 48.6198 ± 0.0792 | −0.0396 | −0.47 |

Sign convention: Δ > 0 for rmse/mae/qlike means pooled has the higher (worse) value;
Δ > 0 for r²/dir_acc means pooled is higher (better).

## Verdict

Does pooling improve QLIKE/RMSE/R² over common-date? Mixed and marginal.
- P1 (price-only): common-date is marginally better on rmse (Δ +0.000009, ~0.6% relative,
  t = +4.48), r² (Δ −0.0033, t = −4.47) and dir_acc; qlike difference is not significant
  (t = −0.82). Pooling does not help P1 and slightly hurts it.
- P2 (price+news): pooled is marginally better on rmse (t = −6.39), r² (t = +6.41), and qlike
  (Δ −0.0094, t = −2.96); dir_acc unchanged (t = +0.64). Pooling helps P2 slightly.
- All effects are sub-1% relative in rmse and ≤0.007 in r².

Does pooling close/beat the HAR gap that common-date could not? No. On the identical val set,
HAR (P0) is on par with or slightly ahead of the deep models on rmse/r² under BOTH regimes
(pooled: P0 r² 0.7351 ≥ P1 0.7287, P2 0.7344; common-date: P0 r² 0.7329 ≥ P1 0.7320,
P2 0.7283). The deep models edge HAR only on qlike, and only marginally. Pooling does not
produce a regime where the deep model decisively beats HAR while common-date does not.

Conclusion: the Track B premise — that a pooled data advantage lets the deep model reach/beat
HAR whereas common-date starved it — is NOT supported by this 5-epoch, 3-seed, horizon-5
screening. A ~7.6× increase in training samples moves validation metrics only marginally and
with mixed sign, and does not change the HAR-vs-deep ordering. This is a null result for the
pooling-data-advantage hypothesis at this screening budget.

## Caveats / follow-ups

- Screening budget: 5 epochs, n = 3 seeds. Paired-t has df = 2 (low power); |t| values here
  are screening signals, not confirmatory. n ≥ 5 seeds and a longer-epoch run are required
  before any final paper claim.
- The near-invariance of validation metrics to a 7.6× training-data change indicates the
  models are not training-data-starved at this horizon/epoch budget — itself evidence against
  the "pooling unlocks capacity" framing.
- Only horizon 5 was run; other horizons (1/10/22) are untested here.

## Checks run (real output)

- `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/ -q`: 125 passed.
- `ruff check code/ test/`: All checks passed.
- Adversarial code review (subagent): no CRITICAL/HIGH; 3 LOW resolved (graph+regime guard,
  P2 news-path test, docstring). See
  `code_review/code_review_2026-08-08_a1_regime.md`.
- RED→GREEN: `test_regime.py` first failed on missing `common_trading_dates` import (RED),
  passed after implementation (GREEN).
- 6 GPU cells: `results/a1_{pooled,commondate}_seed{42,123,2026}/h5/`.
- Diff-coverage gate (`diff-cover --fail-under=100`): Not run — tooling not installed in this
  repo (documented gap in CLAUDE.md); changed lines are covered by `test_regime.py`.

## DoD

- [x] Code satisfies request (regime flag; train-only common-date restriction; leakage-safe).
- [x] Tests written test-first (RED) then GREEN; 125 passed.
- [x] Lint clean (ruff).
- [x] Code review run + findings resolved.
- [x] Results captured with provenance (seeds, manifest hashes in `screening_metadata.json`).
- [ ] Push: intentionally NOT pushed — parent agent verifies and pushes.
