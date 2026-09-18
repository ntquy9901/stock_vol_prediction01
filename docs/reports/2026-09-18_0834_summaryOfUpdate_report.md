# Summary of update — Foundation-model feature baseline (Hướng B falsification)

Date: 2026-09-18. Baseline: `baselines/2026-09-18_foundation_feature/`.

## Objective
Cheaply falsify Hướng B (foundation-model prior + GBME): does a zero-shot time-series foundation-model forecast,
added as ONE causal ADDITIVE feature to the champion own-history gamma-GBM (GBME) — not the multiplicative anchor
— beat GBME out-of-sample on QLIKE? Prior belief = NO-GO (Brini 2026).

## Install outcome
Foundation model IS installable on this machine. `pip install chronos-forecasting` (Amazon Chronos-Bolt 2.3.2)
succeeded into `.venv_gpu_encode` (torch 2.6+cu124, CUDA available). `amazon/chronos-bolt-tiny` loads and runs
zero-shot; measured ~20k forecasts/sec on the RTX 4060, so the ~1.4M-forecast HOSE cache builds in ~1–2 min. No
fabrication was needed.

## What changed (files)
| file | purpose |
|------|---------|
| `baselines/2026-09-18_foundation_feature/code/foundation_config.py` | tunable constants (single source) |
| `.../code/foundation_forecaster.py` | causal zero-shot forecaster + per-(ticker,date) parquet cache |
| `.../code/run_foundation.py` | walk-forward Stage-0/Stage-1 driver, DM/verdict/spike, checkpoints |
| `.../test/test_foundation.py` | fake-forecaster driver tests + opt-in real-Chronos slice test |
| `.../requirements/requirements.md`, `.../design/design.md`, `.../code_review/code_review_2026-09-18.md` | SDD artifacts |
| `results/gamma_gbm/foundation_hose_h{1,5,10,22}.json` | per-horizon results (evidence-carrying) |
| `results/gamma_gbm/foundation_cache_hose.parquet` | cached zero-shot forecasts (71 MB, computed once) |

## Method (causal, cached, batched)
- For each stock/date t, Chronos-Bolt forecasts PRED_LEN=22 steps from the trailing own parkinson-variance context
  ≤ t (CTX_LEN=256). Step h predicts variance at t+h = the panel target `parkinson_variance.shift(-h)`. Frozen
  model, zero-shot, no leakage. Cached to parquet (built once, reused across horizons).
- Stage 0: the zero-shot forecast used directly vs HAR (date-clustered DM per horizon).
- Stage 1: GBME vs GBME+FND (forecast + quantile spread as extra causal GBM columns) vs GBME+PLAC (wrong-ticker
  placebo). Walk-forward over 8 folds (verified n_folds==8 all horizons), seed-ensembled champion HGBR gamma.

## Results — HOSE, all 4 horizons, 8 folds each (pooled QLIKE, floor 1e-8)

| h | GBME | GBME+FND | gain vs GBME | DM p | GBME+PLAC | HAR | Chronos zero-shot |
|---|------|----------|-------------:|-----:|-----------|-----|-------------------|
| 1 | 1.5791 | 1.5914 | −0.78% | 0.0004 (sig worse) | 1.8708 | 1.8169 | 121.54 |
| 5 | 1.6527 | 1.6556 | −0.18% | 0.2133 | 1.6627 | 1.8272 | 167.63 |
| 10| 1.6905 | 1.7007 | −0.60% | 0.0860 | 1.7542 | 1.8328 | 155.29 |
| 22| 1.7292 | 1.8168 | −5.07% | 0.0080 (sig worse) | 1.8558 | 1.8393 | 164.92 |

- **Stage 0 (zero-shot vs HAR): NO-GO, decisively.** Raw Chronos zero-shot QLIKE is 121–168 at every horizon —
  ~60–90× worse than HAR (~1.8). The frozen model does not even approach HAR; `zeroshot_beats_har=False`,
  p=0.0000 all h. (Chronos-Bolt is not scale-adapted to ~1e-4 variance magnitudes; the point of Stage 0 was to
  measure this honestly.)
- **Stage 1 (additive feature): NO-GO at all horizons.** GBME+FND never beats GBME (negative gain everywhere;
  significantly worse at h1 and h22). Spike-excluded (COVID/2022/Apr-2025) results also never beat
  (`beats_ex_spike=False` all h). Placebo also fails to beat. Fit diagnostics: all models "ok" (no over/underfit).
- **Pre-registered success = False.** (Needed ≥2 horizons DM-sig beat + placebo negative + spike-robust.)

**Verdict: NO-GO.** The zero-shot foundation-model forecast neither matches HAR alone nor adds value as a causal
feature to the champion GBME on HOSE. Consistent with the Brini 2026 prior. Hướng B is falsified for this market.

## Tests + coverage
`.venv_gpu_encode` pytest: **23 passed**. diff-cover on changed lines: **C0 = 100%, C1 = 100%** (0 partial
branches) across `foundation_config`, `foundation_forecaster`, `run_foundation`. A fast fake forecaster keeps the
driver tests off the slow model; one opt-in test runs the real Chronos model on a tiny slice; step-varying fake
tests lock the step→horizon→target alignment (added per code review).

## Code review + actions
Adversarial 3-lens review (`code_review/code_review_2026-09-18.md`): no Critical. One Major (test-adequacy: the
persistence fake gave identical per-step values, so nothing locked the causal step→horizon alignment) — FIXED with
a step-varying fake + two alignment tests. Minors: removed dead `KILL_HORIZONS`; renamed the misnamed
`zeroshot_matches_har`→`zeroshot_beats_har`. A memory OOM found during the first real run (h5) was fixed by
trimming the panel to model columns, float32 predictions, and gc per fold (numerically inert — h1 reproduced
identically before/after).

## Performance / batching
Zero-shot inference is batched (FORECAST_BATCH=512 contexts/call) on GPU, never batch=1; ~20k forecasts/s.
Forecasts cached to parquet and computed once; the walk-forward reuses the cache for all horizons. GBME reuses the
tested HGBR champion, seed-ensembled.

## Data-quality gate
N/A (no data change). This baseline reads existing `data/processed_enriched/hose` and the crawled HOSE earnings
parquet; it adds no raw/processed data. It writes only result JSONs + a forecast cache under `results/`.

## Risks / follow-ups
- Accepted low-risk follow-ups from review: `VALID_LEN`/embargo live in the driver (mirrors the sibling battery);
  `USE_SPREAD` flip against an existing spread-less cache would `KeyError` (single global per run); `_placebo`
  degenerates to self-forecast only for a 1-ticker universe (unreachable on HOSE/SP500).
- SP500 not run (HOSE was the mandated market; the driver supports `sp500`). Given the decisive HOSE NO-GO and the
  published prior, no SP500 run was warranted here.

## Git
No git commit/push performed (coordinator pushes). The result JSONs pass the pre-push overfit-evidence gate
(`check_overfit_evidence` → OK: carry train/val/test fit evidence, no over/underfit).
