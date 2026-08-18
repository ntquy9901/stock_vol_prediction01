# R² anomaly investigation: VN30 (~0.77) vs VN100 (~0.09) in the volatility ablation

Date: 2026-08-16
Scope: read-only diagnosis. No code/data modified. Numeric checks run with
`.venv_gpu_encode/Scripts/python.exe` (PYTHONIOENCODING=utf-8) over existing result dumps and the
processed data. `archive/` and `backlog/` excluded.

## Verdict

The low VN100 R² is **NOT a code bug and NOT a scale bug**. It is the combination of a **data
artifact** (phantom-holiday zero-inflation in the Yahoo VN100 source) and a **pooled-global-R²
methodology sensitivity**, on a **genuinely harder/shorter universe**. The R² formula is standard,
correct, and applied identically to both universes. Scale/units are handled correctly (Parkinson is
scale-invariant; no double-scaling).

Three independent facts rule out a bug:

1. **HAR (a plain pooled OLS, no neural network, no per-ticker neural scaler) also collapses to
   R²≈0.09 on VN100** while giving R²≈0.77 on VN30 through the *same* code. A GNN/scaler/denorm bug
   cannot explain a linear-regression baseline behaving the same way.
2. **Parkinson recomputed from the raw OHLCV matches the processed target to 6 decimals** (VN100
   ACB: recomputed `[0.000593, 0.000364, 0.000093]` == processed `[0.000593, 0.000364, 0.000093]`).
   No double-scaling. Median close is ~14-16 for both universes, so no unit mix.
3. **The 0.77 vs 0.09 gap reproduces with a trivial trailing-mean predictor** pushed through the
   same pooled formula (VN30 pooled-global R²=0.206 vs VN100=-0.003), i.e. the gap is a property of
   the data and the aggregation, not of any trained model.

## How R² is computed (file:line)

- `src/common/evaluation.py:134-136`
  ```python
  ss_res = np.sum((y_true - y_pred) ** 2)
  ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
  r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
  ```
  R² is **pooled across ALL ticker-days at once** on **raw (denormalized) Parkinson values**, with
  the **SST baseline = the single GLOBAL mean** of all pooled targets (not a per-ticker mean).
- Call path: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/train.py:45`
  (`evaluate_records` → `evaluate_predictions`). Predictions are inverse-transformed per ticker at
  `train.py:39` (`store.inverse_targets`) before metrics, so R² is on the raw scale.
- The volatility ablation reaches this via
  `baselines/2026-08-15_volatility/code/run_volatility.py:171` (`_evaluate_rung` → `evaluate_records`)
  and HAR via `eda_ladder.run_e0` (same `evaluate_predictions`).

This pooled-global-mean choice is the key methodology lever (see below). It is a defensible choice
but it is **sensitive to between-ticker dispersion and heteroskedasticity**, which differ sharply
between the two universes.

## The key number: zero-fraction (data artifact)

Parkinson variance is exactly 0 on any bar where high == low (a phantom/flat holiday row). The
Yahoo VN100 raw is heavily contaminated; the VN30 raw and the clean vnstock VN100 are not.

Phantom rows in RAW OHLCV (`high == low` AND `volume == 0`):

| Source | files | rows | phantom rows | % |
|---|---|---|---|---|
| VN30 raw (`data/raw/prices/`) | 33 | 106,648 | 523 | **0.49%** |
| VN100 Yahoo (`data/raw/prices/vn100/`) — **used by the ablation** | 104 | 318,555 | 19,942 | **6.26%** |
| VN100 vnstock (`data/raw/prices/vn100_vnstock/`) — clean alternative | 104 | 357,852 | 4,199 | **1.17%** |

Exact-zero fraction of the processed target `parkinson_volatility`:

| Processed dir | exact-zero fraction (all rows) | exact-zero fraction (TEST window, last 15%) |
|---|---|---|
| VN30 (`data/processed/`) | **1.74%** | **0.39%** |
| VN100 (`data/processed/vn100/`) | **7.82%** | **4.68%** |

The VN100 test window carries **~12× more exact-zero targets than VN30** (4.68% vs 0.39%). Those
zeros are pure unpredictable noise: the model predicts a small positive value, the target is 0, and
every such point adds to SSE while contributing little explainable structure, depressing R². The
VN100 ablation ran on the Yahoo source (confirmed: processed VN100 ACB == Yahoo raw ACB Parkinson,
and the vnstock source starts 2008 with different values).

## Per-ticker vs pooled R² on VN100 (methodology sensitivity)

From the VN100 HAR (P0) test dump `results/trackA_ablation_h1_seed42_2026-08-16_vn100quick/P0/predictions_test.json`:

| horizon | pooled-global R² (reported) | within-ticker-mean-SST R² | per-ticker-then-averaged R² | between-ticker var share |
|---|---|---|---|---|
| h1 | 0.0895 | 0.0589 | 0.127 (median 0.133) | 0.033 |
| h5 | 0.0451 | 0.0131 | 0.001 (median 0.013) | 0.032 |
| h10 | 0.0338 | 0.0012 | -0.037 (median -0.005) | 0.033 |

Interpretation: on VN100 the between-ticker variance is only ~3% of pooled variance, so the
pooled-global R² gets almost no "free" lift from per-ticker level differences. Per-ticker R² (~0.13
at h1) is actually a bit *higher* than pooled here, i.e. VN100's low R² is not an artifact of
pooling *inflating* it — it is genuinely low across every basis.

The contrast with VN30 is where pooling matters. Test-window structure differs:

| TEST window (last 15%, per ticker) | VN30 | VN100 |
|---|---|---|
| date range | 2023-08 → 2026-08 | 2024-02 → 2026-08 |
| target mean / std | 3.57e-4 / 5.18e-4 | 4.00e-4 / 1.06e-3 |
| exact-zero fraction | **0.39%** | **4.68%** |
| between-ticker variance share | **11.1%** | **2.3%** |
| lag-1 autocorr (median) | 0.317 | 0.341 |

VN30's recent test window has **~5× more between-ticker dispersion** (11.1% vs 2.3%) — large-caps
diverged in volatility level, and the per-ticker denormalization reproduces those level differences
"for free", which the pooled-global-mean SST rewards heavily. Pooled R² is additionally dominated by
the largest-variance ticker-days (squared errors); VN30's recent window contains larger, persistent
volatility that HAR tracks, VN100's is flatter and zero-inflated.

Same-code, same-formula proof with a trivial trailing-22-day-mean predictor:

| predictor = trailing 22d mean, TEST window | VN30 | VN100 |
|---|---|---|
| POOLED-global R² | **0.206** | **-0.003** |
| WITHIN-ticker R² | 0.107 | -0.026 |

A predictor with no trained parameters, run through the identical pooled formula, already reproduces
the qualitative VN30 ≫ VN100 gap and the pooled ≫ within-ticker inflation. The remaining distance
from 0.206 to the reported 0.77 (VN30) is simply that the real HAR is a fitted 3-feature OLS
(daily+weekly+monthly), which predicts far better than a single trailing mean on VN30's persistent
large-cap volatility — and still cannot on VN100.

## Whole-series stats are nearly identical (why this is period-specific, not universe-wide)

Over each ticker's full history the two universes look almost the same (between-ticker share ~2.5%,
autocorr ~0.35, similar mean/std). The R² divergence is therefore **specific to the recent test
window**: VN30's 2023-2026 slice happens to have large, persistent, cross-sectionally dispersed
volatility (pooled-R²-friendly); VN100's 2024-2026 slice is flatter, homogeneous across mid-caps,
and zero-inflated (pooled-R²-hostile). This is a genuine data property, not a defect.

## Scale handling (the user's specific concern) — clean

- Parkinson `(ln(H/L))²/(4 ln2)` is scale-invariant; verified it matches the processed target
  bit-for-bit from raw. No accidental double-scaling.
- Price magnitudes are consistent (median close ~14-16 across VN30 / VN100 Yahoo / VN100 vnstock),
  so no thousands-VND vs VND unit mix.
- Per-ticker StandardScaler denorm (`run_volatility.py:53-58`, `_StoreShim.inverse_targets`) is only
  used by the neural rungs; HAR bypasses it entirely and is equally low, so the scaler is not the
  cause. `evaluate_records` even guards against a broken scaler (`train.py:40-42` rejects >1%
  nonpositive predictions) and did not trip.

## Secondary finding (a real wrapper bug, but NOT the R² cause)

`scripts/run_vn100_ablation.py:28` overrides `combo_ladder._PROCESSED` to the VN100 dir but does
**not** override `combo_ladder._PRICE_DIR` (`combo_ladder.py:53`, still `data/raw/prices/` = VN30).
`augment_split_frames` therefore looks up each ticker's OHLCV under the VN30 raw dir; for the ~71 of
104 VN100 tickers absent from VN30, `volume_zscore_series` (`features.py:70-72`) finds no file and
returns all-zeros. So the `volume_zscore_20` node feature is silently neutralized for most VN100
tickers in this quick ablation. This is a genuine plumbing defect worth fixing before any headline
VN100 numbers, but it affects only 1 of 5 node features (used by the neural/GNN rungs); HAR ignores
it and is equally low, so it does not drive the R² anomaly.

## What would raise the VN100 R²

1. **Use the clean vnstock source** (`data/raw/prices/vn100_vnstock/`, 1.17% phantom vs 6.26%) or
   drop phantom holiday rows (`high==low & volume==0`) before computing Parkinson. This removes most
   of the zero-inflation noise. It will *raise* R² but not to VN30 levels, because the pooling and
   period effects remain.
2. **Report per-ticker R²** (or a within-ticker SST baseline) for cross-universe comparison — it is
   not inflated by between-ticker dispersion. For VN100 h1 that is ~0.13; still modest, confirming
   the universe is genuinely harder.
3. Accept that pooled-global R² is not comparable across universes with different heteroskedasticity
   and between-ticker dispersion; QLIKE/RMSE per ticker are more robust headline metrics here.

## Evidence index

- R² formula: `src/common/evaluation.py:134-136`; call path `.../2026-08-08_pooled_news_gnn_ablation_baseline/code/train.py:39,45`.
- VN100 ablation wrapper: `scripts/run_vn100_ablation.py:28` (+ un-overridden `combo_ladder.py:53`).
- VN100 result dumps: `results/trackA_ablation_h{1,5,10}_seed42_2026-08-16_vn100quick/`.
- VN30 pooled R² reference (~0.77): `docs/reports/2026-08-09_1927_summaryOfUpdate_report.md`
  ("P0 ... r2 0.7668").
- Zero-fraction / phantom / decomposition numbers: recomputed live from `data/processed/`,
  `data/processed/vn100/`, `data/raw/prices/{,vn100/,vn100_vnstock/}` (see tables above).
