# Clean vnstock VN100 dataset + re-run VN100 ablation — R2 anomaly investigation

Date: 2026-08-16

## Objective

Produce a clean vnstock (VCI) VN100 processed dataset and re-run the VN100 leave-one-out
ablation on it, to test whether the low VN100 R2 in the prior ablation was an artifact of a
dirty Yahoo source (phantom-holiday flat rows that inject fake zero-Parkinson targets) rather
than a genuine property of mid-cap volatility. Report clean-vs-dirty-vs-VN30 R2.

All commands run from repo root with `.venv_gpu_encode` python and `PYTHONIOENCODING=utf-8`.
No commit/push (coordinator review). VN30 data, `data/raw/prices/vn100/` (Yahoo), and the
top-level 33 were not modified.

---

## STEP 1 — Clean the vnstock raw

Input: `data/raw/prices/vn100_vnstock/` (104 tickers, source VCI via vnstock).

Command:
```
.venv_gpu_encode/Scripts/python.exe -m src.data.clean_ohlc data/raw/prices/vn100_vnstock
```
Result: `cleaned 104 files; 53 changed; 239 cells total` (positive-aware max/min OHLC repair).
Re-running the same command reports `0 changed; 0 cells` — the pass is idempotent as documented.

Verify:
```
.venv_gpu_encode/Scripts/python.exe -m src.data.verify_raw_prices data/raw/prices/vn100_vnstock
```
Output after cleaning:
- HARD defects: **1 ticker -> NT2**
- zero-volume WITH price move (SUSPICIOUS): 6 tickers -> NT2=48, HHV=40, BCM=5, EVF=3, NLG=2, VHC=1
- leading zero-volume backfill run >20: 0 tickers
- highest zero-volume fraction: HHV=41.6%, VHM=37.7%, NT2=7.8%, VIX=5.9%, TLG=4.1%

### Residual HARD defect (reported, not fabricated)

The one remaining HARD defect is NT2, rows 0–1:

| date | open | high | low | close | volume |
|------|------|------|-----|-------|--------|
| 2010-01-26 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 2010-01-27 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |

These are pre-listing all-zero backfill rows. `clean_ohlc` intentionally leaves rows whose OHLC
are ALL nonpositive untouched (it cannot reconstruct a row from itself; documented in
`src/data/clean_ohlc.py`). This is the exact "residual nonpositive row that clean_ohlc can't fix
because all-OHLC-nonpositive" case anticipated in the task — reported, not silently zeroed.

Impact on the processed target: none. In `process_single_stock`, Parkinson for these rows is
`log(0/0)**2 = NaN`, which is dropped by `dropna()`. The processed NT2 file contains no NaN and
no fake zeros from these rows (confirmed in Step 2: 0 NaN files, monotonic).

### zero_vol_price_moved note

The 6 `zero_vol_price_moved` rows (volume==0 but high!=low) are a feed diagnostic, not a
target-contamination issue: because high!=low they produce a **nonzero** Parkinson value, so they
do NOT inject fake zeros and do NOT depress R2 through the fake-zero mechanism. They were left as-is
(clean_ohlc only repairs OHLC internal consistency, never volume). This is not the phantom-flat-row
problem, which is `zero_vol_flat` (high==low -> Parkinson exactly 0).

---

## STEP 2 — Process to Parkinson

Command:
```
.venv_gpu_encode/Scripts/python.exe -m src.common.process_parkinson_pipeline \
    --raw data/raw/prices/vn100_vnstock --out data/processed/vn100_vnstock
```
Result: 104 `*_processed.csv`, 357,850 total records.

Verify (all three datasets audited with the same script):

| dataset | files | rows | last date | NaN files | non-monotonic | exact-zero target frac |
|---------|-------|------|-----------|-----------|---------------|------------------------|
| CLEAN vnstock VN100 (`data/processed/vn100_vnstock`) | 104 | 357,850 | 2026-08-14 | 0 | 0 | **3.34%** (11,963) |
| DIRTY Yahoo VN100 (`data/processed/vn100`) | 104 | 318,555 | 2026-08-14 | 0 | 0 | **7.82%** (24,914) |
| VN30 (`data/processed`) | 33 | 106,648 | 2026-08-14 | 0 | 0 | **1.74%** (1,858) |

The dirty Yahoo VN100 target has **2.3x** the exact-zero fraction of the clean vnstock VN100
(7.82% vs 3.34%) — i.e. ~4.5 percentage points of the Yahoo target are fake zeros (holiday flat
rows, H==L -> Parkinson=0) that the vnstock source does not carry. VN30 is the cleanest at 1.74%.
An exact-zero target is unpredictable noise for a volatility model, so a higher fake-zero fraction
mechanically caps achievable R2.

Note the datasets differ in vendor (vnstock/VCI vs Yahoo) and history length; the zero-fraction
gap reflects both the cleaner vnstock source and the `clean_ohlc` pass.

---

## STEP 3 — Re-run VN100 ablation on CLEAN data

Runner: `baselines/2026-08-15_volatility/code/run_ablation.py` via a thin wrapper
`tmp/run_vn100_vnstock_ablation.py` that only patches module globals (no baseline code edited):
- `combo_ladder._PROCESSED = data/processed/vn100_vnstock`
- `combo_ladder._PRICE_DIR = data/raw/prices/vn100_vnstock`  **(coordinator fix, applied)**

Config identical to the prior dirty run for comparability: seed=42, epochs=8, horizons (1,5,10),
SEQ=22, default 70/15/15, device cuda (RTX 4060).

**Coordinator fix applied and verified.** Setting only `_PROCESSED` is insufficient — with the
default `_PRICE_DIR` (VN30 raw), `features.volume_zscore_series` returns all-zeros for non-VN30
tickers, neutralizing the `volume_zscore_20` node feature. After also overriding `_PRICE_DIR`, the
wrapper verified a non-VN30 ticker has a live feature: `volume_zscore check HHV: col=volume_zscore_20
nonzero_rows=1354/1857` (not all-zero). The earlier run started before this fix was discarded and
re-run from scratch.

Basis (h1, clean): snapshots=6256, train=245,967, val=49,153, test=49,314, vol2pk_directed_edges=519,
price_dim=5, news_dim=146.

News caveat: news panel is VN30-only, so the ~71/104 non-VN30 tickers get zero-filled news in the
FULL / minus_gate / minus_news rungs (same caveat as the dirty run). The **HAR rung has no news**,
so the HAR R2 comparison below is free of this confound and is the most robust line.

Results dirs: `results/volatility_ablation_h{1,5,10}_seed42_2026-08-16_vn100vnstock_clean/`.

For an apples-to-apples VN30 reference (no comparable VN30 run existed in this ladder format), the
same runner + config was run on VN30 (default `_PROCESSED`/`_PRICE_DIR`):
`results/volatility_ablation_h{1,5,10}_seed42_2026-08-16_vn30_cmp/`.

---

## STEP 4 — Comparison + verdict

### Held-out TEST R2 per rung, per horizon

**h=1**
| rung | CLEAN vnstock VN100 | DIRTY Yahoo VN100 | VN30 |
|------|--------------------:|------------------:|-----:|
| HAR | 0.2156 | 0.0895 | 0.2870 |
| FULL | 0.2157 | 0.0934 | 0.2906 |
| minus_graph | 0.2197 | 0.0935 | 0.2963 |
| minus_gate | 0.2143 | 0.0935 | 0.2900 |
| minus_news | 0.2140 | 0.0931 | 0.2899 |
| lstm_only | 0.2210 | 0.0933 | 0.2960 |

**h=5**
| rung | CLEAN vnstock VN100 | DIRTY Yahoo VN100 | VN30 |
|------|--------------------:|------------------:|-----:|
| HAR | 0.1355 | 0.0451 | 0.1868 |
| FULL | 0.1416 | 0.0492 | 0.1792 |
| minus_graph | 0.1406 | 0.0482 | 0.1835 |
| minus_gate | 0.1407 | 0.0492 | 0.1783 |
| minus_news | 0.1425 | 0.0521 | 0.1823 |
| lstm_only | 0.1403 | 0.0513 | 0.1820 |

**h=10**
| rung | CLEAN vnstock VN100 | DIRTY Yahoo VN100 | VN30 |
|------|--------------------:|------------------:|-----:|
| HAR | 0.0952 | 0.0338 | 0.1382 |
| FULL | 0.0883 | 0.0284 | 0.1285 |
| minus_graph | 0.0892 | 0.0293 | 0.1282 |
| minus_gate | 0.0892 | 0.0270 | 0.1247 |
| minus_news | 0.0891 | 0.0275 | 0.1285 |
| lstm_only | 0.0890 | 0.0294 | 0.1286 |

### QLIKE per rung, per horizon (lower is better)

**h=1** — CLEAN / DIRTY / VN30: HAR 0.4785 / 0.8945 / 0.4633; FULL 0.4714 / 0.8818 / 0.4529.
**h=5** — HAR 0.5438 / 0.9727 / 0.5503; FULL 0.5376 / 0.9609 / 0.5556.
**h=10** — HAR 0.5775 / 0.9948 / 0.5933; FULL 0.5820 / 1.0007 / 0.5980.

The dirty-VN100 QLIKE is ~0.88–1.00 across horizons; cleaning drops it to ~0.47–0.58, in the same
band as VN30 — a large, consistent improvement that mirrors the R2 result.

### HAR R2 lift and gap-closure to VN30 (HAR rung = the baseline, no news confound)

| horizon | dirty | clean | lift | multiple | clean % of VN30 | gap-to-VN30 closed |
|---------|------:|------:|-----:|---------:|----------------:|-------------------:|
| h1 | 0.0895 | 0.2156 | +0.1261 | 2.41x | 75% | 64% |
| h5 | 0.0451 | 0.1355 | +0.0904 | 3.01x | 73% | 64% |
| h10 | 0.0338 | 0.0952 | +0.0614 | 2.82x | 69% | 59% |

### Verdict

1. **Did cleaning raise R2? Yes, substantially.** On the identical VN100 universe and identical
   ablation config, moving from the dirty Yahoo source to the clean vnstock source raised HAR test
   R2 by roughly 2.4x–3.0x (h1 +0.126, h5 +0.090, h10 +0.061). Every rung shows the same jump, and
   QLIKE improves in lockstep (from ~0.88–1.00 down to ~0.47–0.58). The prior "VN100 R2 is
   anomalously low" observation is **largely a data-dirtiness artifact**: the dirty target carried
   2.3x more fake-zero (phantom holiday flat) rows, and those unpredictable zeros were depressing R2.

2. **Does clean VN100 approach VN30? Most of the way, but not all.** Clean VN100 HAR R2 reaches
   69–75% of VN30 and closes 59–64% of the dirty->VN30 gap, but stays ~0.04–0.07 R2 below VN30 at
   every horizon (e.g. h1: 0.2156 vs 0.2870). So a genuine residual remains: mid-caps are somewhat
   harder to forecast than large-caps even after cleaning.

3. **Overall answer to "is the R2 anomaly a data-dirtiness artifact":** predominantly yes. Data
   dirtiness was the dominant driver of the low VN100 R2 (it accounts for ~60% of the gap to VN30
   and a 2.4–3.0x R2 improvement); a smaller, real mid-cap difficulty accounts for the rest. The
   headline low-R2 anomaly is not a genuine property of the market — it was mostly bad data.

4. **Ablation components remain a parsimony null on clean data.** Leave-one-out effects are tiny
   (h1 clean: graph −0.0003, gate −0.0012, news −0.0013 QLIKE; FULL R2 0.2157 barely differs from
   HAR 0.2156), consistent with the established finding that no learned component meaningfully beats
   HAR. Cleaning changed the R2 level, not the ablation conclusion. (Component rungs carry the
   VN30-only news zero-fill caveat; the HAR-vs-FULL near-tie is robust to it.)

---

## Artifacts

- Clean processed data: `data/processed/vn100_vnstock/` (104 files)
- Clean VN100 ablation: `results/volatility_ablation_h{1,5,10}_seed42_2026-08-16_vn100vnstock_clean/`
- VN30 reference ablation: `results/volatility_ablation_h{1,5,10}_seed42_2026-08-16_vn30_cmp/`
- Dirty Yahoo VN100 (prior): `results/trackA_ablation_h{1,5,10}_seed42_2026-08-16_vn100quick/`
- Wrappers: `tmp/run_vn100_vnstock_ablation.py`, `tmp/run_vn30_ablation.py`
