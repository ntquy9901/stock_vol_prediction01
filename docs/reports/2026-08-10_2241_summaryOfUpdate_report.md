# Summary of update — GARCH family lifted to 33/33 tickers (LPB OHLCV recovered)

## What changed
LPB's raw OHLCV was recovered (SSI iBoard API) and committed, so the GARCH-family baselines
(GARCH(1,1) / GJR-GARCH / EGARCH) now cover all 33 of 33 tickers on the EXACT same 14,418 val /
14,464 test observations as the vol-only baselines and the ladder. The vol-only baselines
(Persistence / EWMA / HAR / HARQ / log-HAR) already covered the full set and are byte-identical to
the prior run (re-verified). The overall conclusion is unchanged: GARCH is still far worse (R² ≈ 0),
HAR/HARQ still tie the deep models.

## Data change
- `data/raw/prices/LPB_ohlcv.csv` (new, 1434 rows, 2020-11-09→2026-08-10, from SSI iBoard) +
  `data/raw/prices/LPB_SOURCE_NOTE.md` (provenance + adjustment-convention caveat).
- OHLC validity checked: high≥low≥0, dates strictly increasing, no duplicate dates.
- Parkinson variance recomputed from LPB High/Low reproduces `LPB_processed.csv` (corr with the
  variance formula = 1.0; median |diff| ≈ 4.8e-6), confirming the correct series.
- Caveat: SSI uses a different price-adjustment convention than the other tickers (levels differ by
  ~1.16x). Immaterial for return-GARCH because log-returns are scale-invariant except on a few
  ex-dividend days. LPB price LEVELS must not be treated as consistent with the other tickers' feeds.

## Calendar gap handling (honest)
18 LPB observations fall on holidays present in the processed calendar but absent from the SSI
trading calendar (Tết 2025-01-27..31, Reunification/Labour 2025-04-30..05-02, New Year 2026-01-01..02).
For these the GARCH forecast is carried forward from the last available trading origin (conditional
variance is persistent over a short gap). This is 18 / 14,464 = 0.12% of test observations; documented
in the canonical file (`GARCH_calendar_gap` note) and covered by a unit test.

## New 33-ticker GARCH numbers (h5) and what changed vs the 32-ticker version
TEST (val in the canonical file):
| baseline | mse | rmse | mae | r2 | qlike | dir_acc | n_test |
|---|---|---|---|---|---|---|---|
| GARCH(1,1) | 2.23971e-05 | 0.00473256 | 0.00115920 | 0.003355 | 1.75138 | 48.492 | 14464 |
| GJR-GARCH | 2.24496e-05 | 0.00473811 | 0.00116291 | 0.001016 | 1.81412 | 48.401 | 14464 |
| EGARCH | 2.25369e-05 | 0.00474731 | 0.00116734 | -0.002868 | 1.86327 | 48.641 | 14464 |

Change vs 32-ticker (test): QLIKE 1.76100→1.75138 / 1.82432→1.81412 / 1.87379→1.86327; RMSE
~0.004761→~0.004733; R² 0.003075→0.003355 (GARCH), −0.003130→−0.002868 (EGARCH); coverage
14,292→14,464 (+172 LPB obs). The shifts are small; GARCH remains decisively worse than HAR/G1
(RMSE ~2x, R² ≈ 0), and DirAcc stays ~48%.

## Files updated for consistency
- Canonical results (33-ticker): `docs/reports/classical_baselines_h5_2026-08-10_221043.{json,md}`
  (new versioned file; the old 32-ticker `..._2026-08-09_182129.md` carries a SUPERSEDED banner for
  provenance).
- Paper: `docs/paper/track_b_paper_draft.md` and `docs/paper/soict2026_trackb_v1.tex` — GARCH table
  rows, `\GARCHtestQ`/`\GARCHtestRsq` macros, QLIKE range 1.76–1.87→1.75–1.86, the "32/33, LPB
  excluded" caveat → "33/33 (LPB OHLCV recovered from SSI; adjustment-convention note)", Table 1
  caption, MCS parenthetical, limitations, and the classical source-file pointer.
- Bundle: `submission_track_b/results/classical_baselines_h5.json` (= new canonical), `PAPER_MAP.md`,
  `submission_track_b/reproduce.py` (comment + `view` header). `reproduce.py view` re-run confirms the
  new numbers.

## Cross-artifact consistency check
- GARCH QLIKE/RMSE/R² match across canonical JSON ↔ paper .md ↔ .tex (macros + table) ↔ bundle JSON
  ↔ `reproduce.py view`. Scan for stale strings (1.7610, 0.004761, 2.26642e-05, 32/33, 32-ticker,
  14,292, 14,247, old canonical filename) returns none in the paper/bundle/PAPER_MAP/reproduce.
  Bundle `rung_metrics.test` == canonical `rung_metrics.test` (exact).

## Tests / gates
- `pytest baselines/classical_baselines/test` — 19 pass (18 unit + 1 smoke). New tests:
  GARCH carry-forward over a calendar gap; GARCH raises when no requested origin is a trading day.
- ruff: clean. diff-cover (origin/master…HEAD): C0 = 100% on the 12 changed lines.
- Data-quality gate: Pandera `check_schema()` = PASS (34/34 processed artifacts valid). The new
  LPB_ohlcv.csv is a raw-prices file (outside the processed-schema scope) and was validated manually
  (OHLC validity + Parkinson reproduction above). Evidently drift: N/A — this change adds a raw price
  file and re-scores GARCH; it does not alter the processed train/test feature distributions the ladder
  uses. The pre-push hook re-runs the data-quality gate.
- Code review: the LPB provenance/caveat and carry-forward are documented; carry-forward is a genuine
  GARCH forecast (not a fabricated fallback) and is unit-tested. No fabricated numbers.

## DoD checklist
- [x] LPB OHLC validity checked; Parkinson reproduction confirmed.
- [x] GARCH re-fit with LPB → 33/33 tickers, exact 14,418/14,464 alignment.
- [x] Canonical results updated (new versioned file; old file superseded-banner).
- [x] Paper .md + .tex + PAPER_MAP + bundle + reproduce.py updated and consistent.
- [x] Cross-artifact consistency verified.
- [x] TDD (RED→GREEN), pytest green, ruff clean, diff-cover C0=100%.
- [x] Pandera check_schema PASS; LPB_ohlcv.csv + LPB_SOURCE_NOTE.md committed.
- [x] Ledger entry + dashboard regenerated; pull --rebase before push.
