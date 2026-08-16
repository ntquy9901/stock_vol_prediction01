# Summary of update — data-quality fixes (OHLC + VHM), quality tests, and enforced ingestion harness

## What changed

1. **OHLC data fix (16 tickers, 46 rows):** rows where open/close fell outside [low,high] (uneven
   dividend/split back-adjustment) or a nonpositive value existed (SSI 2006-12-15 low=0) were made
   internally consistent: `high = max(positive OHLC)`, `low = min(positive OHLC)`; nonpositive
   open/close clamped into the valid range. Parkinson uses only H/L, so the effect on the target is
   confined to these few old rows.
2. **VHM trim:** removed the pre-listing synthetic backfill. VHM now starts **2018-05-23** (real HOSE
   liquid listing) instead of 2011-11-10; 3474 → 2058 rows. Zero-variance fraction dropped 40.5% →
   0.2% (remaining zeros are legitimate limit-up/down days).
3. **Reprocessed** all 33 `data/processed/*_processed.csv` and synced to the worktree copy the
   training reads (`.worktrees/volatility-gat/data/processed`).
4. **Two data-quality test scripts** (written by audit agents) finalized and now GREEN:
   `tests/test_raw_prices_quality.py` (167 passed) and `tests/test_processed_data_quality.py`
   (134 passed).
5. **Enforced ingestion harness (per user request 2026-08-16):** the pre-push hook now runs the two
   data-quality tests and blocks the push when data is touched; a CLAUDE.md rule documents the
   mandatory raw-ingestion workflow.

## Files

| Path | Purpose |
|---|---|
| `data/raw/prices/*.csv` | 16 tickers OHLC-corrected; VHM trimmed to 2018-05-23 |
| `data/processed/*_processed.csv` | regenerated from fixed raw |
| `tests/test_raw_prices_quality.py` | raw OHLCV quality test (schema/dates/OHLC/backfill), 167 passed |
| `tests/test_processed_data_quality.py` | processed Parkinson quality test, 134 passed |
| `scripts/git_hooks/pre-push` | step 4 now runs the two data-quality tests + blocks on failure when data changes |
| `CLAUDE.md` | new rule: raw-data ingestion MUST run + pass data-quality tests (enforced by hook) |
| `docs/reports/2026-08-16_raw_prices_data_quality_report.md` | per-ticker raw audit |
| `docs/reports/2026-08-16_processed_data_quality_audit_report.md` | per-ticker processed audit |
| `docs/reports/2026-08-16_processed_data_parkinson_verification_report.md` | independent Parkinson recompute (PASS) |
| `results/quality_gate/data_final_2026-08-16/drift.html` | Evidently drift evidence |

## Verification

- `tests/test_raw_prices_quality.py`: 167 passed (no OHLC violations remain; no nonpositive OHLC in
  any ticker).
- `tests/test_processed_data_quality.py`: 134 passed (VHM no longer flagged; all invariants hold).
- Independent Parkinson recompute vs raw: max abs diff ~1e-16, 0 rows differ > 1e-9 (prior agent).
- Quality gate: Pandera SCHEMA PASS (34/34), Evidently DRIFT report emitted.
- All 33 processed series end 2026-08-14.

## Enforcement (harness)

- `scripts/git_hooks/pre-push` step 4: when the push diff touches data/manifest, it runs
  `python -m pytest tests/test_raw_prices_quality.py tests/test_processed_data_quality.py` and sets
  the push to FAIL on any failure (in addition to the existing Pandera/Evidently checks). Guarded to
  skip only if the test files are absent on the pushed branch.
- `CLAUDE.md` Definition of Done: new mandatory rule "Raw-data ingestion quality tests — ENFORCED":
  crawling/appending raw data requires running + passing both tests, reprocessing, syncing, and the
  data-quality gate before commit/use; fix bad data correctly (nonpositive → positive max/min;
  pre-listing backfill → trim to listing date), no silent skips.

## Impact / notes

- Data changed → the in-flight seq44@90/10 run (on the pre-fix data) was cancelled; a clean
  seq44/seq22 @90/10 re-run on the finalized data is the next step.
- VN100 crawl agent runs independently into `data/raw/prices/vn100/` (subfolder, not read by active
  training); its output will also be checked by the ingestion harness.

## DoD checklist

- [x] Data fixed correctly (OHLC positive-aware; VHM trimmed to listing date).
- [x] Tests written + GREEN (raw 167, processed 134).
- [x] Quality gate PASS (Pandera 34/34, Evidently).
- [x] Enforcement wired (pre-push hook) + documented (CLAUDE.md).
- [x] Summary report (this file).
- [x] Push after task (below).
