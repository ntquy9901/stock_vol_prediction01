# Summary of update — integrate latest price data (through 2026-08-14) + archive vn30_only + re-run

## What changed

1. **Archived** `data/processed/vn30_only` (30-ticker subset) → `archive/data_processed_vn30_only/`.
   Active volatility training reads the top-level `data/processed` (33 tickers) via
   `combo_ladder.py` (`_PROCESSED = ROOT/"data"/"processed"`, non-recursive `*_processed.csv` glob),
   so the subset was unused by the current pipeline. 9 non-archive legacy VN30 scripts still
   reference `vn30_only` (see Blast radius) — not part of the active path.
2. **Appended latest daily prices** to all 33 raw series and regenerated processed volatility,
   extending coverage from 2026-06-09 (31 tickers) / 2026-06-19 (VPB, VRE) / 2026-08-10 (LPB) to a
   uniform **2026-08-14** (latest VN trading day; the source stopped there — 08-15/16 are weekend).
3. **Re-ran** the seq-lookback experiments on the refreshed data under the 90/10 (train+val merged /
   test) protocol requested by the user (see Training).

## Data integration (append, non-destructive source; regenerate processed)

- Source: staged fetch from the audit agent, `data/raw/prices_update_2026-08-16/append_ready/`
  (per-ticker gap rows after each ticker's prior last date, rescaled per-ticker for level
  continuity). Crawler: `src/data/crawl_vietnam_stocks.py` (yfinance `.VN`); LPB via SSI iBoard.
- Backup: `data/raw/prices` copied to `data/raw/prices_backup_pre_update_2026-08-16/` before append
  (local safety; git history is the durable backup — backup dir not committed).
- Append: 1524 rows added across 33 tickers (48 for the 31 long series, 40 for VPB/VRE, 4 for LPB);
  verified no date overlap with existing rows and each ticker strictly monotonic afterwards.
- Reprocess: `python -m src.common.process_parkinson_pipeline` → 33 `*_processed.csv` regenerated
  (columns `date,parkinson_volatility`). Parkinson = (ln(H/L))²/(4·ln2) is scale-invariant, so the
  three raw storage conventions (thousands-VND, full-VND tz-aware for VPB/VRE, SSI for LPB) do not
  affect the target; `process_single_stock` normalizes dates to plain `YYYY-MM-DD`
  (`str.split(' ').str[0]`), so the mixed VPB/VRE datetime format is handled.
- Sync: the 33 processed files copied to `.worktrees/volatility-gat/data/processed/` because the active
  code resolves `ROOT` to the worktree and reads the worktree's own `data/processed`.

## Verification

- Processed: 33 files, all last date = 2026-08-14; no duplicate dates, all monotonic increasing, no
  NaN in `parkinson_volatility`.
- Worktree copy: 33 files, all last date = 2026-08-14.
- **Quality gate (system Python 3.14; pandera 0.32.1 + evidently 0.7.21):**
  - Pandera SCHEMA: **PASS — 34/34 artifacts valid**.
  - Evidently DRIFT: report generated → `results/quality_gate/data_update_2026-08-16/drift.html`.
  - (LINT/TESTS not re-run in this data-only step; the reprocess used the existing tested pipeline
    `src/common/process_parkinson_pipeline.py` + `parkinson_utils.py`, covered by
    `tests/test_data_processing.py`.)

## Training (re-run on refreshed data)

- Prior in-flight runs on the OLD data were cancelled per user instruction.
- Protocol: train+val merged to 90%, test 10% (`ratios=(0.80,0.10,0.10)`), fixed 15 epochs, no
  early-stop, test read once (no selection on test → no leakage). All 6 rungs (HAR, FULL,
  minus_graph, minus_gate, minus_news, lstm_only) via `run_retrain_trainval.py`.
- Running: seq=44 lookback (2 processes, h1/h5 and h10/h22); seq=22 auto-chained after (file-based
  trigger on the h5/h22 result JSONs). Comparison seq44-vs-seq22 under identical 90/10 on new data.
- Runner override (throwaway): `combo_ladder.SEQ` and a `load_and_split_price_data` ratio patch
  `(0.80,0.10,0.10)`; `features._VOLUME_WINDOW=22`.

## Blast radius (vn30_only archive)

9 non-archive scripts reference `vn30_only` and will not find it after the move (all legacy,
pre-volatility): `train_lstm_har_vn30.py`, `train_har_vn30.py`, `train_simple_lstm_vn30.py`,
`train_all_models_vn30.py`, `test_har_leakage_fix.py`, `src/experiment/train_with_config.py`,
`src/lstm_har_enhanced/train_with_overfitting_prevention.py`, `src/train_with_config.py`. Left
as-is (not deleted, not path-updated) pending a user decision on whether to archive/update them.

## Follow-ups

- seq44 / seq22 @90/10 results on new data → all-metric + DM comparison when runs finish.
- Downstream reports/papers currently cite the OLD test window (to 2026-06); they will need
  refreshing if the new-data results are adopted.
- `/code-review` + diff-cover: data-only change (no new production code); pipeline is pre-existing.

## DoD checklist

- [x] Change matches request (integrate data + archive vn30_only + re-run).
- [x] Data verified (dates, monotonic, no NaN) + quality gate (Pandera PASS, Evidently report).
- [x] Backup taken before mutation.
- [x] Summary report (this file).
- [x] Push after task (archive move pushed d66a107; data commit below).
- [ ] seq44/seq22 new-data results + DM — pending run completion.
