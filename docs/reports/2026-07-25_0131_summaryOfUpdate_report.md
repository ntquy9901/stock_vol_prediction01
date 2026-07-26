# Summary Report — Dual-Group News Embedding Baseline (2026-07-25)

**Baseline:** `baselines/2026-07-25_dual_group_news_embedding_baseline/`
**Type:** New baseline (full SDD lifecycle per CLAUDE.md §1.5), autonomous run — user approved
proceeding without per-step confirmation and explicitly requested the 10→20 epoch comparison.

## What changed

Brought the dual-group (khach_quan/tong_hop) PhoBERT-PCA-EWMA news-embedding pipeline from the
sibling project `data_eda` (documented in `data_eda/docs/embedding_pipeline_reference.md`) into
this project as a new baseline, comparing against the existing `2026-07-07_embedding_baseline`
(PCA-64, single-group, 68.76% DirAcc @40ep).

Per explicit user instruction: no file in `C:\luanvan\data_eda` was modified — everything used
was copied into this project first, then worked on from the copy.

### Files (path → purpose)

| Path | Purpose |
|---|---|
| `data/external_news_embeddings/raw_cache/` | Verbatim copy of data_eda's PhoBERT article-embedding cache (48 files, 4.4GB) — the expensive step (PhoBERT encoding), never re-run |
| `baselines/.../requirements/requirements.md` | Specify — problem, data source decision, scope (Simplicity Gate: 146 cols not 480), success criteria |
| `baselines/.../design/design.md` | Plan — 2-stage data flow, file list, isolation, risks |
| `baselines/.../TASKS.md` | Epic → 4 stories → numbered tasks with verify criteria |
| `code/vendor_config.py` | Path config pointing at this project (no hardcoded absolute paths); vendored 30-ticker VN30 list (see Findings) |
| `code/vendor_data_eda/*.py` | Trimmed copies of data_eda's discovery/aggregation code (discover_news, news_embeddings, dual_news_features, phase04 helpers, phobert_embeddings) |
| `code/build_dual_group_panel.py` | One-off script: rebuilds PCA+EWMA aggregation against the copied cache → `data/features/dual_group_news_panel.parquet` |
| `code/dataset_dual_news.py` | `MultiStockDatasetWithDualNews` — HAR + pre-aggregated per-day news vector (no padding/mask needed, unlike the original article-set baseline) |
| `code/model_dual_news.py` | `DualGroupNewsBaseline` — reuses `ParallelLSTMGNN.get_embeddings` (read-only) + new `NewsFeatureLSTM` branch |
| `code/train_dual_news.py` | Train loop, 6 mandatory metrics, learning curves every 5 epochs |
| `test/*.py` | 6 tests: 2 real-data-sample smokes (aggregation against real small cached sources), 2 dataset shape/coverage, 2 model forward/backward |
| `code_review/code_review_2026-07-25.md` | Self-directed adversarial review — 1 HIGH, 1 MEDIUM, 2 LOW findings, all resolved |

## Tests + coverage

- `pytest baselines/2026-07-25_dual_group_news_embedding_baseline/test/ -v` → **6/6 passed**.
- Real-data-sample smoke: `test_build_panel_smoke.py` runs the actual vendored aggregation
  against real (small) cached sources (`vnexpress_objective`, `ssi`), not synthetic fixtures —
  this is what surfaced the ticker-list mismatch during implementation (see Findings).
- Coverage tool (`diff-cover`): **Not run** — not installed in this repo yet (known gap, see
  CLAUDE.md "Tooling gaps"). C0/C1 not measured; manual review + the 6 passing tests are the
  verification for this change.

## Code review result + actions

Self-directed adversarial review (the `/code-review` skill's automated tooling expects a GitHub
PR; the interactive BMAD workflow halts for human input at each step, which conflicted with the
user's explicit "don't wait for approval" instruction for this session). Full findings in
`code_review/code_review_2026-07-25.md`. Summary:

- **HIGH (fixed):** train/test leakage in the shared PCA fit. The vendored `TRAIN_CUTOFF`
  ("2020-01-01", copied from data_eda) assumed one global split date, but this project's actual
  split cuts every ticker at the same row index, giving each ticker a DIFFERENT calendar
  val/test start (earliest: 2010-06-30 for STB/VNM; latest: 2024-11-11 for SSB). ~19/30 tickers'
  own val/test-period news therefore leaked into the PCA "train" fit under the old cutoff. Fixed
  by setting `TRAIN_CUTOFF = "2010-06-30"` (the provably-safe minimum across all 30 tickers) and
  rebuilding the panel + retraining from scratch.
- **MEDIUM (fixed):** the leakage fix's smaller cutoff left `tong_hop`'s own legacy-feature PCA
  with too few pre-cutoff rows for 32 components, crashing with `KeyError`. Fixed by dropping the
  unused legacy-feature computation entirely (this baseline never consumed those columns anyway).
- **LOW (fixed):** panel scope crept from the documented 146 cols to 185 (39 unused legacy
  columns). Fixed with an explicit column allowlist + fail-loud guard.
- **LOW (by design, user-confirmed):** 316 articles newer than the copied cache snapshot are
  skipped, not encoded (never invoke PhoBERT — explicit user decision 2026-07-25).

## Commands run

```
cp -f data_eda/data/features/*.parquet stock_vol_prediction01/data/external_news_embeddings/raw_cache/
python build_dual_group_panel.py                          # x3 (initial, leakage fix, crash fix)
python -m pytest baselines/.../test/ -v                    # 6/6 passed, re-run after each fix
python train_dual_news.py --epochs 10                      # pre-fix
python train_dual_news.py --epochs 20                      # pre-fix
python train_dual_news.py --epochs 10                      # post-fix
python train_dual_news.py --epochs 20                      # post-fix
python train_dual_news.py --epochs 40                      # post-fix, user-requested extension, early-stopped epoch 36
```

## Results — 6 mandatory metrics

**Corrected (post-leakage-fix) results — the ones that matter:**

| Epochs | Split | MSE | RMSE | MAE | R² | QLIKE | DirAcc |
|---|---|---|---|---|---|---|---|
| 10 | Val | 0.000006 | 0.002460 | 0.000731 | 0.657 | 0.700 | **69.68%** |
| 10 | Test | 0.000007 | 0.002636 | 0.000723 | 0.716 | 0.565 | **68.50%** |
| 20 | Val | 0.000006 | 0.002445 | 0.000725 | 0.661 | 0.700 | **70.00%** |
| 20 | Test | 0.000007 | 0.002642 | 0.000723 | 0.714 | 0.556 | **68.25%** |
| 40 (early-stopped ep36) | Val | 0.000006 | 0.002452 | 0.000724 | 0.659 | 0.693 | **70.54%** |
| 40 (early-stopped ep36) | Test | 0.000007 | 0.002640 | 0.000716 | 0.715 | 0.546 | **68.71%** |

**10 → 20 → 40 epoch comparison (user requested both extensions): plateau confirmed, with mild
noise, not a trend.** Test DirAcc across the three runs: 68.50% → 68.25% → 68.71% — bounces
within a ~0.5pp band, no monotonic improvement. Val DirAcc creeps up slightly each time
(69.68% → 70.00% → 70.54%) but Test doesn't track it, which is the classic sign of the val
checkpoint selection just getting slightly lucky rather than genuine improvement. The 40-epoch
run's early stopping (patience=15, triggered at epoch 36) confirms the model had already stopped
improving on val_loss well before the epoch budget ran out. QLIKE at 40 epochs (0.546 test) is
the best of the three runs and close to REST-TS's project-wide-best 0.543 (see the all-baselines
comparison report). **Recommendation: 10 epochs remains the practical sweet spot — the marginal
QLIKE/DirAcc gains from training to 40 don't justify 4x the compute for this architecture.**

**Comparison to existing baseline** (`2026-07-07_embedding_baseline`, PCA-64 single-group,
concat+MLP): 68.76% DirAcc @40ep, or 68.44%/70.29% @5ep. This new dual-group+EWMA baseline's own
40-epoch run (68.71% Test DirAcc) lands essentially ON PAR with the original at the same epoch
budget — not a win, but a genuinely fair, epoch-matched comparison point (unlike the 10/20-epoch
numbers above, which compared fewer epochs against the original's 40). The richer dual-group+EWMA
feature set does NOT clearly beat the simpler single-group PCA-64 approach when both are given
the same training budget; its main edge is QLIKE (0.546 vs the original's 0.553), a small margin.

**For reference only (pre-fix, methodologically unsound, superseded):** 10ep Val 69.83%/Test
68.52%; 20ep Val 69.30%/Test 67.77% — kept in `results/dual_group_news_2026-07-25_00{39,46}*/`
for traceability but should not be cited as this baseline's result.

## Risks / follow-ups

- **PCA training data shrank** (~4 years pre-2010-06-30 vs. the original ~14 years pre-2020) as
  the direct cost of the leakage fix. If a future iteration wants a richer PCA basis, the correct
  fix is making the cutoff genuinely ticker-aware (fit one PCA cutoff per ticker's own val
  start) rather than reverting to a single unsafe global date — not done here (out of scope for
  this baseline; the shared-PCA design itself assumes one global cutoff).
- **diff-cover/ruff not run** — repo-wide tooling gap predating this change (see CLAUDE.md).
- **316 recent articles excluded** from the panel by design (see Findings) — a future run could
  re-copy a fresher cache snapshot from data_eda if the most-recent news matters more than the
  "never invoke PhoBERT" constraint for a given experiment.
- **`load_news_panel`'s per-row `iterrows()`** is unvectorized (146,700 rows) — fine at this
  size, flagged for later if the panel grows.

## Definition of Done checklist

- [x] Code satisfies the request (dual-group news embedding baseline, no rebuild of PhoBERT, no data_eda mutation)
- [x] Tests written + run (6/6 pass); real-data-sample smoke included
- [ ] diff-cover C0=100%/C1≥80% — Not run (tooling gap, pre-existing, see CLAUDE.md)
- [x] Lint — Not run (ruff not installed, pre-existing gap); no obvious style issues on manual read
- [x] Code review (self-directed adversarial) — 1 HIGH + 1 MEDIUM + 1 LOW found and fixed
- [x] Summary report (this file)
- [x] Smoke test(s) pass (tagged `smoke`, boots the real aggregation + model forward/backward)
- [x] Impact analysis — this session only ever used `Read`/`cp` (never `Write`/`Edit`) against
  `C:\luanvan\data_eda`; the specific `data/features/` cache directory copied from is confirmed
  unchanged (48 files, same as before, matches the copy). Note: `git status` in data_eda shows
  unrelated modified/untracked files from that project's own separate work (not from this
  session — no file under data_eda was ever opened with a write tool here). Confirmed no edits
  to this project's shared `src/` or other baselines (read-only imports only).
- [x] Similar-pattern check — n/a (first baseline of this kind in the project)
