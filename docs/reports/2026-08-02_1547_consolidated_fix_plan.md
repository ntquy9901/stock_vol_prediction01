# Consolidated Fix Plan — Paper Readiness (2026-08-02)

## 1. Scope note

This consolidates four reports — the original audit
(`2026-08-02_1056_paper_readiness_audit_report.md`), its fix-round summary
(`2026-08-02_1527_summaryOfUpdate_report.md`), the independent adversarial re-audit
(`2026-08-02_152253_summaryOfUpdate_report.md`, findings AUD-001..AUD-024, verdict
**NO-GO**), and the proposed `verify-audit-fixes` skill spec
(`2026-08-02_152758_summaryOfUpdate_report.md`) — against the **current** repo state, verified
directly (not read off the reports).

**What changed between report-time (~15:27) and now (~15:47), undocumented in any report:**
The working tree is dirty with real, in-progress, **uncommitted** work not mentioned in any of
the four reports:
- `train_simple_lstm_vn30.py`, `train_lstm_har_vn30.py`, `src/lstm_baseline/dataset.py`,
  `src/lstm_har_baseline/dataset.py` (unstaged `M`): a genuine per-ticker temporal-split fix for
  AUD-003/original-audit-§1.2 has been implemented for **2 of the 5** flagged files (train-only
  scaler fit, per-ticker chronological cut). Not run, not tested, not committed.
- `project-context.md` (unstaged `M`): partial fix for AUD-023/original-§1.10 (Feature
  Categories, MODELS_TO_COMPARE, header date corrected).
- `baselines/2026-07-25_news_usefulness_ablation/test/test_ablation_deltas.py` (untracked): a
  real smoke test now exists for the baseline AUD-020 flagged as missing one. **Verified: passes**
  (`pytest ... -v` → 2 passed).
- `results/per_ticker_gate_2026-08-02_*` (7 dirs, staged `A`) and the two newest audit reports
  (`152253`, `152758`, untracked) are sitting uncommitted alongside commit `fccaf6a`.
- `AGENTS.md` (untracked, 1000 lines, root) — unrelated to the audits, not investigated further;
  flagged only so it isn't lost in a future `git clean`.

Everything below reflects fresh verification (git, grep, `pytest --collect-only`, `ruff check`),
not report prose.

## 2. Already fixed and verified (do not re-plan)

| Item | Evidence |
|---|---|
| Seeding added to ~23 training scripts | commit `fccaf6a` (`git log --oneline -5`) |
| DirAcc per-ticker fix wired into 22 news-fusion baseline files (`n_stocks=`) | commit `fccaf6a`; confirmed absent from the 4 non-news-fusion `lstm_gat_hybrid` trainers (see AUD-004 below — not fully fixed project-wide) |
| Per-ticker-gate 5-seed epoch-20 reproducibility check | `results/per_ticker_gate_2026-08-02_{150559,150913,151224,151827,152448}/results.json` exist (staged, uncommitted); mean 0.5530±0.0115 QLIKE, worse than REST-TS single-seed 0.5431 on 5/5 seeds |
| `ruff` is installed and runnable (CLAUDE.md's "not installed" note is stale) | `ruff --version` → `0.16.1`; `ruff check src tests baselines --exclude ...` → **947 errors, 583 auto-fixable** — reran fresh just now, count matches AUD-013 exactly, so treat as current fact, not stale |
| AUD-020 (missing test for `news_usefulness_ablation`) | `test_ablation_deltas.py` exists, `pytest.mark.smoke`, 2/2 pass — **but uncommitted**, so not "done" until committed |

## 3. Prioritized fix plan

### Phase 1 — Critical, blocks any claim from the core LSTM-GAT architecture

**P1.1 — Normalizer leakage in `src/lstm_gat_hybrid/dataset.py`** (merges AUD-001; no
counterpart in the original audit — genuinely new).
- What's wrong: `_initialize_normalizers()` (line 172, called from `__init__`) fits
  `VolatilityNormalizer` on the entire stock CSV before any split. `create_multi_stock_dataloaders`
  (lines 412-478) then instantiates **three full `MultiStockDataset` objects** — each re-runs
  `__init__` (and re-fits normalizers on 100% of the data) — and only afterward slices each with
  `torch.utils.data.Subset` by position (lines 481-483). Verified still present at these exact
  lines as of this session.
- Why it matters: every reported LSTM-GAT metric (RMSE/QLIKE/DirAcc) is inflated by test/val
  leakage into the normalization statistics — this is the project's core architecture (per
  CLAUDE.md §4), not a side experiment.
- Fix approach: refactor so normalizers are fit **once**, on the train-only date range, then
  reused (not refit) for val/test instances — mirroring the pattern used by
  `src/common/temporal_split.py`'s `TemporalSplitter` (fit-on-train, transform-elsewhere) and the
  in-flight fix already applied to `PooledVolatilityDataset`/`HARVolatilityDataset` (see the
  uncommitted diff in §1): compute the train/val/test index cut first, pass a `fit_stats=None`
  vs `fit_stats=train_stats` argument into `MultiStockDataset.__init__`, and only call
  `.fit()` when `fit_stats is None` (train instance). Requires **retraining** to get valid
  numbers — bounded by CLAUDE.md's 5-10 epoch experimentation policy before any >10 epoch run is
  approved.

**P1.2 — Cross-stock date misalignment in `src/lstm_gat_hybrid/dataset.py`** (AUD-002, new).
- What's wrong: `_load_multi_stock_data` (lines 180-241) removes outliers **independently per
  ticker** (`remove_outliers()`, lines 222-229), so each stock's `DataFrame` ends up with a
  different row count / different dates removed. `_create_sequences` (lines 278-330) then
  indexes every ticker's frame with the **same positional** `i:i+seq_length` / `iloc[target_idx]`
  (lines 305-315) and stacks results with `np.stack(..., axis=1)`. Position `i` does not
  correspond to the same calendar date across stocks once outlier removal has desynced them.
- Why it matters: the graph/panel structure — the entire point of the LSTM-GAT hybrid — silently
  mixes dates across stocks; graph edges (correlation/spillover, `graph_utils_fixed.py`) are
  computed over temporally incoherent panels.
- Fix approach: build sequences from a date-indexed join (e.g., `pd.concat` on a shared `date`
  index with `outer`/`inner` join per split window, or drop outlier rows only for the target
  column while keeping the row so the index stays aligned) instead of positional stacking; assert
  matching dates across tickers before `np.stack`. Requires retraining, same policy as P1.1 — do
  both fixes together since they touch the same `dataset.py` and both require a rerun.

**P1.3 — Headline DirAcc still flatten-biased in the original LSTM-GAT trainers** (merges AUD-004
with original-audit §1.3; the 9-baseline news-fusion fix already committed does **not** cover
these 3 files — confirmed still missing `n_stocks=`):
- `src/lstm_gat_hybrid/train.py:195`, `train_parallel.py:195`,
  `train_parallel_enhanced.py:378,422` — all call `evaluate_predictions(all_targets, all_predictions)`
  with no `n_stocks=`, so console/`results.json` headline DirAcc is the flattened, cross-ticker
  version already shown (via the horizon-1 case) to sit **below random** once corrected.
- Fix: same one-line change already applied to the 22 news-fusion files —
  `evaluate_predictions(all_targets, all_predictions, n_stocks=len(dataset.stock_names))` (or
  equivalent variable already in scope in each function). Pure code fix, **no retraining needed**
  — DirAcc can be recomputed from existing saved predictions if raw predictions were persisted,
  otherwise needs a rerun at existing epoch count (not a new experiment).

### Phase 2 — High severity, blocks specific baseline families or test/CI trust

**P2.1 — Commit the in-progress AUD-003 fix and finish the remaining 3 files.**
`train_simple_lstm_vn30.py`/`train_lstm_har_vn30.py`/underlying datasets already have an
uncommitted per-ticker temporal-split fix (§1) — needs a test run to confirm it executes, then
commit. `src/experiment/train_best_lstm.py:115`, `optimize_lstm.py:105`,
`optimize_lstm_fast.py:181` still call `torch.utils.data.random_split` on ordered time-series data
— confirmed present, unfixed. Fix: apply the same `split=`/train-only-scaler pattern, or point them
at `TemporalSplitter` directly if their data shape allows. Pure code fix; a rerun is only needed if
these scripts' output numbers are cited in the paper (original audit already confirmed they are
not currently cited).

**P2.2 — TimesNet leakage + unspecified CSV selection** (AUD-005, AUD-006 — not covered by any
prior audit round). `src/timesnet_baseline/dataset.py:55-74,135-145,255-271` fits scalers before
split (same class of bug as P1.1); `dataset.py:84-93` + `train.py:288-300` pick `csv_files[0]` from
an unsorted glob with no ticker check. Fix: mirror P1.1's fit-train-only approach; replace
`csv_files[0]` with an explicit configured ticker/path. Needs retraining if TimesNet numbers are
paper-relevant (see call-out §4).

**P2.3 — Repository test discovery/smoke gate is broken** (merges AUD-007/008/009/019). Fresh
`python -m pytest --collect-only -q` just now: **220 collected, 4 errors** (same 3 baseline
families: `2026-07-08_market_fallback`, `2026-07-18_alignment_loss_baseline`,
`2026-07-18_gated_crossattn_baseline` — generic module names `model_embedding`/`dataset_embedding`
collide via `sys.path` when collected together). `pytest.ini:4-6` `testpaths` still only
`src`/`baselines` — confirmed root `tests/` (26 files) excluded from default discovery.
`python -m pytest -m smoke -q` still fails at collection for the same reason (93 deselected, 4
errors). Fix: rename the colliding modules to baseline-unique names (e.g.
`market_fallback_dataset_embedding.py`) or isolate each baseline's `sys.path` insert/pop around
import (real refactor, touches ≥3 baseline `code/` dirs); add `tests` to `testpaths`. Pure code
fix, no retraining — but re-verify the whole repo `pytest --collect-only` and `-m smoke` afterward.
Also found in passing, same root class: `tests/test_evaluation.py:24` imports the non-existent
`src.evaluation` (should be `src.common.evaluation`) — one-line fix, currently outside default
discovery so silently broken.

**P2.4 — Headline result selection unresolved** (merges AUD-010/AUD-011 with original-audit
§1.1/1.5/1.6). Confirmed unchanged: REST-TS (0.5431) has never been multi-seed verified; per-ticker
gate's verified 5-seed mean (0.5530±0.0115) is worse than REST-TS's single-seed number on 5/5
seeds. No canonical one-table comparison exists yet. Fix: run REST-TS 3-5 seeds at the same
epoch used for its 0.5431 number (retraining, bounded by 5-10 epoch policy per run, needs
sign-off if the historical number came from >10 epochs), then build one comparison table (HAR-only,
REST-TS, per-ticker-gate, matched epoch/seed-count/split) before choosing the paper headline.

### Phase 3 — Medium severity, needed for scientific rigor but not architecture-blocking

- **AUD-012 / original §1.7 (VN30 universe stale)** — merge; still open, still un-frozen (32
  vs official membership, VPB/VRE gap). Needs an explicit decision: freeze to a documented
  point-in-time cohort or add a Limitations caveat. Pure doc/config fix once the ticker list is
  decided — no retraining unless the universe itself changes (in which case every model needs
  rerunning — flag as high-cost if chosen).
- **AUD-015 (result provenance incomplete)** — extend whatever function writes `results.json`
  (shared across baselines) to also record git SHA, seed, config hash. Pure code fix.
- **AUD-016 (non-finite metrics can be persisted)** — `src/common/evaluation.py:16-61` plus
  training scripts' `json.dump` calls: add `allow_nan=False` (or explicit finite-check before
  dump) so a NaN/Inf metric fails loudly instead of silently landing in a results file. Pure code
  fix, add a unit test with a deliberately-NaN input.
- **AUD-017 (TimesNet has no seeding)** — same pattern as the ~23 already-seeded scripts; add
  `torch.manual_seed`/`np.random.seed` to `src/timesnet_baseline`'s active training path. Pure
  code fix, trivial, do alongside P2.2 since it's the same file family.
- **AUD-018 (`temporal_split.py` edge-case guards)** — `src/common/temporal_split.py:25-118`
  lacks validation for invalid ratios, empty partitions, unparseable/NaT dates, duplicate dates,
  mixed timezones. Since P1.1/P1.2/P2.1 fixes are pushing more code toward using this utility,
  hardening it first reduces risk of a new leakage bug hiding in an edge case. Pure code fix +
  unit tests (this is exactly the kind of pattern CLAUDE.md's memory note
  `cross-project vendoring` warns about — don't copy split-boundary math, reuse this utility).
- **AUD-022 (no statistical inference)** / original §3.5 — for the per-ticker-gate 5-seed data
  that already exists, compute mean/std/CI now (no retraining needed, just an analysis script). A
  proper paired/bootstrap comparison against REST-TS requires REST-TS's multi-seed run first
  (depends on P2.4).
- **AUD-023 / original §1.10 (docs stale)** — partially fixed uncommitted (`project-context.md`,
  see §1); finish and commit. Pure doc fix.
- Original-audit §1.8 (two pairs of suspiciously identical 11-13-digit metrics across unseeded
  runs) — not covered by any AUD item, still uninvestigated. Root-cause it (checkpoint/cache reuse
  vs results.json overwrite bug) before citing either run.
- Original-audit §1.9 remaining structural gaps (`2026-07-11_sentiment_decay` missing results dir,
  `2026-07-15_objective_news_baseline` incomplete) — see call-out §4 (AUD-021 overlaps the second
  one).

### Phase 4 — Final assembly (depends on Phases 1-3)

- **AUD-024 (no submission package)** — canonical comparison table, manuscript, bibliography,
  reproducibility statement, limitations. Blocked on Phase 1-3 producing a settled headline model
  and verified numbers.

## 4. Explicit call-outs needing a human decision

1. **Which architecture is the paper's headline model?** If it's the news-fusion lineage
   (`baselines/2026-07-26_per_ticker_news_gate_baseline`, already shown to lose to REST-TS on
   5-seed QLIKE — P2.4), then P1.1/P1.2's core `lstm_gat_hybrid` bugs matter mainly as the
   **backbone** the news-fusion models are built on (their trainers likely reuse similar dataset
   code — needs a quick check) rather than as a standalone headline. If the paper still intends to
   present the plain LSTM-GAT hybrid as a result in its own right, P1.1/P1.2 are outright blocking
   and must be fixed + retrained before any number from it is cited.
2. **Is TimesNet (AUD-005/006/017) in scope for the paper at all?** If not, deprioritize P2.2 to
   Phase 3/defer.
3. **VN30 universe** (§3, AUD-012): freeze-to-official vs. document-as-limitation — changes cost
   dramatically (freeze = rerun everything cited).
4. **`src/experiment/train_best_lstm.py`/`optimize_lstm*.py`** (P2.1 remainder): confirm none of
   their numbers are cited anywhere before deciding whether to fix or just delete/archive them.
5. **`2026-07-15_objective_news_baseline`** (original §1.9, AUD-021): finish it or formally close
   as abandoned — currently ambiguous.
6. **REST-TS multi-seed rerun** (P2.4): epoch count for the rerun must match whatever epoch
   produced the historical 0.5431 — confirm that epoch number before spending a training budget;
   if it's >10 epochs this needs explicit sign-off per CLAUDE.md's training policy.

## 5. Out of scope / defer

- **`verify-audit-fixes` skill** (report 4, VER-001..VER-011, Gates 1-11): a real, well-specified
  process recommendation, but it is tooling to catch *future* false "fixed" claims, not itself a
  paper blocker. Defer until after Phase 1-2 land; building it now would compete for time with the
  actual fixes it's meant to verify.
- **Code hygiene for public release** (original §3.10: hardcoded `D:\` paths, bare `except:`) —
  real but doesn't block submission unless "code available" is claimed at submission time.
- **Related-work / positioning** (original §3.9) — explicitly out of this audit's scope; a writing
  task, not a code-fix task.
- **AUD-019's deeper import-isolation refactor** beyond the minimal rename fix in P2.3 — full
  namespace isolation (e.g., per-baseline subprocess execution) is more than needed to make
  `pytest --collect-only` pass; the rename is sufficient for paper purposes.
