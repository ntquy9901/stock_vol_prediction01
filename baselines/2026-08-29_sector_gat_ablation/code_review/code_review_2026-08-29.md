# Code review — sector-graph ablation (2026-08-29)

Adversarial self-review (Blind Hunter + Edge Case Hunter + Acceptance Auditor lenses) before "done".
Scope: `code/` and `test/` of this baseline only. Live-training-path files are out of scope (imported
read-only, never edited).

## Findings & resolutions

### Blind Hunter (hidden bugs)

- **[MAJOR — FIXED] CPU-force could leak the GPU.** `run_sector_ablation.py` originally used
  `os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")`. If the caller had `CUDA_VISIBLE_DEVICES=0`
  already exported, `setdefault` would keep it and the run would land on the live-GPU job — violating
  the hard constraint. Changed to a hard assignment `os.environ["CUDA_VISIBLE_DEVICES"] = ""` under
  `SECTOR_ABLATION_FORCE_CPU=1` (the default). Verified the `device` field reports `cpu`.
- **[MINOR — ACCEPTED] `_OWN` sentinel collision.** A real sector literally named `"__own__:<TICKER>"`
  would merge with an unmapped node. Impossible for GICS/ICB labels; accepted.
- **[INFO] Private-symbol imports** (`RMR._pred_dict/_ens/_metrics/_dm_all`, `EFA._write_estimator_processed`).
  These are the documented reuse surface (the task specifies them); imported read-only, no edit. Acceptable
  per the isolation rule.

### Edge Case Hunter

- **Unmapped tickers** → singleton own-sector (self-loop only); two different unmapped tickers do NOT
  share a bucket. Covered: `test_unmapped_ticker_is_singleton_own_sector`,
  `test_two_unmapped_tickers_do_not_share_a_sector`.
- **`top_k` boundaries:** `top_k=0` → self-loop only; `top_k<0` → `ValueError`; `top_k=None` →
  fully-connected. Covered.
- **Empty test split / non-finite forward output** → explicit `RuntimeError`. Covered
  (`test_forward_pass_smoke_empty_test_raises`, `test_forward_pass_non_finite_raises`).
- **<2 processed files** → `RuntimeError` (`test_build_panel_masked_too_few_raises`).
- **Blank sector cell** in CSV/frame → treated unmapped (dropped), not an empty-string sector. Covered
  in `load_sector_map` and both fetch builders.
- **Ticker case / dot-dash variants:** VN uppercased; GICS `.`→`-` (BRK.B→BRK-B). Covered.

### Acceptance Auditor (vs requirements)

- Builder TDD property tests present and passing (RED→GREEN confirmed). ✅
- HNX coverage reported (98.8%, 160/162, 23 ICB sectors). ✅
- CPU smoke aligns to `D.tickers` + finite forward pass, no training loop. ✅
- 3-way comparison, all 5 metrics + date-clustered DM, produced on CPU, labelled directional. ✅
- No live file edited (grep-verified: only new files under this baseline). ✅
- Provenance CSVs with fixed date string (no `datetime.now()`). ✅

## Performance lens (train/inference code)

- No new batch=1 loop introduced: training reuses the delivered batched GAT (`batch_size=512`,
  block-mask adjacency, single tensor forward). The only per-item Python loop is the intended
  seed/variant loop (3 variants × N seeds) — sequential by necessity (shared CPU), documented.
- CPU is forced **only** to protect the live GPU job, not because the compute is CPU-bound; the
  ready-to-run scale-up flips `SECTOR_ABLATION_FORCE_CPU=0` for GPU.

## Tests & gate

- 34 tests pass under `.venv_gpu_encode` (GPU venv). Diff-coverage on changed lines: **C0 line 100%**,
  **C1 branch 98%** (≥95 gate). Residual branch partials are the module-level CPU-force guard, two
  dry-mode convenience `if`s, and one sector inner-loop arc — all non-defect.
- ruff `--select F` (the blocking set): clean. `E702` semicolons match existing house style
  (WARN-only per CLAUDE.md).
- Data-quality gate: **N/A (no data change)** — no `data/` files added/modified; sector CSVs are new
  metadata artifacts inside the baseline folder.

## Verdict

No open critical/major findings (the one MAJOR was fixed). Ready to commit.
