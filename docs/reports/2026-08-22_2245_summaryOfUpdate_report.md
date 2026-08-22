# Summary of update — output-param option, adversarial code review, triage + fixes

Date: 2026-08-22. Scope: the `--output-param` runner option (config C, review-validated), a 2-layer
adversarial code review of it (per DoD), triage of findings, and the fixes applied.

## What changed
- `baselines/2026-08-21_har_anchored_residual/code/run_masked_rich.py`: added `--output-param
  {zscore_floor(default)|ratio_exp}`; the deep model can now use node-scaled ratio target + exp output
  (positive by construction, no relative floor) instead of z-score target + linear denorm + 1e-2 floor.
  `ratio_exp` writes to a SEPARATE dir (`results/masked_rich_ratio_exp/`) and never clobbers `floor1e2`.
- `scripts/garch_masked/`: output-parameterization ablation scripts (`ablation_vn_5seed.py`,
  `ablation_output_param.py`), GARCH add-on (`compute_garch_masked.py`), + tests.
- Fixes from the review (this update): see below.

## Code review (DoD) — 2-layer adversarial, `/bmad-code-review`
Two parallel layers on the 542-line diff (Blind Hunter = diff only; Edge Case Hunter = diff + repo).
Acceptance Auditor skipped (no spec). Verdict: **no HIGH functional bug** — no leakage, no crash on normal
use, no data corruption; `ratio_exp` verified correct (train/infer links match, bias-match exp(0)=1, clamp,
machine-eps); no clobber of `floor1e2`; `_flat`/DM date alignment correct; per-seed vs ensemble honored;
report numbers + statistical wording match the JSON results.

### Findings and resolution
| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | MAJOR | Default `zscore_floor` is NOT byte-identical to the *committed* prior code: the positivity floor is `1e-2` while commit `0871b15` had `1e-3`. | **Clarified, not a regression.** The delivered `results/masked_rich_floor1e2/` (which the paper uses) were generated with `1e-2`; the current default reproduces them (confirmed: HAR-X h1 vn100 QLIKE = 0.5115, matches the paper, and the GARCH basis-guard passed at 1e-2). The `1e-3` committed baseline was a *stale, un-synced* version; a prior uncommitted `1e-3→1e-2` change was inadvertently bundled into commit `765f80c`, whose message overclaimed "byte-identical". This report is the correction. Fixed the misleading `gentler 1e-2` comment (1e-2 > 1e-3 is a HIGHER floor). Orphan `results/masked_rich/` (1e-3) is historical (referenced only by the qlike-blowup diagnosis report), left in place. |
| 2 | MAJOR/doc | Under `ratio_exp`, the deep model uses a machine-eps floor while HAR/HAR-X keep the `1e-2*mean` relative floor — mixed prediction-floor across compared models. | QLIKE compares all models at the shared `cfg.qlike_floor=1e-8`, so the QLIKE contrast is fair; documented the floor treatment in the runner comment. For a fully floor-consistent paper suite, **config B (ratio + linear + 1e-2 floor)** is an alternative that keeps the SAME floor for every model while capturing the primary (ratio-normalization) stability gain — pending user decision (B vs C as primary). |
| 3 | MINOR | `ablation_output_param.py` (S&P500 diagnosis) omits bias-match. | Acceptable: S&P500 is used only for the instability diagnosis; the fair C-vs-D is done on VN with bias-match. Noted in the robustness report. |
| 4 | MAJOR | `ablation_vn_5seed.py` DM block KeyErrors if run on a config subset excluding C/D. | **Fixed:** extracted `_dm_pairs(ens, har_v, dm_fn)` that emits only contrasts whose configs are present; added a regression test. |
| 5 | MINOR | GARCH DM key `GARCH_vs_HAR` actually compares vs HAR-X. | **Fixed** the code label to `GARCH_vs_HARX` (+ comment); the paper does not use this DM key. Existing JSONs keep the old key (documented). |
| 6 | MINOR | GARCH docstring said "train-valid" but code fits train-only. | **Fixed** docstring to "TRAIN-ONLY" (the correct, leakage-safe behavior). |
| — | cleared | "default output dir renamed"; DM sign convention; train/infer link mismatch | Non-issues (default dir unchanged = `masked_rich_floor1e2`; sign convention correct; links match). |

## Tests + lint
- `scripts/garch_masked/test_ablation.py` + `test_garch_masked.py`: **7 passed** (incl. new `_dm_pairs` guard test).
- `baselines/.../code/test_masked_rich.py`: 10 passed (incl. the `ratio_exp` positivity test).
- ruff clean on edited files (pre-existing semicolon/E501 style in the delivered runner is unchanged, warn-only).
- diff-cover: `Not run` (tooling not installed in this repo — see CLAUDE.md tooling gap).

## Runs in flight (independent resources)
- **Config-C suite** (GPU): VN30/VN100 × {1,5,10,22} × 5 seeds, `--output-param ratio_exp` → `results/masked_rich_ratio_exp/` (6/8 cells at time of writing). Bounded-2 concurrency (VN panels small; no VRAM thrash).
- **HOSE+HNX crawl** (network): complete — 405 + 299 tickers, all `status=ok`.
- **HOSE/HNX ETL** (CPU): complete — `data/processed/{hose,hnx}` (405 + 299), Pandera PASS; report `2026-08-22_1530_hose_hnx_raw_review.md`.

## Risks / follow-ups
- **Config B vs C decision** (finding #2): pick the paper-suite primary — B (floor-consistent) vs C (positive-by-construction) — before locking the re-run into the paper.
- Very short HOSE histories (LPS 4 rows, DMX 12) recommended for exclusion; stale last-dates flagged for delisting review before any live-universe use.
- diff-cover tooling still not installed (C0/C1 gate `Not run`).

## DoD checklist
- [x] Code satisfies request (output-param option) — reviewed correct.
- [x] Tests written + pass (incl. regression test for the fixed KeyError).
- [x] Lint run (ruff, warn-only pre-existing style).
- [x] `/code-review` (2-layer adversarial) run + all MAJOR triaged/fixed or documented.
- [x] Smoke (ratio_exp CLI + masked-rich test) pass.
- [ ] Data-quality gate: N/A for this code change (no data/manifest change); the separate HOSE/HNX ETL ran Pandera (PASS).
- [x] Summary report (this file).
- [x] Push after task.
