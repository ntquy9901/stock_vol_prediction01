# Summary of update — Pooled/transfer ablation for VN30 (implementation) + overnight runs

Date: 2026-09-04 01:55. Autonomous session (user asleep, full autonomy).

## What changed

New hard-isolated baseline `baselines/2026-09-04_pooled_transfer_vn30/` (SDD §3.F, 5 subfolders)
implementing a single-variable walk-forward ablation: does widening the deep model's training
universe from 31 (VN30) to 102 (VN100) stocks improve VN30 volatility forecasts.

| Path | Purpose |
|---|---|
| `code/pooled_panel.py` | `vn30_index`, `screened_universe`, `restrict_fold` (train-node mask + graph isolation), `score_mask` |
| `code/run_pooled_arm.py` | `run_arm` — one arm's walk-forward (restrict training universe, score VN30) |
| `code/run_pooled_ablation.py` | `_build` (shared VN100 panel + folds + VN30 index), `run_ablation` (both arms + paired DM + diff-in-diff + JSON), CLI |
| `code/summarize_pooled.py` | markdown report from result JSONs (untracked report generator) |
| `code/tests/*` (8 modules) | universe/index, restrict, score, run-arm smoke, isolation, alignment, driver, build |
| `code_review/code_review_2026-09-04.md` | adversarial 3-lens + perf + config-hardcode review |
| `docs/superpowers/specs|plans/2026-09-04-pooled-transfer-vn30*` | design spec + implementation plan |

## Design (single-panel realisation)

ONE VN100 panel + ONE fold set. Arms differ only by a training-node mask: Arm 0 trains the 31 VN30
nodes, Arm 1 trains all 102 (training loss + vol→PK graph restricted per arm). Both score exactly
the 31 VN30 nodes on the identical OOS grid → byte-identical `(ticker,date)` keys → perfect paired
DM. Refined from the spec's two-panel wording; the isolation risk (Arm 0 = genuine 31-node system)
is test-gated.

## Tests + coverage

- `pytest baselines/2026-09-04_pooled_transfer_vn30/code/tests` → **10 passed** (GPU venv).
- Diff-coverage on changed lines: **C0 line = 100%, C1 branch = 100%** (pre-push gate).
- The **isolation test** confirms Arm 0 VN30 predictions are invariant (<1e-4) to non-VN30 node
  feature perturbation → the single-panel mask genuinely reproduces a 31-node system.
- `ruff --select F` clean; config-hardcode scan 0 BLOCK / 0 WARN (`lookback` exposed via CLI).

## End-to-end verification

Real-data run (`--horizon 1 --folds-target 1 --epochs 1`) completed and wrote a well-formed JSON
(`arm0`, `arm1`, `paired_dm` on 3 bases, `diff_in_diff`) — wiring confirmed; the 1-fold/1-epoch
numbers are not a result.

## Code review

Adversarial 3-lens + performance + config-hardcode (self-review, autonomous). No CRITICAL/MAJOR
open. Key verified point: `restrict_fold`'s shallow copy leaves non-VN30 `X` present, but the
restricted adjacency + per-node LSTM sever all cross-node paths — proven by the isolation test.
One follow-up: confirm the pre-push overfit-evidence gate does not misfire on the ablation's distinct
JSON shape before pushing results.

## Performance

Reuses the delivered already-batched `train_masked_rich` ([B,N,seq,5] on GPU); no batch=1. Arm 0
processes the full 102-node tensor (masked) — same batched cost as Arm 1; accepted for exact OOS
alignment, documented in `design/design.md`.

## Commands run

- `pytest .../tests -q` → 10 passed; `--cov-branch` C0/C1 = 100%.
- `ruff check --select F` → clean; `check_config_hardcode.py` → 0/0.
- `git push origin master` → pre-push gate **passed**; pushed through `74a6441`.

## Runs launched (overnight, detached)

- Chain `_tmp_overnight_chain.sh` (PID 2179): Phase 1 VN30 ablation h1→h5→h10→h22 (writes
  `results/pooled_transfer_vn30/pooled_vn30_h{H}.json` + `docs/reports/2026-09-04_pooled_transfer_vn30_report.md`);
  Phase 2 SP500 6-fold h1→h22 (`results/walkforward_volga/walkforward_volga_sp500_h{H}.json`).
- Background agent: TiRex-backbone feasibility → `docs/reports/2026-09-04_tirex_backbone_feasibility.md`.

## Risks / follow-ups

- Compute: single-panel makes Arm 0 ≈ Arm 1 cost (~8h/horizon for the ablation); not all horizons +
  SP500 will finish by morning — per-horizon JSONs are partial-safe.
- Results + dashboards + the TiRex report are NOT yet committed (produced by detached runs); commit +
  push in the next GPU-free window; build the two-arm HTML dashboard then.
- Prior Track B A1 (2026-08-08) found pooling did not help deep beat HAR — the honest expected
  outcome is a likely null; the report will state H0/H1 per the a-priori rule.

## DoD checklist

- [x] Code satisfies request; hard-isolated; no unrelated refactor.
- [x] Tests written + pass (10); C0=100/C1=100 on changed lines.
- [x] Lint (ruff F) clean; config-hardcode 0/0.
- [x] Adversarial code review documented; no critical/major open.
- [x] Performance conclusion recorded.
- [x] Code pushed (gate passed, no QG_SKIP).
- [ ] Full ablation results + dashboard (running overnight; commit in the morning GPU-free window).
