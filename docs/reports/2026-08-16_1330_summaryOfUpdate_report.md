# Summary of update — GNNHAR (2308.01419) optimizations P1/P2/P3: code + smoke

Implements the three prioritized experiments from
`docs/reports/2026-08-16_1212_gnnhar_2308.01419_optimization_experiments.md` (derived from
arXiv:2308.01419, Zhang et al.) on the delivered baseline `baselines/2026-08-15_volatility`. Plan:
`docs/superpowers/plans/2026-08-16-gnnhar-optimizations-p1-p2-p3.md`.

## What changed (code)

- **P1 — QLIKE training loss** (`code/train_resume.py`, `code/run_ablation.py`)
  - `_masked_qlike(pred_norm, target_norm, presence, mean_node, std_node, eps=1e-8)`: differentiable,
    presence-masked QLIKE on the physical (variance) scale, matching `src.common.evaluation.qlike_loss`
    exactly (denormalize via the per-ticker scaler → positive floored pred + positive target, clamp
    at 1e-8, `mean(ratio − log(ratio) − 1)`).
  - `train_with_resume(..., loss="mse"|"qlike")`: training step and val-selection both use the chosen
    criterion, so the best-val checkpoint is selected on the same objective (paper: select on
    validation). `run_ablation` threads `loss` (env `ABLATION_LOSS=qlike` → `_qlike` results dir +
    ladder `"loss"` field). HAR stays OLS; its QLIKE is still reported.
- **P2 — configurable GAT depth (1-hop) + MAD** (`code/model.py`, `code/mad.py`, `code/run_ablation.py`)
  - `VolatilityModel(..., gat_layers=1|2)`: 1-hop keeps only `gat1` at the same output dim (head
    unchanged); 2-hop (default) preserves current behavior. `gat_layer_outputs()` exposes per-layer
    embeddings. `mad.mad(emb, presence)` = mean over present node pairs of (1 − cosine) — lower =
    more over-smoothed (paper's over-smoothing diagnostic). `run_ablation` threads `gat_layers`
    (env `ABLATION_GAT_LAYERS=1` → `_gat1` results dir).
- **P3 — regime-split (calm/turbulent) metrics + DM** (`code/regime_report.py`)
  - Post-hoc over existing `predictions_test.json` dumps (no retrain): `split_regime`
    (top `turbulent_frac` by target volatility), `regime_metrics` (5 raw-scale + QLIKE per subset;
    DirAcc omitted — not meaningful on a non-contiguous subset), `regime_dm` (HLN, HAC lag h−1 per
    regime). `run_regime` aligns FULL + comparators and reports per-regime metrics + DM(FULL vs each).
    Reuses `dm_report._ensemble/_qlike/LOSSES` + `diebold_mariano`.

## Tests (TDD, all RED→GREEN)

Full baseline suite **45 passed** (was 24 pre-session). New: `test_qlike_loss.py` (4),
`test_ablation_qlike.py` (2), `test_mad.py` (4), `test_gat_depth.py` (4), `test_ablation_depth.py`
(2), `test_regime_report.py` (5, incl. end-to-end runner on synthetic dumps). Ruff clean on all
changed files. Each of P1/P2/P3 committed+pushed after its tests went green (gate passed, real
`gate_results/*.json`: 02a9077, c2fee15, f28d63e).

## Smoke (real VN30 data, GPU) — proves the pipelines run end-to-end

Undertrained (3 epochs, 1 seed, h=1) — **for wiring verification only, NOT reportable numbers.**

- **P1**: QLIKE ablation produced the full ladder (HAR/FULL/minus_graph/minus_gate/minus_news/
  lstm_only). FULL QLIKE 0.4740 vs HAR 0.4633 at 3 epochs (undertrained; HAR still ahead here).
- **P2**: FULL QLIKE — 2-hop 0.4740 vs 1-hop 0.4798 (ladder `gat_layers` meta 2 vs 1 recorded).
- **P3**: regime report over the P1 dumps: turbulent QLIKE (1.21) ≈ 3× calm (0.39); on turbulent
  days DM(FULL vs minus_gate)=−9.07* (removing the gate worsens QLIKE → gate helps under turbulence)
  — the kind of regime-concealed signal a pooled average hides. Preliminary only (3 epochs/1 seed).

Smoke result dirs were deleted after inspection (throwaway 3-epoch runs).

## Not run (deferred — needs user approval per training policy)

Full converged multi-horizon (h∈{1,5,10,22}) × multi-seed runs for MSE vs QLIKE and 1-hop vs 2-hop,
then DM + regime tables + paper-number update. These are hours of sequential GPU (training policy:
>10 epochs / full runs need explicit approval). Commands:
`ABLATION_LOSS=qlike .venv_gpu_encode/Scripts/python.exe .../run_ablation.py <TS> cuda <seed> 15 1 5 10 22`
(+`ABLATION_GAT_LAYERS=1` for the depth study); then `dm_report.py` and `regime_report.py`.

## DoD

- [x] Code = request (P1/P2/P3), surgical (loss/depth params default to current behavior).
- [x] TDD, 45 pass; ruff clean; data-quality tests run in pre-push gate every push.
- [x] Smoke: all three pipelines boot + run on real data.
- [x] Pushed after each task (3 commits, real gate results).
- [ ] Full converged runs + paper update — deferred (user approval).
- Data-quality (Pandera/Evidently): N/A (no data/manifest/feature change; code-only). Pre-push data
  tests still ran.
