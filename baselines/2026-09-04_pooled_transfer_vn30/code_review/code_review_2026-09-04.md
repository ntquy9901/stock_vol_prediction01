# Code review — Pooled/transfer ablation for VN30 (2026-09-04)

Adversarial 3-lens (Blind Hunter / Edge-Case Hunter / Acceptance Auditor) + performance +
config-hardcode lenses. Self-review (autonomous session); no CRITICAL/MAJOR open.

## Blind Hunter (hidden bugs)
- **`restrict_fold` shallow copy** via `dataclasses.replace`: only `adj_vol2pk` + 4 masks are replaced
  (copied); `X_tr/X_va/X_te` are shared with the original `D`. For Arm 0 the non-VN30 nodes' `X` are
  still present in the forward pass. **Verified safe** by `test_isolation.py`: Arm 0 VN30 predictions
  are invariant (<1e-4) to perturbing non-VN30 node features — the restricted adjacency + per-node
  LSTM sever every cross-node path. If the net ever gains cross-node mixing, that test fails loud.
- **Arm 1 = full training:** `restrict_fold(D, arange(N))` keeps all nodes (mask all-True, adjacency
  unchanged) → equals unrestricted `D`. Confirmed by inspection; Arm 1 trains on all 102 nodes.
- **Scoring source:** `score_mask` is built from the ORIGINAL `D.tmask_te` (not `Dr`) in both arms →
  identical VN30 `(node,date)` keys. Confirmed by `test_alignment.py`.
- **HAR/HAR-X train universe:** `_har_ols_preds(Dr, ...)` fits pooled-OLS on `Dr.tmask_tr`, so Arm 0
  fits HAR on VN30 rows only and Arm 1 on all 102 — the train-universe variable applies to HAR too
  (intended; diff-in-diff interprets it).

## Edge-Case Hunter
- VN30 ticker screened out of the VN100 panel → dropped from the score set with a printed notice
  (defensive; VN30 ⊂ VN100 so it does not trigger on real data — `# pragma: no cover`).
- Small panels / folds: covered by `test_run_arm_smoke` (folds_target=1) and `test_alignment`
  (folds_target=2). `assert_no_leakage` runs on the shared folds.
- `nfloor` uses `Dr.t_mean` (unchanged by `restrict_fold`) → identical positivity floor across arms.

## Acceptance Auditor (vs spec)
- One independent variable (training node set): ✓ only `train_idx` differs between arms.
- Identical OOS grid: ✓ single panel + single fold set; `test_alignment` asserts identical keys.
- Headline paired DM (Arm1 vs Arm0, LSTM & VolGA, 3 bases QLIKE/SE/AE): ✓ `_dm_all`.
- Secondary diff-in-diff gap(deep−HAR): ✓ `_diff_in_diff`.
- Leakage: ✓ reused train-only scalers/graph (`pack_fold`) + `assert_no_leakage`.
- Isolation (Arm 0 = genuine 31-node system): ✓ test-gated.

## Performance lens
Reuses the delivered already-batched `train_masked_rich` (`[B,N,seq,5]` on GPU, mask-aware loss,
per-node train scalers). No batch=1 loop. Arm 0 processes the full 102-node tensor (loss + graph
masked to VN30) — same batched cost as Arm 1; accepted as the price of exact OOS alignment and
documented in `design/design.md`.

## Config-hardcode lens
`check_config_hardcode.py` → 0 BLOCK / 0 WARN. `lookback` exposed via `--lookback` (default 22, the
approved experiment value = variation over canonical `pc.LOOKBACK`=10, matching the delivered VolGA
walk-forward); floors/val-tail/test-frac/seeds/epochs all sourced from `pipeline_config` /
`training_config`.

## Tests + coverage
`pytest baselines/2026-09-04_pooled_transfer_vn30/code/tests` → 10 passed. Coverage on the three new
modules: C0 line = 100%, C1 branch = 100% (`--cov-branch`); residual `# pragma: no cover` only on the
defensive dropped-ticker branch and the default-out path. `ruff check --select F` clean.

## Follow-ups (minor)
- The pre-push overfit-evidence gate keys on masked_rich `result.json` shape; this ablation writes a
  distinct JSON (`arm0/arm1/paired_dm/diff_in_diff`). Confirm the gate does not misfire on
  `results/pooled_transfer_vn30/*.json` before pushing results (plan Task 10).
