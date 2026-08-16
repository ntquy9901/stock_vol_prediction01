# Root-cause: silent volume-feature zeroing (VN100 wrapper) + prevention

## The bug
An exploratory VN100 wrapper (`scripts/run_vn100_ablation.py`) set `combo_ladder._PROCESSED` to the
VN100 processed dir but left `combo_ladder._PRICE_DIR` at the VN30 raw dir. `features.volume_zscore_series`
returns all-zeros for any ticker whose `*_ohlcv.csv` is absent in `price_dir` (intended for a rare
volume-less ticker). Result: the `volume_zscore_20` node feature was **silently zeroed for ~71/104
VN100 tickers**, with no error. (This did NOT cause the low VN100 R² — that is a phantom-holiday
zero-inflation + pooled-R² + hard-universe effect, see `2026-08-16_r2_anomaly_investigation.md` — but
it is a real silent-correctness defect that headline VN100 numbers must not carry.)

## Why it slipped the quality gate / harness / dashboard (5-whys)
1. **Why zeroed?** wrapper set one of two COUPLED globals (`_PROCESSED`) but not the other (`_PRICE_DIR`).
2. **Why no failure?** `volume_zscore_series` degrades SILENTLY (returns zeros) for a missing ticker —
   designed for one legit case, but it masks a mass-missing config error.
3. **Why not caught by tests?** it lived in a throwaway experiment wrapper (not TDD-covered), and the
   pre-push hook did NOT run the delivered baseline's own tests (its PILOT test dir had been removed →
   the hook fell back to `scripts/quality_gate`, which can't import torch anyway).
4. **Why easy to misconfigure?** experiments monkeypatch loose module globals; coupled globals were
   never validated together, and there was no `--universe` CLI.
5. **Why dashboard didn't flag?** the dashboard tracks per-commit GATE pass/fail on production commits,
   not the semantic correctness of an exploratory experiment's config.

**Root cause = silent degradation on misconfigured input + coupled-config-via-loose-globals + the
delivered baseline's tests running outside the gate.** Not a scale/GNN bug (scale verified correct).

## Prevention (implemented this change)
1. **Fail loud (features.py `_check_price_coverage`):** `augment_split_frames` now RAISES when >1
   ticker lacks a raw `*_ohlcv.csv` in `price_dir` (bounded allowlist for the rare legit case). The
   exact bug case now raises "71 of 104 tickers have no *_ohlcv.csv …". +3 unit tests.
2. **Gate the delivered baseline tests (pre-push step 5):** the hook now runs
   `baselines/2026-08-15_volatility/test` + the feature-guard test via the GPU venv on EVERY push
   (the generic fallback uses system python without torch, so it could not run them before).
3. **CLAUDE.md rules (ENFORCED):** "No silent degradation" (fail-loud/bounded-allowlist on
   missing/misconfigured inputs); "coupled config validated together / tested CLI not monkeypatched
   globals"; "experiment/wrapper scripts need a basis-sanity smoke-assert and the delivered baseline's
   tests run in the pre-push gate".

## Follow-ups (not blocking)
- Add a `--universe/--processed/--price-dir` CLI to the runner so experiments stop monkeypatching
  coupled globals (removes the whole class of mismatch).
- Fix `scripts/run_vn100_ablation.py` to set `_PRICE_DIR` too (the clean vnstock re-run already
  applies both, per the coordinator message).
- Task dashboard: give exploratory experiments a `research` ledger entry noting config + the
  basis-sanity assertion result.
