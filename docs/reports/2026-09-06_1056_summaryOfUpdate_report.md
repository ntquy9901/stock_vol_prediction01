# Summary of update — regime / change-point features (Tier-1 shock probe)

## What changed
Added the `baselines/2026-09-06_regime_features/` baseline (§3.F: 5 sub-folders) to test whether causal
market-regime features improve the QLIKE-champion baseline (HAR-X). This is the last of the shock-detection
Tier-1 tasks; the high-frequency jump estimators (bipower/HARQ) were ruled out earlier because they need
intraday returns, so regime/change-point features were the remaining daily-native lever.

## Files
| Path | Purpose |
|------|---------|
| `baselines/2026-09-06_regime_features/code/regime_features.py` | Causal market-regime features (vol_z, regime, days-since-change) |
| `baselines/2026-09-06_regime_features/code/run_regime_har.py` | HAR-X vs HAR-X+regime walk-forward OLS + DM comparison |
| `baselines/2026-09-06_regime_features/test/test_regime_features.py` | 5 tests incl. prefix-invariance causality |
| `baselines/2026-09-06_regime_features/test/test_run_regime_har.py` | 2 tests (OLS recovery, design shapes) |
| `baselines/2026-09-06_regime_features/{requirements,design,code_review}/*.md` | Spec / plan / review |
| `results/regime_features/regime_{vn30,vn100}_h{1,5,10,22}.json` | Results (8 files) |

## Result (NO-GO)
HAR-X+regime raises QLIKE (worse) on every market×horizon tested (8/8). No Diebold-Mariano test favors the
regime variant; every DM favors plain HAR-X.

| Market | h | HAR-X QLIKE | +regime QLIKE | DM p (regime vs HAR-X) | Favors |
|--------|---|-------------|---------------|------------------------|--------|
| VN30 | 1 | 0.4967 | 0.5086 | 0.214 | HAR-X |
| VN30 | 5 | 0.5929 | 0.5998 | 0.420 | HAR-X |
| VN30 | 10 | 0.6385 | 0.6431 | 0.106 | HAR-X |
| VN30 | 22 | 0.6976 | 0.6978 | 0.972 | HAR-X |
| VN100 | 1 | 0.5006 | 0.5135 | 0.088 | HAR-X |
| VN100 | 5 | 0.5611 | 0.5667 | 0.332 | HAR-X |
| VN100 | 10 | 0.6000 | 0.6049 | 0.243 | HAR-X |
| VN100 | 22 | 0.6386 | 0.6420 | 0.266 | HAR-X |

The three causal regime features add estimation variance without adding forecast signal beyond the HAR lags
that already summarise recent volatility. Integrating them into the LSTM/VolGA input (in_dim 5→8 + full retrain)
is therefore not justified. The shock-detection difficulty on this data is intrinsic (real limit-lock and spike
days), and the robust-QLIKE metric (task #12, already delivered) is the sound way to keep those days from
dominating evaluation, rather than a regime feature that does not forecast them.

## Tests + coverage
- `pytest baselines/2026-09-06_regime_features/test/ -q` → 7 passed.
- Pure logic (`regime_features.py`, `_design`, `ols_fit_predict`) at 100% line+branch; driver `run()`/`main()`
  are `# pragma: no cover` entry glue.

## Code review
Self + adversarial (`code_review/code_review_2026-09-06.md`): causal (prefix-invariance test), hard-isolated
(delivered modules read-only), vectorised OLS (no per-row loop), no config-hardcode (window from
`pc.HAR_MONTHLY_WINDOW`). No critical/major defects. The negative result is a substantive finding, not a bug.

## Data-quality gate
N/A (no data change) — reuses the already-cleaned VN30/VN100 enriched panels (corruption=0); regime features
are derived in-memory from the existing `market_pk` column.

## Risks / follow-ups
- SP500 regime probe not run here (VN markets already give a consistent NO-GO across both panels and all
  horizons; an SP500 run would need Colab and is unlikely to reverse an 8/8 negative). Flagged, not blocking.
- Paper: no regime-feature claim will be added; the honest finding is recorded in this report and can be cited
  as a tried-and-rejected extension if a reviewer asks.

## DoD checklist
- [x] Code matches request (regime probe), no unrelated refactor
- [x] Tests + coverage on changed lines
- [x] Code review done + findings addressed
- [x] Summary report (this file)
- [ ] Push (after commit)
