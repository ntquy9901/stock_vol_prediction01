# Code review — regime_features (2026-09-06)

Scope: `code/regime_features.py` (causal regime features) + `code/run_regime_har.py` (HAR-X vs HAR-X+regime
walk-forward OLS). Layers: correctness, leakage, isolation, performance.

## Findings

### Leakage (checked — clean)
Regime features are causal: `vol_z` uses a trailing window; `regime` compares the trailing mean to its
EXPANDING median (past values only); `dsc` counts forward from a flip. `test_causal_no_lookahead` asserts
prefix-invariance (features for days 0..k unchanged when future days are appended). The regime series is aligned
to fold anchors via `panel.anchors[fold.train/forecast]`, and OLS is fit on TRAIN rows only, so no test
information enters training.

### Isolation (§3.F — clean)
The experiment imports the delivered panel/fold/metrics machinery read-only and reimplements only OLS; it
modifies no other baseline's code.

### Performance (clean)
Regime features are O(T·window) once; OLS uses vectorised `lstsq` on flattened arrays — no per-row Python loop.

### Config-hardcode (clean)
The regime window defaults to `pc.HAR_MONTHLY_WINDOW`; no tunable literal is hardcoded.

### Result integrity (MINOR — noted, not a code defect)
The result is a clean NEGATIVE: HAR-X+regime never lowers QLIKE below HAR-X on VN30/VN100 at any horizon (8/8),
and no Diebold-Mariano test favors the regime variant. Reported honestly; this is a NO-GO for integrating regime
features into the deep model (which would require an in_dim 5->8 retrain).

## Status
Tests 100% line+branch on the pure logic; ruff-F clean. No critical/major defects. Conclusion (NO-GO) is a
substantive finding, not a bug.
