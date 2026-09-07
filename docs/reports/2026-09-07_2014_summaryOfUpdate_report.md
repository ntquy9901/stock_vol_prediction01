# Summary of update — h5/h10 failure analysis + regime-blend baseline (2026-09-07 20:14)

## What changed
Investigated why the QLIKE-loss deep models (LSTM/VolGA) lose their edge at h5/h10, and built a baseline
that targets it. Analysis, visual dashboard, a validation spike, and a full baseline with tests + review.

## Files
| path | purpose |
|---|---|
| `docs/reports/2026-09-07_1927_anchor_and_h5h10_failure_analysis.md` | Analysis report (§1 anchor=none provenance, §2 calm-cell mechanism, §3 where/why h5/h10 fails, §4 the h-step lag + literature/citations, §5 spike, §6 baseline results) |
| `docs/reports/2026-09-07_perstock_daily_dashboard.html` | Per-stock daily dashboard: decile damage (SP500+VN), delayed-echo storm episodes, storm-vs-calm context |
| `baselines/2026-09-07_regime_blend/` | New baseline (requirements, design, code, code_review, test) — causal regime-conditional blend of deep + HAR-X |
| `results/qlike_anchor/regime_blend_result.json` | Baseline results (VN100/VN30 × h1/5/10/22 × VolGA/LSTM) |
| `results/qlike_anchor/regime_blend_spike.json` | Prior spike results (binned regime, pooled-val) |

Local-only generators (kept out of git, per the repo's report-generator pattern like
`build_qlikeloss_edge_dashboard.py`; diff-cover would flag untested report scripts):
`build_perstock_daily_dashboard.py`, `scripts/eda/regime_blend_spike.py`. SP500 cells (~3.85 GB) are
gitignored; the dashboard/report rebuild from the committed `decile_summary.json` + result JSONs.

## Key result (VN100 / VN30, deep base VolGA)
| market | h | HAR-X | blend | DM p (blend vs HAR-X) | verdict |
|---|---|---|---|---|---|
| VN100 | 1 | 0.5000 | 0.4857 | 0.000 | beats HAR-X (sig) |
| VN100 | 5/10/22 | 0.5607/0.5999/0.6385 | 0.5589/0.5977/0.6371 | 0.78/0.80/0.93 | ≤ HAR-X, not sig |
| VN30 | 1 | 0.4801 | 0.4698 | 0.001 | beats HAR-X (sig) |
| VN30 | 5/10/22 | 0.5602/0.6091/0.6782 | 0.5556/0.5975/0.6712 | 0.61/0.19/0.79 | ≤ HAR-X, not sig |

Honest verdict: significant daily-horizon (h1) win over HAR-X on both panels (both deep bases); at h5/h10/h22
the blend is ≤ HAR-X but the difference is not significant — the genuine ceiling (at long horizons the deep
signal adds no QLIKE value over HAR-X, so a convex blend cannot significantly beat it). No-harm everywhere.
Target of "significantly beat HAR-X at all four horizons" NOT met; MUST criterion (never worse) met.

## Tests + coverage
- `baselines/2026-09-07_regime_blend/test/test_regime_blend.py`: 12 passed.
- Coverage on `code/`: **100% line, 100% branch** (driver `main()`/`__main__` `# pragma: no cover`).
- ruff `--select F`: clean. Overfit-evidence gate: exit 0 (post-hoc blend classified non-learned).

## Code review
3-lens adversarial review (subagent) — no CRITICAL; causality/leakage clean, QLIKE floor consistent, DM
orientation correct. 3 MAJOR + 4 MINOR triaged in `code_review/code_review_2026-09-07.md`: MAJOR-1 (overfit
gate) refuted by running the gate; MAJOR-2 (empirical assertion in smoke test) and MAJOR-3 (vacuous
independence assert) fixed; MINORs documented/fixed. Post-fix suite green.

## Performance
Post-hoc combination over frozen forecasts; fully vectorised numpy, CPU-only; the only loop is 7 folds × 4
horizons × 2 markets × 2 deep bases of Nelder–Mead on validation arrays (seconds). No batch=1 hot loop, no GPU
needed. N/A for the GPU-batching concerns.

## Data-quality gate
N/A (no data change) — this baseline consumes frozen forecast cells, adds no raw/processed data.

## SonarScan
SonarQube server (`svp_sonar`, sonarqube:community) started and healthy at localhost:9000, but the admin
password is not the default and no `SONAR_TOKEN` is available, so an authenticated scan could not be run in
this session. Follow-up: with a token, `docker run --rm -v <repo>:/usr/src sonarsource/sonar-scanner-cli
-Dsonar.host.url=http://host.docker.internal:9000 -Dsonar.token=<token>` (config in `sonar-project.properties`).

## Follow-ups / risks
- SonarScan pending a token (above).
- The long-horizon tie is a real ceiling; further gains would need a component that genuinely beats HAR-X at
  long h, not a blend of deep+HAR-X.
- Minor: at h1 the val-fit gate shrinks the blend slightly below the pure deep model (still far above HAR-X).
