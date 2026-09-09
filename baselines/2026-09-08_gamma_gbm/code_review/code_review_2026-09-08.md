# Code review — gamma-loss GBM baseline (2026-09-08)

Adversarial 3-lens review (subagent) of `code/gamma_gbm_walkforward.py` + `test/test_gbm.py`, cross-checked
against the delivered `wf_enriched_panel.pack_fold`, `run_masked_rich`, and `stats.date_clustered_dm`.

## Verdict: NO CRITICAL, NO MAJOR. Correctness prerequisites for "GBM beats HAR-X on SP500" all hold.
Verified CLEAN:
- **Causality + anchor alignment:** `extra_feature_panels` uses only backward-looking rolling/shift/diff on
  channel-0 pk; `_design` flattens `har5 [n,N,5]→[n*N,5]` and appends `extras[k][anchors]` in the SAME
  anchor-major/node-minor order; target `y_tr.reshape(-1)` and `tmask_tr.reshape(-1)` share that layout — every
  row `a*N+j` carries HAR + extras + target + mask for the same (anchor, node). No off-by-one vs `pk[t+h]`.
- **Floors:** GBM gamma target floored `>0`; HAR-X and GBM predictions floored by the same per-node `nfloor`;
  QLIKE metric floor shared via `_metrics`/`_dm_all`. HAR-X reproduces canonical.
- **DM orientation:** `_dm_all(GBM, HAR-X)` → favors "A" iff GBM lower loss. Correct.
- **Train/test discipline:** GBM fits only on masked TRAIN rows of each fold, predicts that fold's test; no
  val/test leak; per-fold TRAIN-only scalers; gamma `y>0` handled.

## Findings and resolutions
| # | Sev | Finding | Resolution |
|---|-----|---------|------------|
| MINOR-1 | test gap | Smoke did not prove the 6 extra features actually enter the fitted model | **Fixed**: added `test_extra_features_are_not_a_noop` — on synthetic data where the target depends on an extra column, asserts the GBM-with-extras predictions differ from GBM-on-har5-only AND fit better. |
| MINOR-2 | note | rolling features computed on the union-calendar grid (a ticker with gaps may span <5 own-trading-days) | **Documented** in design.md; strictly causal, negligible on SP500 (single NYSE calendar), irrelevant to the SP500 win. |
| MINOR-3 | framing | HAR-X is OLS (squared loss) while GBM optimises gamma≡QLIKE — part of the edge could be "trains on the metric" | **Resolved with a control (documented)**: a LINEAR gamma model is much WORSE than HAR-X OLS (SP500 h5: 0.6466 vs 0.4870), and extras don't help it (0.6503); only the NONLINEAR GBM wins (+4.63%). So the gain is from nonlinearity + feature interactions, not the objective alone. Disclosed in design.md/requirements. |

## Post-fix verification
- Tests 4/4 pass; **100% line + 100% branch** coverage on `code/`. ruff `--select F` clean.
- No CRITICAL/MAJOR; MINOR-1 fixed, MINOR-2/3 documented (MINOR-3 refuted by the gamma-GLM control).
