# Next-Iteration Verification Audit (V1–V10)

Scope: resolve the blocking verification items in
`docs/experement_guide/HAR_GAT_Result_Diagnosis_and_Next_Experiments.md` §4 with file-and-line evidence,
before running new graph models. Verdict codes: PASS / CONFIRMED-ISSUE / DOCUMENTED / NEEDS-ACTION.

## V1 — target is variance, not double-transformed  → PASS
`src/common/parkinson_utils.py:36`: `parkinson = (np.log(high / low) ** 2) / (4 * np.log(2))` — Parkinson
**variance** (σ²), no square root. Stored to the `parkinson_volatility` column and consumed raw:
`data_utils.py:58` (`pk = df["parkinson_volatility"].to_numpy(float)`), `har_features` (`data_utils.py:32-44`)
uses `pk` directly as the daily term and rolling means for weekly/monthly; the target is `pk[t+h]` raw
(`run_lstm.py:78`, `experts.py`). No sqrt or re-square anywhere in the HAR/target/QLIKE path. QLIKE
`y/ŷ+log ŷ` is therefore applied to variance, internally consistent. Semantic risk: the column NAME says
"volatility" but the content is variance — a naming trap, documented; values are variance-scale.

## V2 — signed graph is reduced to a BINARY, UNSIGNED mask  → CONFIRMED-ISSUE (critical)
`submission/soict_lstm_gat/model.py` `GATLayer.forward` (lines 31-45):
```
wh   = self.W(h)...                       # transformed node features
e_src= (wh * self.a_src).sum(-1)          # logit depends ONLY on node features
e_dst= (wh * self.a_dst).sum(-1)
e    = leaky(e_dst[:, :, None] + e_src[:, None, :])
mask = (adjacency != 0)                    # line 40 — adjacency used ONLY as presence mask
e    = e.masked_fill(~mask, -inf)
alpha= softmax(e, dim=2)
out  = einsum("bijh,bjho->biho", alpha, wh)   # message = attention·wh; adjacency value never multiplied
```
The numerical value and sign of `A[i,j]` enter neither the attention logit `e` nor the message. Hence the
effective graph is `A_model = 1(A != 0)`; edges +0.6, −0.6, +0.1 are indistinguishable. The `edges.py`
glasso adjacency is signed/weighted, but the model consumes only its support. Consequence: every "graph"
result in this study (E2 full-target, E6/E7 residual) and the model-free S0 neighbour-mean screen tested a
**binary equal-weight** graph, not the signed/weighted graphical-lasso graph. The prior "graph adds no OOS
value" conclusion is valid ONLY for the binary graph family. This must be fixed (weight/sign consumed)
before any no-signal claim about the signed graph.

## V3 — prediction cutoff / day-t OHLC  → DOCUMENTED
Contract: forecast is produced **at/after the close of day t**; features use the window ending at day t
(`data_utils.make_windows` anchors at t, target `pk[t+h]`). HAR daily = `pk(t)` uses day-t High/Low, which
is known at the close of t, so it is admissible under a post-close cutoff. No feature uses data after t.
Action taken: cutoff stated explicitly here and in the handoff report; feature generation already enforces
window-end = t. If any deployment produced forecasts before the close of t, day-t H/L would leak — not the
case in this pipeline.

## V4 — target-interval purge  → PASS
Terminal target interval is the single day `t+h`. `folds.purged_split` (`folds.py:14-33`) drops the last
`horizon` anchors of train and of val; `experts._purge_snapshots` (`experts.py`) drops the last `horizon`
train/val snapshots. Property `max(train_anchor)+h < min(val_anchor)` (and val/test) is unit-tested
(`test_folds.py::test_purge_no_target_overlap`, 4 horizons). No label interval crosses a split boundary.

## V5 — test not used for selection  → DOCUMENTED (exploratory, not locked-confirmatory)
Early stopping and best-checkpoint use validation QLIKE (`experts.train_neural`); α (E3) and λ (E9) are fit
on validation only (`blend.blend`, `gate.fit_lambda_static`); HAR/GARCH are unselected; the test set is
scored once per config. No hyperparameter reads the test set. HOWEVER: the current results are
**exploratory** — the whole ladder was run and inspected on the same single test split; `min_common` and
the positive floor were chosen from data structure / training, not test loss, but there is no separate
locked confirmatory test period. Action: Phase D must use walk-forward with a held-out locked window for
any confirmatory claim (VN100 E3 h10/h22).

## V6 — DM HAC lag / HLN  → PASS (block-bootstrap to add)
`metrics.diebold_mariano` uses the HLN small-sample correction with HAC lag `h-1` (Newey–West triangular
weights); `stats.date_clustered_dm` aggregates the loss differential to one value per date before applying
it (removes cross-sectional dependence; serial dependence handled by the h-1 HAC). `stats.block_bootstrap_ci`
(circular block, block=ceil(T^{1/3})) is implemented and used in `reports/experiment_results.md`.
NEEDS-ACTION: report the effective number of test dates and the HAC bandwidth per horizon explicitly.

## V7 — cross-fitted HAR residual warm-up  → DOCUMENTED
`har_cv.crossfit_har` / `experts._crossfit_har_pooled` use expanding blocks (`n_folds=5`); block 0 is a
training seed with NaN OOS and is excluded via `cf_mask`. Warm-up = 1/5 of train dates. Residual targets
use the pooled per-horizon HAR (the deployment spec). NEEDS-ACTION: expose a minimum warm-up parameter and
record residual lineage; residual experts train on ~80% of train dates (m-1 asymmetry vs E1/E2, documented).

## V8 — common-date / survivorship  → CONFIRMED-ISSUE (documented, to quantify)
`snapshots._load_panel` keeps the complete-case intersection, dropping latest-listed tickers until
`min_common` common dates remain. This (a) shortens the test window drastically (VN100 ~49–130 test dates;
S&P 500 default gave only 34) and (b) favours long-history stocks (survivorship). The S&P 500 rerun uses
`min_common=3000` = an explicitly-named long-history subset (457 nodes, ~300 test dates), NOT the full
index. NEEDS-ACTION: report total dates before intersection, retained per split, excluded tickers/dates,
and consider the masked-panel union (doc §10).

## V9 — "E2 hurts" is not uniform  → CONFIRMED-ISSUE (reword conclusion)
The graph effect is panel/horizon dependent, not uniformly harmful. Counter-example: VN100 h1 full-target
E2 QLIKE 0.5207 < E1 0.5371 (graph HELPS vs no-graph LSTM there), while VN30 h1 E2 0.4915 > E1 0.4383
(graph hurts). Both still lose to HAR. Action: the handoff/paper conclusion must state "full-target graph
is harmful on VN30 short horizons and mixed on VN100; residual-anchored graph is neutral" per panel, not a
blanket "graph hurts".

## V10 — residual R² 0.039 vs no QLIKE gain  → NEEDS-ACTION
Pooled residual R²_OOS ≈ 0.039 (VN100 h22, E6) with no significant QLIKE improvement suggests the residual
fit explains variance-scale/regime structure that does not translate to QLIKE forecast loss. Action: add
date-level and per-ticker (macro) residual R², calibration slope `r = a + b·r̂`, and
`Var(r̂)/Var(r)` (doc §11) to separate genuine forecast value from scale artefacts.

## Overall leakage verdict
**PASS WITH CONDITIONS.** No target leakage or test-in-selection defect found (V1,V3,V4,V5,V6 clean or
documented). Two substantive issues gate any graph conclusion: **V2 (signed→binary)** — the signed graph
was never actually consumed — and **V8/V9** — panel-dependent effect + survivorship/short-window. The
"no cross-stock signal" claim is therefore premature; it holds only for a binary equal-weight static graph.
Next: signed-graph implementation audit (V2 unit tests) → model-free screening S1–S7 on weighted/signed/
directed graphs → corrected graph model only if screening shows stable signal.
