# Adversarial code review — masked union-date panel (2026-08-22)

Scope reviewed (only these):
- `baselines/2026-08-21_har_anchored_residual/code/masked_snapshots.py`
- `baselines/2026-08-21_har_anchored_residual/code/run_masked.py`
- `baselines/2026-08-21_har_anchored_residual/test/test_masked.py`

Reused read-only (inspected for correctness of the calls, not re-audited): `submission/soict_lstm_gat/{model.py (GATLayer), baselines.py (har_fit/har_predict), metrics.py, data_utils.py (har_features)}` and `code/stats.py (date_clustered_dm)`. Out of scope: `archive/`, the rest of `submission/soict_lstm_gat/`, other baseline files.

Method: 3 layers — Blind Hunter (hidden bugs), Edge Case Hunter (boundaries/masks/degenerate data), Acceptance Auditor (does it satisfy the stated intent: union-of-dates masked panel, TRAIN-only scalers/HAR/edge, purge=h, all-metrics + per-metric date-clustered DM, no leakage). Two findings were verified empirically (test run + a reproduction script on the fixture panel).

## Severity counts
- CRITICAL: 0
- MAJOR: 2
- MINOR: 3
- Verified-correct (no finding): mask-aware GAT attention, leakage boundaries (scaler/HAR/edge), purge, loss/eval target masking against invalid-target cells, per-metric DM key alignment.

## Top 3 findings

### 1. MAJOR — Target/loss mask is `tgt_ok`, not `win_ok & tgt_ok`: cells with an invalid (zero-filled) input window are trained on, HAR-fit on, and scored
- Where: `masked_snapshots.py:118` (`pack`: `nm = win_ok[sl]; tm = tgt_ok[sl]` — `tmask` is `tgt_ok` alone); consumed at `run_masked.py:102` (training loss `* tmb`), `run_masked.py:104-106` (val early-stop MSE), `run_masked.py:167-169` (HAR OLS fit + test scoring), and `_pred_dict` (`run_masked.py:120-126`, test metrics + DM).
- The builder computes `node_ok = win_ok & tgt_ok` (`masked_snapshots.py:66`) and uses it correctly for the anchor `min_valid` filter and the node-drop `train_rows`, but `node_ok` is never exported. The exported `tmask` is `tgt_ok` only. A cell can have `tgt_ok=True` while `win_ok=False` (target `pk[t+h]` exists but the lookback feature window contains a NaN — the ~`lookback+21`-day band right after a ticker's listing on the ragged union panel). For such cells the features/window are `np.nan_to_num`→0 (`masked_snapshots.py:125`, `har_te`/`har_tr` zero at `:128`), yet they enter:
  - the LSTM/GAT training loss (zero input window → real target);
  - the pooled HAR OLS (`har_fit` on all-zero feature rows → intercept-only, biases coefficients);
  - the test metrics and the date-clustered DM.
- Failure scenario (verified on the smoke fixture, staggered listings): train had 4301 scored cells, of which **105 had `win_ok=False`** (no valid input window) and **15 HAR train rows were entirely zero-feature** yet included in the OLS fit. Test contamination was 0 in the fixture (no ticker lists inside the final 10%), but on a real ragged panel any late IPO that lists inside the test window contributes mis-scored cells. The contamination is ~0.3% of train rows here; it is data-dependent and larger the more late-listers a universe has.
- Impact on the reported numbers: bounded and applied to all three models on the same cells, so it biases absolute metric values (and marginally the DM) rather than being pure leakage. Note the GAT variant is the only model that can produce a non-degenerate forecast on a zero-own-input cell (it can attend to valid neighbours), so if anything this slightly favours the graph — it does not manufacture the "graph hurts" conclusion.
- Fix: export `node_ok` (i.e. `win_ok & tgt_ok`) as the target mask used by loss/HAR/scoring, or intersect in `pack` (`tm = (win_ok & tgt_ok)[sl]`). Keep `win_ok` as the separate graph-neighbour mask.

### 2. MAJOR — `np.fill_diagonal` on a read-only `.to_numpy()` result: `build_masked` crashes under pandas 3.0 (the project's documented env)
- Where: `masked_snapshots.py:106-107` — `corr = wide.iloc[:last_tr_row+1].corr(...).to_numpy(); np.fill_diagonal(corr, 0.0)`.
- Under pandas 3.x Copy-on-Write (default), `DataFrame.corr().to_numpy()` returns a read-only array; `np.fill_diagonal` then raises `ValueError: underlying array is read-only`.
- Verified: both tests FAIL in the repository base interpreter (Python 3.14, numpy 2.4.4, **pandas 3.0.3** — the version CLAUDE.md states is installed and "verified"). The same two tests PASS in `.venv_gpu_encode` (Python 3.10, numpy 2.2.6, **pandas 2.3.3**), which is where the existing `results/masked_panel/*` were produced. The already-produced VN30/VN100/SP500 numbers are therefore unaffected, but the code is not reproducible on the documented pandas-3.0 stack and the smoke gate fails there.
- Fix: `corr = np.array(wide.iloc[:last_tr_row+1].corr(min_periods=edge_min_overlap))` (or `.to_numpy().copy()`) before `fill_diagonal`.

### 3. MINOR — Node universe is horizon-dependent, so cross-horizon comparison is not on an identical set of tickers
- Where: `masked_snapshots.py:63` (`anchors` end at `T - horizon`), `:65-66` (`tgt_ok` uses `pk[t+horizon]`), `:71-72` split indices, `:77-78` (`train_rows = node_ok[sl_tr].sum(0)`, `keepn = train_rows >= min_train_rows`).
- Larger `horizon` shortens the anchor array and shifts the train slice, changing each node's `train_rows` and thus which nodes survive `min_train_rows`. Node selection is deterministic given `(files, horizon, min_train_rows)` (no RNG), so the reported VN100 `N=102` (h1/h5) vs an earlier `N=104` is a deterministic consequence of the horizon, not nondeterminism — but different horizons score different ticker universes, so cross-horizon statements ("HAR best at long horizon") mix a horizon effect with a universe effect. Within a horizon all three models share one `MaskedData`, so the model-vs-model comparison at a fixed horizon is fair.
- Fix (if cross-horizon parity is wanted): compute the kept-node set once (e.g. at the longest horizon, or intersect across horizons) and reuse it for all horizons; otherwise document the per-horizon universe difference next to the cross-horizon table.

## Other findings

### 4. MINOR — Model-specific prediction floors before a shared QLIKE floor
- LSTM/GAT predictions are floored per node at `1e-3*t_mean + 1e-12` (`run_masked.py:94`); HAR predictions are floored at `qlike_floor = 1e-8` (`run_masked.py:168` via `har_predict`). The QLIKE metric then clamps both `y` and `p` to a shared `1e-8` (`metrics.py:67-70`), so the *metric basis* is identical (satisfies the recorded H2 rule), but the LSTM/GAT forecasts cannot fall below ~`1e-7` while HAR forecasts can reach `1e-8`. On near-zero-target cells this is a small asymmetry in the QLIKE/SE inputs. Magnitude is tiny relative to typical variance (~`1e-4`); no evidence it affects the ranking. Fix: apply one common floor to all three models' raw predictions.

### 5. MINOR (edge) — Degenerate-data fallbacks
- `masked_snapshots.py:79-80`: if fewer than 2 nodes reach `min_train_rows`, the fallback keeps nodes with `train_rows >= 1`; a 1-row node gets `t_std = nanstd(single) + 1e-8 ≈ 1e-8`, i.e. an essentially constant (mean) forecast. Not a blow-up, but not a real scaler.
- `masked_snapshots.py:105`: if `tr_anchor` is empty (`i_tr - horizon <= 0`, only in pathological tiny panels) `last_tr_row = anchors[i_tr-1]`; with `i_tr==0` this indexes `anchors[-1]`, so the correlation edge would be built over ALL dates (val/test included) — a leakage path, reachable only when there is effectively no training data. Real runs have ample anchors, so this does not affect the produced results. Fix: guard `i_tr==0` / empty-train explicitly and raise.

## Verified correct (no change needed)
- **Mask-aware GAT** (`run_masked.py:82-83` `adj_batch = base * nm.unsqueeze(1)`, feeding `model.GATLayer`): columns of invalid neighbours (`win_ok=False`) are zeroed, so a valid node never attends to a not-yet-listed node; the self-loop `base[i,i]=1` is preserved for valid nodes (`adj_b[b,i,i]=nm[b,i]=1`), so a valid node always has ≥1 finite logit and the `softmax(masked_fill(-inf))` cannot go all-`-inf`; `GATLayer` additionally applies `nan_to_num(alpha)` (`model.py:43`) as a backstop. Zero-filled invalid input nodes cannot inject messages into valid nodes.
- **Leakage boundaries:** per-node target/feature scalers (`masked_snapshots.py:94-102`) use only `node_ok[sl_tr]` train rows; the correlation Top-K edge uses `wide.iloc[:last_tr_row+1]` with `last_tr_row = last-train-target index` (`:105-106`), which is < the first val/test anchor date — no val/test rows; HAR OLS uses `tmask_tr` (train) only. Purge (`sl_tr`/`sl_va` drop the last `horizon` anchors, `:72,:91`) keeps train targets < first val anchor and val targets < first test anchor.
- **Loss/eval exclude invalid-target cells:** training loss and val MSE divide by the valid-target count (`run_masked.py:102`, `:104-106`); `_pred_dict` emits only `tmask`-true cells; zero-filled `y` never enters metrics/DM. (This is correct w.r.t. invalid *targets*; the separate defect in Finding 1 is that "valid" here omits the input-window condition.)
- **Per-metric DM alignment:** `_dm_all` (`run_masked.py:143-158`) keys on `(node, date)`; HAR/LSTM/GAT dicts share the identical `tmask_te` key set, `_ens` intersects keys, and `set(a)&set(b)` guarantees QLIKE/SE/AE are computed on the same aligned observations with the same `y`. `favors="A" if mean_diff<0` matches `metrics.diebold_mariano`'s "negative favours A" convention.
- **Test hygiene:** `test_run_masked_all_metrics_and_dm` monkeypatches `RM.REPO` to `tmp_path` (`test_masked.py:49`), so `results/masked_panel/` is not clobbered; `test_build_masked_union_and_masks` does not call `run`. No in-scope test writes to real `results/`.

## Trust verdict
The masked-panel per-metric results (LSTM beats HAR on MAE at all horizons; graph hurts QLIKE/SE; HAR strongest on squared-error / long horizon) are broadly **trustworthy as qualitative findings**. At a fixed horizon all three models are evaluated on the same node set and the same scored cells, with train-only scalers, train-only HAR coefficients, a train-only correlation edge, and a correct `h`-purge; the mask-aware GAT is implemented correctly and cannot NaN or leak future/neighbour listings. No bug was found that would fabricate the "graph hurts" conclusion — the one mask defect (Finding 1) can only give the graph a small artificial advantage on zero-own-input cells, making the graph-hurts result conservative rather than manufactured.

Two caveats before quoting exact metric values or claiming reproducibility:
1. Finding 1 (mask = `tgt_ok` instead of `win_ok & tgt_ok`) contaminates a small, data-dependent fraction of train cells (verified 105/4301 ≈ 2.4% window-invalid, 15/4301 all-zero HAR rows in the fixture) and possibly a few test cells for universes with late IPOs. It biases absolute numbers slightly, applied identically across models; it is very unlikely to flip the ranking but should be fixed before the numbers are treated as final.
2. Finding 2 means the code only runs under pandas 2.x; the existing results were produced in `.venv_gpu_encode` (pandas 2.3.3) and are valid, but the pipeline must be patched to run on the documented pandas-3.0 environment for reproduction, and the smoke gate currently fails there.

Cross-horizon comparisons additionally mix a horizon effect with a differing-universe effect (Finding 3).
