# Code review — masked_rich "fairest graph test" (2-hop weighted-GAT)

Date: 2026-08-22
Scope reviewed (only):
- `baselines/2026-08-21_har_anchored_residual/code/masked_rich.py`
- `baselines/2026-08-21_har_anchored_residual/code/run_masked_rich.py`
- `baselines/2026-08-21_har_anchored_residual/code/test_masked_rich.py` (8 tests)

Reused read-only (not reviewed for defects, only checked for consistent use): `submission/soict_lstm_gat/{metrics.py, baselines.py, config.py, data_utils.py}`, `code/masked_snapshots.py`, `code/stats.py`.

Verification method: static read of the four in-scope concerns (leakage, layer correctness, fairness, metric integrity) plus the CPU-fast unit suite. GPU training not run.

`python -m pytest .../code/test_masked_rich.py -q` -> **8 passed** (incl. end-to-end train on the ragged real VN30 slice returning finite, strictly-positive predictions for both no-graph and weighted-GAT paths).

## Severity counts
- Critical: 0
- High: 1 (interpretation / claim risk, not a code defect)
- Medium: 3
- Low: 2
- Positive confirmations: 4

---

## Findings (most severe first)

### H1 (HIGH — interpretation, not a code bug) "Graph beats HAR" conflates a feature-set win with a graph win
`run_masked_rich.run` compares HAR (3 HAR columns only, `masked_rich.py:46`, `run_masked_rich.py:222-224`) against LSTM / GAT that see **5** node features (`masked_rich.py:32,137-141`). The docstring states this explicitly (`masked_rich.py:15-17`). Therefore a "graph beats HAR on MSE/RMSE/MAE/R2" headline is the sum of three effects — (a) two extra features `market_pk` + `volume_zscore_20`, (b) LSTM nonlinearity, (c) the graph — and does **not** isolate the graph.

Failure scenario: reporting the vn30/vn100 h1 and vn100 h5 point-error gains as evidence "the graph helps" when they are largely a 5-feature-vs-3-feature win, reversing every prior project conclusion (graph adds no OOS value) on a confound.

Mitigating fact (why this is not also a Critical): the code DOES build the correct control — a same-5-feature **no-graph** LSTM (`use_graph=False`, `run_masked_rich.py:225`) — and computes the graph's marginal value directly as `wGAT_vol2pk_vs_LSTM` (and `wGAT_corr_vs_LSTM`) via date-clustered DM (`run_masked_rich.py:234,237`). `_ens`/`_dm_all` intersect keys (`run_masked_rich.py:185-190,200`) so graph-vs-LSTM is on identical observations. The graph's contribution must be read from **wGAT_vs_LSTM**, not wGAT_vs_HAR.

Fix (reporting, not code): headline the graph's value on `wGAT_vol2pk_vs_LSTM` (5-feat vs 5-feat) with the DM p-value; present wGAT_vs_HAR and LSTM_vs_HAR only as the (feature+model) context rows. If wGAT_vs_LSTM is null, state the point-error gain is a feature win, not a graph win.

### M1 (MEDIUM) Prediction-floor asymmetry between HAR and the deep models
HAR predictions are floored at `cfg.qlike_floor = 1e-8` (`run_masked_rich.py:223`), while LSTM/GAT predictions are floored per-node at `1e-3 * D.t_mean + 1e-12` (`run_masked_rich.py:151`). The two model families therefore operate on different effective supports: the deep models can never emit a small prediction, HAR can go near-zero.

The **metric** floor is consistent — `_metrics` and `_dm_all` re-clamp both y and p to `1e-8` for every model (`metrics.per_obs_qlike`, `run_masked_rich.py:195-196,203`) — so this is not a "different floor in the metric" bug. But because the deep floor (`1e-3*t_mean`) far exceeds `1e-8`, the deep models structurally over-predict on calm days, which QLIKE (asymmetric, penalises over-prediction of low-vol obs) punishes heavily while MSE/MAE (dominated by high-vol obs) rewards. This asymmetry is a *partial mechanical driver* of the observed "point-error favours deep, QLIKE favours HAR" split, i.e. the +270% QLIKE at h1 is a genuine over-prediction penalty on the deep models, not a tiny-denominator/floor artifact.

Failure scenario: attributing the point-error/QLIKE divergence to model quality when part of it is the differing prediction floor.
Fix: apply one common prediction floor to all models (either floor HAR at `1e-3*t_mean` too, or floor the deep models at `1e-8`) so the QLIKE/point-error trade-off reflects the models, not the flooring convention. At minimum, document the asymmetry next to the QLIKE numbers.

### M2 (MEDIUM) Silent zero fallback in `volume_zscore_20` — recurrence of a documented bug class
`_volume_zscore_wide` returns `0.0` (neutral shock) for any ticker whose `{tk}_ohlcv.csv` is missing or whose window is flat/zero-volume (`masked_rich.py:87-89`). This is a silent neutral fallback of exactly the kind CLAUDE.md §"No silent degradation" forbids, and matches the prior incident (project memory: `volume_zscore` silently zeroed for 71/104 tickers due to a wrong price dir). If a price-dir/filename mismatch occurs, one of the two "extra" features silently collapses to all-zero for the affected tickers, quietly weakening the very feature advantage H1 is about — with no gate to catch it.

Verified NOT active for the reported runs: vn30 has 33/33 `_ohlcv.csv` matches, vn100 has 104 processed (`data/vn100/`) with 104 raw matches (`data/raw/prices/vn100_vnstock/`). sp500 has 0 processed files under `submission/.../data/sp500` so it does not run.

Failure scenario: running sp500 (or after a dir rename) with volume silently 0 for many/all tickers; the 5-feature model degrades to 4 features undetected.
Fix: raise (or bounded-allowlist a small count) when `{tk}_ohlcv.csv` is absent for a ticker being scored, mirroring the `augment_split_frames._check_price_coverage` precedent; log a coverage summary.

### M3 (MEDIUM) Possible NaN gradient from a fully-masked attention row
`adj_batch` zeroes source columns of invalid neighbours: `base.unsqueeze(0) * nm.unsqueeze(1)` (`run_masked_rich.py:141`). A **valid** target keeps its self-loop (`base[i,i]*nm[b,i]=1`), so it is never fully masked. An **invalid** target has its own self column zeroed, so its adjacency row can be all-zero; `WeightedGATLayer` then `masked_fill(-inf)` the whole row and `softmax` yields NaN, patched forward by `nan_to_num` (`run_masked_rich.py:70-73`). The forward is finite, but `softmax` backward with saved NaN output and zero upstream grad can produce `0 * NaN = NaN` gradients into the shared `W`, which would corrupt all nodes.

Evidence it is not currently corrupting: the invalid-row grad path is otherwise fully isolated (its loss weight `tmb=0` and its column is zeroed at both hops, so no valid node receives its message), and the real-data smoke `test_train_masked_rich_smoke` trains on the ragged VN30 union (invalid nodes present) and asserts finite, positive predictions — which passes. The NaN only materialises if an *invalid* node's row is fully -inf AND the backward actually propagates the NaN; empirically it has not on VN30/VN100.

Failure scenario: a ragged panel (e.g. sp500, or a VN config) where the backward NaN does propagate silently NaNs a seed's training; `_ens` mean would then be NaN and surface as NaN metrics (loud, not silent-wrong).
Fix (robustness): guarantee every attention row has at least one finite entry before softmax (e.g. force `A[i,i]=1` for all i in `adj_batch`, or replace a fully-masked row with a uniform self-only row), so invalid nodes never hit the all-`-inf` path.

### L1 (LOW) One-seed point-error signal is fragile
The reported win is single-seed (`--single` -> `seeds=(42,)`, `run_masked_rich.py:269-270`), the standout delta is one metric on one dataset (vn100 h5 MAE -5.1%), R2 is pooled across heterogeneous nodes (`metrics.r2` denominator dominated by a few high-vol obs), and the direction disagrees with QLIKE. `torch.backends.cudnn.benchmark = True` (`run_masked_rich.py:24`) also allows minor run-to-run nondeterminism on GPU. These are the hallmarks of a small/mixed signal near noise.
Fix: judge on the 5-seed date-clustered DM p-values (already implemented) rather than single-seed percentage deltas.

### L2 (LOW) `market_pk` includes the node's own contemporaneous value
`market_pk = np.nanmedian(np.sqrt(pk), axis=1)` (`masked_rich.py:135`) is the cross-sectional median at day t over all valid nodes, including node j itself, and is shared as feature 4 across every node. This is causal (day-t only, inside the input window) so it is not leakage, but the "market" factor is mildly self-inclusive.
Fix (optional): leave-one-out median per node, or accept as-is (effect is negligible for N>=30).

---

## Positive confirmations (checked, correct)

- **P1 No temporal leakage in the three new ingredients.** `market_pk` uses only day-t cross-section (`masked_rich.py:135`); `volume_zscore_20` is a trailing rolling z-score (`masked_rich.py:84-87`, causal); the directed vol->PK edge and the correlation edge are estimated on TRAIN rows only up to `last_tr_row = tr_anchor[-1] + horizon` and frozen (`masked_rich.py:185-198`), with `src=v[:-1]`, `tgt=p[1:]` so the t+1 target side stays inside train. Per-node target and 5-dim feature scalers are fit on TRAIN valid rows only (`masked_rich.py:176-183`). Purge = horizon is respected (`sl_tr = slice(0, i_tr - horizon)`, `masked_rich.py:156`).
- **P2 WeightedGATLayer consumes edge weight AND sign, at both hops, mask-aware.** Weight/sign enter both the attention logit (`a_term = A * edge_bias`, `run_masked_rich.py:68-69`) and the message (`msg = alpha * A`, `run_masked_rich.py:75-76`); the 2-hop stack is `gat1: 5->256`, `gat2: 256->256` with ELU inside each layer, matching the deliverable's `gat_layers=2` (`run_masked_rich.py:96-106`). Confirmed by 4 passing tests: `test_wgat_consumes_edge_sign`, `test_wgat_consumes_edge_weight`, `test_two_hop_net_consumes_edge_sign_and_weight`, `test_wgat_mask_aware` + `test_two_hop_mask_aware`.
- **P3 Metric integrity / F1-F2 preserved.** Metrics and DM are computed only on `target_mask = win_ok & tgt_ok` cells (`masked_rich.py:204`, `_pred_dict` gate `run_masked_rich.py:180`); the QLIKE floor is identical (`1e-8`) across all models at metric and DM time (`run_masked_rich.py:195-196,203`); the F1/F2 masking fixes from `masked_snapshots` are replicated, not reintroduced.
- **P4 Graph-vs-LSTM computed on identical observations.** `_ens` intersects per-seed keys, `_dm_all` intersects the two models' keys, and all models share `tmask_te`, so `wGAT_vs_LSTM`, `wGAT_vs_HAR`, `LSTM_vs_HAR` are all on the same obs (`run_masked_rich.py:185-214,225-237`).

---

## Verdict

**Is the code correct enough for a full 5-seed run?** Yes. Leakage checks pass, the weighted-GAT is correct (weight+sign consumed at both hops, mask-aware), metrics/DM use a shared floor on masked cells, and the essential control (same-5-feature no-graph LSTM) is present. The 8-test suite passes including a ragged real-data train. The three MEDIUMs are robustness/interpretation issues, not correctness blockers; M2 and M3 would only bite sp500 / a renamed price dir and would surface loudly (NaN metrics or missing files), not silently corrupt the VN numbers.

**Is the 1-seed short-horizon "graph beats HAR" point-error signal real or an artifact?** Most likely **not a graph win**. It is (i) primarily a *feature win* — 5 features (adding `market_pk` + `volume_zscore_20`) vs HAR's 3 — which the graph-vs-HAR headline cannot separate; the graph's own marginal value must be read from `wGAT_vol2pk_vs_LSTM`, which the code computes. It is (ii) partly a *metric-geometry effect* — the deep models' higher prediction floor makes MSE/MAE look good while QLIKE (+270%) reflects genuine calm-day over-prediction, i.e. the point-error/QLIKE split is coherent, not a metric bug. And it is (iii) *single-seed and small* (one MAE delta, pooled R2).

**Go/no-go:** GO on the 5-seed run (cheap insurance, code is sound), conditioned on the reporting discipline:
1. Center the graph claim on **wGAT_vs_LSTM** (5-feat vs 5-feat) with 5-seed date-clustered DM p-values — not wGAT_vs_HAR percentage deltas.
2. Apply/state a **common prediction floor** (M1) so the QLIKE-vs-point-error trade-off is not partly a flooring artifact.
3. Before running sp500, fix the **silent volume-zero fallback** (M2) and add the fully-masked-row guard (M3).
Expected outcome if the graph-vs-LSTM DM is null across seeds: the point-error gain is a feature win, consistent with the project's standing finding that the graph adds no out-of-sample value.
