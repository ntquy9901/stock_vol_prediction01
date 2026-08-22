# Why the graph (GAT) branch adds no out-of-sample value — HAR-anchored residual study

Date: 2026-08-22
Scope: `baselines/2026-08-21_har_anchored_residual` (E5 LSTM-residual, E6 GAT-residual, E7 both) over the
`submission/soict_lstm_gat` snapshot design. Panels: VN30 (33 nodes) and VN100 (104 nodes). Read-only
data analysis + small single-seed reproduction runs on GPU (`.venv_gpu_encode`). No data or committed
model files were modified. Builds on `docs/reports/2026-08-21_gat_why_no_help.md` (that report covered
the SOICT full-target model; this one covers the HAR-anchored residual experts and adds a
model-independent spillover test).

## Verdict

**Genuine null, not a bug.** The graph path is implemented correctly (adjacency is a real non-identity
graph, branch toggles work, masking/softmax/broadcast are correct, the graph is frozen train-only and
leakage-safe, and the GAT branch is demonstrably alive — it emits a non-trivial correction at VN100
h22). The graph adds no OOS value because the cross-sectional spillover it would exploit is **not
out-of-sample-transferable on these panels**: a simple linear, leakage-safe neighbor-lag regressor
adds ~0 incremental OOS R² beyond HAR at every horizon on both panels, and the glasso dependence
structure itself does not persist from train to test.

**Single most decisive piece of evidence:** a leakage-safe linear regression of the target on HAR + the
mean Parkinson variance of a node's Top-5 glasso neighbors at day t — which bypasses the GAT, attention,
and residual head entirely — yields incremental OOS R² of essentially zero everywhere (VN30: −0.0001 /
+0.0013 / −0.0014 / −0.0004 at h1/h5/h10/h22; VN100: −0.0007 / +0.009 / −0.001 / +0.0007). If exploitable
cross-sectional spillover existed OOS, this simplest possible test would find it. It does not, so no graph
architecture — regardless of tuning — can recover value that is not in the data.

## Item 1 — Edge structure and OOS stability (STRONGLY supports null)

Glasso Top-5 partial-correlation adjacency rebuilt independently on the TRAIN and TEST date panels
(`_tmp_graph_diag/edge_and_spillover.py`, results in `edge_spillover_results.json`):

| | VN30 train | VN30 test | VN100 train | VN100 test |
|---|---:|---:|---:|---:|
| edges (undirected) | 95 | 98 | 322 | 303 |
| density | 0.180 | 0.186 | 0.060 | 0.057 |
| negative-edge share | 0.032 | 0.194 | 0.022 | 0.155 |
| degree mean (min–max) | 5.8 (5–8) | 5.9 (5–8) | 6.2 (5–10) | 5.8 (5–8) |

Train↔test transfer of the edge set:

| metric | VN30 | VN100 |
|---|---:|---:|
| Top-5 neighbor-set Jaccard (mean / median) | 0.167 / 0.111 | 0.094 / 0.087 |
| fraction of nodes with Jaccard ≥ 0.5 | 0.000 | 0.000 |
| undirected edge-set Jaccard | 0.156 | 0.085 |
| sign agreement on shared edges | 0.923 | 0.939 |
| off-diagonal weight correlation (train vs test) | 0.193 | 0.155 |

The graph is a real, well-formed non-identity structure (density 6–18%, degree 5–8), so the GAT is not
accidentally attending to self-loops only. But the structure barely survives out of sample: no node keeps
half its Top-5 neighbors, edge-set Jaccard is 0.09–0.16, and the negative-edge share flips from ~2–3%
(train) to 15–19% (test) — the sign pattern is unstable too. The frozen train graph the GAT averages over
is close to noise relative to the test-period dependence. Sign agreement on the (small) shared-edge subset
stays high (~0.93), but that subset is only ~10–16% of edges.

## Item 2 — Is there cross-sectional spillover signal at all? (DECISIVE — supports null)

Leakage-safe test, no neural net involved. Neighbors are taken from the TRAIN glasso Top-5 sets; the
feature is `neighbor_lag[j,t] = mean over Top-5 neighbors of pk[·,t]`. Two pooled OLS models are fit on
TRAIN anchors and scored on TEST anchors: HAR (3 features) vs HAR + neighbor_lag (4 features).

| horizon | VN30 incr. OOS R² | VN30 resid-explained R² | VN100 incr. OOS R² | VN100 resid-explained R² |
|---|---:|---:|---:|---:|
| h1 | −0.00012 | 0.0026 | −0.00073 | 0.0001 |
| h5 | +0.00132 | 0.0025 | +0.00905 | 0.0075 |
| h10 | −0.00135 | −0.0013 | −0.00103 | −0.0027 |
| h22 | −0.00037 | 0.0000 | +0.00067 | 0.0009 |

`incr. OOS R²` = (SSE_HAR − SSE_HAR+nbr) / SS_tot on test; `resid-explained R²` = OOS R² of a train-fit
regression of the HAR residual on neighbor_lag. Both are ~0 at every horizon on both panels. The single
non-negligible cell is VN100 h5 (+0.9% incremental R²), which is small and does not correspond to the
horizon where E6's point estimate is largest (VN100 h22). Even the simplest, most generous linear use of
neighbor information — uniform neighbor averaging, exactly what the collapsed GAT computes — carries no
exploitable OOS signal beyond a node's own HAR persistence. This is model-independent: it holds with no
attention, no head capacity, no training dynamics.

## Item 3 — Attention collapse (SUPPORTS: GAT reduces to uniform neighbor averaging)

E6 (GAT-residual) trained single-seed, attention extracted on test snapshots
(`_tmp_graph_diag/train_e6_diag.py`, `train_e6_results.json`). Normalized attention entropy (1.0 = uniform
over neighbors = plain averaging):

| | VN100 h22 | VN30 h1 |
|---|---:|---:|
| entropy mean / median | 0.996 / 0.999 | 0.993 / 0.999 |
| entropy p10 | 0.993 | 0.983 |
| fraction of (node,snapshot) near-uniform (>0.9) | 0.998 | 0.989 |

Attention is inert: `α ≈ 1/degree` everywhere, so the "attention" in Graph Attention Network reduces to an
unweighted mean over each node's Top-5 neighbors. This matches the prior SOICT-model finding and is
expected — the logits are a LeakyReLU of a low-rank projection of only 3 normalized features softmaxed over
~5 neighbors, with no gradient pressure to sharpen because (per Item 2) the averaged quantity has no signal
to reward discrimination. Attention collapse is a symptom, not the root cause: Item 2 shows uniform
neighbor-averaging itself is worthless OOS, so a "fixed" attention would attend to the same signal-free
average.

## Item 4 — Does the GAT branch learn a non-trivial correction? (branch is ALIVE — rules out "dead branch" bug)

Magnitude of the per-node residual correction `c` and its raw-scale reconstruction `add_scale · c` vs HAR:

| variant | params | c_std | corr/HAR magnitude ratio | best val QLIKE |
|---|---:|---:|---:|---:|
| **VN100 h22** E5 (LSTM) | 55,169 | 2e-5 | 0.004 | 0.5366 |
| **VN100 h22** E6 (GAT) | 17,793 | 0.095 | **0.150** | 0.5036 |
| **VN100 h22** E7 (both) | 72,833 | 0.133 | 0.264 | 0.5082 |
| **VN30 h1** E5 (LSTM) | 55,169 | 6e-5 | 0.007 | 0.4117 |
| **VN30 h1** E6 (GAT) | 17,793 | 0.0009 | 0.004 | 0.4113 |
| **VN30 h1** E7 (both) | 72,833 | 0.0013 | 0.004 | 0.4103 |

Two findings:
1. The GAT branch is **not dead/suppressed**: at VN100 h22 E6 emits a correction ~15% of HAR magnitude
   (c_std 0.095), gradients flow, the zero-init head has moved well off zero. So a training/architecture
   defect that silences the graph is ruled out. Yet this correction does not beat HAR under the
   date-clustered paired DM (E6 vs E5 p=0.47, E6 resid R²=0.037 point estimate but not significant per
   `reports/experiment_results.md`). The correction it makes is essentially injecting a common-factor
   (market-level) term via uniform neighbor averaging, which nudges long-horizon VN100 point estimates but
   is within noise.
2. At VN30 h1 both E5 and E6 collapse to ~0 correction (ratio 0.004–0.007) — the zero-init head correctly
   falls back to HAR because the residual is unpredictable, so E5≈E6≈HAR (matching QLIKE ≈ 0.394 in the
   results table). E5 (LSTM-residual) collapses to ~HAR at both configs (c_std 2e-5, 6e-5): the ticker's own
   window carries no residual signal beyond HAR either.

The E5-vs-E6 correction correlation is −0.37 (VN100 h22) / +0.17 (VN30 h1): the two experts are not
duplicating each other, but neither carries robust OOS-exploitable information (Item 2).

## Item 5 — Bug hunt in the graph path (NO defect found)

Inspected `submission/soict_lstm_gat/model.py` (`GATLayer`), `edges.py`, and
`baselines/2026-08-21_har_anchored_residual/code/{models.py,experts.py,snapshots.py}`:

- **Adjacency masking**: `mask = (adjacency != 0)`, `masked_fill(~mask, -inf)`, `softmax(dim=2)` over the
  source axis, `nan_to_num`. Correct. Self-loops (diagonal = 1.0) guarantee every target row has ≥1 valid
  entry, so no all-masked row / no NaN row. The `!= 0` mask correctly keeps signed (negative) partial-corr
  edges (the earlier `>0` bug that dropped negative edges is already fixed, per the code comment).
- **Non-identity adjacency**: confirmed by Item 1 — 95–322 real off-diagonal edges, density 6–18%. The GAT
  sees a genuine graph, not the identity.
- **Node-feature-at-t indexing**: `price[:, :, -1, :]` = last window day = day t. Correct; the GAT reads raw
  HAR features at t (not LSTM output — parallel design, as documented).
- **Adjacency broadcast**: `[N,N] → [B,N,N]` via `unsqueeze(0).expand(b,n,n)`. Correct.
- **Leakage / freezing**: glasso is fit on `snap.adj_pk_train = panel.iloc[a_tr]` (train anchor dates only)
  and frozen for val/test. Snapshots purge the last `horizon` train/val snapshots to remove target overlap.
  Per-ticker scalers fit on train rows. No leakage.
- **`use_graph=False` truly removes the branch**: `models.ResidualNet` only builds `self.gat` and adds
  `hidden*heads` to `feat_dim` when `use_graph`. Confirmed by param counts — E6 (GAT-only) 17,793 <
  E5 (LSTM-only) 55,169 < E7 (both) 72,833; the branches are independently toggled and additive.
- **Residual reconstruction**: `additive_pred = har + add_scale · c`, zero-init head ⇒ exact HAR fallback at
  init. Correct and identical across E5/E6/E7 (same anchor, same floor), so the comparison is apples-to-
  apples.

No masking bug, no indexing/broadcast bug, no leakage, no accidental identity graph, no dead branch.

## Item 6 — Over-smoothing / capacity

Node-representation dispersion (mean pairwise 1−cosine across nodes on test snapshots): GAT output vs its
raw input.

| | raw input | GAT output | ratio |
|---|---:|---:|---:|
| VN100 h22 | 0.725 | 0.529 | 0.73 |
| VN30 h1 | 0.815 | 0.543 | 0.67 |

Uniform neighbor averaging pulls node representations ~30% closer together (partial smoothing, not full
collapse — output dispersion stays well above zero). This washes out part of the strong own-node HAR
signal by mixing in neighbors whose (train-frozen, OOS-unstable) relationship to the node does not hold on
test. Consistent with the plan's over-smoothing hypothesis, but secondary: the primary reason for no value
is the absence of OOS-transferable spillover (Item 2), not smoothing per se.

## Overfitting diagnosis (train vs val vs test)

Requested explicit check: does the graph branch overfit? E5/E6/E7 retrained (3 seeds averaged) with the
trained net's HAR-anchored reconstruction evaluated on TRAIN, VAL, and TEST snapshots
(`_tmp_graph_diag/overfit_diag.py`, `overfit_results.json`). QLIKE is not level-comparable across splits
(different periods/difficulty), so the overfitting signal is the **residual R² train-vs-test gap**
(`R²_tr − R²_te`); a large positive gap = train-specific memorization that fails OOS.

**VN100 h22** (largest graph point-effect; 346 train / 47 test snapshots):

| model | params | QLIKE tr | QLIKE va | QLIKE te | resid R² tr | resid R² te | R² gap (tr−te) | c_std tr | c_std te |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E5 LSTM | 55,169 | 0.657 | 0.536 | 0.617 | 0.0002 | 0.0005 | −0.0002 | 7e-5 | 6e-5 |
| E6 GAT | 17,793 | 0.660 | 0.505 | 0.592 | 0.0017 | **0.0389** | **−0.037** | 0.119 | 0.082 |
| E7 both | 72,833 | 0.662 | 0.508 | 0.597 | −0.0020 | 0.0345 | −0.036 | 0.123 | 0.084 |

**VN30 h1** (low-signal control; 1049 train / 132 test):

| model | params | QLIKE tr | QLIKE va | QLIKE te | resid R² tr | resid R² te | R² gap (tr−te) | c_std tr | c_std te |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E5 LSTM | 55,169 | 0.469 | 0.412 | 0.395 | −0.0001 | −0.0001 | +0.00002 | 1e-4 | 1e-4 |
| E6 GAT | 17,793 | 0.469 | 0.411 | 0.395 | 0.0001 | −0.0001 | +0.0002 | 0.0016 | 0.0013 |
| E7 both | 72,833 | 0.469 | 0.411 | 0.395 | 0.0002 | 0.00003 | +0.0002 | 0.0021 | 0.0019 |

Findings:
1. **The graph residual does NOT overfit.** For E6 the residual R² gap is *negative* on VN100 h22
   (train 0.0017 vs test 0.039) — test R² exceeds train R² — and ~0 on VN30 h1. There is no "fits train,
   fails OOS" pattern. On QLIKE too, E6 fits TRAIN essentially the same as E5 (0.660 vs 0.657, marginally
   *worse*) yet does *better* on val/test (0.505/0.592 vs 0.536/0.617). This differs from the SOICT
   full-target model (`2026-08-21_gat_why_no_help.md`), which did overfit — the HAR-anchored zero-init
   residual anchor constrains the head to a small correction and removes the overfitting failure mode.
2. **There is no train-side edge advantage to begin with.** E6's TRAIN residual R² is only 0.0017
   (VN100 h22) / 0.0001 (VN30 h1) — barely above E5's ~0. The GAT's +17k-vs-55k parameters do not buy a
   better train fit that then fails OOS; they buy essentially nothing on train. So the null is not
   overfitting.
3. **It is also not attention-collapse hiding a real train signal.** If a learnable train-side
   cross-sectional structure existed, the collapsed (uniform-averaging) GAT would still capture part of it
   as a train R² well above zero. It does not (train R² ≈ 0.002). Attention collapses onto an
   average that carries almost no signal even in-sample.
4. **The small VN100 h22 test effect (R²≈0.039) is a non-transferable common-factor coincidence, not
   learned structure.** Test R² > train R², attention is uniform (Item 3), the underlying edge set does not
   persist (Item 1 Jaccard 0.09), and the linear neighbor-average (Item 2) is ~0 — so the effect is a
   long-horizon market-level alignment on the test window, and it is not significant (E6 vs E5 date-
   clustered DM p=0.47).

**Overfitting verdict: (c) genuine no-signal.** Not (a) overfitting — there is no positive train→test
residual-R² gap (it is zero or negative), and the graph's extra capacity buys no train advantage. Not (b)
pure underfitting masking a real train signal — the train residual R² is itself ~0, so there is no learnable
in-sample cross-sectional structure being missed. The three-way split shows the graph neither memorizes
train nor discovers transferable structure, because the transferable cross-sectional signal is absent from
the data (Items 1–2).

## Distinguishing the two explanations

- **(a) graph path correct, data has no OOS-transferable spillover** — SUPPORTED by all four independent
  lines: the model-independent linear spillover test is ~0 everywhere (Item 2); the glasso structure does
  not persist train→test (Item 1); the GAT branch is provably functional and still fails to beat HAR
  (Item 4); the code path is correct (Item 5).
- **(b) training/architecture defect suppresses the graph** — NOT SUPPORTED: no code defect found (Item 5),
  and the branch is not suppressed — it emits a 15%-of-HAR correction at VN100 h22 (Item 4). Attention
  collapse (Item 3) is real but is a downstream symptom of a signal-free averaging target, not a defect that
  silences the graph; even ideal (linear, uncollapsed) use of the neighbor average adds nothing (Item 2).

## Bottom line

For per-ticker Parkinson-variance forecasting on these VN30/VN100 panels, the cross-sectional graph adds no
OOS value because the exploitable signal is not there out of sample, not because of a bug. The target is
dominated by each ticker's own volatility persistence (already captured by HAR / the LSTM branch); the
residual cross-sectional dependence the glasso graph encodes does not transfer across the train/test
boundary, and even a leakage-safe linear neighbor-average regressor confirms ~0 incremental OOS R². The GAT
result belongs in the paper as a clean negative ablation, with this mechanistic evidence.

## Reproduction

```
PYTHONIOENCODING=utf-8 .venv_gpu_encode/Scripts/python.exe _tmp_graph_diag/edge_and_spillover.py   # items 1,2 (CPU)
PYTHONIOENCODING=utf-8 .venv_gpu_encode/Scripts/python.exe _tmp_graph_diag/train_e6_diag.py         # items 3,4,6 (GPU, 1 seed)
PYTHONIOENCODING=utf-8 .venv_gpu_encode/Scripts/python.exe _tmp_graph_diag/overfit_diag.py           # overfitting (GPU, 3 seeds)
```
Raw outputs: `_tmp_graph_diag/edge_spillover_results.json`, `_tmp_graph_diag/train_e6_results.json`,
`_tmp_graph_diag/overfit_results.json`.
```
