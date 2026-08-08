# 40-Epoch Convergence Check — Headline Comparison and Three Ablations

Date: 2026-08-06 (runs executed 2026-08-06 → 2026-08-08)
Scope: 4 model variants × 3 seeds (42, 123, 2026), extended from 20 to 40 epochs.

## 1. Motivation

The SOICT-2026 draft (`docs/paper/soict2026_draft_v3.tex`) reports a headline comparison and
three ablations, each trained for 20 epochs on 3 seeds. Post-hoc inspection of the saved
val-loss bookkeeping showed several seeds whose minimum validation loss occurred **at epoch 20**
(the training cap), i.e. validation loss was still falling when training stopped:

- Price-only backbone (`train_parallel_enhanced.py`): seed42 best@17, seed123 best@20, seed2026 best@19.
- Full / per-ticker gate (`train_per_ticker_gate.py`): seed42 best@10, seed123 best@20, seed2026 best@20.
- No-graph ablation (`run_no_graph_ablation.py`): seed42 best@19, seed123 best@15, seed2026 best@14.
- No-gate ablation (`train_dual_news.py`): best_epoch not recorded (loss history not saved in the
  original run).

This check extends every seed of every variant by 20 more epochs (to 40 total) and asks whether
the reported metrics — and the paper's significance conclusions — are stable with respect to the
training budget.

## 2. Method

### 2.1 Resume mechanism
A resume path was added to the three scripts that lacked one
(`train_parallel_enhanced.py`, `run_no_graph_ablation.py`, `train_dual_news.py`), matching the
existing `--resume_checkpoint` pattern in `train_per_ticker_gate.py`: load the prior best
checkpoint, continue the epoch axis (21→40), and seed the running best from the prior run so the
extension never regresses below the checkpoint's already-found best. The `train_per_ticker_gate.py`
per-invocation cap is 10 epochs, so the full model reached 40 via two chained +10 resumes
(20→30→40), reusing the same 10+10 chaining that produced its original 20-epoch numbers. The +20
extension (cumulative 40) was user-authorized for this check.

Resume was verified live: each variant's first resumed epoch continued the numbering (epoch 21,
not 1) and reproduced a validation loss adjacent to the epoch-20 checkpoint (e.g. backbone seed42
epoch 21 val = 0.9208 vs prior best 0.9103; dual-news seed on resume re-measured the loaded
checkpoint's val loss to seed the running best).

### 2.2 Checkpoint convention (important caveat)
The four variants do **not** share a single "which checkpoint is reported" rule; this asymmetry
already existed in the paper's 20-epoch numbers and is preserved here:

- **backbone, no-graph, no-gate**: single continuous resume; the reported checkpoint is the
  **global best over epochs 1–40** (running best seeded from the 20-epoch best).
- **full / per-ticker gate**: built by 10-epoch chained resumes, and `train_per_ticker_gate.py`
  re-initialises its running best to +∞ at each resume. The reported checkpoint is therefore the
  **best over the final 10-epoch window** (epochs 31–40) — exactly analogous to how the paper's
  20-epoch full number was the best over epochs 11–20, not 1–20.

Because this convention matters for one conclusion, Section 5 reports the full model under **both**
its native windowed convention and a consistent global-best-over-1–40 convention.

### 2.3 Aggregation
Per-variant mean ± std over the 3 seeds for the six mandatory metrics; paired `t`-tests across the
3 seeds (n=3, df=2; two-sided 5% critical value |t| > 4.303). Reproduced by
`scripts/convergence_check/aggregate_convergence.py` on `mapping_old_20ep.json` and
`mapping_new_40ep.json`. The aggregator reproduces the paper's published 20-epoch macros exactly
(e.g. backbone QLIKE 0.4603, news QLIKE 0.4430; Ablation-1 QLIKE t=6.22, RMSE t=9.38), confirming
the run identification.

## 3. Convergence: best_epoch per seed (20-epoch → 40-epoch)

| Variant   | seed42 | seed123 | seed2026 | Diagnosis at 40 |
|-----------|--------|---------|----------|-----------------|
| backbone  | 17 → 37 | 20 → 40 | 19 → 39 | still at/near cap — **not converged** |
| full      | 10 → 40 | 20 → 20 | 20 → 30 | flat/noisy val surface; global min scattered 20–40 |
| no-graph  | 19 → 39 | 15 → 40 | 14 → 34 | still at/near cap — **not converged** |
| no-gate   | —  → 34 | —  → 37 | —  → 40 | still at/near cap — **not converged** |

(Full best_epoch over 1–40 read from `loss_history.json`: seed42=40, seed123=20, seed2026=30.)

**Finding.** Extending to 40 epochs did **not** produce clean convergence. For backbone, no-graph
and no-gate, the best validation epoch simply moved to the new cap (34–40) instead of settling
earlier — the deep models keep improving slowly and are still not converged at 40. For the full
model the validation surface is flat and noisy (≈0.88–0.93 with occasional dips): its original
"best@20" was a single noise dip that later epochs did not consistently beat (seed123's global
minimum is still epoch 20), so "best@20" reflected noise, not a genuine climbing trend.

The direct answer to the check's question: **20 epochs was well short of convergence, and 40
epochs is still not clearly converged.** Every deep-model comparison here remains budget-sensitive.

## 4. Six-metric comparison (mean ± std, n=3): 20-epoch vs 40-epoch

Full model shown under its native windowed (best-31–40) convention; see Section 5 for the
global-best alternative.

### QLIKE (lower better)
| Variant  | 20-epoch | 40-epoch |
|----------|----------|----------|
| backbone | 0.4603 ± 0.0205 | 0.4597 ± 0.0144 |
| full     | 0.4430 ± 0.0185 | 0.4392 ± 0.0060 |
| no-graph | 0.4657 ± 0.0112 | 0.4376 ± 0.0065 |
| no-gate  | 0.4366 ± 0.0116 | 0.4284 ± 0.0013 |

### RMSE (lower better)
| Variant  | 20-epoch | 40-epoch |
|----------|----------|----------|
| backbone | 0.002923 ± 0.000090 | 0.002916 ± 0.000063 |
| full     | 0.002734 ± 0.000096 | 0.002739 ± 0.000067 |
| no-graph | 0.002788 ± 0.000058 | 0.002622 ± 0.000027 |
| no-gate  | 0.002723 ± 0.000074 | 0.002706 ± 0.000033 |

### MAE (lower better)
| Variant  | 20-epoch | 40-epoch |
|----------|----------|----------|
| backbone | 0.000811 ± 0.000014 | 0.000809 ± 0.000009 |
| full     | 0.000793 ± 0.000012 | 0.000794 ± 0.000004 |
| no-graph | 0.000788 ± 0.000011 | 0.000760 ± 0.000002 |
| no-gate  | 0.000787 ± 0.000010 | 0.000786 ± 0.000002 |

### R² (higher better)
| Variant  | 20-epoch | 40-epoch |
|----------|----------|----------|
| backbone | 0.7749 ± 0.0140 | 0.7762 ± 0.0098 |
| full     | 0.8031 ± 0.0139 | 0.8024 ± 0.0096 |
| no-graph | 0.7953 ± 0.0085 | 0.8191 ± 0.0038 |
| no-gate  | 0.8047 ± 0.0106 | 0.8073 ± 0.0047 |

### Directional accuracy (%, higher better)
| Variant  | 20-epoch | 40-epoch |
|----------|----------|----------|
| backbone | 48.47 ± 0.35 | 48.97 ± 0.58 |
| full     | 47.77 ± 0.52 | 48.42 ± 0.26 |
| no-graph | 48.29 ± 0.04 | 48.42 ± 0.23 |
| no-gate  | 48.22 ± 0.27 | 48.27 ± 0.31 |

### MSE (lower better)
| Variant  | 20-epoch | 40-epoch |
|----------|----------|----------|
| backbone | 8.55e-6 | 8.51e-6 |
| full     | 7.47e-6 | 7.51e-6 |
| no-graph | 7.78e-6 | 6.87e-6 |
| no-gate  | 7.42e-6 | 7.33e-6 |

**Observation.** The no-graph (identity-adjacency) model improved the most with the extra 20 epochs
(RMSE 0.002788 → 0.002622, R² 0.7953 → 0.8191), while the backbone (k-NN graph) barely moved
(RMSE 0.002923 → 0.002916). The graph-free model pulls ahead of the graph model as training
continues.

## 5. Paired t-tests (n=3, df=2; |t| > 4.303 significant at 5%)

### Ablation 1 — news branch (backbone − full)
| Metric | 20-epoch t | 40-epoch t (full windowed) | 40-epoch t (full global-best) |
|--------|-----------|-----------------------------|-------------------------------|
| MSE    | 9.67 (SIG)  | 3.15 (ns) | 10.48 (SIG) |
| RMSE   | 9.38 (SIG)  | 3.17 (ns) | 11.62 (SIG) |
| MAE    | 3.79 (ns)   | 2.73 (ns) | 4.72 (SIG)  |
| R²     | −9.67 (SIG) | −3.15 (ns)| −10.48 (SIG)|
| QLIKE  | 6.22 (SIG)  | 2.14 (ns) | 4.06 (ns, borderline) |
| DirAcc | 2.97 (ns)   | 2.17 (ns) | 2.27 (ns)   |

### Ablation 2 — cross-stock graph (no-graph − backbone)
| Metric | 20-epoch t | 40-epoch t |
|--------|-----------|-----------|
| MSE    | −1.63 (ns) | −5.72 (SIG) |
| RMSE   | −1.64 (ns) | −5.87 (SIG) |
| MAE    | −1.89 (ns) | −8.30 (SIG) |
| R²     | 1.63 (ns)  | 5.72 (SIG)  |
| QLIKE  | 0.30 (ns)  | −2.01 (ns)  |
| DirAcc | −0.98 (ns) | −1.20 (ns)  |

### Ablation 3 — per-ticker gate (full − no-gate)
| Metric | 20-epoch t | 40-epoch t (full windowed) | 40-epoch t (full global-best) |
|--------|-----------|-----------------------------|-------------------------------|
| MSE    | 0.39 (ns) | 1.22 (ns) | −0.47 (ns) |
| RMSE   | 0.37 (ns) | 1.23 (ns) | −0.47 (ns) |
| MAE    | 1.01 (ns) | 3.15 (ns) | 0.87 (ns)  |
| R²     | −0.39 (ns)| −1.22 (ns)| 0.47 (ns)  |
| QLIKE  | 1.24 (ns) | 3.19 (ns) | 2.52 (ns)  |
| DirAcc | −2.28 (ns)| 0.87 (ns) | −0.51 (ns) |

## 6. Do the paper's conclusions hold at 40 epochs?

### Ablation 3 (gate adds no measurable value): HOLDS, robustly.
No metric passes the 5% threshold at 40 epochs under either full-model convention (max |t| = 3.19,
QLIKE, windowed). Same conclusion as the paper.

### Ablation 1 (only the news branch earns a significant gain): PARTIALLY holds / FRAGILE.
- The **direction** is unchanged on all six metrics at 40 epochs: removing news raises error and
  QLIKE and lowers R² on the seed means.
- The **magnitude of significance is convention-dependent**:
  - Under the full model's native windowed (best-31–40) convention — the same convention the
    paper's 20-epoch full number used — **no metric passes at 40 epochs** (RMSE t drops 9.38 → 3.17,
    QLIKE 6.22 → 2.14).
  - Under a consistent global-best-over-1–40 convention, **RMSE/MSE/MAE/R² remain significant**
    (RMSE t = 11.62), but **QLIKE drops to t = 4.06, just below the 4.303 threshold**.
- Root cause of the drop: the full model's validation surface is flat and noisy, and its windowed
  convention discards a good epoch-20/epoch-30 checkpoint for seeds 123/2026 in favour of a noisier
  best-over-31–40 one; this both worsens the full model's 40-epoch metrics and inflates the
  cross-seed variance that the n=3 paired `t` is very sensitive to.
- Net: the paper's specific published statistic **"QLIKE t = −6.22"** does **not** reproduce at 40
  epochs (it becomes 2.14 windowed / 4.06 global-best). The RMSE news benefit is the robust part
  (still highly significant under the consistent convention); the QLIKE significance is not robust.

### Ablation 2 (cross-stock graph changes nothing): DIRECTION/SIGNIFICANCE CHANGED.
At 20 epochs no metric was significant ("the graph moves no metric past the threshold"). At 40
epochs, **MSE/RMSE/MAE/R² are all significant in favour of REMOVING the graph** (identity beats
k-NN; RMSE t = −5.87, MAE t = −8.30). The graph is therefore not neutral at 40 epochs — it is mildly
harmful on error metrics, and removing it significantly improves them. This still supports the
paper's larger thesis that the graph does not help, but the paper's specific wording ("no metric
passes the threshold" / "adds no measurable value") is **no longer accurate** — several metrics pass,
in the direction of the graph hurting.

## 7. Bottom line

1. **The models are not converged at 20 epochs, and still not clearly converged at 40** — best
   validation epochs sit at 34–40 for most seeds. All deep-model comparisons here are
   budget-sensitive.
2. **Ablation 3 (gate) holds** unchanged.
3. **Ablation 1 (news) is fragile**: direction holds; the headline QLIKE significance (t=−6.22)
   does not reproduce at 40 epochs; the RMSE/MSE/R² news benefit is robust only under a
   consistent best-over-1–40 checkpoint convention.
4. **Ablation 2 (graph) materially changed**: from "no measurable effect" to "significantly harmful
   on RMSE/MAE/R²/MSE (removing the graph helps)".

These are **material changes** to two of the paper's three ablation conclusions (Ablations 1 and 2).
Per the task's instruction, `docs/paper/*.tex` was **not** modified. A separate integration pass
should decide how to fold this in — options include reporting results at a fixed, pre-registered
budget with an explicit convergence caveat, unifying the full model's checkpoint-selection
convention with the other variants, and re-checking the news-branch significance and the graph
claim against these 40-epoch numbers.

## 8. Artifacts

- Resume code: `src/lstm_gat_hybrid/train_parallel_enhanced.py`,
  `scripts/ablation_no_graph/run_no_graph_ablation.py`,
  `baselines/2026-07-25_dual_group_news_embedding_baseline/code/train_dual_news.py`.
- Orchestration / aggregation: `scripts/convergence_check/` (`master_run.py`,
  `aggregate_convergence.py`, `mapping_old_20ep.json`, `mapping_new_40ep.json`).
- 40-epoch result dirs (paths in `mapping_new_40ep.json`): backbone
  `parallel_lstm_gnn_knn_seed{42,123,2026}_2026-08-08_*`, full
  `per_ticker_gate_2026-08-08_{080538,141440,144454}`, no-graph
  `no_graph_ablation_seed{42,123,2026}_2026-08-08_*`, no-gate
  `dual_group_news_2026-08-08_{163057,164533,165636}`.
- Reproduce: `python scripts/convergence_check/aggregate_convergence.py scripts/convergence_check/mapping_new_40ep.json`
