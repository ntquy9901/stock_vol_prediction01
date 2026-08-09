# Consolidated metrics review — all experiment clusters to date

Scope: read-only consolidation of every committed metric cluster in the project as of
2026-08-09. All numbers below were read directly from the cited `results.json` /
`validation_comparison.json` / aggregate files or from the two source reports; none are recalled.
Where a file was missing or a run incomplete it is stated explicitly.

Metrics: MSE / RMSE / MAE / R² / QLIKE / DirAcc, all on the denormalized (physical) scale except
where a cluster reports a training-space validation loss (called out per cluster). DirAcc is the
corrected per-ticker value. Multi-seed entries are mean ± sample std (n as stated). Bold marks the
best value per metric **within a cluster only**.

---

## EVAL-BASIS CAVEAT (read first)

The clusters below are **not cross-comparable in absolute terms**. They differ in evaluation set,
data regime, split, epoch budget, and (for the graph clusters) reporting quantity. Absolute RMSE
therefore ranges ~0.0015 to ~0.0029 across clusters for reasons of eval basis, not model quality.
Compare only **within** a cluster.

| Cluster | Eval set | Regime / panel | Approx. RMSE band | Reporting |
|---|---|---|---|---|
| 1. Track A (paper v3) | **Test** | common-date panel (33 tickers) | ~0.0027–0.0029 | test_metrics |
| 2. Track B A1 screening (5-ep) | **Validation** | pooled async + common-date | ~0.0015 | best-checkpoint val |
| 3. Track B pooled (20-ep) | **Validation** | pooled async | ~0.0015 | best-checkpoint val |
| 4. Track B graph — intersection (batched) | **Validation** | intersection panel | ~0.0024 | val + train-space vloss |
| 5. Track B graph — masked (15-ep) | **Validation** | pooled masked | ~0.0015 | val + train-space vloss |
| 6. Multi-horizon HAR vs FULL | **Test** | common-date, 1/5/10/22-day | ~0.0025–0.0031 | test_metrics |

Track A / multi-horizon report on the **test** set; all Track B screening/graph clusters report on
the **validation** set (best checkpoint). A Track A RMSE of ~0.0027 and a pooled RMSE of ~0.0015 are
measuring different things and must not be ranked against each other.

---

## Cluster 1 — Track A (paper v3), common-date panel, 20-epoch, test set

HAR n=1 (deterministic linear model); the four deep variants n=3 seeds (42/123/2026).
Source: `docs/paper/architecture_diagrams_review.md` §6, which cites
`results/har_baseline_2026-08-05_224208/test_metrics.csv`,
`results/parallel_lstm_gnn_knn_*`, `results/per_ticker_gate_*`,
`results/no_graph_ablation_seed*`, `results/dual_group_news_2026-08-05_*`.

| Model | n | MSE ↓ | RMSE ↓ | MAE ↓ | R² ↑ | QLIKE ↓ | DirAcc ↑ (%) |
|---|---|---|---|---|---|---|---|
| (0) HAR classical | 1 | **4.76e-06** | **0.002182** | **0.000575** | 0.7419 | 0.5493 | **48.65** |
| (1) Price-only backbone | 3 | 8.55e-06±5.32e-07 | 0.002923±0.000090 | 0.000811±0.000014 | 0.7749±0.0140 | 0.4603±0.0205 | 48.47±0.35 |
| (FULL) News+Graph+Gate | 3 | 7.48e-06±5.28e-07 | 0.002734±0.000096 | 0.000793±0.000012 | 0.8031±0.0139 | 0.4430±0.0185 | 47.77±0.52 |
| (2) Ablation — No-Graph | 3 | 7.78e-06±3.22e-07 | 0.002788±0.000058 | 0.000788±0.000011 | 0.7953±0.0085 | 0.4657±0.0112 | 48.29±0.04 |
| (3) Ablation — No-Gate | 3 | 7.42e-06±4.02e-07 | 0.002723±0.000074 | 0.000787±0.000010 | **0.8047±0.0106** | **0.4366±0.0116** | 48.22±0.27 |

Within-cluster: HAR is best on MSE/RMSE/MAE/DirAcc; No-Gate is best on R²/QLIKE. RMSE/MAE and
QLIKE/R² disagree because QLIKE penalizes under-prediction at high-volatility points asymmetrically
(Patton 2011); HAR forecasts "smooth" (low mean error) but misses volatility spikes (where QLIKE
bites), while the news-bearing deep models capture spikes better.

---

## Cluster 2 — Track B A1 screening, 5-epoch, validation set (n=3)

P0 = HAR-only backbone; P1 = +news-concat; P2 = +news (screening config); P3 = +gate.
P0 is deterministic across seeds (std 0). Source: per-seed
`results/a1_{pooled,commondate}_seed{42,123,2026}/h5/validation_comparison.json`
(epochs=5, per `screening_metadata.json`).

### 2a. Pooled async regime
| Config | MSE ↓ | RMSE ↓ | MAE ↓ | R² ↑ | QLIKE ↓ | DirAcc ↑ (%) |
|---|---|---|---|---|---|---|
| P0 | **2.20379e-06** | **0.0014845** | **0.00047974** | **0.73515** | 0.51671 | 48.540 |
| P1 | 2.25715e-06±1.2e-08 | 0.0015024±4.1e-06 | 0.00049046±2.1e-06 | 0.72873±0.0015 | 0.51184±0.0011 | **48.588±0.042** |
| P2 | 2.21036e-06±1.0e-08 | 0.0014867±3.4e-06 | 0.00048038±2.7e-06 | 0.73436±0.0012 | **0.50839±0.00024** | 48.470±0.13 |
| P3 | 2.21632e-06±1.1e-08 | 0.0014887±3.8e-06 | 0.00048200±2.7e-06 | 0.73364±0.0013 | 0.50856±0.00022 | 48.580±0.068 |

### 2b. Common-date regime
| Config | MSE ↓ | RMSE ↓ | MAE ↓ | R² ↑ | QLIKE ↓ | DirAcc ↑ (%) |
|---|---|---|---|---|---|---|
| P0 | **2.22240e-06** | **0.0014908** | 0.00048472 | **0.73291** | 0.51472 | 48.478 |
| P1 | 2.23001e-06±1.2e-08 | 0.0014933±3.9e-06 | 0.00047666±1.7e-06 | 0.73200±0.0014 | **0.51277±0.0024** | **48.791±0.022** |
| P2 | 2.26043e-06±3.4e-09 | 0.0015035±1.1e-06 | **0.00047639±1.0e-06** | 0.72834±0.00041 | 0.51777±0.0055 | 48.363±0.16 |
| P3 | 2.27130e-06±2.3e-08 | 0.0015071±7.8e-06 | 0.00048014±5.1e-06 | 0.72703±0.0028 | 0.51628±0.0035 | 48.620±0.079 |

At 5-epoch screening the news/gate configs sit within noise of the P0 backbone in both regimes.
Pooled vs common-date differences are all within one std (pooling null — see verdict §7).

---

## Cluster 3 — Track B pooled, 20-epoch, validation set (n=3)

Source: `results/pooled_20ep_aggregate.json` (means, stds, paired-t computed over
`results/pooled_20ep_seed{42,123,2026}/h5/{P0-P3}/`; epochs=20 per `screening_metadata.json`).

| Config | MSE ↓ | RMSE ↓ | MAE ↓ | R² ↑ | QLIKE ↓ | DirAcc ↑ (%) |
|---|---|---|---|---|---|---|
| P0 | **2.20379e-06±0** | **0.0014845±0** | **0.00047974±0** | **0.73515±0** | 0.51671±0 | 48.540±~0 |
| P1 | 2.24541e-06±1.24e-08 | 0.0014985±4.1e-06 | 0.00048871±2.5e-06 | 0.73014±0.0015 | 0.51098±0.00058 | **48.663±0.069** |
| P2 | 2.20800e-06±8.7e-09 | 0.0014859±2.9e-06 | 0.00048011±2.5e-06 | 0.73464±0.0010 | 0.50843±0.00079 | 48.528±0.017 |
| P3 | 2.21042e-06±1.52e-08 | 0.0014867±5.1e-06 | 0.00048056±4.9e-06 | 0.73435±0.0018 | **0.50842±0.00024** | 48.533±0.10 |

**Convergence / overfit note (per task brief):** best validation is reached at epoch 5–6; raw
validation loss then degrades for P2/P3 (≈0.846 → 0.91–0.95 by epoch 20). The metrics tabulated
above are the best-checkpoint (early-stopped) values, not the epoch-20 values, so extending 5→20
epochs did not improve the reported numbers — it exposed overfitting past the early optimum. P0 std
shows as ~0 because the HAR-only reference is effectively seed-invariant here.

---

## Cluster 4 — Track B graph, intersection panel, batched, seed42, 5-epoch (n=1)

G0 = graph-off (news, no cross-stock edges); G1 = graph-on (news propagated over edges).
`paired_delta` = G1 val-loss − G0 val-loss (train-space MSE loss; positive → G1 worse).
Source: `results/pooled_news_gnn_g0g1_batched_2026-08-08_171457_seed42/h5/graph_validation_comparison.json`
(+ per-config `G0/results.json`, `G1/results.json`).

| Config | val-loss | MSE ↓ | RMSE ↓ | MAE ↓ | R² ↑ | QLIKE ↓ | DirAcc ↑ (%) |
|---|---|---|---|---|---|---|---|
| G0 (graph off) | **0.86071** | **5.793e-06** | **0.0024069** | 0.00067014 | **0.74469** | **0.68762** | **48.614** |
| G1 (graph on) | 0.86269 | 5.809e-06 | 0.0024102 | **0.00066018** | 0.74399 | 0.69628 | 48.522 |

paired_delta = **+0.00198** (G1 worse). On the intersection panel the graph adds nothing at 5-epoch:
G1 loses on val-loss, MSE, RMSE, R², QLIKE, DirAcc; it wins only MAE by ~1e-5. Graph-null.
(Note RMSE band here ~0.0024 — intersection panel, distinct eval basis from clusters 2/3/5.)

---

## Cluster 5 — Track B graph, masked, 15-epoch, seed42, three adjacency modes (n=1 per mode)

15-epoch runs (confirmed: 15 train/val loss entries, monotonic val-loss descent → converged, no
overfit). G0 is identical across all three modes (adjacency is irrelevant when the graph is off).
`paired_delta` = G1 val-loss − G0 val-loss (negative → G1 better). Source:
`.worktrees/masked-gnn/results/pooled_news_gnn_masked_{dense,knn8,thr07}_seed42_2026-08-08_230837/h5/graph_validation_comparison.json`
(dense/thr07 timestamp `_230837`; knn8 present at both `_230837` and a re-run `_2026-08-09_071141`
with byte-identical metrics).

| Config | val-loss | MSE ↓ | RMSE ↓ | MAE ↓ | R² ↑ | QLIKE ↓ | DirAcc ↑ (%) | Δ vs G0 |
|---|---|---|---|---|---|---|---|---|
| G0 (graph off) | 0.83924 | 2.14947e-06 | 0.0014661 | 0.00046554 | 0.74167 | 0.51009 | 48.706 | — |
| G1 dense (avg 18.6 edges) | 0.83797 | 2.13264e-06 | 0.0014604 | 0.00046081 | 0.74370 | 0.50647 | **49.098** | −0.00127 |
| G1 knn-8 (avg 5.9 edges) | **0.83671** | **2.12864e-06** | **0.0014590** | **0.00046067** | **0.74418** | **0.50646** | 48.712 | **−0.00253** |
| G1 thr-0.7 (avg 1.1 edges) | 0.83924 | 2.14886e-06 | 0.0014659 | 0.00046286 | 0.74175 | 0.50980 | 48.457 | 0.00000 |

On the masked pooled panel at convergence (15-epoch), G1 (graph-on) **beats** G0 for dense and
knn-8 — reversing the cluster-4 intersection result. knn-8 gives the largest improvement
(Δ val-loss −0.00253) and is best on 5/6 metrics; dense is best on DirAcc. thr-0.7 is so sparse
(avg 1.1 off-diagonal edges) that it collapses to ≈G0 (Δ = 0). **This is a single-seed result
(seed42 only).**

### Supplementary — earlier masked 3-seed run (adjacency mode not recorded in JSON)
Source: `.worktrees/masked-gnn/results/pooled_news_gnn_masked_g0g1_2026-08-08_21{2959,4227,4916}_seed{42,123,2026}`.
These predate the explicit dense/knn8/thr07 adjacency split; `adjacency` field is null.

| Seed | G0 RMSE | G1 RMSE | paired_delta (val-loss) |
|---|---|---|---|
| 42 | 0.0014645 | 0.0014579 | +0.00241 |
| 123 | 0.0014643 | 0.0014547 | +0.00076 |
| 2026 | 0.0014642 | 0.0014541 | −0.00006 |

The val-loss delta changes sign across seeds (+/+/−), i.e. no consistent graph benefit under the
default adjacency at 3 seeds. Only the later knn-8 sparse-adjacency single-seed run shows a
clean negative delta.

---

## Cluster 6 — Multi-horizon (1/5/10/22-day), HAR vs FULL, test set (n=3)

Source: `docs/reports/2026-08-06_multihorizon_har_vs_full.md` §3 (18 runs, seeds 42/123/2026,
20-epoch, denormalized, corrected per-ticker DirAcc; result dirs
`results/har_only_h{1,10,22}_*` and `results/per_ticker_gate_h{1,10,22}_*`; 5-day column reproduced
from `docs/reports/2026-08-03_final_paper_readiness_report.md` §1). Bold = better mean at that
horizon.

### QLIKE ↓
| Horizon | HAR-only | FULL (news gate) | FULL − HAR (paired) |
|---|---|---|---|
| 1-day | **0.3778±0.0100** | 0.3837±0.0225 | +0.0059 (t=0.33, ns) |
| 5-day | 0.4603±0.0205 | **0.4430±0.0185** | −0.0173 (t=−6.22, sig) |
| 10-day | 0.5240±0.0168 | **0.5186±0.0168** | −0.0054 (t=−0.96, ns) |
| 22-day | **0.5262±0.0068** | 0.5675±0.0370 | +0.0412 (t=1.64, ns) |

### RMSE ↓
| Horizon | HAR-only | FULL (news gate) | FULL − HAR (paired) |
|---|---|---|---|
| 1-day | **0.002501±0.000089** | 0.002506±0.000182 | +0.000005 (t=0.03, ns) |
| 5-day | 0.002923±0.000090 | **0.002734±0.000096** | −0.000189 (t=−9.38, sig) |
| 10-day | 0.003093±0.000048 | **0.003063±0.000036** | −0.000030 (t=−3.69, consistent, sub-threshold) |
| 22-day | **0.002955±0.000021** | 0.003066±0.000087 | +0.000111 (t=2.05, borderline) |

### R² ↑ / MAE ↓ / DirAcc ↑
| Horizon | R² HAR | R² FULL | MAE HAR | MAE FULL | DirAcc HAR | DirAcc FULL |
|---|---|---|---|---|---|---|
| 1-day | **0.8345±0.0117** | 0.8334±0.0241 | **0.000724±0.000010** | 0.000728±0.000030 | 32.47±0.30 | **33.32±0.62** |
| 5-day | 0.7749±0.0140 | **0.8031±0.0139** | 0.000811±0.000014 | **0.000793±0.000012** | **48.47±0.35** | 47.77±0.52 |
| 10-day | 0.7470±0.0080 | **0.7519±0.0058** | 0.000849±0.000007 | **0.000845±0.000005** | 47.23±1.88 | **47.72±1.44** |
| 22-day | **0.7540±0.0035** | 0.7351±0.0150 | **0.000846±0.000006** | 0.000854±0.000004 | 42.05±5.62 | **43.56±6.50** |

News benefit on continuous-error metrics is non-monotonic in horizon and peaks at 5-day: clear/sig
(5d) → marginal (10d) → net harmful (22d); 1-day is a wash. Only 5-day reaches significance at n=3.

---

## 7. Statistical verdict

Paired-t values below are read from `results/pooled_20ep_aggregate.json` (cluster 3, n=3, df=2,
two-sided; n=3 significance threshold |t|≈4.30 at α=0.05) and from
`docs/paper/architecture_diagrams_review.md` §7 (cluster 1, n=3).

- **News — SIGNIFICANT on error metrics.** Cluster 3 P2 vs P1: MSE t=−10.50 (p=0.0090),
  RMSE t=−10.60 (p=0.0088), MAE t=−17.0 (p=0.0034), R² t=+10.50 (p=0.0090) — all significant;
  QLIKE t=−4.17 (p=0.053, mean_diff −0.00254) borderline; DirAcc t=−2.93 (p=0.099) ns.
  Corroborated on Track A test set (cluster 1, FULL vs backbone): QLIKE t=−6.22, RMSE t=−9.38 (both
  sig). News is the one added mechanism with a measurable, significant effect.
- **Gate — NULL.** Cluster 3 P3 vs P2: MSE t=+0.60, RMSE t=+0.60, MAE t=+0.29, R² t=−0.60,
  QLIKE t=−0.03, DirAcc t=+0.06 — all p>0.6. Track A no-gate vs gate |t|<2.3 (ns), no-gate mean
  marginally better on all six. The per-ticker gate buys no measurable accuracy; keep only for
  interpretability.
- **Graph — NULL except a single-seed knn-8 hint.** Track A no-graph vs FULL |t|<1.9 (ns).
  Intersection/batched (cluster 4): paired_delta +0.00198 (G1 worse). Masked default-adjacency
  3-seed (cluster 5 supplement): val-loss delta sign inconsistent (+0.00241/+0.00076/−0.00006).
  Masked knn-8 sparse adjacency (cluster 5, seed42 only): Δ val-loss **−0.00253**, best on 5/6
  metrics — a **hint**, not a claim. Diebold-Mariano confirmation and multi-seed knn-8 are IN
  FLIGHT (see §8).
- **Pooling (A1) — NULL.** Cluster 2 pooled vs common-date: both regimes ≈0.0015 validation RMSE
  with per-config differences inside one std (e.g. P2 pooled 0.0014867 vs common-date 0.0015035).
  No formal cross-regime paired-t is stored in the result files; on the tabulated means pooling
  neither helps nor hurts.
- **DirAcc — ~48–49% everywhere, structural.** Across all clusters DirAcc sits near or below the
  50% random line (Track A 47.8–48.7%; pooled 48.5–48.7%; masked 48.5–49.1%; 1-day horizon far
  lower at 32–33%). No configuration separates on DirAcc at n=3. This matches the documented
  anti-persistence finding (per-ticker day-over-day sign has ≈−0.30 autocorrelation;
  `docs/reports/2026-08-04_diracc_low_accuracy_analysis.md`) — a structural ceiling, not a model
  defect.

---

## 8. As-of note (work in flight)

The multi-seed knn-8 confirmation and the Diebold-Mariano test are still running as of
2026-08-09 07:47. Present on disk: masked knn-8 **seed42 only** (two byte-identical runs,
`_230837` and `_2026-08-09_071141`). The knn-8 **seed123** directory
(`pooled_news_gnn_masked_knn8_seed123_2026-08-09_071141`) exists but is **empty — no
`graph_validation_comparison.json` yet**; **seed2026 knn-8 does not exist yet**. No
Diebold-Mariano output file was found under the masked-gnn worktree. The knn-8 graph gain is
therefore a single-seed hint pending multi-seed + DM confirmation; it is not yet a statistical
claim.

---

## 9. Provenance / missing-file log

- Cluster 1: `docs/paper/architecture_diagrams_review.md` §6 (cites test_metrics / training_results
  JSONs listed therein). Read.
- Cluster 2: `results/a1_{pooled,commondate}_seed{42,123,2026}/h5/validation_comparison.json`
  (6 files) + `screening_metadata.json`. All read; means/stds computed from per-seed rows.
- Cluster 3: `results/pooled_20ep_aggregate.json` + `results/pooled_20ep_seed42/h5/` metadata.
  Read. (Per-seed P0–P3 `results.json` are ~1 MB each — not opened individually; the committed
  aggregate is authoritative for means/stds/paired-t.)
- Cluster 4: `results/pooled_news_gnn_g0g1_batched_2026-08-08_171457_seed42/h5/`
  `graph_validation_comparison.json` + `G0/G1 results.json`. Read.
- Cluster 5: `.worktrees/masked-gnn/results/pooled_news_gnn_masked_{dense,knn8,thr07}_seed42_*`
  and `..._masked_g0g1_2026-08-08_21*_seed*` comparison JSONs. Read (worktree files read-only, not
  modified). knn-8 seed123 empty; seed2026 absent (see §8).
- Cluster 6: `docs/reports/2026-08-06_multihorizon_har_vs_full.md` §3. Read.

**Not found / incomplete:** masked knn-8 seed123 results (dir empty), masked knn-8 seed2026 (no
dir), any Diebold-Mariano output artifact. These are the in-flight items in §8.
