# Direction A validation — the VolGA graph helps on VN100 (not just null on HNX)

**Question (improvement-strategy Direction A):** does VolGA's graph branch help on a market that
actually has cross-sectional structure (VN100: 104 nodes, median |ρ|=0.33, ~62% of pairs |ρ|>0.3),
rather than HNX where structure is near-absent (median |ρ|=0.09) and 8 graph probes were null?

**Answer: yes, significantly, at short horizons.** From the delivered multi-seed masked_rich results
(`results/masked_rich_floor1e2/vn100_h{1,5,10,22}`; per-seed mean QLIKE + date-clustered
Diebold–Mariano), no new run required:

| h | LSTM (no-graph) | VolGA (LSTM+GAT vol→PK) | Δ QLIKE | DM VolGA vs no-graph LSTM |
|---:|---:|---:|---:|---|
| 1 | 0.6229 | **0.5504** | −0.0725 | **p = 1.1e-06 (favours VolGA)** |
| 5 | 0.6021 | **0.5759** | −0.0262 | **p = 4.6e-04 (favours VolGA)** |
| 10 | 0.6100 | 0.6103 | +0.0003 | p = 0.45 (n.s.) |
| 22 | 0.6525 | 0.6576 | +0.0052 | p = 0.14 (n.s.) |

- vs **HAR-X**: VolGA is a statistical tie at every horizon (DM p = 0.24 / 0.75 / 0.41 / 0.12) — i.e.
  VolGA is competitive with the strong parsimonious baseline on VN100.

## Interpretation
The graph's value is **market- and horizon-dependent**, exactly as the cross-sectional-structure EDA
predicts:
- **VN100** (real structure): the graph beats the no-graph LSTM decisively at h1 (p≈1e-6) and h5
  (p≈5e-4); the effect fades by h10/h22.
- **HNX** (near-zero structure, 45.3% floored targets): 8 structurally distinct graph approaches all
  null; two significantly worse.

This converts the project's "graph is null" result into a sharper, evidence-backed statement: **a
graph edge adds significant out-of-sample QLIKE value where a stable cross-sectional structure exists
and at short horizons, and adds nothing where it does not.** It also confirms the strategy's diagnosis
that the earlier null was HNX-specific — the single market with the weakest cross-sectional signal.

## Caveats
- Existing delivered data (not a fresh rerun); numbers trace to committed `result.json` (per-seed mean,
  date-clustered DM), so they are multi-seed and significance-tested, not single-run.
- The gain is at short horizons only; the graph does not help at h10/h22 on VN100.
- VolGA does not beat HAR-X on VN100 (tie) — the graph closes the gap to HAR-X from the no-graph LSTM,
  it does not surpass the parsimonious baseline.

## Next (optional, to strengthen)
Repeat the same DM comparison on **SP500** (498 liquid nodes) and confirm whether the short-horizon
graph gain holds on the largest liquid universe; and re-run VN100 with the *densified heterogeneous*
variant to see if separating linear/non-linear edges adds beyond the shipped vol→PK edge.

## Follow-up: densified heterogeneous variant on VN100 (does separating linear/non-linear edges help?)

Ran the densified heterogeneous 2-relation model on VN100 h1 (3 seeds, corr+lift graph: 2984 linear
ρ>0.25 edges avg-deg 58 + 179 non-linear lift>1.2 edges). Date-clustered DM (QLIKE), seed-ensembled:

| comparison | DM p (QLIKE) | favours |
|---|---|---|
| squashed corr+lift vs no-graph LSTM | **1.8e-09** | squashed (graph helps) |
| heterogeneous (separated) vs squashed | **3.2e-07** | **squashed** |
| heterogeneous vs no-graph LSTM | 0.062 | heterogeneous (n.s.) |

**Conclusion:** separating the linear and non-linear relations does **not** improve on simply combining
them — it is **significantly worse** than the squashed graph on QLIKE (p=3.2e-7). The combined corr+lift
graph, by contrast, significantly beats the no-graph LSTM (p=1.8e-9), reconfirming Direction A with a
second edge family. Across both markets the heterogeneous separation is never beneficial: neutral on
HNX (both null), harmful on VN100. Recommendation: keep the graph edges combined; do not pursue the
heterogeneous separation as a VolGA improvement.

## Correction: fair vol→PK vs corr+lift comparison (same harness)

A reviewer flagged that the earlier "vol→PK (0.55) beats corr+lift (0.62)" was a CROSS-RUN comparison
with different configs (vol→PK: 5 seeds + train-to-convergence, no-graph QLIKE 0.6229; corr+lift: 3
seeds / 10 epochs, no-graph QLIKE 0.6448 — the *identical* no-graph model differs by 0.022 across the
two runs). That claim was confounded and is retracted.

Re-run in ONE harness (VN100 h1, 3 seeds, 10 epochs, same no-graph baseline 0.6448), QLIKE + DM:

| model | QLIKE | DM vs no-graph LSTM |
|---|---|---|
| no_graph_LSTM | 0.6448 | — |
| stat GAT vol→PK (shipped VolGA edge) | 0.5413 | p = 5.6e-08 |
| corr+lift GAT | 0.5514 | p = 4.1e-07 |
| **vol→PK vs corr+lift** | | **p = 0.065 (not significant — a tie)** |

**Corrected conclusion:** on VN100 h1 the two edge families are statistically **indistinguishable**
(DM p=0.065); both significantly beat the no-graph LSTM (p≈1e-7). The robust effect is the graph
itself (~0.09 QLIKE vs no-graph), not the specific edge construction. Combined with the heterogeneous
result (separating the two relations is *worse* than combining them, p=3.2e-7), the practical guidance
is: use a single combined graph; the shipped vol→PK edge and the corr+lift edge are equivalent choices.
