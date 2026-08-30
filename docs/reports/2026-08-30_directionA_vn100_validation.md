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
