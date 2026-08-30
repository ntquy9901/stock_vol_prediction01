# GAT depth: 1-hop vs 2-hop (preliminary/exploratory)

**Question:** the delivered VolGA uses a 2-hop GAT (`MaskedRichNet(gat_layers=2)`), while GNNHAR
(arXiv:2308.01419) recommends 1-hop. Does 1-hop improve VolGA on the markets where the graph helps?

**Method (exploratory, throwaway harness — NOT yet a gated baseline):** monkeypatched
`MaskedRichNet` to `gat_layers=1`, reused `run_masked_rich.train_masked_rich` read-only, trained
no-graph / vol→PK-2hop / vol→PK-1hop in ONE harness (same panel, 3 seeds {42,123,2026}, 10 epochs),
date-clustered Diebold–Mariano between depths. Seed-ensembled QLIKE.

| panel/h | no-graph | vol→PK 2-hop | vol→PK 1-hop | DM 1-hop vs 2-hop |
|---|---:|---:|---:|---|
| VN100 h1 | 0.5772 | 0.5339 | **0.5103** | **p=0.021 (favours 1-hop)** |
| VN100 h5 | 0.5851 | 0.5690 | 0.5735 | p=0.130 (favours 2-hop, n.s.) |
| SP500 h1 | — | — | — | did not complete (498-node panel build hung/OOM) |

(Both depths beat no-graph at both VN100 horizons, p ≤ 1.5e-3.)

## Finding
1-hop is **significantly better than 2-hop at VN100 h1** (p=0.021, ~4.4% lower QLIKE) — consistent with
GNNHAR's 1-hop recommendation. But the advantage is **horizon-specific**: at VN100 h5 the two depths are
statistically indistinguishable (p=0.13, point estimate marginally favours 2-hop). So 1-hop is **not a
blanket improvement** — it helps at the shortest horizon (where the graph benefit is largest) and not at
h5. SP500 h1 is inconclusive (technical failure on the 498-node correlation-adjacency build).

## Caveats
- Exploratory throwaway harness (monkeypatch, single 10-epoch config); not a gated baseline. The
  no-graph QLIKE varies run-to-run (0.577 here vs 0.623/0.645 in other runs) — compare only WITHIN a run.
- Single config, two VN100 horizons only. Not sufficient to change the delivered default.

## Recommendation
1-hop is a **promising but not robust** lever. Before adopting it in VolGA, formalize as a proper
gat-depth baseline (tests + over/underfit evidence + gate) and run it across all horizons {1,5,10,22}
on VN100 and SP500 with the delivered seed/epoch protocol; adopt 1-hop only if it holds (or is neutral)
across horizons, not just at h1.
