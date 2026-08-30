# VolGA improvement strategy — grounded in the overnight EDA + 8-probe graph sweep (2026-08-30)

## 1. What the overnight evidence establishes

**8 graph approaches were tested on HNX h1 and NONE beats the no-graph LSTM** (statistical vol→PK,
sector ICB, MTGNN learned-adjacency, Diebold–Yilmaz spillover, Graph-WaveNet TCN+adaptive, corr+lift
paper edge, heterogeneous 2-relation, serial LSTM→GNN hybrid). Two are significantly *worse*
(DY 2.20 p<0.001; corr+lift p=0.0037). The serial-vs-parallel architecture change is a statistical
tie (DM p=0.92). **The null is a property of the market, not of the architecture.**

**The EDA quantifies WHY** (per-market cross-sectional structure + liquidity):

| market | nodes (screened) | median \|ρ\| | pairs \|ρ\|>0.3 | zero-Parkinson | OHLC geom-violations |
|---|---|---|---|---|---|
| VN30 | 33 | 0.349 | ~46% | 1.7% | 0 |
| VN100 | 104 | 0.333 | ~62% | 3.3% | 0 |
| HOSE | 378 | 0.145 | ~11% | 12.6% | 3,623 |
| **HNX** | **154** | **0.095** | **~3%** | **45.3%** | 3,474 |
| SP500 | 498 | 0.216 | — | 1.1% | 0 |

Key facts:
- On **HNX** the linear edge |ρ|>0.7 fires on only **3 / 11,781 pairs** and Apriori lift>1.7 on 12 →
  142/154 nodes are singletons. There is almost no cross-sectional signal to propagate.
- **45.3% of HNX Parkinson targets are exactly zero** (H==L limit/no-trade days) → the QLIKE target is
  floor-dominated; floor-activation reaches 0.72–0.92 on several kept tickers (S55, STP, HJS).
- Correlation magnitude **alone** does not predict graph/deep lift: VN30/VN100 are *more* correlated
  than SP500, yet deep/graph helped most on SP500. The real driver is
  **node-count × liquidity × the availability of STABLE, estimable edges out-of-sample.**

## 2. Implication — where the improvement budget should go

VolGA's graph branch is being asked to extract cross-sectional signal from HNX, the single market where
that signal is weakest and the target is most floor-corrupted. No edge construction can manufacture
signal that is not in the data. The improvement directions therefore rank as follows.

### Direction A (highest expected value) — move the graph to a universe where it can work
Run VolGA's graph on the markets that actually have cross-sectional structure AND enough liquid nodes:
**VN100** (104 nodes, median |ρ|=0.33, 62% of pairs |ρ|>0.3) and **SP500** (498 liquid nodes, where
deep/graph already helped). Report HNX as the honest hard case. This tests the graph where it has
headroom instead of where it is provably empty. *Caveat learned tonight:* VN30/VN100 have high ρ but
few nodes; SP500 has moderate ρ but many nodes — so run BOTH and separate the node-count effect from
the correlation-magnitude effect before crediting the graph.

### Direction B — fix the target before fixing the model (HNX-specific)
The HNX QLIKE is dominated by 45.3% zero-Parkinson days, not by model quality. Higher-value than any
graph on HNX:
- a **liquidity/zero-target screen** (drop or down-weight tickers with floor-activation above a
  disclosed threshold; report the screened universe);
- a **robust/again-floored estimator** or an explicit floor model, disclosed as a data-quality caveat;
- clean the **systematic feed-artifact dates** the EDA found (corrupt OHLC clusters on shared dates:
  54 tickers on 2013-11-22, 48 on 2013-05-24, 43 on 2011-10-13) — a targeted date-level fix, not
  per-ticker.

### Direction C — use the graph as a REGULARIZER, not a signal source
The only reproducible graph effect tonight was **variance reduction**: the heterogeneous 2-relation
model cut seed-variance (QLIKE std 0.011 vs 0.063/0.070) and the serial hybrid gave a small significant
**MAE** reduction (p<0.001, ~1%) while leaving QLIKE unchanged. If the graph is kept, frame and use it
as a light regularizer / ensemble-stabiliser (report MAE + seed-stability), not as a QLIKE-beating
mechanism — and prefer the *densified* edge (ρ>0.25) which removed the paper-threshold harm.

### Direction D — spend model capacity on the temporal/feature side
Because cross-sectional signal on VN is weak, marginal gains are more likely from the temporal branch
and node features than from edges: richer per-stock features, HARQ-style realised-quantity terms, or a
stronger temporal backbone — evaluated against HAR, which the EDA (excess kurtosis ≈102 + persistent
|return| autocorrelation) shows is a genuinely strong baseline on this data.

## 3. Recommended next experiment (concrete, single, decisive)
Run VolGA (delivered parallel LSTM+GAT vol→PK) and the densified heterogeneous variant on **VN100 and
SP500** at h1/h5, 3 seeds, with the same masked-panel / HAR-X anchor / QLIKE / DM protocol used on HNX.
Success = graph beats no-graph LSTM on QLIKE with DM p<0.05 surviving multiple-comparison correction on
at least one of these liquid, structured universes. If it fails there too, the honest paper conclusion
is strengthened: for daily volatility, the graph adds no robust QLIKE signal even where cross-sectional
structure exists — and VolGA's contribution is the temporal model + the rigorous multi-market,
multi-horizon, DM-tested negative-result on the graph.

## 4. What NOT to do
- Do not keep searching for a better *edge construction on HNX* — 8 structurally distinct approaches
  (correlation, sector, learned, spillover, TCN-adaptive, association-rule, heterogeneous, serial) all
  returned null; the ceiling is the data, not the edge.
- Do not report a graph "win" from a borderline single-comparison p-value (e.g. hetero p=0.0499) without
  multiple-comparison correction and seed-stability.
- Do not lower the correlation threshold just to densify the HNX graph and call the resulting
  noise-edges a signal.

---
*Sources: docs/reports/2026-08-30_{hnx_full,vn_markets_eda_comparison,sp500}_eda.* + per-ticker
diagnostic; baselines/2026-08-30_{hetero_graph,lstm_gnn_serial_hybrid}_ablation + the 6 earlier HNX
edge probes; results/*ablation* (h1–h22). All numbers trace to committed result.json / EDA HTML.*
