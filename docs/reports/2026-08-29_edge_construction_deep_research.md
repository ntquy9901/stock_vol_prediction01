# Edge-Construction Methods for Stock-Volatility GNNs — Deep-Research Synthesis (2026-08-29)

Deep-research workflow (96 agents, 5 web-search angles → fetch → 3-vote adversarial verification → synthesis).
**Caveat:** the safety classifier was unavailable during part of the adversarial-verify pass, so the citations
below should be re-checked (open the arXiv/DOI links) before being cited in the paper.

## Headline answer

The 2020–2025 literature offers a rich menu of graph-edge constructions, but **the evidence that a graph
improves DAILY VOLATILITY forecasting over strong non-graph baselines (HAR, LSTM) is mixed and modest, not
decisive.** This is consistent with (and supports) our own finding that the statistical volume→volatility edge
adds no robust value and the graph's contribution is horizon-dependent.

## Edge families (verified, with citations)

1. **Statistical correlation / distance graphs on the volatility series** (metadata-light).
   - Chen & Robert (2022), Graph Transformer Network fusing LOB + cross-sectional relations — arXiv:2112.09015.
   - Mattera & Otto (2023), network log-ARCH; edges from distance/correlation between volatility series; beats
     independent univariate — arXiv:2303.11064.
   - Gorduza et al. (2022), company-correlation graph; GAE reconstruction error tracks market volatility —
     arXiv:2212.04974.
   - Stability: estimated on train → can go stale OOS (matches our vol2pk instability finding).

2. **Diebold–Yilmaz FEVD spillover connectedness** (principled directed weighted edges).
   - Diebold & Yilmaz (2014), J. Econometrics 182 — "variance decompositions define weighted, directed networks";
     the FEVD matrix maps directly onto a GNN adjacency. DOI:10.1016/j.jeconom.2014.04.012.
   - Used as GNN edges: arXiv:2409.15320 (volatility-spillover-index graphs, time-varying directed weights).
   - Already in our paper's refs (`dy2012`). Strong econometric basis; a candidate we have NOT tried.

3. **Learned / adaptive graph-structure-learning** (most metadata-light, most transferable to VN).
   - MTGNN — Wu et al. (2020), "Connecting the Dots", KDD.
   - Graph WaveNet (2019, self-adaptive adjacency); GTS — Shang et al. (ICLR 2021, learns a discrete graph).
   - No volatility-specific proof it beats HAR OOS, but removes the predefined-graph requirement.

4. **Sector / industry (GICS/ICB) graphs** — help, but relation-type-dependent (our sector-GAT experiment).

5. **Knowledge-graph / relational edges (Wikidata competitor/subsidiary, supply-chain)** — DEPRIORITISED for VN.
   - RSR/TGC — Feng et al. (2019), ACM TOIS — Temporal Graph Convolution on a stock relation network.
   - HATS — Kim et al. (2019), arXiv:1908.07999 — hierarchical attention over multiple relation types; explicitly
     finds "performance can change depending on the relational data used" (edge choice is NOT neutral).
   - MGRN (2021), arXiv:2107.10941 — news sentiment + multiple relational graphs.
   - **Key caveat:** this evidence is for RETURN / MOVEMENT / trading prediction, **not daily volatility**, and it
     needs rich metadata (Wikidata, supply-chain) that Vietnam largely lacks.

## Answers to the two targeted questions

(a) **Do GNN graphs beat HAR/LSTM on daily volatility?** Mixed and modest. Relational-graph gains are mostly on
return/movement, not volatility; volatility-GNN papers show improvements over *independent univariate* baselines
but rarely a decisive win over a strong HAR. Our honest framing is well-supported by the literature.

(b) **Most stable OOS + suitable for an emerging market (Vietnam)?** Static, structural edges (sector, ownership)
are stable but sector needs a taxonomy (we have ICB via vnstock). Estimated edges (correlation, lead-lag) are the
least stable OOS. Learned/adaptive adjacency is the most metadata-light and transferable to thinly-traded VN
stocks that lack supply-chain/Wikidata data.

## Ranked recommendation for a Vietnam volatility GNN (reproducibility + OOS stability first)

1. **Correlation / distance graph on realized volatility** (metadata-light, well-cited) — but guard OOS stability.
2. **Diebold–Yilmaz spillover connectedness** (principled directed edges; already citable in our refs).
3. **Learned / adaptive adjacency** (MTGNN-style; most transferable to VN, no metadata needed).
4. **Sector / ICB graph** (stable, we have the data; our directional result is promising).
5. Knowledge-graph / supply-chain — deprioritise (missing VN metadata).
- Train with **QLIKE**, not MSE.

## Implications for our work

- The literature CONFIRMS our honest, mixed result — we can cite it to frame "graphs are not a guaranteed win
  for daily volatility" rather than over-claiming.
- Two principled, citable edges we have NOT tried and could add as future work / a stronger contribution:
  **Diebold–Yilmaz spillover** and **learned/adaptive adjacency** — both stronger-cited than our statistical
  vol2pk edge, and adaptive is the modern metadata-light direction.
- Our **sector-GAT** experiment (rank #4) is a legitimate, stable-edge contribution; combine with the above framing.
