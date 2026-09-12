# Complex-Network Topology Features for the gamma-GBM: Technical Design

**Date:** 2026-09-12
**Status:** DESIGN-BEFORE-RUN (for advisor review; the experiment is NOT yet run)
**Scope:** How we adapt the Complexity-2026 complex-network method as market-level features for our per-stock daily gamma-GBM volatility model.
**Implementation under review:** `scripts/eda/complex_network_gbm.py`
**Pipeline figure:** `docs/paper/figures/fig_complex_network_pipeline.png`

Source paper: N-K-K. Nguyen, H-T. Dinh, Q. Nguyen, "Complex Network Built From Stock Price Returns and Volumes to Predict Market Volatility and Volume," *Complexity*, 2026. DOI 10.1155/cplx/5670093. Framework figure: `docs/experement_guide/cplx5670093-fig-0001-m.jpg`.

---

## 1. Purpose and the honest question

The paper builds time-varying financial networks from combined return-and-volume correlations and shows that GLOBAL network-topology metrics predict one-month-ahead VNIndex volatility. This document specifies how we test whether the SAME topology metrics add incremental value to our own per-stock daily gamma-GBM, and it fixes every design choice BEFORE running so the advisor can approve or reject the protocol.

The design mirrors the code exactly. Where our application deviates from the paper (target, scoring, window sizes), the deviation is stated explicitly with its rationale.

---

## 2. The paper's exact method (stated faithfully)

Reference: paper Section 2 (Methodology), pages 2-8; framework in Figure 1.

**Data.** HOSE (Ho Chi Minh City Stock Exchange), 2015 to end of 2024, 1750 trading days. Liquidity filter: exclude stocks with average daily volume below 10,000 shares or with more than 70% trading-day gaps; 279 stocks of 1679 survive. For each stock and day the paper records the log return `r_it = ln S_it - ln S_{i,t-1}` and the log volume `ln(V_i(t))`. VNIndex is the market target series.

**Rolling window and network construction (Section 2.2).** A sliding window spans `ΔT = 125` trading days (approximately 6 months) and is shifted forward by `L = 21` trading days (approximately 1 month). Within each window, for the N surviving stocks:

1. Compute the return correlation matrix `ρ^R` and the log-volume correlation matrix `ρ^V` (Pearson).
2. Blend them: `ρ_mix = α · ρ^R + (1 - α) · ρ^V`, with `α ∈ [0, 1]` controlling the return-vs-volume weight. The paper's optimal weight for volatility is `α = 0.7` (paper eq. 1).
3. Build a network by one of three filters: **Threshold** (undirected edge if `|ρ_mix| ≥ τ`), **MST** (Kruskal on distance `d_ij = sqrt(2(1 - ρ_ij))`), or **Top-k** (each node keeps its k highest-correlation neighbours, k = 5). The paper reports **Threshold is best for volatility**.

**Seven global topology metrics (Section 2.3).** From each network the paper extracts a standardized set of seven metrics: Density, Average Degree, Average Clustering Coefficient, Average Weight (mean `|corr|` over edges), Diameter (of the largest connected component), Average Betweenness Centrality, Average Eigenvector Centrality.

**Targets and models (Sections 2.4, 2.5).** Each window yields one feature-target pair. The three market-level targets over the next `L = 21` days are: average VNIndex log return, average log volume, and the standard deviation of log returns (volatility). Models are Linear Regression and Random Forest under walk-forward validation, scored by `R^2` and RMSE.

**Headline result.** Threshold + Random Forest at `α = 0.7` gives volatility `R^2 ≈ 0.563` and volume `R^2 ≈ 0.95`. Mechanism claim (Section 3.6, Discussion): surges in network density and centrality precede heightened market volatility, interpreted as market synchronization / a systemic-risk regime where idiosyncratic risk is replaced by systemic risk.

---

## 3. Our application: what differs and why

### 3.1 Target and scoring difference (the honest question)

| Axis | Paper (Complexity-2026) | Our application |
|---|---|---|
| Prediction unit | MARKET-LEVEL (VNIndex), one series | PER-STOCK, every ticker in the panel |
| Horizon / frequency | one-month-ahead (21-day), one value per window | daily, `h ∈ {1, 5, 10, 22}` steps ahead |
| Target | avg return / avg log volume / return std-dev | Parkinson variance `pk_{i,t+h}` |
| Predictors | 7 topology metrics ALONE | 9 own-history features + the 7 topology metrics |
| Score | `R^2`, RMSE | pooled per-observation QLIKE + date-clustered Diebold-Mariano |
| Question | Do topology features explain market volatility? | Do topology features add INCREMENTAL QLIKE to own-history GBM? |

The paper answers an explanatory question ("can topology alone explain market volatility?", scored by `R^2`). We answer a different, stricter question: do these market-level topology features, broadcast to every stock, beat the model's own history on our loss-of-record (QLIKE) with a significance test that respects cross-sectional dependence (date-clustered DM)?

**Why QLIKE + DM incremental test is the honest question.** A high `R^2` for market volatility does not imply the features help a strong per-stock forecaster. Our gamma-GBM already encodes each stock's own volatility persistence; the only defensible claim is that topology adds signal ON TOP of that. QLIKE is our scoring loss of record for volatility (gamma deviance equals QLIKE up to a constant, so the GBM trains and is scored on the same objective), and the date-clustered DM collapses each day's cross-section to one value before testing so that many tickers sharing a date are not counted as independent observations. `R^2` on a single market series cannot make the incremental claim; QLIKE-vs-GBM under DM can.

### 3.2 Window-size deviation (documented)

Our implementation uses `WIN = 66` and `STEP = 22`, NOT the paper's `ΔT = 125` / `L = 21`.

Quoting the code (`scripts/eda/complex_network_gbm.py`):

```python
# scripts/eda/complex_network_gbm.py
WIN = 66          # trailing window ΔT = 3 months at 22 trading days/month (our monthly convention;
#                   paper used 125 ~= 6 months, we align with the project's month=22 features)
STEP = 22         # recompute + slide by one month = 22 trading days, forward-fill in between
THR = 0.5         # threshold tau on |combined corr| (paper "Threshold" network; best for volatility)
ALPHA = 0.7       # combined = ALPHA*return-corr + (1-ALPHA)*log-volume-corr (paper's optimal alpha=0.7)
```

**Rationale.** The project's convention is 1 month = 22 trading days. `har_monthly` and `volume_zscore_22` both use a 22-day month. We therefore set the trailing correlation window to `ΔT = 3 months = 66 trading days` and recompute / slide the network by one month = `22 trading days`, forward-filling the metrics on the days in between. This aligns the topology window with the project's monthly features rather than importing the paper's 125/21 calendar. Everything else (`α = 0.7`, `τ = 0.5`, the 7 metrics, combined return+volume correlation, causal construction, forward-fill) matches the paper.

---

## 4. Graph construction (matching the code)

Quoting `scripts/eda/complex_network_gbm.py::build_topo`:

```python
# scripts/eda/complex_network_gbm.py
def build_topo(frames, market):
    ret = pd.DataFrame({tk: d.set_index("date")["daily_return"] for tk, d in frames.items()}).sort_index()
    vcol = "volume_zscore_22"                                   # log-volume proxy (normalised volume)
    vol = pd.DataFrame({tk: d.set_index("date")[vcol] for tk, d in frames.items()
                        if vcol in d.columns}).sort_index()
    dates = ret.index
    rows = {}
    for i in range(WIN, len(dates), STEP):
        d0 = dates[i]
        Rw = ret.iloc[i - WIN:i].dropna(axis=1, thresh=int(WIN * 0.8))
        Vw = vol.iloc[i - WIN:i].dropna(axis=1, thresh=int(WIN * 0.8))
        common = Rw.columns.intersection(Vw.columns)            # same tickers to combine the two matrices
        if len(common) < 20:
            continue
        rc = np.nan_to_num(Rw[common].corr().to_numpy()); vc = np.nan_to_num(Vw[common].corr().to_numpy())
        combined = ALPHA * rc + (1 - ALPHA) * vc                # paper: alpha-weighted combined corr matrix
        rows[d0] = global_feats(combined)
    F = pd.DataFrame.from_dict(rows, orient="index", columns=TOPO).reindex(dates).ffill()
    return F
```

Construction notes, all matching the paper except the window sizes above:

- **Two correlation matrices on a common ticker set.** The return matrix uses `daily_return`; the volume matrix uses `volume_zscore_22`. Only tickers present in BOTH windows (`common`) enter the blend, so the two matrices are conformable. A window needs at least 20 common tickers or it is skipped.
- **Volume proxy.** Our stand-in for the paper's log-volume is `volume_zscore_22` (normalised volume). This is a deviation in the volume variable name only; the correlation structure it captures is the co-movement of trading activity, which is the paper's intent.
- **Blend `α = 0.7`.** `combined = ALPHA * rc + (1 - ALPHA) * vc`, exactly the paper's eq. (1) with the paper's optimal `α` for volatility.
- **Threshold `τ = 0.5`.** The network keeps an undirected edge wherever `|combined| ≥ 0.5` (paper's best filter for volatility). MST and Top-k are the paper's alternatives; this design uses Threshold only, as the paper found it best for volatility.
- **Causal / leakage-safe.** For a network dated `d0 = dates[i]`, every correlation uses only rows `iloc[i - WIN : i]`, i.e. data strictly up to (not including) `d0`. Metrics are then forward-filled (`.ffill()`) to daily so each trading day inherits the most recent past network. No future window ever contributes to a day's features.

---

## 5. The seven global metrics (matching the code)

Quoting `scripts/eda/complex_network_gbm.py::global_feats`:

```python
# scripts/eda/complex_network_gbm.py
def global_feats(C):
    """The paper's 7 global topology metrics of the Threshold graph from combined correlation matrix C:
    density, average degree, average clustering, average weight, diameter (largest component), average
    betweenness centrality, average eigenvector centrality."""
    n = C.shape[0]
    absC = np.abs(C); np.fill_diagonal(absC, 0.0)
    A = (absC > THR)
    G = nx.from_numpy_array(A.astype(float))
    dens = nx.density(G)
    avg_deg = float(np.mean([d for _, d in G.degree()]))
    clus = nx.average_clustering(G)
    ew = absC[A]
    avg_w = float(ew.mean()) if ew.size else 0.0
    if G.number_of_edges():
        H = G.subgraph(max(nx.connected_components(G), key=len))
        diam = float(nx.diameter(H)) if H.number_of_nodes() > 1 else 0.0
    else:
        diam = 0.0
    betw = float(np.mean(list(nx.betweenness_centrality(G).values()))) if n > 2 else 0.0
    try:
        eig = float(np.mean(list(nx.eigenvector_centrality_numpy(G).values())))
    except Exception:
        eig = 0.0
    return [dens, avg_deg, clus, avg_w, diam, betw, eig]
```

`TOPO = ["dens", "avg_deg", "clus", "avg_w", "diam", "betw", "eig"]`.

| # | Feature (code) | Metric | One-line meaning / formula |
|---|---|---|---|
| 1 | `dens` | Density | actual edges / maximum possible edges; overall interconnectedness (market integration). |
| 2 | `avg_deg` | Average Degree | mean node degree; average number of co-movement links per stock. |
| 3 | `clus` | Average Clustering Coefficient | mean fraction of a node's neighbours that are themselves linked; local herding / sector cohesion. |
| 4 | `avg_w` | Average Weight | mean `|corr|` over the edges that pass the threshold; average strength of retained links. |
| 5 | `diam` | Diameter | longest shortest path in the largest connected component; network spread / fragmentation. |
| 6 | `betw` | Average Betweenness Centrality | mean node betweenness; extent to which nodes act as bridges on shortest paths (contagion pathways). |
| 7 | `eig` | Average Eigenvector Centrality | mean eigenvector centrality; influence of nodes via connection to other well-connected nodes (core hubs). |

These seven metrics are GLOBAL: each is a single scalar per day describing the whole market network. There is one 7-vector per day, and it is broadcast IDENTICALLY to every stock on that day (Section 6).

---

## 6. Feature integration (matching the code)

The topology 7-vector is merged onto every ticker frame by date and then fed to the GBM alongside the own-history block. Quoting `scripts/eda/complex_network_gbm.py::main`:

```python
# scripts/eda/complex_network_gbm.py
    F = build_topo(frames, market)
    Fr = F.reset_index().rename(columns={"index": "date"})
    for tk in list(frames):
        frames[tk] = frames[tk].merge(Fr, on="date", how="left")
    ...
    for h in (1, 5, 10, 22):
        a = FM.panel(frames, {}, h)
        for c in TOPO:
            a[c] = a[c].ffill().fillna(0.0) if c in a else 0.0
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        models = {"GBM": FM.OWN, "GBM+topo": FM.OWN + TOPO}
```

- **OWN block (9 features).** From `full_matrix.py`: `OWN = HAR + ["rq", "mr_change", "mr_slope5", "mr_slope10", "mr_dev5", "mr_z22"]`, where `HAR = ["har_daily", "har_weekly", "har_monthly"]`. These are own-history only (no market or volume scalar).
- **GBM (baseline):** gamma `HistGradientBoostingRegressor` on the 9 OWN features.
- **GBM+topo (treatment):** the same GBM on OWN + the 7 topology features = **16-dim** input. The topology block is identical across stocks on a given date; the model must find incremental value from a market-level regime signal.
- **Ensemble and floor.** Predictions are averaged over 3 seeds (`FM.SEEDS = (0, 1, 2)`) and floored at `FL = QLIKE_FLOOR = 1e-8`.

The estimator, quoted from `scripts/eda/full_matrix.py::gbm`:

```python
# scripts/eda/full_matrix.py
def gbm(tr, te, cols, seed):
    m = HistGradientBoostingRegressor(loss="gamma", max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                      l2_regularization=1.0, random_state=seed)
    m.fit(tr[cols].to_numpy(float), np.maximum(tr["y"].to_numpy(float), FL))
    return np.maximum(m.predict(te[cols].to_numpy(float)), FL)
```

The gamma loss is chosen because gamma deviance equals QLIKE up to a constant, so the model is trained and scored on the same volatility objective.

---

## 7. Evaluation protocol (matching the code)

Quoting the walk-forward loop from `scripts/eda/complex_network_gbm.py::main`:

```python
# scripts/eda/complex_network_gbm.py
        models = {"GBM": FM.OWN, "GBM+topo": FM.OWN + TOPO}
        preds = {m: [] for m in models}; yy, dts = [], []
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]; te = a[(a.date >= ts) & (a.date < tend)]
            if len(te) == 0 or len(tr) < min_rows:
                continue
            for m, cols in models.items():
                preds[m].append(np.mean([FM.gbm(tr, te, cols, s) for s in FM.SEEDS], 0))
            yy.append(te["y"].to_numpy(float)); dts.append(te["date"].to_numpy())
        y = np.concatenate(yy); dates = np.concatenate(dts)
        e = {m: M.per_obs_qlike(y, np.concatenate(preds[m]), floor=FL) for m in models}
        q = {m: float(np.mean(e[m])) for m in models}
        p = ST.date_clustered_dm(e["GBM+topo"], e["GBM"], dates, h)["p_value"]
```

- **Expanding walk-forward folds.** `S1.FOLDS` (from `vn_gbm_graph_stage1.py`) are semi-annual boundaries `["2022-07-01", "2023-01-01", ..., "2026-01-01", "2100-01-01"]`; training always starts at `S1.TRAIN_START = "2015-01-01"` and expands. Each fold trains on `[TRAIN_START, ts - embargo)` and tests on `[ts, tend)`.
- **Target-horizon embargo.** A gap of `int(h * 1.6) + 5` days is removed at the train/test boundary so the shifted target `pk_{t+h}` from late-train rows cannot leak into the test window.
- **Minimum train rows.** `min_rows = 30000` (S&P 500) or `3000` (HOSE); folds with too little history or an empty test are skipped.
- **Pooled per-observation QLIKE.** `M.per_obs_qlike` (from `submission/soict_lstm_gat/metrics.py`) clamps both `y` and `p` to the shared floor `FL`, forms `r = y / p`, and returns `r - log(r) - 1` per observation. The reported number is the mean over all pooled test observations.
- **Date-clustered Diebold-Mariano.** `ST.date_clustered_dm` (from `baselines/2026-08-21_har_anchored_residual/code/stats.py`) collapses each loss series to one cross-sectional mean per unique date, then runs the HLN-corrected DM with the HAC lag set to `h - 1`. This removes the cross-sectional dependence of many tickers sharing a date. `mean_diff < 0` favours GBM+topo. We report `gain_pct = (QLIKE_GBM - QLIKE_GBM+topo) / QLIKE_GBM * 100` and the DM p-value at each `h ∈ {1, 5, 10, 22}`.
- **Markets.** Run on HOSE and S&P 500 (the driver takes the market as `sys.argv[1]`).

Output is written to `results/gamma_gbm/complex_network_<market>.json`.

---

## 8. Leakage controls and honest prior expectation

**Leakage controls (each also a review checkpoint):**

1. **Causal network windows.** Every correlation matrix uses only `iloc[i - WIN : i]`, strictly before the network date; topology is forward-filled, never back-filled.
2. **Train-only fold boundary.** The GBM trains on `[TRAIN_START, ts - embargo)` and is evaluated on the disjoint `[ts, tend)`; the embargo covers the horizon so the shifted target cannot cross the boundary.
3. **Shared positivity floor.** GBM and GBM+topo are scored with the identical `FL = 1e-8` clamp inside `per_obs_qlike`, so no model gains from a different floor.
4. **Identical basis for DM.** Both loss series are aggregated over the SAME `dates` array before the DM test.
5. **Broadcast is market-level, not cross-sectional leakage.** The 7 metrics are the same for every stock on a date and are built only from past windows, so a stock never sees its own future or another stock's contemporaneous target.

**Honest prior.** Market-level regime features have NOT beaten own-history on our per-stock QLIKE before: the market-factor / market-aggregate feature was NO-GO or unstable on HOSE in earlier runs (see `scripts/eda/full_matrix.py`'s `GBM+market` and `scripts/eda/vn_gbm_graph_stage1.py`'s `M1`). Prior probability that a broadcast market feature helps per-stock QLIKE is therefore low. What is new here is the specific combined return+volume, time-varying network topology (not a single market-volatility scalar): it encodes market synchronization / breadth in a way a single aggregate does not, so it is worth a rigorous test. The expectation is skeptical, and the outcome (help or NO-GO under DM) will be reported honestly after the advisor approves running it.

---

## 9. Pipeline figure

`docs/paper/figures/fig_complex_network_pipeline.png` (generator: `docs/paper/figures/generate_complex_network_pipeline.py`, matplotlib, dpi 180) shows the full pipeline:

Part A (market-level topology, causal): N-stock returns + volume over a trailing `ΔT = 66` day window -> return correlation matrix `ρ^R` and volume correlation matrix `ρ^V` -> blend `ρ_mix = α ρ^R + (1 - α) ρ^V` with `α = 0.7` -> Threshold network (`τ = 0.5`) -> 7 global metrics -> forward-fill to daily (recompute every `STEP = 22` days) and broadcast identically to all stocks.

Part B (per-stock GBM): the broadcast 7-vector is concatenated with the per-stock OWN(9) feature vector to form a 16-dim input -> gamma-GBM (3-seed ensemble, floor FL) -> Parkinson-variance forecast `pk_{i,t+h}` for `h ∈ {1, 5, 10, 22}` -> QLIKE and date-clustered DM of GBM+topo vs GBM. The legend states the dimensions: GBM uses `R^9` (OWN); GBM+topo uses `R^16` (OWN plus 7 topology). Both markets (HOSE and S&P 500) are trained.

---

## 10. Go / no-go for the advisor

Approve running `scripts/eda/complex_network_gbm.py` on HOSE and S&P 500 if the design above is acceptable, in particular:

- the window-size deviation (`WIN = 66`, `STEP = 22`) aligned to the project's 22-day month, versus the paper's 125/21;
- the volume proxy `volume_zscore_22` in place of raw log volume;
- Threshold-only network at `τ = 0.5`, `α = 0.7`;
- the incremental QLIKE + date-clustered DM test (GBM+topo vs GBM) as the decision metric, replacing the paper's `R^2` on the market series.

The experiment is NOT run in this document; it waits for advisor approval.
