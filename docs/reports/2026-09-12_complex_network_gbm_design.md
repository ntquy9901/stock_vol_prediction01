# Complex-Network Topology Features for Market-Index Volatility Forecasting: Technical Design

**Date:** 2026-09-12
**Status:** DESIGN-BEFORE-RUN (for advisor review; neither experiment is run yet)
**Scope:** Replicate the Complexity-2026 complex-network method faithfully (Experiment A: predict the future MARKET-INDEX volatility from network topology, exactly as the paper's Section 2.4 and Figure 1), then test whether the same topology features add incremental value to our per-stock daily gamma-GBM (Experiment B).
**Implementation under review:** `scripts/eda/complex_network_index.py` (Experiment A, to be written), `scripts/eda/complex_network_gbm.py` (Experiment B).
**Pipeline figure:** `docs/paper/figures/fig_complex_network_pipeline.png`

Source paper: N-K-K. Nguyen, H-T. Dinh, Q. Nguyen, "Complex Network Built From Stock Price Returns and Volumes to Predict Market Volatility and Volume," *Complexity*, 2026. DOI 10.1155/cplx/5670093. Framework figure: `docs/experement_guide/cplx5670093-fig-0001-m.jpg`.

---

## 1. Purpose and the two questions

The paper builds time-varying financial networks from a combined return-and-volume correlation matrix and shows that seven GLOBAL network-topology metrics predict the **future one-month VNIndex volatility** (the standard deviation of the index's daily log returns over the next 21 days), reaching `R^2 ≈ 0.563`. This document fixes every design choice BEFORE running so the advisor can approve or reject the protocol.

Two experiments, in priority order:

- **Experiment A (faithful replication of Section 2.4).** Reproduce the paper's actual task on our data: features = the 7 topology metrics of a rolling network; target = the future 21-day **market-index** volatility (and, as the paper also does, the future average index return and average log volume). Models = Linear Regression and Random Forest. Score = `R^2` and RMSE. Market index = the real **VNIndex** for HOSE and the **S&P 500 index** for the US. This answers the paper's own question on our data.

- **Experiment B (our thesis extension).** Take the same 7 topology metrics, broadcast them to every stock, and test whether they add INCREMENTAL value to our per-stock gamma-GBM that forecasts each stock's Parkinson variance. Score = pooled per-observation QLIKE + date-clustered Diebold-Mariano. This answers the stricter question our thesis cares about: does the paper's graph beat own-history on our loss of record?

The design mirrors the code exactly. Every deviation from the paper (window size, volume proxy, US index source) is stated with its rationale.

---

## 2. The paper's exact method (stated faithfully)

Reference: paper Section 2 (Methodology), pages 2-8; framework in Figure 1.

**Data (Section 2.1).** HOSE, 2015 to end of 2024, 1750 trading days. Liquidity filter: exclude stocks with average daily volume below 10,000 shares, then remove stocks with less than 70% trading-day availability; 279 stocks of 1679 survive. For each stock and day the paper records the log return `r_it = ln S_it - ln S_{i,t-1}` and the log volume `ln V_i(t)`. It also collects the daily close **VNIndex**, the market index of HOSE.

**Rolling window and network construction (Section 2.2).** A sliding window spans `ΔT = 125` trading days (about 6 months), shifted forward by `L = 21` trading days (about 1 month). Window `W_k = [t, t + ΔT - 1]` is aligned with the prediction period `[t + ΔT, t + ΔT + L]`. Within each window, for the N surviving stocks:

1. Compute the return correlation matrix `ρ^R` and the log-volume correlation matrix `ρ^V` (Pearson).
2. Blend them: `ρ_mix = α · ρ^R + (1 - α) · ρ^V`, `α ∈ [0, 1]` (paper eq. 1). The optimal weight for volatility is `α = 0.7`.
3. Build a network by one of three filters: **Threshold** (undirected edge if `|ρ_mix| ≥ τ`), **MST** (Kruskal on `d_ij = sqrt(2(1 - ρ_ij))`), or **Top-k** (`k = 5`). **Threshold is best for volatility.**

**Seven global topology metrics (Section 2.3).** Density, Average Degree, Average Clustering Coefficient, Average Weight (mean `|corr|` over edges), Diameter (largest connected component), Average Betweenness Centrality, Average Eigenvector Centrality.

**Future market-index targets (Section 2.4, the part this design now follows).** After building the network for window `W_k`, the paper extracts THREE market-level targets from the **VNIndex** over the next `L = 21` days `[t + ΔT, t + ΔT + L]`:

- (i) average index log return: `r_index = (1/L) Σ r_index(t + p)` (paper eq. 2);
- (ii) average log volume: `lnV_index = (1/L) Σ log V_index(t + p)` (paper eq. 3);
- (iii) **standard deviation of index log returns (volatility)**: `σ_r,index = sqrt( (1/(L-1)) Σ (r_index(t+p) - r_index)^2 )` (paper eq. 4).

Each rolling window therefore produces ONE supervised sample: a 7-metric feature vector and a 3-target output vector. Figure 1 shows exactly this: N-stock return and volume series plus the VNIndex feed the window; the network gives 7 metrics; the 7 metrics predict the 3 future VNIndex quantities.

**Models and evaluation (Section 2.5).** Linear Regression and Random Forest, walk-forward (train on past windows, test on the subsequent window, no look-ahead). Score: `R^2` (eq. 6) and RMSE (eq. 5).

**Headline result.** Threshold + Random Forest at `α = 0.7` gives volatility `R^2 ≈ 0.563`; volume `R^2 ≈ 0.95`; return `R^2` is modest (Top-k `α = 0.8` RF, `R^2 ≈ 0.56`, most settings 0.1-0.35). Mechanism claim (Section 3.6): surges in network density and centrality precede heightened index volatility, read as market synchronization where idiosyncratic risk is replaced by systemic risk.

---

## 3. Experiment A: faithful replication (predict the market-index volatility)

This is the paper's actual task on our data. It follows Figure 1 and Section 2.4 directly.

### 3.1 Data

| Role | HOSE | S&P 500 |
|---|---|---|
| Network nodes (stocks) | our HOSE processed panel (`daily_return`, `volume_zscore_22`), 300+ tickers | our S&P 500 processed panel, 498 tickers |
| **Market index (target series)** | **real VNINDEX** (`data/raw/prices/_market_index/vnindex.csv`, daily OHLC + volume, 2000-07-31 to 2026-09-11, 6,361 trading days) | **real S&P 500 index `^GSPC`** (`data/raw/prices/_market_index/gspc.csv`, Yahoo Finance, daily OHLC + Adj Close + volume, 2000-01-03 to 2026-09-11, 6,713 trading days) |

The VNINDEX target is real and covers the index's full history back to its inception week, exceeding the paper's 2015-2024 window. It is stitched from two independent sources with a documented cross-check: the Zenodo VN-Index daily dataset (DOI 10.5281/zenodo.21873555, CC BY 4.0) for 2000-07-31 to 2024-12-16, and the vnstock VCI feed for 2024-12-17 to 2026-09-11. The two sources overlap on 1,566 trading days, on which the daily Close agrees to a mean absolute difference of 0.0006% and a maximum of 0.29% (no day above 0.5%); the last session matches the reported market close of 1,795.21 on 2026-09-11. Full provenance and the discrepancy check are in `docs/reports/2026-09-12_vnindex_data_provenance.md` (builder `scripts/etl_vn_index/build_vnindex.py`). Volume display units differ across sources (tagged in `vol_source`) and feed only the secondary volume target, which is not stitched across the join; the headline volatility target is Close-based and unaffected.

The S&P 500 index target is the real `^GSPC` (Yahoo Finance, 2000-01-03 to 2026-09-11, 6,713 trading days, a single source, no stitch). It is cross-checked against the FRED `SP500` series on their 2,514-day overlap: Close agrees to a mean absolute difference of 0.00007% and a maximum of 0.12% (no day above 0.5%). Index volume is not comparable to a stock or ETF volume, so only OHLC is used (Close for the return-volatility target). Provenance in `data/raw/prices/_market_index/gspc_provenance.json` (builder `scripts/etl_vn_index/build_gspc.py`).

### 3.2 One supervised sample per window

For each window end `i` (stepping by `STEP` trading days over the common trading calendar):

- **Feature vector (7-dim):** `global_feats(combined)` where `combined = α ρ^R + (1 - α) ρ^V` is built from the stock returns and volumes over the trailing window `[i - WIN, i)` (strictly past). Same construction as Experiment B (Section 5, Section 6).
- **Target vector (3-dim), strictly future:** from the market index over `[i, i + L)`:
  - `idx_ret` = mean daily log return of the index;
  - `idx_lnvol` = mean daily `log(index volume)`;
  - `idx_vol` = standard deviation (ddof=1) of the index's daily log returns = the paper's volatility target (headline).

A window is emitted only if it has at least `WIN * 0.8` past rows and a full `L`-day future block; the last incomplete future block is dropped (no partial-horizon target).

### 3.3 Models and score (paper protocol)

- **Models:** `sklearn.linear_model.LinearRegression` and `sklearn.ensemble.RandomForestRegressor` (paper's LR + RF). RF hyperparameters are the project config defaults (`n_estimators`, `max_depth`, `min_samples_leaf` from `pipeline_config`; no test-set tuning). Features are standardized with a scaler fit on the training windows only.
- **Walk-forward:** expanding, one target at a time. Train on all windows strictly before a cutoff, test on the windows after it; slide the cutoff across the sample. Because a window's target ends at `i + L`, the training set stops at windows whose target end is at or before the test window's feature start (an `L`-day embargo between train targets and test features). Pooled `R^2` and RMSE are reported across the held-out test windows.
- **Headline configuration:** Threshold network, `α = 0.7`, `τ = 0.5` (the paper's best-for-volatility). A small `α` grid `{0.0, 0.5, 0.7, 1.0}` is reported as robustness (the paper's Table 1 structure), not as the headline.
- **Targets reported:** all three (`idx_vol` primary; `idx_ret` and `idx_lnvol` for completeness, as in the paper).

### 3.4 What a faithful result would and would not show

`R^2 ≈ 0.56` for `idx_vol` on VNIndex would replicate the paper. It would show topology explains a large share of the variance of the FUTURE INDEX volatility. It would NOT, on its own, show topology helps a strong per-stock forecaster (that is Experiment B). The sample is small (about 90 monthly windows over 2017-2024), so `R^2` is reported with the number of test windows and is not over-interpreted.

---

## 4. Graph construction (matching the code)

Quoting `scripts/eda/complex_network_gbm.py::build_topo` (shared by both experiments):

```python
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

Construction notes, all matching the paper except the window sizes (Section 7):

- **Two correlation matrices on a common ticker set.** Return matrix from `daily_return`; volume matrix from `volume_zscore_22`. Only tickers in BOTH windows enter the blend. A window needs at least 20 common tickers or it is skipped.
- **Volume proxy.** Our stand-in for the paper's log volume is `volume_zscore_22` (normalised volume). Deviation in the volume variable only; it still captures co-movement of trading activity.
- **Blend `α = 0.7`.** `combined = ALPHA * rc + (1 - ALPHA) * vc`, exactly paper eq. (1) at the paper's optimal `α` for volatility.
- **Threshold `τ = 0.5`.** Undirected edge wherever `|combined| ≥ 0.5` (best filter for volatility). MST and Top-k are the paper's alternatives; this design uses Threshold, as the paper found it best for volatility.
- **Causal.** A network dated `d0 = dates[i]` uses only rows `iloc[i - WIN : i]`, strictly before `d0`. In Experiment B the daily panel forward-fills these metrics; in Experiment A each window is a sample in its own right (no forward-fill needed).

---

## 5. The seven global metrics (matching the code)

Quoting `scripts/eda/complex_network_gbm.py::global_feats`:

```python
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

| # | Feature (code) | Metric | One-line meaning |
|---|---|---|---|
| 1 | `dens` | Density | actual edges / maximum possible edges; overall interconnectedness (market integration). |
| 2 | `avg_deg` | Average Degree | mean node degree; average number of co-movement links per stock. |
| 3 | `clus` | Average Clustering Coefficient | mean fraction of a node's neighbours that are themselves linked; local herding / sector cohesion. |
| 4 | `avg_w` | Average Weight | mean `|corr|` over retained edges; average strength of retained links. |
| 5 | `diam` | Diameter | longest shortest path in the largest connected component; network spread / fragmentation. |
| 6 | `betw` | Average Betweenness Centrality | mean node betweenness; nodes acting as bridges on shortest paths (contagion pathways). |
| 7 | `eig` | Average Eigenvector Centrality | mean eigenvector centrality; influence via connection to other well-connected nodes (core hubs). |

Each metric is one scalar per network (global). Experiment A uses the 7-vector directly as the sample's features; Experiment B broadcasts the 7-vector identically to every stock on that date.

---

## 6. Experiment B: per-stock gamma-GBM extension (secondary)

The topology 7-vector is merged onto every ticker frame by date and fed to the GBM alongside the own-history block. Quoting `scripts/eda/complex_network_gbm.py::main`:

```python
    F = build_topo(frames, market)
    Fr = F.reset_index().rename(columns={"index": "date"})
    for tk in list(frames):
        frames[tk] = frames[tk].merge(Fr, on="date", how="left")
    for h in (1, 5, 10, 22):
        a = FM.panel(frames, {}, h)
        for c in TOPO:
            a[c] = a[c].ffill().fillna(0.0) if c in a else 0.0
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        models = {"GBM": FM.OWN, "GBM+topo": FM.OWN + TOPO}
```

- **OWN block (9 features).** `OWN = HAR + ["rq", "mr_change", "mr_slope5", "mr_slope10", "mr_dev5", "mr_z22"]`, `HAR = ["har_daily", "har_weekly", "har_monthly"]`; own-history only.
- **GBM (baseline):** gamma `HistGradientBoostingRegressor` on the 9 OWN features.
- **GBM+topo (treatment):** the same GBM on OWN + 7 topology = **16-dim** input; the topology block is identical across stocks on a date.
- **Ensemble and floor:** 3 seeds `(0, 1, 2)`, floored at `FL = 1e-8`.
- **Score:** pooled per-observation QLIKE + date-clustered Diebold-Mariano at `h ∈ {1, 5, 10, 22}`, on HOSE and S&P 500. `gain_pct = (QLIKE_GBM - QLIKE_GBM+topo) / QLIKE_GBM * 100`.

The gamma loss equals QLIKE up to a constant, so the model is trained and scored on the same volatility objective. Output: `results/gamma_gbm/complex_network_<market>.json`.

---

## 7. Window-size deviation (documented, applies to both experiments)

Our implementation uses `WIN = 66` and `STEP = 22`, NOT the paper's `ΔT = 125` / `L = 21`.

```python
WIN = 66          # trailing window ΔT = 3 months at 22 trading days/month (project month=22 convention;
#                   paper used 125 ~= 6 months)
STEP = 22         # recompute + slide by one month = 22 trading days
THR = 0.5         # threshold tau on |combined corr| (paper "Threshold" network; best for volatility)
ALPHA = 0.7       # combined = ALPHA*return-corr + (1-ALPHA)*log-volume-corr (paper's optimal alpha=0.7)
```

**Rationale.** The project's convention is 1 month = 22 trading days (`har_monthly`, `volume_zscore_22` both use 22). We therefore set the trailing correlation window to `ΔT = 3 months = 66 trading days` and slide by one month = `22 trading days`. For Experiment A the future target block is `L = 22` days (one project-month), replacing the paper's 21. Everything else (`α = 0.7`, `τ = 0.5`, the 7 metrics, combined return+volume correlation, causal construction) matches the paper.

**Robustness row.** Because the paper's window is 6 months, Experiment A will also report `WIN = 132` (`6 months × 22`) so the advisor can compare our 3-month choice against a faithful 6-month window on the same data. The headline stays `WIN = 66` per the project convention unless the advisor prefers 132.

---

## 8. Evaluation protocol summary

| | Experiment A (faithful) | Experiment B (extension) |
|---|---|---|
| Sample | one per window (about 90 monthly windows, 2017-2024 HOSE) | one per ticker-day (pooled panel) |
| Features | 7 topology metrics | 9 OWN + 7 topology (16) |
| Target | future 22-day market-index vol / return / log-volume | per-stock `pk_{i,t+h}`, `h ∈ {1,5,10,22}` |
| Models | LinearRegression, RandomForest | gamma HistGradientBoosting (3 seeds) |
| Score | `R^2`, RMSE (paper) | pooled QLIKE + date-clustered DM |
| Walk-forward | expanding, `L`-day embargo between train targets and test features | expanding semi-annual folds, `int(h*1.6)+5`-day embargo |
| Markets | HOSE (real VNIndex), S&P 500 (index/proxy) | HOSE, S&P 500 |
| Output | `results/gamma_gbm/complex_network_index_<market>.json` | `results/gamma_gbm/complex_network_<market>.json` |

---

## 9. Leakage controls and honest prior expectation

**Leakage controls (each a review checkpoint):**

1. **Causal network windows.** Every correlation matrix uses only rows strictly before the network date.
2. **Strictly-future targets.** Experiment A's target spans `[i, i + L)`, entirely after the feature window `[i - WIN, i)`; Experiment B shifts `pk` by `+h` and embargoes the fold boundary.
3. **Train-only scaling and fitting.** Feature scaler and both models fit on training windows/rows only.
4. **Shared positivity floor (Experiment B).** GBM and GBM+topo scored with the identical `FL = 1e-8` clamp.
5. **Identical basis for DM (Experiment B).** Both loss series aggregated over the same `dates` array before the test.
6. **Market-level broadcast, not cross-sectional leakage (Experiment B).** The 7 metrics are the same for every stock on a date and built only from past windows.

**Honest prior.**

- *Experiment A* is a genuine replication. The paper reports `R^2 ≈ 0.56` for VNIndex volatility with the real index, 2015-2024, `ΔT = 125`. On 2017-2024 with `ΔT = 66` and about 90 windows the `R^2` may come out lower; whatever it is will be reported with the test-window count, not oversold.
- *Experiment B* is skeptical. Market-level regime features have not beaten own-history on our per-stock QLIKE before (`GBM+market` and `vn_gbm_graph_stage1` M1 were NO-GO or unstable on HOSE). What is new is the specific combined return+volume time-varying topology, so a rigorous DM test is warranted. The expected outcome is NO-GO on QLIKE, and it will be reported honestly.

---

## 10. Pipeline figure

`docs/paper/figures/fig_complex_network_pipeline.png` (generator `generate_complex_network_pipeline.py`, matplotlib, dpi 180) will be updated to show BOTH branches:

- **Part A (faithful, Section 2.4):** N-stock returns + volume over a trailing `ΔT = 66` window plus the market index -> `ρ^R`, `ρ^V` -> blend `ρ_mix` (`α = 0.7`) -> Threshold network (`τ = 0.5`) -> 7 metrics -> predict the future 22-day index volatility / return / log-volume (LR, RF; `R^2`, RMSE).
- **Part B (extension):** the same 7-vector, broadcast to all stocks, concatenated with per-stock OWN(9) -> 16-dim gamma-GBM -> `pk_{i,t+h}` -> QLIKE + date-clustered DM (GBM+topo vs GBM).

---

## 11. Go / no-go for the advisor

Approve running both experiments if the design is acceptable, in particular:

- **Experiment A now follows Section 2.4 and Figure 1**: features = 7 topology metrics, target = future market-index volatility (real VNIndex for HOSE, S&P 500 index for the US), models = LR + RF, score = `R^2` + RMSE.
- the window-size deviation (`WIN = 66`, `STEP = 22`, `L = 22`) aligned to the 22-day month, with `WIN = 132` reported as a 6-month robustness row;
- the volume proxy `volume_zscore_22` in place of raw log volume;
- Threshold-only network at `τ = 0.5`, `α = 0.7` (with an `α` grid as robustness);
- the S&P 500 index source (real `^GSPC` if the feed is available, else a documented panel proxy);
- Experiment B's incremental QLIKE + date-clustered DM as the decision metric for the per-stock question.

Neither experiment is run in this document; both wait for advisor approval.
