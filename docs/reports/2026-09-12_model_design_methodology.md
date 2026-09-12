# Model Design and Methodology: Multi-Horizon Parkinson-Variance Forecasting

**Date:** 2026-09-12
**Scope:** Detailed design of how the paper's models are built, trained, validated, and tested. All code blocks below are quoted verbatim from the project source files (path given as a comment header on each block); no code is paraphrased.

---

## 1. Overview

The task is multi-horizon forecasting of daily **Parkinson variance**. For a trading day `t` with intraday high `H` and low `L`, the target series is

```
pk_t = ln(H/L)^2 / (4 ln 2)
```

This quantity is a **variance** (squared units), not a standard deviation. The `parkinson_variance` column on each enriched per-ticker CSV holds `pk_t`. The forecast target at horizon `h` is the future value `pk_{t+h}` (see the `panel` builder below, `y = parkinson_variance.shift(-h)`).

**Horizons:** `h in {1, 5, 10, 22}` trading days (daily, weekly, biweekly, monthly).

**Markets:** two equity universes are evaluated independently.

- **S&P 500** (497 tickers), enriched CSVs under `data/processed_enriched/sp500_clean/`.
- **HOSE** (Ho Chi Minh Stock Exchange, 405 tickers), enriched CSVs under `data/processed_enriched/hose/`.

**Models compared:**

| Name | Estimator | Feature set |
|------|-----------|-------------|
| HAR | OLS | 3 HAR lags |
| HARQ | OLS + RQ interaction | 3 HAR lags + realized-quarticity interaction |
| GBM | gamma HistGradientBoostingRegressor | OWN (9 own-history features) |
| GBME | gamma HistGradientBoostingRegressor | OWN + EARN (4 earnings features) |
| GBME+graph | gamma HistGradientBoostingRegressor | OWN + EARN + correlation-graph neighbour aggregate `g` |

**GBM definition.** GBM here is scikit-learn's `HistGradientBoostingRegressor` configured with the **gamma deviance loss** (`loss="gamma"`). The gamma deviance equals QLIKE up to an additive constant, so the training objective is aligned with the reported evaluation loss. This is why GBM is trained to minimize an objective in the same family as the metric it is scored on.

Earnings features (EARN) apply to S&P 500 natively and to HOSE via real crawled announcement dates (`hose_earnings_combined.parquet`). Vietnam has no per-firm scheduled-date coverage on the primary vendor, so HOSE earnings are partial (dense only for 2025-2026); the code injects them where available.

---

## 2. Data organization (train / validation / test)

### 2.1 Per-ticker enriched CSVs

Each ticker is one CSV under `data/processed_enriched/<market>/`. Files ending in `_rejections` are skipped. Each CSV is parsed with a `date` column, sorted chronologically, and passed through the feature builder `_feat` (Section 3). This is the loader in `full_matrix.py`:

```python
# scripts/eda/full_matrix.py
def load(market):
    if market == "sp500":
        sect = json.load(open(REPO / "results" / "gamma_gbm" / "sp500_sectors.json"))
        d = "sp500_clean"
    else:
        sect = pd.read_csv(S1.SECT).set_index("symbol")["industry_code"].to_dict(); d = market
    frames = {}
    for p in glob.glob(str(REPO / "data" / "processed_enriched" / d / "*.csv")):
        tk = Path(p).stem
        if tk.endswith("_rejections"):
            continue
        fr = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        frames[tk] = S1._feat(fr).assign(ticker=tk, sector=sect.get(tk, -1))
    edates = {}
    if market == "sp500":
        e = pd.read_parquet(REPO / "results" / "gamma_gbm" / "sp500_earnings.parquet")
        edates = {tk: np.sort(g["earnings_date"].to_numpy()) for tk, g in e.groupby("ticker")}
    return frames, sect, edates
```

### 2.2 Pooled panel

All tickers are stacked into one long panel (all ticker-days). The target is created per ticker as `pk.shift(-h)` before stacking, so the horizon shift is causal and per-ticker (no cross-ticker leakage of the future). Rows with any missing OWN feature or missing `y` are dropped:

```python
# scripts/eda/full_matrix.py
def panel(frames, edates, h):
    rows = []
    for tk, d in frames.items():
        e = d.copy(); e["y"] = e["parkinson_variance"].shift(-h)
        if edates:
            T = e["date"].shift(-h).to_numpy(); nxt, prv = _signed(T, edates.get(tk)); dist = np.minimum(nxt, prv)
            e["earn_prox"] = np.maximum(0.0, 1.0 - dist / WK); e["earn_soon"] = (dist <= 3).astype(float)
            e["earn_pre"] = np.maximum(0.0, 1.0 - nxt / WK); e["earn_post"] = np.maximum(0.0, 1.0 - prv / (2 * WK))
        rows.append(e)
    a = pd.concat(rows, ignore_index=True)
    return a.dropna(subset=OWN + ["y"]).reset_index(drop=True)
```

### 2.3 Expanding-window walk-forward folds

The evaluation is an **expanding-window walk-forward**. Fold boundaries and the training start date are constants in `vn_gbm_graph_stage1.py`:

```python
# scripts/eda/vn_gbm_graph_stage1.py
TRAIN_START = "2015-01-01"
FOLDS = ["2022-07-01", "2023-01-01", "2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01",
         "2025-07-01", "2026-01-01", "2100-01-01"]
TOPK, RNG_SEED = 10, 20260910
```

There are `len(FOLDS) - 1 = 8` test folds. Fold `k` has test window `[FOLDS[k], FOLDS[k+1])`; the training window always starts at `TRAIN_START` and grows (expands) as `k` increases. The eight test windows are:

| Fold `k` | Test start `ts` | Test end `tend` |
|----------|-----------------|-----------------|
| 0 | 2022-07-01 | 2023-01-01 |
| 1 | 2023-01-01 | 2023-07-01 |
| 2 | 2023-07-01 | 2024-01-01 |
| 3 | 2024-01-01 | 2024-07-01 |
| 4 | 2024-07-01 | 2025-01-01 |
| 5 | 2025-01-01 | 2025-07-01 |
| 6 | 2025-07-01 | 2026-01-01 |
| 7 | 2026-01-01 | 2100-01-01 (open-ended, all remaining) |

### 2.4 Target-horizon embargo

Between the training window end and the test window start there is a horizon-dependent **embargo** of `int(h*1.6) + 5` calendar days. Because the target is `pk_{t+h}`, a training row dated near `ts` would otherwise have its label reach into the test window; the embargo removes that overlap. The training slice ends at `ts - embargo`:

```python
# scripts/eda/full_matrix.py
for h in (1, 5, 10, 22):
    a = panel(frames, edates, h); embargo = pd.Timedelta(days=int(h * 1.6) + 5)
    tickers = sorted(a["ticker"].unique()); n = len(tickers); Wu = _uniform(n)
    preds = {m: [] for m in ["HAR", "HARQ"] + list(GMODELS)}; yy, dts = [], []
    for k in range(len(S1.FOLDS) - 1):
        ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
        tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]; te = a[(a.date >= ts) & (a.date < tend)]
        if len(te) == 0 or len(tr) < (30000 if market == "sp500" else 3000):
            continue
```

Embargo widths: `h=1 -> 6 days`, `h=5 -> 13 days`, `h=10 -> 21 days`, `h=22 -> 40 days`.

### 2.5 Per-fold minimum-rows guard

A fold is skipped if the test slice is empty or the training panel has fewer than `30000` rows for S&P 500 (`3000` for HOSE / VN). This prevents fitting on a too-thin early expanding window. The guard is the `if len(te) == 0 or len(tr) < (30000 if market == "sp500" else 3000): continue` line above.

### 2.6 Validation policy (important)

The gamma-GBM models use **fixed hyperparameters**: no early stopping, no separate held-out validation split, no hyperparameter search. The `HistGradientBoostingRegressor` is fit once per (fold, seed) with constant settings (Section 4). The **walk-forward test folds themselves are the out-of-sample evaluation**. There is no invented held-out validation set for the tree models; model selection is not performed inside the loop.

For completeness: an earlier GNNHAR line of work used a trailing 22-day validation slice for early stopping. That mechanism is **separate and is not part of the reframed paper's GBM/HAR/HARQ models** described here.

### 2.7 ASCII timeline (expanding folds + embargo)

```
TRAIN_START=2015-01-01
|============================ TRAIN (expands) ============================|--embargo--|=== TEST fold0 ===|
2015 ................................................. 2022-07-01 - (h*1.6+5d)          [2022-07-01, 2023-01-01)

|============================ TRAIN (expands further) ====================================|--embargo--|=== TEST fold1 ===|
2015 ....................................................... 2023-01-01 - (h*1.6+5d)                  [2023-01-01, 2023-07-01)

... one fold per 6-month block through 2026-01-01, final fold open-ended to 2100-01-01 ...

Legend:  TRAIN = [2015-01-01, ts - embargo)     TEST = [ts, tend)     embargo = int(h*1.6)+5 calendar days
```

---

## 3. Feature construction

### 3.1 Own-history features (OWN, 9 features)

`OWN` for S&P 500 is defined in `full_matrix.py` as the three HAR lags plus six log-vol / quarticity descriptors:

```python
# scripts/eda/full_matrix.py
FL = S1.FL
HAR = ["har_daily", "har_weekly", "har_monthly"]
OWN = HAR + ["rq", "mr_change", "mr_slope5", "mr_slope10", "mr_dev5", "mr_z22"]   # own-history, no market/volume
EARN = ["earn_prox", "earn_soon", "earn_pre", "earn_post"]
SEEDS = (0, 1, 2)
PLAC_SEEDS = (11, 12, 13)
WK = 5
```

The HAR lags (`har_daily`, `har_weekly`, `har_monthly`) are rolling means of `pk` over 1, 5, and 22 trading days respectively, precomputed on the enriched CSV. The remaining six OWN descriptors are built in `_feat` (`vn_gbm_graph_stage1.py`), which also produces the log-vol series `logpk` used for the graph:

```python
# scripts/eda/vn_gbm_graph_stage1.py
def _feat(d):
    pk = d["parkinson_variance"].to_numpy(float); lpk = pd.Series(np.log(np.maximum(pk, FL)), index=d.index)
    d["logpk"] = lpk
    d["rq"] = np.sqrt(pd.Series(pk ** 2, index=d.index).rolling(WK, min_periods=1).mean())
    d["mr_change"] = lpk.diff(1); d["mr_slope5"] = (lpk - lpk.shift(WK)) / WK
    d["mr_slope10"] = (lpk - lpk.shift(2 * WK)) / (2 * WK); d["mr_dev5"] = lpk - lpk.rolling(WK).mean()
    d["mr_z22"] = (lpk - lpk.rolling(MO).mean()) / (lpk.rolling(MO).std() + FL)
    return d
```

with `WK, MO = 5, 22`. Feature formulas (all observable at origin `t`, causal, using only history up to `t`):

| Feature | Formula | Meaning |
|---------|---------|---------|
| `har_daily` | mean of `pk` over last 1 day | HAR daily component |
| `har_weekly` | mean of `pk` over last 5 days | HAR weekly component |
| `har_monthly` | mean of `pk` over last 22 days | HAR monthly component |
| `rq` | `sqrt(rolling_mean(pk^2, 5))` | realized-quarticity proxy |
| `mr_change` | `logpk_t - logpk_{t-1}` | 1-day log-vol change |
| `mr_slope5` | `(logpk_t - logpk_{t-5}) / 5` | 5-day log-vol slope |
| `mr_slope10` | `(logpk_t - logpk_{t-10}) / 10` | 10-day log-vol slope |
| `mr_dev5` | `logpk_t - rolling_mean(logpk, 5)` | deviation from 5-day mean |
| `mr_z22` | `(logpk_t - rolling_mean(logpk, 22)) / (rolling_std(logpk, 22) + FL)` | 22-day z-score |

Note the VN pipeline (`vn_gbm_graph_stage1.py`) uses a slightly different `OWN` list that also includes `volume_zscore_22`; the SP500 paper `OWN` (from `full_matrix.py`, quoted above) does not, keeping OWN strictly to own-history vol descriptors.

### 3.2 External earnings features (EARN, 4 features)

Earnings features are built from the **signed distance to the nearest scheduled announcement date**. `_signed` returns, for each date, the days to the next announcement (`nxt`) and the days since the previous one (`prv`); missing coverage returns a large sentinel `1e9`:

```python
# scripts/eda/full_matrix.py
def _signed(T, ed):
    n = len(T)
    if ed is None or len(ed) == 0:
        return np.full(n, 1e9), np.full(n, 1e9)
    tv = pd.DatetimeIndex(pd.to_datetime(T)).asi8 / 86_400_000_000_000.0
    ev = np.sort(pd.DatetimeIndex(pd.to_datetime(ed)).asi8 / 86_400_000_000_000.0)
    idx = np.searchsorted(ev, tv, side="left")
    nxt = np.where(idx < len(ev), ev[np.clip(idx, 0, len(ev) - 1)] - tv, 1e9)
    prv = np.where(idx > 0, tv - ev[np.clip(idx - 1, 0, len(ev) - 1)], 1e9)
    return np.maximum(nxt, 0.0), np.maximum(prv, 0.0)
```

The four EARN columns are then derived in `panel` (quoted in Section 2.2) with `WK = 5`:

| Feature | Formula | Meaning |
|---------|---------|---------|
| `earn_prox` | `max(0, 1 - dist/5)` where `dist = min(nxt, prv)` | proximity to nearest announcement |
| `earn_soon` | `1[dist <= 3]` | binary: announcement within 3 days |
| `earn_pre` | `max(0, 1 - nxt/5)` | ramp before an announcement |
| `earn_post` | `max(0, 1 - prv/10)` | decay after an announcement |

The distances are measured at the **target date** `T = date.shift(-h)` so the earnings signal aligns with the day being predicted, and are computed against **real announcement dates**, not filing deadlines.

### 3.3 Cross-firm graph feature (g)

The graph feature is a neighbour aggregate under a **correlation top-k adjacency** estimated on training data only. `build_graph` computes the correlation of the standardized TRAIN `logpk` panel, keeps the top-10 positive neighbours per node, and row-normalizes:

```python
# scripts/eda/vn_gbm_graph_stage1.py
def _std_cols(df):
    z = (df - df.mean()) / df.std().replace(0, np.nan)
    return z.fillna(0.0).to_numpy(float), z.notna().to_numpy(float)


def build_graph(train, tickers, rng):
    """per-fold static vol-correlation top-k graph on TRAIN logpk; returns real+placebo row-normalised weight
    matrices W[i,j] (neighbour j of node i), positive-only."""
    piv = train.pivot_table(index="date", columns="ticker", values="logpk").reindex(columns=tickers)
    X, Xm = _std_cols(piv)
    corr = (X.T @ X) / np.maximum(Xm.T @ Xm, 1.0); np.fill_diagonal(corr, -np.inf)
    n = len(tickers); W = np.zeros((n, n)); Wp = np.zeros((n, n))
    for i in range(n):
        top = np.argsort(corr[i])[::-1][:TOPK]
        W[i, top] = np.clip(corr[i, top], 0.0, None)
        rand = rng.choice(np.delete(np.arange(n), i), size=min(TOPK, n - 1), replace=False)
        Wp[i, rand] = 1.0
    W /= np.maximum(np.abs(W).sum(1, keepdims=True), 1e-12)
    Wp /= np.maximum(np.abs(Wp).sum(1, keepdims=True), 1e-12)
    return W, Wp
```

The neighbour aggregate `g` is then, for row `(i, t)`, `g[i,t] = sum_j W_ij * pk[j,t]`. This is computed by pivoting `parkinson_variance` to a `[date x ticker]` matrix and multiplying by `W.T`, with NaNs filled by the cross-sectional mean:

```python
# scripts/eda/full_matrix.py
def nb(fold, tickers, W):
    piv = fold.pivot_table(index="date", columns="ticker", values="parkinson_variance").reindex(columns=tickers).sort_index()
    V = piv.to_numpy(float); Vf = np.nan_to_num(np.where(np.isnan(V), np.nanmean(V, axis=1, keepdims=True), V))
    NB = Vf @ W.T; dpos = {d: i for i, d in enumerate(piv.index)}; cpos = {c: j for j, c in enumerate(tickers)}
    return NB[fold["date"].map(dpos).to_numpy(), fold["ticker"].map(cpos).to_numpy()]
```

A uniform-adjacency market aggregate (all off-diagonal weights equal) is available for the market-mean control:

```python
# scripts/eda/full_matrix.py
def _uniform(n):
    return (np.ones((n, n)) - np.eye(n)) / (n - 1)
```

**Leakage-safety of the graph.** The adjacency `W` is estimated with `build_graph(tr, ...)` where `tr` is the training slice only (`fold.date < ts - embargo`), and it is rebuilt per fold with a fold-dependent RNG (`np.random.default_rng(S1.RNG_SEED + k)`). The neighbour aggregate at `(i,t)` uses only contemporaneous observed `pk[j,t]` values (same day `t`, not future), so `g` is observable at origin `t`.

---

## 4. Models (with code)

### 4.1 HAR (OLS on 3 lags)

HAR is ordinary least squares on the three HAR lags with an intercept. Targets are floored at `FL` before fitting; predictions are floored at `1%` of the mean training target:

```python
# scripts/eda/full_matrix.py
def _har_ols(tr, te):
    x = lambda df: np.column_stack([np.ones(len(df)), df[HAR].to_numpy(float)])  # noqa: E731
    c = np.linalg.lstsq(x(tr), np.maximum(tr["y"].to_numpy(float), FL), rcond=None)[0]
    return np.maximum(x(te) @ c, 0.01 * np.maximum(tr["y"], FL).mean())
```

### 4.2 HARQ (OLS + RQ interaction)

HARQ augments the HAR design matrix with an interaction between the daily HAR term and the realized-quarticity proxy `rq`:

```python
# scripts/eda/full_matrix.py
def _harq_ols(tr, te):
    def dm(df):
        x = df[HAR].to_numpy(float); return np.column_stack([np.ones(len(x)), x, x[:, 0] * df["rq"].to_numpy(float)])
    c = np.linalg.lstsq(dm(tr), np.maximum(tr["y"].to_numpy(float), FL), rcond=None)[0]
    nf = 0.01 * np.maximum(tr["y"], FL).mean()
    return np.maximum(dm(te) @ c, nf)
```

The design matrix columns are: intercept, `har_daily`, `har_weekly`, `har_monthly`, and `har_daily * rq`.

### 4.3 GBM / GBME / GBME+graph (gamma HistGradientBoostingRegressor)

All three tree models share one fit function; only the `cols` (feature column list) differs. The gamma-loss configuration is fixed:

```python
# scripts/eda/full_matrix.py
def gbm(tr, te, cols, seed):
    m = HistGradientBoostingRegressor(loss="gamma", max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                      l2_regularization=1.0, random_state=seed)
    m.fit(tr[cols].to_numpy(float), np.maximum(tr["y"].to_numpy(float), FL))
    return np.maximum(m.predict(te[cols].to_numpy(float)), FL)
```

Hyperparameters (identical for every tree model, no tuning): `loss="gamma"`, `max_iter=300`, `learning_rate=0.05`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `random_state=seed`. Predictions are floored at `FL = 1e-8`.

Feature-column sets, from the model registry in `full_matrix.main`:

```python
# scripts/eda/full_matrix.py
GMODELS = {"GBM": OWN, "GBM+market": OWN + ["g_market"], "GBM+graph": OWN + ["g_corr"],
           "GBM+plac": OWN + ["g_plac"]}
if has_earn:
    GMODELS.update({"GBM+earn": OWN + EARN, "GBM+earn+graph": OWN + EARN + ["g_corr"]})
```

Mapping to the paper names:

| Paper name | Registry key | Feature columns |
|------------|--------------|-----------------|
| GBM | `GBM` | `OWN` (9) |
| GBME | `GBM+earn` | `OWN + EARN` (13) |
| GBME+graph | `GBM+earn+graph` | `OWN + EARN + ["g_corr"]` (14) |

`g_corr` is the correlation-graph neighbour aggregate; `g_market` (uniform) and `g_plac` (degree-matched random edges) are controls, not headline models.

---

## 5. Training

Each tree model is fit once per fold per seed. The `gbm` fit call (Section 4.3) trains on the fold's training slice `trf` with targets floored at `FL`. Three seeds `SEEDS = (0, 1, 2)` are averaged into a **seed ensemble**; the mean prediction is what enters the metric:

```python
# scripts/eda/full_matrix.py
for mdl, cols in GMODELS.items():
    preds[mdl].append(np.mean([gbm(trf, tef, cols, s) for s in SEEDS], 0))   # seed-ensemble
```

Training properties:

- **Objective:** gamma deviance (`loss="gamma"`), which equals QLIKE up to an additive constant, so the fitting loss and the reported loss are in the same family.
- **QLIKE floor `FL = 1e-8`:** applied to training targets (`np.maximum(tr["y"], FL)`) and to predictions (`np.maximum(m.predict(...), FL)`); the same floor is later used in the metric so the positivity basis is identical across models.
- **No early stopping / no validation split:** fixed `max_iter=300`; the walk-forward test folds are the only out-of-sample check.
- **Seed ensemble:** mean over 3 seeds reduces boosting variance; per-seed spread is reported separately in the driver output.

HAR and HARQ are deterministic OLS fits (no seeds).

---

## 6. Validation and testing

Testing is **pooled per-observation QLIKE over the concatenated test folds**. For each horizon, each model's per-fold predictions are concatenated across all folds, then scored against the concatenated targets. The full driver loop ties data, folds, features, models, and scoring together:

```python
# scripts/eda/full_matrix.py
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]; te = a[(a.date >= ts) & (a.date < tend)]
            if len(te) == 0 or len(tr) < (30000 if market == "sp500" else 3000):
                continue
            Wc, _ = S1.build_graph(tr, tickers, np.random.default_rng(S1.RNG_SEED + k))
            fold = a[(a.date >= S1.TRAIN_START) & (a.date < tend)].copy()
            fold["g_market"] = nb(fold, tickers, Wu); fold["g_corr"] = nb(fold, tickers, Wc)
            plc = []
            for ps in PLAC_SEEDS:
                rng = np.random.default_rng(ps + k); Wp = np.zeros((n, n))
                for i in range(n):
                    j = rng.choice(np.delete(np.arange(n), i), size=min(S1.TOPK, n - 1), replace=False); Wp[i, j] = 1.0 / len(j)
                plc.append(nb(fold, tickers, Wp))
            fold["g_plac"] = np.mean(plc, 0)
            for c in ["g_market", "g_corr", "g_plac"]:
                fold[c] = fold[c].fillna(0.0)
            trf = fold[(fold.date >= S1.TRAIN_START) & (fold.date < ts - embargo)]; tef = fold[(fold.date >= ts) & (fold.date < tend)]
            preds["HAR"].append(_har_ols(trf, tef)); preds["HARQ"].append(_harq_ols(trf, tef))
            for mdl, cols in GMODELS.items():
                preds[mdl].append(np.mean([gbm(trf, tef, cols, s) for s in SEEDS], 0))   # seed-ensemble
            yy.append(tef["y"].to_numpy(float)); dts.append(tef["date"].to_numpy())
        y = np.concatenate(yy); dates = np.concatenate(dts)
        e = {m: M.per_obs_qlike(y, np.concatenate(preds[m]), floor=FL) for m in preds}
        q = {m: float(np.mean(e[m])) for m in e}
```

The all-metrics variant (`paper_metrics_sp500.py`) reports MSE / RMSE / MAE / R2 / QLIKE per model, and is where HOSE earnings are injected from the crawled parquet:

```python
# scripts/eda/paper_metrics_sp500.py
def all_metrics(y, p):
    return {"mse": M.mse(y, p), "rmse": M.rmse(y, p), "mae": M.mae(y, p),
            "r2": M.r2(y, p), "qlike": float(np.mean(M.per_obs_qlike(y, p, floor=FL)))}
```

```python
# scripts/eda/paper_metrics_sp500.py
    frames, sect, edates = FM.load(market)
    if market != "sp500":                                           # inject REAL crawled VN announcement dates
        _ep = REPO / "results" / "gamma_gbm" / "hose_earnings_combined.parquet"
        if _ep.exists():
            _e = pd.read_parquet(_ep)
            edates = {tk: np.sort(g["earnings_date"].to_numpy()) for tk, g in _e.groupby("ticker")}
    has_earn = bool(edates)                                          # VN earnings partial (dense 2025-26)
```

### 6.1 Significance: date-clustered Diebold-Mariano

Significance between two models is a **date-clustered Diebold-Mariano** test. Because the panel has many tickers per date, per-observation loss differentials are cross-sectionally dependent; the test first collapses the differential to one value per trading date (cross-sectional mean), then runs the HLN-corrected DM on the date-level series:

```python
# baselines/2026-08-21_har_anchored_residual/code/stats.py
def date_clustered_dm(
    loss_a: np.ndarray, loss_b: np.ndarray, dates: np.ndarray, h: int
) -> dict[str, float]:
    """Date-clustered Diebold-Mariano on a dependent panel.

    Aggregates each loss series to one value per unique date (cross-sectional mean), then runs the
    reused HLN Diebold-Mariano on the date-level series. Because the mean is linear,
    ``mean(loss_a) - mean(loss_b)`` per date equals the mean loss differential per date, so this is
    exactly DM on the date-aggregated differential. ``mean_diff < 0`` favours model A (smaller loss).

    Returns ``{"dm_hln", "p_value", "mean_diff", "n_dates"}``.
    """
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    if loss_a.shape != loss_b.shape:                 # both aggregate over the SAME `dates` array; require
        raise ValueError("loss_a and loss_b must have the same shape (aligned per-observation)")
    _, a_by_date = _aggregate_by_date(loss_a, dates)
    _, b_by_date = _aggregate_by_date(loss_b, dates)
    res = _metrics.diebold_mariano(a_by_date, b_by_date, h=h)
    return {
        "dm_hln": res.dm_hln,
        "p_value": res.p_value,
        "mean_diff": res.mean_diff,
        "n_dates": int(a_by_date.size),
    }
```

The per-date collapse itself:

```python
# baselines/2026-08-21_har_anchored_residual/code/stats.py
def _aggregate_by_date(values: np.ndarray, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse ``values`` to the mean within each unique date.

    Returns ``(unique_dates_sorted, mean_per_date)``. ``np.unique`` yields dates in chronological
    order (dates are sortable ints/datetimes), giving a time-ordered aggregated series.
    """
    values = np.asarray(values, dtype=float)
    dates = np.asarray(dates)
    if values.shape[0] != dates.shape[0]:
        raise ValueError("values and dates must have the same length")
    uniq, inverse = np.unique(dates, return_inverse=True)
    sums = np.zeros(uniq.size, dtype=float)
    counts = np.zeros(uniq.size, dtype=float)
    np.add.at(sums, inverse, values)
    np.add.at(counts, inverse, 1.0)
    return uniq, sums / counts
```

The underlying DM statistic (HLN small-sample correction, Newey-West Bartlett HAC truncated at `h-1` lags, Student-t p-value) is:

```python
# submission/soict_lstm_gat/metrics.py
def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> DMResult:
    """Two-sided Diebold-Mariano test on per-observation losses.

    Loss differential ``d = loss_a - loss_b``; a negative statistic means series A carries
    the smaller loss (A more accurate). The long-run variance uses a Newey-West Bartlett
    HAC estimator truncated at ``h - 1`` lags; the HLN (1997) small-sample correction is
    applied and the statistic referred to Student-t with ``n - 1`` degrees of freedom.

    Copied from baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/diebold_mariano.py.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("loss_a and loss_b must have the same shape")
    if a.ndim != 1:
        raise ValueError("losses must be one-dimensional per-observation series")
    if isinstance(h, bool) or not float(h).is_integer():
        raise ValueError(f"horizon h must be an integer, got {h!r}")   # R-13: h=1.5 fails loud, not TypeError
    h = int(h)
    if h < 1:
        raise ValueError("horizon h must be >= 1")
    n = a.size
    if n < 2:
        raise ValueError("Diebold-Mariano requires at least two observations")
    if h >= n:
        raise ValueError(f"horizon h must be < n (HLN factor / HAC lag undefined for h >= n), got h={h}, n={n}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("losses must be finite")

    d = a - b
    mean_diff = float(d.mean())
    dev = d - mean_diff
    max_lag = h - 1

    gamma0 = float(np.dot(dev, dev) / n)
    long_run = gamma0
    for lag in range(1, max_lag + 1):
        gamma = float(np.dot(dev[lag:], dev[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1)
        long_run += 2.0 * weight * gamma

    if long_run <= 0.0:
        raise ValueError("non-positive long-run variance; DM statistic undefined")

    dm_stat = mean_diff / np.sqrt(long_run / n)
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = float(dm_stat * hln_factor)
    p_value = float(2.0 * stats.t.cdf(-abs(dm_hln), df=n - 1))

    return DMResult(dm_hln=dm_hln, p_value=p_value, mean_diff=mean_diff, n=n)
```

**Bolding rule in the results tables.** In each horizon block the best model on QLIKE is bolded only when its improvement over the relevant comparator is **DM-significant** under this date-clustered test. A lower QLIKE that is not DM-significant is not bolded as a win.

---

## 7. Metrics

All metrics are self-contained in `submission/soict_lstm_gat/metrics.py` and operate on paired `(y, p)` arrays with a shared shape/finite guard.

**Point-forecast metrics.**

```python
# submission/soict_lstm_gat/metrics.py
def mse(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error."""
    y, p = _check_pair(y, p)
    return float(np.mean((y - p) ** 2))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y, p)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    """Mean absolute error."""
    y, p = _check_pair(y, p)
    return float(np.mean(np.abs(y - p)))
```

**QLIKE (per-observation and mean).** Both target and prediction are clamped to `>= floor` on a single shared floor, then `r = y/p` and loss `= r - log(r) - 1` (zero when the forecast is exact):

```python
# submission/soict_lstm_gat/metrics.py
def per_obs_qlike(y: np.ndarray, p: np.ndarray, floor: float = _EPSILON) -> np.ndarray:
    """Per-observation QLIKE with a single shared positivity floor.

    Both target and prediction are clamped to ``>= floor`` (identical basis), then
    ``r = y / p`` and loss ``= r - log(r) - 1`` (== 0 when the forecast is exact).

    ``floor`` must be finite and positive (else the positivity clamp is meaningless), and
    ``y``/``p`` must be finite (NaN/inf fail loud rather than silently surviving ``np.maximum``).
    """
    if not (np.isfinite(floor) and floor > 0.0):
        raise ValueError(f"floor must be finite and positive, got {floor}")
    y, p = _check_pair(y, p)
    y = np.maximum(y, floor)
    p = np.maximum(p, floor)
    ratio = y / p
    return ratio - np.log(ratio) - 1.0
```

The QLIKE formula is therefore, per observation:

```
QLIKE(y, f) = y/f - ln(y/f) - 1
```

**Gamma-deviance / QLIKE relation.** The scikit-learn gamma deviance for target `y` and prediction `f` is `2 * (log(f/y) + y/f - 1) = 2 * (y/f - log(y/f) - 1)`. This is `2 * QLIKE`, i.e. QLIKE up to a positive multiplicative-and-additive constant. Minimizing the gamma deviance during boosting therefore minimizes QLIKE, which is why `loss="gamma"` is the training objective for a QLIKE-scored task.

---

## 8. Leakage controls

The design enforces the following controls, each visible in the quoted code:

1. **Adjacency / correlation on train only.** `build_graph(tr, ...)` fits the correlation top-k graph on the training slice `tr` (`fold.date < ts - embargo`); the graph is rebuilt per fold. No test-window data enters the adjacency (Section 3.3).
2. **Standardization on train panel only.** `_std_cols` inside `build_graph` standardizes the TRAIN `logpk` pivot; there is no scaler fit on test data.
3. **Target-horizon embargo.** `embargo = int(h*1.6) + 5` calendar days separates train end from test start, so no training label `pk_{t+h}` reaches into the test window (Section 2.4).
4. **Test read once.** Each fold's test slice `tef` is scored a single time and concatenated; there is no repeated peeking or model selection on test folds (Section 6).
5. **Per-ticker causal features.** All OWN features are rolling / lag operations over each ticker's own past (`_feat`), and the target `y = pk.shift(-h)` is created per ticker before pooling, so no future or cross-ticker leakage occurs in feature or label construction (Sections 2.2, 3.1).
6. **Earnings distances to real announcement dates.** EARN features use `_signed` distances to actual crawled announcement dates (`sp500_earnings.parquet`; `hose_earnings_combined.parquet` for HOSE), measured at the target date `date.shift(-h)`; they are not filing deadlines (Section 3.2).
7. **Neighbour aggregate observable at origin.** `g[i,t] = sum_j W_ij * pk[j,t]` uses same-day `pk[j,t]` of other firms (contemporaneous, observable at `t`), never a future value; the illegal `g_oracle` (neighbour target at `t+h`) exists only as a separate contemporaneous upper-bound diagnostic in `paper_metrics_sp500.py`, not as a reported model.
8. **Shared positivity floor.** The same `FL = 1e-8` floor is applied to training targets, predictions, and the QLIKE metric across every model, so no model gains an artifactual edge from a different floor.
```