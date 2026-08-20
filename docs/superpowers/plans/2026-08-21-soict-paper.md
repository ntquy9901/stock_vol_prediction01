# SOICT HAR-LSTM-GAT Paper — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal self-contained submission folder that trains/evaluates a proposed
HAR-LSTM-GAT volatility model against HAR + GARCH baselines (VN30 main, VN100/S&P500 variations),
produces a reviewer reproduce script, and drafts the SOICT paper.

**Architecture:** Extract tested components from the repo into `submission/soict_lstm_gat/`, TDD the
new glue, run main + leave-one-out ablation + 3 variation studies × {h1,h5} × 5 seeds on 80/10/10
per-stock pooled splits, evaluate 5 metrics + Diebold–Mariano.

**Tech Stack:** Python 3.11, PyTorch (GPU), numpy/pandas, scikit-learn (graphical_lasso), arch (GARCH),
matplotlib (learning curves), pytest.

## Global Constraints (verbatim from spec)

- Lookback = 10 (main), 22 (variation). Horizons h ∈ {1, 5}. Target = Parkinson VARIANCE at t+h.
- Features (3) = HAR [parkinson(t), rolling-5 mean, rolling-22 mean]; shared LSTM input + GAT node feats.
- Split: per-stock chronological 80/10/10; ONE pooled model; per-ticker StandardScaler fit TRAIN-only.
- **Training loss = MSE**; early-stop on **val MSE** (patience 3, min_epochs 5, 20 epochs max). 5 seeds
  {42,123,2026,7,2024}. dropout 0.2, weight_decay 1e-5, grad-clip 1.0, ReduceLROnPlateau.
- GAT edges = graphical-lasso partial-correlation Top-5, estimated on TRAIN rows only + frozen.
- **QLIKE positivity floor = 1e-8, identical across every compared model.**
- Evaluate 5 metrics (MSE/RMSE/MAE/QLIKE/R2), seed-averaged. **Success + DM decided on QLIKE** at h1/h5
  (p<0.05); MSE is the training loss only. Report honestly (win/tie/lose).
- Model name: **HAR-LSTM-GAT (Ours)**; ablation = **LSTM (w/o GAT)**. Row order in tables:
  HAR → GARCH → LSTM (w/o GAT) → HAR-LSTM-GAT (Ours).
- Ship PROCESSED data only (date, parkinson_volatility) for vn30/vn100/sp500 — no raw OHLCV.
- No news/gate, no 5-feature set, no DirAcc.

## File Structure

```
submission/soict_lstm_gat/
├── config.py         # hyperparams, seeds, dataset paths, horizons, lookbacks
├── metrics.py        # mse/rmse/mae/qlike(floor)/r2 + diebold_mariano
├── data_utils.py     # load_processed, har_features, make_windows, per_stock_split, TickerScaler, pool
├── edges.py          # glasso_adjacency(train_panel, top_k) -> [N,N] frozen
├── model.py          # HARLSTMGAT(use_graph) forward [B,N]
├── baselines.py      # har_ols_fit/predict, garch_forecast
├── train.py          # train_pooled(config, seed) -> checkpoint + curves + log
├── evaluate.py       # evaluate_run -> metrics dict; dm_compare
├── run_all.py        # orchestrate main + ablation + variations
├── config_smoke.py   # tiny config for smoke tests
├── data/{vn30,vn100,sp500}/*_processed.csv   # shipped (Phase 9)
├── README.md, REPRODUCE.md, EXTRACTION_LOG.md, requirements.txt, reproduce.sh
└── tests/test_*.py
```

Reuse (read-only reference, then extract minimal): `baselines/2026-08-15_volatility/code/{model.py,
edges_glasso.py,dm_report.py}`, `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/diebold_mariano.py`,
`baselines/2026-08-14_pooled_news_edanode_gnn/code/eda_ladder.py` (HAR run_e0), `baselines/classical_baselines/code`
(GARCH), `scripts/sp500_crossmarket/run_sp500_crossmarket.py` (proven pooled HAR+LSTM pattern).

---

## Phase 0 — Scaffold

### Task 0: Folder + config + deps

**Files:** Create `submission/soict_lstm_gat/{__init__.py,config.py,config_smoke.py,requirements.txt,tests/__init__.py}`

- [ ] **Step 1:** Create `config.py` with a frozen dataclass:
```python
from dataclasses import dataclass, field
@dataclass(frozen=True)
class Config:
    lookback: int = 10
    horizons: tuple = (1, 5)
    seeds: tuple = (42, 123, 2026, 7, 2024)
    epochs: int = 20
    patience: int = 3
    min_epochs: int = 5
    hidden: int = 64
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    top_k: int = 5           # glasso Top-K edges
    qlike_floor: float = 1e-8
    batch_size: int = 512
    train_frac: float = 0.80
    val_frac: float = 0.10   # test = remaining 0.10
    data_root: str = "data"  # relative to the submission folder
```
`config_smoke.py`: same but epochs=2, seeds=(42,), lookback=10.
`requirements.txt`: `torch`, `numpy`, `pandas`, `scikit-learn`, `arch`, `matplotlib`, `pytest`.

- [ ] **Step 2:** Commit `chore: scaffold submission/soict_lstm_gat config`.

---

## Phase 1 — metrics.py (TDD)

### Task 1: Point metrics + QLIKE floor + DM

**Files:** Create `submission/soict_lstm_gat/metrics.py`, `tests/test_metrics.py`
**Interfaces — Produces:** `mse(y,p)`, `rmse`, `mae`, `r2`, `qlike(y,p,floor=1e-8)` → float;
`per_obs_qlike(y,p,floor)`, `per_obs_se(y,p)` → np.ndarray; `diebold_mariano(loss_a,loss_b,h)` →
`DMResult(dm_hln,p_value,mean_diff,n)`.

- [ ] **Step 1: failing test**
```python
import numpy as np, pytest
from submission.soict_lstm_gat import metrics as m
def test_qlike_perfect_is_zero_and_floored():
    y=np.array([0.5,1e-12]); p=np.array([0.5,0.3])
    assert m.qlike(np.array([0.5]),np.array([0.5])) == pytest.approx(0.0)
    # floor: y below 1e-8 clamped, no inf/nan
    assert np.isfinite(m.qlike(y,p))
def test_rmse_is_sqrt_mse():
    y=np.array([1.,2.,3.]); p=np.array([1.,2.,4.])
    assert m.rmse(y,p)==pytest.approx(np.sqrt(m.mse(y,p)))
def test_dm_sign_favors_lower_loss():
    a=np.full(200,0.1); b=np.full(200,0.2)+np.random.default_rng(0).normal(0,1e-3,200)
    r=m.diebold_mariano(a,b,h=1)
    assert r.mean_diff<0 and r.dm_hln<0   # A lower loss -> negative
```
- [ ] **Step 2:** Run → FAIL (module missing).
- [ ] **Step 3:** Implement `metrics.py` (copy QLIKE floor + HLN DM from `dm_report._qlike` /
  `diebold_mariano.py`; `mse/rmse/mae/r2` standard; `qlike` = mean of `per_obs_qlike`).
- [ ] **Step 4:** Run tests → PASS. Add a test that DM `h=5` uses HAC lag 4 (n≥2 guard raises on n<2).
- [ ] **Step 5:** Commit `feat(metrics): 5 metrics + QLIKE floor + Diebold-Mariano (TDD)`.

---

## Phase 2 — data_utils.py (TDD)

### Task 2: HAR features + windows + per-stock split + scaler (no leakage)

**Files:** Create `submission/soict_lstm_gat/data_utils.py`, `tests/test_data_utils.py`
**Interfaces — Produces:**
- `har_features(pk: np.ndarray) -> np.ndarray[N,3]` (daily, roll5, roll22; NaN until valid).
- `make_windows(pk, lookback, horizon) -> np.ndarray[anchors]` (monthly-valid anchor indices).
- `per_stock_split(anchors, train_frac, val_frac) -> (a_tr,a_va,a_te)`.
- `TickerScaler.fit(feats_rows)/transform(windows)` and target mean/std (fit on TRAIN feats/targets).
- `build_pooled(files, lookback, horizon, cfg) -> PooledData` with concatenated normalized
  `X_tr[M,lookback,3], y_tr_norm, X_va, ..., X_te, y_te_raw, te_ticker_ids, te_target_dates, scalers`.

- [ ] **Step 1: failing tests**
```python
def test_har_features_rolling_validity():
    pk=np.arange(1,60,dtype=float); f=du.har_features(pk)
    assert f.shape==(59,3)
    assert np.isnan(f[:4,1]).all() and not np.isnan(f[4,1])    # weekly valid @idx4
    assert np.isnan(f[:21,2]).all() and not np.isnan(f[21,2])  # monthly valid @idx21
    assert f[30,0]==pk[30]                                     # daily==pk(t)
def test_split_fractions_and_order():
    a=np.arange(1000); tr,va,te=du.per_stock_split(a,0.8,0.1)
    assert len(tr)==800 and len(va)==100 and len(te)==100
    assert tr[-1]<va[0]<te[0]                                  # chronological, no overlap
def test_scaler_fit_train_only_no_leakage():
    s=du.TickerScaler(); s.fit_features(np.array([[1.,1,1],[3,3,3]]))
    before=(s.f_mean.copy(),s.f_std.copy())
    # transforming test rows must NOT change the fitted stats
    s.transform(np.zeros((2,10,3)))
    assert np.allclose(s.f_mean,before[0]) and np.allclose(s.f_std,before[1])
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `data_utils.py` (adapt `build_ticker_features`/`make_windows`/split from
  `scripts/sp500_crossmarket/run_sp500_crossmarket.py`; generalize `data_root` + `train_frac/val_frac`
  to 80/10/10; add `TickerScaler` with `fit_features`, `fit_target`, `transform`).
- [ ] **Step 4:** Run → PASS. Add a real-data-sample smoke: load 2 VN30 processed files, `build_pooled`
  h1 → assert `X_tr.shape[1:]==(10,3)`, finite, `te_ticker_ids` maps to scalers.
- [ ] **Step 5:** Commit `feat(data): HAR features + windows + 80/10/10 per-stock split + scaler (TDD)`.

---

## Phase 3 — edges.py (TDD, graphical-lasso, train-only)

### Task 3: glasso partial-correlation adjacency

**Files:** Create `submission/soict_lstm_gat/edges.py`, `tests/test_edges.py`
**Interfaces — Produces:** `glasso_adjacency(pk_wide_train: pd.DataFrame, top_k=5, alpha=None) ->
np.ndarray[N,N]` (symmetric, self-loop diagonal=1, Top-K partial-corr per node, TRAIN rows only).

- [ ] **Step 1: failing tests**
```python
def test_adjacency_shape_symmetric_selfloop():
    rng=np.random.default_rng(0); df=pd.DataFrame(rng.normal(size=(300,6)),columns=list("ABCDEF"))
    A=eg.glasso_adjacency(df,top_k=2)
    assert A.shape==(6,6) and np.allclose(np.diag(A),1.0)
    assert (A!=0).sum(1).max()<=2+1                       # top-2 + self-loop
def test_train_only_frozen():
    rng=np.random.default_rng(1); tr=pd.DataFrame(rng.normal(size=(300,5)))
    A1=eg.glasso_adjacency(tr,top_k=2)
    tr2=pd.concat([tr,pd.DataFrame(rng.normal(size=(50,5)))],ignore_index=True)  # +test rows
    # edge must be a pure function of the ROWS PASSED; passing only-train reproduces A1
    A1b=eg.glasso_adjacency(tr,top_k=2)
    assert np.allclose(A1,A1b)
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `edges.py` (extract `glasso_partial_corr` + `precision_to_partial_corr` +
  Top-K selection from `baselines/2026-08-15_volatility/code/edges_glasso.py`; alpha auto-raise loop
  until `graphical_lasso` converges; fallback `np.linalg.pinv`). Caller passes TRAIN pk-wide only.
- [ ] **Step 4:** Run → PASS. Add non-convergence test (near-singular corr) → still returns finite A.
- [ ] **Step 5:** Commit `feat(edges): train-only graphical-lasso partial-corr adjacency (TDD)`.

---

## Phase 4 — model.py (TDD)

### Task 4: HAR-LSTM-GAT + use_graph toggle

**Files:** Create `submission/soict_lstm_gat/model.py`, `tests/test_model.py`
**Interfaces — Produces:** `HARLSTMGAT(price_dim=3,hidden=64,heads=4,dropout=0.2,use_graph=True)`;
`forward(x[B,N,seq,3], adj[N,N]) -> [B,N]` normalized-scale output.

- [ ] **Step 1: failing tests**
```python
def test_forward_shape_and_graph_toggle():
    import torch
    full=md.HARLSTMGAT(use_graph=True); nog=md.HARLSTMGAT(use_graph=False)
    x=torch.randn(4,6,10,3); adj=torch.eye(6)
    assert full(x,adj).shape==(4,6) and nog(x,adj).shape==(4,6)
    assert sum(p.numel() for p in full.parameters())>sum(p.numel() for p in nog.parameters())
def test_no_graph_ignores_adjacency():
    import torch; nog=md.HARLSTMGAT(use_graph=False).eval()
    x=torch.randn(2,5,10,3)
    a=nog(x,torch.eye(5)); b=nog(x,torch.ones(5,5))
    assert torch.allclose(a,b)   # adjacency has no effect when use_graph=False
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `model.py` (minimal from `VolatilityModel`: `price_lstm` 2-layer;
  when `use_graph`, a `GATLayer(price_dim→hidden*heads)` on `x[:,:,-1,:]` raw day-t node feats +
  concat; head `Linear(hidden[+gnn_dim])→1`; drop news/gate/positivity-buffer complexity; keep a
  simple inverse handled in evaluate). Extract `GATLayer` from `gat.py`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(model): HAR-LSTM-GAT with use_graph toggle (TDD)`.

---

## Phase 5 — baselines.py (TDD)

### Task 5: HAR OLS + GARCH

**Files:** Create `submission/soict_lstm_gat/baselines.py`, `tests/test_baselines.py`
**Interfaces — Produces:** `har_fit(X[n,3],y[n]) -> coef[4]`; `har_predict(X,coef) -> yhat` (floored ≥0);
`garch_forecast(train_series, n_test, horizon) -> yhat[n_test]` (per-ticker GARCH(1,1) variance forecast).

- [ ] **Step 1: failing tests**
```python
def test_har_ols_recovers_linear():
    rng=np.random.default_rng(0); X=rng.normal(size=(500,3)); coef=np.array([0.1,0.2,0.3,0.4])
    y=coef[0]+X@coef[1:]; c=bl.har_fit(X,y)
    assert np.allclose(c,coef,atol=1e-6) and np.all(bl.har_predict(X,c)>=0)
def test_garch_smoke_positive():
    rng=np.random.default_rng(0); s=np.abs(rng.normal(0,1e-2,800))+1e-4
    f=bl.garch_forecast(s,n_test=20,horizon=1); assert f.shape==(20,) and np.all(f>0)
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `baselines.py` (HAR = `np.linalg.lstsq` with intercept, floor at
  `qlike_floor`; GARCH via `arch.arch_model(rescaled, vol='Garch', p=1,q=1)`, rolling/one-shot forecast
  variance, rescale back). Reference `baselines/classical_baselines/code`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(baselines): HAR OLS + GARCH(1,1) (TDD)`.

---

## Phase 6 — train.py (TDD smoke)

### Task 6: pooled training loop (MSE, early-stop, curves, log)

**Files:** Create `submission/soict_lstm_gat/train.py`, `tests/test_train.py`
**Interfaces — Produces:** `train_pooled(pooled, cfg, seed, use_graph, adj, out_dir) ->
{best_state, history, epochs_ran}`; writes `learning_curve_ep{e}.png` every 5 epochs + `train.log`.

- [ ] **Step 1: failing smoke test** (2-epoch synthetic pooled, cfg_smoke): asserts returns dict with
  `epochs_ran<=2`, a checkpoint file, ≥1 PNG, `train.log` exists and contains "hyperparameters".
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `train.py` (adapt sp500 crossmarket loop: MSE loss, Adam+wd, grad-clip,
  ReduceLROnPlateau, early-stop on **val MSE** patience/min_epochs, `torch.manual_seed(seed)`, batched
  GPU, save learning-curve PNG every 5 epochs, print+log hyperparams & per-epoch val metrics).
- [ ] **Step 4:** Run smoke → PASS.
- [ ] **Step 5:** Commit `feat(train): pooled MSE training loop + early-stop + learning curves (TDD)`.

---

## Phase 7 — evaluate.py (TDD)

### Task 7: test metrics + DM compare

**Files:** Create `submission/soict_lstm_gat/evaluate.py`, `tests/test_evaluate.py`
**Interfaces — Produces:** `predict_test(model, pooled, adj) -> preds_raw[n_test]` (inverse-scaled,
floored); `all_metrics(y_raw,pred_raw,cfg) -> {mse,rmse,mae,qlike,r2}`; `dm_compare(pred_a,pred_b,y,h,
loss) -> DMResult`.

- [ ] **Step 1: failing test** — synthetic preds → `all_metrics` keys + values match `metrics.*`;
  `dm_compare` on QLIKE returns negative dm when A closer.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `evaluate.py` (inverse-transform per te_ticker_ids scaler, floor, compute
  metrics; DM via `metrics.diebold_mariano` on per-obs QLIKE/SE, seed-ensemble = mean preds over seeds).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(evaluate): test metrics + Diebold-Mariano compare (TDD)`.

---

## Phase 8 — run_all.py (integration smoke)

### Task 8: orchestrate main + ablation + variations

**Files:** Create `submission/soict_lstm_gat/run_all.py`, `tests/test_run_all.py`
**Interfaces — Produces:** `run_experiment(dataset, lookback, horizon, use_graph, cfg) -> result dict`
seed-ensembled with all metrics + baselines + DM; `main()` writes `results/soict/<stamp>/results.json`.

- [ ] **Step 1: failing integration smoke** (cfg_smoke, 2 tiny synthetic tickers, h1, 1 seed): asserts
  `results.json` has keys for `HAR-LSTM-GAT`, `LSTM (w/o GAT)`, `HAR`, `GARCH` + DM cells; runs <60s.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `run_all.py`: for each (config in {main VN30 lb10, ablation −gat lb10,
  var lb22, var VN100, var SP500}) × horizon × seed → build_pooled, glasso adj (train-only), train,
  predict; ensemble over seeds; compute baselines (HAR, GARCH); DM Ours-vs-{HAR,GARCH,w/o-GAT};
  write results.json. Parallelize across (seed×config) with a process pool / concurrent runs.
- [ ] **Step 4:** Run smoke → PASS.
- [ ] **Step 5:** Commit `feat(run_all): orchestrate main+ablation+variations + DM (TDD)`.

---

## Phase 9 — Ship data + reproduce + docs

### Task 9: extract data, reproduce.sh, EXTRACTION_LOG, README

**Files:** Create `submission/soict_lstm_gat/{data/**,reproduce.sh,README.md,REPRODUCE.md,EXTRACTION_LOG.md}`

- [ ] **Step 1:** Copy processed CSVs → `data/vn30` (from `data/processed/*_processed.csv`),
  `data/vn100` (from `data/processed/vn100_vnstock`), `data/sp500` (from `data/processed/sp500`).
  Write `EXTRACTION_LOG.md` listing every extracted source file → dest (ENFORCE §1.6), incl. which
  repo module each `submission/*.py` was distilled from.
- [ ] **Step 2:** `reproduce.sh`: create venv, `pip install -r requirements.txt`, run `python -m
  run_all` (train+test), print output paths. `REPRODUCE.md`: env + commands for reviewers. `README.md`:
  overview, provenance/license note (VN100=vnstock, SP500 derived-from-Yahoo non-commercial), how to run.
- [ ] **Step 3:** Gitignore the shipped `submission/**/data/sp500` (Yahoo-derived) OR ship it —
  DECISION: ship processed vn30+vn100; **sp500 processed is derived-from-Yahoo → gitignore in the repo
  but let reproduce.sh regenerate it** (documented in README). vn30/vn100 processed are committed.
- [ ] **Step 4:** Commit `feat(submission): ship processed data + reproduce.sh + extraction log + docs`.

---

## Phase 10 — Run the full experiment suite

### Task 10: execute + collect results (real data)

- [ ] **Step 1:** Run `run_all` full (real Config): main VN30 lb10 + ablation −gat + variations
  (lb22 VN30, VN100, SP500) × {h1,h5} × 5 seeds, GPU, parallel. Background; poll.
- [ ] **Step 2:** Verify `results/soict/<stamp>/results.json` complete; learning-curve PNGs present.
- [ ] **Step 3:** Sanity: metrics finite; DM n large; no all-zero features; QLIKE floor identical.
- [ ] **Step 4:** Commit results.json + curves + a `docs/reports/<stamp>_soict_results_report.md`
  (all-metric tables per horizon, DM verdicts, honest win/tie/lose).

---

## Phase 11 — Paper draft + .svg diagram

### Task 11: markdown paper + architecture SVG

- [ ] **Step 1:** Generate compact `.svg` architecture diagram (HAR features → per-node LSTM → GAT over
  glasso graph → head) into `docs/paper/diagrams/soict_harlstmgat.svg` (reuse `generate_diagrams.py`).
- [ ] **Step 2:** Write `docs/paper/soict_harlstmgat_draft.md` (objective style; Abstract, Intro,
  Related Work, Method, Data, Experiments, Results tables + DM + curves, Discussion, Conclusion; row
  order HAR→GARCH→LSTM(w/o GAT)→HAR-LSTM-GAT(Ours); all horizons; honest verdicts).
- [ ] **Step 3:** Commit `docs(paper): SOICT HAR-LSTM-GAT markdown draft + .svg architecture`.

---

## Phase 12 — LaTeX + review + wrap

### Task 12: SOICT LaTeX + code-review + summary

- [ ] **Step 1:** WebFetch `https://soict.org/submission/paper-submission/` for the required template
  (likely Springer LNCS / IEEE). If accessible, produce `docs/paper/soict_harlstmgat.tex` per template;
  else deliver best-effort LaTeX + note the gap in the report.
- [ ] **Step 2:** Run `/code-review` (3-layer) on `submission/soict_lstm_gat/` new code; fix findings.
- [ ] **Step 3:** verification-before-completion: run full `pytest submission/soict_lstm_gat/tests`,
  confirm green; run the reproduce.sh smoke path.
- [ ] **Step 4:** Write `docs/reports/<stamp>_soict_summaryOfUpdate_report.md` (what built, results,
  code-review, honest verdict, DoD). Commit + push.

---

## Self-Review (plan vs spec)

- **Coverage:** spec §3 folder→Task0/9; §4 data→Task2; §5 model/ablation/variations→Task4/8/10;
  baselines→Task5; §6 training→Task6; §7 eval+DM→Task7; §8 paper+svg→Task11/12; §9 reproducibility→Task9;
  §10 TDD tests→each task; §11 defaults→config Task0. All covered.
- **Placeholder scan:** each task has concrete test + implementation direction + commit. No TBDs.
- **Type consistency:** `build_pooled`→PooledData consumed by train/evaluate/run_all; `glasso_adjacency`
  used in run_all(train-only); `HARLSTMGAT(use_graph)` toggled for Ours vs w/o-GAT; `diebold_mariano`
  signature shared metrics↔evaluate. Consistent.
- **Honest-result note:** success is conditional; Phase 10/11 report true DM verdicts (no fabrication).
