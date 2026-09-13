# GARCH volatility baseline — requirements (spec)

> SDD phase 2 (Specify). Constitution = repo `CLAUDE.md`. This file is the source of truth for the
> baseline; code is a translation of it.

## 1. Goal

Provide a correctly-implemented, per-stock **GARCH conditional-variance benchmark** that plugs into the
project comparison table (HAR / HARQ / GBM / …) on the **same walk-forward folds** and the **same
pooled per-observation QLIKE basis** so the paper can state "our model beats the classic GARCH
benchmark". Two variants are reported:

- **GARCH(1,1)** — symmetric conditional variance.
- **GJR-GARCH(1,1,1)** — adds a leverage (asymmetry) term, directly relevant because downside
  semivariance was the one signal that helped on this data.

GARCH is the pure econometric benchmark: it uses **only each stock's own daily log-return history**,
no exogenous features.

## 2. Inputs / outputs

**Input:** per-ticker enriched frames from `scripts/eda/full_matrix.py::load(market)` — each frame has
`date`, `daily_return` (daily log-return), `parkinson_variance` (the σ² target), and the HAR columns.
GARCH consumes ONLY `daily_return` (+ `date` for ordering); the target/eval basis reuses
`full_matrix.panel(frames, {}, h)` (`y = parkinson_variance.shift(-h)`).

**Output:** `results/gamma_gbm/garch_<market>.json`. Per horizon h ∈ {1, 5, 10, 22}:
```
h{h}: {
  n: <#pooled test obs scored>,
  n_excluded: <#test obs dropped (ticker-fold below MIN_TRAIN_OBS own history)>,
  qlike: {GARCH, GJR-GARCH, HAR},
  gain_vs_HAR_pct: {GARCH, GJR-GARCH},          # (HAR - model)/HAR*100, +ve = model better
  dm_vs_HAR: {GARCH: {p_value, mean_diff}, GJR-GARCH: {...}},  # mean_diff<0 => model beats HAR
  n_fallback: {GARCH, GJR-GARCH},               # #(ticker,fold) that fell back to unconditional var
  train_metrics: {GARCH, GJR-GARCH, HAR},       # in-sample pooled QLIKE (last active fold train rows)
  fit_diagnostics: {model: {verdict, train_qlike, test_qlike}}
}
```

## 3. Model / method spec

### GARCH(1,1)
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}, ε_t = r_t − μ (constant-mean), Normal errors.

### GJR-GARCH(1,1,1)
σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·1(ε_{t-1}<0) + β·σ²_{t-1}.

### Estimation
`arch` package (Kevin Sheppard), verified present in `.venv_gpu_encode` (arch 8.0.0). Fit by ML per
(ticker, fold, horizon).

### Units (CRITICAL)
`parkinson_variance` is the variance of daily LOG-returns (dimensionless, O(1e-4)). `arch` expects
returns scaled (×100) for numerical conditioning. So fit on `daily_return × SCALE` (SCALE=100); the
fitted conditional variance is on the (return×100)² scale, so every forecast is rescaled back by
**÷ SCALE²** to sit on the same scale as `parkinson_variance`. A unit test asserts scale recovery on a
synthetic known-variance series (fallback path returns the sample variance in ORIGINAL scale, not
×10⁴ / ÷10⁴).

### Multi-step forecast (causal)
For horizon h, forecast σ²_{t+h|t}. With reversion persistence φ (φ = α+β for GARCH,
φ = α+β+γ/2 for GJR under a symmetric zero-mean innovation), long-run variance σ̄² = ω/(1−φ):
```
σ²_{t+h|t} = σ̄² + φ^{h-1} · (σ²_{t+1|t} − σ̄²)
```
σ²_{t+1|t} = ω + (α + γ·1(ε_t<0))·ε²_t + β·σ²_t is computable at t (uses returns ≤ t). For h=1 this
reduces to σ²_{t+1|t}. As h→∞ it reverts to σ̄². Both properties are unit-tested against an
independent iteration of the expectation recursion.

### Walk-forward (causal)
Reuse `vn_gbm_graph_stage1.FOLDS` / `TRAIN_START` and `embargo = int(h·1.6)+5` days (identical to the
sibling GBM/HAR baselines). Per (fold, horizon):
- Fold gate: skip if pooled train rows < min_rows (30000 sp500 / 3000 else) or test rows == 0 —
  identical to the siblings, so GARCH/GJR/HAR are scored on **exactly the same rows**.
- For each ticker: fit GARCH/GJR on its returns with `date < ts − embargo`; filter the conditional
  variance recursion causally over the full return history; emit σ²_{t+h|t} for each test date t.

### Positivity floor / metric / DM
Pooled per-observation QLIKE with the shared floor `full_matrix.FL`; date-clustered Diebold-Mariano
(`stats.date_clustered_dm`) of each variant vs `full_matrix._har_ols` on the identical rows, all
horizons.

## 4. Leakage argument
- GARCH parameters are estimated only on returns with `date < ts − embargo` (train window).
- The conditional-variance recursion at test date t uses only returns with index ≤ t (causal filter);
  σ²_{t+h|t} is a pure function of past returns and fixed params.
- GARCH training is unsupervised on the return series (never sees the shifted target `y`), so the
  target embargo is inherited from the sibling protocol only to keep the train window identical.
- Unit test: perturbing a FUTURE return leaves an earlier test date's forecast unchanged.

## 5. Convergence / degeneracy handling (no silent garbage)
- **Estimability gate (scoring):** a per-stock GARCH is undefined without own history. A ticker-fold is
  scored ONLY when it has ≥ `MIN_TRAIN_OBS` returns before the fold; otherwise its rows are excluded
  from the pooled comparison for **every** model (HAR/GARCH/GJR stay on identical rows) and counted in
  `n_excluded`. Without this gate a newly-listed ticker's unconditional variance is ~0 → the forecast
  floors to `FL` → `y/FL` is astronomical → the pooled QLIKE is dominated by degenerate garbage rather
  than a meaningful GARCH-vs-HAR signal (observed: 2 zero-history HOSE tickers drove the mean from ~1 to
  ~80). `MIN_TRAIN_OBS` thus gates both fitting and scoring.
- **Convergence fallback:** for a scored ticker-fold (≥ `MIN_TRAIN_OBS` returns) whose `arch` fit raises
  or returns degenerate params (ω ≤ 0, or reversion persistence φ ∉ (0,1)), the forecast falls back to
  the **unconditional sample variance** of its train returns (a positive constant, original scale) —
  counted in `n_fallback`, never zeros / NaN. `arch` itself is imported at module load so a broken
  install fails loud instead of silently degrading every fit to fallback.

## 6. Acceptance criteria (go/no-go)
- **GO** when: (a) all unit tests pass (multi-step closed-form vs iteration; reversion to σ̄²;
  units/scale recovery; causality) with C0=100% / C1≥95% on changed lines; (b) `run_garch("hose")`
  completes on real data and writes a JSON with real numbers for every horizon; (c) HAR is scored on
  the identical rows (same fold gate) so the DM is apples-to-apples.
- **Verdict is descriptive, not a pass condition:** prior expectation is GARCH ≈ / worse-than HAR-X on
  QLIKE (this project already found deep/HAR beat GARCH). Whatever the DM says is reported without
  spin; the deliverable is a correct benchmark, not a win.

## 7. [NEEDS CLARIFICATION] — resolved defaults
- Mean model: **Constant** (arch default; ε = r − μ). Distribution: **Normal** for both variants.
- `MIN_TRAIN_OBS = 250` (~1 trading year) minimum returns to attempt an ML fit.
- SP500 is NOT run locally (heavy); a note is left for a Colab run. HOSE is run locally.
