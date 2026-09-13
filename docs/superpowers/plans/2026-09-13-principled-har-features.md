# Principled HAR-family Feature Set — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the heuristic `mr_*` + `rq`-proxy features with a principled, citation-backed HAR-family feature set (data-driven causal lags) in an isolated baseline, and DM-verify it is non-inferior to the current own-history GBM.

**Architecture:** A new isolated baseline `baselines/2026-09-13_principled_har_features/` that imports read-only from the shared `scripts/eda/full_matrix.py` (FM), `vn_gbm_graph_stage1.py` (S1), `metrics.py` (M), `stats.py` (ST). Daily OHLC → published variance estimators (Parkinson/Garman-Klass/Rogers-Satchell) + realized semivariance (SHAR) + multi-scale HAR aggregates with windows chosen causally from training data; fed to the same gamma-HistGBM + walk-forward + pooled-QLIKE + date-clustered-DM protocol as the sibling `2026-09-12_complex_network` baseline.

**Tech Stack:** Python 3.10 (`.venv_gpu_encode`), numpy, pandas, scikit-learn (HistGradientBoostingRegressor via `FM.gbm`), pytest + pytest-cov + diff-cover.

> **REVISION 2026-09-13 (option A — supersedes the tasks below where they conflict):** The design was
> simplified. Lag windows are FIXED at the standard Corsi (1, 5, 22) — there is **no data-driven selection**,
> so **Task 2 (lag_select / GPH / AIC) is REMOVED entirely** and `run_har` uses fixed windows (no
> `_choose_windows`, no `gph_d`). The GK/RS/YZ estimators are **used from the precomputed enriched columns**
> (`garman_klass_variance`, `rogers_satchell_variance`, `yang_zhang_n20`) — NOT recomputed from OHLC — so
> `estimators.py` implements only **realized semivariance**. Net tasks: (1) scaffold + semivariance estimator,
> (2 ← was Task 3) build_panel over precomputed columns + semivariance at fixed windows, (3 ← was Task 4)
> DM runner principled-vs-own + leave-one-out (fixed windows), (4 ← was Task 5) HOSE run + Colab + review +
> gate. Feature set = `[har_daily, har_weekly, har_monthly, garman_klass_variance, rogers_satchell_variance,
> yang_zhang_n20, semi_neg, semi_pos]` vs `FM.OWN`. Follow the code in Tasks 1/3/4/5 below but drop all GPH/AIC
> window-selection logic and read GK/RS/YZ from columns.

## Global Constraints

- Daily OHLC only — NO intraday. RQ (HARQ), jumps/bipower (HAR-CJ) are NOT buildable; document as a limitation, do not ship a mislabelled proxy.
- Do NOT modify `scripts/eda/full_matrix.py` `FM.OWN` or any of its 42 dependents. Import read-only.
- Every named estimator MUST match its published formula and carry a formula-exact test (independent recompute) — per the project's "named estimators" rule.
- Causal / no look-ahead: every feature at row-date `t` uses data dated `≤ t`; lag windows chosen once from the training data preceding the first test fold, fixed across folds; per-ticker scaling via the existing `FM.gbm`.
- Success = non-inferior to `FM.OWN` GBM on QLIKE under date-clustered DM (h=1,5,10,22), both markets. A DM-significant gap < ~0.1% (n≈400k) counts as non-inferior.
- Constants live in `code/config.py` (no magic numbers in pipeline modules). Parkinson σ² = (ln(H/L))²/(4·ln2).
- Tests: C0 line = 100%, C1 branch ≥ 95% on changed lines (pre-push gate). Run under `.venv_gpu_encode`.
- SP500 runs via Colab (git-centric); HOSE runs locally.

---

### Task 1: Baseline scaffold + daily variance estimators

**Files:**
- Create: `baselines/2026-09-13_principled_har_features/{requirements/requirements.md, design/design.md, code/__init__.py, code/config.py, test/__init__.py, test/conftest.py}`
- Create: `baselines/2026-09-13_principled_har_features/code/estimators.py`
- Test: `baselines/2026-09-13_principled_har_features/test/test_estimators.py`

**Interfaces:**
- Produces:
  - `estimators.parkinson(high, low) -> np.ndarray` (daily Parkinson variance σ²).
  - `estimators.garman_klass(o, h, l, c) -> np.ndarray`.
  - `estimators.rogers_satchell(o, h, l, c) -> np.ndarray`.
  - `estimators.semivariance(ret, window, min_periods) -> tuple[np.ndarray, np.ndarray]` returning `(rs_minus, rs_plus)` = trailing mean of `ret²·1(ret<0)` and `ret²·1(ret>0)` over `window` days (daily-frequency realized semivariance, Barndorff-Nielsen et al. 2010 / Patton-Sheppard 2015 adaptation).
- `config` exposes: `LN2 = np.log(2.0)`, `SEMI_WINDOW` candidate unused here (used Task 3), `FLOOR` = `S1.FL`.

- [ ] **Step 1: Create baseline folders + conftest path bootstrap**

`test/conftest.py` (mirror the sibling baseline):
```python
import sys
from pathlib import Path
_CODE = Path(__file__).resolve().parents[1] / "code"
REPO = _CODE.parents[2]
for _p in (str(REPO), str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
```
`code/__init__.py` and `test/__init__.py` empty. `requirements/requirements.md` + `design/design.md` = adapt `docs/superpowers/specs/2026-09-13-principled-har-features-design.md` (objective, I/O, success criteria, feature catalog, go/no-go).

- [ ] **Step 2: Write the failing test for the estimators**

```python
# test_estimators.py
import numpy as np
import estimators as E


def test_parkinson_matches_formula():
    h, l = np.array([102.0]), np.array([100.0])
    expected = (np.log(102/100) ** 2) / (4 * np.log(2))
    assert np.isclose(E.parkinson(h, l)[0], expected)


def test_garman_klass_matches_formula():
    o, h, l, c = map(np.array, ([100.0], [103.0], [99.0], [101.0]))
    exp = 0.5 * np.log(103/99) ** 2 - (2*np.log(2) - 1) * np.log(101/100) ** 2
    assert np.isclose(E.garman_klass(o, h, l, c)[0], exp)


def test_rogers_satchell_matches_formula():
    o, h, l, c = map(np.array, ([100.0], [103.0], [99.0], [101.0]))
    exp = np.log(103/101)*np.log(103/100) + np.log(99/101)*np.log(99/100)
    assert np.isclose(E.rogers_satchell(o, h, l, c)[0], exp)


def test_semivariance_splits_by_sign_and_is_trailing():
    ret = np.array([0.0, -0.02, 0.01, -0.03, 0.0])
    rs_minus, rs_plus = E.semivariance(ret, window=3, min_periods=1)
    # at t=3 (window {t-2,t-1,t} = idx 1,2,3): negatives -0.02,-0.03 -> mean of squares
    assert np.isclose(rs_minus[3], np.mean([0.02**2, 0.0, 0.03**2]))
    assert np.isclose(rs_plus[3], np.mean([0.0, 0.01**2, 0.0]))
    assert np.isnan(rs_minus[0]) is np.False_ or rs_minus[0] >= 0  # min_periods=1 -> defined at t=0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd baselines/2026-09-13_principled_har_features && /c/luanvan/stock_vol_prediction01/.venv_gpu_encode/Scripts/python.exe -m pytest test/test_estimators.py -q`
Expected: FAIL (`No module named 'estimators'`).

- [ ] **Step 4: Implement `estimators.py`**

```python
"""Published OHLC daily variance estimators + daily-frequency realized semivariance (all formula-exact).

Parkinson (1980), Garman-Klass (1980), Rogers-Satchell (1991); semivariance per Barndorff-Nielsen, Kinnebrock
& Shephard (2010) / Patton-Sheppard (2015), at daily frequency (one signed return per day).
"""
import numpy as np
import pandas as pd

import config


def parkinson(high, low):
    return np.log(np.asarray(high, float) / np.asarray(low, float)) ** 2 / (4 * config.LN2)


def garman_klass(o, h, l, c):
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    return 0.5 * np.log(h / l) ** 2 - (2 * config.LN2 - 1) * np.log(c / o) ** 2


def rogers_satchell(o, h, l, c):
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


def semivariance(ret, window, min_periods):
    r = pd.Series(np.asarray(ret, float))
    neg = (r ** 2).where(r < 0, 0.0)
    pos = (r ** 2).where(r > 0, 0.0)
    rm = neg.rolling(window, min_periods=min_periods).mean().to_numpy()
    rp = pos.rolling(window, min_periods=min_periods).mean().to_numpy()
    return rm, rp
```
`config.py`:
```python
"""Single source of truth for the principled-HAR baseline constants."""
import sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"))
import pipeline_config as pc  # noqa: E402  (QLIKE floor)
LN2 = float(np.log(2.0))
FLOOR = pc.QLIKE_FLOOR
HAR_WINDOW_GRID = (3, 5, 10, 22, 44, 66)     # candidate HAR aggregation windows (data-driven pick, Task 2)
GPH_BANDWIDTH = 0.5                           # GPH long-memory periodogram bandwidth exponent m = n**GPH_BANDWIDTH
SEMI_MIN_PERIODS = 3                          # min obs before a trailing semivariance is defined
```
(If `pipeline_config` import path differs, use `import vn_gbm_graph_stage1 as S1; FLOOR = S1.FL`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `... -m pytest test/test_estimators.py -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add baselines/2026-09-13_principled_har_features
git commit -m "principled_har: baseline scaffold + formula-exact OHLC estimators + semivariance"
```

---

### Task 2: Causal data-driven lag selection

**Files:**
- Create: `baselines/2026-09-13_principled_har_features/code/lag_select.py`
- Test: `baselines/2026-09-13_principled_har_features/test/test_lag_select.py`

**Interfaces:**
- Consumes: `config.HAR_WINDOW_GRID`, `config.GPH_BANDWIDTH`.
- Produces:
  - `lag_select.gph_d(series) -> float` — GPH fractional-integration estimate (long-memory parameter).
  - `lag_select.select_windows(logrv, grid=config.HAR_WINDOW_GRID) -> tuple[int,int,int]` — picks `(w_short, w_mid, w_long)` from the grid by minimising the AIC of a HAR-OLS of `logrv[t]` on its own trailing means over candidate windows, using ONLY the passed (training) series.

- [ ] **Step 1: Write the failing test**

```python
# test_lag_select.py
import numpy as np
import lag_select as L


def test_gph_d_detects_long_memory():
    # fractionally-integrated-like: cumulative sum has strong persistence (d ~ 1); white noise d ~ 0
    rng = np.random.default_rng(0)
    wn = rng.standard_normal(2000)
    rw = np.cumsum(wn)
    assert L.gph_d(rw) > L.gph_d(wn)            # persistent series has larger d
    assert abs(L.gph_d(wn)) < 0.3               # white noise near 0


def test_select_windows_returns_three_distinct_grid_values():
    rng = np.random.default_rng(1)
    # AR(1)-ish log-rv with persistence so longer windows help
    x = np.zeros(1500)
    for t in range(1, 1500):
        x[t] = 0.95 * x[t-1] + 0.2 * rng.standard_normal()
    ws, wm, wl = L.select_windows(x, grid=(3, 5, 10, 22, 44))
    assert ws < wm < wl
    assert {ws, wm, wl}.issubset({3, 5, 10, 22, 44})
```

- [ ] **Step 2: Run to verify fail**

Run: `... -m pytest test/test_lag_select.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement `lag_select.py`**

```python
"""Causal, training-only selection of HAR aggregation windows (ACF/long-memory justified).

gph_d: Geweke-Porter-Hudak (1983) log-periodogram estimator of the fractional-integration parameter d,
evidence of slowly-decaying (long-memory) volatility. select_windows: choose 3 HAR component windows from a
candidate grid by minimising the AIC of an OLS HAR regression of the series on its own trailing means.
"""
import numpy as np
import pandas as pd

import config


def gph_d(series):
    x = np.asarray(series, float)
    x = x - x.mean()
    n = len(x)
    m = int(n ** config.GPH_BANDWIDTH)
    per = np.abs(np.fft.rfft(x)) ** 2 / (2 * np.pi * n)
    freqs = 2 * np.pi * np.arange(1, m + 1) / n
    y = np.log(per[1:m + 1])
    z = np.log(2 * np.sin(freqs / 2))
    z = z - z.mean()
    d = -np.sum(z * (y - y.mean())) / np.sum(z ** 2)
    return float(d)


def _har_aic(logrv, windows):
    s = pd.Series(logrv)
    X = [np.ones(len(s))]
    for w in windows:
        X.append(s.rolling(w, min_periods=w).mean().shift(1).to_numpy())
    X = np.column_stack(X)
    y = s.to_numpy()
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    Xo, yo = X[ok], y[ok]
    beta, *_ = np.linalg.lstsq(Xo, yo, rcond=None)
    resid = yo - Xo @ beta
    rss = float(np.sum(resid ** 2))
    k = Xo.shape[1]
    n = len(yo)
    return n * np.log(rss / n) + 2 * k


def select_windows(logrv, grid=config.HAR_WINDOW_GRID):
    grid = sorted(grid)
    best, best_aic = None, np.inf
    for i in range(len(grid)):
        for j in range(i + 1, len(grid)):
            for k in range(j + 1, len(grid)):
                trip = (grid[i], grid[j], grid[k])
                aic = _har_aic(logrv, trip)
                if aic < best_aic:
                    best_aic, best = aic, trip
    return best
```

- [ ] **Step 4: Run to verify pass**

Run: `... -m pytest test/test_lag_select.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add baselines/2026-09-13_principled_har_features/code/lag_select.py baselines/2026-09-13_principled_har_features/test/test_lag_select.py
git commit -m "principled_har: causal GPH long-memory + data-driven HAR window selection"
```

---

### Task 3: Causal feature panel

**Files:**
- Create: `baselines/2026-09-13_principled_har_features/code/build_panel.py`
- Test: `baselines/2026-09-13_principled_har_features/test/test_build_panel.py`

**Interfaces:**
- Consumes: `estimators`, `config`; per-ticker frames with columns `date, open, high, low, close, daily_return, parkinson_variance` (from `FM.load`).
- Produces:
  - `build_panel.FEATURES(windows) -> list[str]` — principled feature names for chosen `windows=(ws,wm,wl)`: `["rv_d","rv_w","rv_m","gk_d","rs_d","semi_neg","semi_pos"]` (+ the HAR aggregates use the windows internally).
  - `build_panel.add_features(frame, windows) -> pd.DataFrame` — adds those columns causally to one ticker frame.
  - `build_panel.feature_frames(frames, windows) -> dict` — maps over all tickers.

- [ ] **Step 1: Write the failing test**

```python
# test_build_panel.py
import numpy as np
import pandas as pd
import build_panel as B


def _frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.cumprod(1 + 0.01 * rng.standard_normal(n))
    o = c * (1 + 0.001 * rng.standard_normal(n))
    h = np.maximum(o, c) * (1 + 0.005 * np.abs(rng.standard_normal(n)))
    l = np.minimum(o, c) * (1 - 0.005 * np.abs(rng.standard_normal(n)))
    pk = np.log(h / l) ** 2 / (4 * np.log(2))
    return pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=n), "open": o, "high": h,
                         "low": l, "close": c, "daily_return": np.r_[0.0, np.diff(np.log(c))],
                         "parkinson_variance": pk})


def test_features_listed_and_added():
    f = B.add_features(_frame(), (3, 5, 10))
    for col in B.FEATURES((3, 5, 10)):
        assert col in f.columns


def test_features_are_causal():
    base = _frame()
    f0 = B.add_features(base.copy(), (3, 5, 10))
    perturbed = base.copy()
    perturbed.loc[40:, ["high", "low", "close", "daily_return"]] *= 1.5   # change the FUTURE
    f1 = B.add_features(perturbed, (3, 5, 10))
    # feature at t=30 (< 40) must be unchanged by future perturbation
    cols = B.FEATURES((3, 5, 10))
    assert np.allclose(f0.loc[30, cols].to_numpy(float), f1.loc[30, cols].to_numpy(float), equal_nan=True)
```

- [ ] **Step 2: Run to verify fail** — `... -m pytest test/test_build_panel.py -q` → FAIL.

- [ ] **Step 3: Implement `build_panel.py`**

```python
"""Causal principled-HAR feature panel: HAR-RV core (Parkinson) + GK/RS daily + realized semivariance."""
import numpy as np
import pandas as pd

import config
import estimators as E


def FEATURES(windows):
    return ["rv_d", "rv_w", "rv_m", "gk_d", "rs_d", "semi_neg", "semi_pos"]


def add_features(frame, windows):
    ws, wm, wl = windows
    d = frame.sort_values("date").reset_index(drop=True).copy()
    pk = d["parkinson_variance"].to_numpy(float)
    s = pd.Series(pk)
    d["rv_d"] = pk
    d["rv_w"] = s.rolling(wm, min_periods=wm).mean().to_numpy()   # mid window = "weekly" scale
    d["rv_m"] = s.rolling(wl, min_periods=wl).mean().to_numpy()   # long window = "monthly" scale
    d["gk_d"] = E.garman_klass(d["open"], d["high"], d["low"], d["close"])
    d["rs_d"] = E.rogers_satchell(d["open"], d["high"], d["low"], d["close"])
    rm, rp = E.semivariance(d["daily_return"].to_numpy(float), ws, config.SEMI_MIN_PERIODS)
    d["semi_neg"], d["semi_pos"] = rm, rp
    return d


def feature_frames(frames, windows):
    return {tk: add_features(fr, windows) for tk, fr in frames.items()}
```
(Note: `ws` is the short window used for the daily-split semivariance; `rv_d` is the 1-day term. The three HAR scales map to `(1, wm, wl)` with `rv_w`/`rv_m` aggregates; `ws` feeds the semivariance smoothing. Adjust the mapping if lag_select returns a window ≤ 1.)

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**

```bash
git add baselines/2026-09-13_principled_har_features/code/build_panel.py baselines/2026-09-13_principled_har_features/test/test_build_panel.py
git commit -m "principled_har: causal feature panel (HAR-RV core + GK/RS + semivariance)"
```

---

### Task 4: DM runner (principled vs own-history) + leave-one-out

**Files:**
- Create: `baselines/2026-09-13_principled_har_features/code/run_har.py`
- Test: `baselines/2026-09-13_principled_har_features/test/test_run_har.py`

**Interfaces:**
- Consumes: `FM.load/panel/gbm/SEEDS/OWN`, `S1.FOLDS/TRAIN_START`, `M.per_obs_qlike`, `ST.date_clustered_dm`, `build_panel`, `lag_select`, `config.FLOOR`.
- Produces: `run_har.run(market, load_fn=None) -> dict` with per-horizon `{n, qlike:{principled,own}, vs_own:{gain_vs_own_pct, dm_p}, leave_one_out:{feat:{gain_vs_full_pct, dm_p}}, windows, gph_d, fit_diagnostics}`.

- [ ] **Step 1: Write the failing smoke test**

```python
# test_run_har.py  (stub loaders + patched folds; small synthetic panel)
import json
import numpy as np
import pandas as pd
import config
import run_har as R


def _frames(nt=25, n=195, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for t in range(nt):
        c = 100 * np.cumprod(1 + 0.01 * rng.standard_normal(n))
        o = c * (1 + 1e-3 * rng.standard_normal(n))
        h = np.maximum(o, c) * (1 + 5e-3 * np.abs(rng.standard_normal(n)))
        l = np.minimum(o, c) * (1 - 5e-3 * np.abs(rng.standard_normal(n)))
        pk = np.log(h / l) ** 2 / (4 * np.log(2))
        df = pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=n), "open": o, "high": h, "low": l,
                           "close": c, "daily_return": np.r_[0.0, np.diff(np.log(c))], "parkinson_variance": pk})
        for col in ["har_daily", "har_weekly", "har_monthly", "rq", "mr_change", "mr_slope5",
                    "mr_slope10", "mr_dev5", "mr_z22"]:
            df[col] = rng.standard_normal(n)
        df["ticker"] = f"T{t}"; df["sector"] = t % 3
        out[f"T{t}"] = df
    return out


def test_run_smoke(monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(R.S1, "FOLDS", ["2015-02-01", "2015-07-01", "2015-08-01", "2100-01-01"])
    monkeypatch.setattr(R, "MIN_ROWS", {"sp500": 30000, "default": 500})  # small for the synthetic panel
    out = R.run("hose", load_fn=lambda m: (_frames(), {}, {}))
    r = out["h1"]
    assert set(r) >= {"n", "qlike", "vs_own", "leave_one_out", "windows", "fit_diagnostics"}
    assert set(r["qlike"]) == {"principled", "own"}
    json.dumps(out)
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement `run_har.py`**

```python
"""Principled HAR feature set vs own-history GBM: walk-forward pooled QLIKE + date-clustered DM + leave-one-out.

Windows chosen ONCE (causal) from the training log-RV preceding the first test fold via lag_select, fixed
across folds. HORIZONS/config imported; HOSE local, SP500 via Colab.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO), str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover
import config  # noqa: E402
import build_panel as BP  # noqa: E402
import lag_select as LS  # noqa: E402
import full_matrix as FM  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402

FL = FM.FL
MIN_ROWS = {"sp500": 30000, "default": 3000}
HORIZONS = (1, 5, 10, 22)


def _choose_windows(frames):
    # pool training log-RV (dates < first test fold) across tickers, choose windows once (causal)
    cut = pd.Timestamp(S1.FOLDS[0])
    logrv = []
    for fr in frames.values():
        m = (fr["date"] >= S1.TRAIN_START) & (fr["date"] < cut)
        logrv.append(np.log(np.maximum(fr.loc[m, "parkinson_variance"].to_numpy(float), FL)))
    series = np.concatenate(logrv) if logrv else np.array([0.0, 0.0])
    return LS.select_windows(series), LS.gph_d(series)


def _pooled(a, feat_cols, h, min_rows):
    embargo = pd.Timedelta(days=int(h * 1.6) + 5)
    preds, yy, dts, last_tr = [], [], [], None
    for k in range(len(S1.FOLDS) - 1):
        ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
        tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
        te = a[(a.date >= ts) & (a.date < tend)]
        if len(te) == 0 or len(tr) < min_rows:
            continue
        preds.append(np.mean([FM.gbm(tr, te, feat_cols, s) for s in FM.SEEDS], 0))
        yy.append(te["y"].to_numpy(float)); dts.append(te["date"].to_numpy()); last_tr = tr
    if not yy:
        return None
    return np.concatenate(yy), np.concatenate(preds), np.concatenate(dts), last_tr


def run(market, load_fn=None):
    load_fn = load_fn or FM.load
    min_rows = MIN_ROWS.get(market, MIN_ROWS["default"])
    frames, _, _ = load_fn(market)
    windows, d_hat = _choose_windows(frames)
    feat = BP.FEATURES(windows)
    framed = BP.feature_frames(frames, windows)
    out = {}
    for h in config.HORIZONS if hasattr(config, "HORIZONS") else HORIZONS:
        a = FM.panel(framed, {}, h)
        models = {"principled": feat, "own": FM.OWN}
        errs = {}
        base = None
        for name, cols in models.items():
            res = _pooled(a, cols, h, min_rows)
            if res is None:
                break
            y, p, dates, last_tr = res
            errs[name] = M.per_obs_qlike(y, p, floor=FL)
            base = (y, dates, last_tr)
        if base is None or "principled" not in errs:
            continue
        y, dates, last_tr = base
        q = {m: float(np.mean(errs[m])) for m in errs}
        dm = float(ST.date_clustered_dm(errs["principled"], errs["own"], dates, h)["p_value"])
        loo = {}
        for f in feat:
            res = _pooled(a, [c for c in feat if c != f], h, min_rows)
            if res is None:
                continue
            yl, pl, dl, _ = res
            el = M.per_obs_qlike(yl, pl, floor=FL)
            ql = float(np.mean(el))
            loo[f] = {"gain_vs_full_pct": (q["principled"] - ql) / q["principled"] * 100.0,
                      "dm_p": float(ST.date_clustered_dm(el, errs["principled"], dl, h)["p_value"])}
        tr_q = {m: float(np.mean(M.per_obs_qlike(last_tr["y"].to_numpy(float),
                np.mean([FM.gbm(last_tr, last_tr, cols, s) for s in FM.SEEDS], 0), floor=FL)))
                for m, cols in models.items()}
        out[f"h{h}"] = {"n": int(len(y)), "qlike": q,
                        "vs_own": {"gain_vs_own_pct": (q["own"] - q["principled"]) / q["own"] * 100.0, "dm_p": dm},
                        "leave_one_out": loo, "windows": list(windows), "gph_d": d_hat,
                        "train_metrics": tr_q,
                        "fit_diagnostics": {m: {"verdict": "overfit" if q[m] > tr_q[m] * 1.25 else "ok",
                                                "train_qlike": tr_q[m], "test_qlike": q[m]} for m in models}}
    return out


def main():  # pragma: no cover - entry driver
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    out = run(market)
    (REPO / "results" / "gamma_gbm" / f"principled_har_{market}.json").write_text(json.dumps(out, indent=2))
    print("saved", market, {h: round(out[h]["vs_own"]["gain_vs_own_pct"], 3) for h in out}, flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 4: Run to verify pass** — `... -m pytest test/test_run_har.py -q` → PASS. Then run the full baseline test suite + coverage:
`... -m pytest test/ --cov=estimators --cov=lag_select --cov=build_panel --cov=run_har --cov-branch --cov-report=term-missing -q` → all pass, C0=100%, C1≥95% (add tests for any uncovered branch, e.g. `_pooled` returns None / `loo` skip).

- [ ] **Step 5: Commit**

```bash
git add baselines/2026-09-13_principled_har_features/code/run_har.py baselines/2026-09-13_principled_har_features/test/test_run_har.py
git commit -m "principled_har: DM runner (principled vs own) + leave-one-out + causal window choice"
```

---

### Task 5: HOSE run, Colab notebook, code review, gate, report

**Files:**
- Create: `notebooks/principled_har_sp500_colab.ipynb`; modify `.gitignore` (allowlist the notebook).
- Create: `results/gamma_gbm/principled_har_hose.json` (real run), `baselines/2026-09-13_principled_har_features/code_review/code_review_2026-09-13.md`, `docs/reports/2026-09-13_<HHMM>_summaryOfUpdate_report.md`.

- [ ] **Step 1: Run HOSE locally**

Run: `/c/luanvan/stock_vol_prediction01/.venv_gpu_encode/Scripts/python.exe baselines/2026-09-13_principled_har_features/code/run_har.py hose`
Record per-horizon `vs_own.gain_vs_own_pct` + `dm_p`, the chosen `windows`, `gph_d`, and `leave_one_out`. Interpret vs the non-inferiority criterion (§2 of the spec).

- [ ] **Step 2: Write the Colab notebook**

Model it on `notebooks/complex_network_sp500_colab.ipynb` (git-clone code + Drive-mount the two SP500 bundles + run `run_har.py sp500` + push the result JSON). Add `!notebooks/principled_har_sp500_colab.ipynb` to `.gitignore`. Validate with `nbformat`.

- [ ] **Step 3: Adversarial code review**

Run `/code-review` (or dispatch a 3-layer review) on the new modules. Focus: formula-exactness of each estimator vs its paper, causality/leakage in `add_features` and `_choose_windows` (windows must use train-only data), the HAR window→feature mapping when `select_windows` returns a window of 1. Fix HIGH/MEDIUM; document in `code_review/`.

- [ ] **Step 4: Summary report + gate + push**

Write `docs/reports/<date>_<HHMM>_summaryOfUpdate_report.md` (what changed, HOSE DM table + non-inferiority verdict, windows/gph_d chosen, leave-one-out pruning, tests+coverage, code-review result, SP500-pending-via-Colab). Then:
```bash
git add baselines/2026-09-13_principled_har_features results/gamma_gbm/principled_har_hose.json \
  notebooks/principled_har_sp500_colab.ipynb .gitignore docs/reports/<date>_<HHMM>_summaryOfUpdate_report.md
git commit -m "principled_har: HOSE DM result (non-inferiority verdict) + SP500 Colab notebook + report"
git push origin master   # pre-push gate: pytest + diff-cover C0/C1 + data-quality + config-hardcode + checklist
```
Expected: quality gate passes. If SP500 is to be run, the user executes the Colab notebook and pushes `principled_har_sp500.json`; then analyse both markets for the final non-inferiority verdict.

---

## Self-Review

**Spec coverage:** §3 feature catalog → Tasks 1,3 (estimators + panel). §4 data-driven causal lags → Task 2 (gph_d/select_windows) + Task 4 `_choose_windows` (train-only, fixed). §5 validation (DM + leave-one-out, HOSE local / SP500 Colab) → Tasks 4,5. §6 deliverables (5 subfolders, tests, notebook, report) → Tasks 1,5. §7 constraints (no FM.OWN change, formula-exact, no intraday, parsimony) → Global Constraints + Task 4 imports read-only + leave-one-out. Covered.

**Placeholder scan:** no TBD/TODO; all code blocks concrete. The only deferred detail is the window→scale mapping note in Task 3 (documented decision, not a placeholder).

**Type consistency:** `FEATURES(windows)`, `add_features(frame,windows)`, `feature_frames`, `select_windows→(ws,wm,wl)`, `gph_d→float`, `run(market,load_fn)→dict` consistent across Tasks 2–5. `MIN_ROWS` patched in the Task 4 test matches its module definition. `config.HORIZONS` guarded (`hasattr`) since config defines the grid only if present — note: add `HORIZONS = (1,5,10,22)` to config.py in Task 1 to avoid the guard; update Step 4 config accordingly.

**Fix applied inline:** add `HORIZONS = (1, 5, 10, 22)` to `config.py` (Task 1 Step 4) and use `config.HORIZONS` directly in `run_har.py` (drop the `hasattr` guard).
