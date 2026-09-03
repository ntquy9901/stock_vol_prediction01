# Pooled/transfer ablation for VN30 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether widening the deep model's training universe from 31 (VN30) to 102 (VN100) stocks improves VN30 volatility forecasts, via a single-variable walk-forward ablation with paired Diebold–Mariano.

**Architecture:** ONE VN100 enriched panel + ONE fold structure. Each arm differs only by a *training-node mask*: Arm 1 trains on all 102 nodes, Arm 0 trains on the 31 VN30 nodes (graph adjacency + training loss restricted to VN30). Both arms score exactly the 31 VN30 nodes on the identical OOS grid → perfect paired alignment by construction. Reuses the delivered VolGA panel reader, masked-rich trainer, and metric/DM helpers read-only.

**Tech Stack:** Python 3.11, numpy, pandas, torch (GPU venv `.venv_gpu_encode`), pytest. Reuses `baselines/2026-08-31_walkforward_volga/code/*` and `baselines/2026-08-21_har_anchored_residual/code/*`.

## Global Constraints

- New baseline dir `baselines/2026-09-04_pooled_transfer_vn30/` with 5 subfolders (SDD §3.F): `requirements/ design/ code/ code_review/ test/`. Hard isolation: import reused modules read-only; edit no other baseline's files.
- No hardcoded tunables — all windows/thresholds/hparams from `pipeline_config as pc` (single source of truth).
- Leakage-safe: graph + all scalers fit on the fold TRAIN window only; reuse `assert_no_leakage`.
- Target = `parkinson_variance` (σ², range-based) at t+h; positivity floor from config.
- Score set fixed = the frozen VN30 universe; both arms score the identical `(ticker, date)` OOS points.
- Decision rule fixed a priori: headline = paired DM Arm1-vs-Arm0 (VolGA and LSTM, 3 loss bases QLIKE/SE/AE); secondary = diff-in-diff of gap(deep−HAR). Report H0/H1 honestly; the Track B A1 (2026-08-08) null is stated in the report.
- Tests: C0 line = 100%, C1 branch ≥ 95% on changed lines (`--cov-branch`, `diff-cover`). ruff `F` clean.
- Run under the GPU venv: `.venv_gpu_encode/Scripts/python.exe`.
- Bootstrap `sys.path` in each script (folder name has `-`; not `python -m`-importable). Run with `python <path>/<script>.py`.

**Reused interfaces (verbatim signatures):**
- `wf_enriched_panel.build_enriched_panel(files, lookback, horizon, keep_tickers) -> EnrichedPanel` (`.tickers`, `.N`, `.anchors`, `.node_ok`, `.target_dates`).
- `wf_enriched_panel.frozen_universe(files, lookback, horizon) -> list[str]`.
- `wf_enriched_panel.pack_fold(panel, fold, lookback, horizon) -> MaskedRichData` (`.adj_vol2pk` [N,N], `.tmask_tr/_va/_te` [n,N], `.nmask_tr` , `.y_te`, `.d_te`, `.N`, `.har_tr/_te`, `.har5_tr/_te`, `.t_mean`).
- `wf_folds.make_folds(n, test_start, K, val, horizon) -> list[Fold]` (`.idx`, `.train`, `.val`, `.forecast` slices); `wf_folds.assert_no_leakage(folds, target_dates, horizon)`.
- `run_masked_rich.train_masked_rich(D, cfg, seed, use_graph, adj, return_splits=True) -> {"train","val","test","train_curve","val_curve","best_epoch"}`.
- `run_masked_rich._pred_dict(pred, y, tmask, dates, N) -> {(node_j, date): (y, pred)}`; `._ens(dicts)`; `._metrics(pred_dict, floor)`; `.seed_metric_stats(seed_dicts, floor)`; `._dm_all(a, b, horizon, floor) -> {"qlike"/"se"/"ae": {"p_value","mean_diff","favors"}}`.
- `run_walkforward._har_ols_preds(D, floor, nfloor) -> (har, harx)` each `{"tr","va","te"}` (fits pooled-OLS on `D.tmask_tr` rows).
- `masked_rich._directed_vol2pk(vshock, sqrt_pk, last_row, top_k) -> adj[N,N]`.
- `run_volga_walkforward.VolgaWFConfig`, `.enriched_glob(market)`, `.training_config(...)`.

---

### Task 1: Baseline scaffold + universe/index helpers

**Files:**
- Create: `baselines/2026-09-04_pooled_transfer_vn30/requirements/requirements.md` (copy objective + success criteria from the spec).
- Create: `baselines/2026-09-04_pooled_transfer_vn30/design/design.md` (copy the design spec).
- Create: `baselines/2026-09-04_pooled_transfer_vn30/code/__init__.py` (empty).
- Create: `baselines/2026-09-04_pooled_transfer_vn30/code/pooled_panel.py` (sys.path bootstrap + helpers).
- Test: `baselines/2026-09-04_pooled_transfer_vn30/code/tests/test_universe.py`.

**Interfaces:**
- Produces: `vn30_index(panel, vn30_tickers) -> np.ndarray[int]` — indices of the VN30 tickers within a VN100 panel's `.tickers`, raising if any VN30 ticker is absent. `screened_universe(market) -> list[str]` — `frozen_universe` over that market's enriched glob at the experiment lookback/horizon.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_universe.py
import sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import pooled_panel as pp

def test_vn30_index_maps_and_is_subset():
    class Panel:  # minimal stand-in
        tickers = ["AAA", "FPT", "VIC", "ZZZ"]
    idx = pp.vn30_index(Panel(), ["FPT", "VIC"])
    assert list(idx) == [1, 2]

def test_vn30_index_raises_on_missing():
    class Panel:
        tickers = ["AAA", "FPT"]
    try:
        pp.vn30_index(Panel(), ["FPT", "MISSING"])
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: Run test to verify it fails** — `.venv_gpu_encode/Scripts/python.exe -m pytest baselines/2026-09-04_pooled_transfer_vn30/code/tests/test_universe.py -v` → FAIL (module/func missing).

- [ ] **Step 3: Write minimal implementation**
```python
# pooled_panel.py
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "baselines" / "2026-08-31_walkforward_volga" / "code",
           _REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           _REPO / "submission" / "soict_lstm_gat"):
    sys.path.insert(0, str(_p))
from wf_enriched_panel import frozen_universe  # noqa: E402
from run_volga_walkforward import enriched_glob  # noqa: E402
import glob as _glob

def vn30_index(panel, vn30_tickers) -> np.ndarray:
    pos = {t: j for j, t in enumerate(panel.tickers)}
    missing = [t for t in vn30_tickers if t not in pos]
    if missing:
        raise ValueError(f"VN30 tickers absent from panel: {missing}")
    return np.array([pos[t] for t in vn30_tickers], dtype=int)

def screened_universe(market: str, lookback: int, horizon: int) -> list:
    files = _glob.glob(enriched_glob(market))
    return frozen_universe(files, lookback, horizon)
```

- [ ] **Step 4: Run test to verify it passes** — same pytest cmd → PASS.

- [ ] **Step 5: Commit**
```bash
git add baselines/2026-09-04_pooled_transfer_vn30/requirements baselines/2026-09-04_pooled_transfer_vn30/design baselines/2026-09-04_pooled_transfer_vn30/code
git commit -m "pooled-vn30: baseline scaffold + universe/index helpers"
```

---

### Task 2: `restrict_fold` — training-node mask + graph restriction

**Files:**
- Modify: `baselines/2026-09-04_pooled_transfer_vn30/code/pooled_panel.py`
- Test: `baselines/2026-09-04_pooled_transfer_vn30/code/tests/test_restrict.py`

**Interfaces:**
- Produces: `restrict_fold(D, train_idx: np.ndarray) -> D2` — returns a shallow copy of the `MaskedRichData` `D` whose `tmask_tr`/`nmask_tr`/`tmask_va`/`nmask_va` columns **outside** `train_idx` are zeroed, and whose `adj_vol2pk` keeps only edges with **both** endpoints in `train_idx` (rows/cols outside zeroed). `tmask_te` is left unchanged (scoring is handled separately). Uses `dataclasses.replace`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_restrict.py
import sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import pooled_panel as pp
from run_masked_rich import _pred_dict  # noqa (path set by pooled_panel import)

def _toy_D():
    import masked_rich as MR
    N = 4
    ones = np.ones((2, N), np.float32)
    adj = np.ones((N, N), np.float32)
    return MR.MaskedRichData(
        tickers=["a","b","c","d"], adj_vol2pk=adj, adj_corr=np.eye(N, dtype=np.float32),
        X_tr=np.zeros((2,N,1,5),np.float32), X_va=np.zeros((2,N,1,5),np.float32), X_te=np.zeros((2,N,1,5),np.float32),
        nmask_tr=ones.copy(), nmask_va=ones.copy(), nmask_te=ones.copy(),
        tmask_tr=ones.copy(), tmask_va=ones.copy(), tmask_te=ones.copy(),
        y_tr=np.ones((2,N)), y_va=np.ones((2,N)), y_te=np.ones((2,N)),
        har_tr=np.zeros((2,N,3)), har_va=np.zeros((2,N,3)), har_te=np.zeros((2,N,3)),
        d_va=["2020-01-01","2020-01-02"], d_te=["2020-01-01","2020-01-02"],
        t_mean=np.ones(N), t_std=np.ones(N),
        har5_tr=np.zeros((2,N,5)), har5_va=np.zeros((2,N,5)), har5_te=np.zeros((2,N,5)))

def test_restrict_zeros_train_outside_idx_and_isolates_graph():
    D = _toy_D()
    D2 = pp.restrict_fold(D, np.array([0, 1]))
    assert D2.tmask_tr[:, 2:].sum() == 0 and D2.tmask_tr[:, :2].sum() == 4
    assert D2.adj_vol2pk[2:, :].sum() == 0 and D2.adj_vol2pk[:, 2:].sum() == 0
    assert D2.adj_vol2pk[:2, :2].sum() == 4          # VN30-block edges preserved
    assert D.tmask_tr[:, 2:].sum() == 4              # original untouched (copy, not in-place)
    assert D2.tmask_te.sum() == D.tmask_te.sum()     # test mask left unchanged
```

- [ ] **Step 2: Run to verify fail** — `pytest .../tests/test_restrict.py -v` → FAIL.

- [ ] **Step 3: Implement**
```python
# append to pooled_panel.py
from dataclasses import replace  # noqa: E402

def restrict_fold(D, train_idx):
    N = D.adj_vol2pk.shape[0]
    keep = np.zeros(N, bool); keep[np.asarray(train_idx, int)] = True
    def zc(m):
        m2 = m.copy(); m2[:, ~keep] = 0.0; return m2
    adj = D.adj_vol2pk.copy(); adj[~keep, :] = 0.0; adj[:, ~keep] = 0.0
    return replace(D, adj_vol2pk=adj,
                   tmask_tr=zc(D.tmask_tr), nmask_tr=zc(D.nmask_tr),
                   tmask_va=zc(D.tmask_va), nmask_va=zc(D.nmask_va))
```

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit** — `git commit -am "pooled-vn30: restrict_fold (train-node mask + graph isolation)"`

---

### Task 3: `score_dicts` — restrict per-fold predictions to VN30

**Files:**
- Modify: `pooled_panel.py`
- Test: `tests/test_score.py`

**Interfaces:**
- Produces: `score_mask(tmask_te, score_idx) -> np.ndarray` — a copy of `tmask_te` with all columns outside `score_idx` zeroed (feed to `_pred_dict` so only VN30 `(node,date)` keys are produced).

- [ ] **Step 1: Failing test**
```python
# tests/test_score.py
import sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import pooled_panel as pp
from run_masked_rich import _pred_dict

def test_score_mask_keeps_only_score_nodes():
    tm = np.ones((2, 4), np.float32)
    sm = pp.score_mask(tm, np.array([1, 3]))
    assert sm[:, [0, 2]].sum() == 0 and sm[:, [1, 3]].sum() == 4
    d = _pred_dict(np.zeros((2,4)), np.ones((2,4)), sm, ["2020-01-01","2020-01-02"], 4)
    assert {j for (j, _) in d} == {1, 3}
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement**
```python
def score_mask(tmask_te, score_idx):
    keep = np.zeros(tmask_te.shape[1], bool); keep[np.asarray(score_idx, int)] = True
    m = tmask_te.copy(); m[:, ~keep] = 0.0; return m
```
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `git commit -am "pooled-vn30: score_mask (restrict scoring to VN30)"`

---

### Task 4: `run_arm` — one arm's walk-forward over the shared folds

**Files:**
- Create: `baselines/2026-09-04_pooled_transfer_vn30/code/run_pooled_arm.py`
- Test: `tests/test_run_arm_smoke.py`

**Interfaces:**
- Consumes: Task 1–3 helpers; reused `pack_fold`, `train_masked_rich`, `_har_ols_preds`, `_pred_dict`, `_ens`, `_metrics`, `seed_metric_stats`.
- Produces: `run_arm(panel, folds, wf, cfg, train_idx, score_idx) -> dict` with keys `metrics` (per-model `_metrics` on VN30 pooled preds), `seed_stats` (LSTM/VolGA), `preds` (per-model VN30 pooled pred dict, for later DM), `evidence` (per-fold n_train/n_val/n_forecast + fit_diagnostics). For Arm 0 pass `train_idx = score_idx` (VN30); for Arm 1 pass `train_idx = arange(N)` (all 102).

- [ ] **Step 1: Write the smoke/behaviour test** (real enriched data, tiny slice; skip if data absent)
```python
# tests/test_run_arm_smoke.py
import sys, glob, numpy as np, pytest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import pooled_panel as pp, run_pooled_arm as ra
from run_volga_walkforward import enriched_glob, VolgaWFConfig, training_config
from wf_enriched_panel import build_enriched_panel
from wf_folds import make_folds
import pipeline_config as pc

def _slice_panel():
    files = sorted(glob.glob(enriched_glob("vn100")))
    if not files: pytest.skip("enriched vn100 absent")
    keep = [Path(f).stem for f in files][:12]
    return build_enriched_panel(files, 22, 1, keep), keep

def test_run_arm_scores_only_score_idx_and_runs():
    panel, keep = _slice_panel()
    wf = VolgaWFConfig(lookback=22, horizon=1, folds_target=1)
    n = len(panel.anchors); ts = int(n * wf.test_frac); K = max(1, n - ts)
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    cfg = training_config(epochs=2, seeds=(0,))
    score_idx = np.array([0, 1, 2])            # first 3 tickers = "vn30" stand-in
    out = ra.run_arm(panel, folds, wf, cfg, np.arange(panel.N), score_idx)
    scored_nodes = {j for (j, _) in out["preds"]["HAR"]}
    assert scored_nodes.issubset({0, 1, 2})    # only score_idx scored
    assert set(out["metrics"]) == {"HAR", "HAR-X", "LSTM", "LSTM_wGAT_vol2pk"}
    assert out["metrics"]["HAR"]["n"] > 0
```

- [ ] **Step 2: Run → FAIL** (module missing).

- [ ] **Step 3: Implement** (mirror delivered `run_fold`/`run_walkforward`, restrict + score)
```python
# run_pooled_arm.py
from __future__ import annotations
import sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import pooled_panel as pp
from wf_enriched_panel import pack_fold
import run_masked_rich as RMR
from run_walkforward import _har_ols_preds
import pipeline_config as pc
_MODELS = ("HAR", "HAR-X", "LSTM", "LSTM_wGAT_vol2pk")

def run_arm(panel, folds, wf, cfg, train_idx, score_idx):
    fl = cfg.qlike_floor
    pooled = {m: {} for m in ("HAR", "HAR-X")}
    lstm_pool = [{} for _ in cfg.seeds]; volga_pool = [{} for _ in cfg.seeds]
    per_fold = []
    for fold in folds:
        D = pack_fold(panel, fold, wf.lookback, wf.horizon)
        Dr = pp.restrict_fold(D, train_idx)
        smask = pp.score_mask(D.tmask_te, score_idx)
        nfloor = pc.POS_FLOOR_FRAC * Dr.t_mean + pc.POS_FLOOR_EPS
        har, harx = _har_ols_preds(Dr, fl, nfloor)
        eye = np.eye(Dr.N, dtype=np.float32)
        lstm = [RMR.train_masked_rich(Dr, cfg, s, False, eye, return_splits=True) for s in cfg.seeds]
        volga = [RMR.train_masked_rich(Dr, cfg, s, True, Dr.adj_vol2pk, return_splits=True) for s in cfg.seeds]
        pooled["HAR"].update(RMR._pred_dict(har["te"], D.y_te, smask, D.d_te, D.N))
        pooled["HAR-X"].update(RMR._pred_dict(harx["te"], D.y_te, smask, D.d_te, D.N))
        for si, o in enumerate(lstm):
            lstm_pool[si].update(RMR._pred_dict(o["test"], D.y_te, smask, D.d_te, D.N))
        for si, o in enumerate(volga):
            volga_pool[si].update(RMR._pred_dict(o["test"], D.y_te, smask, D.d_te, D.N))
        per_fold.append({"idx": fold.idx, "n_train": int(Dr.tmask_tr.sum()),
                         "n_forecast": int(smask.sum())})
    lstm_ens = RMR._ens(lstm_pool); volga_ens = RMR._ens(volga_pool)
    preds = {"HAR": pooled["HAR"], "HAR-X": pooled["HAR-X"], "LSTM": lstm_ens, "LSTM_wGAT_vol2pk": volga_ens}
    metrics = {m: RMR._metrics(preds[m], fl) for m in _MODELS}
    seed_stats = {"LSTM": RMR.seed_metric_stats(lstm_pool, fl),
                  "LSTM_wGAT_vol2pk": RMR.seed_metric_stats(volga_pool, fl)}
    return {"metrics": metrics, "seed_stats": seed_stats, "preds": preds, "per_fold": per_fold}
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `git commit -am "pooled-vn30: run_arm (per-arm walk-forward, restrict+score)"`

---

### Task 5: Isolation test — Arm 0 VN30 preds invariant to non-VN30 features

**Files:** Test only: `tests/test_isolation.py`. (Gate that the single-panel-mask genuinely reproduces a 31-node system; if it fails, the design must fall back to a separate VN30 panel — STOP and flag.)

**Interfaces:** Consumes `run_arm`.

- [ ] **Step 1: Write the test**
```python
# tests/test_isolation.py
import sys, glob, numpy as np, pytest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import run_pooled_arm as ra
from run_volga_walkforward import enriched_glob, VolgaWFConfig, training_config
from wf_enriched_panel import build_enriched_panel
from wf_folds import make_folds

def test_arm0_vn30_preds_independent_of_nonvn30_features():
    files = sorted(glob.glob(enriched_glob("vn100")))
    if not files: pytest.skip("enriched vn100 absent")
    keep = [Path(f).stem for f in files][:8]
    panel = build_enriched_panel(files, 22, 1, keep)
    wf = VolgaWFConfig(lookback=22, horizon=1, folds_target=1)
    n = len(panel.anchors); ts = int(n * wf.test_frac); K = max(1, n - ts)
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    cfg = training_config(epochs=2, seeds=(0,))
    score_idx = np.array([0, 1, 2])            # "vn30" = first 3 nodes
    base = ra.run_arm(panel, folds, wf, cfg, score_idx, score_idx)   # Arm0: train_idx == score_idx
    p1 = {k: v[1] for k, v in base["preds"]["LSTM_wGAT_vol2pk"].items()}
    panel.feats[:, 3:, :] += 5.0               # perturb ONLY non-vn30 nodes' features
    base2 = ra.run_arm(panel, folds, wf, cfg, score_idx, score_idx)
    p2 = {k: v[1] for k, v in base2["preds"]["LSTM_wGAT_vol2pk"].items()}
    for k in p1:
        assert abs(p1[k] - p2[k]) < 1e-4, "Arm0 VN30 preds leaked from non-VN30 nodes"
```

- [ ] **Step 2: Run.** Expected PASS (adj isolation + loss mask sever all cross-node paths). If FAIL → the net mixes nodes outside `adj`; STOP, switch Arm 0 to a separate VN30 panel with date-aligned folds, and revise the plan. Record the outcome in `code_review/`.
- [ ] **Step 3: Commit** — `git commit -am "pooled-vn30: isolation test (Arm0 independent of non-VN30 nodes)"`

---

### Task 6: Alignment + leakage tests over the shared folds

**Files:** Test: `tests/test_alignment.py`.

**Interfaces:** Consumes `run_arm`, `assert_no_leakage`.

- [ ] **Step 1: Write the test** — both arms produce the identical VN30 `(ticker,date)` OOS key set, and the shared folds pass the leakage guard.
```python
# tests/test_alignment.py
import sys, glob, numpy as np, pytest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import run_pooled_arm as ra
from run_volga_walkforward import enriched_glob, VolgaWFConfig, training_config
from wf_enriched_panel import build_enriched_panel
from wf_folds import make_folds, assert_no_leakage

def test_arms_share_identical_vn30_oos_keys_and_no_leakage():
    files = sorted(glob.glob(enriched_glob("vn100")))
    if not files: pytest.skip("enriched vn100 absent")
    keep = [Path(f).stem for f in files][:10]
    panel = build_enriched_panel(files, 22, 1, keep)
    wf = VolgaWFConfig(lookback=22, horizon=1, folds_target=2)
    n = len(panel.anchors); ts = int(n * wf.test_frac); K = max(1, (n - ts) // 2)
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    assert_no_leakage(folds, panel.target_dates, wf.horizon)
    cfg = training_config(epochs=2, seeds=(0,))
    score_idx = np.array([0, 1, 2])
    a1 = ra.run_arm(panel, folds, wf, cfg, np.arange(panel.N), score_idx)  # pooled
    a0 = ra.run_arm(panel, folds, wf, cfg, score_idx, score_idx)           # baseline
    assert set(a1["preds"]["HAR"]) == set(a0["preds"]["HAR"])              # identical OOS keys
```

- [ ] **Step 2: Run → PASS.**
- [ ] **Step 3: Commit** — `git commit -am "pooled-vn30: alignment+leakage tests"`

---

### Task 7: Driver — both arms + paired DM + diff-in-diff + JSON

**Files:**
- Create: `baselines/2026-09-04_pooled_transfer_vn30/code/run_pooled_ablation.py`
- Test: `tests/test_driver.py`

**Interfaces:**
- Produces: `run_ablation(horizon, folds_target=22, epochs=16, market="vn100", vn30_market="vn30", out=None) -> dict` with `arm0`, `arm1` (each a `run_arm` result), `paired_dm` (`{"LSTM": _dm_all(a1,a0), "VolGA": _dm_all(a1,a0)}` on the shared VN30 keys, favors A = pooled better), `diff_in_diff` (per-model gap(deep−HAR) in each arm + Δ), and `meta`. CLI `--horizon {1,5,10,22} --folds-target --epochs --out`. Builds ONE VN100 panel, computes ONE fold set, screens the VN30 universe once.

- [ ] **Step 1: Write the test** (monkeypatch `run_arm` to avoid GPU; assert wiring)
```python
# tests/test_driver.py
import sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import run_pooled_ablation as rp

def test_paired_dm_and_did_wiring(monkeypatch):
    keys = [(0, "2020-01-01"), (1, "2020-01-02")]
    def fake_arm(panel, folds, wf, cfg, train_idx, score_idx):
        base = 0.5 if len(train_idx) > 5 else 0.6   # pooled better
        preds = {m: {k: (1.0, base) for k in keys} for m in ("HAR","HAR-X","LSTM","LSTM_wGAT_vol2pk")}
        return {"metrics": {m: {"qlike": base, "n": 2} for m in preds}, "preds": preds,
                "seed_stats": {}, "per_fold": []}
    monkeypatch.setattr(rp.ra, "run_arm", fake_arm)
    monkeypatch.setattr(rp, "_build", lambda h, ft: ("panel", "folds", "wf", "cfg",
                                                     np.arange(10), np.arange(3)))
    out = rp.run_ablation(1)
    assert "LSTM" in out["paired_dm"] and "VolGA" in out["paired_dm"]
    assert set(out["diff_in_diff"]) == {"LSTM", "LSTM_wGAT_vol2pk"}
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** — `_build(horizon, folds_target)` screens VN30 (`pp.screened_universe("vn30",...)`), builds the VN100 panel over `frozen_universe("vn100")`, `vn30_index`, `make_folds`; `run_ablation` runs both arms, `RMR._dm_all(a1["preds"][m], a0["preds"][m], horizon, floor)` for LSTM & VolGA, computes gap(deep−HAR) per arm + Δ, `assert_no_leakage`, writes JSON to `results/pooled_transfer_vn30/pooled_vn30_h{h}.json`, `argparse` main under `# pragma: no cover`.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `git commit -am "pooled-vn30: driver (both arms, paired DM, diff-in-diff, JSON)"`

---

### Task 8: Coverage + ruff + code review, then run h1 (go/no-go)

**Files:** `code_review/code_review_2026-09-04.md`.

- [ ] **Step 1: Coverage gate** — `.venv_gpu_encode/Scripts/python.exe -m pytest baselines/2026-09-04_pooled_transfer_vn30/code/tests --cov=baselines/2026-09-04_pooled_transfer_vn30/code --cov-branch -q` ; then `diff-cover` on changed lines: C0 = 100%, C1 ≥ 95%. Add tests for any uncovered branch.
- [ ] **Step 2: Lint** — `ruff check --select F baselines/2026-09-04_pooled_transfer_vn30/code` → clean.
- [ ] **Step 3: Adversarial `/code-review`** (3-lens + perf + config-hardcode lens); fix critical/major; note minors in `code_review/`.
- [ ] **Step 4: Run h1 both arms (GPU, detached)** —
```bash
nohup .venv_gpu_encode/Scripts/python.exe -u \
  baselines/2026-09-04_pooled_transfer_vn30/code/run_pooled_ablation.py \
  --horizon 1 --folds-target 22 --epochs 16 \
  > results/pooled_transfer_vn30/_h1.log 2>&1 & disown
```
Poll `results/pooled_transfer_vn30/pooled_vn30_h1.json`. Inspect headline paired DM (VolGA/LSTM Arm1-vs-Arm0, 3 bases) + diff-in-diff. **Go/no-go:** clear signal (either sign) → run h5/h10/h22; else stop with the h1 conclusion.
- [ ] **Step 5: Commit** the h1 result JSON + code_review — `git commit`.

---

### Task 9: Two-arm dashboard + honest report

**Files:**
- Create: `baselines/2026-09-04_pooled_transfer_vn30/code/build_pooled_dashboard.py` (reuse the `embed`/table patterns from `2026-08-31_walkforward_volga/code/build_dashboards.py`, including the Section-1 walk-forward schematic).
- Create: `docs/reports/2026-09-04_pooled_transfer_vn30_dashboard.html` (generated).
- Create: `docs/reports/2026-09-04_pooled_transfer_vn30_report.md` (neutral technical style; state the Track B A1 prior + the a-priori decision rule + honest H0/H1 verdict).

- [ ] **Step 1:** Generate the dashboard from the result JSON(s): Section 1 data-organisation schematic; Arm0-vs-Arm1 metric tables (all 5 metrics × 4 models); headline paired-DM table (3 bases); diff-in-diff table; per-arm fit evidence.
- [ ] **Step 2:** Write the report (what changed, method, results, honest verdict, limitations incl. the sp500 run stopped to free GPU).
- [ ] **Step 3: Commit** — `git commit`.

---

### Task 10: Quality gate + push

- [ ] **Step 1:** Ensure GPU-free window (the h1/other runs finished) before pushing (pre-push step 5 runs GPU baseline tests).
- [ ] **Step 2:** `git push origin master` — the pre-push gate runs (tests, diff-cover C0=100/C1≥95, ruff F, data-quality, config-hardcode, delivered-baseline). Fix any real block; **no `QG_SKIP`** without explicit user consent.
- [ ] **Step 3:** Write `docs/reports/<YYYY-MM-DD_HHMM>_summaryOfUpdate_report.md` (DoD checklist, code-review result, commands run).

## Self-Review

**Spec coverage:** §1 objective → Task 9 report verdict. §2 two-arm single-variable → Tasks 2/4/7 (train-node mask). §3 shared calendar → single-panel design (Tasks 4/6 alignment) — *note: refined from spec's two-panel to one-panel-mask; §3 intent (identical OOS) is met more strongly and is test-gated (Task 6)*; the isolation risk is gated by Task 5. §4 leakage → Task 6 `assert_no_leakage` + reused train-only scalers/graph. §5 metrics/decision rule → Task 7 paired DM + diff-in-diff; §5 overfit evidence → per-fold fit in `run_arm` (extend evidence in Task 7 if the overfit gate needs the full block). §6 code changes/isolation → Task 1 scaffold. §7 testing → Tasks 2/3/5/6 + Task 8 coverage. §8 compute/h1-first → Task 8. §9 deliverables → Tasks 9/10.

**Placeholder scan:** none — all steps carry runnable code/commands.

**Type consistency:** `run_arm(panel, folds, wf, cfg, train_idx, score_idx)` used identically in Tasks 4/5/6/7; `restrict_fold(D, train_idx)`, `score_mask(tmask_te, score_idx)`, `vn30_index(panel, tickers)`, `screened_universe(market, lookback, horizon)` consistent across tasks; pred dicts keyed `(node_j, date)` consistent with `_pred_dict`/`_dm_all`.

**Gap flagged:** the overfit-evidence gate (`check_overfit_evidence.py`) expects `train_metrics`/`val_metrics`/`fit_diagnostics`/`learning_curves` in a *masked_rich result.json*. This ablation writes a different JSON shape → confirm in Task 7 whether the pre-push overfit gate inspects `results/pooled_transfer_vn30/*.json` (it keys on masked_rich result files); if it does, emit the evidence block per arm; if not, no action. Verify before Task 10 push.
