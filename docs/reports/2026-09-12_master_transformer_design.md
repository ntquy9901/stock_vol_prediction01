---
title: "MASTER transformer: design and code"
---

# MASTER (Market-Guided Stock Transformer) — reimplementation design and code

**Purpose of this document.** Detailed design and code walkthrough of our reimplementation of
MASTER (Market-Guided Stock Transformer, Li et al., AAAI 2024) adapted to our daily Parkinson-variance
forecasting task, for discussion with the advisor. It quotes the real code and design in the repository
and reports the actual measured outcome.

**Upstream reference.** MASTER: Li, Wang, Yang, Luo, Qu, Wang, Zhang, Zhou, "MASTER: Market-Guided Stock
Transformer for Stock Price Forecasting", AAAI 2024, arXiv:2312.15235. Reference implementation:
<https://github.com/SJTU-DMTai/MASTER> (MIT License), vendored under `_master_ref/`.

**Where the experiment lives.** `baselines/2026-09-06_master_transformer/` (code, tests, review,
requirements, design) and `results/master_transformer/*.json` (8 runs: VN30 and VN100 × h ∈ {1,5,10,22}).
The full prose report is `docs/reports/2026-09-06_master_transformer_report.md`.

---

## 1. Overview

### What MASTER is

MASTER is a transformer for cross-sectional stock forecasting built on three mechanisms, all in the
upstream `master.py`:

1. **Market-guided gating.** A market-status vector (index price + volume aggregates over several
   horizons) is mapped by a linear layer then `softmax(·/beta)` to a per-feature scaling vector
   `alpha(m) ∈ R^F`; the input features are rescaled `x_tilde = alpha(m) ⊙ x`. The market regime decides
   which stock features matter (regime-adaptive feature selection). `beta` is a temperature controlling
   the sharpness of selection.
2. **Intra-stock temporal attention (TAttention).** Per-stock transformer self-attention over TIME,
   keeping a sequence of local embeddings (not collapsed). This replaces an LSTM temporal encoder.
3. **Inter-stock (cross-sectional) attention (SAttention).** At each timestep, all-pairs self-attention
   ACROSS stocks — learned, with no predefined graph. This replaces a GAT over fixed edges.

A **temporal aggregation** (attention pooling over time, query = last step) collapses the sequence to
one embedding per stock, and a **linear head** produces the forecast.

The upstream task is cross-sectional **return ranking** (predict a d-day normalized return ratio, rank
stocks), evaluated by IC / RankIC / ICIR + portfolio metrics on CSI300/CSI800 with Alpha158 factors +
63 market indicators, lookback τ=8, horizon d=5.

### Why we tried it on our volatility panel

Our task is daily **Parkinson-variance** (σ²) regression at t+h on VN30/VN100, evaluated by
MSE/RMSE/MAE/QLIKE/R² with date-clustered Diebold-Mariano (DM). The project prior is strong: graph /
attention structure has repeatedly failed to beat the linear HAR-X on this noisy daily target, and our
VolGA model (LSTM + a *sparse* horizon-matched vol→PK GAT edge) only wins at h1. MASTER offers a clean
contrast to test: **dense learned cross-sectional attention** (softmax over ALL stocks) plus a
**market-volatility-regime gate** we had never tried. The motivation and mapping are laid out in
`docs/reports/2026-09-05_MASTER_analysis_application.md`, which ranked the market-guided gate as the
highest-value, lowest-cost idea to port, and the dense inter-stock attention as a medium-value / medium-risk
second probe, tempered by the fact that prior learned adjacencies (MTGNN, Graph-WaveNet-adaptive) were
already null on thin VN data.

A **negative** result (MASTER also fails to beat HAR-X, or its dense attention loses to VolGA's sparse
edge) was declared a valid, publishable finding up front; the narrative was not to be optimised
(`requirements/requirements.md`).

### What our reimplementation adapts

| | Upstream MASTER | Our reimplementation |
|---|---|---|
| Target | d-day return **ranking** | Parkinson **variance** regression at t+h |
| Loss / metric | MSE on z-scored labels; IC/RankIC | masked MSE (z-scored target); QLIKE + date-clustered DM |
| Universe | CSI300/CSI800 (300–800 stocks) | VN30/VN100 (33 / 102 stocks) |
| Features | 158 factors + 63 market | 5 stock node features + 4 causal market features |
| Cross-stock | learned per-timestep dense attention | same dense attention (kept), vs VolGA's sparse vol→PK edge |
| Temporal | transformer (TAttention) | transformer (TAttention), kept |
| Positivity | (ranking, N/A) | linear output + inverse z-score + shared positivity floor |

The IC/RankIC results do not transfer; what we port are the **architectural ideas**, tempered to a
small-N, few-feature, regression setting with a QLIKE objective and a strict leakage-safe walk-forward.

---

## 2. Data organization (train / validation / test)

### Panel and sequence construction

Inputs are the enriched causal panels `data/processed_enriched/{vn30,vn100}/<ticker>.csv`, each carrying
the 5 stock node features `[parkinson_variance, har_weekly, har_monthly, market_pk, volume_zscore_22]`.
They are read READ-ONLY through the delivered `build_enriched_panel`. Each **anchor** (one trading date t)
is one MASTER sample: the cross-section of N stocks, each with a T=lookback window of D features. The
batched forward consumes `[B, N, T, D]` (B anchors).

The data flow (from `design/design.md`):

```text
# baselines/2026-09-06_master_transformer/design/design.md
enriched CSVs ──build_enriched_panel──▶ EnrichedPanel(pk[T,N], feats[T,N,5], anchors[A], masks, target_dates)
                                          │
              frozen_universe (train-row screen, once)
                                          │
   make_folds(n, test_start, K, val, h) ──▶ expanding-window folds (assert_no_leakage)
                                          │  per fold:
   pack_fold ──▶ D: X_*[A,N,T,5] (train-only per-node feature scaler), y_*[A,N] (raw pk[t+h]),
                    tmask/nmask, t_mean/t_std[N], d_va/d_te, har_*/har5_*  (all delivered, READ-ONLY)
   market_features.compute_market_raw(pk, vshock, window) ──▶ mkt_raw[T,4] (causal)
   market_features.fit_market_scaler(mkt_raw, last_train_anchor) ──▶ (mean,std)  (TRAIN-only)
   market_features.pack_market(mkt_std, anchors, lookback) ──▶ Xm_*[A,T,4]
```

### The split, retrain cadence and seeds

The walk-forward is an **expanding window** with `folds_target=7` retrain points over the OOS region,
lookback `lb=10` (canonical), a validation block for early stopping, and a purge equal to the horizon.
The exact fold arithmetic in `run_master.run`:

```python
# baselines/2026-09-06_master_transformer/code/run_master.py  (run)
wf = VolgaWFConfig(lookback=lookback, horizon=horizon, folds_target=folds_target)
n = len(panel.anchors)
ts = int(n * wf.test_frac)
K = max(1, math.ceil((n - ts) / wf.folds_target))
folds = make_folds(n, ts, K, wf.val, wf.horizon)
assert_no_leakage(folds, panel.target_dates, wf.horizon)
```

Within each fold, `pack_fold` returns train/val/test tensors `X_*[A,N,T,5]`, targets `y_*[A,N]` (raw
`pk[t+h]`), the target-validity masks `tmask_*`, the node-validity masks `nmask_*`, per-node target
mean/std `t_mean/t_std`, and the OOS dates `d_te`. Feature and target scalers are fit on **train only**.
The 5 seeds are the canonical `(42, 123, 2026, 7, 2024)`; per-seed predictions are ensembled for the
pooled metrics and DM.

**Fairness.** HAR, HAR-X, LSTM, VolGA and MASTER are all trained on the SAME folds, seeds, lookback,
per-node train-only scalers and the SAME shared QLIKE positivity floor, so every DM test is exactly
paired. `assert_no_leakage` runs each fold.

---

## 3. Model architecture

The network is vendored from the upstream `master.py` into `code/master_net.py` and **adapted** to add a
leading anchor-batch dimension `B` (documented in the file header for honesty; the per-anchor math is
identical, pinned by a test that a B=1 batched forward equals the reference).

### Forward pass, step by step

The top-level `MASTER.forward` takes `x [B, N, T, d_feat + d_gate]` and an optional stock-validity
`key_mask [B, N]`:

```python
# baselines/2026-09-06_master_transformer/code/master_net.py  (MASTER.forward)
def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
    # x [B, N, T, d_feat + d_gate]; key_mask [B, N] (1=valid stock)
    src = x[:, :, :, :self.gate_input_start_index]                       # [B, N, T, d_feat]
    gate_input = x[:, :, -1, self.gate_input_start_index:self.gate_input_end_index]   # [B, N, d_gate]
    src = src * self.feature_gate(gate_input).unsqueeze(2)               # gate broadcasts over T
    h = self.pos(self.proj(src))
    h = self.tattn(h)
    h = self.sattn(h, key_mask)
    h = self.temporal(h)
    return self.decoder(h).squeeze(-1)                                   # [B, N]
```

Step by step:

1. **Split.** `src` is the 5 stock features; `gate_input` is the 4 market features read at the LAST
   timestep only (the current, causal market state). `gate_input_start_index = 5`,
   `gate_input_end_index = 9`.
2. **Market gate.** `feature_gate(gate_input)` → a per-feature soft weight, broadcast over T and applied
   to `src`.
3. **Project + positional encoding.** `proj: Linear(d_feat → d_model)` then sinusoidal `PositionalEncoding`
   over time.
4. **Intra-stock temporal attention** `tattn` over T.
5. **Dense inter-stock attention** `sattn` over N (masked by `key_mask`).
6. **Temporal aggregation** `temporal` → one `[B, N, d_model]` vector per stock.
7. **Linear decoder** → `[B, N]` variance forecast per stock.

### Market-guided gate

```python
# baselines/2026-09-06_master_transformer/code/master_net.py  (Gate)
class Gate(nn.Module):
    """Market-guided feature gate: market features -> soft per-feature weight (reference master.py)."""
    def __init__(self, d_input: int, d_output: int, beta: float = 1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta
    def forward(self, gate_input: torch.Tensor) -> torch.Tensor:
        output = torch.softmax(self.trans(gate_input) / self.t, dim=-1)
        return self.d_output * output
```

The softmax over the `d_output = d_feat = 5` feature axis, scaled by temperature `beta`, produces a
weighting that sums to `d_output`, so it re-weights the 5 stock features by the market regime.

### Intra-stock temporal attention (TAttention)

Multi-head self-attention over the T axis, batched over `(B, N)` — every stock's own sequence attends to
itself in time:

```python
# baselines/2026-09-06_master_transformer/code/master_net.py  (TAttention.forward)
def forward(self, x: torch.Tensor) -> torch.Tensor:      # x [B, N, T, d_model]
    x = self.norm1(x)
    q, k, v = self.qtrans(x), self.ktrans(x), self.vtrans(x)
    dim = self.d_model // self.nhead
    outs = []
    for i in range(self.nhead):
        sl = slice(i * dim, None) if i == self.nhead - 1 else slice(i * dim, (i + 1) * dim)
        qh, kh, vh = q[..., sl], k[..., sl], v[..., sl]   # [B, N, T, dim]
        att = torch.softmax(torch.matmul(qh, kh.transpose(-1, -2)), dim=-1)   # [B, N, T, T]
        if self.attn_dropout is not None:
            att = self.attn_dropout[i](att)
        outs.append(torch.matmul(att, vh))               # [B, N, T, dim]
    att_output = torch.cat(outs, dim=-1)
    xt = self.norm2(x + att_output)
    return xt + self.ffn(xt)
```

Scores are `[B, N, T, T]`; residual + LayerNorm + FFN as in a standard transformer block.

### Inter-stock / cross-sectional attention (SAttention)

The defining mechanism: at each timestep every stock attends to EVERY other stock in the same anchor's
cross-section (dense softmax over N), the contrast to VolGA's sparse Top-5 edge. Stocks are permuted to
the second-to-last axis so attention runs over N within each `(B, T)`; invalid (non-trading) stocks are
masked out as keys:

```python
# baselines/2026-09-06_master_transformer/code/master_net.py  (SAttention.forward)
def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
    # x [B, N, T, d_model]; key_mask [B, N] (1=valid stock, 0=invalid/non-trading -> excluded as a key).
    x = self.norm1(x)
    # move stocks to the second-to-last axis so attention runs over N within each (B, T)
    q = self.qtrans(x).permute(0, 2, 1, 3)               # [B, T, N, d_model]
    k = self.ktrans(x).permute(0, 2, 1, 3)
    v = self.vtrans(x).permute(0, 2, 1, 3)
    dim = self.d_model // self.nhead
    # invalid KEY stocks get -inf logits so valid stocks never attend to zero-padded non-trading stocks
    neg = None if key_mask is None else (key_mask < 0.5).view(key_mask.shape[0], 1, 1, key_mask.shape[1])
    outs = []
    for i in range(self.nhead):
        sl = slice(i * dim, None) if i == self.nhead - 1 else slice(i * dim, (i + 1) * dim)
        qh, kh, vh = q[..., sl], k[..., sl], v[..., sl]   # [B, T, N, dim]
        logit = torch.matmul(qh, kh.transpose(-1, -2)) / self.temperature   # [B, T, N, N]
        if neg is not None:
            logit = logit.masked_fill(neg, float("-inf"))
        att = torch.softmax(logit, dim=-1)
        att = torch.nan_to_num(att, nan=0.0)             # invalid TARGET rows (all-masked) -> 0 (discarded)
        att = self.attn_dropout[i](att)
        outs.append(torch.matmul(att, vh))               # [B, T, N, dim]
    att_output = torch.cat(outs, dim=-1).permute(0, 2, 1, 3)   # back to [B, N, T, d_model]
    xt = self.norm2(x + att_output)
    return xt + self.ffn(xt)
```

The cross-sectional scores are `[B, T, N, N]` — this is the VRAM-bounding tensor, kept modest by
`d_model=128` and `batch≤64` for N=102. Masking invalid keys is the fair analogue of VolGA's masked
adjacency.

### Temporal aggregation and head

```python
# baselines/2026-09-06_master_transformer/code/master_net.py  (TemporalAttention.forward)
def forward(self, z: torch.Tensor) -> torch.Tensor:      # z [B, N, T, d_model]
    h = self.trans(z)
    query = h[:, :, -1, :].unsqueeze(2)                  # [B, N, 1, d_model]
    lam = (h * query).sum(-1)                            # [B, N, T]
    lam = torch.softmax(lam, dim=-1).unsqueeze(-1)       # [B, N, T, 1]
    return (lam * z).sum(2)                              # [B, N, d_model]
```

Attention pooling with the last timestep as query collapses T; `decoder = nn.Linear(d_model, 1)` maps to
one scalar variance forecast per stock, giving `[B, N]`.

### Construction and dimensions

```python
# baselines/2026-09-06_master_transformer/code/master_net.py  (build_master)
def build_master(cfg) -> MASTER:
    return MASTER(d_feat=cfg.n_stock_feat, d_model=cfg.d_model, t_nhead=cfg.t_nhead, s_nhead=cfg.s_nhead,
                  t_dropout=cfg.dropout, s_dropout=cfg.dropout, gate_input_start_index=cfg.n_stock_feat,
                  gate_input_end_index=cfg.n_stock_feat + cfg.n_market_feat, beta=cfg.beta,
                  max_len=cfg.max_len)
```

Resolved dimensions (`master_config.py`): `d_model=128`, `t_nhead=4`, `s_nhead=2`, `dropout=0.2`,
`beta=2.0`, `n_stock_feat=5`, `n_market_feat=4`, `max_len=100`.

---

## 4. Market features (the market-status vector)

The 4 causal market features are built in `code/market_features.py`. All use only information up to day t.

### Cross-sectional aggregates (per date, all-NaN → neutral 0)

```python
# baselines/2026-09-06_master_transformer/code/market_features.py  (_cross_sectional)
def _cross_sectional(pk: np.ndarray, vshock: np.ndarray):
    valid = np.isfinite(pk)
    any_valid = valid.any(axis=1)
    m_mean = np.zeros(pk.shape[0]); m_disp = np.zeros(pk.shape[0]); m_vshock = np.zeros(pk.shape[0])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN rows -> handled explicitly below
        mm = np.nanmean(np.where(valid, pk, np.nan), axis=1)
        md = np.nanstd(np.where(valid, pk, np.nan), axis=1)
        mv = np.nanmean(np.where(np.isfinite(vshock), vshock, np.nan), axis=1)
    m_mean[any_valid] = mm[any_valid]
    m_disp[any_valid] = md[any_valid]
    mv_valid = np.isfinite(vshock).any(axis=1)
    m_vshock[mv_valid] = mv[mv_valid]
    return m_mean, m_disp, m_vshock
```

### Causal trailing z of the market vol level

```python
# baselines/2026-09-06_master_transformer/code/market_features.py  (causal_rolling_z)
def causal_rolling_z(series: np.ndarray, window: int) -> np.ndarray:
    """Causal trailing z-score over [max(0, t-window+1) .. t] (min_periods=1). No look-ahead."""
    v = np.asarray(series, dtype=float)
    n = len(v)
    z = np.zeros(n)
    for t in range(n):
        w = v[max(0, t - window + 1):t + 1]
        s = w.std()
        z[t] = (v[t] - w.mean()) / s if s > 0 else 0.0
    return z
```

The 4 features (`compute_market_raw` stacks them into `[T, 4]`):

1. `m_mean` — cross-sectional mean of Parkinson variance over stocks (market vol level).
2. `m_disp` — cross-sectional dispersion (std) over stocks (market dispersion).
3. `m_vshock` — cross-sectional mean of the volume z-shock (market volume shock).
4. `m_volz` — causal 22-day rolling z of `m_mean` (how extreme today's market vol is).

### Leakage-safe (train-only) scaler

The market scaler is fit on rows `[0 : last_train_anchor + 1]` — every train input window ends at or
before the last train anchor, so no val/test date enters the scaler:

```python
# baselines/2026-09-06_master_transformer/code/market_features.py  (fit_market_scaler)
def fit_market_scaler(mkt_raw, train_end_row, eps):
    if not 0 <= train_end_row < len(mkt_raw):
        raise ValueError(f"train_end_row {train_end_row} out of range for T={len(mkt_raw)}")
    block = mkt_raw[:train_end_row + 1]
    mean = block.mean(axis=0)
    std = block.std(axis=0) + eps
    return mean, std
```

`standardize` maps any residual non-finite to 0 (neutral), and `pack_market` builds the per-anchor
window `mkt_std[t - lookback + 1 : t + 1]` (all dates ≤ t, causal). The market row is the same for every
stock at a given date, broadcast across N by the caller (`_cat_input`).

---

## 5. Training and evaluation

### Loss, optimizer, early stopping (mirrors `train_masked_rich`)

MASTER is trained with masked MSE on the per-node **z-scored** target, a LINEAR decoder output (no
Softplus/ReLU), and inverse-transform + a shared positivity floor at eval — the project's proven pattern:

```python
# baselines/2026-09-06_master_transformer/code/run_master.py  (train_master, epoch loop)
for ep in range(cfg.epochs):
    net.train()
    idx = rng.permutation(len(xtr))
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        xb, kmb, tmb, yb = xtr[b], kmtr[b], tmtr[b], ytr_n[b]
        opt.zero_grad()
        pred = net(xb, kmb)
        loss = (((pred - yb) ** 2) * tmb).sum() / tmb.sum().clamp(min=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
    pva = infer(xva, kmva)
    mva = D.tmask_va.astype(bool)
    vmse = float(np.mean((pva[mva] - D.y_va[mva]) ** 2))
    ...
    sched.step(vmse)
    if vmse < best - 1e-12:
        best = vmse
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        wait = 0; best_ep = ep + 1
    else:
        wait += 1
    if ep + 1 >= cfg.min_epochs and wait >= cfg.patience:
        break
```

Optimizer is `Adam(lr=5e-4, weight_decay=1e-5)` with `ReduceLROnPlateau(factor=0.5, patience=2)`,
`grad_clip=1.0`, `epochs=16` max, `patience=5`, `min_epochs=5`, anchor `batch_size=64`. The positivity
floor and inverse z-score:

```python
# baselines/2026-09-06_master_transformer/code/run_master.py  (_floor)
def _floor(pn, t_mean, t_std, cfg):
    """Inverse per-node z-score then apply the shared positivity floor (identical basis to LSTM/VolGA)."""
    return np.maximum(pn * t_std + t_mean, cfg.pos_floor_frac * t_mean + cfg.pos_floor_eps)
```

**Performance.** The concatenated `[A,N,T,D]` inputs and key masks are built ONCE per split as GPU
tensors; the epoch loop and per-epoch learning-curve inference index minibatch views on the GPU (no
per-item batch=1, no per-epoch host re-allocation — see code-review F10 below).

### Metrics and DM

For each fold, HAR/HAR-X (OLS), LSTM (no graph), VolGA (LSTM + horizon-matched vol→PK GAT) and MASTER are
all trained; predictions are pooled per-(node,date) over the whole OOS region. Reported metrics per model:
MSE/RMSE/MAE/QLIKE/R², non-lock conditional QLIKE (drop limit-lock obs where target ≤ 2·floor), and
per-obs win-rate vs HAR-X. The comparison is date-clustered Diebold-Mariano on per-obs QLIKE:

```python
# baselines/2026-09-06_master_transformer/code/run_master.py  (run, DM block)
dm = {"MASTER_vs_HARX": RMR._dm_all(ens["MASTER"], pooled["HAR-X"], horizon, fl),
      "MASTER_vs_LSTM": RMR._dm_all(ens["MASTER"], ens["LSTM"], horizon, fl),
      "MASTER_vs_VolGA": RMR._dm_all(ens["MASTER"], ens["VolGA"], horizon, fl),
      "VolGA_vs_HARX": RMR._dm_all(ens["VolGA"], pooled["HAR-X"], horizon, fl),
      "LSTM_vs_HARX": RMR._dm_all(ens["LSTM"], pooled["HAR-X"], horizon, fl)}
```

The DM convention "favours A if mean_diff < 0" means A is the first (MASTER); a positive mean-diff and
"favors B" means the baseline is better. Every learned model's result JSON carries `train_metrics`,
`val_metrics`, test `metrics`, per-model `fit_diagnostics`, and per-fold/seed `learning_curves`, so the
overfit gate can verify no over/under-fit.

**Pipeline validation.** The in-run HAR-X/LSTM/VolGA QLIKE reproduce the delivered
`results/edge_hmatched/` numbers exactly (VN30 h1: HAR-X 0.4801, LSTM 0.4787, VolGA 0.4740 — identical),
confirming identical folds/seeds/config.

---

## 6. Result

**Outcome: NO-GO. MASTER does not beat HAR-X (nor LSTM, nor VolGA) on QLIKE at any horizon on either
panel.** It is the worst of the five models at every VN30 horizon, and the only DM tests that reach
significance are MASTER **losing** to VolGA at h1 (VN30 p=0.016, VN100 p=0.006) — the sparse learned edge
significantly beats the dense attention at exactly the horizon most favourable to cross-sectional
structure.

Pooled OOS QLIKE (lower is better); "DM vs X" gives the p-value and which model it favours. Numbers from
`results/master_transformer/master_<market>_h<h>.json`.

| Market | h | HAR-X | LSTM | VolGA | **MASTER** | DM MASTER vs HAR-X | DM MASTER vs VolGA | DM MASTER vs LSTM |
|---|---|---|---|---|---|---|---|---|
| VN30 | 1 | 0.4801 | 0.4787 | 0.4740 | **0.4910** | p=0.165 favors HAR-X | **p=0.016 favors VolGA** | p=0.093 favors LSTM |
| VN30 | 5 | 0.5602 | 0.5738 | 0.5672 | **0.6436** | p=0.179 favors HAR-X | p=0.107 favors VolGA | p=0.108 favors LSTM |
| VN30 | 10 | 0.6091 | 0.6048 | 0.6061 | **0.6416** | p=0.337 favors HAR-X | p=0.117 favors VolGA | p=0.135 favors LSTM |
| VN30 | 22 | 0.6782 | 0.6832 | 0.6882 | **0.6973** | p=0.687 favors HAR-X | p=0.595 favors VolGA | p=0.583 favors LSTM |
| VN100 | 1 | 0.5000 | 0.5155 | **0.4879** | 0.5053 | p=0.562 favors HAR-X | **p=0.006 favors VolGA** | p=0.188 favors MASTER |
| VN100 | 5 | **0.5607** | 0.5759 | 0.5662 | 0.5697 | p=0.555 favors HAR-X | p=0.476 favors VolGA | p=0.214 favors MASTER |
| VN100 | 10 | **0.5999** | 0.6127 | 0.6127 | 0.6303 | p=0.300 favors HAR-X | p=0.093 favors VolGA | p=0.093 favors LSTM |
| VN100 | 22 | **0.6385** | 0.6452 | 0.6476 | 0.6628 | p=0.457 favors HAR-X | p=0.272 favors VolGA | p=0.255 favors LSTM |

Best model per row in bold in the QLIKE columns; MASTER's QLIKE is bolded to mark it. VolGA is best at
VN100 h1, HAR-X is best at VN100 h5/h10/h22, and no MASTER cell is best. MASTER's fit diagnostics are
`ok` (no over/under-fit) at the pooled level for every run.

**VN30 h1 detail** (the horizon most favourable to cross-sectional structure): MASTER pooled QLIKE 0.4910
is the WORST of five (HAR 0.4779, VolGA 0.4740, LSTM 0.4787, HAR-X 0.4801). Non-lock QLIKE tells the same
story (MASTER 0.4566 vs VolGA 0.4396, HAR-X 0.4455). MASTER's per-obs QLIKE beats HAR-X on only 52.9% of
ticker-days yet loses on the aggregate because a minority of large errors dominate. Per-seed QLIKE mean is
0.5718 ± 0.0467.

**Interpretation.** A market-guided dense cross-sectional transformer underperforms both a sparse learned
edge (VolGA) and a linear HAR-X on this noisy daily variance target; the dense attention adds noise, not
signal. This is consistent with the project-wide prior that dense/learned graph structure does not beat
HAR-X here. The finding is honest and negative — a valid result under the pre-registered go/no-go.

---

## 7. Code review findings

Self-review (`code_review/code_review_2026-09-06.md`, 3 lenses) plus the coordinator's `/code-review` and
pre-push gate. No unresolved critical/major findings. Summary:

- **F1–F4 (leakage / fairness, all OK).** Per-node feature and target scalers come from `pack_fold`
  (train-only); the market scaler is fit on `mkt_raw[0 : last_train_anchor+1]` (test pins the row range);
  market features are causal (truncation test); folds reuse `make_folds` / `assert_no_leakage`; the VolGA
  edge is the horizon-matched train-only edge.
- **F5 (design choice, documented).** The dense SAttention masks invalid (non-trading) stocks as keys
  via `key_mask=nmask`, the fair analogue of VolGA's masked adjacency.
- **F6 (correctness, OK).** Batched attention does not mix anchors —
  `test_per_anchor_independence_of_batched_attention` perturbs one anchor and asserts the other's output
  is unchanged.
- **F7 (positivity, OK).** Positivity via inverse-transform + shared floor (no Softplus/ReLU); QLIKE uses
  the single shared `qlike_floor=1e-8`, identical across all five models.
- **F8 / F9 (edge cases, handled).** An all-masked key set yields a NaN softmax that `torch.nan_to_num`
  zeroes; all-NaN cross-sectional rows → neutral 0 market feature.
- **F10 (performance, fixed).** The initial loop rebuilt the concatenated `[A,N,T,D]` input every
  epoch/inference call (a batch/GPU-underutilization anti-pattern). Fixed: inputs + key masks are built
  ONCE per split as GPU tensors; the epoch loop indexes minibatch views. Anchors are batched, tensors
  stay on GPU, no per-item batch=1.
- **F11–F13 (config / hygiene / isolation, OK).** All MASTER tunables live in `master_config.py`; floors/
  windows/seeds/lookback come from `pipeline_config` (no magic numbers). `run()`/`main()` are thin
  `# pragma: no cover` drivers with logic in tested helpers. Hard isolation: only read-only imports from
  other baselines; the MASTER nn.Module is vendored with an attribution header (arXiv:2312.15235 + MIT
  SJTU-DMTai repo).

Tests (`test/test_master.py`, 23 tests) cover forward shape, gate slicing (last-timestep-only), per-anchor
independence of batched attention, key masking, reproducibility, causal market features, train-only
scaler fit, positive floored output, the driver helpers, and two end-to-end tiny-fold smoke tests.

---

## 8. Where this lives in the repo

- `baselines/2026-09-06_master_transformer/requirements/requirements.md` — objective, inputs/outputs,
  success criteria, go/no-go.
- `baselines/2026-09-06_master_transformer/design/design.md` — data flow, `[N,T,D]` + gate design,
  normalization, batching, config gates, comparison plan.
- `baselines/2026-09-06_master_transformer/code/master_net.py` — vendored + batched MASTER nn.Module.
- `baselines/2026-09-06_master_transformer/code/master_config.py` — `MasterConfig` (single source of
  tunables).
- `baselines/2026-09-06_master_transformer/code/market_features.py` — causal market features + train-only
  scaler + per-anchor packing.
- `baselines/2026-09-06_master_transformer/code/run_master.py` — walk-forward driver (train all five
  models, pool, metrics, DM, evidence, JSON).
- `baselines/2026-09-06_master_transformer/code_review/code_review_2026-09-06.md` — review findings.
- `baselines/2026-09-06_master_transformer/test/test_master.py` — tests.
- `results/master_transformer/master_{vn30,vn100}_h{1,5,10,22}.json` — the 8 result files.
- `docs/reports/2026-09-05_MASTER_analysis_application.md` — analysis / application report (motivation +
  mapping to our task).
- `docs/reports/2026-09-06_master_transformer_report.md` — the full prose result report.
- `_master_ref/` — the vendored MIT-licensed upstream reference clone.

### Reproduce

```bash
.venv_gpu_encode/Scripts/python.exe baselines/2026-09-06_master_transformer/code/run_master.py --market vn30 --horizon 1
.venv_gpu_encode/Scripts/python.exe -m pytest baselines/2026-09-06_master_transformer/test -q
```
