# Track-A GAT + node-features + volume→PK edge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Track-A-style GNN (real multi-head GAT branch, concatenated) using the E2 node
features (HAR + MarketPK + volume_zscore_20) + PhoBERT news + a directed volume→PK Top-5 edge, and
test whether the graph adds out-of-sample value (graph-on vs graph-off, nested) and vs HAR — at 15
epochs (1 seed first, then 3), with train checkpoint + resume.

**Architecture:** See `baselines/2026-08-15_trackA_gat_edge/design/ARCHITECTURE.md`. Reuse the
existing leakage-safe basis (5 node features + news + frozen vol→PK Top-5 graph snapshots) from the
`combo`/`eda_gnn` pipelines; the only NEW model code is `TrackAGatModel` (self-written multi-head
GAT concat branch) + a checkpoint/resume training loop. Ablation nested on one checkpoint:
NODE=graph-off (adjacency=identity), GNN=graph-on (adjacency=vol→PK).

**Tech Stack:** Python 3.10 (`.venv_gpu_encode`, torch 2.6+cu124, no torch_geometric → GAT written
by hand), pytest, ruff, diff-cover. Reuse `baselines/2026-08-14_pooled_news_edanode_gnn/code`
(combo build_basis), `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code` (data/models/
run_pilot helpers), `baselines/2026-08-11_eda_gnn_baseline/code` (features/edges).

---

## File Structure

- Create `baselines/2026-08-15_trackA_gat_edge/code/gat.py` — self-written multi-head `GATLayer`.
- Create `baselines/2026-08-15_trackA_gat_edge/code/model.py` — `TrackAGatModel` (LSTM + GAT concat + news + gate + fusion + positivity; graph on/off).
- Create `baselines/2026-08-15_trackA_gat_edge/code/train_resume.py` — checkpointed train loop with resume + graph-snapshot eval.
- Create `baselines/2026-08-15_trackA_gat_edge/code/run_trackA.py` — build basis (reuse combo), train (NODE+GNN readout on one checkpoint), dump predictions + HAR (P0) + metrics.
- Create `baselines/2026-08-15_trackA_gat_edge/code/aggregate.py` — 3-seed DM (GNN vs NODE, GNN vs HAR, NODE vs HAR) (adapt combo aggregate).
- Create tests under `.../test/`: `test_gat.py`, `test_model.py`, `test_resume.py`, `test_run.py` (smoke).
- Results → `results/trackA_gat_seed{seed}_<TS>/`; checkpoints → `models/trackA_gat_seed{seed}_<TS>.pt`.

---

## Task 1: Self-written multi-head GAT layer

**Files:**
- Create: `baselines/2026-08-15_trackA_gat_edge/code/gat.py`
- Test: `baselines/2026-08-15_trackA_gat_edge/test/test_gat.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_gat.py
import sys
from pathlib import Path
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
from gat import GATLayer  # noqa: E402


def test_gat_shapes_and_identity_is_self_only():
    torch.manual_seed(0)
    B, N, din = 2, 4, 8
    layer = GATLayer(din, out_dim=5, heads=3)   # -> out width 15
    h = torch.randn(B, N, din)
    full = torch.ones(B, N, N)                  # fully connected
    out_full = layer(h, full)
    assert out_full.shape == (B, N, 15)
    # identity adjacency => each node attends only to itself => deterministic wrt others:
    eye = torch.eye(N).unsqueeze(0).expand(B, N, N)
    out_self = layer(h, eye)
    # perturbing OTHER nodes must not change a node's self-only output
    h2 = h.clone(); h2[:, 1:] += 3.0
    out_self2 = layer(h2, eye)
    assert torch.allclose(out_self[:, 0], out_self2[:, 0], atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv_gpu_encode/Scripts/python.exe -m pytest baselines/2026-08-15_trackA_gat_edge/test/test_gat.py -q`
Expected: FAIL (ModuleNotFoundError: gat).

- [ ] **Step 3: Write minimal implementation**

```python
# code/gat.py
"""Self-written multi-head Graph Attention layer (Velickovic-style), masked by adjacency."""
from __future__ import annotations
import torch
from torch import nn


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, negative_slope: float = 0.2):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(heads, out_dim))
        self.a_dst = nn.Parameter(torch.zeros(heads, out_dim))
        self.leaky = nn.LeakyReLU(negative_slope)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # h [B,N,in], adjacency [B,N,N] (>0 where edge j->i allowed; diagonal kept by caller)
        b, n, _ = h.shape
        wh = self.W(h).view(b, n, self.heads, self.out_dim)          # [B,N,H,O]
        e_src = (wh * self.a_src).sum(-1)                            # [B,N,H]
        e_dst = (wh * self.a_dst).sum(-1)                            # [B,N,H]
        e = self.leaky(e_dst.unsqueeze(2) + e_src.unsqueeze(1))      # [B,N(dst i),N(src j),H]
        mask = (adjacency > 0).unsqueeze(-1)                         # [B,N,N,1]
        e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=2)                             # over source j
        alpha = torch.nan_to_num(alpha, nan=0.0)                    # isolated node -> all -inf row
        out = torch.einsum("bijh,bjho->biho", alpha, wh)           # [B,N,H,O]
        return torch.nn.functional.elu(out.reshape(b, n, self.heads * self.out_dim))
```

- [ ] **Step 4: Run test to verify it passes**

Run: same command as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add baselines/2026-08-15_trackA_gat_edge/code/gat.py baselines/2026-08-15_trackA_gat_edge/test/test_gat.py
git commit -m "trackA-gat: self-written multi-head GAT layer + test"
```

---

## Task 2: TrackAGatModel (LSTM + GAT concat + news + gate + fusion + positivity; graph on/off)

**Files:**
- Create: `baselines/2026-08-15_trackA_gat_edge/code/model.py`
- Test: `baselines/2026-08-15_trackA_gat_edge/test/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_model.py
import sys
from pathlib import Path
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
from model import TrackAGatModel  # noqa: E402


def _batch(B=2, N=4, seq=22):
    return {
        "price": torch.randn(B, N, seq, 5),
        "news": torch.randn(B, N, seq, 146),
        "news_mask": torch.ones(B, N, seq),
        "ticker_ids": torch.arange(N).unsqueeze(0).expand(B, N),
        "adjacency": torch.eye(N).unsqueeze(0).expand(B, N, N).clone(),
    }


def test_forward_shape_and_graph_changes_output():
    torch.manual_seed(0)
    m = TrackAGatModel(price_dim=5, news_dim=146, num_tickers=4)
    b = _batch()
    pred_off = m(b["price"], b["news"], b["news_mask"], b["ticker_ids"], b["adjacency"], apply_graph=False)
    assert pred_off.shape == (2, 4)
    assert torch.all(pred_off > 0)                       # positivity floor
    vol2pk = torch.ones(2, 4, 4)                          # a non-identity graph
    pred_on = m(b["price"], b["news"], b["news_mask"], b["ticker_ids"], vol2pk, apply_graph=True)
    assert not torch.allclose(pred_off, pred_on)         # graph residual moves predictions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv_gpu_encode/Scripts/python.exe -m pytest baselines/2026-08-15_trackA_gat_edge/test/test_model.py -q`
Expected: FAIL (ModuleNotFoundError: model).

- [ ] **Step 3: Write minimal implementation**

```python
# code/model.py
"""Track-A-style GNN: LSTM temporal + real multi-head GAT (concat branch) + news + per-ticker gate."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gat import GATLayer  # noqa: E402

POSITIVITY_EPSILON = 1e-6


class TrackAGatModel(nn.Module):
    def __init__(self, price_dim: int, news_dim: int, num_tickers: int,
                 hidden: int = 64, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.price_lstm = nn.LSTM(price_dim, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.news_proj = nn.Linear(news_dim, hidden)
        self.news_lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.gate_logits = nn.Parameter(torch.zeros(num_tickers))
        self.gat1 = GATLayer(hidden, hidden, heads)          # 64 -> 256
        self.gat2 = GATLayer(hidden * heads, hidden, heads)  # 256 -> 256
        gnn_dim = hidden * heads
        self.head = nn.Sequential(
            nn.Linear(hidden + gnn_dim + hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))
        # per-ticker target scaler stats set by configure_positivity(store) before eval
        self.register_buffer("scaler_mean", torch.zeros(num_tickers))
        self.register_buffer("scaler_std", torch.ones(num_tickers))

    def configure_positivity(self, mean: torch.Tensor, std: torch.Tensor) -> "TrackAGatModel":
        self.scaler_mean.copy_(mean); self.scaler_std.copy_(std); return self

    def _encode_seq(self, lstm: nn.LSTM, x, proj=None):
        b, n, seq, d = x.shape
        flat = x.reshape(b * n, seq, d)
        if proj is not None:
            flat = proj(flat)
        out, _ = lstm(flat)
        return out[:, -1].reshape(b, n, -1)                 # last hidden [B,N,hidden]

    def forward(self, price, news, news_mask, ticker_ids, adjacency, apply_graph: bool = True):
        h_lstm = self._encode_seq(self.price_lstm, price)                       # [B,N,64]
        news_hidden = self._encode_seq(self.news_lstm, news, proj=self.news_proj)  # [B,N,64]
        gate = torch.sigmoid(self.gate_logits[ticker_ids]).unsqueeze(-1)
        gated_news = gate * news_hidden
        b, n, _ = h_lstm.shape
        adj = adjacency if apply_graph else torch.eye(n, device=h_lstm.device).unsqueeze(0).expand(b, n, n)
        h_gnn = self.gat2(self.gat1(h_lstm, adj), adj)                          # [B,N,256]
        h = torch.cat([h_lstm, h_gnn, gated_news], dim=-1)                     # [B,N,384]
        raw = self.head(h).squeeze(-1)                                         # normalized pred
        mean = self.scaler_mean[ticker_ids]; std = self.scaler_std[ticker_ids]
        denorm = raw * std + mean
        eps = POSITIVITY_EPSILON
        floored = eps * torch.nn.functional.softplus(denorm / eps) + eps
        return (floored - mean) / std                                          # back to normalized, >0
```

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add baselines/2026-08-15_trackA_gat_edge/code/model.py baselines/2026-08-15_trackA_gat_edge/test/test_model.py
git commit -m "trackA-gat: TrackAGatModel (LSTM+GAT concat+news+gate+positivity, graph on/off) + test"
```

---

## Task 3: Checkpointed train loop with RESUME

**Files:**
- Create: `baselines/2026-08-15_trackA_gat_edge/code/train_resume.py`
- Test: `baselines/2026-08-15_trackA_gat_edge/test/test_resume.py`

- [ ] **Step 1: Write the failing test** (train 2 epochs on tiny synthetic snapshots, save, resume 1 → epoch counter = 3, checkpoint reloadable)

```python
# test/test_resume.py
import sys
from pathlib import Path
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
from model import TrackAGatModel          # noqa: E402
from train_resume import train_with_resume, load_checkpoint  # noqa: E402


def _snaps(n_snap=3, N=4, seq=22):
    torch.manual_seed(0)
    snaps = []
    for _ in range(n_snap):
        snaps.append({
            "price": torch.randn(N, seq, 5), "news": torch.randn(N, seq, 146),
            "news_mask": torch.ones(N, seq), "ticker_ids": torch.arange(N),
            "adjacency": torch.eye(N), "target": torch.rand(N) + 0.1,
        })
    return snaps


def test_train_then_resume_advances_epoch(tmp_path):
    m = TrackAGatModel(5, 146, 4)
    ckpt = tmp_path / "m.pt"
    train_with_resume(m, _snaps(), _snaps(), ckpt, epochs=2, device=torch.device("cpu"), seed=0)
    state = load_checkpoint(ckpt)
    assert state["epoch"] == 2
    m2 = TrackAGatModel(5, 146, 4)
    train_with_resume(m2, _snaps(), _snaps(), ckpt, epochs=1, device=torch.device("cpu"),
                      seed=0, resume=True)
    assert load_checkpoint(ckpt)["epoch"] == 3          # resumed, not restarted
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv_gpu_encode/Scripts/python.exe -m pytest baselines/2026-08-15_trackA_gat_edge/test/test_resume.py -q`
Expected: FAIL (ModuleNotFoundError: train_resume).

- [ ] **Step 3: Write minimal implementation**

```python
# code/train_resume.py
"""Checkpointed graph training with resume. Batches whole snapshots (variable N handled per-snap)."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn


def _forward_snap(model, snap, device, apply_graph=True):
    pred = model(snap["price"].unsqueeze(0).to(device), snap["news"].unsqueeze(0).to(device),
                 snap["news_mask"].unsqueeze(0).to(device),
                 snap["ticker_ids"].unsqueeze(0).to(device),
                 snap["adjacency"].unsqueeze(0).to(device), apply_graph=apply_graph)
    return pred.squeeze(0)


def _val_loss(model, snaps, device):
    model.eval(); tot, cnt = 0.0, 0
    with torch.no_grad():
        for s in snaps:
            p = _forward_snap(model, s, device)
            t = s["target"].to(device)
            tot += torch.mean((p - t) ** 2).item() * len(t); cnt += len(t)
    return tot / max(cnt, 1)


def save_checkpoint(path, model, optimizer, epoch, best_val, best_state):
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, "best_val": best_val, "best_state": best_state}, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def train_with_resume(model, train_snaps, val_snaps, ckpt_path: Path, epochs: int,
                      device, seed: int, resume: bool = False):
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    start_epoch, best_val = 0, float("inf")
    best_state = copy.deepcopy(model.state_dict())
    if resume and Path(ckpt_path).exists():
        ck = load_checkpoint(ckpt_path)
        model.load_state_dict(ck["model_state"]); optimizer.load_state_dict(ck["optimizer_state"])
        start_epoch, best_val, best_state = ck["epoch"], ck["best_val"], ck["best_state"]
    rng = np.random.default_rng(seed)
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        for i in rng.permutation(len(train_snaps)):
            s = train_snaps[i]
            optimizer.zero_grad()
            loss = torch.mean((_forward_snap(model, s, device) - s["target"].to(device)) ** 2)
            if not torch.isfinite(loss):
                raise ValueError("non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        vl = _val_loss(model, val_snaps, device)
        if vl < best_val:
            best_val, best_state = vl, copy.deepcopy(model.state_dict())
    save_checkpoint(ckpt_path, model, optimizer, start_epoch + epochs, best_val, best_state)
    return {"epoch": start_epoch + epochs, "best_val": best_val, "best_state": best_state}
```

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add baselines/2026-08-15_trackA_gat_edge/code/train_resume.py baselines/2026-08-15_trackA_gat_edge/test/test_resume.py
git commit -m "trackA-gat: checkpointed train loop with resume + test"
```

---

## Task 4: Basis + run script (reuse combo build_basis; NODE+GNN readout on one checkpoint)

**Files:**
- Create: `baselines/2026-08-15_trackA_gat_edge/code/run_trackA.py`
- Test: `baselines/2026-08-15_trackA_gat_edge/test/test_run.py`

- [ ] **Step 1: Write the failing smoke test** (monkeypatch to a tiny snapshot set; assert NODE/GNN/HAR metrics + checkpoint written)

```python
# test/test_run.py — smoke: build a 2-snapshot fake basis, run 1-epoch NODE+GNN, assert outputs
import sys
from pathlib import Path
import pytest, torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import run_trackA as rt  # noqa: E402


@pytest.mark.smoke
def test_run_seed_smoke(tmp_path, monkeypatch):
    def _fake_basis(*a, **k):
        N, seq = 4, 22
        def snap(split):
            return {"price": torch.randn(N, seq, 5), "news": torch.randn(N, seq, 146),
                    "news_mask": torch.ones(N, seq), "ticker_ids": torch.arange(N),
                    "adjacency": torch.ones(N, N), "target": torch.rand(N) + 0.1,
                    "target_raw": (torch.rand(N) + 0.1).tolist(), "split": split}
        snaps = [snap("train"), snap("train"), snap("val"), snap("test")]
        return {"snaps": snaps, "num_tickers": N, "scaler_mean": torch.zeros(N),
                "scaler_std": torch.ones(N), "har": {"val": {"qlike": 0.5}, "test": {"qlike": 0.5}}}
    monkeypatch.setattr(rt, "build_trackA_basis", _fake_basis)
    out = rt.run_seed(seed=0, epochs=1, ts="T", out_base=tmp_path, device=torch.device("cpu"))
    assert set(out["rungs"]) >= {"HAR", "NODE", "GNN"}
    assert (tmp_path / "ckpt.pt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv_gpu_encode/Scripts/python.exe -m pytest baselines/2026-08-15_trackA_gat_edge/test/test_run.py -q`
Expected: FAIL (ModuleNotFoundError: run_trackA).

- [ ] **Step 3: Write minimal implementation**

`run_trackA.py` responsibilities (write real code, no placeholders):
- `build_trackA_basis(...)`: import combo `build_basis` from
  `baselines/2026-08-14_pooled_news_edanode_gnn/code/combo_ladder.py`; convert its graph snapshots
  (which already carry 5-feature price seq + news + vol→PK adjacency + targets) into the per-snapshot
  dicts the model consumes; extract per-ticker target-scaler mean/std for `configure_positivity`;
  compute HAR (P0) metrics via the same `run_e0`. Return `{"snaps", "num_tickers", "scaler_mean",
  "scaler_std", "har"}`.
- `run_seed(seed, epochs, ts, out_base, device, resume=False)`: build `TrackAGatModel`,
  `configure_positivity`, `train_with_resume` on train/val snaps writing `out_base/ckpt.pt`; load
  `best_state`; evaluate on val+test snaps twice — `apply_graph=False` → **NODE**, `apply_graph=True`
  → **GNN** — into all-6-metrics via the shared `evaluate_records`; dump per-obs predictions for DM;
  write `ladder_metrics.json` with rungs HAR/NODE/GNN. Reuse `_write_graph_predictions`, `_write_json`,
  `evaluate_records` from the pooled pipeline.
- `main(ts, device_name, seeds, epochs, resume)`: loop seeds; results to
  `results/trackA_gat_seed{seed}_{ts}/`; checkpoints to `models/`.

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS (smoke on fake basis).

- [ ] **Step 5: Commit**

```bash
git add baselines/2026-08-15_trackA_gat_edge/code/run_trackA.py baselines/2026-08-15_trackA_gat_edge/test/test_run.py
git commit -m "trackA-gat: basis reuse + run_seed (NODE/GNN/HAR) + smoke"
```

---

## Task 5: DM aggregate (GNN vs NODE, GNN vs HAR, NODE vs HAR)

**Files:**
- Create: `baselines/2026-08-15_trackA_gat_edge/code/aggregate.py` (adapt `combo_aggregate.py`; RUNGS=(HAR,NODE,GNN), DUMP rungs = the three with per-obs dumps; comparisons GNN_vs_NODE, GNN_vs_HAR, NODE_vs_HAR; QLIKE floor 1e-8 identical; per-pair DM guard; median/tie handling as in combo_aggregate).
- Test: reuse the combo `test_combo_aggregate` pattern (synthetic dumps → verdict + degenerate guard).

- [ ] Steps mirror `combo_aggregate.py` (already reviewed): copy structure, rename rungs/comparisons, keep the try/except degenerate-pair guard, `allow_nan=False`, tie label. Test with synthetic dumps. Commit.

---

## Task 6: Run 1 seed × 15 epochs → report → resume → 3 seeds

- [ ] Run `run_trackA.main` with `seeds=[42], epochs=15` on `.venv_gpu_encode` (GPU). Report NODE/GNN/HAR val+test metrics (all 6) + graph effect (GNN−NODE).
- [ ] Present to user; on approval `resume=True, epochs=5` (or 10) from the checkpoint; report again.
- [ ] If configuration is settled, run `seeds=[42,123,2026]` and `aggregate.py` → DM (GNN vs NODE = graph helpful?, GNN/NODE vs HAR). Report honestly.

---

## Quality gates (per CLAUDE.md, every task)
- `.venv_gpu_encode/Scripts/python.exe -m pytest <baseline>/test/ -q` green; ruff clean; diff-cover C0=100% on changed lines (pure model/gat/resume/aggregate covered by unit tests; `run_seed` heavy path covered by the monkeypatched smoke; the 15-epoch GPU run validated by results JSON).
- 3-layer adversarial code review before "done"; data-quality gate N/A (reuses processed data + news panel; no new dataset).
- Push branch after each verified task; objective report tone.

## Self-review notes
- Spec coverage: node features (5) ✓ Task 4 basis; real multi-head GAT ✓ Task 1–2; news+gate ✓ Task 2; vol→PK edge ✓ Task 4 (reused); graph on/off nested ✓ Task 2 (`apply_graph`); resume ✓ Task 3; DM ablation ✓ Task 5–6; architecture file ✓ ARCHITECTURE.md.
- Type consistency: `TrackAGatModel(price_dim, news_dim, num_tickers)`, `GATLayer(in_dim,out_dim,heads)`, `train_with_resume(model, train, val, ckpt, epochs, device, seed, resume)`, `load_checkpoint(path)["epoch"]` used consistently across tasks.
- No placeholders in code steps (Tasks 1–3 full code; Task 4–5 reuse-described with exact source files).
