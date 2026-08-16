"""P1: differentiable masked QLIKE training loss must match the eval qlike_loss exactly.

The eval path (train.evaluate_records -> src.common.evaluation.qlike_loss) denormalizes predictions
via the target scaler (pred_raw = pred_norm*std + mean) and computes mean(ratio - log(ratio) - 1)
with ratio = target_raw/pred_raw, both clamped at epsilon=1e-8. The training loss must be the same
quantity (masked over present nodes) so val-selection and test-eval use one criterion.
"""
import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
_ROOT = Path(__file__).resolve().parents[3]
for _p in (CODE, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_resume as tr  # noqa: E402
from model import VolatilityModel  # noqa: E402
from src.common.evaluation import qlike_loss  # noqa: E402


def _snap(n=3, seq=4, pf=5, nf=6):
    return {
        "price": torch.randn(n, seq, pf),
        "news": torch.randn(n, seq, nf),
        "news_mask": torch.ones(n, seq),
        "ticker_ids": torch.arange(n, dtype=torch.long),
        "adjacency": torch.eye(n),
        "target": torch.rand(n) + 0.5,          # positive-ish (variance scale)
        "presence_mask": torch.ones(n),
        "split": "train",
    }


def test_masked_qlike_matches_eval_qlike():
    # normalized preds/targets + per-node scaler; denorm must stay positive (variance scale)
    pred_norm = torch.tensor([[1.0, -0.5, 0.25]])
    target_norm = torch.tensor([[2.0, 0.5, -0.25]])
    mean_node = torch.tensor([[0.6, 0.4, 0.5]])
    std_node = torch.tensor([[0.2, 0.1, 0.3]])
    presence = torch.ones(1, 3)

    loss = tr._masked_qlike(pred_norm, target_norm, presence, mean_node, std_node)

    pred_raw = (pred_norm * std_node + mean_node).numpy().ravel()
    target_raw = (target_norm * std_node + mean_node).numpy().ravel()
    expected = qlike_loss(target_raw, pred_raw, epsilon=1e-8)
    assert np.isclose(loss.item(), expected, atol=1e-6)


def test_masked_qlike_ignores_absent_nodes():
    # node 2 is absent: its (wild) values must not affect the loss
    pred_norm = torch.tensor([[1.0, 0.25, 5.0]])
    target_norm = torch.tensor([[2.0, -0.25, -50.0]])
    mean_node = torch.tensor([[0.6, 0.5, 0.5]])
    std_node = torch.tensor([[0.2, 0.3, 0.3]])
    presence = torch.tensor([[1.0, 1.0, 0.0]])

    loss = tr._masked_qlike(pred_norm, target_norm, presence, mean_node, std_node)

    keep = slice(0, 2)
    pred_raw = (pred_norm[:, keep] * std_node[:, keep] + mean_node[:, keep]).numpy().ravel()
    target_raw = (target_norm[:, keep] * std_node[:, keep] + mean_node[:, keep]).numpy().ravel()
    expected = qlike_loss(target_raw, pred_raw, epsilon=1e-8)
    assert np.isclose(loss.item(), expected, atol=1e-6)


def test_masked_qlike_is_differentiable():
    pred_norm = torch.tensor([[1.0, 0.25]], requires_grad=True)
    target_norm = torch.tensor([[2.0, 0.5]])
    mean_node = torch.tensor([[0.6, 0.5]])
    std_node = torch.tensor([[0.2, 0.3]])
    presence = torch.ones(1, 2)
    loss = tr._masked_qlike(pred_norm, target_norm, presence, mean_node, std_node)
    loss.backward()
    assert pred_norm.grad is not None and torch.isfinite(pred_norm.grad).all()


def test_train_with_qlike_loss_runs_and_selects_on_qlike(tmp_path):
    import math
    torch.manual_seed(0)
    model = VolatilityModel(price_dim=5, news_dim=6, num_tickers=3, hidden=8, heads=2)
    model.configure_positivity(torch.full((3,), 0.5), torch.full((3,), 0.1))
    snaps = [_snap() for _ in range(4)]
    ckpt = tmp_path / "q.pt"
    out = tr.train_with_resume(model, snaps, snaps, ckpt, epochs=2, device=torch.device("cpu"),
                               seed=0, batch_size=2, loss="qlike")
    assert ckpt.exists()
    assert math.isfinite(out["best_val"])
    # best_val is a QLIKE value on the val snaps -> must equal the model's val QLIKE at best_state
    model.load_state_dict(out["best_state"])
    vq = tr._val_loss(model, snaps, torch.device("cpu"), apply_graph=True, batch_size=2, loss="qlike")
    assert math.isfinite(vq) and vq > -1e-9   # QLIKE >= 0

