"""Unit + property tests for LatentNoiseBaseline.

Covers (rule §3.F.5): shape correctness + determinism property (noise OFF in eval,
noise ON in train). Runs on dummy tensors (no real data/cache needed).

Run:  pytest baselines/2026-07-11_latent_noise/test/ -v
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = _ROOT / "baselines" / "2026-07-11_latent_noise" / "code"
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from src.lstm_gat_hybrid.config import LSTMGATConfig
from model_latent_noise import LatentNoiseBaseline


def _make_model(noise_std=0.1, num_features=3):
    cfg = LSTMGATConfig()
    cfg.num_features_per_stock = num_features
    return LatentNoiseBaseline(cfg, emb_dim=64, d_news=64, dropout=0.0, noise_std=noise_std)


def _dummy_batch(B=2, seq=5, S=4, F=3, A=3, emb_dim=64):
    torch.manual_seed(0)
    x_har = torch.randn(B, seq, S, F)
    adj = torch.ones(B, S, S)  # dense graph (all connected) for a small smoke
    x_emb = torch.randn(B, seq, S, A, emb_dim)
    mask = torch.zeros(B, seq, S, A)
    mask[..., 0] = 1.0  # 1 real article per (stock, day) so pooling is well-defined
    return x_har, adj, x_emb, mask


def test_output_shape():
    """forward returns [B, num_stocks]."""
    model = _make_model()
    model.eval()
    B, S = 2, 4
    x_har, adj, x_emb, mask = _dummy_batch(B=B, S=S)
    with torch.no_grad():
        out = model(x_har, adj, x_emb, mask)
    assert out.shape == (B, S), f"expected ({B},{S}), got {tuple(out.shape)}"


def test_noise_off_in_eval_is_deterministic():
    """Property: eval mode -> noise OFF -> two forwards identical."""
    model = _make_model(noise_std=0.5)  # large noise to make any leak obvious
    model.eval()
    x_har, adj, x_emb, mask = _dummy_batch()
    with torch.no_grad():
        out1 = model(x_har, adj, x_emb, mask)
        out2 = model(x_har, adj, x_emb, mask)
    assert torch.allclose(out1, out2, atol=1e-6), "eval forward must be deterministic (noise OFF)"


def test_noise_on_in_train_is_stochastic():
    """Property: train mode + noise_std>0 -> two forwards differ (noise active)."""
    model = _make_model(noise_std=0.5)
    model.train()
    x_har, adj, x_emb, mask = _dummy_batch()
    with torch.no_grad():
        out1 = model(x_har, adj, x_emb, mask)
        out2 = model(x_har, adj, x_emb, mask)
    assert not torch.allclose(out1, out2, atol=1e-4), "train forward must differ when noise_std>0"


def _count_randn(fn, *args, **kwargs):
    """Run fn and count how many times torch.randn_like was called (noise indicator)."""
    calls = {"n": 0}
    orig = torch.randn_like

    def _counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    torch.randn_like = _counting
    try:
        fn(*args, **kwargs)
    finally:
        torch.randn_like = orig
    return calls["n"]


def test_noise_std_zero_calls_no_randn():
    """Property: noise_std=0 -> forward never calls randn_like (no noise injected).

    Counts randn_like directly so the check is not confounded by Dropout (which uses
    bernoulli, not randn) differing between train/eval.
    """
    model = _make_model(noise_std=0.0)
    x_har, adj, x_emb, mask = _dummy_batch()
    model.train()

    def _fwd():
        with torch.no_grad():
            model(x_har, adj, x_emb, mask)

    n = _count_randn(_fwd)
    assert n == 0, f"noise_std=0 must not call randn_like (got {n})"


def test_noise_std_positive_calls_randn_in_train():
    """Property: noise_std>0 + train -> forward calls randn_like (noise active)."""
    model = _make_model(noise_std=0.1)
    x_har, adj, x_emb, mask = _dummy_batch()
    model.train()

    def _fwd():
        with torch.no_grad():
            model(x_har, adj, x_emb, mask)

    assert _count_randn(_fwd) > 0, "noise_std>0 in train must call randn_like"


def test_noise_off_in_eval_calls_no_randn():
    """Property: eval mode -> forward never calls randn_like (noise OFF) even if noise_std>0."""
    model = _make_model(noise_std=0.5)
    x_har, adj, x_emb, mask = _dummy_batch()
    model.eval()

    def _fwd():
        with torch.no_grad():
            model(x_har, adj, x_emb, mask)

    assert _count_randn(_fwd) == 0, "eval mode must not call randn_like (noise OFF)"


def test_backward_runs_with_noise():
    """Smoke: loss backprops through the noisy news_rep without error."""
    model = _make_model(noise_std=0.1)
    model.train()
    x_har, adj, x_emb, mask = _dummy_batch()
    out = model(x_har, adj, x_emb, mask)
    loss = out.pow(2).mean()
    loss.backward()  # should not raise
    # at least the news-temporal params got a grad
    nt_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                  for p in model.news_temporal.parameters())
    assert nt_grad, "news_temporal should receive gradients"


if __name__ == "__main__":
    # plain runner for quick local check (pytest is the primary, per §3.F.5)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
