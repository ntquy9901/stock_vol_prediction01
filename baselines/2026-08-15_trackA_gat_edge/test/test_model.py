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
    # mean=0 keeps the positivity guarantee scaler-independent (floored/std > 0 for any std>0);
    # std shrinks the denormalized scale so it lands near eps instead of deep in softplus's
    # float32 underflow zone (|denorm/eps| > ~87 collapses both branches to the identical floor
    # value regardless of the real GAT-driven difference in raw output -- verified via root-cause
    # inspection: at the default std=1, raw output at this seed is bias-dominated and negative for
    # both apply_graph settings, well past the underflow threshold).
    m.configure_positivity(torch.zeros(4), torch.full((4,), 1e-4))
    b = _batch()
    pred_off = m(b["price"], b["news"], b["news_mask"], b["ticker_ids"], b["adjacency"], apply_graph=False)
    assert pred_off.shape == (2, 4)
    assert torch.all(pred_off > 0)                       # positivity floor
    vol2pk = torch.ones(2, 4, 4)                          # a non-identity graph
    pred_on = m(b["price"], b["news"], b["news_mask"], b["ticker_ids"], vol2pk, apply_graph=True)
    assert not torch.allclose(pred_off, pred_on)         # graph residual moves predictions


def test_ablation_flags_lstm_only_and_no_gate():
    torch.manual_seed(0)
    b = _batch()
    # LSTM-only: use_news=False -> gated_news is a zero vector; still valid positive output
    lstm = TrackAGatModel(5, 146, 4, use_news=False, use_gate=False)
    lstm.configure_positivity(torch.zeros(4), torch.full((4,), 1e-4))
    p = lstm(b["price"], b["news"], b["news_mask"], b["ticker_ids"], b["adjacency"], apply_graph=False)
    assert p.shape == (2, 4) and bool((p > 0).all())
    # +news, no gate: gate fixed at 1.0
    news = TrackAGatModel(5, 146, 4, use_news=True, use_gate=False)
    news.configure_positivity(torch.zeros(4), torch.full((4,), 1e-4))
    p2 = news(b["price"], b["news"], b["news_mask"], b["ticker_ids"], b["adjacency"], apply_graph=False)
    assert p2.shape == (2, 4)
