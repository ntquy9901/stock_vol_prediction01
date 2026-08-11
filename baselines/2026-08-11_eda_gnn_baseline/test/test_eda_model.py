"""Shape, nesting, positivity, and masking tests for PriceGraphModel."""

import numpy as np
import torch

from eda_model import PriceGraphModel


def _inputs(batch=2, nodes=4, seq=6, feat=5):
    torch.manual_seed(0)
    x_price = torch.randn(batch, nodes, seq, feat)
    adjacency = torch.eye(nodes).unsqueeze(0).expand(batch, nodes, nodes).clone()
    adjacency[:, 0, 1] = 0.5  # node 0 attends to node 1
    ticker_ids = torch.arange(nodes).unsqueeze(0).expand(batch, nodes).clone()
    presence = torch.ones(batch, nodes, dtype=torch.bool)
    return x_price, adjacency, ticker_ids, presence


def test_forward_shape():
    model = PriceGraphModel(price_dim=5, num_tickers=4, use_gnn=True).eval()
    x_price, adjacency, ticker_ids, presence = _inputs()
    out = model(x_price, adjacency, ticker_ids, presence)
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()


def test_message_passing_off_matches_no_gnn():
    """apply_message_passing=False must equal the pure backbone+head (nested E3off control)."""

    x_price, adjacency, ticker_ids, presence = _inputs()
    gnn = PriceGraphModel(price_dim=5, num_tickers=4, use_gnn=True).eval()
    with torch.no_grad():
        off = gnn(x_price, adjacency, ticker_ids, presence, apply_message_passing=False)
        # Rebuild a use_gnn=False twin from the same encoder+head weights.
        plain = PriceGraphModel(price_dim=5, num_tickers=4, use_gnn=False).eval()
        plain.price_lstm.load_state_dict(gnn.price_lstm.state_dict())
        plain.head.load_state_dict(gnn.head.state_dict())
        base = plain(x_price, adjacency, ticker_ids, presence)
    assert torch.allclose(off, base, atol=1e-6)
    # And message passing ON must actually move node 0's prediction (guards a no-op edge).
    with torch.no_grad():
        on = gnn(x_price, adjacency, ticker_ids, presence, apply_message_passing=True)
    assert not torch.allclose(on[:, 0], off[:, 0], atol=1e-6)


def test_positivity_floor_makes_predictions_positive():
    class _Scaler:
        def __init__(self, mean, std):
            self.mean = np.array([mean])
            self.std = np.array([std])

    class _Prep:
        def __init__(self, mean, std):
            self.target_scaler = _Scaler(mean, std)

    class _Store:
        preprocessors = {i: _Prep(1e-3, 5e-4) for i in range(4)}

    model = PriceGraphModel(price_dim=5, num_tickers=4, use_gnn=True).eval()
    model.configure_positivity(_Store())
    x_price, adjacency, ticker_ids, presence = _inputs()
    with torch.no_grad():
        out = model(x_price, adjacency, ticker_ids, presence)
    raw = out * model.target_std[ticker_ids] + model.target_mean[ticker_ids]
    assert (raw > 0).all()


def test_absent_node_does_not_change_present_outputs():
    """Zeroing an absent node's features must not change a present node's prediction."""

    model = PriceGraphModel(price_dim=5, num_tickers=4, use_gnn=True).eval()
    x_price, adjacency, ticker_ids, presence = _inputs()
    presence[:, 3] = False  # node 3 absent
    # node 0 attends only to node 1 (present), so mutating absent node 3 must not matter
    with torch.no_grad():
        base = model(x_price, adjacency, ticker_ids, presence)
        mutated = x_price.clone()
        mutated[:, 3] = 12.3
        after = model(mutated, adjacency, ticker_ids, presence)
    assert torch.allclose(base[:, 0], after[:, 0], atol=1e-6)
