"""P2: configurable GAT depth (1-hop vs 2-hop) on VolatilityModel.

Paper finding: one GNN layer (1-hop: node + direct neighbours) is usually enough; stacking to 2/3
hops over-smooths without a stable gain. gat_layers=1 must keep the graph branch dim (head unchanged)
and default (2) must preserve current behavior.
"""
import sys
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from model import VolatilityModel  # noqa: E402


def _inputs(b=1, n=3, seq=4, pf=5, nf=6):
    return dict(
        price=torch.randn(b, n, seq, pf), news=torch.randn(b, n, seq, nf),
        news_mask=torch.ones(b, n, seq), ticker_ids=torch.arange(n).unsqueeze(0).expand(b, n),
        adjacency=torch.eye(n).unsqueeze(0).expand(b, n, n).contiguous())


def test_default_is_two_layers():
    m = VolatilityModel(price_dim=5, news_dim=6, num_tickers=3, hidden=8, heads=2)
    assert m.gat_layers == 2
    assert hasattr(m, "gat2")


def test_one_layer_has_no_gat2_and_forwards():
    m = VolatilityModel(price_dim=5, news_dim=6, num_tickers=3, hidden=8, heads=2, gat_layers=1)
    m.configure_positivity(torch.zeros(3), torch.ones(3))
    assert m.gat_layers == 1
    assert not hasattr(m, "gat2")
    out = m(**_inputs())
    assert out.shape == (1, 3) and torch.isfinite(out).all()


def test_gat_layer_outputs_count_matches_depth():
    inp = _inputs()
    for depth in (1, 2):
        m = VolatilityModel(price_dim=5, news_dim=6, num_tickers=3, hidden=8, heads=2, gat_layers=depth)
        outs = m.gat_layer_outputs(inp["price"], inp["adjacency"])
        assert len(outs) == depth
        assert outs[-1].shape == (1, 3, 8 * 2)     # hidden*heads


def test_gat_layer_outputs_requires_graph():
    m = VolatilityModel(price_dim=5, news_dim=6, num_tickers=3, hidden=8, heads=2, use_graph=False)
    inp = _inputs()
    try:
        m.gat_layer_outputs(inp["price"], inp["adjacency"])
        raise AssertionError("expected ValueError when use_graph=False")
    except ValueError:
        pass
