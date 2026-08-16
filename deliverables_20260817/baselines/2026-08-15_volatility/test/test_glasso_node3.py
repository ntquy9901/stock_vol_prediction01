"""3-feature-node + glasso-edge variant: nodes use ONLY the 3 HAR features (no market_pk/volume_z),
edge = graphical-LASSO. Isolates the graph+nonlinearity from the extra node features vs HAR(3)."""
import sys
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import run_glasso_node3 as g3  # noqa: E402


def test_slice_price_keeps_first_k_features():
    snaps = [{"price": torch.randn(4, 22, 5), "split": "train"},
             {"price": torch.randn(4, 22, 5), "split": "test"}]
    out = g3.slice_price(snaps, 3)
    assert out[0]["price"].shape == (4, 22, 3)
    assert out[1]["split"] == "test"                      # other keys preserved
    assert torch.equal(out[0]["price"], snaps[0]["price"][:, :, :3])


def test_slice_price_width_ge_dim_is_noop():
    snaps = [{"price": torch.randn(2, 4, 5)}]
    assert g3.slice_price(snaps, 5)[0]["price"].shape == (2, 4, 5)
