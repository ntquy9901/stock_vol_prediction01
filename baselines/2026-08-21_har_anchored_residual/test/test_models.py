"""TDD: residual expert net + HAR-anchored reconstruction (plan sections 10, 11).

Invariants: zero-init residual head -> net outputs 0 at init -> hybrid == HAR (additive) or HAR+eps
(multiplicative). Multiplicative reconstruction is always positive. Branch toggles change capacity.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import models as md  # noqa: E402


def test_residual_head_zero_init_outputs_zero():
    for use_lstm, use_graph in [(True, False), (False, True), (True, True)]:
        net = md.ResidualNet(use_lstm=use_lstm, use_graph=use_graph).eval()
        x = torch.randn(3, 5, 10, 3)
        adj = torch.eye(5)
        with torch.no_grad():
            out = net(x, adj)
        assert out.shape == (3, 5)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)   # HAR fallback at init


def test_additive_fallback_equals_har_at_init():
    har = np.array([0.02, 0.03, 0.015])
    c = np.zeros(3)
    pred = md.additive_pred(har, c, res_scale=0.01, lam=1.0)
    assert np.allclose(pred, har)


def test_multiplicative_fallback_and_positivity():
    har = np.array([0.02, 0.0, 0.015])
    c0 = np.zeros(3)
    pred0 = md.multiplicative_pred(har, c0, eps=1e-8, lam=1.0)
    assert np.allclose(pred0, har + 1e-8)                 # exact HAR+eps fallback
    c = np.array([-50.0, 3.0, -1.0])                       # extreme corrections
    pred = md.multiplicative_pred(har, c, eps=1e-8, lam=1.0)
    assert np.all(pred > 0)                                # positive by construction


def test_branch_toggles_change_param_count():
    n_lstm = sum(p.numel() for p in md.ResidualNet(use_lstm=True, use_graph=False).parameters())
    n_gat = sum(p.numel() for p in md.ResidualNet(use_lstm=False, use_graph=True).parameters())
    n_both = sum(p.numel() for p in md.ResidualNet(use_lstm=True, use_graph=True).parameters())
    assert n_both > n_lstm and n_both > n_gat


def test_gat_only_uses_adjacency():
    net = md.ResidualNet(use_lstm=False, use_graph=True).eval()
    # break zero-init so the graph branch actually influences the output
    torch.nn.init.normal_(net.head[-1].weight, std=0.1)
    x = torch.randn(1, 4, 10, 3)
    eye = torch.eye(4)
    dense = torch.ones(4, 4)
    with torch.no_grad():
        a = net(x, eye)
        b = net(x, dense)
    assert not torch.allclose(a, b)
