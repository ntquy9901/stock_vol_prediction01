"""P2: the ablation runner must build the model at the requested GAT depth (1-hop vs 2-hop)."""
import sys
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import run_ablation  # noqa: E402
import train_resume  # noqa: E402


def _capture_gat_layers(tmp_path, monkeypatch, **train_kwargs):
    captured = {}

    def fake_twr(model, train_s, val_s, ckpt, epochs, device, seed, **kw):
        captured["gat_layers"] = model.gat_layers
        opt = torch.optim.Adam(model.parameters())
        train_resume.save_checkpoint(ckpt, model, opt, 1, 0.0, model.state_dict())

    monkeypatch.setattr(run_ablation, "train_with_resume", fake_twr)
    basis = {"price_dim": 5, "news_dim": 6, "num_tickers": 3,
             "scaler_mean": torch.zeros(3), "scaler_std": torch.ones(3)}
    run_ablation._train(basis, [], [], tmp_path / "m.pt", epochs=1, seed=0,
                        device=torch.device("cpu"), use_news=True, use_gate=True,
                        use_graph=True, **train_kwargs)
    return captured["gat_layers"]


def test_train_builds_one_hop_when_requested(tmp_path, monkeypatch):
    assert _capture_gat_layers(tmp_path, monkeypatch, gat_layers=1) == 1


def test_train_defaults_to_two_hops(tmp_path, monkeypatch):
    assert _capture_gat_layers(tmp_path, monkeypatch) == 2
