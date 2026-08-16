"""P1: the ablation runner must thread a `loss` selector into training so a QLIKE-trained ladder
can be produced alongside the MSE one (paper: training-loss choice is as important as architecture).
"""
import sys
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import run_ablation  # noqa: E402
import train_resume  # noqa: E402


def test_train_threads_loss_to_train_with_resume(tmp_path, monkeypatch):
    captured = {}

    def fake_twr(model, train_s, val_s, ckpt, epochs, device, seed, **kw):
        captured["loss"] = kw.get("loss")
        opt = torch.optim.Adam(model.parameters())
        train_resume.save_checkpoint(ckpt, model, opt, 1, 0.0, model.state_dict())

    monkeypatch.setattr(run_ablation, "train_with_resume", fake_twr)
    basis = {"price_dim": 5, "news_dim": 6, "num_tickers": 3,
             "scaler_mean": torch.zeros(3), "scaler_std": torch.ones(3)}
    run_ablation._train(basis, [], [], tmp_path / "x.pt", epochs=1, seed=0,
                        device=torch.device("cpu"), use_news=True, use_gate=True,
                        use_graph=True, loss="qlike")
    assert captured["loss"] == "qlike"


def test_train_defaults_to_mse(tmp_path, monkeypatch):
    captured = {}

    def fake_twr(model, train_s, val_s, ckpt, epochs, device, seed, **kw):
        captured["loss"] = kw.get("loss", "mse")
        opt = torch.optim.Adam(model.parameters())
        train_resume.save_checkpoint(ckpt, model, opt, 1, 0.0, model.state_dict())

    monkeypatch.setattr(run_ablation, "train_with_resume", fake_twr)
    basis = {"price_dim": 5, "news_dim": 6, "num_tickers": 3,
             "scaler_mean": torch.zeros(3), "scaler_std": torch.ones(3)}
    run_ablation._train(basis, [], [], tmp_path / "y.pt", epochs=1, seed=0,
                        device=torch.device("cpu"), use_news=True, use_gate=True, use_graph=True)
    assert captured["loss"] == "mse"
