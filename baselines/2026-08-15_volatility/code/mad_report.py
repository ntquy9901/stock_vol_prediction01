"""FU3a: quantify GAT over-smoothing on VN by measuring per-layer MAD over held-out test snapshots.

Loads a trained FULL 2-hop checkpoint, runs the graph branch on each test snapshot, and reports the
mean MAD (mean average distance = 1 − cosine over present node pairs) of the gat1 output vs the gat2
output. The paper predicts MAD DROPS with depth (over-smoothing); this measures whether that happens
on the VN vol→PK graph (P2 found 2-hop still helps here, so we expect MAD to NOT collapse).

Run: python <.../code/mad_report.py> <TS_with_suffix> <horizon> [seed]
"""
from __future__ import annotations

import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
_ROOT = CODE.resolve().parents[2]
for _p in (CODE, _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code",
           _ROOT / "baselines" / "2026-08-11_eda_gnn_baseline" / "code",
           _ROOT / "baselines" / "2026-08-14_pooled_news_edanode_gnn" / "code", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from mad import mad  # noqa: E402


def layer_mads(model, snaps, device) -> dict[int, float]:
    """Mean MAD (over test snaps) of each GAT layer's node embeddings. Keys 1..gat_layers."""
    model.eval()
    model.to(device)
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    with torch.no_grad():
        for snap in snaps:
            price = snap["price"].unsqueeze(0).to(device)
            adj = snap["adjacency"].unsqueeze(0).to(device)
            presence = snap["presence_mask"].to(device)
            outs = model.gat_layer_outputs(price, adj)             # list of [1, N, hidden*heads]
            for layer_idx, emb in enumerate(outs, start=1):
                value = float(mad(emb.squeeze(0), presence))
                sums[layer_idx] = sums.get(layer_idx, 0.0) + value
                counts[layer_idx] = counts.get(layer_idx, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def main(ts: str, horizon: int, seed: int = 42) -> None:  # pragma: no cover
    import combo_ladder
    from model import VolatilityModel
    from run_volatility import ROOT, build_volatility_basis
    from train_resume import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combo_ladder.HORIZON = horizon
    out_base = ROOT / "results" / f"volatility_ablation_h{horizon}_seed{seed}_{ts}"
    basis = build_volatility_basis(out_base / "_mad_tmp")
    model = VolatilityModel(basis["price_dim"], basis["news_dim"], basis["num_tickers"], gat_layers=2)
    model.configure_positivity(basis["scaler_mean"], basis["scaler_std"])
    ckpt = load_checkpoint(out_base / "FULL.pt")
    model.load_state_dict(ckpt["best_state"])
    test = [s for s in basis["snaps"] if s["split"] == "test"]
    mads = layer_mads(model, test, device)
    print(f"MAD by GAT depth (FULL 2-hop, h{horizon}, seed{seed}, n_test_snaps={len(test)}):")
    for layer, value in sorted(mads.items()):
        print(f"  gat{layer} (={layer}-hop) MAD = {value:.4f}")
    if len(mads) == 2:
        drop = mads[1] - mads[2]
        print(f"  MAD change 1->2 hop: {drop:+.4f} "
              f"({'drops (over-smoothing)' if drop > 0 else 'rises (no collapse)'})")


if __name__ == "__main__":  # pragma: no cover
    _ts = sys.argv[1]
    _h = int(sys.argv[2])
    _seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    main(_ts, _h, _seed)
