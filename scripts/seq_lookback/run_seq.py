"""Throwaway experiment runner: retrain-on-(train+val) ablation at a chosen LSTM lookback (SEQ).

Reuses the delivered ``run_retrain_trainval`` harness verbatim (all 6 rungs, 90/10 protocol =
ratios (0.80,0.10,0.10) with train+val merged, fixed epoch budget, no early stop, test read once).
The ONLY overrides vs the delivered run are:
  * ``combo_ladder.SEQ`` = the requested lookback (default harness value is 22),
  * ``combo_ladder.load_and_split_price_data`` forced to the 90/10 ratios (0.80,0.10,0.10),
  * a smoke wrapper around ``combo_ladder.build_basis`` asserting the basis is well-formed
    (price_dim == 5, price features not all-zero for a valid ticker) — per the project rule that
    experiment/wrapper scripts touching train/eval carry a basic invariant smoke-assert before the
    numbers are trusted.

Output dirs mirror the harness: results/volatility_retrain_h{h}_seed{seed}_{TS}/ where the caller is
expected to pass a TS that encodes the seq (e.g. 2026-08-18_1200_seq5) so cross-seq DM can find them.

Run (GPU venv):
  python scripts/seq_lookback/run_seq.py <SEQ> <TS> [device] [seed] [epochs] [horizons...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (
    _ROOT / "baselines" / "2026-08-15_volatility" / "code",
    _ROOT / "baselines" / "2026-08-14_pooled_news_edanode_gnn" / "code",
    _ROOT / "baselines" / "2026-08-11_eda_gnn_baseline" / "code",
    _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code",
    _ROOT,
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import combo_ladder  # noqa: E402
import run_retrain_trainval  # noqa: E402

_ORIG_SPLIT = combo_ladder.load_and_split_price_data
_ORIG_BUILD = combo_ladder.build_basis
_RATIOS_90_10 = (0.80, 0.10, 0.10)


def _split_9010(processed):
    """Force the 90/10 protocol split (train+val merged downstream = 90%, test = 10%)."""
    return _ORIG_SPLIT(processed, ratios=_RATIOS_90_10)


def _build_with_smoke(stamp):
    """Delegate to the real build_basis, then assert the basis is well-formed before trusting it.

    The check inspects ``graph.snapshots[0].x_price`` — the [nodes, seq, feats] tensor the model
    actually consumes (run_retrain_trainval._build_deep_basis reads x_price, not the windowed
    x_price_raw) — and verifies per feature column that it is not globally all-zero. This catches a
    FULLY zeroed feature column; a column zeroed for only a SUBSET of tickers (the historical
    volume_zscore bug) is guarded upstream by features._check_price_coverage in augment_split_frames,
    not here.
    """
    pooled, graph, store, allowed, edge_count = _ORIG_BUILD(stamp)
    snap = np.asarray(graph.snapshots[0].x_price, dtype=np.float64)
    if snap.ndim != 3:
        raise AssertionError(f"[seq-smoke] snapshot x_price must be 3-D [nodes,seq,feats], got {snap.shape}")
    n_nodes, seq_dim, price_dim = snap.shape
    if price_dim != 5:
        raise AssertionError(f"[seq-smoke] expected price_dim=5, got {price_dim}")
    if seq_dim != combo_ladder.SEQ:
        raise AssertionError(f"[seq-smoke] lookback dim {seq_dim} != SEQ {combo_ladder.SEQ}")
    if not np.isfinite(snap).all():
        raise AssertionError("[seq-smoke] non-finite values in snapshot x_price")
    # per-column guard: a silently-zeroed feature column must fail (not just an all-zero whole block).
    for j in range(price_dim):
        if np.allclose(snap[:, :, j], 0.0):
            raise AssertionError(f"[seq-smoke] snapshot x_price feature column {j} is all-zero (feature build broke)")
    print(f"[seq-smoke] OK: SEQ={combo_ladder.SEQ} price_dim={price_dim} "
          f"seq_dim={seq_dim} nodes0={n_nodes} train_windows={len(allowed)} edges={edge_count}", flush=True)
    return pooled, graph, store, allowed, edge_count


def main() -> None:
    seq = int(sys.argv[1])
    ts = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 15
    horizons = tuple(int(x) for x in sys.argv[6:]) if len(sys.argv) > 6 else (1, 5, 10, 22)

    combo_ladder.SEQ = seq
    combo_ladder.load_and_split_price_data = _split_9010
    combo_ladder.build_basis = _build_with_smoke
    print(f"[seq] SEQ={seq} ts={ts} device={device} seed={seed} epochs={epochs} horizons={horizons}",
          flush=True)
    run_retrain_trainval.main(ts, device, seed, epochs, horizons)


if __name__ == "__main__":
    main()
