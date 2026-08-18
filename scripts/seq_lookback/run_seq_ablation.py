"""Control run: seq-lookback under the STANDARD 70/15/15 split WITH early stopping.

Counterpart to `run_seq.py` (which uses the 90/10 retrain-on-(train+val) protocol). This one wraps
the delivered `run_ablation` harness verbatim — train on train (70%), early-stop on val (15%,
patience=3, min_epochs=6), test read once (15%); HAR refit on train-only — so it isolates the effect
of the SPLIT/early-stop protocol from the lookback. The ONLY override vs the delivered run is
`combo_ladder.SEQ` (the split ratio is left at the harness default 70/15/15). The basis smoke-assert
is reused from `run_seq` and injected at `run_volatility.build_basis` (the name `build_volatility_basis`
resolves at call time), so it fires without touching the split.

Run (GPU venv):
  python scripts/seq_lookback/run_seq_ablation.py <SEQ> <TS> [device] [seed] [epochs] [horizons...]
Writes results/volatility_ablation_h{h}_seed{seed}_<TS>/ladder_metrics.json (+ per-rung test dumps).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (
    str(_ROOT / "baselines" / "2026-08-15_volatility" / "code"),
    str(_ROOT / "baselines" / "2026-08-14_pooled_news_edanode_gnn" / "code"),
    str(_ROOT / "baselines" / "2026-08-11_eda_gnn_baseline" / "code"),
    str(_ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"),
    str(_ROOT),
    str(Path(__file__).resolve().parent),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import combo_ladder  # noqa: E402
import run_ablation  # noqa: E402
import run_volatility  # noqa: E402
from run_seq import _build_with_smoke  # noqa: E402  (same per-column basis smoke-assert)


def main() -> None:
    seq = int(sys.argv[1])
    ts = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    horizons = tuple(int(x) for x in sys.argv[6:]) if len(sys.argv) > 6 else (1, 5, 10, 22)

    combo_ladder.SEQ = seq
    # inject the smoke wrapper where build_volatility_basis looks it up (run_volatility module global);
    # the split ratio is NOT patched -> the harness default (0.7, 0.15, 0.15) applies.
    run_volatility.build_basis = _build_with_smoke
    print(f"[seq-ablation 70/15/15] SEQ={seq} ts={ts} device={device} seed={seed} "
          f"epochs={epochs} horizons={horizons} (early-stop patience=3 min_epochs=6)", flush=True)
    run_ablation.main(ts, device, seed, epochs, horizons)


if __name__ == "__main__":
    main()
