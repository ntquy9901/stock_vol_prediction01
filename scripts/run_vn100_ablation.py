"""Quick VN100 ablation wrapper: point the Track-A leave-one-out pipeline at VN100 processed data.

The Track-A ablation runner (baselines/2026-08-15_trackA_gat_edge/code/run_ablation.py) defaults to
the VN30 processed dir via combo_ladder._PROCESSED. This wrapper overrides that module global to the
VN100 dir (data/processed/vn100), leaving SEQ=22 and the DEFAULT 70/15/15 split ratios untouched,
then calls run_ablation.main. Exploratory single-seed run only; does not modify data.

Run (GPU venv, from repo root):
  PYTHONIOENCODING=utf-8 .venv_gpu_encode/Scripts/python.exe scripts/run_vn100_ablation.py <TS> \
      [device] [seed] [epochs] [horizons...]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKA_CODE = ROOT / "baselines" / "2026-08-15_trackA_gat_edge" / "code"
for _p in (str(TRACKA_CODE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_ablation  # noqa: E402  (imported first: sets up sys.path + imports combo_ladder)
import combo_ladder  # noqa: E402

# Override the processed-data dir BEFORE any basis is built. SEQ stays 22; ratios stay default.
combo_ladder._PROCESSED = ROOT / "data" / "processed" / "vn100"
print(f"[vn100] _PROCESSED = {combo_ladder._PROCESSED}", flush=True)
print(f"[vn100] SEQ = {combo_ladder.SEQ} (unchanged)", flush=True)


def main() -> None:
    ts = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    horizons = tuple(int(x) for x in sys.argv[5:]) if len(sys.argv) > 5 else (1, 5)
    run_ablation.main(ts, device, seed, epochs, horizons)


if __name__ == "__main__":
    main()
