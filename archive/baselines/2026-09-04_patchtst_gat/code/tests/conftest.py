"""Force CPU for every test in this baseline (the GPU is busy with an overnight walk-forward chain)
and put the baseline ``code/`` dir on sys.path (the folder name has a dash, so it is not importable
as a package)."""
import os
import sys
from pathlib import Path

# IMPORTANT: on this Windows + torch build, an EMPTY CUDA_VISIBLE_DEVICES ("") does NOT hide the GPU
# (torch.cuda.is_available() stays True); "-1" is required. Must be set before torch initialises CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Multi-threaded MKL/OpenMP einsum crashes ("Windows fatal exception: access violation") on this box;
# pin CPU math to one thread for stable smokes. Set before torch imports the math backends.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))

import torch  # noqa: E402  (after env pins)

torch.set_num_threads(1)

# Env pins above can be read too late by an already-mapped MKL/OpenMP DLL, so ALSO cap the native
# BLAS/MKL/OpenMP thread pools at RUNTIME (this is what actually prevents the einsum access violation
# on this box regardless of import order). Held for the whole session.
try:
    import threadpoolctl
    _TP_LIMIT = threadpoolctl.threadpool_limits(limits=1)   # noqa: F841 (keep alive for session)
except Exception:  # pragma: no cover - threadpoolctl always present in the project venv
    pass
