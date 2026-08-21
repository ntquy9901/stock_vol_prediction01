"""Smoke test: the architecture-figure generator produces non-empty svg/pdf/png."""
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent


def test_generate_arch_writes_all_formats():
    runpy.run_path(str(HERE / "generate_arch.py"), run_name="__main__")
    for ext in ("svg", "pdf", "png"):
        f = HERE / f"soict_harlstmgat.{ext}"
        assert f.exists() and f.stat().st_size > 1000, f"{ext} missing or too small"
