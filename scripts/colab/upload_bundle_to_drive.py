"""Upload the Colab data bundle to Google Drive via rclone (reusable when the data is updated).

Prereqs (already set up on this machine): rclone installed + an authenticated remote named ``gdrive``
(``rclone listremotes`` shows ``gdrive:``). The bundle is DATA-ONLY (see make_colab_bundle.py); the Colab
notebook reads it from ``MyDrive/luanvan_data/`` and mounts the SAME Google account this remote points to.

Typical use after refreshing the enriched data:
    # rebuild the bundle from current data, then upload it to Drive
    .venv_gpu_encode/Scripts/python.exe scripts/colab/upload_bundle_to_drive.py --market sp500_clean --rebuild

    # or just re-upload the existing zip
    .venv_gpu_encode/Scripts/python.exe scripts/colab/upload_bundle_to_drive.py --market sp500_clean
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

DEFAULT_REMOTE = "gdrive:luanvan_data/"


def rclone_copy_cmd(bundle: Path, remote: str) -> list[str]:
    """Build the rclone command that uploads ``bundle`` into the ``remote`` folder (verifies on transfer)."""
    return ["rclone", "copy", str(bundle), remote, "--progress", "--stats=10s"]


def upload(bundle: Path, remote: str = DEFAULT_REMOTE, run=subprocess.run) -> int:
    """Upload ``bundle`` to ``remote`` with rclone; returns the rclone exit code. Raises if the file is
    missing (nothing to upload)."""
    if not bundle.exists():
        raise SystemExit(f"bundle not found: {bundle} (build it first with make_colab_bundle.py)")
    return run(rclone_copy_cmd(bundle, remote)).returncode


def main():  # pragma: no cover - entry driver; rclone_copy_cmd/upload are unit-tested
    ap = argparse.ArgumentParser(description="Upload the Colab data bundle to Google Drive via rclone.")
    ap.add_argument("--market", default="sp500_clean")
    ap.add_argument("--remote", default=DEFAULT_REMOTE)
    ap.add_argument("--rebuild", action="store_true", help="rebuild the bundle from current data before uploading")
    a = ap.parse_args()
    bundle = REPO / f"colab_bundle_{a.market}.zip"
    if a.rebuild:
        import make_colab_bundle as mb
        n = mb.build_bundle(REPO, a.market, bundle)
        print(f"[upload] rebuilt {bundle.name} ({n} data files)")
    print(f"[upload] rclone copy {bundle.name} -> {a.remote}")
    rc = upload(bundle, a.remote)
    print("[upload] OK" if rc == 0 else f"[upload] FAILED (rclone exit {rc})")
    sys.exit(rc)


if __name__ == "__main__":  # pragma: no cover
    main()
