"""Download competition data via the Kaggle API.

Prerequisites:
  1. pip install kaggle
  2. Set KAGGLE_USERNAME / KAGGLE_KEY in a .env file at the repo root,
     OR drop kaggle.json at %USERPROFILE%\\.kaggle\\kaggle.json (Windows).
  3. Accept the competition rules on the website:
     https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/rules

Usage:
  python scripts/download_data.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

COMPETITION = "nvidia-nemotron-model-reasoning-challenge"
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except OSError as e:
        print(f"[download] Kaggle API init failed: {e}", file=sys.stderr)
        print("  Set KAGGLE_USERNAME/KAGGLE_KEY in .env or place kaggle.json.", file=sys.stderr)
        return 2

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    print(f"[download] Downloading '{COMPETITION}' -> {RAW_DIR}")
    api.competition_download_files(COMPETITION, path=str(RAW_DIR), quiet=False)

    for zp in RAW_DIR.glob("*.zip"):
        print(f"[download] Extracting {zp.name}")
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(RAW_DIR)
        zp.unlink()

    files = sorted(p.name for p in RAW_DIR.iterdir())
    print(f"[download] Files in {RAW_DIR}: {files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
