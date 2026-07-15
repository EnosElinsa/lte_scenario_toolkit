"""Thin command-line wrapper for the scenario selection workflow."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from select_sites import main  # noqa: E402

if __name__ == "__main__":
    main()
