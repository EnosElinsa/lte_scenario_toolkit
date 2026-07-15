"""Thin command-line wrapper for the Earth Engine DEM export workflow."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from lte_scenario_toolkit.newyork_dem import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
