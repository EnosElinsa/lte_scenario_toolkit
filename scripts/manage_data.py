"""Thin source-tree entry point for LTE scenario data management."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lte_scenario_toolkit.data_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
