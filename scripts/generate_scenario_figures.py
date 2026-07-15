"""Thin command-line wrapper for publication terrain figures."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_scenario_figures import main  # noqa: E402

if __name__ == "__main__":
    main()
