"""Generate SHA256 and size metadata for all declared research inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lte_scenario_toolkit.io import create_data_manifest  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "data" / "datasets.yaml",
        help="dataset provenance YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "manifest.json",
        help="generated JSON manifest",
    )
    args = parser.parse_args(argv)
    output = create_data_manifest(args.metadata, args.output, repo_root=ROOT)
    print(f"Data manifest written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
