"""Generate publication terrain figures from an existing scenario CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio

from .config import load_experiment_config
from .io import build_dataset_record, write_run_record
from .spatial import resolve_io_paths
from .terrain import validate_dem_path
from .visualization import render_3d_terrain

REQUIRED_COLUMNS = {
    "rect_id",
    "pt_count",
    "left_x",
    "bottom_y",
    "center_x",
    "center_y",
    "X",
    "Y",
}


def load_scenario_csv(path: str | Path):
    """Read a scenario CSV and reconstruct its rectangle and point geometry."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Scenario CSV does not exist: {csv_path}")
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"Scenario CSV is empty: {csv_path}")
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Scenario CSV missing required columns: {', '.join(missing)}")

    first = frame.iloc[0]
    rectangle = {
        name: first[name]
        for name in (
            "rect_id",
            "pt_count",
            "left_x",
            "bottom_y",
            "center_x",
            "center_y",
        )
    }
    geometry = gpd.points_from_xy(frame["X"], frame["Y"])
    points = gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:3857")
    return frame, rectangle, points


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate 3D figures from an existing CSV")
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML configuration")
    parser.add_argument("--city", help="boundary directory or layer name")
    parser.add_argument("--output-dir", type=Path, help="directory containing CSV and figures")
    parser.add_argument("--size", type=int, help="rectangle size in metres")
    parser.add_argument("--target", type=int, help="target base-station count")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Run the configured existing-CSV figure workflow."""

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    args = _parse_args(argv)
    config = load_experiment_config(
        args.config,
        city=args.city,
        output_dir=args.output_dir,
    )
    if args.size is not None:
        config["rect_size"] = args.size
    if args.target is not None:
        config["target_count"] = args.target
    config.update(resolve_io_paths(config))

    try:
        frame, rectangle, selected_points = load_scenario_csv(config["output_csv"])
        dem_path = validate_dem_path(config["dem_path"])
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Scenario CSV: {config['output_csv']}")
    print(
        f"Rectangle center: ({rectangle['center_x']:.1f}, {rectangle['center_y']:.1f}); "
        f"stations: {rectangle['pt_count']}"
    )
    if "elevation" in frame.columns:
        print(f"Mean elevation: {frame['elevation'].mean():.1f} m")

    with rasterio.open(dem_path) as dem:
        dem_crs = str(dem.crs)
        dem_resolution_m = float(abs(dem.res[0]))
        try:
            render_3d_terrain(
                rectangle,
                selected_points,
                dem,
                config,
                publication_style=True,
            )
        except Exception as exc:
            print(f"WARNING: terrain rendering failed: {exc}", file=sys.stderr)

    try:
        inputs = [
            build_dataset_record(
                config["output_csv"],
                name="scenario_csv",
                source_url="local experiment output",
                license_name="project output",
                crs="EPSG:3857",
            ),
            build_dataset_record(
                dem_path,
                name="dem",
                source_url=(
                    "https://developers.google.com/earth-engine/datasets/catalog/USGS_3DEP_1m"
                ),
                license_name="USGS public-domain data; retain source attribution",
                crs=dem_crs,
                resolution_m=dem_resolution_m,
            ),
        ]
        outputs = [
            path
            for path in (
                config["output_3d_png"],
                config["output_3d_png"].with_suffix(".eps"),
                config["output_3d_html"],
            )
            if path.exists()
        ]
        run_record = write_run_record(
            config["output_dir"],
            config=config,
            inputs=inputs,
            outputs=outputs,
            command=sys.argv if argv is None else ["lte-generate-figures", *argv],
            filename="run-generate-figures.json",
        )
        print(f"Run record: {run_record}")
    except Exception as exc:
        print(f"WARNING: run record generation failed: {exc}", file=sys.stderr)
    return 0
