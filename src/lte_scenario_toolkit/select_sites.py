"""Select an LTE scenario and enrich its stations with DEM elevations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box

from . import io, scenario, spatial, terrain, visualization
from .config import load_experiment_config


def process_selected_rectangles(chosen, points_gdf, dem, config):
    """Extract stations and elevations for the selected rectangles."""

    if not chosen:
        return None, None

    rectangle_size = config["rect_size"]
    point_crs = points_gdf.crs
    frames = []
    selected_groups = []

    for index, result in enumerate(chosen, start=1):
        rectangle = result["geometry"]
        minimum_x, minimum_y, maximum_x, maximum_y = rectangle.bounds
        rough_mask = (
            (points_gdf.geometry.x >= minimum_x)
            & (points_gdf.geometry.x <= maximum_x)
            & (points_gdf.geometry.y >= minimum_y)
            & (points_gdf.geometry.y <= maximum_y)
        )
        candidates = points_gdf[rough_mask]
        selected = candidates[candidates.geometry.within(rectangle)].copy().reset_index(drop=True)
        if selected.empty:
            continue

        elevations = terrain.extract_elevation(selected, dem)
        terrain.require_valid_elevations(elevations)
        selected["elevation"] = elevations
        selected_groups.append(selected)
        frames.append(
            io.build_output_dataframe(
                selected,
                point_crs,
                rect_id=index,
                pt_count=result["pt_count"],
                left_x=result["left_x"],
                bottom_y=result["bottom_y"],
                center_x=result["center_x"],
                center_y=result["center_y"],
                rect_size=rectangle_size,
            )
        )

    if not frames:
        return None, None
    frame = pd.concat(frames, ignore_index=True)
    selected_points = gpd.GeoDataFrame(pd.concat(selected_groups, ignore_index=True), crs=point_crs)
    return frame, selected_points


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Select an LTE scenario using a reproducible YAML configuration"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="experiment YAML; paths are repository-relative",
    )
    parser.add_argument("--city", help="boundary directory or layer name")
    parser.add_argument("--output-dir", type=Path, help="write outputs directly to this directory")
    parser.add_argument("--size", type=int, help="override rectangle size in metres")
    parser.add_argument("--target", type=int, help="override target base-station count")
    parser.add_argument(
        "--select-index",
        type=int,
        help="choose a one-based candidate index without opening the interactive selector",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Run the configured scenario-selection workflow."""

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
    config.update(spatial.resolve_io_paths(config))

    print(f"City: {config['boundary_folder']} ({config['boundary_layer']})")
    print(f"Points: {config['points_shp']}")
    print(f"Boundary: {config['boundary_shp']}")
    print(f"DEM: {config['dem_path']}")
    print(f"Output: {config['output_dir']}")
    points_gdf, boundary, coordinates = spatial.load_and_prepare(config)

    cache_path = config["cache_json"]
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        results = []
        for result in cached:
            result["geometry"] = box(
                result["left_x"],
                result["bottom_y"],
                result["left_x"] + config["rect_size"],
                result["bottom_y"] + config["rect_size"],
            )
            results.append(result)
        print(f"Loaded {len(results)} cached candidates: {cache_path.name}")
    else:
        positions = scenario.generate_scan_positions(
            boundary,
            config["rect_size"],
            config["scan_step"],
            config["strategy"],
            random_seed=config.get("random_seed", 42),
        )
        results = scenario.scan_rectangles(coordinates, boundary, positions, config)
        cache_payload = [
            {key: value for key, value in result.items() if key != "geometry"}
            for result in results
        ]
        cache_path.write_text(
            json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved {len(results)} candidates: {cache_path.name}")

    scenario.verify_results(results, coordinates, config["rect_size"])
    if args.select_index is None:
        chosen = visualization.interactive_select(points_gdf, boundary, results, config)
    else:
        chosen = scenario.choose_result(results, args.select_index)
    if not chosen:
        print("No rectangle selected", file=sys.stderr)
        return 2

    try:
        dem_path = terrain.validate_dem_path(config["dem_path"])
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    with rasterio.open(dem_path) as dem:
        dem_crs = str(dem.crs)
        dem_resolution_m = float(abs(dem.res[0]))
        final_frame, selected_points = process_selected_rectangles(
            chosen, points_gdf, dem, config
        )
        if final_frame is None or final_frame.empty:
            print("No station data extracted", file=sys.stderr)
            return 2

        if config.get("save_csv", True):
            final_frame.to_csv(config["output_csv"], index=False, encoding="utf-8-sig")
            print(f"Scenario CSV: {config['output_csv']}")
        try:
            visualization.render_3d_terrain(chosen[0], selected_points, dem, config)
        except Exception as exc:
            print(f"WARNING: terrain rendering failed: {exc}", file=sys.stderr)

    if config.get("save_preview_png", True):
        try:
            preview_path = visualization.save_preview(points_gdf, boundary, chosen, config)
            print(f"Preview: {preview_path}")
        except Exception as exc:
            print(f"WARNING: preview generation failed: {exc}", file=sys.stderr)

    try:
        input_records = []
        for role, source_path, source_url, license_name in (
            (
                "base_station_points",
                config["points_shp"],
                "data/manifest.json",
                "Public redistribution permission confirmed by repository owner",
            ),
            (
                "administrative_boundary",
                config["boundary_shp"],
                "data/manifest.json",
                "Public redistribution permission confirmed by repository owner",
            ),
        ):
            for component in sorted(source_path.parent.glob(f"{source_path.stem}.*")):
                if component.is_file():
                    input_records.append(
                        io.build_dataset_record(
                            component,
                            name=f"{role}:{component.suffix.lstrip('.')}",
                            source_url=source_url,
                            license_name=license_name,
                            crs=str(points_gdf.crs),
                        )
                    )
        input_records.append(
            io.build_dataset_record(
                dem_path,
                name="dem",
                source_url=(
                    "https://developers.google.com/earth-engine/datasets/catalog/USGS_3DEP_1m"
                ),
                license_name="USGS public-domain data; retain source attribution",
                crs=dem_crs,
                resolution_m=dem_resolution_m,
            )
        )
        outputs = [
            path
            for path in (
                config["output_csv"],
                config["output_3d_png"],
                config["output_3d_png"].with_suffix(".eps"),
                config["output_3d_html"],
                config["preview_png"],
            )
            if path.exists()
        ]
        run_record = io.write_run_record(
            config["output_dir"],
            config=config,
            inputs=input_records,
            outputs=outputs,
            command=sys.argv if argv is None else ["lte-select-sites", *argv],
            filename="run-select-sites.json",
        )
        print(f"Run record: {run_record}")
    except Exception as exc:
        print(f"WARNING: run record generation failed: {exc}", file=sys.stderr)
    return 0
