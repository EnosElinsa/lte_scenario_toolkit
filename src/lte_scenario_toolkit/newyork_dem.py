"""Export the USGS 3DEP 1 m DEM for New York City from Earth Engine.

The default ROI is the five-county New York City boundary assembled from the
official TIGER/2018/Counties FeatureCollection.  Use ``--boundary-mode county``
to export New York County (Manhattan) only.

No Earth Engine export task is started unless ``--export`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

DEM_COLLECTION = "USGS/3DEP/1m"
COUNTY_COLLECTION = "TIGER/2018/Counties"
TARGET_CRS = "EPSG:3857"
DEFAULT_SCALE_M = 1
DEFAULT_FILE_DIMENSIONS = 8192
DEFAULT_SHARD_SIZE = 256
DEFAULT_MAX_PIXELS = 1e13
DEFAULT_DRIVE_FOLDER = "usa-lte-base-station-data"
DEFAULT_PREFIX = "USGS_1M_DEM_NewYorkState_NewYork"
NYC_COUNTY_GEOIDS = ("36005", "36047", "36061", "36081", "36085")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export USGS 3DEP 1 m DEM for New York City in EPSG:3857."
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("EE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="Earth Engine Cloud Project ID (or set EE_PROJECT).",
    )
    parser.add_argument(
        "--boundary-mode",
        choices=("city", "county"),
        default="city",
        help="city = five NYC counties (default); county = one county.",
    )
    parser.add_argument(
        "--county-geoid",
        default="36061",
        help="County GEOID when --boundary-mode county (default: 36061, New York County).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE_M,
        help="Export scale in metres; 1 m is the dataset's native pixel size.",
    )
    parser.add_argument(
        "--file-dimensions",
        type=int,
        default=DEFAULT_FILE_DIMENSIONS,
        help="Per-file tile dimension for Drive export (multiple of --shard-size).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Internal computation shard size in pixels.",
    )
    parser.add_argument(
        "--max-pixels",
        type=float,
        default=DEFAULT_MAX_PIXELS,
        help="Earth Engine maxPixels guard for the export.",
    )
    parser.add_argument(
        "--drive-folder",
        default=DEFAULT_DRIVE_FOLDER,
        help="Google Drive folder for the exported GeoTIFF tiles.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Prefix for the exported GeoTIFF tile names.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print parameters only; do not import or contact Earth Engine.",
    )
    mode.add_argument(
        "--export",
        action="store_true",
        help="Start the Earth Engine Drive export task after building the image.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.scale <= 0:
        raise ValueError("--scale must be greater than zero.")
    if args.file_dimensions <= 0 or args.shard_size <= 0:
        raise ValueError("--file-dimensions and --shard-size must be positive.")
    if args.file_dimensions % args.shard_size != 0:
        raise ValueError("--file-dimensions must be a multiple of --shard-size.")
    if args.max_pixels <= 0:
        raise ValueError("--max-pixels must be greater than zero.")
    if not re.fullmatch(r"\d{5}", args.county_geoid):
        raise ValueError("--county-geoid must be a five-digit Census county GEOID.")


def print_dry_run(args: argparse.Namespace) -> None:
    if args.boundary_mode == "city":
        roi_description = (
            "TIGER/2018/Counties, STATEFP=36, GEOID in "
            + repr(list(NYC_COUNTY_GEOIDS))
            + " (New York City five counties)"
        )
    else:
        roi_description = (
            f"TIGER/2018/Counties, GEOID={args.county_geoid} "
            "(New York County/Manhattan by default)"
        )
    print(
        json.dumps(
            {
                "project": args.project or "<required: pass --project or set EE_PROJECT>",
                "roi": roi_description,
                "dataset": DEM_COLLECTION,
                "band": "elevation",
                "scale_m": args.scale,
                "crs": TARGET_CRS,
                "export_target": "Google Drive GeoTIFF (sharded)",
                "drive_folder": args.drive_folder,
                "file_name_prefix": args.prefix,
                "file_dimensions": args.file_dimensions,
                "shard_size": args.shard_size,
                "max_pixels": args.max_pixels,
                "export_started": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def import_gee() -> tuple[Any, Any]:
    """Import ee/geemap lazily so --dry-run works without GEE packages."""

    # A run artifact named code.py can shadow Python's standard-library code
    # module, which geemap imports indirectly.  Remove that directory first.
    script_dir = Path(__file__).resolve().parent
    if sys.path and Path(sys.path[0] or ".").resolve() == script_dir:
        sys.path.pop(0)
    try:
        import ee  # type: ignore
        import geemap  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Earth Engine packages. Install with "
            "python -m pip install earthengine-api geemap."
        ) from exc
    return ee, geemap


def initialize_gee(project: str) -> Any:
    if not project:
        raise RuntimeError(
            "Cloud Project ID is required. Pass --project "
            "gen-lang-client-0153149292 or set EE_PROJECT."
        )
    ee, _geemap = import_gee()
    try:
        # Use the Earth Engine API directly.  Recent geemap releases do not
        # expose geemap.ee_initialize; importing geemap remains useful for the
        # rest of the project's optional vector workflows.
        ee.Initialize(project=project)
    except Exception as exc:  # pragma: no cover - depends on local credentials
        raise RuntimeError(
            "Earth Engine initialization failed: "
            f"{type(exc).__name__}: {exc}. Run `earthengine authenticate` "
            "once, then retry with the same --project."
        ) from exc
    return ee


def build_roi(ee: Any, args: argparse.Namespace) -> tuple[Any, Any, list[str]]:
    counties = ee.FeatureCollection(COUNTY_COLLECTION).filter(
        ee.Filter.eq("STATEFP", "36")
    )
    if args.boundary_mode == "city":
        boundary_fc = counties.filter(ee.Filter.inList("GEOID", list(NYC_COUNTY_GEOIDS)))
        label = "New York City (five counties)"
    else:
        boundary_fc = counties.filter(ee.Filter.eq("GEOID", args.county_geoid))
        label = f"county {args.county_geoid}"

    count = int(boundary_fc.size().getInfo())
    if count == 0:
        raise RuntimeError(
            f"No county boundary matched {label}. Check TIGER/2018/Counties fields."
        )
    names = [str(name) for name in boundary_fc.aggregate_array("NAME").getInfo()]
    roi = boundary_fc.geometry()
    return boundary_fc, roi, names


def build_dem(ee: Any, roi: Any) -> tuple[Any, int]:
    collection = ee.ImageCollection(DEM_COLLECTION).filterBounds(roi)
    image_count = int(collection.size().getInfo())
    if image_count == 0:
        raise RuntimeError(
            f"{DEM_COLLECTION} returned no tiles intersecting the selected ROI."
        )
    # The collection is tiled. mosaic() joins the intersecting tiles; clip() is
    # applied once at the end so the server does not repeat clipping per tile.
    dem = collection.mosaic().select("elevation").clip(roi)
    return dem, image_count


def start_export(ee: Any, dem: Any, roi: Any, args: argparse.Namespace) -> Any:
    task = ee.batch.Export.image.toDrive(
        image=dem,
        description=args.prefix,
        folder=args.drive_folder,
        fileNamePrefix=args.prefix,
        region=roi,
        scale=args.scale,
        crs=TARGET_CRS,
        maxPixels=args.max_pixels,
        fileDimensions=args.file_dimensions,
        shardSize=args.shard_size,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return task


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.dry_run:
        print_dry_run(args)
        return 0

    ee = initialize_gee(args.project)
    boundary_fc, roi, names = build_roi(ee, args)
    dem, image_count = build_dem(ee, roi)
    print(f"Boundary features: {len(names)} -> {', '.join(names)}")
    print(f"DEM collection tiles intersecting ROI: {image_count}")
    print(f"Dataset: {DEM_COLLECTION}, band=elevation, scale={args.scale:g} m")
    print(f"Export CRS: {TARGET_CRS}")
    print(f"Export region: exact TIGER county geometry ({args.boundary_mode})")
    print(f"Drive output: {args.drive_folder}/{args.prefix}_*.tif")

    if not args.export:
        print("Preview only; no export task started. Re-run with --export to submit it.")
        return 0

    task = start_export(ee, dem, roi, args)
    print(f"Export task started: {task.id}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
