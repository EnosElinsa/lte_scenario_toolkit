"""Manage registered LTE scenario data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .boundary_data import BoundaryImportError, register_scenario
from .data_catalog import CatalogError, load_data_catalog
from .data_validation import validate_scenario_data
from .dem_data import (
    DemIngestError,
    EarthEngineExportError,
    build_dem_export_plan,
    execute_dem_export,
    ingest_dem_shards,
    write_export_run,
)


def _scenario_add(args: argparse.Namespace) -> int:
    register_scenario(
        args.catalog,
        scenario_id=args.scenario_id,
        display_name=args.display_name or args.scenario_id,
        boundary_source=args.boundary_source,
        provider=args.provider,
        license_name=args.license_name,
        redistribution_confirmed=args.redistribution_confirmed,
        layer=args.layer,
        download_date=args.download_date,
        config_path=args.config_path,
    )
    print(f"Registered scenario: {args.scenario_id}")
    return 0


def _scenario_list(args: argparse.Namespace) -> int:
    catalog = load_data_catalog(args.catalog)
    for scenario_id in sorted(catalog.scenarios_by_id):
        scenario = catalog.scenario(scenario_id)
        print(f"{scenario_id}\t{catalog.scenario_status(scenario_id)}\t{scenario['display_name']}")
    return 0


def _scenario_show(args: argparse.Namespace) -> int:
    catalog = load_data_catalog(args.catalog)
    scenario = catalog.scenario(args.scenario_id)
    boundary = catalog.dataset(scenario["boundary_dataset_id"])
    dem_id = scenario["dem_dataset_id"]
    dem_path = "<not declared>" if dem_id is None else catalog.dataset(dem_id)["entrypoint"]
    print(f"scenario_id: {scenario['scenario_id']}")
    print(f"display_name: {scenario['display_name']}")
    print(f"status: {catalog.scenario_status(args.scenario_id)}")
    print(f"boundary: {boundary['entrypoint']}")
    print(f"dem: {dem_path}")
    return 0


def _dem_export(args: argparse.Namespace) -> int:
    catalog = load_data_catalog(args.catalog)
    plan = build_dem_export_plan(
        catalog,
        args.scenario_id,
        project=args.project,
        scale_m=args.scale,
        file_dimensions=args.file_dimensions,
        shard_size=args.shard_size,
        max_pixels=args.max_pixels,
        drive_folder=args.drive_folder,
    )
    if args.dry_run:
        print(json.dumps(plan.json_dict(), ensure_ascii=False, indent=2))
        return 0

    result = execute_dem_export(plan, start=args.start_export)
    run_path = write_export_run(plan, result, runs_root=catalog.root / "runs")
    print(f"Image count: {result.image_count}")
    print(f"Task ID: {result.task_id or '<not started>'}")
    print(f"Run path: {run_path}")
    return 0


def _dem_ingest(args: argparse.Namespace) -> int:
    catalog = load_data_catalog(args.catalog)
    ingested = ingest_dem_shards(catalog, args.scenario_id, args.tiles_dir)
    scenario = ingested.scenario(args.scenario_id)
    dem_id = scenario["dem_dataset_id"]
    if dem_id is None:
        # The ingest function normally raises this first; retain a defensive
        # guard for custom catalog implementations used by callers/tests.
        raise DemIngestError(
            f"Scenario {args.scenario_id!r} does not declare a DEM"
        )
    dem_path = ingested.resolve(ingested.dataset(dem_id)["entrypoint"])
    print(f"Final DEM: {dem_path}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    """Validate one scenario or every scenario in stable ID order."""

    if bool(args.all_scenarios) == bool(args.scenario_id):
        raise CatalogError("validate requires exactly one scenario ID or --all")
    catalog = load_data_catalog(args.catalog)
    scenario_ids = (
        sorted(catalog.scenarios_by_id)
        if args.all_scenarios
        else [args.scenario_id]
    )
    reports = [
        validate_scenario_data(
            catalog,
            scenario_id,
            full_checksum=args.full_checksum,
        )
        for scenario_id in scenario_ids
    ]
    for report in reports:
        state = "ok" if report.ok else "failed"
        print(f"{report.scenario_id}: {report.status} ({state})")
        for message in report.messages:
            print(f"  {message.level.upper()} {message.code}: {message.message}")
    return 0 if all(report.ok for report in reports) else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the extensible data-management command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/datasets.yaml"),
        help="schema-v2 dataset catalog (default: data/datasets.yaml)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    scenario = commands.add_parser("scenario", help="manage scenario registrations")
    scenario_commands = scenario.add_subparsers(dest="scenario_command", required=True)

    add = scenario_commands.add_parser("add", help="register a scenario boundary")
    add.add_argument("scenario_id", metavar="id")
    add.add_argument("--display-name")
    add.add_argument("--boundary-source", required=True)
    add.add_argument("--provider", required=True)
    add.add_argument("--license", dest="license_name", required=True)
    add.add_argument("--layer")
    add.add_argument("--download-date")
    add.add_argument("--config-path", type=Path)
    add.add_argument("--redistribution-confirmed", action="store_true")
    add.set_defaults(handler=_scenario_add)

    list_command = scenario_commands.add_parser("list", help="list scenarios")
    list_command.set_defaults(handler=_scenario_list)

    show = scenario_commands.add_parser("show", help="show one scenario")
    show.add_argument("scenario_id", metavar="id")
    show.set_defaults(handler=_scenario_show)

    dem = commands.add_parser("dem", help="manage scenario DEM data")
    dem_commands = dem.add_subparsers(dest="dem_command", required=True)
    export = dem_commands.add_parser(
        "export",
        help="preflight or submit a registered-scenario Earth Engine DEM export",
    )
    export.add_argument("scenario_id")
    export.add_argument(
        "--project",
        default=os.environ.get("EE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="Earth Engine Cloud Project ID (or set EE_PROJECT)",
    )
    export.add_argument(
        "--scale",
        type=float,
        help="export scale in metres (default: registered DEM native scale)",
    )
    export.add_argument(
        "--file-dimensions",
        type=int,
        default=8192,
        help="per-file tile dimension (default: 8192)",
    )
    export.add_argument(
        "--shard-size",
        type=int,
        default=256,
        help="Earth Engine computation shard size (default: 256)",
    )
    export.add_argument(
        "--max-pixels",
        type=float,
        default=1e13,
        help="Earth Engine maxPixels guard (default: 1e13)",
    )
    export.add_argument(
        "--drive-folder",
        help="Google Drive folder (default: registered DEM folder)",
    )
    mode = export.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved plan without importing or contacting Earth Engine",
    )
    mode.add_argument(
        "--export",
        dest="start_export",
        action="store_true",
        help="explicitly start the Earth Engine Drive export task",
    )
    export.set_defaults(handler=_dem_export)
    ingest = dem_commands.add_parser(
        "ingest",
        help="merge manually downloaded DEM shards and register the final raster",
    )
    ingest.add_argument("scenario_id")
    ingest.add_argument(
        "--tiles-dir",
        type=Path,
        required=True,
        help="directory containing downloaded GeoTIFF shards",
    )
    ingest.set_defaults(handler=_dem_ingest)

    validate = commands.add_parser(
        "validate",
        help="validate registered scenario boundary, manifest, DEM, and config data",
    )
    validate.add_argument("scenario_id", nargs="?", metavar="id")
    validate.add_argument(
        "--all",
        dest="all_scenarios",
        action="store_true",
        help="validate every registered scenario in sorted ID order",
    )
    validate.add_argument(
        "--full-checksum",
        action="store_true",
        help="compute and compare SHA256 values (fast mode checks sizes only)",
    )
    validate.set_defaults(handler=_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the data-management CLI."""

    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (
        BoundaryImportError,
        CatalogError,
        DemIngestError,
        EarthEngineExportError,
        FileNotFoundError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
