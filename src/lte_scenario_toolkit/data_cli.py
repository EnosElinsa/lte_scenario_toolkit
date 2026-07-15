"""Manage registered LTE scenario data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .boundary_data import BoundaryImportError, register_scenario
from .data_catalog import CatalogError, load_data_catalog


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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the data-management CLI."""

    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (BoundaryImportError, CatalogError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
