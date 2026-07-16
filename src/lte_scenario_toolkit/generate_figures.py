"""Generate previewable terrain figures from a run or scenario CSV."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import load_experiment_config
from .data_catalog import load_data_catalog
from .figure_service import (
    REQUIRED_COLUMNS as FIGURE_REQUIRED_COLUMNS,
)
from .figure_service import (
    FigureService,
    FigureSource,
    FigureSpec,
)
from .run_service import RunService
from .spatial import resolve_io_paths
from .terrain import validate_dem_path

REQUIRED_COLUMNS = set(FIGURE_REQUIRED_COLUMNS)


def load_scenario_csv(path: str | Path):
    """Compatibility loader returning the former frame/rectangle/points tuple."""

    source = FigureService.load_source(path)
    return source.frame, source.rectangle, source.points


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate 3D figures from a selection run or scenario CSV"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="experiment YAML source or context for --csv",
    )
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--run-dir", type=Path, help="selection run directory")
    sources.add_argument("--csv", type=Path, help="legacy scenario CSV")
    parser.add_argument("--city", help="legacy boundary directory or layer name")
    parser.add_argument("--output-dir", type=Path, help="root for new figure runs")
    parser.add_argument("--size", type=int, help="legacy rectangle size in metres")
    parser.add_argument("--target", type=int, help="legacy target base-station count")
    parser.add_argument("--rect-id", type=float, help="rectangle ID for a multi-rectangle CSV")
    parser.add_argument(
        "--preset",
        choices=("preview", "publication"),
        default="publication",
    )
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--azimuth", type=float)
    parser.add_argument("--elevation-angle", type=float)
    parser.add_argument("--vertical-exaggeration", type=float)
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=("png", "eps", "html"),
        help="repeat for each requested output format",
    )
    args = parser.parse_args(argv)
    if args.config is None and args.run_dir is None and args.csv is None:
        parser.error("one of --config, --run-dir, or --csv is required")
    if args.config is not None and args.run_dir is not None:
        parser.error("--config cannot be combined with --run-dir")
    return args


def _figure_spec(args) -> FigureSpec:
    spec = FigureSpec.from_preset(args.preset)
    changes: dict[str, Any] = {}
    for argument, field in (
        (args.dpi, "dpi"),
        (args.azimuth, "azimuth"),
        (args.elevation_angle, "elevation_angle"),
        (args.vertical_exaggeration, "vertical_exaggeration"),
    ):
        if argument is not None:
            changes[field] = argument
    return replace(spec, **changes).validate()


def _legacy_formats(config: dict[str, Any]) -> tuple[str, ...]:
    formats: list[str] = []
    if config.get("save_terrain_png", True):
        formats.append("png")
    if config.get("save_terrain_eps", False):
        formats.append("eps")
    if config.get("save_terrain_html", True):
        formats.append("html")
    return tuple(formats)


def _contextual_source(
    source: FigureSource,
    *,
    dem_path: Path,
    target_crs: str,
    rectangle_size: float,
    scenario_id: str,
    profile_id: str,
) -> FigureSource:
    if source.run_id is not None:
        return replace(
            source,
            dem_path=source.dem_path or dem_path.resolve(),
        )
    points = source.points.set_crs(target_crs, allow_override=True)
    return replace(
        source,
        points=points,
        target_crs=target_crs,
        rectangle_size_m=rectangle_size,
        dem_path=dem_path.resolve(),
        warnings=(),
        scenario_id=scenario_id,
        profile_id=profile_id,
    )


def _legacy_config_source(
    args,
    config: dict[str, Any],
) -> tuple[FigureSource, Path, tuple[str, ...]]:
    if args.csv is None:
        resolved = resolve_io_paths(config, create_output=False)
        csv_path = resolved["output_csv"]
        dem_value = resolved.get("dem_path", config.get("dem_path"))
    else:
        csv_path = args.csv
        dem_value = config.get("dem_path")
    source = FigureService.load_source(csv_path, rect_id=args.rect_id)
    dem_path = validate_dem_path(dem_value)
    target_crs = str(config.get("target_crs", source.target_crs))
    rectangle_size = float(config.get("rect_size", source.rectangle_size_m))
    source = _contextual_source(
        source,
        dem_path=Path(dem_path),
        target_crs=target_crs,
        rectangle_size=rectangle_size,
        scenario_id=config.get("scenario_id") or "legacy",
        profile_id=config.get("profile_id") or "figures",
    )
    output_root = Path(config["output_root"]).resolve()
    return source, output_root, _legacy_formats(config)


def _latest_profile_run(
    output_root: Path,
    scenario_id: str,
    profile_id: str,
) -> FigureSource:
    service = RunService(output_root)
    discovery = service.discover()
    errors = [item["error"] for item in discovery.diagnostics]
    for record in reversed(discovery.records):
        metadata = record.get("metadata", {})
        if (
            record["scenario_id"] != scenario_id
            or record["profile_id"] != profile_id
            or not isinstance(metadata, Mapping)
            or metadata.get("run_kind") != "selection"
        ):
            continue
        try:
            _, final_path = service._expected_paths(
                scenario_id=record["scenario_id"],
                profile_id=record["profile_id"],
                created_at=record["created_at"],
                run_id=record["run_id"],
            )
            return FigureService.load_source(final_path)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{record['run_id']}: {exc}")
    suffix = f" Checked: {'; '.join(errors)}" if errors else ""
    raise ValueError(
        "No compatible selection run was found for this profile; provide --csv "
        f"or --run-dir.{suffix}"
    )


def _v2_config_source(
    args,
    config,
) -> tuple[FigureSource, Path, tuple[str, ...]]:
    profile = config.profile_snapshot
    if profile is None:
        raise ValueError("schema-version-2 configuration has no profile snapshot")
    repository = Path(config["repo_root"]).resolve()
    catalog = load_data_catalog(
        repository / "data" / "datasets.yaml",
        repo_root=repository,
    )
    scenario = catalog.scenario(profile.scenario_id)
    dem_id = scenario.get("dem_dataset_id")
    if type(dem_id) is not str or not dem_id:
        raise ValueError(f"Scenario {profile.scenario_id!r} has no registered DEM")
    dem_record = catalog.dataset(dem_id)
    if dem_record["role"] != "dem":
        raise ValueError(f"Registered dataset {dem_id!r} is not a DEM")
    dem_path = validate_dem_path(catalog.resolve(dem_record["entrypoint"]))
    output_root = Path(config["output_root"]).resolve()
    if args.csv is None:
        source = _latest_profile_run(
            output_root,
            profile.scenario_id,
            profile.profile_id,
        )
    else:
        source = FigureService.load_source(args.csv, rect_id=args.rect_id)
        source = _contextual_source(
            source,
            dem_path=Path(dem_path),
            target_crs=str(config["target_crs"]),
            rectangle_size=float(config["rect_size"]),
            scenario_id=profile.scenario_id,
            profile_id=profile.profile_id,
        )
    return source, output_root, _legacy_formats(config)


def _config_source(args) -> tuple[FigureSource, Path, tuple[str, ...]]:
    config = load_experiment_config(
        args.config,
        city=args.city,
        output_dir=args.output_dir,
    )
    if args.size is not None:
        config["rect_size"] = args.size
    if args.target is not None:
        config["target_count"] = args.target
    if getattr(config, "profile_snapshot", None) is not None:
        return _v2_config_source(args, config)
    return _legacy_config_source(args, config)


def _default_output_root(source: FigureSource) -> Path:
    if (
        source.path.is_dir()
        and source.profile_id is not None
        and source.scenario_id is not None
        and source.path.parent.name == source.profile_id
        and source.path.parent.parent.name == source.scenario_id
    ):
        return source.path.parents[2]
    base = source.path if source.path.is_dir() else source.path.parent
    return (base / "figure-runs").resolve()


def _source_and_output(args) -> tuple[FigureSource, Path, tuple[str, ...]]:
    if args.config is not None:
        return _config_source(args)
    if args.city is not None or args.size is not None or args.target is not None:
        raise ValueError("--city, --size, and --target require --config")
    source_path = args.run_dir if args.run_dir is not None else args.csv
    source = FigureService.load_source(source_path, rect_id=args.rect_id)
    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _default_output_root(source)
    )
    return source, output_root, ("png", "html")


def main(argv=None) -> int:
    """Run the figure workflow and map invalid input to CLI exit code 2."""

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    args = _parse_args(argv)
    try:
        spec = _figure_spec(args)
        source, output_root, defaults = _source_and_output(args)
        formats = tuple(args.formats) if args.formats is not None else defaults
        run_dir = FigureService.render(
            source,
            spec,
            RunService(output_root),
            formats,
            parent_run_id=source.run_id,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Scenario CSV: {source.csv_path}")
    print(
        f"Rectangle center: ({source.rectangle['center_x']:.1f}, "
        f"{source.rectangle['center_y']:.1f}); stations: "
        f"{source.rectangle['pt_count']}"
    )
    if "elevation" in source.frame.columns:
        print(f"Mean elevation: {source.frame['elevation'].mean():.1f} m")
    for warning in source.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"Figure run: {run_dir}")
    return 0


__all__ = ["REQUIRED_COLUMNS", "load_scenario_csv", "main"]
