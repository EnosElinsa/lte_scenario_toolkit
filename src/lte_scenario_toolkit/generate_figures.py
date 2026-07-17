"""Generate previewable terrain figures from a run or scenario CSV."""

from __future__ import annotations

import argparse
import os
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
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument(
        "--output-root",
        type=Path,
        help="root for a new uniquely named figure run",
    )
    outputs.add_argument(
        "--output-dir",
        type=Path,
        help="legacy exact output directory; existing artifacts are not overwritten",
    )
    parser.add_argument("--size", type=int, help="legacy rectangle size in metres")
    parser.add_argument("--target", type=int, help="legacy target base-station count")
    parser.add_argument("--rect-id", type=float, help="rectangle ID for a multi-rectangle CSV")
    parser.add_argument(
        "--preset",
        choices=("preview", "publication"),
    )
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--azimuth", type=float)
    parser.add_argument("--elevation-angle", type=float)
    parser.add_argument("--vertical-exaggeration", type=float)
    parser.add_argument("--colormap")
    parser.add_argument("--station-color")
    parser.add_argument("--station-size", type=float)
    parser.add_argument("--title")
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


def _profile_figure_spec(settings: Any) -> FigureSpec:
    base = FigureSpec.from_preset(settings.preset)
    return replace(
        base,
        colormap=settings.colormap,
        dpi=settings.dpi,
        azimuth=settings.azimuth_deg,
        elevation_angle=settings.elevation_deg,
        vertical_exaggeration=settings.vertical_exaggeration,
        station_color=settings.station_color,
        station_size=settings.station_marker_size,
        title=settings.title,
    ).validate()


def _figure_spec(args: Any, configured: FigureSpec | None = None) -> FigureSpec:
    spec = (
        FigureSpec.from_preset(args.preset)
        if args.preset is not None
        else configured or FigureSpec.from_preset("publication")
    )
    changes: dict[str, Any] = {}
    for argument, field in (
        (args.dpi, "dpi"),
        (args.azimuth, "azimuth"),
        (args.elevation_angle, "elevation_angle"),
        (args.vertical_exaggeration, "vertical_exaggeration"),
        (args.colormap, "colormap"),
        (args.station_color, "station_color"),
        (args.station_size, "station_size"),
        (args.title, "title"),
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
    discovery = service.discover_entries()
    errors = [item["error"] for item in discovery.diagnostics]
    for entry in reversed(discovery.entries):
        record = entry.record
        metadata = record.get("metadata", {})
        if (
            record["scenario_id"] != scenario_id
            or record["profile_id"] != profile_id
            or not isinstance(metadata, Mapping)
            or metadata.get("run_kind") != "selection"
        ):
            continue
        try:
            return FigureService.load_source(entry.run_dir)
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


def _config_source(
    args: Any,
) -> tuple[FigureSource, Path, tuple[str, ...], FigureSpec | None]:
    config = load_experiment_config(
        args.config,
        city=args.city,
        output_dir=None,
    )
    if args.size is not None:
        config["rect_size"] = args.size
    if args.target is not None:
        config["target_count"] = args.target
    profile = getattr(config, "profile_snapshot", None)
    if profile is not None:
        source, output_root, formats = _v2_config_source(args, config)
        return source, output_root, formats, _profile_figure_spec(profile.figure)
    source, output_root, formats = _legacy_config_source(args, config)
    return source, output_root, formats, None


def _default_output_root(source: FigureSource) -> Path:
    if source.path is None:
        raise ValueError("figure source has no path for a default output root")
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


def _source_and_output(
    args: Any,
) -> tuple[FigureSource, Path, tuple[str, ...], FigureSpec | None]:
    if args.config is not None:
        return _config_source(args)
    if args.city is not None or args.size is not None or args.target is not None:
        raise ValueError("--city, --size, and --target require --config")
    source_path = args.run_dir if args.run_dir is not None else args.csv
    source = FigureService.load_source(source_path, rect_id=args.rect_id)
    return source, _default_output_root(source), ("png", "html"), None


def _exact_output_directory(path: Path, formats: tuple[str, ...]) -> Path:
    raw = path.expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    if raw.is_symlink():
        raise ValueError(f"exact output directory must not be a symlink: {raw}")
    exact = raw.resolve(strict=False)
    if os.path.lexists(exact) and not exact.is_dir():
        raise ValueError(f"exact output directory must be a real directory: {exact}")
    names = {
        "source.csv",
        "run.json",
        "run-generate-figures.json",
        *(f"terrain.{value}" for value in formats),
    }
    conflicts = sorted(name for name in names if os.path.lexists(exact / name))
    if conflicts:
        raise FileExistsError(f"exact output conflict: {', '.join(conflicts)}")
    return exact


def _repository_for_command(args: Any) -> Path:
    if args.config is not None:
        config_path = args.config.expanduser().resolve(strict=False)
        for candidate in (config_path.parent, *config_path.parents):
            if (candidate / "data" / "datasets.yaml").is_file():
                return candidate.resolve()
    return Path.cwd().resolve()


def main(argv=None) -> int:
    """Run the figure workflow and map invalid input to CLI exit code 2."""

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    args = _parse_args(argv)
    try:
        source, default_output_root, defaults, configured_spec = _source_and_output(args)
        spec = _figure_spec(args, configured_spec)
        formats = tuple(args.formats) if args.formats is not None else defaults
        exact_output = (
            _exact_output_directory(args.output_dir, formats)
            if args.output_dir is not None
            else None
        )
        output_root = (
            args.output_root.resolve()
            if args.output_root is not None
            else exact_output or default_output_root
        )
        run_service = RunService(output_root)
        run_dir = FigureService.render(
            source,
            spec,
            run_service,
            formats,
            parent_run_id=source.run_id,
            entrypoint=(
                sys.argv
                if argv is None
                else ("lte-generate-figures", *argv)
            ),
            repository=_repository_for_command(args),
        )
        if exact_output is not None:
            run_dir = run_service.relocate_to_exact_directory(
                run_dir,
                exact_output,
                compatibility_record="run-generate-figures.json",
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
