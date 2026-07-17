"""Generate terrain figures from one completed toolkit run."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .figure_service import REQUIRED_COLUMNS as FIGURE_REQUIRED_COLUMNS
from .figure_service import FigureService, FigureSource, FigureSpec
from .run_service import RunService

REQUIRED_COLUMNS = set(FIGURE_REQUIRED_COLUMNS)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate 3D figures from a completed selection or figure run"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="completed run directory or its run.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="root for a new uniquely named figure run",
    )
    parser.add_argument("--preset", choices=("preview", "publication"))
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
    return parser.parse_args(argv)


def _figure_spec(args: Any) -> FigureSpec:
    spec = FigureSpec.from_preset(args.preset or "publication")
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


def _default_output_root(source: FigureSource) -> Path:
    if source.path is None or not source.path.is_dir():
        raise ValueError("completed run source has no directory")
    if (
        source.profile_id is not None
        and source.scenario_id is not None
        and source.path.parent.name == source.profile_id
        and source.path.parent.parent.name == source.scenario_id
    ):
        return source.path.parents[2]
    return (source.path / "figure-runs").resolve()


def main(argv=None) -> int:
    """Run the current figure workflow and map invalid input to exit code 2."""

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    args = _parse_args(argv)
    try:
        source = FigureService.load_source(args.run_dir)
        spec = _figure_spec(args)
        formats = tuple(args.formats) if args.formats is not None else ("png", "html")
        output_root = (
            args.output_root.expanduser().resolve(strict=False)
            if args.output_root is not None
            else _default_output_root(source)
        )
        run_dir = FigureService.render(
            source,
            spec,
            RunService(output_root),
            formats,
            parent_run_id=source.run_id,
            entrypoint=(
                sys.argv if argv is None else ("lte-generate-figures", *argv)
            ),
            repository=Path.cwd().resolve(),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"Rectangle center: ({source.rectangle['center_x']:.1f}, "
        f"{source.rectangle['center_y']:.1f}); stations: "
        f"{source.rectangle['pt_count']}"
    )
    if "elevation" in source.frame.columns:
        print(f"Mean elevation: {source.frame['elevation'].mean():.1f} m")
    print(f"Figure run: {run_dir}")
    return 0


__all__ = ["REQUIRED_COLUMNS", "main"]
