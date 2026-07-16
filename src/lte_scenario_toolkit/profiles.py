"""Versioned experiment-profile models and loaders."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROFILE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)

DEFAULT_PROFILE_VALUES: dict[str, Any] = {
    "target_crs": "EPSG:3857",
    "rect_size": 3000,
    "target_count": 30,
    "tolerance": 0,
    "scan_mode": "fast",
    "strategy": "uniform",
    "scan_step": 10,
    "max_rects": 100,
    "min_spacing": 3000,
    "random_seed": 42,
}

_MISSING = object()


@dataclass(frozen=True)
class OutputSettings:
    """Artifact switches owned by an experiment profile."""

    save_csv: bool = True
    save_preview_png: bool = True
    save_terrain_png: bool = True
    save_terrain_eps: bool = True
    save_terrain_html: bool = True


@dataclass(frozen=True)
class FigureSettings:
    """Figure styling owned by an experiment profile."""

    preset: str = "publication"
    colormap: str = "terrain"
    dpi: int = 300
    azimuth_deg: float = -60.0
    elevation_deg: float = 30.0
    vertical_exaggeration: float = 1.0
    station_color: str = "red"
    station_marker_size: float = 20.0
    title: str | None = None


@dataclass(frozen=True)
class ExperimentProfile:
    """Validated schema-version-2 experiment settings."""

    schema_version: int
    profile_id: str
    display_name: str
    scenario_id: str
    points_dataset_id: str
    random_seed: int
    target_crs: str
    rect_size: int
    target_count: int
    tolerance: int
    scan_mode: str
    strategy: str
    scan_step: int
    max_rects: int
    min_spacing: int
    output_root: Path
    outputs: OutputSettings = field(default_factory=OutputSettings)
    figure: FigureSettings = field(default_factory=FigureSettings)
    source_path: Path | None = None

    def runtime_values(self) -> dict[str, Any]:
        """Return the flat mapping consumed by existing workflow services."""

        return {
            "profile_id": self.profile_id,
            "scenario_id": self.scenario_id,
            "points_dataset_id": self.points_dataset_id,
            "random_seed": self.random_seed,
            "target_crs": self.target_crs,
            "rect_size": self.rect_size,
            "target_count": self.target_count,
            "tolerance": self.tolerance,
            "scan_mode": self.scan_mode,
            "strategy": self.strategy,
            "scan_step": self.scan_step,
            "max_rects": self.max_rects,
            "min_spacing": self.min_spacing,
            "output_root": self.output_root,
            "save_csv": self.outputs.save_csv,
            "save_preview_png": self.outputs.save_preview_png,
            "save_terrain_png": self.outputs.save_terrain_png,
            "save_terrain_eps": self.outputs.save_terrain_eps,
            "save_terrain_html": self.outputs.save_terrain_html,
            "config_path": self.source_path,
        }


def _mapping(
    document: Mapping[str, Any],
    section: str,
) -> Mapping[str, Any]:
    value = document.get(section, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section {section} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, prefix: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(
            f"Missing required configuration value: {prefix}.{key}"
        ) from exc


def _value(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    *,
    default: Any = _MISSING,
) -> Any:
    if key in mapping:
        return mapping[key]
    if default is not _MISSING:
        return default
    prefix, _, _ = path.rpartition(".")
    return _required(mapping, key, prefix)


def _string_value(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    *,
    default: Any = _MISSING,
) -> str:
    value = _value(mapping, key, path, default=default)
    if type(value) is not str:
        raise ValueError(f"{path} must be a string")
    return value


def _optional_string_value(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    *,
    default: str | None,
) -> str | None:
    value = _value(mapping, key, path, default=default)
    if value is not None and type(value) is not str:
        raise ValueError(f"{path} must be null or a string")
    return value


def _integer_value(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    *,
    default: Any = _MISSING,
) -> int:
    value = _value(mapping, key, path, default=default)
    if type(value) is not int:
        raise ValueError(f"{path} must be an integer")
    return value


def _boolean_value(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    *,
    default: bool,
) -> bool:
    value = _value(mapping, key, path, default=default)
    if type(value) is not bool:
        raise ValueError(f"{path} must be a boolean")
    return value


def _finite_float_value(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    *,
    default: float,
) -> float:
    value = _value(mapping, key, path, default=default)
    if type(value) not in {int, float}:
        raise ValueError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_profile(profile: ExperimentProfile) -> ExperimentProfile:
    if (
        PROFILE_ID_PATTERN.fullmatch(profile.profile_id) is None
        or profile.profile_id in WINDOWS_RESERVED
    ):
        raise ValueError(
            "profile.id must be a safe lowercase slug and not a Windows device name"
        )
    if profile.scan_mode not in {"fast", "complete"}:
        raise ValueError("scan.mode must be one of: complete, fast")
    if profile.strategy not in {"sequential", "uniform"}:
        raise ValueError("scan.strategy must be one of: sequential, uniform")
    for key, value in (
        ("rect_size", profile.rect_size),
        ("scan_step", profile.scan_step),
        ("max_rects", profile.max_rects),
        ("min_spacing", profile.min_spacing),
    ):
        if value <= 0:
            raise ValueError(f"{key} must be greater than zero")
    if profile.target_count < 0 or profile.tolerance < 0:
        raise ValueError("target_count and tolerance must be non-negative")
    if profile.figure.dpi <= 0:
        raise ValueError("figures.dpi must be greater than zero")
    if profile.figure.vertical_exaggeration <= 0:
        raise ValueError("figures.vertical_exaggeration must be greater than zero")
    return profile


def load_profile(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> ExperimentProfile:
    """Load and validate one schema-version-2 experiment profile."""

    source_path = Path(path).resolve()
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else source_path.parent
    )
    document = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        raise ValueError("Configuration document must be a mapping")
    if "schema_version" not in document:
        raise ValueError("Missing required configuration value: schema_version")
    schema_version = document["schema_version"]
    if type(schema_version) is not int or schema_version != 2:
        raise ValueError("schema_version must be the integer 2")

    profile_section = _mapping(document, "profile")
    inputs = _mapping(document, "inputs")
    experiment = _mapping(document, "experiment")
    spatial = _mapping(document, "spatial")
    scan = _mapping(document, "scan")
    outputs = _mapping(document, "outputs")
    figures = _mapping(document, "figures")

    output_defaults = OutputSettings()
    output_settings = OutputSettings(
        save_csv=_boolean_value(
            outputs,
            "save_csv",
            "outputs.save_csv",
            default=output_defaults.save_csv,
        ),
        save_preview_png=_boolean_value(
            outputs,
            "save_preview_png",
            "outputs.save_preview_png",
            default=output_defaults.save_preview_png,
        ),
        save_terrain_png=_boolean_value(
            outputs,
            "save_terrain_png",
            "outputs.save_terrain_png",
            default=output_defaults.save_terrain_png,
        ),
        save_terrain_eps=_boolean_value(
            outputs,
            "save_terrain_eps",
            "outputs.save_terrain_eps",
            default=output_defaults.save_terrain_eps,
        ),
        save_terrain_html=_boolean_value(
            outputs,
            "save_terrain_html",
            "outputs.save_terrain_html",
            default=output_defaults.save_terrain_html,
        ),
    )
    figure_defaults = FigureSettings()
    figure_settings = FigureSettings(
        preset=_string_value(
            figures,
            "preset",
            "figures.preset",
            default=figure_defaults.preset,
        ),
        colormap=_string_value(
            figures,
            "colormap",
            "figures.colormap",
            default=figure_defaults.colormap,
        ),
        dpi=_integer_value(
            figures,
            "dpi",
            "figures.dpi",
            default=figure_defaults.dpi,
        ),
        azimuth_deg=_finite_float_value(
            figures,
            "azimuth_deg",
            "figures.azimuth_deg",
            default=figure_defaults.azimuth_deg,
        ),
        elevation_deg=_finite_float_value(
            figures,
            "elevation_deg",
            "figures.elevation_deg",
            default=figure_defaults.elevation_deg,
        ),
        vertical_exaggeration=_finite_float_value(
            figures,
            "vertical_exaggeration",
            "figures.vertical_exaggeration",
            default=figure_defaults.vertical_exaggeration,
        ),
        station_color=_string_value(
            figures,
            "station_color",
            "figures.station_color",
            default=figure_defaults.station_color,
        ),
        station_marker_size=_finite_float_value(
            figures,
            "station_marker_size",
            "figures.station_marker_size",
            default=figure_defaults.station_marker_size,
        ),
        title=_optional_string_value(
            figures,
            "title",
            "figures.title",
            default=figure_defaults.title,
        ),
    )

    profile = ExperimentProfile(
        schema_version=2,
        profile_id=_string_value(
            profile_section,
            "id",
            "profile.id",
        ),
        display_name=_string_value(
            profile_section,
            "display_name",
            "profile.display_name",
        ),
        scenario_id=_string_value(
            profile_section,
            "scenario_id",
            "profile.scenario_id",
        ),
        points_dataset_id=_string_value(
            inputs,
            "points_dataset_id",
            "inputs.points_dataset_id",
        ),
        random_seed=_integer_value(
            experiment,
            "random_seed",
            "experiment.random_seed",
            default=DEFAULT_PROFILE_VALUES["random_seed"],
        ),
        target_crs=_string_value(
            spatial,
            "target_crs",
            "spatial.target_crs",
        ),
        rect_size=_integer_value(
            spatial,
            "rectangle_size_m",
            "spatial.rectangle_size_m",
        ),
        target_count=_integer_value(
            spatial,
            "target_base_station_count",
            "spatial.target_base_station_count",
        ),
        tolerance=_integer_value(
            spatial,
            "count_tolerance",
            "spatial.count_tolerance",
        ),
        scan_mode=_string_value(
            scan,
            "mode",
            "scan.mode",
            default=DEFAULT_PROFILE_VALUES["scan_mode"],
        ),
        strategy=_string_value(
            scan,
            "strategy",
            "scan.strategy",
        ),
        scan_step=_integer_value(
            scan,
            "step_m",
            "scan.step_m",
        ),
        max_rects=_integer_value(
            scan,
            "max_rectangles",
            "scan.max_rectangles",
        ),
        min_spacing=_integer_value(
            scan,
            "minimum_center_spacing_m",
            "scan.minimum_center_spacing_m",
        ),
        output_root=_resolve_path(
            _string_value(
                outputs,
                "root",
                "outputs.root",
                default="results",
            ),
            root,
        ),
        outputs=output_settings,
        figure=figure_settings,
        source_path=source_path,
    )
    return _validate_profile(profile)
