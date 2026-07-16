"""Versioned experiment-profile models and loaders."""

from __future__ import annotations

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
    if not isinstance(document, Mapping) or document.get("schema_version") != 2:
        raise ValueError("Expected a schema-version-2 experiment profile")

    profile_section = _mapping(document, "profile")
    inputs = _mapping(document, "inputs")
    experiment = _mapping(document, "experiment")
    spatial = _mapping(document, "spatial")
    scan = _mapping(document, "scan")
    outputs = _mapping(document, "outputs")
    figures = _mapping(document, "figures")

    output_defaults = OutputSettings()
    output_settings = OutputSettings(
        save_csv=bool(outputs.get("save_csv", output_defaults.save_csv)),
        save_preview_png=bool(
            outputs.get("save_preview_png", output_defaults.save_preview_png)
        ),
        save_terrain_png=bool(
            outputs.get("save_terrain_png", output_defaults.save_terrain_png)
        ),
        save_terrain_eps=bool(
            outputs.get("save_terrain_eps", output_defaults.save_terrain_eps)
        ),
        save_terrain_html=bool(
            outputs.get("save_terrain_html", output_defaults.save_terrain_html)
        ),
    )
    figure_defaults = FigureSettings()
    title = figures.get("title", figure_defaults.title)
    figure_settings = FigureSettings(
        preset=str(figures.get("preset", figure_defaults.preset)),
        colormap=str(figures.get("colormap", figure_defaults.colormap)),
        dpi=int(figures.get("dpi", figure_defaults.dpi)),
        azimuth_deg=float(figures.get("azimuth_deg", figure_defaults.azimuth_deg)),
        elevation_deg=float(
            figures.get("elevation_deg", figure_defaults.elevation_deg)
        ),
        vertical_exaggeration=float(
            figures.get(
                "vertical_exaggeration",
                figure_defaults.vertical_exaggeration,
            )
        ),
        station_color=str(
            figures.get("station_color", figure_defaults.station_color)
        ),
        station_marker_size=float(
            figures.get(
                "station_marker_size",
                figure_defaults.station_marker_size,
            )
        ),
        title=None if title is None else str(title),
    )

    profile = ExperimentProfile(
        schema_version=2,
        profile_id=str(_required(profile_section, "id", "profile")),
        display_name=str(_required(profile_section, "display_name", "profile")),
        scenario_id=str(_required(profile_section, "scenario_id", "profile")),
        points_dataset_id=str(
            _required(inputs, "points_dataset_id", "inputs")
        ),
        random_seed=int(
            experiment.get("random_seed", DEFAULT_PROFILE_VALUES["random_seed"])
        ),
        target_crs=str(_required(spatial, "target_crs", "spatial")),
        rect_size=int(_required(spatial, "rectangle_size_m", "spatial")),
        target_count=int(
            _required(spatial, "target_base_station_count", "spatial")
        ),
        tolerance=int(_required(spatial, "count_tolerance", "spatial")),
        scan_mode=str(scan.get("mode", DEFAULT_PROFILE_VALUES["scan_mode"])),
        strategy=str(_required(scan, "strategy", "scan")),
        scan_step=int(_required(scan, "step_m", "scan")),
        max_rects=int(_required(scan, "max_rectangles", "scan")),
        min_spacing=int(
            _required(scan, "minimum_center_spacing_m", "scan")
        ),
        output_root=_resolve_path(outputs.get("root", "results"), root),
        outputs=output_settings,
        figure=figure_settings,
        source_path=source_path,
    )
    return _validate_profile(profile)
