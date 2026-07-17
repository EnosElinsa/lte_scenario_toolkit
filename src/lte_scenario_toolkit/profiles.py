"""Versioned experiment-profile models and loaders."""

from __future__ import annotations

import math
import os
import re
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from .data_catalog import (
    CatalogError,
    ConcurrentCatalogUpdateError,
    DataCatalog,
    catalog_transaction_lock,
    load_data_catalog,
    save_data_catalog,
)

PROFILE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
LEGACY_SECTION_KEYS = frozenset({"experiment", "inputs", "spatial", "scan", "outputs"})

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
_CATALOG_ROLLBACK_CONFIRMED = "_lte_catalog_rollback_confirmed"


class ConcurrentProfileUpdateError(RuntimeError):
    """Raised when a stored profile changes during a mutating transaction."""


def _mark_catalog_rollback(error: Exception, *, confirmed: bool) -> None:
    setattr(error, _CATALOG_ROLLBACK_CONFIRMED, confirmed)


def _catalog_rollback_confirmed(error: Exception) -> bool:
    return getattr(error, _CATALOG_ROLLBACK_CONFIRMED, False) is True


def _ensure_catalog_unchanged(catalog: DataCatalog) -> None:
    try:
        current_mtime_ns = catalog.path.stat().st_mtime_ns
    except FileNotFoundError as exc:
        raise ConcurrentCatalogUpdateError(
            f"Catalog {catalog.path} changed since it was loaded"
        ) from exc
    if current_mtime_ns != catalog.loaded_mtime_ns:
        raise ConcurrentCatalogUpdateError(
            f"Catalog {catalog.path} changed since it was loaded"
        )


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


@dataclass(frozen=True, slots=True)
class LegacyProfileValues(Mapping[str, Any]):
    """Immutable effective legacy values plus their discovered source revision."""

    _items: tuple[tuple[str, Any], ...]
    source_path: Path
    source_sha256: str
    catalog_owner_scenario_id: str | None = None

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def source_revision(self) -> str:
        return f"sha256:{self.source_sha256}"


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
    legacy_source: LegacyProfileValues | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def is_legacy_preview(self) -> bool:
        return self.legacy_source is not None

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
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{path} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_profile(profile: ExperimentProfile) -> ExperimentProfile:
    if type(profile.schema_version) is not int or profile.schema_version != 2:
        raise ValueError("schema_version must be the integer 2")
    for path, value in (
        ("profile.id", profile.profile_id),
        ("profile.display_name", profile.display_name),
        ("profile.scenario_id", profile.scenario_id),
        ("inputs.points_dataset_id", profile.points_dataset_id),
        ("spatial.target_crs", profile.target_crs),
        ("scan.mode", profile.scan_mode),
        ("scan.strategy", profile.strategy),
    ):
        if type(value) is not str or not value.strip():
            raise ValueError(f"{path} must be a non-empty string")
    if (
        PROFILE_ID_PATTERN.fullmatch(profile.profile_id) is None
        or profile.profile_id in WINDOWS_RESERVED
    ):
        raise ValueError(
            "profile.id must be a safe lowercase slug and not a Windows device name"
        )
    if PROFILE_ID_PATTERN.fullmatch(profile.scenario_id) is None:
        raise ValueError("profile.scenario_id must be a safe lowercase slug")
    if profile.scan_mode not in {"fast", "complete"}:
        raise ValueError("scan.mode must be one of: complete, fast")
    if profile.strategy not in {"sequential", "uniform"}:
        raise ValueError("scan.strategy must be one of: sequential, uniform")
    for path, value in (
        ("experiment.random_seed", profile.random_seed),
        ("spatial.rectangle_size_m", profile.rect_size),
        ("spatial.target_base_station_count", profile.target_count),
        ("spatial.count_tolerance", profile.tolerance),
        ("scan.step_m", profile.scan_step),
        ("scan.max_rectangles", profile.max_rects),
        ("scan.minimum_center_spacing_m", profile.min_spacing),
    ):
        if type(value) is not int:
            raise ValueError(f"{path} must be an integer")
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
    if not isinstance(profile.output_root, (str, os.PathLike)):
        raise ValueError("outputs.root must be a path")
    if not isinstance(profile.outputs, OutputSettings):
        raise ValueError("outputs must be OutputSettings")
    for key, value in vars(profile.outputs).items():
        if type(value) is not bool:
            raise ValueError(f"outputs.{key} must be a boolean")
    if not isinstance(profile.figure, FigureSettings):
        raise ValueError("figures must be FigureSettings")
    for key in ("preset", "colormap", "station_color"):
        value = getattr(profile.figure, key)
        if type(value) is not str:
            raise ValueError(f"figures.{key} must be a string")
    if profile.figure.title is not None and type(profile.figure.title) is not str:
        raise ValueError("figures.title must be null or a string")
    if type(profile.figure.dpi) is not int:
        raise ValueError("figures.dpi must be an integer")
    for key in (
        "azimuth_deg",
        "elevation_deg",
        "vertical_exaggeration",
        "station_marker_size",
    ):
        value = getattr(profile.figure, key)
        if type(value) not in {int, float}:
            raise ValueError(f"figures.{key} must be a finite number")
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(f"figures.{key} must be a finite number") from exc
        if not finite:
            raise ValueError(f"figures.{key} must be a finite number")
    if profile.figure.dpi <= 0:
        raise ValueError("figures.dpi must be greater than zero")
    if profile.figure.vertical_exaggeration <= 0:
        raise ValueError("figures.vertical_exaggeration must be greater than zero")
    if profile.legacy_source is not None:
        if not isinstance(profile.legacy_source, LegacyProfileValues):
            raise ValueError("legacy_source must be LegacyProfileValues")
        if profile.source_path is None or (
            Path(profile.source_path).resolve() != profile.legacy_source.source_path
        ):
            raise ValueError("legacy preview source path must match its provenance")
        if re.fullmatch(r"[0-9a-f]{64}", profile.legacy_source.source_sha256) is None:
            raise ValueError("legacy source SHA256 must be a lowercase hexadecimal digest")
    return profile


def validate_profile(profile: ExperimentProfile) -> ExperimentProfile:
    """Validate an in-memory profile, including directly constructed values."""

    if not isinstance(profile, ExperimentProfile):
        raise ValueError("profile must be an ExperimentProfile")
    return _validate_profile(profile)


def _profile_repository(path: Path) -> Path:
    for parent in path.parents:
        if parent.name.casefold() == "configs":
            return parent.parent.resolve()
    return path.parent.resolve()


def _serialized_output_root(profile: ExperimentProfile, path: Path) -> str:
    output_root = Path(profile.output_root)
    if not output_root.is_absolute():
        return output_root.as_posix()
    repository = _profile_repository(path)
    try:
        return output_root.resolve().relative_to(repository).as_posix()
    except ValueError:
        return str(output_root.resolve())


def _profile_document(profile: ExperimentProfile, path: Path) -> dict[str, Any]:
    _validate_profile(profile)
    if profile.is_legacy_preview:
        raise ValueError("Legacy profile previews are read-only until explicit Save")
    return {
        "schema_version": 2,
        "profile": {
            "id": profile.profile_id,
            "display_name": profile.display_name,
            "scenario_id": profile.scenario_id,
        },
        "inputs": {
            "points_dataset_id": profile.points_dataset_id,
        },
        "experiment": {
            "random_seed": profile.random_seed,
        },
        "spatial": {
            "target_crs": profile.target_crs,
            "rectangle_size_m": profile.rect_size,
            "target_base_station_count": profile.target_count,
            "count_tolerance": profile.tolerance,
        },
        "scan": {
            "mode": profile.scan_mode,
            "strategy": profile.strategy,
            "step_m": profile.scan_step,
            "max_rectangles": profile.max_rects,
            "minimum_center_spacing_m": profile.min_spacing,
        },
        "outputs": {
            "root": _serialized_output_root(profile, path),
            "save_csv": profile.outputs.save_csv,
            "save_preview_png": profile.outputs.save_preview_png,
            "save_terrain_png": profile.outputs.save_terrain_png,
            "save_terrain_eps": profile.outputs.save_terrain_eps,
            "save_terrain_html": profile.outputs.save_terrain_html,
        },
        "figures": {
            "preset": profile.figure.preset,
            "colormap": profile.figure.colormap,
            "dpi": profile.figure.dpi,
            "azimuth_deg": profile.figure.azimuth_deg,
            "elevation_deg": profile.figure.elevation_deg,
            "vertical_exaggeration": profile.figure.vertical_exaggeration,
            "station_color": profile.figure.station_color,
            "station_marker_size": profile.figure.station_marker_size,
            "title": profile.figure.title,
        },
    }


def _atomic_write_profile_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_restore_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".rollback",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _profile_text(profile: ExperimentProfile, path: str | Path) -> str:
    """Return the deterministic UTF-8 text written for one profile."""

    destination = Path(path).resolve()
    document = _profile_document(profile, destination)
    text = yaml.safe_dump(
        document,
        sort_keys=False,
        allow_unicode=True,
    )
    if not text.endswith("\n"):
        text += "\n"
    return text


def dump_profile(profile: ExperimentProfile, path: str | Path) -> Path:
    """Validate and atomically write a deterministic schema-version-2 profile."""

    destination = Path(path).resolve()
    text = _profile_text(profile, destination)
    _atomic_write_profile_text(destination, text)
    return destination


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
        else _profile_repository(source_path)
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


def load_legacy_profile(
    path: str | Path,
    repo_root: str | Path,
) -> LegacyProfileValues:
    """Load one legacy YAML as an immutable mapping of effective values."""

    source_path = Path(path).resolve()
    source_bytes = source_path.read_bytes()
    document = yaml.safe_load(source_bytes.decode("utf-8")) or {}
    if not isinstance(document, Mapping):
        raise ValueError("The legacy experiment configuration must be a YAML mapping")
    if "schema_version" in document:
        raise ValueError("load_legacy_profile only accepts legacy YAML without schema_version")

    from .config import load_experiment_config

    effective = load_experiment_config(
        source_path,
        repo_root=Path(repo_root).resolve(),
    )
    try:
        current_bytes = source_path.read_bytes()
    except FileNotFoundError as exc:
        raise ConcurrentProfileUpdateError(
            f"Legacy profile changed while it was loaded: {source_path}"
        ) from exc
    if current_bytes != source_bytes:
        raise ConcurrentProfileUpdateError(
            f"Legacy profile changed while it was loaded: {source_path}"
        )
    return LegacyProfileValues(
        tuple(effective.items()),
        source_path,
        sha256(source_bytes).hexdigest(),
    )


def _legacy_profile_from_values(
    legacy: Mapping[str, Any],
    *,
    profile_id: str,
    scenario_id: str,
    points_dataset_id: str,
    display_name: str | None = None,
    legacy_source: LegacyProfileValues | None = None,
) -> ExperimentProfile:
    if not isinstance(legacy, Mapping):
        raise ValueError("legacy profile values must be a mapping")

    def required(key: str) -> Any:
        try:
            return legacy[key]
        except KeyError as exc:
            raise ValueError(f"Missing required legacy profile value: {key}") from exc

    output_defaults = OutputSettings()
    source_path = legacy.get("config_path")
    profile = ExperimentProfile(
        schema_version=2,
        profile_id=profile_id,
        display_name=(
            display_name
            if display_name is not None
            else legacy.get("experiment_name", profile_id)
        ),
        scenario_id=scenario_id,
        points_dataset_id=points_dataset_id,
        random_seed=legacy.get(
            "random_seed",
            DEFAULT_PROFILE_VALUES["random_seed"],
        ),
        target_crs=required("target_crs"),
        rect_size=required("rect_size"),
        target_count=required("target_count"),
        tolerance=required("tolerance"),
        scan_mode=legacy.get("scan_mode", DEFAULT_PROFILE_VALUES["scan_mode"]),
        strategy=required("strategy"),
        scan_step=required("scan_step"),
        max_rects=required("max_rects"),
        min_spacing=required("min_spacing"),
        output_root=Path(required("output_root")),
        outputs=OutputSettings(
            save_csv=legacy.get("save_csv", output_defaults.save_csv),
            save_preview_png=legacy.get(
                "save_preview_png",
                output_defaults.save_preview_png,
            ),
            save_terrain_png=legacy.get(
                "save_terrain_png",
                output_defaults.save_terrain_png,
            ),
            save_terrain_eps=legacy.get(
                "save_terrain_eps",
                output_defaults.save_terrain_eps,
            ),
            save_terrain_html=legacy.get(
                "save_terrain_html",
                output_defaults.save_terrain_html,
            ),
        ),
        source_path=Path(source_path).resolve() if source_path is not None else None,
        legacy_source=legacy_source,
    )
    return _validate_profile(profile)


def convert_legacy_profile(
    legacy: Mapping[str, Any],
    *,
    profile_id: str,
    scenario_id: str,
    points_dataset_id: str,
    display_name: str | None = None,
) -> ExperimentProfile:
    """Explicitly convert immutable legacy values into a writable v2 profile."""

    return _legacy_profile_from_values(
        legacy,
        profile_id=profile_id,
        scenario_id=scenario_id,
        points_dataset_id=points_dataset_id,
        display_name=display_name,
    )


class ProfileStore:
    """Discover and mutate repository profiles under one shared catalog lock."""

    def __init__(
        self,
        repo_root: str | Path,
        catalog_path: str | Path,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        raw_catalog_path = Path(catalog_path)
        if not raw_catalog_path.is_absolute():
            raw_catalog_path = self.repo_root / raw_catalog_path
        self.catalog_path = raw_catalog_path.resolve()
        self.configs_root = self.repo_root / "configs"

    def _resolved_configs_root(self) -> Path:
        resolved = self.configs_root.resolve(strict=False)
        try:
            resolved.relative_to(self.repo_root)
        except ValueError as exc:
            raise ValueError(
                f"Profile configs directory is outside repository: {self.configs_root}"
            ) from exc
        return resolved

    def _profile_path(
        self,
        value: str | Path,
        *,
        must_exist: bool,
    ) -> Path:
        raw = Path(value)
        if raw.is_absolute():
            candidate = raw
        elif raw.parts and raw.parts[0].casefold() == "configs":
            candidate = self.repo_root / raw
        else:
            candidate = self.configs_root / raw
        configs_root = self._resolved_configs_root()
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"Profile path must remain inside {self.configs_root}: {value}"
            ) from exc
        if candidate.is_symlink():
            raise ValueError(f"Profile path must not be a symlink: {candidate}")
        if must_exist and not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved

    def _destination(self, profile: ExperimentProfile) -> Path:
        _validate_profile(profile)
        return self._profile_path(
            Path(profile.scenario_id) / f"{profile.profile_id}.yaml",
            must_exist=False,
        )

    def _prepare_legacy_migration(
        self,
        profile: ExperimentProfile,
        catalog: DataCatalog,
    ) -> tuple[ExperimentProfile, Path, bytes, bool] | None:
        legacy = profile.legacy_source
        if legacy is None:
            return None
        source = self._profile_path(legacy.source_path, must_exist=False)
        owners = self._catalog_owners_for_source(catalog, source)
        expected_owner = legacy.catalog_owner_scenario_id
        if expected_owner is None and owners:
            raise ConcurrentProfileUpdateError(
                f"Legacy profile ownership changed since discovery: {source}"
            )
        if expected_owner is not None and owners != (expected_owner,):
            raise ConcurrentProfileUpdateError(
                f"Legacy profile ownership changed since discovery: {source}"
            )
        try:
            matched_scenario_id, matched_points_id = self._match_legacy_catalog(
                legacy,
                catalog,
                owners,
            )
        except ValueError as exc:
            raise ConcurrentProfileUpdateError(
                f"Legacy profile catalog mapping changed since discovery: {source}"
            ) from exc
        if (
            matched_scenario_id != profile.scenario_id
            or matched_points_id != profile.points_dataset_id
        ):
            raise ConcurrentProfileUpdateError(
                f"Legacy profile catalog mapping changed since discovery: {source}"
            )
        try:
            content = source.read_bytes()
        except FileNotFoundError as exc:
            raise ConcurrentProfileUpdateError(
                f"Legacy profile changed since discovery: {source}"
            ) from exc
        if sha256(content).hexdigest() != legacy.source_sha256:
            raise ConcurrentProfileUpdateError(
                f"Legacy profile changed since discovery: {source}"
            )
        converted = convert_legacy_profile(
            legacy,
            profile_id=profile.profile_id,
            scenario_id=profile.scenario_id,
            points_dataset_id=profile.points_dataset_id,
            display_name=profile.display_name,
        )
        return converted, source, content, expected_owner is not None

    def _load_catalog(self) -> DataCatalog:
        return load_data_catalog(self.catalog_path, repo_root=self.repo_root)

    def _save_catalog(
        self,
        catalog: DataCatalog,
        document: dict[str, Any],
    ) -> DataCatalog:
        original = catalog.path.read_bytes()
        expected = yaml.safe_dump(
            document,
            sort_keys=False,
            allow_unicode=True,
        ).encode("utf-8")
        expected_with_native_newlines = expected.replace(
            b"\n",
            os.linesep.encode("ascii"),
        )
        try:
            return save_data_catalog(catalog, document)
        except ConcurrentCatalogUpdateError as exc:
            _mark_catalog_rollback(exc, confirmed=False)
            exc.add_note(
                "Catalog rollback was not attempted because concurrent catalog "
                "state is externally owned"
            )
            raise
        except Exception as exc:
            try:
                try:
                    current = catalog.path.read_bytes()
                except FileNotFoundError:
                    _atomic_restore_bytes(catalog.path, original)
                else:
                    if current in {expected, expected_with_native_newlines}:
                        _atomic_restore_bytes(catalog.path, original)
                    elif current != original:
                        exc.add_note(
                            f"Catalog rollback skipped for {catalog.path}: "
                            "catalog changed after this operation wrote it"
                        )
            except Exception as rollback_error:
                exc.add_note(
                    f"Catalog rollback also failed for {catalog.path}: {rollback_error}"
                )
            try:
                rollback_confirmed = catalog.path.read_bytes() == original
            except Exception as verification_error:
                rollback_confirmed = False
                exc.add_note(
                    f"Catalog rollback verification failed for {catalog.path}: "
                    f"{verification_error}"
                )
            _mark_catalog_rollback(exc, confirmed=rollback_confirmed)
            raise

    @staticmethod
    def _validate_profile_catalog(
        profile: ExperimentProfile,
        catalog: DataCatalog,
    ) -> None:
        _validate_profile(profile)
        catalog.scenario(profile.scenario_id)
        dataset = catalog.dataset(profile.points_dataset_id)
        if dataset["role"] != "points":
            raise ValueError(
                f"Profile points dataset {profile.points_dataset_id!r} "
                "must have role 'points'"
            )

    def _load_stored_profile(
        self,
        path: str | Path,
        catalog: DataCatalog,
    ) -> tuple[Path, ExperimentProfile, bytes]:
        resolved = self._profile_path(path, must_exist=True)
        original = resolved.read_bytes()
        profile = load_profile(resolved, repo_root=self.repo_root)
        try:
            current = resolved.read_bytes()
        except FileNotFoundError as exc:
            raise ConcurrentProfileUpdateError(
                f"Profile changed while it was loaded: {resolved}"
            ) from exc
        if current != original:
            raise ConcurrentProfileUpdateError(
                f"Profile changed while it was loaded: {resolved}"
            )
        self._validate_profile_catalog(profile, catalog)
        return resolved, profile, original

    @staticmethod
    def _unlink_profile_if_unchanged(
        path: Path,
        expected: bytes,
        catalog: DataCatalog,
    ) -> None:
        _ensure_catalog_unchanged(catalog)
        try:
            current = path.read_bytes()
        except FileNotFoundError as exc:
            raise ConcurrentProfileUpdateError(
                f"Profile changed before it could be removed: {path}"
            ) from exc
        if current != expected:
            raise ConcurrentProfileUpdateError(
                f"Profile changed before it could be removed: {path}"
            )
        _ensure_catalog_unchanged(catalog)
        try:
            path.unlink()
        except FileNotFoundError as exc:
            raise ConcurrentProfileUpdateError(
                f"Profile changed before it could be removed: {path}"
            ) from exc

    def _catalog_profile_path(
        self,
        catalog: DataCatalog,
        scenario_id: str,
    ) -> Path | None:
        config_path = catalog.scenario(scenario_id)["config_path"]
        if config_path is None:
            return None
        return self._profile_path(catalog.resolve(config_path), must_exist=False)

    def _catalog_owners_for_source(
        self,
        catalog: DataCatalog,
        source: Path,
    ) -> tuple[str, ...]:
        return tuple(
            scenario_id
            for scenario_id in catalog.scenarios_by_id
            if self._catalog_profile_path(catalog, scenario_id) == source
        )

    @staticmethod
    def _legacy_input_paths(legacy: LegacyProfileValues) -> tuple[Path, Path, Path]:
        from .spatial import resolve_io_paths

        try:
            paths = resolve_io_paths(dict(legacy), create_output=False)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"Valid legacy profile {legacy.source_path} could not resolve its "
                f"input paths: {exc}"
            ) from exc
        return tuple(
            Path(paths[key]).resolve(strict=False)
            for key in ("points_shp", "boundary_shp", "dem_path")
        )

    def _match_legacy_catalog(
        self,
        legacy: LegacyProfileValues,
        catalog: DataCatalog,
        owners: tuple[str, ...],
    ) -> tuple[str, str]:
        points_path, boundary_path, dem_path = self._legacy_input_paths(legacy)
        point_ids = tuple(
            dataset_id
            for dataset_id, dataset in catalog.datasets_by_id.items()
            if dataset["role"] == "points"
            and catalog.resolve(dataset["entrypoint"]).resolve(strict=False)
            == points_path
        )
        if not point_ids:
            raise ValueError(
                f"Valid legacy profile {legacy.source_path} points path {points_path} "
                "does not match any catalog points dataset"
            )
        if len(point_ids) != 1:
            raise ValueError(
                f"Valid legacy profile {legacy.source_path} points path matches "
                f"multiple catalog datasets: {', '.join(point_ids)}"
            )

        matching_scenarios: list[str] = []
        for scenario_id, scenario in catalog.scenarios_by_id.items():
            boundary = catalog.dataset(scenario["boundary_dataset_id"])
            if (
                catalog.resolve(boundary["entrypoint"]).resolve(strict=False)
                != boundary_path
            ):
                continue
            dem_dataset_id = scenario["dem_dataset_id"]
            if dem_dataset_id is None:
                continue
            dem = catalog.dataset(dem_dataset_id)
            if catalog.resolve(dem["entrypoint"]).resolve(strict=False) == dem_path:
                matching_scenarios.append(scenario_id)

        if owners:
            if len(owners) != 1:
                raise ValueError(
                    f"Legacy profile {legacy.source_path} is referenced by multiple "
                    "scenarios"
                )
            owner = owners[0]
            if owner not in matching_scenarios:
                raise ValueError(
                    f"catalog-owned legacy profile {legacy.source_path} does not "
                    f"match the registered paths for scenario {owner!r}"
                )
            return owner, point_ids[0]

        if not matching_scenarios:
            raise ValueError(
                f"Valid legacy profile {legacy.source_path} boundary and DEM paths "
                "do not match any catalog scenario"
            )
        if len(matching_scenarios) != 1:
            superseding = [
                candidate
                for candidate in matching_scenarios
                if self._catalog_v2_supersedes_legacy(catalog, candidate, legacy)
            ]
            if len(superseding) == 1:
                return superseding[0], point_ids[0]
            raise ValueError(
                f"Valid legacy profile {legacy.source_path} matches multiple scenarios: "
                f"{', '.join(matching_scenarios)}"
            )
        return matching_scenarios[0], point_ids[0]

    def _catalog_v2_supersedes_legacy(
        self,
        catalog: DataCatalog,
        scenario_id: str,
        legacy: LegacyProfileValues,
    ) -> bool:
        profile_path = self._catalog_profile_path(catalog, scenario_id)
        if (
            profile_path is None
            or profile_path == legacy.source_path
            or not profile_path.is_file()
        ):
            return False
        profile = load_profile(profile_path, repo_root=self.repo_root)
        return (
            profile.scenario_id == scenario_id
            and profile.profile_id == legacy.source_path.stem
        )

    def _relative_profile_path(self, path: Path) -> str:
        return path.relative_to(self.repo_root).as_posix()

    @staticmethod
    def _default_document(
        catalog: DataCatalog,
        scenario_id: str,
        relative_path: str,
    ) -> dict[str, Any]:
        document = deepcopy(catalog.document)
        for scenario in document["scenarios"]:
            if scenario["scenario_id"] == scenario_id:
                scenario["config_path"] = relative_path
                return document
        raise CatalogError(f"Unknown scenario ID: {scenario_id}")

    @staticmethod
    def _restore_profile_after_error(
        path: Path,
        original: bytes | None,
        expected: bytes,
        operation_error: Exception,
    ) -> None:
        try:
            try:
                current = path.read_bytes()
            except FileNotFoundError:
                if original is not None:
                    _atomic_restore_bytes(path, original)
                return
            if current != expected:
                operation_error.add_note(
                    f"Profile rollback skipped for {path}: target changed "
                    "after this operation wrote it"
                )
                return
            if original is not None:
                _atomic_restore_bytes(path, original)
                return
            path.unlink()
        except Exception as rollback_error:
            operation_error.add_note(
                f"Profile rollback also failed for {path}: {rollback_error}"
            )

    @staticmethod
    def _ensure_unique_profile_identities(
        profiles: list[ExperimentProfile],
    ) -> None:
        seen: dict[tuple[str, str], ExperimentProfile] = {}
        for profile in profiles:
            identity = (profile.scenario_id, profile.profile_id)
            previous = seen.get(identity)
            if previous is None:
                seen[identity] = profile
                continue
            raise ValueError(
                "duplicate profile identity "
                f"({profile.scenario_id!r}, {profile.profile_id!r}): "
                f"{previous.source_path} and {profile.source_path}"
            )

    def discover(self, scenario_id: str | None = None) -> list[ExperimentProfile]:
        """Recursively load profiles in stable repository-relative order."""

        configs_root = self._resolved_configs_root()
        catalog = self._load_catalog() if self.catalog_path.is_file() else None
        catalog_owners: dict[Path, list[str]] = {}
        if catalog is not None:
            for configured_scenario_id, scenario in catalog.scenarios_by_id.items():
                config_path = scenario["config_path"]
                if config_path is None:
                    continue
                resolved = self._profile_path(
                    catalog.resolve(config_path),
                    must_exist=False,
                )
                catalog_owners.setdefault(resolved, []).append(
                    configured_scenario_id
                )
        if not configs_root.is_dir():
            return []
        paths = sorted(
            (
                path
                for path in configs_root.rglob("*")
                if path.is_file() and path.suffix.casefold() in {".yaml", ".yml"}
            ),
            key=lambda path: path.relative_to(configs_root).as_posix(),
        )
        if catalog is None:
            profiles = [
                load_profile(
                    self._profile_path(path, must_exist=True),
                    repo_root=self.repo_root,
                )
                for path in paths
            ]
            self._ensure_unique_profile_identities(profiles)
            if scenario_id is None:
                return profiles
            return [
                profile for profile in profiles if profile.scenario_id == scenario_id
            ]
        profiles: list[ExperimentProfile] = []
        for path in paths:
            resolved = self._profile_path(path, must_exist=True)
            document = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            owners = tuple(catalog_owners.get(resolved, []))
            if isinstance(document, Mapping) and "schema_version" in document:
                profiles.append(load_profile(resolved, repo_root=self.repo_root))
                continue
            if not isinstance(document, Mapping):
                if not owners:
                    continue
                raise ValueError(
                    "The catalog-owned legacy experiment configuration must be a "
                    "YAML mapping"
                )
            try:
                legacy = load_legacy_profile(resolved, self.repo_root)
            except ValueError as exc:
                if not owners and not LEGACY_SECTION_KEYS.intersection(document):
                    continue
                raise ValueError(f"Invalid legacy profile {resolved}: {exc}") from exc
            matched_scenario_id, matched_points_id = self._match_legacy_catalog(
                legacy,
                catalog,
                owners,
            )
            if not owners and self._catalog_v2_supersedes_legacy(
                catalog,
                matched_scenario_id,
                legacy,
            ):
                continue
            if owners:
                legacy = replace(
                    legacy,
                    catalog_owner_scenario_id=matched_scenario_id,
                )
            profiles.append(
                _legacy_profile_from_values(
                    legacy,
                    profile_id=resolved.stem,
                    scenario_id=matched_scenario_id,
                    points_dataset_id=matched_points_id,
                    legacy_source=legacy,
                )
            )
        self._ensure_unique_profile_identities(profiles)
        if scenario_id is None:
            return profiles
        return [profile for profile in profiles if profile.scenario_id == scenario_id]

    def save(
        self,
        profile: ExperimentProfile,
        *,
        overwrite: bool = False,
        set_default: bool = False,
    ) -> Path:
        """Save a profile and optionally update its scenario default atomically."""

        with catalog_transaction_lock(self.repo_root):
            catalog = self._load_catalog()
            self._validate_profile_catalog(profile, catalog)
            legacy_migration = self._prepare_legacy_migration(profile, catalog)
            persisted_profile = (
                legacy_migration[0] if legacy_migration is not None else profile
            )
            self._validate_profile_catalog(persisted_profile, catalog)
            destination = (
                legacy_migration[1]
                if legacy_migration is not None and not legacy_migration[3]
                else self._destination(persisted_profile)
            )
            destination_exists = os.path.lexists(destination)
            if destination_exists and not destination.is_file():
                raise FileExistsError(destination)
            if destination_exists and not overwrite:
                raise FileExistsError(destination)
            original = destination.read_bytes() if destination_exists else None
            expected = _profile_text(persisted_profile, destination).encode("utf-8")
            try:
                dump_profile(persisted_profile, destination)
            except Exception as exc:
                self._restore_profile_after_error(
                    destination,
                    original,
                    expected,
                    exc,
                )
                raise
            update_catalog = set_default
            if legacy_migration is not None:
                try:
                    _, source, source_bytes, is_catalog_default = legacy_migration
                    try:
                        current_source = source.read_bytes()
                    except FileNotFoundError as exc:
                        raise ConcurrentProfileUpdateError(
                            f"Legacy profile changed during migration: {source}"
                        ) from exc
                    expected_source = expected if source == destination else source_bytes
                    if current_source != expected_source:
                        raise ConcurrentProfileUpdateError(
                            f"Legacy profile changed during migration: {source}"
                        )
                    update_catalog = set_default or (
                        is_catalog_default and source != destination
                    )
                except Exception as exc:
                    self._restore_profile_after_error(
                        destination,
                        original,
                        expected,
                        exc,
                    )
                    raise
            if not update_catalog:
                return destination
            document = self._default_document(
                catalog,
                persisted_profile.scenario_id,
                self._relative_profile_path(destination),
            )
            try:
                self._save_catalog(catalog, document)
            except Exception as exc:
                if _catalog_rollback_confirmed(exc):
                    self._restore_profile_after_error(
                        destination,
                        original,
                        expected,
                        exc,
                    )
                else:
                    exc.add_note(
                        f"Profile rollback skipped for {destination}: catalog "
                        "rollback was not confirmed, so the target was retained"
                    )
                raise
            return destination

    def copy(
        self,
        source: str | Path,
        profile_id: str,
        display_name: str,
    ) -> Path:
        """Create a named copy without changing the source or catalog default."""

        with catalog_transaction_lock(self.repo_root):
            catalog = self._load_catalog()
            _, source_profile, _ = self._load_stored_profile(source, catalog)
            copied = replace(
                source_profile,
                profile_id=profile_id,
                display_name=display_name,
                source_path=None,
            )
            self._validate_profile_catalog(copied, catalog)
            destination = self._destination(copied)
            if os.path.lexists(destination):
                raise FileExistsError(destination)
            expected = _profile_text(copied, destination).encode("utf-8")
            try:
                dump_profile(copied, destination)
            except Exception as exc:
                self._restore_profile_after_error(
                    destination,
                    None,
                    expected,
                    exc,
                )
                raise
            return destination

    def rename(
        self,
        source: str | Path,
        profile_id: str,
        display_name: str,
    ) -> Path:
        """Write a renamed profile before removing its source."""

        with catalog_transaction_lock(self.repo_root):
            catalog = self._load_catalog()
            source_path, source_profile, source_bytes = self._load_stored_profile(
                source,
                catalog,
            )
            renamed = replace(
                source_profile,
                profile_id=profile_id,
                display_name=display_name,
                source_path=None,
            )
            self._validate_profile_catalog(renamed, catalog)
            destination = self._destination(renamed)
            if destination == source_path or os.path.lexists(destination):
                raise FileExistsError(destination)
            default_path = self._catalog_profile_path(catalog, source_profile.scenario_id)
            is_default = default_path == source_path
            expected = _profile_text(renamed, destination).encode("utf-8")
            try:
                dump_profile(renamed, destination)
            except Exception as exc:
                self._restore_profile_after_error(
                    destination,
                    None,
                    expected,
                    exc,
                )
                raise
            saved_catalog: DataCatalog | None = None
            try:
                unlink_catalog = catalog
                if is_default:
                    document = self._default_document(
                        catalog,
                        source_profile.scenario_id,
                        self._relative_profile_path(destination),
                    )
                    saved_catalog = self._save_catalog(catalog, document)
                    if (
                        self._catalog_profile_path(
                            saved_catalog,
                            source_profile.scenario_id,
                        )
                        != destination
                    ):
                        raise ConcurrentCatalogUpdateError(
                            f"Catalog {saved_catalog.path} changed since it was loaded"
                        )
                    unlink_catalog = saved_catalog
                self._unlink_profile_if_unchanged(
                    source_path,
                    source_bytes,
                    unlink_catalog,
                )
            except Exception as exc:
                catalog_restored = not is_default
                if is_default and saved_catalog is None:
                    catalog_restored = _catalog_rollback_confirmed(exc)
                if saved_catalog is not None:
                    try:
                        save_data_catalog(saved_catalog, deepcopy(catalog.document))
                        catalog_restored = True
                    except Exception as rollback_error:
                        exc.add_note(
                            f"Catalog rollback also failed during rename: {rollback_error}"
                        )
                if catalog_restored:
                    self._restore_profile_after_error(
                        destination,
                        None,
                        expected,
                        exc,
                    )
                else:
                    exc.add_note(
                        f"Profile rollback skipped for {destination}: catalog "
                        "rollback was not confirmed, so the target was retained"
                    )
                raise
            return destination

    def set_default(self, scenario_id: str, profile_path: str | Path) -> Path:
        """Set an existing same-scenario profile as the catalog default."""

        with catalog_transaction_lock(self.repo_root):
            catalog = self._load_catalog()
            catalog.scenario(scenario_id)
            resolved, profile, _ = self._load_stored_profile(profile_path, catalog)
            if profile.scenario_id != scenario_id:
                raise ValueError(
                    f"Profile scenario {profile.scenario_id!r} does not match {scenario_id!r}"
                )
            document = self._default_document(
                catalog,
                scenario_id,
                self._relative_profile_path(resolved),
            )
            self._save_catalog(catalog, document)
            return resolved

    def delete(
        self,
        profile_path: str | Path,
        *,
        replacement_default: str | Path | None = None,
    ) -> None:
        """Delete a profile, switching a scenario default first when required."""

        with catalog_transaction_lock(self.repo_root):
            catalog = self._load_catalog()
            source, profile, source_bytes = self._load_stored_profile(
                profile_path,
                catalog,
            )
            default_path = self._catalog_profile_path(catalog, profile.scenario_id)
            if default_path != source:
                self._unlink_profile_if_unchanged(source, source_bytes, catalog)
                return
            if replacement_default is None:
                raise ValueError(
                    "Cannot delete the default profile without a replacement default"
                )
            replacement_path, replacement, _ = self._load_stored_profile(
                replacement_default,
                catalog,
            )
            if replacement_path == source:
                raise ValueError("Replacement default must be a different profile")
            if replacement.scenario_id != profile.scenario_id:
                raise ValueError(
                    "Replacement default must belong to the same scenario"
                )
            document = self._default_document(
                catalog,
                profile.scenario_id,
                self._relative_profile_path(replacement_path),
            )
            saved_catalog = self._save_catalog(catalog, document)
            try:
                if (
                    self._catalog_profile_path(saved_catalog, profile.scenario_id)
                    != replacement_path
                ):
                    raise ConcurrentCatalogUpdateError(
                        f"Catalog {saved_catalog.path} changed since it was loaded"
                    )
                self._unlink_profile_if_unchanged(
                    source,
                    source_bytes,
                    saved_catalog,
                )
            except Exception as exc:
                try:
                    save_data_catalog(saved_catalog, deepcopy(catalog.document))
                except Exception as rollback_error:
                    exc.add_note(
                        f"Catalog rollback also failed during delete: {rollback_error}"
                    )
                raise
