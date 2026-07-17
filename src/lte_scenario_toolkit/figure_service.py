"""Validated scenario sources and previewable terrain-figure workflows."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
import yaml
from matplotlib.colors import is_color_like
from pyproj import CRS

from . import io, visualization
from .data_catalog import load_data_catalog
from .run_service import RunService

REQUIRED_COLUMNS = frozenset(
    {
        "rect_id",
        "pt_count",
        "left_x",
        "bottom_y",
        "center_x",
        "center_y",
        "X",
        "Y",
    }
)
_RECTANGLE_COLUMNS = (
    "rect_id",
    "pt_count",
    "left_x",
    "bottom_y",
    "center_x",
    "center_y",
)
_FORMATS = frozenset({"png", "eps", "html"})


@dataclass(frozen=True)
class SelectionFigureIdentity:
    """Stable provenance for a figure prepared from an unexported selection."""

    scenario_id: str
    profile_id: str
    profile_fingerprint: str
    points_fingerprint: str
    boundary_fingerprint: str
    dem_fingerprint: str
    scan_algorithm_version: str
    scan_checked_positions: int
    scan_total_positions: int
    candidate_index: int
    candidate_flat_grid_id: int
    candidate_point_count: int
    candidate_left_x: float
    candidate_bottom_y: float
    candidate_center_x: float
    candidate_center_y: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FigureCsvIdentity:
    path: Path
    size_bytes: int
    mtime_ns: int
    sha256: str


def _csv_identity(path: Path) -> FigureCsvIdentity:
    if path.is_symlink() or not path.is_file():
        raise ValueError("figure source CSV must be a regular file")
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return FigureCsvIdentity(
        path=resolved,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=io.sha256_file(resolved),
    )


def validate_csv_identity(source: FigureSource) -> FigureCsvIdentity:
    identity = source.csv_identity
    if source.csv_path is None or identity is None:
        raise ValueError("figure source has no captured CSV identity")
    current = _csv_identity(source.csv_path)
    if current != identity:
        raise ValueError("figure source CSV changed after it was loaded; reload the source")
    return identity


@dataclass(frozen=True)
class FigureSource:
    """One validated rectangle and its points, without an open raster handle."""

    path: Path | None
    csv_path: Path | None
    frame: pd.DataFrame
    rectangle: dict[str, Any]
    points: gpd.GeoDataFrame
    target_crs: str
    rectangle_size_m: float
    source_kind: str = "csv"
    warnings: tuple[str, ...] = ()
    dem_path: Path | None = None
    run_id: str | None = None
    scenario_id: str | None = None
    profile_id: str | None = None
    dem_fingerprint: str | None = None
    selection_identity: SelectionFigureIdentity | None = None
    csv_identity: FigureCsvIdentity | None = None

    def __post_init__(self) -> None:
        if self.source_kind not in {"csv", "run", "selection"}:
            raise ValueError("figure source kind must be csv, run, or selection")
        if self.source_kind == "selection":
            if (
                self.selection_identity is None
                or self.path is not None
                or self.csv_path is not None
                or self.csv_identity is not None
            ):
                raise ValueError(
                    "selection figure source requires frozen identity and no source path"
                )

    def snapshot(self) -> FigureSource:
        """Copy mutable tabular members before handing the source to a worker."""

        return replace(
            self,
            frame=self.frame.copy(deep=True),
            rectangle=dict(self.rectangle),
            points=self.points.copy(deep=True),
        )


@dataclass(frozen=True)
class FigureSourceInspection:
    """Safe source summary used before a legacy rectangle is selected."""

    path: Path
    source_kind: str
    rectangle_ids: tuple[Any, ...]
    warnings: tuple[str, ...] = ()
    run_id: str | None = None

    @property
    def requires_rectangle(self) -> bool:
        return len(self.rectangle_ids) > 1


@dataclass(frozen=True)
class _SourceContext:
    root: Path
    csv_path: Path
    record: Mapping[str, Any]
    target_crs: str
    rectangle_size_m: float
    dem_path: Path | None
    dem_fingerprint: str | None
    warnings: tuple[str, ...]
    source_kind: str


@dataclass(frozen=True)
class FigureSpec:
    """Complete, validated terrain-figure styling and sampling settings."""

    preset: str
    colormap: str
    dpi: int
    azimuth: float
    elevation_angle: float
    vertical_exaggeration: float
    station_color: str
    station_size: float
    title: str | None
    max_pixels: int

    @classmethod
    def from_preset(cls, preset: str) -> FigureSpec:
        if preset == "preview":
            return cls(
                preset="preview",
                colormap="terrain",
                dpi=120,
                azimuth=-60.0,
                elevation_angle=30.0,
                vertical_exaggeration=1.0,
                station_color="red",
                station_size=20.0,
                title=None,
                max_pixels=600,
            )
        if preset == "publication":
            return cls(
                preset="publication",
                colormap="terrain",
                dpi=300,
                azimuth=-60.0,
                elevation_angle=30.0,
                vertical_exaggeration=1.0,
                station_color="red",
                station_size=20.0,
                title=None,
                max_pixels=1800,
            )
        raise ValueError("preset must be one of: preview, publication")

    def validate(self) -> FigureSpec:
        if self.preset not in {"preview", "publication"}:
            raise ValueError("figure preset must be one of: preview, publication")
        if type(self.colormap) is not str or not self.colormap.strip():
            raise ValueError("figure colormap must be a non-empty string")
        try:
            matplotlib.colormaps.get_cmap(self.colormap)
        except ValueError as exc:
            raise ValueError(f"Unknown figure colormap: {self.colormap}") from exc
        if type(self.dpi) is not int or self.dpi <= 0:
            raise ValueError("Figure DPI must be a positive integer")
        for name, value in (
            ("azimuth", self.azimuth),
            ("elevation angle", self.elevation_angle),
            ("vertical exaggeration", self.vertical_exaggeration),
            ("station size", self.station_size),
        ):
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise ValueError(f"Figure {name} must be a finite number")
        if not -90 <= float(self.elevation_angle) <= 90:
            raise ValueError("Figure elevation angle must be between -90 and 90")
        if self.vertical_exaggeration <= 0:
            raise ValueError("Figure vertical exaggeration must be greater than zero")
        if self.station_size <= 0:
            raise ValueError("Figure station size must be greater than zero")
        if type(self.station_color) is not str or not is_color_like(self.station_color):
            raise ValueError("Figure station color must be a valid Matplotlib color")
        if self.title is not None and type(self.title) is not str:
            raise ValueError("Figure title must be null or a string")
        if type(self.max_pixels) is not int or self.max_pixels <= 0:
            raise ValueError("Figure max_pixels must be a positive integer")
        return self

    def resolved_title(self, rectangle_size_m: float, point_count: int) -> str:
        if self.title is not None:
            return self.title
        if self.preset == "publication":
            return {1000: "DCMOP1", 2000: "DCMOP2", 3000: "DCMOP3"}.get(
                int(rectangle_size_m),
                f"{rectangle_size_m:g}m",
            )
        return (
            f"Terrain | {rectangle_size_m:g}m x {rectangle_size_m:g}m | "
            f"{point_count} stations"
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FigureResult:
    """Immutable description of a rendered or published figure result."""

    path: Path
    artifacts: tuple[Path, ...]
    errors: tuple[dict[str, str], ...] = ()


def _mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read run record {path}: {exc}") from exc
    return _mapping(document, description="run record")


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read run snapshot {path}: {exc}") from exc
    return _mapping(document, description="run snapshot")


def _contained_regular_file(parent: Path, relative: Any, *, description: str) -> Path:
    if type(relative) is not str or not relative:
        raise ValueError(f"{description} must be a contained relative path")
    candidate_relative = Path(relative)
    if (
        candidate_relative.is_absolute()
        or candidate_relative == Path(".")
        or ".." in candidate_relative.parts
    ):
        raise ValueError(f"{description} must be a contained relative path: {relative}")
    candidate = parent / candidate_relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{description} must be an existing regular file: {relative}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(parent)
    except ValueError as exc:
        raise ValueError(f"{description} escapes its run directory: {relative}") from exc
    return resolved


def _validated_dem_path(value: Any, *, description: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{description} must be an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{description} must be an absolute path: {path}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be an existing regular file: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError(f"{description} must be a resolved path: {path}")
    try:
        with rasterio.open(resolved) as dem:
            if dem.crs is None:
                raise ValueError("DEM has no CRS")
            if dem.count < 1:
                raise ValueError("DEM has no raster band")
    except (OSError, rasterio.errors.RasterioError, ValueError) as exc:
        raise ValueError(f"{description} is not a readable georeferenced raster: {exc}") from exc
    return resolved


def _catalog_dem_path(catalog_value: Any, dataset_id: str) -> Path:
    if not isinstance(catalog_value, (str, os.PathLike)):
        raise ValueError("run DEM catalog_path must be an absolute path")
    catalog_path = Path(catalog_value)
    if not catalog_path.is_absolute():
        raise ValueError(f"run DEM catalog_path must be absolute: {catalog_path}")
    if catalog_path.is_symlink() or not catalog_path.is_file():
        raise ValueError(f"run DEM catalog_path must be a regular file: {catalog_path}")
    catalog_path = catalog_path.resolve(strict=True)
    try:
        catalog = load_data_catalog(catalog_path, repo_root=catalog_path.parent.parent)
        record = catalog.dataset(dataset_id)
        if record["role"] != "dem":
            raise ValueError(f"dataset {dataset_id!r} is not a DEM")
        return _validated_dem_path(
            catalog.resolve(record["entrypoint"]),
            description="catalog DEM entrypoint",
        )
    except Exception as exc:
        raise ValueError(
            f"Cannot resolve DEM dataset {dataset_id!r} from {catalog_path}: {exc}"
        ) from exc


def _run_dem_path(
    run_dir: Path,
    record: Mapping[str, Any],
) -> tuple[Path | None, str | None]:
    metadata = record.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("run metadata must be a mapping")
    inputs = metadata.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise ValueError("run metadata inputs must be a mapping")
    dem = inputs.get("dem")
    if dem is None:
        return None, None
    if not isinstance(dem, Mapping):
        raise ValueError("run metadata DEM input must be a mapping")
    fingerprint = dem.get("fingerprint")
    if fingerprint is not None and (type(fingerprint) is not str or not fingerprint):
        raise ValueError("run DEM fingerprint must be a non-empty string")
    if "path" in dem:
        return (
            _validated_dem_path(dem["path"], description="run DEM path"),
            fingerprint,
        )
    catalog_path = dem.get("catalog_path")
    dataset_id = dem.get("dataset_id")
    if catalog_path is None:
        return None, fingerprint
    if type(dataset_id) is not str or not dataset_id:
        raise ValueError(
            "run DEM dataset_id must be a non-empty string when catalog_path is recorded"
        )
    return _catalog_dem_path(catalog_path, dataset_id), fingerprint


def _run_source(
    path: Path,
) -> tuple[
    Path,
    Path,
    Mapping[str, Any],
    str,
    float,
    Path | None,
    str | None,
]:
    run_dir = path.parent if path.name.casefold() == "run.json" else path
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError(f"Figure run source must be a real directory: {run_dir}")
    run_dir = run_dir.resolve(strict=True)
    record_path = run_dir / "run.json"
    if record_path.is_symlink() or not record_path.is_file():
        raise ValueError(f"Figure run source is missing run.json: {run_dir}")
    record = _read_json(record_path)
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("run artifacts must be a list")
    csv_artifacts = [
        item
        for item in artifacts
        if type(item) is str and Path(item).suffix.casefold() == ".csv"
    ]
    if len(csv_artifacts) != 1:
        raise ValueError("Figure run must contain exactly one scenario CSV artifact")
    csv_path = _contained_regular_file(
        run_dir,
        csv_artifacts[0],
        description="scenario CSV artifact",
    )
    metadata = _mapping(record.get("metadata", {}), description="run metadata")
    if metadata.get("run_kind") == "figure":
        source = _mapping(metadata.get("source", {}), description="figure source")
        if source.get("artifact") != csv_artifacts[0]:
            raise ValueError("figure source artifact does not match its CSV artifact")
        target_crs = metadata.get("target_crs")
        rectangle_size = metadata.get("rectangle_size_m")
    else:
        config_path = run_dir / "run-config.yaml"
        if config_path.is_symlink() or not config_path.is_file():
            raise ValueError(f"Figure run source is missing run-config.yaml: {run_dir}")
        snapshot = _read_yaml(config_path)
        if "profile" in snapshot:
            profile = _mapping(
                snapshot["profile"],
                description="run snapshot profile section",
            )
            for snapshot_field, record_field in (
                ("id", "profile_id"),
                ("scenario_id", "scenario_id"),
            ):
                if snapshot_field in profile and record_field in record:
                    if profile[snapshot_field] != record[record_field]:
                        raise ValueError(
                            f"run snapshot profile.{snapshot_field} does not match "
                            f"run.json {record_field}"
                        )
        spatial = _mapping(
            snapshot.get("spatial"),
            description="run snapshot spatial section",
        )
        target_crs = spatial.get("target_crs")
        rectangle_size = spatial.get("rectangle_size_m")
    if type(target_crs) is not str or not target_crs:
        raise ValueError("run figure target CRS must be a non-empty string")
    try:
        CRS.from_user_input(target_crs)
    except Exception as exc:
        raise ValueError(f"Invalid run snapshot target CRS: {target_crs}") from exc
    if type(rectangle_size) not in {int, float} or not math.isfinite(rectangle_size):
        raise ValueError("run figure rectangle_size_m must be a finite number")
    if rectangle_size <= 0:
        raise ValueError("run snapshot rectangle_size_m must be greater than zero")
    dem_path, dem_fingerprint = _run_dem_path(run_dir, record)
    return (
        run_dir,
        csv_path,
        record,
        target_crs,
        float(rectangle_size),
        dem_path,
        dem_fingerprint,
    )


def _read_csv_frame(csv_path: Path, *, allow_empty: bool = False) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            csv_path,
            dtype={
                "run_id": "string",
                "scenario_id": "string",
                "profile_id": "string",
                "candidate_id": "string",
            },
        )
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError(f"Cannot read scenario CSV {csv_path}: {exc}") from exc
    if frame.empty and not allow_empty:
        raise ValueError(f"Scenario CSV is empty: {csv_path}")
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Scenario CSV missing required columns: {', '.join(missing)}")
    numeric_columns = set(REQUIRED_COLUMNS)
    if "elevation" in frame.columns:
        numeric_columns.add("elevation")
    numeric = frame.loc[:, sorted(numeric_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Scenario CSV required numeric values must be finite")
    frame = frame.copy()
    frame.loc[:, sorted(numeric_columns)] = numeric
    return frame


def _rectangle_ids(frame: pd.DataFrame) -> tuple[Any, ...]:
    values: list[Any] = []
    for value in frame["rect_id"].drop_duplicates().tolist():
        values.append(value.item() if hasattr(value, "item") else value)
    return tuple(values)


def _read_frame(
    csv_path: Path,
    *,
    rect_id: Any,
    empty_rectangle: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _read_csv_frame(csv_path, allow_empty=empty_rectangle is not None)
    if frame.empty:
        if empty_rectangle is None:
            raise ValueError(f"Scenario CSV is empty: {csv_path}")
        missing = [name for name in _RECTANGLE_COLUMNS if name not in empty_rectangle]
        if missing:
            raise ValueError(
                "figure source snapshot rectangle is missing: " + ", ".join(missing)
            )
        rectangle: dict[str, Any] = {}
        for name in _RECTANGLE_COLUMNS:
            value = empty_rectangle[name]
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(
                    f"empty figure source snapshot {name} must be finite numeric"
                )
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"empty figure source snapshot {name} must be finite numeric"
                ) from exc
            if not math.isfinite(numeric):
                raise ValueError(
                    f"empty figure source snapshot {name} must be finite numeric"
                )
            rectangle[name] = int(numeric) if name == "pt_count" else numeric
        if float(empty_rectangle["pt_count"]) != 0:
            raise ValueError("empty figure source snapshot must have pt_count 0")
        if rect_id is not None and float(rect_id) != float(rectangle["rect_id"]):
            raise ValueError(f"rect_id {rect_id!r} is not present in the scenario CSV")
        return frame.copy(), rectangle
    rectangle_ids = _rectangle_ids(frame)
    if rect_id is None:
        if len(rectangle_ids) != 1:
            raise ValueError("Scenario CSV has multiple rect_id values; choose rect_id explicitly")
        selected_id = rectangle_ids[0]
    else:
        try:
            selected_numeric = float(rect_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rect_id must be numeric: {rect_id!r}") from exc
        matching_ids = [value for value in rectangle_ids if float(value) == selected_numeric]
        if len(matching_ids) != 1:
            raise ValueError(f"rect_id {rect_id!r} is not present in the scenario CSV")
        selected_id = matching_ids[0]
    selected = frame.loc[frame["rect_id"] == selected_id].copy().reset_index(drop=True)
    rectangle: dict[str, Any] = {}
    for column in _RECTANGLE_COLUMNS:
        values = selected[column].drop_duplicates().tolist()
        if len(values) != 1:
            raise ValueError(f"Scenario CSV rectangle column {column} is inconsistent")
        value = values[0]
        rectangle[column] = value.item() if hasattr(value, "item") else value
    point_count_value = rectangle["pt_count"]
    if isinstance(point_count_value, (bool, np.bool_)):
        raise ValueError("Scenario CSV pt_count must be a non-negative integer")
    point_count = float(point_count_value)
    if point_count < 0 or not point_count.is_integer():
        raise ValueError("Scenario CSV pt_count must be a non-negative integer")
    if int(point_count) != len(selected):
        raise ValueError(
            "Scenario CSV pt_count must match the selected rectangle row count"
        )
    rectangle["pt_count"] = int(point_count)
    return selected, rectangle


def _recorded_figure_rectangle(
    record: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    metadata_value = record.get("metadata", {})
    metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
    value = metadata.get("rectangle")
    if metadata.get("run_kind") == "figure" and isinstance(value, Mapping):
        return value
    candidate_value = metadata.get("candidate")
    candidate = candidate_value if isinstance(candidate_value, Mapping) else {}
    required = {
        "point_count",
        "left_x",
        "bottom_y",
        "center_x",
        "center_y",
    }
    if metadata.get("run_kind") == "selection" and required.issubset(candidate):
        return {
            "rect_id": 1,
            "pt_count": candidate["point_count"],
            "left_x": candidate["left_x"],
            "bottom_y": candidate["bottom_y"],
            "center_x": candidate["center_x"],
            "center_y": candidate["center_y"],
        }
    return None


def _json_safe_rectangle(rectangle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: (
            rectangle[name].item()
            if hasattr(rectangle[name], "item")
            else rectangle[name]
        )
        for name in _RECTANGLE_COLUMNS
    }


def _resolve_source_context(source_path: Path) -> _SourceContext:
    if not source_path.exists():
        raise FileNotFoundError(f"Figure source does not exist: {source_path}")
    if source_path.is_dir() or source_path.name.casefold() == "run.json":
        (
            root,
            csv_path,
            record,
            target_crs,
            rectangle_size,
            dem_path,
            dem_fingerprint,
        ) = _run_source(source_path)
        return _SourceContext(
            root=root,
            csv_path=csv_path,
            record=record,
            target_crs=target_crs,
            rectangle_size_m=rectangle_size,
            dem_path=dem_path,
            dem_fingerprint=dem_fingerprint,
            warnings=(),
            source_kind="run",
        )
    if source_path.is_file() and source_path.suffix.casefold() == ".csv":
        if source_path.is_symlink():
            raise ValueError(f"Scenario CSV must be a regular file: {source_path}")
        resolved_csv = source_path.resolve(strict=True)
        sibling_record = source_path.parent / "run.json"
        sibling_snapshot = source_path.parent / "run-config.yaml"
        if sibling_record.is_file() or sibling_snapshot.is_file():
            (
                root,
                csv_path,
                record,
                target_crs,
                rectangle_size,
                dem_path,
                dem_fingerprint,
            ) = _run_source(source_path.parent)
            if csv_path != resolved_csv:
                raise ValueError(
                    "Selected CSV is not the scenario CSV recorded by its sibling run.json"
                )
            return _SourceContext(
                root=root,
                csv_path=csv_path,
                record=record,
                target_crs=target_crs,
                rectangle_size_m=rectangle_size,
                dem_path=dem_path,
                dem_fingerprint=dem_fingerprint,
                warnings=(),
                source_kind="run",
            )
        return _SourceContext(
            root=resolved_csv,
            csv_path=resolved_csv,
            record={},
            target_crs="EPSG:3857",
            rectangle_size_m=0.0,
            dem_path=None,
            dem_fingerprint=None,
            warnings=(
                "Legacy scenario CSV has no run snapshot; assuming EPSG:3857.",
            ),
            source_kind="csv",
        )
    raise ValueError("Figure source must be a run directory, run.json, or CSV")


def _selection_run_consistency(
    record: Mapping[str, Any],
    frame: pd.DataFrame,
    rectangle: Mapping[str, Any],
) -> None:
    metadata = record.get("metadata", {})
    if not isinstance(metadata, Mapping) or metadata.get("run_kind") != "selection":
        return

    for column, record_field in (
        ("run_id", "run_id"),
        ("scenario_id", "scenario_id"),
        ("profile_id", "profile_id"),
    ):
        if frame.empty:
            continue
        if column not in frame.columns or record_field not in record:
            continue
        values = frame[column].drop_duplicates().tolist()
        if len(values) != 1 or values[0] != record[record_field]:
            raise ValueError(
                f"Scenario CSV {column} does not match run.json {record_field}"
            )

    if "candidate" not in metadata:
        return
    candidate = _mapping(
        metadata["candidate"],
        description="run metadata candidate",
    )
    if not frame.empty and "candidate_id" in frame.columns and "candidate_id" in candidate:
        values = frame["candidate_id"].drop_duplicates().tolist()
        if len(values) != 1 or values[0] != candidate["candidate_id"]:
            raise ValueError(
                "Scenario CSV candidate_id does not match run metadata candidate_id"
            )
    if "point_count" in candidate:
        try:
            point_count = float(candidate["point_count"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("run metadata candidate point_count is invalid") from exc
        if not math.isfinite(point_count) or point_count != rectangle["pt_count"]:
            raise ValueError(
                "run metadata candidate point_count does not match Scenario CSV pt_count"
            )
    for name in ("center_x", "center_y"):
        if name not in candidate:
            continue
        try:
            value = float(candidate[name])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"run metadata candidate {name} is invalid") from exc
        if not math.isfinite(value) or not math.isclose(
            value,
            float(rectangle[name]),
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"run metadata candidate {name} does not match Scenario CSV {name}"
            )


def _figure_run_consistency(
    record: Mapping[str, Any],
    rectangle: Mapping[str, Any],
) -> None:
    metadata_value = record.get("metadata", {})
    metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
    if metadata.get("run_kind") != "figure":
        return
    if metadata.get("rect_id") != rectangle["rect_id"]:
        raise ValueError("figure run rect_id does not match its source snapshot")
    source_value = metadata.get("source", {})
    source = source_value if isinstance(source_value, Mapping) else {}
    selection_value = source.get("selection")
    if isinstance(selection_value, Mapping):
        if selection_value.get("candidate_point_count") != rectangle["pt_count"]:
            raise ValueError(
                "figure run selection point count does not match its source snapshot"
            )


def _infer_rectangle_size(rectangle: Mapping[str, Any]) -> float:
    width = 2.0 * (float(rectangle["center_x"]) - float(rectangle["left_x"]))
    height = 2.0 * (float(rectangle["center_y"]) - float(rectangle["bottom_y"]))
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError("Scenario CSV cannot infer a positive rectangle size")
    if not math.isclose(width, height, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("Scenario CSV rectangle is not square")
    return width


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")


def _promote_nonempty(temporary: Path, destination: Path) -> Path:
    if temporary.is_symlink() or not temporary.is_file() or temporary.stat().st_size <= 0:
        raise ValueError(f"Rendered figure is missing or empty: {temporary}")
    temporary.replace(destination)
    return destination


def _complete_staging(run_path: Path, artifacts: Iterable[str]) -> bool:
    try:
        paths = (run_path / "run.json", *(run_path / name for name in artifacts))
        return all(
            not path.is_symlink()
            and path.is_file()
            and path.stat().st_size > 0
            for path in paths
        )
    except OSError:
        return False


def _normalise_formats(formats: Iterable[str]) -> tuple[str, ...]:
    if isinstance(formats, (str, bytes, os.PathLike)):
        raise ValueError("formats must be a non-empty collection")
    try:
        values = tuple(formats)
    except TypeError as exc:
        raise ValueError("formats must be a non-empty collection") from exc
    if not values:
        raise ValueError("formats must be a non-empty collection")
    normalised: list[str] = []
    for value in values:
        if type(value) is not str:
            raise ValueError("figure format must be text")
        token = value.casefold().removeprefix(".")
        if token not in _FORMATS:
            raise ValueError("figure format must be one of: eps, html, png")
        if token in normalised:
            raise ValueError(f"duplicate figure format: {token}")
        normalised.append(token)
    return tuple(normalised)


def _normalise_entrypoint(entrypoint: Iterable[str] | None) -> list[str]:
    if entrypoint is None:
        return []
    if isinstance(entrypoint, (str, bytes, os.PathLike)):
        raise ValueError("entrypoint must be a collection of command arguments")
    try:
        command = list(entrypoint)
    except TypeError as exc:
        raise ValueError(
            "entrypoint must be a collection of command arguments"
        ) from exc
    if any(type(item) is not str for item in command):
        raise ValueError("entrypoint arguments must be text")
    return command


def _require_dem(source: FigureSource) -> Path:
    if source.dem_path is None:
        raise ValueError(
            "A DEM path has not been resolved for this figure source; load a selection "
            "run with recorded DEM provenance or use the legacy config workflow"
        )
    return _validated_dem_path(source.dem_path, description="figure source DEM path")


def _input_metadata(source: FigureSource, dem_path: Path) -> dict[str, Any]:
    dem_stat = dem_path.stat()
    dem_fingerprint = source.dem_fingerprint
    fingerprint_source = "run"
    if dem_fingerprint is None:
        dem_fingerprint = io.sha256_file(dem_path)
        fingerprint_source = "sha256"
    result: dict[str, Any] = {
        "dem": {
            "path": str(dem_path),
            "size_bytes": dem_stat.st_size,
            "fingerprint": dem_fingerprint,
            "fingerprint_source": fingerprint_source,
        },
    }
    if source.csv_path is not None:
        csv_identity = validate_csv_identity(source)
        result["csv"] = {
            "path": str(csv_identity.path),
            "size_bytes": csv_identity.size_bytes,
            "sha256": csv_identity.sha256,
        }
    elif source.selection_identity is not None:
        result["selection"] = source.selection_identity.as_dict()
    else:
        raise ValueError("figure source has no CSV or frozen selection identity")
    return result


class FigureService:
    """Load, preview, and publish figures through one reusable service."""

    @staticmethod
    def inspect_source(path: str | Path) -> FigureSourceInspection:
        """Inspect a source and enumerate rectangle IDs without choosing one."""

        context = _resolve_source_context(Path(path))
        empty_rectangle = _recorded_figure_rectangle(context.record)
        frame = _read_csv_frame(
            context.csv_path,
            allow_empty=empty_rectangle is not None,
        )
        rectangle_ids = (
            (empty_rectangle["rect_id"],)
            if frame.empty and empty_rectangle is not None
            else _rectangle_ids(frame)
        )
        record_run_id = context.record.get("run_id")
        return FigureSourceInspection(
            path=context.root,
            source_kind=context.source_kind,
            rectangle_ids=rectangle_ids,
            warnings=context.warnings,
            run_id=record_run_id if type(record_run_id) is str else None,
        )

    @staticmethod
    def attach_dem(
        source: FigureSource,
        path: str | os.PathLike[str],
        *,
        fingerprint: str | None = None,
    ) -> FigureSource:
        """Return a source with one explicitly validated local DEM."""

        if not isinstance(source, FigureSource):
            raise ValueError("source must be a FigureSource")
        if fingerprint is not None and (type(fingerprint) is not str or not fingerprint):
            raise ValueError("DEM fingerprint must be non-empty text")
        dem_path = _validated_dem_path(path, description="figure source DEM path")
        return replace(
            source,
            dem_path=dem_path,
            dem_fingerprint=fingerprint,
            warnings=tuple(
                warning
                for warning in source.warnings
                if "DEM path has not been resolved" not in warning
            ),
        )

    @staticmethod
    def load_source(path: str | Path, rect_id: Any = None) -> FigureSource:
        context = _resolve_source_context(Path(path))
        frame, rectangle = _read_frame(
            context.csv_path,
            rect_id=rect_id,
            empty_rectangle=_recorded_figure_rectangle(context.record),
        )
        _selection_run_consistency(context.record, frame, rectangle)
        _figure_run_consistency(context.record, rectangle)
        rectangle_size = context.rectangle_size_m
        if rectangle_size == 0:
            rectangle_size = _infer_rectangle_size(rectangle)
        elif frame.empty:
            inferred_size = _infer_rectangle_size(rectangle)
            if not math.isclose(
                rectangle_size,
                inferred_size,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    "run rectangle_size_m does not match its source snapshot"
                )
        geometry = gpd.points_from_xy(frame["X"], frame["Y"])
        points = gpd.GeoDataFrame(
            frame.copy(),
            geometry=geometry,
            crs=context.target_crs,
        )
        return FigureSource(
            path=context.root,
            csv_path=context.csv_path,
            frame=frame,
            rectangle=rectangle,
            points=points,
            target_crs=context.target_crs,
            rectangle_size_m=rectangle_size,
            source_kind=context.source_kind,
            warnings=context.warnings,
            dem_path=context.dem_path,
            run_id=(
                context.record.get("run_id")
                if type(context.record.get("run_id")) is str
                else None
            ),
            scenario_id=(
                context.record.get("scenario_id")
                if type(context.record.get("scenario_id")) is str
                else None
            ),
            profile_id=(
                context.record.get("profile_id")
                if type(context.record.get("profile_id")) is str
                else None
            ),
            dem_fingerprint=context.dem_fingerprint,
            csv_identity=_csv_identity(context.csv_path),
        )

    @staticmethod
    def preview(source: FigureSource, spec: FigureSpec, output: str | Path) -> Path:
        if not isinstance(source, FigureSource):
            raise ValueError("source must be a FigureSource")
        if not isinstance(spec, FigureSpec):
            raise ValueError("spec must be a FigureSpec")
        spec.validate()
        dem_path = _require_dem(source)
        destination = Path(output).resolve()
        if destination.suffix.casefold() != ".png":
            raise ValueError("preview output must use a .png suffix")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_sibling(destination)
        try:
            with rasterio.open(dem_path) as dem:
                visualization.render_3d_terrain(
                    source.rectangle,
                    source.points,
                    dem,
                    spec,
                    rectangle_size=source.rectangle_size_m,
                    target_crs=source.target_crs,
                    png_path=temporary,
                )
            return _promote_nonempty(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def render(
        source: FigureSource,
        spec: FigureSpec,
        run_service: RunService,
        formats: Iterable[str],
        parent_run_id: str | None = None,
        *,
        entrypoint: Iterable[str] | None = None,
        repository: str | os.PathLike[str] | None = None,
    ) -> Path:
        if not isinstance(source, FigureSource):
            raise ValueError("source must be a FigureSource")
        if not isinstance(spec, FigureSpec):
            raise ValueError("spec must be a FigureSpec")
        if not isinstance(run_service, RunService):
            raise ValueError("run_service must be a RunService")
        spec.validate()
        requested = _normalise_formats(formats)
        command = _normalise_entrypoint(entrypoint)
        repository_root = (
            Path.cwd().resolve()
            if repository is None
            else Path(repository).expanduser().resolve(strict=False)
        )
        dem_path = _require_dem(source)
        inputs = _input_metadata(source, dem_path)
        with rasterio.open(dem_path) as dem:
            terrain_arrays = visualization.prepare_terrain_arrays(
                source.rectangle,
                source.points,
                dem,
                source.rectangle_size_m,
                source.target_crs,
                max_pixels=spec.max_pixels,
            )
        scenario_id = source.scenario_id or "legacy"
        profile_id = source.profile_id or "figures"
        run = run_service.begin(
            scenario_id,
            profile_id,
            parent_run_id=parent_run_id,
        )
        artifacts: list[str] = []
        rendered_artifacts: list[str] = []
        errors: list[dict[str, str]] = []
        publishing = False
        try:
            source_name = "source.csv"
            source_destination = run.path / source_name
            source_temporary = _temporary_sibling(source_destination)
            try:
                source.frame.to_csv(source_temporary, index=False, encoding="utf-8-sig")
                _promote_nonempty(source_temporary, source_destination)
                artifacts.append(source_name)
            finally:
                source_temporary.unlink(missing_ok=True)
            for token in requested:
                name = f"terrain.{token}"
                destination = run.path / name
                temporary = _temporary_sibling(destination)
                paths = {
                    "png_path": temporary if token == "png" else None,
                    "eps_path": temporary if token == "eps" else None,
                    "html_path": temporary if token == "html" else None,
                }
                try:
                    visualization.render_3d_terrain(
                        source.rectangle,
                        source.points,
                        None,
                        spec,
                        rectangle_size=source.rectangle_size_m,
                        target_crs=source.target_crs,
                        terrain_arrays=terrain_arrays,
                        **paths,
                    )
                    _promote_nonempty(temporary, destination)
                    artifacts.append(name)
                    rendered_artifacts.append(name)
                except Exception as exc:
                    errors.append(
                        {
                            "artifact": token,
                            "code": f"figure.{token}.failed",
                            "message": f"{type(exc).__name__}: {exc}",
                        }
                    )
                finally:
                    temporary.unlink(missing_ok=True)
            if not rendered_artifacts:
                raise ValueError("No requested figure format was rendered successfully")
            metadata = {
                "schema_version": 1,
                "run_kind": "figure",
                "source": {
                    "kind": source.source_kind,
                    "artifact": source_name,
                    "path": str(source.path) if source.path is not None else None,
                    "csv": (
                        str(source.csv_path) if source.csv_path is not None else None
                    ),
                    "run_id": source.run_id,
                    "selection": (
                        source.selection_identity.as_dict()
                        if source.selection_identity is not None
                        else None
                    ),
                },
                "inputs": inputs,
                "target_crs": source.target_crs,
                "rectangle_size_m": source.rectangle_size_m,
                "parameters": {
                    "target_crs": source.target_crs,
                    "rectangle_size_m": source.rectangle_size_m,
                },
                "rect_id": source.rectangle["rect_id"],
                "rectangle": _json_safe_rectangle(source.rectangle),
                "figure_spec": spec.as_dict(),
                "requested_formats": list(requested),
                "artifact_paths": {
                    Path(name).suffix.removeprefix("."): name
                    for name in rendered_artifacts
                },
                "warnings": list(source.warnings),
                "entrypoint": command,
                "git_commit": io._git_commit(repository_root),
                "software_versions": io.software_versions(),
            }
            publishing = True
            return run_service.publish(
                run,
                status="completed" if not errors else "partial",
                artifacts=artifacts,
                metadata=metadata,
                errors=errors,
            )
        except Exception as exc:
            if publishing:
                if _complete_staging(run.path, artifacts):
                    exc.add_note(f"Figure run staging retained for recovery: {run.path}")
                else:
                    run_service.abandon(run)
            else:
                run_service.abandon(run)
            raise


__all__ = [
    "FigureResult",
    "FigureCsvIdentity",
    "FigureService",
    "FigureSource",
    "FigureSourceInspection",
    "FigureSpec",
    "SelectionFigureIdentity",
    "validate_csv_identity",
]
