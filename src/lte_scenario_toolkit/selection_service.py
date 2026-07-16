"""Shared preflight, candidate scanning, and DEM-statistics services."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.errors import WindowError
from rasterio.features import geometry_mask, geometry_window
from rasterio.windows import intersect as windows_intersect
from shapely.geometry import box, mapping
from shapely.ops import transform as transform_geometry

from .candidate_cache import CandidateCache, cache_key
from .candidate_scanner import (
    Candidate,
    ScanCancelled,
    ScanRequest,
    ScanResult,
    scan_candidates,
)
from .data_validation import validate_scenario_data
from .profiles import ExperimentProfile
from .spatial import prepare_spatial_data

SCANNER_ALGORITHM_VERSION = "row-sweep-v1"


class SelectionError(ValueError):
    """Base error with a stable GUI/CLI mapping code and optional details."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = {} if details is None else dict(details)


class SelectionPreflightError(SelectionError):
    """Raised when selection inputs cannot pass the read-only preflight."""


@dataclass(frozen=True)
class DemStatistics:
    minimum: float
    maximum: float
    mean: float
    elevation_range: float
    valid_pixel_count: int


@dataclass(frozen=True)
class SelectionPreflight:
    scenario_id: str
    profile: ExperimentProfile
    points_path: Path
    boundary_path: Path
    dem_path: Path
    output_root: Path
    boundary_fingerprint: str
    points_fingerprint: str
    dem_fingerprint: str


def _canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_records(catalog: Any) -> dict[str, dict[str, Any]]:
    manifest_path = Path(catalog.root).resolve() / "data" / "manifest.json"
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Scenario manifest is unreadable: {manifest_path}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 2:
        raise ValueError("Scenario manifest schema_version must be 2")
    datasets = document.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("Scenario manifest datasets must be a list")
    records: dict[str, dict[str, Any]] = {}
    for record in datasets:
        if not isinstance(record, dict):
            raise ValueError("Scenario manifest contains a malformed dataset")
        dataset_id = record.get("dataset_id")
        if type(dataset_id) is not str or not dataset_id:
            raise ValueError("Scenario manifest contains a dataset without an ID")
        if dataset_id in records:
            raise ValueError(f"Scenario manifest repeats dataset {dataset_id!r}")
        records[dataset_id] = record
    return records


def _dataset_fingerprint(
    records: dict[str, dict[str, Any]],
    dataset_id: str,
) -> str:
    try:
        record = records[dataset_id]
    except KeyError as exc:
        raise ValueError(f"Scenario manifest has no dataset {dataset_id!r}") from exc
    files = record.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Scenario manifest dataset {dataset_id!r} has no files")
    normalized: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError(f"Scenario manifest dataset {dataset_id!r} has malformed files")
        path = item.get("path")
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if (
            type(path) is not str
            or not path
            or type(size) is not int
            or size < 0
            or type(digest) is not str
            or len(digest) != 64
        ):
            raise ValueError(f"Scenario manifest dataset {dataset_id!r} has malformed files")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(
                f"Scenario manifest dataset {dataset_id!r} has malformed files"
            ) from exc
        normalized.append(
            {
                "path": path.replace("\\", "/"),
                "size_bytes": size,
                "sha256": digest.casefold(),
            }
        )
    normalized.sort(key=lambda item: item["path"])
    return _canonical_fingerprint(normalized)


def _resolved_output_root(catalog: Any, output_root: str | os.PathLike[str]) -> Path:
    try:
        path = Path(output_root).expanduser()
    except TypeError as exc:
        raise ValueError("output_root must be a path") from exc
    if not path.is_absolute():
        path = Path(catalog.root).resolve() / path
    path = path.resolve()
    writable = path if path.exists() else path.parent
    if not writable.is_dir():
        raise ValueError(f"Output root parent is not a directory: {writable}")
    if not os.access(writable, os.W_OK):
        raise ValueError(f"Output root is not writable: {writable}")
    return path


def _target_crs(value: str) -> CRS:
    try:
        crs = CRS.from_user_input(value)
    except Exception as exc:
        raise ValueError(f"Invalid target CRS: {value!r}") from exc
    if not crs.is_projected:
        raise ValueError("Target CRS must be projected for metre-based scan parameters")
    units = {axis.unit_name.casefold() for axis in crs.axis_info if axis.unit_name}
    if units and not units.issubset({"metre", "meter"}):
        raise ValueError("Target CRS axes must use metres")
    return crs


def _validation_error(report: Any) -> str:
    messages = getattr(report, "messages", ())
    details = "; ".join(
        f"{getattr(message, 'code', 'validation')}: "
        f"{getattr(message, 'message', message)}"
        for message in messages
        if str(getattr(message, "level", "error")).casefold() == "error"
    )
    return details or "scenario data validation failed"


def stream_dem_statistics(
    dem: Any,
    geometry: Any,
    *,
    geometry_crs: Any,
) -> DemStatistics:
    """Accumulate exact valid-pixel statistics one native DEM block at a time."""

    if getattr(dem, "crs", None) is None:
        raise ValueError("DEM requires a CRS")
    if getattr(dem, "count", 0) < 1:
        raise ValueError("DEM requires at least one band")
    if geometry is None or getattr(geometry, "is_empty", True):
        raise ValueError("Candidate geometry must not be empty")
    try:
        source_crs = CRS.from_user_input(geometry_crs)
    except Exception as exc:
        raise ValueError(f"Invalid geometry CRS: {geometry_crs!r}") from exc
    target_crs = CRS.from_user_input(dem.crs)
    projected = geometry
    if source_crs != target_crs:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        projected = transform_geometry(transformer.transform, geometry)
    if projected.is_empty:
        raise ValueError("Candidate geometry does not overlap valid DEM pixels")

    try:
        candidate_window = geometry_window(dem, [mapping(projected)])
    except WindowError as exc:
        raise ValueError("Candidate geometry does not overlap valid DEM pixels") from exc

    valid_count = 0
    value_sum = 0.0
    minimum = np.inf
    maximum = -np.inf
    for _, block_window in dem.block_windows(1):
        if not windows_intersect(block_window, candidate_window):
            continue
        data = dem.read(1, window=block_window, masked=True)
        inside = geometry_mask(
            [mapping(projected)],
            out_shape=data.shape,
            transform=dem.window_transform(block_window),
            invert=True,
        )
        array = np.asarray(data.data)
        valid = inside & ~np.ma.getmaskarray(data) & np.isfinite(array)
        if not bool(valid.any()):
            continue
        values = np.asarray(array[valid], dtype=np.float64)
        valid_count += int(values.size)
        value_sum += float(values.sum(dtype=np.float64))
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))

    if valid_count == 0:
        raise ValueError("Candidate geometry contains no valid DEM pixels")
    return DemStatistics(
        minimum=float(minimum),
        maximum=float(maximum),
        mean=value_sum / valid_count,
        elevation_range=float(maximum - minimum),
        valid_pixel_count=valid_count,
    )


class SelectionService:
    """Share one validated selection implementation between CLI and GUI."""

    def __init__(self, catalog: Any) -> None:
        self.catalog = catalog
        self.cache = CandidateCache(catalog.root)

    def preflight(
        self,
        profile: ExperimentProfile,
        output_root: str | os.PathLike[str],
    ) -> SelectionPreflight:
        """Resolve and validate selection inputs without creating output paths."""

        scenario_id = getattr(profile, "scenario_id", None)
        if type(scenario_id) is not str or not scenario_id:
            raise SelectionPreflightError(
                "profile.scenario_id",
                "Profile scenario_id must be a non-empty string",
            )
        try:
            status = self.catalog.scenario_status(scenario_id)
        except Exception as exc:
            raise SelectionPreflightError(
                "scenario.unknown",
                f"Cannot resolve scenario {scenario_id!r}: {exc}",
                details={"scenario_id": scenario_id},
            ) from exc
        if status != "ready":
            raise SelectionPreflightError(
                "scenario.not_ready",
                f"Scenario {scenario_id!r} must be ready before selection (status: {status})",
                details={"scenario_id": scenario_id, "status": status},
            )
        if not isinstance(profile, ExperimentProfile):
            raise SelectionPreflightError(
                "profile.invalid",
                "profile must be an ExperimentProfile",
            )
        try:
            _target_crs(profile.target_crs)
        except ValueError as exc:
            raise SelectionPreflightError(
                "profile.target_crs",
                str(exc),
                details={"target_crs": profile.target_crs},
            ) from exc
        try:
            report = validate_scenario_data(self.catalog, scenario_id)
        except Exception as exc:
            raise SelectionPreflightError(
                "scenario.validation_failed",
                f"Scenario data validation could not run: {exc}",
                details={"scenario_id": scenario_id},
            ) from exc
        if getattr(report, "status", status) != "ready" or not report.ok:
            raise SelectionPreflightError(
                "scenario.validation_failed",
                _validation_error(report),
                details={"scenario_id": scenario_id},
            )

        try:
            scenario = self.catalog.scenario(scenario_id)
        except Exception as exc:
            raise SelectionPreflightError(
                "scenario.unknown",
                f"Cannot resolve scenario {scenario_id!r}: {exc}",
                details={"scenario_id": scenario_id},
            ) from exc
        try:
            points = self.catalog.dataset(profile.points_dataset_id)
        except Exception as exc:
            raise SelectionPreflightError(
                "inputs.points_dataset_id",
                f"Cannot resolve points dataset {profile.points_dataset_id!r}: {exc}",
                details={"points_dataset_id": profile.points_dataset_id},
            ) from exc
        if points.get("role") != "points":
            raise SelectionPreflightError(
                "inputs.points_dataset_id",
                f"Profile dataset {profile.points_dataset_id!r} must have role 'points'",
                details={"points_dataset_id": profile.points_dataset_id},
            )
        boundary_id = scenario.get("boundary_dataset_id")
        dem_id = scenario.get("dem_dataset_id")
        if type(boundary_id) is not str or not boundary_id:
            raise SelectionPreflightError(
                "scenario.boundary_dataset",
                f"Scenario {scenario_id!r} has no boundary dataset",
            )
        if type(dem_id) is not str or not dem_id:
            raise SelectionPreflightError(
                "scenario.dem_dataset",
                f"Scenario {scenario_id!r} has no DEM dataset",
            )
        try:
            boundary = self.catalog.dataset(boundary_id)
            dem = self.catalog.dataset(dem_id)
        except Exception as exc:
            raise SelectionPreflightError(
                "scenario.datasets",
                f"Cannot resolve registered scenario datasets: {exc}",
                details={"scenario_id": scenario_id},
            ) from exc
        if boundary.get("role") != "boundary" or dem.get("role") != "dem":
            raise SelectionPreflightError(
                "scenario.dataset_roles",
                f"Scenario {scenario_id!r} has invalid registered dataset roles",
            )
        try:
            points_path = self.catalog.resolve(points["entrypoint"])
            boundary_path = self.catalog.resolve(boundary["entrypoint"])
            dem_path = self.catalog.resolve(dem["entrypoint"])
        except Exception as exc:
            raise SelectionPreflightError(
                "inputs.paths",
                f"Cannot resolve registered input paths: {exc}",
                details={"scenario_id": scenario_id},
            ) from exc
        for label, path in (
            ("points", points_path),
            ("boundary", boundary_path),
            ("DEM", dem_path),
        ):
            if not path.is_file():
                raise SelectionPreflightError(
                    "inputs.missing",
                    f"Registered {label} entrypoint does not exist: {path}",
                    details={"input": label, "path": str(path)},
                )
        try:
            with rasterio.open(dem_path) as opened_dem:
                if opened_dem.crs is None or opened_dem.count < 1:
                    raise SelectionPreflightError(
                        "inputs.dem",
                        f"Registered DEM is missing raster metadata: {dem_path}",
                        details={"path": str(dem_path)},
                    )
        except (OSError, rasterio.errors.RasterioError) as exc:
            raise SelectionPreflightError(
                "inputs.dem",
                f"Registered DEM is unreadable: {dem_path}",
                details={"path": str(dem_path)},
            ) from exc

        try:
            records = _manifest_records(self.catalog)
            boundary_fingerprint = _dataset_fingerprint(records, boundary_id)
            points_fingerprint = _dataset_fingerprint(
                records,
                profile.points_dataset_id,
            )
            dem_fingerprint = _dataset_fingerprint(records, dem_id)
        except ValueError as exc:
            raise SelectionPreflightError(
                "inputs.manifest",
                str(exc),
            ) from exc
        try:
            resolved_output = _resolved_output_root(self.catalog, output_root)
        except ValueError as exc:
            raise SelectionPreflightError(
                "outputs.root",
                str(exc),
                details={"output_root": str(output_root)},
            ) from exc
        return SelectionPreflight(
            scenario_id=scenario_id,
            profile=profile,
            points_path=points_path,
            boundary_path=boundary_path,
            dem_path=dem_path,
            output_root=resolved_output,
            boundary_fingerprint=boundary_fingerprint,
            points_fingerprint=points_fingerprint,
            dem_fingerprint=dem_fingerprint,
        )

    @staticmethod
    def _request(profile: ExperimentProfile) -> ScanRequest:
        return ScanRequest(
            rectangle_size=profile.rect_size,
            target_count=profile.target_count,
            tolerance=profile.tolerance,
            step=profile.scan_step,
            max_candidates=profile.max_rects,
            minimum_spacing=profile.min_spacing,
            strategy=profile.strategy,
            mode=profile.scan_mode,
            random_seed=profile.random_seed,
            algorithm_version=SCANNER_ALGORITHM_VERSION,
        )

    def scan(
        self,
        preflight: SelectionPreflight,
        force: bool = False,
        progress: Any = None,
        cancel: Any = None,
    ) -> ScanResult:
        """Load or compute a completed candidate scan through the shared cache."""

        if not isinstance(preflight, SelectionPreflight):
            raise ValueError("preflight must be a SelectionPreflight")
        if cancel is not None and cancel.is_set():
            raise ScanCancelled("Candidate scan was cancelled")
        points = gpd.read_file(preflight.points_path)
        boundaries = gpd.read_file(preflight.boundary_path)
        _, boundary, coordinates = prepare_spatial_data(
            points,
            boundaries,
            target_crs=preflight.profile.target_crs,
        )
        request = self._request(preflight.profile)
        key = cache_key(
            request,
            preflight.scenario_id,
            preflight.boundary_fingerprint,
            preflight.points_fingerprint,
            preflight.profile.target_crs,
        )
        if not force:
            cached = self.cache.load(key, request)
            if cached is not None:
                if cancel is not None and cancel.is_set():
                    raise ScanCancelled("Candidate scan was cancelled")
                return cached
        result = scan_candidates(
            request,
            boundary,
            coordinates,
            progress=progress,
            cancel=cancel,
        )
        if result.completed:
            self.cache.store(key, request, result)
        return result

    def candidate_statistics(
        self,
        preflight: SelectionPreflight,
        candidate: Candidate,
    ) -> DemStatistics:
        """Compute exact native-resolution DEM statistics for one candidate."""

        if not isinstance(preflight, SelectionPreflight):
            raise ValueError("preflight must be a SelectionPreflight")
        if not isinstance(candidate, Candidate):
            raise ValueError("candidate must be a Candidate")
        geometry = box(
            candidate.left_x,
            candidate.bottom_y,
            candidate.left_x + preflight.profile.rect_size,
            candidate.bottom_y + preflight.profile.rect_size,
        )
        with rasterio.open(preflight.dem_path) as dem:
            return stream_dem_statistics(
                dem,
                geometry,
                geometry_crs=preflight.profile.target_crs,
            )


__all__ = [
    "DemStatistics",
    "SCANNER_ALGORITHM_VERSION",
    "SelectionError",
    "SelectionPreflight",
    "SelectionPreflightError",
    "SelectionService",
    "stream_dem_statistics",
]
