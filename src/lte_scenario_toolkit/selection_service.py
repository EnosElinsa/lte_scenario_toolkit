"""Shared preflight, candidate scanning, and DEM-statistics services."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
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

from . import io, visualization
from .candidate_cache import CandidateCache, cache_key
from .candidate_scanner import (
    Candidate,
    ScanCancelled,
    ScanProgress,
    ScanRequest,
    ScanResult,
    scan_candidates,
)
from .data_validation import validate_scenario_data
from .profiles import ExperimentProfile, dump_profile, validate_profile
from .run_service import RunService
from .spatial import prepare_spatial_data

SCANNER_ALGORITHM_VERSION = "row-sweep-v1"
EXPORT_ARTIFACTS = frozenset(
    {"csv", "preview_png", "terrain_png", "terrain_eps", "terrain_html"}
)
_ARTIFACT_ORDER = (
    "csv",
    "preview_png",
    "terrain_png",
    "terrain_eps",
    "terrain_html",
)


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


class SelectionScanError(SelectionError):
    """Raised when a candidate scan cannot produce a reusable result."""


class SelectionStatisticsError(SelectionError):
    """Raised when exact candidate DEM statistics cannot be computed."""


class SelectionExportError(SelectionError):
    """Raised when a selected candidate cannot be published safely."""


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
    boundary_dataset_id: str | None = None
    dem_dataset_id: str | None = None


@dataclass(frozen=True)
class PreparedSelection:
    """One prepared vector snapshot shared by scan, selector, and export."""

    preflight: SelectionPreflight
    points: gpd.GeoDataFrame
    boundary: Any
    coordinates: np.ndarray


@dataclass(frozen=True)
class SelectionProgress:
    """Cache-aware immutable progress emitted by the selection service."""

    phase: str
    checked_positions: int
    total_positions: int
    candidate_count: int
    elapsed_seconds: float
    added_candidates: tuple[Candidate, ...]
    removed_flat_grid_ids: tuple[int, ...]
    cache_status: str
    cache_key: str

    @classmethod
    def from_scan(
        cls,
        event: ScanProgress,
        *,
        cache_status: str,
        cache_key: str,
    ) -> SelectionProgress:
        return cls(
            phase=event.phase,
            checked_positions=event.checked_positions,
            total_positions=event.total_positions,
            candidate_count=event.candidate_count,
            elapsed_seconds=event.elapsed_seconds,
            added_candidates=event.added_candidates,
            removed_flat_grid_ids=event.removed_flat_grid_ids,
            cache_status=cache_status,
            cache_key=cache_key,
        )


def _canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity_value(value: Any) -> Any:
    """Convert a frozen profile document into canonical JSON primitives."""

    if isinstance(value, Path):
        return str(value.resolve(strict=False))
    if isinstance(value, dict):
        return {str(key): _identity_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_identity_value(item) for item in value]
    return value


def _manifest_records(catalog: Any) -> dict[str, dict[str, Any]]:
    manifest_path = Path(catalog.root).resolve() / "data" / "manifest.json"
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Scenario manifest is unreadable: {manifest_path}") from exc
    if not isinstance(document, dict):
        raise ValueError("Scenario manifest must be a JSON object")
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
    writable = path
    while not writable.exists() and writable != writable.parent:
        writable = writable.parent
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


def _sample_point_elevations(points: gpd.GeoDataFrame, dem: Any) -> np.ndarray:
    """Sample point elevations without reading a complete native DEM band."""

    if points.crs is None:
        raise ValueError("Point data requires a CRS before DEM sampling")
    if getattr(dem, "crs", None) is None:
        raise ValueError("DEM requires a CRS before elevation sampling")
    if points.empty:
        return np.asarray([], dtype=float)
    projected = points.to_crs(dem.crs) if points.crs != dem.crs else points
    coordinates = zip(
        projected.geometry.x.to_numpy(),
        projected.geometry.y.to_numpy(),
        strict=True,
    )
    nodata = getattr(dem, "nodata", None)
    elevations: list[float] = []
    for sample in dem.sample(coordinates, indexes=1, masked=True):
        value = sample[0]
        if np.ma.is_masked(value):
            elevations.append(np.nan)
            continue
        numeric = float(value)
        if (
            not np.isfinite(numeric)
            or (nodata is not None and np.isclose(numeric, nodata, equal_nan=True))
        ):
            elevations.append(np.nan)
            continue
        elevations.append(numeric)
    values = np.asarray(elevations, dtype=float)
    invalid_count = int((~np.isfinite(values)).sum())
    if invalid_count:
        raise ValueError(
            f"{invalid_count} selected station(s) have no valid DEM elevation"
        )
    return values


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(
        f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}"
    )


def _require_nonempty_regular_file(path: Path) -> Path:
    status = path.lstat()
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ValueError(f"Required output is not a regular file: {path}")
    if status.st_size <= 0:
        raise ValueError(f"Required output is empty: {path}")
    return path


def _promote_nonempty_file(temporary: Path, destination: Path) -> Path:
    _require_nonempty_regular_file(temporary)
    temporary.replace(destination)
    return destination


def _render_terrain_artifact(
    token: str,
    temporary: Path,
    *,
    rectangle: dict[str, Any],
    selected_points: gpd.GeoDataFrame,
    dem_path: Path,
    rectangle_size: float,
    target_crs: str,
    figure_spec: Any,
) -> None:
    """Render one independently requested terrain artifact with a full spec."""

    with rasterio.open(dem_path) as dem:
        visualization.render_3d_terrain(
            rectangle,
            selected_points,
            dem,
            figure_spec,
            rectangle_size=rectangle_size,
            target_crs=target_crs,
            png_path=temporary if token == "terrain_png" else None,
            eps_path=temporary if token == "terrain_eps" else None,
            html_path=temporary if token == "terrain_html" else None,
        )


def _profile_figure_spec(profile: ExperimentProfile) -> Any:
    from .figure_service import FigureSpec

    settings = profile.figure
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


class SelectionService:
    """Share one validated selection implementation between CLI and GUI."""

    def __init__(self, catalog: Any) -> None:
        self.catalog = catalog
        self.cache = CandidateCache(catalog.root)
        self._prepared: PreparedSelection | None = None

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
            validate_profile(profile)
        except ValueError as exc:
            raise SelectionPreflightError(
                "profile.invalid",
                str(exc),
            ) from exc
        try:
            _target_crs(profile.target_crs)
        except ValueError as exc:
            raise SelectionPreflightError(
                "profile.target_crs",
                str(exc),
                details={"target_crs": profile.target_crs},
            ) from exc
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
            report = validate_scenario_data(
                self.catalog,
                scenario_id,
                dataset_ids=(profile.points_dataset_id,),
            )
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
            boundary_dataset_id=boundary_id,
            dem_dataset_id=dem_id,
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

    def _prepare_selection(
        self,
        preflight: SelectionPreflight,
        *,
        refresh: bool,
    ) -> PreparedSelection:
        if not isinstance(preflight, SelectionPreflight):
            raise SelectionScanError(
                "scan.request",
                "preflight must be a SelectionPreflight",
            )
        if (
            not refresh
            and self._prepared is not None
            and self._prepared.preflight is preflight
        ):
            return self._prepared
        try:
            points = gpd.read_file(preflight.points_path)
            boundaries = gpd.read_file(preflight.boundary_path)
            selected, boundary, coordinates = prepare_spatial_data(
                points,
                boundaries,
                target_crs=preflight.profile.target_crs,
            )
        except Exception as exc:
            raise SelectionScanError(
                "scan.inputs",
                f"Cannot load selection vector inputs: {exc}",
                details={
                    "points_path": str(preflight.points_path),
                    "boundary_path": str(preflight.boundary_path),
                },
            ) from exc
        coordinates.setflags(write=False)
        prepared = PreparedSelection(
            preflight=preflight,
            points=selected,
            boundary=boundary,
            coordinates=coordinates,
        )
        self._prepared = prepared
        return prepared

    def prepared_selection(
        self,
        preflight: SelectionPreflight,
    ) -> PreparedSelection:
        """Return the one vector snapshot associated with this selection run."""

        return self._prepare_selection(preflight, refresh=False)

    def scan(
        self,
        preflight: SelectionPreflight,
        force: bool = False,
        progress: Any = None,
        cancel: Any = None,
    ) -> ScanResult:
        """Load or compute a completed candidate scan through the shared cache."""

        if not isinstance(preflight, SelectionPreflight):
            raise SelectionScanError(
                "scan.request",
                "preflight must be a SelectionPreflight",
            )
        try:
            cancelled = cancel is not None and cancel.is_set()
        except Exception as exc:
            raise SelectionScanError(
                "scan.request",
                "cancel must expose is_set()",
            ) from exc
        if cancelled:
            raise ScanCancelled("Candidate scan was cancelled")
        try:
            request = self._request(preflight.profile)
            key = cache_key(
                request,
                preflight.scenario_id,
                preflight.boundary_fingerprint,
                preflight.points_fingerprint,
                preflight.profile.target_crs,
            )
        except ValueError as exc:
            raise SelectionScanError(
                "scan.request",
                str(exc),
            ) from exc
        if not force:
            try:
                cached = self.cache.load(key, request)
            except (OSError, ValueError) as exc:
                raise SelectionScanError(
                    "scan.cache",
                    f"Candidate cache read failed: {exc}",
                    details={"cache_key": key},
                ) from exc
            if cached is not None:
                if cancel is not None and cancel.is_set():
                    raise ScanCancelled("Candidate scan was cancelled")
                if progress is not None:
                    progress(
                        SelectionProgress(
                            phase="completed",
                            checked_positions=cached.checked_positions,
                            total_positions=cached.total_positions,
                            candidate_count=len(cached.candidates),
                            elapsed_seconds=0.0,
                            added_candidates=cached.candidates,
                            removed_flat_grid_ids=(),
                            cache_status="hit",
                            cache_key=key,
                        )
                    )
                return cached
        prepared = self._prepare_selection(preflight, refresh=force)
        cache_status = "forced" if force else "miss"
        if progress is not None:
            progress(
                SelectionProgress(
                    phase="cache",
                    checked_positions=0,
                    total_positions=0,
                    candidate_count=0,
                    elapsed_seconds=0.0,
                    added_candidates=(),
                    removed_flat_grid_ids=(),
                    cache_status=cache_status,
                    cache_key=key,
                )
            )

        def forward_progress(event: ScanProgress) -> None:
            if progress is not None:
                progress(
                    SelectionProgress.from_scan(
                        event,
                        cache_status=cache_status,
                        cache_key=key,
                    )
                )

        try:
            result = scan_candidates(
                request,
                prepared.boundary,
                prepared.coordinates,
                progress=forward_progress if progress is not None else None,
                cancel=cancel,
            )
        except ScanCancelled:
            raise
        except (OSError, ValueError) as exc:
            raise SelectionScanError(
                "scan.failed",
                f"Candidate scan failed: {exc}",
                details={"cache_key": key},
            ) from exc
        if result.completed:
            try:
                self.cache.store(key, request, result)
            except (OSError, ValueError) as exc:
                raise SelectionScanError(
                    "scan.cache",
                    f"Candidate cache write failed: {exc}",
                    details={"cache_key": key},
                ) from exc
        return result

    @staticmethod
    def _export_artifacts(artifacts: Iterable[str]) -> tuple[str, ...]:
        if isinstance(artifacts, (str, bytes, os.PathLike)):
            raise SelectionExportError(
                "export.artifacts",
                "artifacts must be a collection of artifact tokens",
            )
        try:
            requested = list(artifacts)
        except TypeError as exc:
            raise SelectionExportError(
                "export.artifacts",
                "artifacts must be an iterable of artifact tokens",
            ) from exc
        if not requested:
            raise SelectionExportError(
                "export.artifacts",
                "At least one export artifact is required",
            )
        if any(type(item) is not str or item not in EXPORT_ARTIFACTS for item in requested):
            choices = ", ".join(_ARTIFACT_ORDER)
            raise SelectionExportError(
                "export.artifacts",
                f"Export artifacts must be selected from: {choices}",
            )
        selected = set(requested)
        if len(selected) != len(requested):
            raise SelectionExportError(
                "export.artifacts",
                "Export artifacts must not contain duplicates",
            )
        return tuple(token for token in _ARTIFACT_ORDER if token in selected)

    @staticmethod
    def _export_entrypoint(entrypoint: Iterable[str] | None) -> list[str]:
        if entrypoint is None:
            return []
        if isinstance(entrypoint, (str, bytes, os.PathLike)):
            raise SelectionExportError(
                "export.entrypoint",
                "entrypoint must be a sequence of strings",
            )
        try:
            command = list(entrypoint)
        except TypeError as exc:
            raise SelectionExportError(
                "export.entrypoint",
                "entrypoint must be a sequence of strings",
            ) from exc
        if any(type(item) is not str for item in command):
            raise SelectionExportError(
                "export.entrypoint",
                "entrypoint must contain only strings",
            )
        return command

    def _export_dataset_ids(
        self,
        preflight: SelectionPreflight,
    ) -> dict[str, str]:
        """Resolve catalog IDs, with stable path-stem IDs for manual fixtures."""

        scenario: dict[str, Any] = {}
        if (
            preflight.boundary_dataset_id is None
            or preflight.dem_dataset_id is None
        ):
            try:
                candidate = self.catalog.scenario(preflight.scenario_id)
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
            else:
                if isinstance(candidate, dict):
                    scenario = candidate
        declared = {
            "points": preflight.profile.points_dataset_id,
            "boundary": preflight.boundary_dataset_id
            or scenario.get("boundary_dataset_id"),
            "dem": preflight.dem_dataset_id or scenario.get("dem_dataset_id"),
        }
        paths = {
            "points": preflight.points_path,
            "boundary": preflight.boundary_path,
            "dem": preflight.dem_path,
        }
        return {
            role: (
                dataset_id
                if type(dataset_id) is str and dataset_id
                else Path(paths[role]).stem or role
            )
            for role, dataset_id in declared.items()
        }

    @staticmethod
    def _candidate_geometry(
        preflight: SelectionPreflight,
        candidate: Candidate,
    ) -> Any:
        return box(
            candidate.left_x,
            candidate.bottom_y,
            candidate.left_x + preflight.profile.rect_size,
            candidate.bottom_y + preflight.profile.rect_size,
        )

    @staticmethod
    def _selected_stations(
        prepared: PreparedSelection,
        candidate: Candidate,
    ) -> gpd.GeoDataFrame:
        size = prepared.preflight.profile.rect_size
        maximum_x = candidate.left_x + size
        maximum_y = candidate.bottom_y + size
        points = prepared.points
        mask = (
            (points.geometry.x >= candidate.left_x)
            & (points.geometry.x <= maximum_x)
            & (points.geometry.y >= candidate.bottom_y)
            & (points.geometry.y <= maximum_y)
        )
        selected = points[mask].copy().reset_index(drop=True)
        if len(selected) != candidate.point_count:
            raise SelectionExportError(
                "export.station_count",
                "Selected station count does not match the completed scan candidate",
                details={
                    "candidate_point_count": candidate.point_count,
                    "selected_station_count": len(selected),
                },
            )
        return selected

    @staticmethod
    def _render_candidate(
        preflight: SelectionPreflight,
        candidate: Candidate,
        geometry: Any,
    ) -> dict[str, Any]:
        return {
            "geometry": geometry,
            "flat_grid_id": candidate.flat_grid_id,
            "pt_count": candidate.point_count,
            "left_x": candidate.left_x,
            "bottom_y": candidate.bottom_y,
            "center_x": candidate.center_x,
            "center_y": candidate.center_y,
            "rect_size": preflight.profile.rect_size,
        }

    def _authoritative_candidate(
        self,
        preflight: SelectionPreflight,
        scan_result: ScanResult,
        candidate: Candidate,
    ) -> tuple[int, Candidate, ScanRequest]:
        if not isinstance(preflight, SelectionPreflight):
            raise SelectionExportError(
                "export.request",
                "preflight must be a SelectionPreflight",
            )
        if not isinstance(scan_result, ScanResult) or scan_result.completed is not True:
            raise SelectionExportError(
                "export.scan",
                "export requires a completed ScanResult",
            )
        if not isinstance(candidate, Candidate):
            raise SelectionExportError(
                "export.candidate",
                "candidate must be a Candidate from the completed scan",
            )
        matches = [
            (index, stored)
            for index, stored in enumerate(scan_result.candidates)
            if stored == candidate
        ]
        if len(matches) != 1:
            raise SelectionExportError(
                "export.candidate",
                "candidate must match exactly one completed scan candidate",
            )
        try:
            request = self._request(preflight.profile)
        except ValueError as exc:
            raise SelectionExportError("export.scan", str(exc)) from exc
        if scan_result.algorithm_version != request.algorithm_version:
            raise SelectionExportError(
                "export.scan",
                "scan_result algorithm_version does not match the profile request",
            )
        index, authoritative = matches[0]
        return index, authoritative, request

    def prepare_figure_source(
        self,
        preflight: SelectionPreflight,
        scan_result: ScanResult,
        candidate: Candidate,
    ) -> Any:
        """Prepare a traceable in-memory figure source without publishing a run."""

        from .figure_service import FigureSource, SelectionFigureIdentity

        candidate_index, candidate, _request = self._authoritative_candidate(
            preflight,
            scan_result,
            candidate,
        )
        try:
            prepared = self.prepared_selection(preflight)
            selected = self._selected_stations(prepared, candidate)
        except SelectionExportError:
            raise
        except Exception as exc:
            raise SelectionExportError(
                "export.inputs",
                f"Cannot prepare selected-station inputs: {exc}",
            ) from exc
        try:
            with rasterio.open(preflight.dem_path) as dem:
                selected["elevation"] = _sample_point_elevations(selected, dem)
        except Exception as exc:
            raise SelectionExportError(
                "export.elevation",
                f"Cannot sample selected-station elevations: {exc}",
            ) from exc

        candidate_id = f"candidate-{candidate_index + 1:04d}"
        frame = io.build_output_dataframe(
            selected,
            selected.crs,
            rect_id=1,
            pt_count=candidate.point_count,
            left_x=candidate.left_x,
            bottom_y=candidate.bottom_y,
            center_x=candidate.center_x,
            center_y=candidate.center_y,
            scenario_id=preflight.scenario_id,
            profile_id=preflight.profile.profile_id,
            candidate_id=candidate_id,
            target_crs=preflight.profile.target_crs,
        )
        rectangle = {
            "rect_id": 1,
            "pt_count": candidate.point_count,
            "left_x": candidate.left_x,
            "bottom_y": candidate.bottom_y,
            "center_x": candidate.center_x,
            "center_y": candidate.center_y,
        }
        identity = SelectionFigureIdentity(
            scenario_id=preflight.scenario_id,
            profile_id=preflight.profile.profile_id,
            profile_fingerprint=_canonical_fingerprint(
                _identity_value(asdict(preflight.profile))
            ),
            points_fingerprint=preflight.points_fingerprint,
            boundary_fingerprint=preflight.boundary_fingerprint,
            dem_fingerprint=preflight.dem_fingerprint,
            scan_algorithm_version=scan_result.algorithm_version,
            scan_checked_positions=scan_result.checked_positions,
            scan_total_positions=scan_result.total_positions,
            candidate_index=candidate_index + 1,
            candidate_flat_grid_id=candidate.flat_grid_id,
            candidate_point_count=candidate.point_count,
            candidate_left_x=candidate.left_x,
            candidate_bottom_y=candidate.bottom_y,
            candidate_center_x=candidate.center_x,
            candidate_center_y=candidate.center_y,
        )
        return FigureSource(
            path=None,
            csv_path=None,
            frame=frame,
            rectangle=rectangle,
            points=selected.copy(deep=True),
            target_crs=preflight.profile.target_crs,
            rectangle_size_m=float(preflight.profile.rect_size),
            source_kind="selection",
            dem_path=preflight.dem_path.resolve(strict=True),
            scenario_id=preflight.scenario_id,
            profile_id=preflight.profile.profile_id,
            dem_fingerprint=preflight.dem_fingerprint,
            selection_identity=identity,
        )

    def export(
        self,
        preflight: SelectionPreflight,
        scan_result: ScanResult,
        candidate: Candidate,
        *,
        output_root: str | os.PathLike[str],
        artifacts: Iterable[str],
        entrypoint: Iterable[str] | None = None,
    ) -> Path:
        """Publish one completed-scan candidate through an atomic run staging."""

        candidate_index, candidate, request = self._authoritative_candidate(
            preflight,
            scan_result,
            candidate,
        )
        dataset_ids = self._export_dataset_ids(preflight)
        requested = self._export_artifacts(artifacts)
        command = self._export_entrypoint(entrypoint)
        try:
            resolved_output = _resolved_output_root(self.catalog, output_root)
        except ValueError as exc:
            raise SelectionExportError("export.output_root", str(exc)) from exc
        if resolved_output != preflight.output_root.resolve():
            raise SelectionExportError(
                "export.output_root",
                "output_root must match the validated preflight output root",
            )
        candidate_id = f"candidate-{candidate_index + 1:04d}"
        try:
            key = cache_key(
                request,
                preflight.scenario_id,
                preflight.boundary_fingerprint,
                preflight.points_fingerprint,
                preflight.profile.target_crs,
            )
        except ValueError as exc:
            raise SelectionExportError("export.scan", str(exc)) from exc
        try:
            prepared = self.prepared_selection(preflight)
            geometry = self._candidate_geometry(preflight, candidate)
            selected = self._selected_stations(prepared, candidate)
        except SelectionExportError:
            raise
        except Exception as exc:
            raise SelectionExportError(
                "export.inputs",
                f"Cannot prepare selected-station inputs: {exc}",
            ) from exc
        try:
            with rasterio.open(preflight.dem_path) as dem:
                statistics = stream_dem_statistics(
                    dem,
                    geometry,
                    geometry_crs=preflight.profile.target_crs,
                )
                selected["elevation"] = _sample_point_elevations(selected, dem)
        except SelectionExportError:
            raise
        except Exception as exc:
            raise SelectionExportError(
                "export.elevation",
                f"Cannot sample selected-station elevations: {exc}",
            ) from exc

        figure_spec = _profile_figure_spec(preflight.profile)
        service = RunService(resolved_output)
        run = None
        publication_started = False
        published_files: tuple[str, ...] = ()
        try:
            run = service.begin(
                preflight.scenario_id,
                preflight.profile.profile_id,
            )
            snapshot = replace(
                preflight.profile,
                output_root=resolved_output,
            )
            run_config_path = run.path / "run-config.yaml"
            dump_profile(snapshot, run_config_path)
            _require_nonempty_regular_file(run_config_path)
            frame = io.build_output_dataframe(
                selected,
                selected.crs,
                rect_id=1,
                pt_count=candidate.point_count,
                left_x=candidate.left_x,
                bottom_y=candidate.bottom_y,
                center_x=candidate.center_x,
                center_y=candidate.center_y,
                run_id=run.run_id,
                scenario_id=preflight.scenario_id,
                profile_id=preflight.profile.profile_id,
                candidate_id=candidate_id,
                target_crs=preflight.profile.target_crs,
            )
            selection_document = {
                "run_id": run.run_id,
                "scenario_id": preflight.scenario_id,
                "profile_id": preflight.profile.profile_id,
                "candidate_id": candidate_id,
                "target_crs": preflight.profile.target_crs,
                "scan": {
                    "algorithm_version": scan_result.algorithm_version,
                    "cache_key": key,
                    "checked_positions": scan_result.checked_positions,
                    "total_positions": scan_result.total_positions,
                    "completed": scan_result.completed,
                },
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "candidate_index": candidate_index + 1,
                        **asdict(candidate),
                        "bounds": list(geometry.bounds),
                        "geometry": mapping(geometry),
                        "dem_statistics": asdict(statistics),
                        "selected_station_id_field": "cell",
                        "selected_station_ids": frame["cell"].tolist(),
                    }
                ],
            }
            selection_path = run.path / "selection.json"
            io.atomic_write_json(selection_path, selection_document)
            _require_nonempty_regular_file(selection_path)

            size = preflight.profile.rect_size
            base = (
                f"{preflight.scenario_id}_{size}m_"
                f"target{preflight.profile.target_count}_tol{preflight.profile.tolerance}"
            )
            names = {
                "csv": f"{base}.csv",
                "preview_png": f"{base}.png",
                "terrain_png": f"{base}_3d.png",
                "terrain_eps": f"{base}_3d.eps",
                "terrain_html": f"{base}_3d.html",
            }
            artifact_paths: dict[str, str] = {}
            errors: list[dict[str, str]] = []
            render_candidate = self._render_candidate(
                preflight,
                candidate,
                geometry,
            )
            for token in requested:
                destination = run.path / names[token]
                temporary = _temporary_sibling(destination)
                try:
                    if token == "csv":
                        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
                    elif token == "preview_png":
                        visualization.save_preview(
                            prepared.points,
                            prepared.boundary,
                            [render_candidate],
                            {
                                "rect_size": size,
                                "boundary_layer": preflight.scenario_id,
                                "preview_png": temporary,
                            },
                        )
                    else:
                        _render_terrain_artifact(
                            token,
                            temporary,
                            rectangle=render_candidate,
                            selected_points=selected,
                            dem_path=preflight.dem_path,
                            rectangle_size=size,
                            target_crs=preflight.profile.target_crs,
                            figure_spec=figure_spec,
                        )
                    _promote_nonempty_file(temporary, destination)
                    artifact_paths[token] = names[token]
                except Exception as exc:
                    errors.append(
                        {
                            "artifact": token,
                            "code": f"artifact.{token}.failed",
                            "message": f"{type(exc).__name__}: {exc}",
                        }
                    )
                finally:
                    temporary.unlink(missing_ok=True)
            if not artifact_paths:
                detail = "; ".join(
                    f"{item.get('code')}: {item.get('message')}" for item in errors
                )
                raise SelectionExportError(
                    "export.no_artifacts",
                    "No requested export artifact was published"
                    + (f": {detail}" if detail else ""),
                    details={"errors": errors},
                )
            _require_nonempty_regular_file(run_config_path)
            _require_nonempty_regular_file(selection_path)
            profile = preflight.profile
            metadata = {
                "run_kind": "selection",
                "candidate": {
                    "candidate_id": candidate_id,
                    "flat_grid_id": candidate.flat_grid_id,
                    "point_count": candidate.point_count,
                    "left_x": candidate.left_x,
                    "bottom_y": candidate.bottom_y,
                    "center_x": candidate.center_x,
                    "center_y": candidate.center_y,
                },
                "scanner": {
                    "algorithm_version": scan_result.algorithm_version,
                    "checked_positions": scan_result.checked_positions,
                    "total_positions": scan_result.total_positions,
                },
                "cache": {
                    "key": key,
                },
                "inputs": {
                    "points": {
                        "dataset_id": dataset_ids["points"],
                        "fingerprint": preflight.points_fingerprint,
                    },
                    "boundary": {
                        "dataset_id": dataset_ids["boundary"],
                        "fingerprint": preflight.boundary_fingerprint,
                    },
                    "dem": {
                        "dataset_id": dataset_ids["dem"],
                        "fingerprint": preflight.dem_fingerprint,
                        "path": str(preflight.dem_path.resolve()),
                    },
                },
                "parameters": {
                    "target_crs": profile.target_crs,
                    "rectangle_size_m": profile.rect_size,
                    "target_base_station_count": profile.target_count,
                    "count_tolerance": profile.tolerance,
                    "scan_mode": profile.scan_mode,
                    "strategy": profile.strategy,
                    "step_m": profile.scan_step,
                    "max_candidates": profile.max_rects,
                    "minimum_center_spacing_m": profile.min_spacing,
                    "random_seed": profile.random_seed,
                },
                "sidecars": {
                    "config": "run-config.yaml",
                    "selection": "selection.json",
                },
                "requested_artifacts": list(requested),
                "artifact_paths": dict(artifact_paths),
                "figure_spec": figure_spec.as_dict(),
                "git_commit": io._git_commit(Path(self.catalog.root).resolve()),
                "software_versions": io.software_versions(),
                "entrypoint": command,
            }
            status = "completed" if not errors else "partial"
            published_files = tuple(artifact_paths.values())
            publication_started = True
            return service.publish(
                run,
                status=status,
                artifacts=published_files,
                metadata=metadata,
                errors=errors,
            )
        except Exception as exc:
            retain_staging = False
            if (
                run is not None
                and publication_started
                and run.path.is_dir()
            ):
                try:
                    for relative in (
                        "run.json",
                        "run-config.yaml",
                        "selection.json",
                        *published_files,
                    ):
                        _require_nonempty_regular_file(run.path / relative)
                except (OSError, ValueError):
                    pass
                else:
                    retain_staging = True
                    exc.add_note(
                        f"Complete run staging retained for recovery: {run.path}"
                    )
            if run is not None and run.path.is_dir() and not retain_staging:
                try:
                    service.abandon(run)
                except Exception as cleanup_error:
                    exc.add_note(f"Run staging cleanup also failed: {cleanup_error}")
            if isinstance(exc, SelectionExportError):
                raise
            raise SelectionExportError(
                "export.failed",
                f"Selected candidate export failed: {exc}",
            ) from exc

    def candidate_statistics(
        self,
        preflight: SelectionPreflight,
        candidate: Candidate,
    ) -> DemStatistics:
        """Compute exact native-resolution DEM statistics for one candidate."""

        if not isinstance(preflight, SelectionPreflight):
            raise SelectionStatisticsError(
                "statistics.request",
                "preflight must be a SelectionPreflight",
            )
        if not isinstance(candidate, Candidate):
            raise SelectionStatisticsError(
                "statistics.request",
                "candidate must be a Candidate",
            )
        geometry = box(
            candidate.left_x,
            candidate.bottom_y,
            candidate.left_x + preflight.profile.rect_size,
            candidate.bottom_y + preflight.profile.rect_size,
        )
        try:
            with rasterio.open(preflight.dem_path) as dem:
                return stream_dem_statistics(
                    dem,
                    geometry,
                    geometry_crs=preflight.profile.target_crs,
                )
        except Exception as exc:
            raise SelectionStatisticsError(
                "statistics.failed",
                f"Cannot compute candidate DEM statistics: {exc}",
                details={
                    "candidate_flat_grid_id": candidate.flat_grid_id,
                    "dem_path": str(preflight.dem_path),
                },
            ) from exc


__all__ = [
    "DemStatistics",
    "EXPORT_ARTIFACTS",
    "PreparedSelection",
    "SCANNER_ALGORITHM_VERSION",
    "SelectionError",
    "SelectionExportError",
    "SelectionPreflight",
    "SelectionPreflightError",
    "SelectionProgress",
    "SelectionScanError",
    "SelectionService",
    "SelectionStatisticsError",
    "stream_dem_statistics",
]
