"""Fast and full integrity checks for registered LTE scenario data."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
from pyproj import CRS

from .config import load_experiment_config
from .data_catalog import DataCatalog
from .dem_data import validate_dem_coverage
from .io import sha256_file
from .spatial import resolve_io_paths

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_POLYGON_TYPES = frozenset({"Polygon", "MultiPolygon"})
_SHAPEFILE_COMPONENTS = (".shp", ".shx", ".dbf", ".prj", ".cpg")


def _canonical_value(value: Any, *, key: str | None = None) -> Any:
    """Return a JSON-safe value with stable path/date representations."""

    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, dict):
        return {
            str(item_key): _canonical_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item, key=key) for item in value]
    if isinstance(value, str) and key in {"path", "entrypoint", "config_path"}:
        return value.replace("\\", "/")
    return value


def _canonical_json(value: Any) -> str:
    """Serialize metadata for exact, order-independent comparison."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class ValidationMessage:
    """One normalized validation diagnostic."""

    level: str
    code: str
    message: str


@dataclass
class ValidationReport:
    """Validation diagnostics for one scenario."""

    scenario_id: str
    status: str
    messages: list[ValidationMessage] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return true when no error-level diagnostics were recorded."""

        return not any(message.level.casefold() == "error" for message in self.messages)

    def add(
        self,
        level: str | ValidationMessage,
        code: str | None = None,
        message: str | None = None,
    ) -> ValidationMessage:
        """Append a diagnostic and return the normalized message."""

        if isinstance(level, ValidationMessage):
            if code is not None or message is not None:
                raise TypeError("code and message are not accepted with ValidationMessage")
            item = level
        else:
            if code is None or message is None:
                raise TypeError("add requires level, code, and message")
            item = ValidationMessage(str(level).casefold(), str(code), str(message))
        self.messages.append(item)
        return item


def _error(report: ValidationReport, code: str, message: str) -> None:
    report.add("error", code, message)


def _warning(report: ValidationReport, code: str, message: str) -> None:
    report.add("warning", code, message)


def _safe_manifest_path(
    root: Path,
    raw_path: Any,
    *,
    dataset_root: Path | None = None,
) -> tuple[Path | None, str | None]:
    """Resolve one manifest path while rejecting absolute/traversal paths."""

    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, "manifest file path must be a non-empty string"
    relative = Path(raw_path)
    # Checking parts explicitly catches traversal even when the target does not
    # exist yet (and keeps the diagnostic stable across Windows/Linux).
    if (
        relative.is_absolute()
        or relative.drive
        or re.match(r"^[A-Za-z]:[\\/]", raw_path) is not None
        or ".." in relative.parts
    ):
        return None, f"manifest file path escapes repository root: {raw_path}"
    repository = root.resolve()
    resolved = (repository / relative).resolve()
    try:
        resolved.relative_to(repository)
    except ValueError:
        return None, f"manifest file path escapes repository root: {raw_path}"
    if dataset_root is not None:
        try:
            resolved.relative_to(dataset_root.resolve())
        except ValueError:
            return None, f"manifest file path is outside its dataset: {raw_path}"
    return resolved, None


def _boundary_sidecars(entrypoint: Path) -> set[str]:
    """Return lower-case component suffixes found beside a Shapefile."""

    found: set[str] = set()
    try:
        siblings = tuple(entrypoint.parent.iterdir())
    except OSError:
        return found
    stem = entrypoint.stem.casefold()
    for sibling in siblings:
        if not sibling.is_file() or sibling.stem.casefold() != stem:
            continue
        suffix = sibling.suffix.casefold()
        if suffix in _SHAPEFILE_COMPONENTS:
            found.add(suffix)
    # The catalog entrypoint is exact: a differently cased .SHP is not the
    # registered path even though sidecar suffixes are intentionally tolerant.
    if entrypoint.is_file() and entrypoint.suffix.casefold() == ".shp":
        found.add(".shp")
    return found


def _validate_boundary(
    catalog: DataCatalog,
    boundary: dict[str, Any],
    report: ValidationReport,
) -> Path | None:
    """Validate the registered boundary entrypoint and its vector contract."""

    try:
        entrypoint = catalog.resolve(boundary.get("entrypoint", ""))
    except Exception as exc:
        _error(report, "boundary.entrypoint", f"Cannot resolve boundary entrypoint: {exc}")
        return None

    if not entrypoint.is_file():
        _error(report, "boundary.missing", f"Boundary entrypoint does not exist: {entrypoint}")
        return None
    if entrypoint.suffix.casefold() != ".shp":
        _error(report, "boundary.entrypoint", f"Boundary entrypoint is not a Shapefile: {entrypoint}")

    found_components = _boundary_sidecars(entrypoint)
    missing_components = [
        component
        for component in _SHAPEFILE_COMPONENTS
        if component not in found_components
    ]
    if missing_components:
        _error(
            report,
            "boundary.sidecar",
            "Boundary Shapefile is missing required components: "
            + ", ".join(missing_components),
        )

    try:
        frame = gpd.read_file(entrypoint)
    except Exception as exc:
        _error(report, "boundary.read", f"Cannot read boundary Shapefile {entrypoint}: {exc}")
        return entrypoint

    frame_crs = getattr(frame, "crs", None)
    expected_crs = boundary.get("crs")
    if frame_crs is None:
        _error(report, "boundary.crs", "Boundary Shapefile does not declare a CRS")
    elif not expected_crs:
        _error(report, "boundary.crs", "Boundary registry does not declare a CRS")
    else:
        try:
            if CRS.from_user_input(frame_crs) != CRS.from_user_input(expected_crs):
                _error(
                    report,
                    "boundary.crs",
                    f"Boundary CRS does not match registry ({frame_crs} != {expected_crs})",
                )
        except Exception as exc:
            _error(report, "boundary.crs", f"Cannot compare boundary CRS: {exc}")

    if frame.empty:
        _error(report, "boundary.empty", "Boundary Shapefile contains no features")
    expected_count = boundary.get("feature_count")
    if isinstance(expected_count, int) and not isinstance(expected_count, bool):
        if len(frame) != expected_count:
            _error(
                report,
                "boundary.count",
                f"Boundary feature count does not match registry ({len(frame)} != {expected_count})",
            )
    else:
        _error(report, "boundary.count", "Boundary registry feature_count is invalid")

    geometry = getattr(frame, "geometry", None)
    if geometry is None:
        _error(report, "boundary.geometry", "Boundary Shapefile has no geometry column")
        return entrypoint
    try:
        if geometry.isna().any() or geometry.is_empty.any():
            _error(report, "boundary.geometry.empty", "Boundary geometries must be nonempty")
    except Exception as exc:
        _error(report, "boundary.geometry", f"Cannot inspect boundary geometries: {exc}")
    try:
        geometry_types = {
            str(value) for value in geometry.geom_type.dropna().unique().tolist()
        }
        if not geometry_types <= _POLYGON_TYPES:
            found = ", ".join(sorted(geometry_types)) or "<none>"
            _error(
                report,
                "boundary.geometry_type",
                f"Boundary geometries must be Polygon or MultiPolygon; found: {found}",
            )
    except Exception as exc:
        _error(report, "boundary.geometry_type", f"Cannot inspect boundary geometry types: {exc}")
    try:
        if not geometry.is_valid.all():
            _error(report, "boundary.geometry.invalid", "Boundary geometries must be valid")
    except Exception as exc:
        _error(report, "boundary.geometry.invalid", f"Cannot validate boundary geometries: {exc}")
    return entrypoint


def _manifest_file(
    catalog: DataCatalog,
    dataset: dict[str, Any],
    item: Any,
    index: int,
    report: ValidationReport,
    *,
    full_checksum: bool,
) -> None:
    """Validate one manifest file record without loading file contents."""

    if not isinstance(item, dict):
        return
    raw_path = item.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return
    dataset_root: Path | None
    try:
        dataset_root = catalog.resolve(dataset.get("path", ""))
    except Exception:
        dataset_root = None
    path, path_error = _safe_manifest_path(
        catalog.root,
        raw_path,
        dataset_root=dataset_root,
    )
    if path_error:
        code = (
            "manifest.traversal"
            if "escapes" in path_error or "outside" in path_error
            else "manifest.file.malformed"
        )
        _error(report, code, path_error)
        return
    size = item.get("size_bytes")
    digest = item.get("sha256")
    if type(size) is not int or size < 0:
        return
    if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
        return
    if path is None:
        return
    if not path.is_file():
        _error(report, "manifest.missing", f"Manifest file does not exist: {raw_path}")
        return
    actual_size = path.stat().st_size
    if actual_size != size:
        _error(
            report,
            "manifest.size",
            f"Manifest size does not match {raw_path} ({actual_size} != {size})",
        )
    if full_checksum:
        try:
            actual_digest = sha256_file(path)
        except Exception as exc:
            _error(report, "manifest.sha256", f"Cannot hash manifest file {raw_path}: {exc}")
        else:
            if actual_digest.casefold() != digest.casefold():
                _error(
                    report,
                    "manifest.sha256",
                    f"Manifest SHA256 does not match {raw_path}",
                )


def _manifest_file_shape(
    root: Path,
    item: Any,
    index: int,
    report: ValidationReport,
    *,
    dataset_root: Path | None = None,
) -> bool:
    """Check one file record's structure without touching the target file."""

    if not isinstance(item, dict):
        _error(report, "manifest.file.malformed", f"Manifest file entry {index} is not a mapping")
        return False
    raw_path = item.get("path")
    path, path_error = _safe_manifest_path(
        root,
        raw_path,
        dataset_root=dataset_root,
    )
    if path_error:
        code = (
            "manifest.traversal"
            if "escapes" in path_error or "outside" in path_error
            else "manifest.file.malformed"
        )
        _error(report, code, path_error)
        return False
    del path
    valid = True
    size = item.get("size_bytes")
    if type(size) is not int or size < 0:
        _error(report, "manifest.file.malformed", f"Manifest size_bytes is invalid for {raw_path!r}")
        valid = False
    digest = item.get("sha256")
    if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
        _error(report, "manifest.file.malformed", f"Manifest sha256 is invalid for {raw_path!r}")
        valid = False
    return valid


def _manifest_record_shape(
    catalog: DataCatalog,
    dataset: dict[str, Any] | None,
    record: Any,
    report: ValidationReport,
) -> bool:
    """Validate record structure globally without stat/hash operations."""

    if not isinstance(record, dict):
        _error(report, "manifest.dataset.malformed", "Manifest dataset entry is not a mapping")
        return False
    valid = True
    for field_name in ("path", "entrypoint"):
        value = record.get(field_name)
        _, path_error = _safe_manifest_path(catalog.root, value)
        if path_error:
            code = (
                "manifest.traversal"
                if "escapes" in path_error or "outside" in path_error
                else "manifest.dataset.malformed"
            )
            _error(report, code, f"Manifest {field_name} is invalid: {path_error}")
            valid = False
    files = record.get("files")
    if not isinstance(files, list):
        _error(report, "manifest.files", "Manifest dataset files must be a list")
        return False
    dataset_root: Path | None = None
    if dataset is not None:
        try:
            dataset_root = catalog.resolve(dataset.get("path", ""))
        except Exception as exc:
            _error(report, "manifest.metadata", f"Cannot resolve dataset root: {exc}")
            return False
    seen_paths: set[str] = set()
    for index, item in enumerate(files):
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            key = item["path"]
            if key in seen_paths:
                _error(report, "manifest.duplicate_file", f"Manifest repeats file path: {key}")
                valid = False
            seen_paths.add(key)
        if not _manifest_file_shape(
            catalog.root,
            item,
            index,
            report,
            dataset_root=dataset_root,
        ):
            valid = False
    return valid


def _manifest_record(
    catalog: DataCatalog,
    dataset: dict[str, Any],
    record: Any,
    report: ValidationReport,
    *,
    full_checksum: bool,
    require_entrypoint: bool,
    shape_valid: bool,
) -> None:
    if not isinstance(record, dict):
        return
    record_metadata = {key: value for key, value in record.items() if key != "files"}
    dataset_metadata = {key: value for key, value in dataset.items() if key != "files"}
    if _canonical_json(record_metadata) != _canonical_json(dataset_metadata):
        _error(
            report,
            "manifest.metadata",
            f"Manifest metadata for {dataset.get('dataset_id')} does not match catalog",
        )
    files = record.get("files")
    if not isinstance(files, list):
        return
    if not shape_valid:
        return
    seen_paths: set[str] = set()
    for index, item in enumerate(files):
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            key = item["path"]
            if key in seen_paths:
                _error(report, "manifest.duplicate_file", f"Manifest repeats file path: {key}")
            seen_paths.add(key)
        _manifest_file(
            catalog,
            dataset,
            item,
            index,
            report,
            full_checksum=full_checksum,
        )
    if require_entrypoint:
        expected = _canonical_value(dataset.get("entrypoint"), key="entrypoint")
        represented = any(
            isinstance(item, dict)
            and _canonical_value(item.get("path"), key="entrypoint") == expected
            for item in files
        )
        if not represented:
            _error(
                report,
                "manifest.entrypoint",
                f"Manifest files for {dataset.get('dataset_id')} omit the registered entrypoint",
            )


def _validate_manifest(
    catalog: DataCatalog,
    boundary: dict[str, Any],
    dem: dict[str, Any] | None,
    report: ValidationReport,
    *,
    full_checksum: bool,
) -> dict[str, dict[str, Any]]:
    """Read and validate the schema-v2 manifest, returning indexed records."""

    path = catalog.root / "data" / "manifest.json"
    if not path.is_file():
        _error(report, "manifest.missing", f"Manifest does not exist: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _error(report, "manifest.parse", f"Cannot parse manifest {path}: {exc}")
        return {}
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 2
    ):
        _error(report, "manifest.schema", "Manifest schema_version must be 2")
        return {}
    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, list):
        _error(report, "manifest.datasets", "Manifest datasets must be a list")
        return {}

    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, list):
        _error(report, "manifest.scenarios", "Manifest scenarios must be a list")
    elif _canonical_json(raw_scenarios) != _canonical_json(catalog.document.get("scenarios", [])):
        _error(report, "manifest.scenarios", "Manifest scenarios do not match catalog")

    records: dict[str, dict[str, Any]] = {}
    shape_valid: dict[str, bool] = {}
    for index, raw_record in enumerate(raw_datasets):
        if not isinstance(raw_record, dict):
            _error(report, "manifest.dataset.malformed", f"Manifest dataset entry {index} is not a mapping")
            continue
        dataset_id = raw_record.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            _error(report, "manifest.dataset.malformed", f"Manifest dataset entry {index} has no dataset_id")
            continue
        if dataset_id in records:
            _error(report, "manifest.duplicate", f"Manifest has duplicate dataset ID: {dataset_id}")
            continue
        records[dataset_id] = raw_record
        dataset = catalog.datasets_by_id.get(dataset_id)
        shape_valid[dataset_id] = _manifest_record_shape(catalog, dataset, raw_record, report)
        if dataset is None:
            _error(report, "manifest.unknown", f"Manifest references unknown dataset: {dataset_id}")

    boundary_id = boundary.get("dataset_id")
    boundary_record = records.get(boundary_id)
    if boundary_record is None:
        _error(report, "manifest.boundary", f"Manifest is missing boundary dataset: {boundary_id}")
    else:
        boundary_path = catalog.resolve(boundary.get("entrypoint", ""))
        _manifest_record(
            catalog,
            boundary,
            boundary_record,
            report,
            full_checksum=full_checksum,
            require_entrypoint=boundary_path.is_file(),
            shape_valid=shape_valid.get(boundary_id, False),
        )

    if dem is not None:
        dem_id = dem.get("dataset_id")
        dem_record = records.get(dem_id)
        if dem_record is None:
            _error(report, "manifest.dem", f"Manifest is missing DEM dataset: {dem_id}")
        else:
            dem_path = catalog.resolve(dem.get("entrypoint", ""))
            # An absent external DEM may have a cached record from a previous
            # ingest.  Its metadata/files are intentionally not checked until
            # the registered entrypoint exists locally.
            if dem_path.is_file():
                _manifest_record(
                    catalog,
                    dem,
                    dem_record,
                    report,
                    full_checksum=full_checksum,
                    require_entrypoint=True,
                    shape_valid=shape_valid.get(dem_id, False),
                )
    return records


def _validate_dem(
    catalog: DataCatalog,
    scenario: dict[str, Any],
    boundary_path: Path | None,
    report: ValidationReport,
) -> Path | None:
    dem_id = scenario.get("dem_dataset_id")
    if dem_id is None:
        _warning(report, "dem.undeclared", "Scenario does not declare a DEM dataset")
        return None
    try:
        dem = catalog.dataset(dem_id)
        dem_path = catalog.resolve(dem.get("entrypoint", ""))
    except Exception as exc:
        _warning(report, "dem.pending", f"Declared DEM is not available: {exc}")
        return None
    if not dem_path.is_file():
        _warning(report, "dem.pending", f"Declared DEM is not available: {dem_path}")
        return dem_path
    if boundary_path is None or not boundary_path.is_file():
        _error(report, "dem.coverage", "Cannot validate DEM coverage without a valid boundary")
        return dem_path
    expected_crs = dem.get("export_crs") or dem.get("crs")
    expected_resolution = dem.get("native_scale_m")
    if not expected_crs or expected_resolution is None:
        _error(report, "dem.metadata", "DEM registry is missing expected CRS or native scale")
        return dem_path
    try:
        validate_dem_coverage(
            dem_path,
            boundary_path,
            expected_crs=str(expected_crs),
            expected_resolution=expected_resolution,
        )
    except Exception as exc:
        _error(report, "dem.coverage", f"DEM coverage validation failed: {exc}")
    return dem_path


def _validate_config(
    catalog: DataCatalog,
    scenario: dict[str, Any],
    boundary: dict[str, Any],
    dem: dict[str, Any] | None,
    report: ValidationReport,
) -> None:
    config_path = scenario.get("config_path")
    if config_path is None:
        return
    try:
        resolved_config_path = catalog.resolve(config_path)
        config = load_experiment_config(resolved_config_path, repo_root=catalog.root)
        paths = resolve_io_paths(config, create_output=False)
    except Exception as exc:
        _error(report, "config.invalid", f"Cannot resolve linked experiment config: {exc}")
        return

    try:
        expected_boundary = catalog.resolve(boundary.get("entrypoint", "")).resolve()
        actual_boundary = Path(paths["boundary_shp"]).resolve()
        if actual_boundary != expected_boundary:
            _error(
                report,
                "config.boundary",
                f"Config boundary does not match registry ({actual_boundary} != {expected_boundary})",
            )
    except Exception as exc:
        _error(report, "config.boundary", f"Cannot compare config boundary: {exc}")

    try:
        actual_dem = Path(paths["dem_path"]).resolve()
        if dem is None:
            _error(report, "config.dem", "Config declares a DEM but scenario does not")
        else:
            expected_dem = catalog.resolve(dem.get("entrypoint", "")).resolve()
            if actual_dem != expected_dem:
                _error(
                    report,
                    "config.dem",
                    f"Config DEM does not match registry ({actual_dem} != {expected_dem})",
                )
    except Exception as exc:
        _error(report, "config.dem", f"Cannot compare config DEM: {exc}")


def validate_scenario_data(
    catalog: DataCatalog,
    scenario_id: str,
    *,
    full_checksum: bool = False,
) -> ValidationReport:
    """Validate one registered scenario in fast or full-checksum mode."""

    scenario = catalog.scenario(scenario_id)
    status = catalog.scenario_status(scenario_id)
    report = ValidationReport(scenario_id=scenario_id, status=status)
    boundary = catalog.dataset(scenario["boundary_dataset_id"])
    dem_id = scenario.get("dem_dataset_id")
    dem = None if dem_id is None else catalog.dataset(dem_id)
    boundary_path = _validate_boundary(catalog, boundary, report)
    _validate_manifest(
        catalog,
        boundary,
        dem,
        report,
        full_checksum=full_checksum,
    )
    _validate_dem(catalog, scenario, boundary_path, report)
    _validate_config(catalog, scenario, boundary, dem, report)
    return report


__all__ = [
    "ValidationMessage",
    "ValidationReport",
    "validate_scenario_data",
]
