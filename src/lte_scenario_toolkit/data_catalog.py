"""Validated dataset catalogs and reproducible checksum manifests."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .io import sha256_file

DATASET_BASE_FIELDS = frozenset(
    {
        "dataset_id",
        "role",
        "path",
        "entrypoint",
        "source_url",
        "provider",
        "license",
        "download_date",
        "crs",
        "spatial_resolution",
        "notes",
    }
)
BOUNDARY_FIELDS = frozenset(
    {"geometry_type", "feature_count", "redistribution_confirmed"}
)
DEM_FIELDS = frozenset(
    {
        "external",
        "earth_engine_collection",
        "band",
        "units",
        "vertical_datum",
        "native_scale_m",
        "export_crs",
        "export_prefix",
        "drive_folder",
    }
)
SCENARIO_FIELDS = frozenset(
    {
        "scenario_id",
        "display_name",
        "boundary_dataset_id",
        "dem_dataset_id",
        "config_path",
    }
)
SCENARIO_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")


class CatalogError(ValueError):
    """Raised when a data catalog does not satisfy the repository contract."""


class ConcurrentCatalogUpdateError(CatalogError):
    """Raised when a catalog changed after it was loaded."""


@dataclass(frozen=True)
class DataCatalog:
    """An indexed, validated data catalog rooted at a repository."""

    path: Path
    root: Path
    document: dict[str, Any]
    loaded_mtime_ns: int
    datasets_by_id: dict[str, dict[str, Any]]
    scenarios_by_id: dict[str, dict[str, Any]]

    def dataset(self, dataset_id: str) -> dict[str, Any]:
        """Return one dataset declaration by ID."""

        try:
            return self.datasets_by_id[dataset_id]
        except KeyError as exc:
            raise CatalogError(f"Unknown dataset ID: {dataset_id}") from exc

    def scenario(self, scenario_id: str) -> dict[str, Any]:
        """Return one scenario declaration by ID."""

        try:
            return self.scenarios_by_id[scenario_id]
        except KeyError as exc:
            raise CatalogError(f"Unknown scenario ID: {scenario_id}") from exc

    def resolve(self, path: str | Path) -> Path:
        """Resolve a repository-relative catalog path."""

        relative = Path(path)
        if relative.is_absolute():
            raise CatalogError(f"Catalog path must be repository-relative: {path}")
        resolved = (self.root / relative).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise CatalogError(f"Catalog path escapes repository root: {path}") from exc
        return resolved

    def scenario_status(self, scenario_id: str) -> str:
        """Report whether a scenario has its boundary and optional DEM entrypoints."""

        scenario = self.scenario(scenario_id)
        boundary = self.dataset(scenario["boundary_dataset_id"])
        if not self.resolve(boundary["entrypoint"]).is_file():
            return "invalid"
        dem_dataset_id = scenario["dem_dataset_id"]
        if dem_dataset_id is None:
            return "boundary-ready"
        dem = self.dataset(dem_dataset_id)
        if not self.resolve(dem["entrypoint"]).is_file():
            return "dem-pending"
        return "ready"


def _require_fields(
    entry: dict[str, Any],
    required: frozenset[str],
    *,
    description: str,
) -> None:
    missing = sorted(required.difference(entry))
    if missing:
        raise CatalogError(f"{description} is missing required fields: {', '.join(missing)}")


def _validate_catalog_path(value: Any, root: Path, *, description: str) -> None:
    if not isinstance(value, str) or not value:
        raise CatalogError(f"{description} must be a non-empty repository-relative path")
    path = Path(value)
    if path.is_absolute():
        raise CatalogError(f"{description} must be repository-relative: {value}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CatalogError(f"{description} escapes repository root: {value}") from exc


def _validate_dataset(dataset: Any, root: Path, index: int) -> tuple[str, dict[str, Any]]:
    if not isinstance(dataset, dict):
        raise CatalogError(f"Dataset entry {index} must be a mapping")
    description = f"Dataset {dataset.get('dataset_id', index)!r}"
    _require_fields(dataset, DATASET_BASE_FIELDS, description=description)

    dataset_id = dataset["dataset_id"]
    if not isinstance(dataset_id, str) or not dataset_id:
        raise CatalogError(f"{description} dataset_id must be a non-empty string")
    role = dataset["role"]
    if not isinstance(role, str) or role not in {"points", "boundary", "dem"}:
        raise CatalogError(
            f"Dataset {dataset_id!r} role must be one of: points, boundary, dem"
        )
    _validate_catalog_path(dataset["path"], root, description=f"Dataset {dataset_id!r} path")
    _validate_catalog_path(
        dataset["entrypoint"],
        root,
        description=f"Dataset {dataset_id!r} entrypoint",
    )

    if role == "boundary":
        missing = sorted(BOUNDARY_FIELDS.difference(dataset))
        if missing:
            raise CatalogError(
                f"Dataset {dataset_id!r} is missing boundary fields: {', '.join(missing)}"
            )
        if not isinstance(dataset["geometry_type"], str) or not dataset["geometry_type"]:
            raise CatalogError(f"Dataset {dataset_id!r} geometry_type must be a string")
        feature_count = dataset["feature_count"]
        if type(feature_count) is not int or feature_count < 0:
            raise CatalogError(
                f"Dataset {dataset_id!r} feature_count must be a non-negative integer"
            )
        if not isinstance(dataset["redistribution_confirmed"], bool):
            raise CatalogError(
                f"Dataset {dataset_id!r} redistribution_confirmed must be a boolean"
            )
    elif role == "dem":
        missing = sorted(DEM_FIELDS.difference(dataset))
        if missing:
            raise CatalogError(f"Dataset {dataset_id!r} is missing DEM fields: {', '.join(missing)}")
        if not isinstance(dataset["external"], bool):
            raise CatalogError(f"Dataset {dataset_id!r} external must be a boolean")
        native_scale_m = dataset["native_scale_m"]
        if (
            not isinstance(native_scale_m, (int, float))
            or isinstance(native_scale_m, bool)
            or native_scale_m <= 0
        ):
            raise CatalogError(f"Dataset {dataset_id!r} native_scale_m must be positive")
    return dataset_id, dataset


def _validate_scenario(scenario: Any, root: Path, index: int) -> tuple[str, dict[str, Any]]:
    if not isinstance(scenario, dict):
        raise CatalogError(f"Scenario entry {index} must be a mapping")
    description = f"Scenario {scenario.get('scenario_id', index)!r}"
    _require_fields(scenario, SCENARIO_FIELDS, description=description)
    scenario_id = scenario["scenario_id"]
    if not isinstance(scenario_id, str) or not SCENARIO_ID_PATTERN.fullmatch(scenario_id):
        raise CatalogError(
            f"{description} scenario_id must match {SCENARIO_ID_PATTERN.pattern}"
        )
    if not isinstance(scenario["display_name"], str) or not scenario["display_name"]:
        raise CatalogError(f"Scenario {scenario_id!r} display_name must be a non-empty string")
    for field in ("boundary_dataset_id", "dem_dataset_id"):
        value = scenario[field]
        if field == "dem_dataset_id" and value is None:
            continue
        if not isinstance(value, str) or not value:
            raise CatalogError(f"Scenario {scenario_id!r} {field} must be a dataset ID")
    config_path = scenario["config_path"]
    if config_path is not None:
        _validate_catalog_path(
            config_path,
            root,
            description=f"Scenario {scenario_id!r} config_path",
        )
    return scenario_id, scenario


def validate_catalog_document(document: dict[str, Any], root: str | Path) -> None:
    """Validate a schema-v2 catalog document."""

    if not isinstance(document, dict) or type(document.get("schema_version")) is not int:
        raise CatalogError("Data catalog schema_version must be 2")
    if document["schema_version"] != 2:
        raise CatalogError("Data catalog schema_version must be 2")
    if not isinstance(document.get("datasets"), list):
        raise CatalogError("Data catalog must contain a datasets list")
    if not isinstance(document.get("scenarios"), list):
        raise CatalogError("Data catalog must contain a scenarios list")

    repository = Path(root).resolve()
    datasets_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_dataset in enumerate(document["datasets"]):
        dataset_id, dataset = _validate_dataset(raw_dataset, repository, index)
        if dataset_id in datasets_by_id:
            raise CatalogError(f"Duplicate dataset ID: {dataset_id}")
        datasets_by_id[dataset_id] = dataset

    scenario_ids: set[str] = set()
    scenarios: list[dict[str, Any]] = []
    for index, raw_scenario in enumerate(document["scenarios"]):
        scenario_id, scenario = _validate_scenario(raw_scenario, repository, index)
        if scenario_id in scenario_ids:
            raise CatalogError(f"Duplicate scenario ID: {scenario_id}")
        scenario_ids.add(scenario_id)
        scenarios.append(scenario)

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        boundary_id = scenario["boundary_dataset_id"]
        boundary = datasets_by_id.get(boundary_id)
        if boundary is None:
            raise CatalogError(
                f"Scenario {scenario_id!r} references unknown boundary dataset {boundary_id!r}"
            )
        if boundary["role"] != "boundary":
            raise CatalogError(
                f"Scenario {scenario_id!r} boundary dataset {boundary_id!r} "
                "must have role 'boundary'"
            )

        dem_id = scenario["dem_dataset_id"]
        if dem_id is None:
            continue
        dem = datasets_by_id.get(dem_id)
        if dem is None:
            raise CatalogError(
                f"Scenario {scenario_id!r} references unknown DEM dataset {dem_id!r}"
            )
        if dem["role"] != "dem":
            raise CatalogError(
                f"Scenario {scenario_id!r} DEM dataset {dem_id!r} must have role 'dem'"
            )


def _load_data_catalog(path: str | Path, root: str | Path | None = None) -> DataCatalog:
    catalog_path = Path(path).resolve()
    document = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    if root is None:
        repository = (
            catalog_path.parent.parent
            if catalog_path.parent.name == "data"
            else catalog_path.parent
        )
    else:
        repository = Path(root)
    repository = repository.resolve()
    validate_catalog_document(document, repository)
    datasets = {item["dataset_id"]: item for item in document["datasets"]}
    scenarios = {item["scenario_id"]: item for item in document["scenarios"]}
    return DataCatalog(
        path=catalog_path,
        root=repository,
        document=document,
        loaded_mtime_ns=catalog_path.stat().st_mtime_ns,
        datasets_by_id=datasets,
        scenarios_by_id=scenarios,
    )


def load_data_catalog(path: str | Path) -> DataCatalog:
    """Load, validate, and index a repository data catalog."""

    return _load_data_catalog(path)


def _atomic_write_text(path: Path, text: str, *, before_replace=None) -> None:
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
        if before_replace is not None:
            before_replace()
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _guard_catalog_mtime(catalog: DataCatalog) -> None:
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


def save_data_catalog(catalog: DataCatalog, document: dict[str, Any]) -> DataCatalog:
    """Validate and atomically save a catalog unless its source changed."""

    validate_catalog_document(document, catalog.root)
    _guard_catalog_mtime(catalog)
    text = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    _atomic_write_text(
        catalog.path,
        text,
        before_replace=lambda: _guard_catalog_mtime(catalog),
    )
    return _load_data_catalog(catalog.path, catalog.root)


def _manifest_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _manifest_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_manifest_json_safe(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _existing_manifest_files(output_path: Path) -> dict[str, list[dict[str, Any]]]:
    if not output_path.exists():
        return {}
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Cannot read existing data manifest: {output_path}") from exc
    datasets = payload.get("datasets") if isinstance(payload, dict) else None
    if not isinstance(datasets, list):
        raise CatalogError(f"Existing data manifest has no datasets list: {output_path}")
    files_by_id: dict[str, list[dict[str, Any]]] = {}
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise CatalogError("Existing data manifest contains a non-mapping dataset")
        dataset_id = dataset.get("dataset_id")
        files = dataset.get("files")
        if not isinstance(dataset_id, str) or not isinstance(files, list):
            raise CatalogError("Existing data manifest contains an invalid dataset entry")
        files_by_id[dataset_id] = files
    return files_by_id


def _dataset_files(catalog: DataCatalog, dataset: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_path = catalog.resolve(dataset["path"])
    if not dataset_path.exists():
        if bool(dataset.get("external", False)):
            return []
        raise FileNotFoundError(dataset_path)

    if dataset_path.is_file():
        paths = [dataset_path]
    else:
        paths = sorted(
            (path for path in dataset_path.rglob("*") if path.is_file()),
            key=lambda path: path.as_posix(),
        )

    files: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(catalog.root)
        except ValueError as exc:
            raise CatalogError(f"Dataset file escapes repository root: {path}") from exc
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return files


def update_data_manifest(
    catalog: DataCatalog,
    output_path: str | Path,
    dataset_ids: Iterable[str] | None = None,
) -> Path:
    """Atomically update manifest metadata and selected dataset checksums."""

    output = Path(output_path)
    if not output.is_absolute():
        output = catalog.root / output
    output = output.resolve()

    if dataset_ids is None:
        targets: set[str] | None = None
    elif isinstance(dataset_ids, str):
        targets = {dataset_ids}
    else:
        targets = set(dataset_ids)
    if targets is not None:
        unknown = sorted(targets.difference(catalog.datasets_by_id))
        if unknown:
            raise CatalogError(f"Unknown dataset IDs: {', '.join(unknown)}")

    previous_files = _existing_manifest_files(output)
    datasets: list[dict[str, Any]] = []
    for raw_dataset in catalog.document["datasets"]:
        dataset_id = raw_dataset["dataset_id"]
        recompute = (
            targets is None or dataset_id in targets or dataset_id not in previous_files
        )
        dataset = _manifest_json_safe(dict(raw_dataset))
        dataset["files"] = (
            _dataset_files(catalog, raw_dataset)
            if recompute
            else previous_files[dataset_id]
        )
        datasets.append(dataset)

    payload = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "datasets": datasets,
        "scenarios": _manifest_json_safe(catalog.document["scenarios"]),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(output, text)
    return output
