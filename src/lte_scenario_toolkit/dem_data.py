"""Plan and execute Earth Engine DEM exports for registered scenarios."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd

from .data_catalog import DataCatalog
from .io import _git_commit, sha256_file, software_versions


class EarthEngineExportError(RuntimeError):
    """Raised when a registered-scenario DEM export cannot be prepared or run."""


@dataclass(frozen=True)
class DemExportPlan:
    """Resolved, reproducible parameters for one registered-scenario DEM export."""

    repo_root: Path
    scenario_id: str
    display_name: str
    boundary_path: Path
    boundary_sha256: str
    dataset_id: str
    band: str
    scale_m: float
    export_crs: str
    drive_folder: str
    export_prefix: str
    file_dimensions: int
    shard_size: int
    max_pixels: float
    estimated_pixels: int
    project: str | None

    def json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the export plan."""

        return {
            "repo_root": str(self.repo_root),
            "scenario_id": self.scenario_id,
            "display_name": self.display_name,
            "boundary_path": str(self.boundary_path),
            "boundary_sha256": self.boundary_sha256,
            "dataset_id": self.dataset_id,
            "band": self.band,
            "scale_m": self.scale_m,
            "export_crs": self.export_crs,
            "drive_folder": self.drive_folder,
            "export_prefix": self.export_prefix,
            "file_dimensions": self.file_dimensions,
            "shard_size": self.shard_size,
            "max_pixels": self.max_pixels,
            "estimated_pixels": self.estimated_pixels,
            "project": self.project,
        }


@dataclass(frozen=True)
class DemExportResult:
    """Outcome of an Earth Engine coverage preflight or submitted export."""

    image_count: int
    task_id: str | None


def build_dem_export_plan(
    catalog: DataCatalog,
    scenario_id: str,
    *,
    project: str | None,
    scale_m: float | None = None,
    file_dimensions: int = 8192,
    shard_size: int = 256,
    max_pixels: float = 1e13,
    drive_folder: str | None = None,
) -> DemExportPlan:
    """Resolve a registered scenario and estimate its DEM export size."""

    try:
        return _build_dem_export_plan(
            catalog,
            scenario_id,
            project=project,
            scale_m=scale_m,
            file_dimensions=file_dimensions,
            shard_size=shard_size,
            max_pixels=max_pixels,
            drive_folder=drive_folder,
        )
    except EarthEngineExportError:
        raise
    except Exception as exc:
        raise EarthEngineExportError(
            f"Cannot build DEM export plan for {scenario_id!r}: {exc}"
        ) from exc


def _positive_number(value: object, *, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field} must be a positive finite number")
    return float(value)


def _positive_integer(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _build_dem_export_plan(
    catalog: DataCatalog,
    scenario_id: str,
    *,
    project: str | None,
    scale_m: float | None,
    file_dimensions: int,
    shard_size: int,
    max_pixels: float,
    drive_folder: str | None,
) -> DemExportPlan:
    scenario = catalog.scenario(scenario_id)
    dem_dataset_id = scenario["dem_dataset_id"]
    if dem_dataset_id is None:
        raise ValueError(f"Scenario {scenario_id!r} does not declare a DEM dataset")
    dem = catalog.dataset(dem_dataset_id)
    boundary = catalog.dataset(scenario["boundary_dataset_id"])
    boundary_path = catalog.resolve(boundary["entrypoint"])
    if not boundary_path.is_file():
        raise FileNotFoundError(f"Scenario boundary file does not exist: {boundary_path}")

    selected_scale = _positive_number(
        dem["native_scale_m"] if scale_m is None else scale_m,
        field="scale_m",
    )
    selected_file_dimensions = _positive_integer(
        file_dimensions,
        field="file_dimensions",
    )
    selected_shard_size = _positive_integer(shard_size, field="shard_size")
    selected_max_pixels = _positive_number(max_pixels, field="max_pixels")
    if selected_file_dimensions % selected_shard_size != 0:
        raise ValueError("file_dimensions must be a multiple of shard_size")

    selected_folder = dem["drive_folder"] if drive_folder is None else drive_folder
    boundary_frame = gpd.read_file(boundary_path).to_crs(dem["export_crs"])
    if boundary_frame.empty:
        raise ValueError(f"Scenario {scenario_id!r} boundary has no features")
    dissolved = boundary_frame.geometry.union_all()
    if dissolved.is_empty or not math.isfinite(dissolved.area) or dissolved.area <= 0:
        raise ValueError(f"Scenario {scenario_id!r} boundary has no positive area")
    estimated_pixels = math.ceil(dissolved.area / (selected_scale**2))
    if estimated_pixels > selected_max_pixels:
        raise ValueError(
            f"estimated pixel count {estimated_pixels} exceeds max_pixels "
            f"{selected_max_pixels:g}"
        )

    return DemExportPlan(
        repo_root=catalog.root,
        scenario_id=scenario_id,
        display_name=scenario["display_name"],
        boundary_path=boundary_path,
        boundary_sha256=sha256_file(boundary_path),
        dataset_id=dem["earth_engine_collection"],
        band=dem["band"],
        scale_m=selected_scale,
        export_crs=dem["export_crs"],
        drive_folder=selected_folder,
        export_prefix=dem["export_prefix"],
        file_dimensions=selected_file_dimensions,
        shard_size=selected_shard_size,
        max_pixels=selected_max_pixels,
        estimated_pixels=estimated_pixels,
        project=project,
    )


def _import_gee() -> tuple[Any, Any]:
    """Import the optional online dependencies only when an online run begins."""

    try:
        import ee  # type: ignore
        import geemap  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise EarthEngineExportError(
            "Earth Engine export requires earthengine-api and geemap"
        ) from exc
    return ee, geemap


def execute_dem_export(
    plan: DemExportPlan,
    *,
    start: bool,
    ee_module: Any = None,
    geemap_module: Any = None,
) -> DemExportResult:
    """Run coverage preflight and optionally submit one Google Drive export."""

    try:
        if not isinstance(plan.project, str) or not plan.project.strip():
            raise ValueError(
                "Earth Engine project is required for online DEM export; "
                "pass --project or set EE_PROJECT"
            )

        if ee_module is None or geemap_module is None:
            imported_ee, imported_geemap = _import_gee()
            ee_module = imported_ee if ee_module is None else ee_module
            geemap_module = imported_geemap if geemap_module is None else geemap_module

        current_boundary_sha256 = sha256_file(plan.boundary_path)
        if current_boundary_sha256 != plan.boundary_sha256:
            raise ValueError(
                "Boundary SHA256 checksum changed after export planning: "
                f"expected {plan.boundary_sha256}, found {current_boundary_sha256}"
            )
        boundary = gpd.read_file(plan.boundary_path).to_crs("EPSG:4326")
        ee_module.Initialize(project=plan.project)
        geometry_column = boundary.geometry.name
        roi = geemap_module.gdf_to_ee(
            boundary[[geometry_column]],
            geodesic=False,
        ).geometry()
        collection = ee_module.ImageCollection(plan.dataset_id).filterBounds(roi)
        image_count = int(collection.size().getInfo())
        if image_count <= 0:
            raise ValueError(
                f"{plan.dataset_id} returned no images for scenario {plan.scenario_id!r}"
            )

        image = collection.mosaic().select(plan.band).clip(roi)
        if not start:
            return DemExportResult(image_count=image_count, task_id=None)

        task = ee_module.batch.Export.image.toDrive(
            image=image,
            description=plan.export_prefix,
            folder=plan.drive_folder,
            fileNamePrefix=plan.export_prefix,
            region=roi,
            scale=plan.scale_m,
            crs=plan.export_crs,
            maxPixels=plan.max_pixels,
            fileDimensions=plan.file_dimensions,
            shardSize=plan.shard_size,
            fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": True},
            skipEmptyTiles=False,
        )
        task.start()
        raw_task_id = getattr(task, "id", None)
        if raw_task_id is None or not str(raw_task_id):
            raise ValueError("Earth Engine export task did not provide a task ID")
        return DemExportResult(image_count=image_count, task_id=str(raw_task_id))
    except EarthEngineExportError:
        raise
    except Exception as exc:
        raise EarthEngineExportError(
            f"Earth Engine DEM export failed: {type(exc).__name__}: {exc}"
        ) from exc


def _atomic_write_text(path: Path, text: str) -> None:
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
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _publish_run_directory(
    staging_path: Path,
    runs_root: Path,
    base_name: str,
) -> Path:
    for attempt in range(100):
        suffix = "" if attempt == 0 else f"-{uuid.uuid4().hex[:8]}"
        run_path = runs_root / f"{base_name}{suffix}"
        if os.path.lexists(run_path):
            continue
        try:
            staging_path.rename(run_path)
        except OSError:
            if os.path.lexists(run_path):
                continue
            raise
        else:
            return run_path
    raise FileExistsError(f"Could not allocate a unique run directory under {runs_root}")


def write_export_run(
    plan: DemExportPlan,
    result: DemExportResult,
    *,
    runs_root: str | Path,
) -> Path:
    """Write human-readable and machine-readable records for an online run."""

    staging_path: Path | None = None
    try:
        created_at = datetime.now(timezone.utc)
        timestamp = created_at.isoformat().replace("+00:00", "Z")
        base_name = (
            f"{created_at:%Y%m%d-%H%M%S}-{plan.scenario_id}-dem-export"
        )
        runs_directory = Path(runs_root)
        runs_directory.mkdir(parents=True, exist_ok=True)
        git_commit = _git_commit(plan.repo_root)
        versions = software_versions()
        task_id = result.task_id or "<not started>"
        commit_label = git_commit or "<unavailable>"
        project_label = plan.project or "<not provided>"

        payload = {
            **plan.json_dict(),
            "timestamp": timestamp,
            "image_count": result.image_count,
            "task_id": result.task_id,
            "git_commit": git_commit,
            "software_versions": versions,
        }
        run_lines = [
            "# DEM export run",
            "",
            f"- Timestamp: {timestamp}",
            f"- Scenario: {plan.display_name} ({plan.scenario_id})",
            f"- Earth Engine project: {project_label}",
            f"- Git commit: {commit_label}",
            f"- Intersecting images: {result.image_count}",
            f"- Earth Engine task ID: {task_id}",
            "",
            "## Export plan",
            "",
            f"- Dataset: {plan.dataset_id}",
            f"- Band: {plan.band}",
            f"- Scale: {plan.scale_m:g} m",
            f"- CRS: {plan.export_crs}",
            f"- Drive destination: {plan.drive_folder}/{plan.export_prefix}_*.tif",
            f"- Estimated pixels: {plan.estimated_pixels}",
            f"- maxPixels: {plan.max_pixels:g}",
            "",
            "## Software versions",
            "",
            *[f"- {name}: {version}" for name, version in sorted(versions.items())],
            "",
        ]
        layer_lines = [
            "# DEM data layer",
            "",
            f"- Scenario: {plan.display_name} ({plan.scenario_id})",
            f"- Earth Engine dataset: {plan.dataset_id}",
            f"- Band: {plan.band}",
            f"- Export scale: {plan.scale_m:g} m",
            f"- Export CRS: {plan.export_crs}",
            "- Format: Cloud Optimized GeoTIFF",
            f"- File dimensions: {plan.file_dimensions}",
            f"- Shard size: {plan.shard_size}",
            "- Empty tiles retained: yes",
            f"- Related task ID: {task_id}",
            "",
        ]
        source_lines = [
            "# Export sources",
            "",
            "## Boundary",
            "",
            f"- Local path: {plan.boundary_path}",
            f"- SHA256: {plan.boundary_sha256}",
            "",
            "## DEM",
            "",
            f"- Earth Engine ImageCollection: {plan.dataset_id}",
            f"- Selected band: {plan.band}",
            "",
        ]

        staging_path = Path(
            tempfile.mkdtemp(
                prefix=f".{base_name}.",
                suffix=".tmp",
                dir=runs_directory,
            )
        )
        _atomic_write_text(staging_path / "RUN.md", "\n".join(run_lines))
        _atomic_write_text(staging_path / "DATA_LAYER.md", "\n".join(layer_lines))
        _atomic_write_text(staging_path / "sources.md", "\n".join(source_lines))
        _atomic_write_text(
            staging_path / "export-plan.json",
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )
        run_path = _publish_run_directory(
            staging_path,
            runs_directory,
            base_name,
        )
        staging_path = None
        return run_path
    except EarthEngineExportError:
        raise
    except Exception as exc:
        raise EarthEngineExportError(
            f"Cannot write DEM export run artifacts: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if staging_path is not None and os.path.lexists(staging_path):
            shutil.rmtree(staging_path)
