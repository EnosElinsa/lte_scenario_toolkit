"""Plan and execute Earth Engine DEM exports for registered scenarios."""

from __future__ import annotations

import copy
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
import numpy as np
import rasterio
import yaml
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.merge import merge as rasterio_merge
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform
from shapely.geometry import box, mapping
from shapely.ops import unary_union

from .boundary_data import (
    _catalog_transaction_lock,
    _optional_bytes,
    _restore_bytes,
    _restore_owned_bytes,
    _safe_repository_path,
)
from .data_catalog import (
    DataCatalog,
    load_data_catalog,
    save_data_catalog,
    update_data_manifest,
)
from .io import _git_commit, sha256_file, software_versions


class EarthEngineExportError(RuntimeError):
    """Raised when a registered-scenario DEM export cannot be prepared or run."""


class DemIngestError(ValueError):
    """Raised when manually downloaded DEM shards cannot be safely ingested."""


@dataclass(frozen=True)
class DemShardSet:
    """Validated, rectangular set of DEM shards ready for a streaming merge."""

    paths: tuple[Path, ...]
    crs: str
    resolution: tuple[float, float]
    dtype: str
    nodata: float | None
    count: int
    bounds: tuple[float, float, float, float]


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


# ---------------------------------------------------------------------------
# Manual DEM shard ingestion

_PIXEL_EPSILON = 1e-8
_TRANSFORM_EPSILON = 1e-12


def _canonical_crs(value: object) -> tuple[CRS, str]:
    """Parse a CRS and return both the comparable object and stable text."""

    try:
        crs = CRS.from_user_input(value)
    except Exception as exc:  # rasterio raises several CRS-specific exceptions
        raise DemIngestError(f"Invalid CRS {value!r}") from exc
    if crs is None:
        raise DemIngestError("DEM CRS is required")
    # ``to_string`` preserves concise EPSG forms for normal catalog entries.
    return crs, crs.to_string()


def _normalise_nodata(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DemIngestError(f"Invalid DEM nodata value: {value!r}") from exc
    # Preserve NaN as a distinct metadata marker.  It behaves as invalid data
    # during coverage checks, but it is not equivalent to an absent nodata
    # declaration when comparing shard metadata.
    return result


def _same_nodata(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    return left == right


def _same_resolution(
    left: tuple[float, float], right: tuple[float, float], *, tolerance: float = _PIXEL_EPSILON
) -> bool:
    return all(
        math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)
        for a, b in zip(left, right, strict=True)
    )


def _resolution_from_dataset(dataset: rasterio.io.DatasetReader) -> tuple[float, float]:
    transform = dataset.transform
    if (
        not math.isfinite(transform.a)
        or not math.isfinite(transform.e)
        or transform.a <= 0
        or transform.e >= 0
        or not math.isclose(transform.b, 0.0, abs_tol=_TRANSFORM_EPSILON)
        or not math.isclose(transform.d, 0.0, abs_tol=_TRANSFORM_EPSILON)
    ):
        raise DemIngestError(
            f"DEM shard {dataset.name} must use a non-rotated north-up transform"
        )
    return (float(transform.a), float(-transform.e))


def _is_float_dtype(dtype: str) -> bool:
    try:
        return bool(np.issubdtype(np.dtype(dtype), np.floating))
    except TypeError:
        return False


def _grid_index(value: float, origin: float, resolution: float, *, axis: str, path: Path) -> int:
    index = (value - origin) / resolution
    rounded = round(index)
    if not math.isclose(index, rounded, rel_tol=0.0, abs_tol=_PIXEL_EPSILON):
        raise DemIngestError(
            f"DEM shard {path.name} has an unaligned {axis} origin "
            f"({value:g}; expected an integer pixel offset)"
        )
    return int(rounded)


def _footprint(dataset: rasterio.io.DatasetReader):
    left, bottom, right, top = dataset.bounds
    if not all(math.isfinite(value) for value in (left, bottom, right, top)):
        raise DemIngestError(f"DEM shard {dataset.name} has non-finite bounds")
    if right <= left or top <= bottom:
        raise DemIngestError(f"DEM shard {dataset.name} has empty bounds")
    return box(left, bottom, right, top)


def inspect_dem_shards(tiles_dir: str | Path, *, prefix: str) -> DemShardSet:
    """Discover and validate a manually downloaded rectangular DEM shard set.

    Validation intentionally uses geospatial footprints rather than raster
    values.  Earth Engine may retain NoData around the exact scenario polygon;
    those pixels do not make a tile missing from the rectangular export grid.
    """

    directory = Path(tiles_dir)
    if not directory.exists() or not directory.is_dir():
        raise DemIngestError(f"DEM tiles directory does not exist or is not a directory: {directory}")
    if not isinstance(prefix, str) or not prefix:
        raise DemIngestError("DEM shard prefix must be a non-empty string")

    candidates = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ),
        key=lambda path: (path.name, path.as_posix()),
    )
    if not candidates:
        raise DemIngestError(f"No GeoTIFF DEM shards found in {directory}")
    unrelated = [path.name for path in candidates if not path.name.startswith(prefix)]
    if unrelated:
        names = ", ".join(unrelated)
        raise DemIngestError(
            f"Found GeoTIFFs unrelated to DEM shard prefix {prefix!r}: {names}"
        )

    paths: list[Path] = []
    footprints = []
    reference_crs: CRS | None = None
    reference_crs_text: str | None = None
    reference_resolution: tuple[float, float] | None = None
    reference_dtype: str | None = None
    reference_nodata: float | None = None
    reference_count: int | None = None
    reference_origin: tuple[float, float] | None = None

    for path in candidates:
        try:
            with rasterio.open(path) as dataset:
                if dataset.crs is None:
                    raise DemIngestError(f"DEM shard {path.name} is missing a CRS")
                shard_crs, shard_crs_text = _canonical_crs(dataset.crs)
                shard_resolution = _resolution_from_dataset(dataset)
                if dataset.count != 1:
                    raise DemIngestError(
                        f"DEM shard {path.name} must contain exactly one band; found {dataset.count}"
                    )
                shard_dtype = str(dataset.dtypes[0])
                if not _is_float_dtype(shard_dtype):
                    raise DemIngestError(
                        f"DEM shard {path.name} must use a floating-point dtype; found {shard_dtype}"
                    )
                shard_nodata = _normalise_nodata(dataset.nodata)
                transform = dataset.transform
                shard_origin = (float(transform.c), float(transform.f))

                if reference_crs is None:
                    reference_crs = shard_crs
                    reference_crs_text = shard_crs_text
                    reference_resolution = shard_resolution
                    reference_dtype = shard_dtype
                    reference_nodata = shard_nodata
                    reference_count = dataset.count
                    reference_origin = shard_origin
                else:
                    if shard_crs != reference_crs:
                        raise DemIngestError(
                            f"DEM shard {path.name} CRS does not match the other shards"
                        )
                    if not _same_resolution(shard_resolution, reference_resolution):
                        raise DemIngestError(
                            f"DEM shard {path.name} resolution does not match the other shards"
                        )
                    if shard_dtype != reference_dtype:
                        raise DemIngestError(
                            f"DEM shard {path.name} dtype does not match the other shards"
                        )
                    if not _same_nodata(shard_nodata, reference_nodata):
                        raise DemIngestError(
                            f"DEM shard {path.name} nodata does not match the other shards"
                        )
                    if dataset.count != reference_count:
                        raise DemIngestError(
                            f"DEM shard {path.name} band count does not match the other shards"
                        )

                assert reference_resolution is not None
                assert reference_origin is not None
                _grid_index(
                    shard_origin[0],
                    reference_origin[0],
                    reference_resolution[0],
                    axis="x",
                    path=path,
                )
                # Raster rows advance downward while north-up origins advance
                # toward decreasing y.
                _grid_index(
                    reference_origin[1] - shard_origin[1],
                    0.0,
                    reference_resolution[1],
                    axis="y",
                    path=path,
                )
                footprints.append(_footprint(dataset))
                paths.append(path.resolve())
        except DemIngestError:
            raise
        except Exception as exc:
            raise DemIngestError(f"Cannot inspect DEM shard {path}: {exc}") from exc

    assert reference_crs_text is not None
    assert reference_resolution is not None
    assert reference_dtype is not None
    assert reference_count is not None

    # Footprint area gives a robust overlap/missing check without reading any
    # raster values.  The tolerance is one tiny fraction of one pixel area,
    # enough for floating-point edge noise but far below a real missing cell.
    union = unary_union(footprints)
    pixel_area = reference_resolution[0] * reference_resolution[1]
    # Include a tiny relative term for GEOS round-off at large projected
    # coordinates, while keeping the tolerance far below one real pixel.
    tolerance = max(pixel_area * 1e-6, abs(float(union.area)) * 1e-15, 1e-12)
    sum_area = sum(float(footprint.area) for footprint in footprints)
    if sum_area - float(union.area) > tolerance:
        raise DemIngestError("DEM shard footprints overlap")
    minimum_x, minimum_y, maximum_x, maximum_y = union.bounds
    rectangular = box(minimum_x, minimum_y, maximum_x, maximum_y)
    if float(rectangular.area) - float(union.area) > tolerance:
        raise DemIngestError("DEM shard grid has missing rectangular coverage")

    # The Shapely union performs the area check in O(number of shards) rather
    # than allocating one object per raster pixel.  This matters for real
    # Earth Engine exports whose shards can each contain millions of pixels.

    return DemShardSet(
        paths=tuple(paths),
        crs=reference_crs_text,
        resolution=reference_resolution,
        dtype=reference_dtype,
        nodata=reference_nodata,
        count=reference_count,
        bounds=(float(minimum_x), float(minimum_y), float(maximum_x), float(maximum_y)),
    )


def _overview_levels(width: int, height: int) -> list[int]:
    minimum_dimension = min(width, height)
    return [factor for factor in (2, 4, 8, 16, 32, 64) if factor <= minimum_dimension]


def merge_dem_shards(shards: DemShardSet, destination: str | Path) -> Path:
    """Stream validated shards into one tiled, compressed GeoTIFF."""

    if not isinstance(shards, DemShardSet):
        raise DemIngestError("merge_dem_shards requires a DemShardSet from inspect_dem_shards")
    output = Path(destination)
    if os.path.lexists(output):
        raise DemIngestError(f"DEM merge destination already exists: {output}")
    if os.path.lexists(output.parent) and not output.parent.is_dir():
        raise DemIngestError(f"DEM merge destination parent is not a directory: {output.parent}")
    if not output.parent.is_dir():
        raise DemIngestError(f"DEM merge destination parent does not exist: {output.parent}")

    try:
        dtype = np.dtype(shards.dtype)
        predictor = 3 if np.issubdtype(dtype, np.floating) else 2
        dst_kwds = {
            "driver": "GTiff",
            "BIGTIFF": "YES",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "LZW",
            "predictor": predictor,
        }
        rasterio_merge(
            list(shards.paths),
            bounds=shards.bounds,
            res=shards.resolution,
            nodata=shards.nodata,
            dtype=shards.dtype,
            mem_limit=64,
            dst_path=output,
            dst_kwds=dst_kwds,
        )
        with rasterio.open(output, "r+") as dataset:
            levels = _overview_levels(dataset.width, dataset.height)
            if levels:
                dataset.build_overviews(levels, Resampling.average)
                dataset.update_tags(ns="rio_overview", resampling="average")
        return output
    except DemIngestError:
        output.unlink(missing_ok=True)
        raise
    except Exception as exc:
        output.unlink(missing_ok=True)
        raise DemIngestError(f"Could not merge DEM shards into {output}: {exc}") from exc


def _boundary_geometry(boundary_path: Path, target_crs: CRS):
    try:
        frame = gpd.read_file(boundary_path)
    except Exception as exc:
        raise DemIngestError(f"Cannot read registered boundary {boundary_path}: {exc}") from exc
    if frame.empty:
        raise DemIngestError(f"Registered boundary has no features: {boundary_path}")
    if frame.crs is None:
        raise DemIngestError(f"Registered boundary is missing a CRS: {boundary_path}")
    try:
        projected = frame.to_crs(target_crs)
        geometry = projected.geometry.union_all()
    except Exception as exc:
        raise DemIngestError(f"Cannot reproject registered boundary {boundary_path}: {exc}") from exc
    if geometry is None or geometry.is_empty:
        raise DemIngestError(f"Registered boundary has empty geometry: {boundary_path}")
    return geometry


def validate_dem_coverage(
    raster_path: str | Path,
    boundary_path: str | Path,
    *,
    expected_crs: str,
    expected_resolution: tuple[float, float] | float,
) -> None:
    """Validate raster metadata and finite elevation coverage of a boundary."""

    raster = Path(raster_path)
    try:
        expected_crs_obj, _ = _canonical_crs(expected_crs)
        if isinstance(expected_resolution, (int, float)) and not isinstance(expected_resolution, bool):
            expected_res = (float(expected_resolution), float(expected_resolution))
        else:
            expected_res = tuple(float(value) for value in expected_resolution)
        if len(expected_res) != 2 or any(value <= 0 or not math.isfinite(value) for value in expected_res):
            raise DemIngestError("Expected DEM resolution must contain two positive finite values")
    except DemIngestError:
        raise
    except Exception as exc:
        raise DemIngestError("Expected DEM resolution must contain two numeric values") from exc

    try:
        with rasterio.open(raster) as dataset:
            if dataset.crs is None:
                raise DemIngestError(f"DEM raster is missing a CRS: {raster}")
            if CRS.from_user_input(dataset.crs) != expected_crs_obj:
                raise DemIngestError(
                    f"DEM raster CRS does not match expected CRS ({dataset.crs} != {expected_crs})"
                )
            if dataset.count != 1:
                raise DemIngestError(f"DEM raster must contain exactly one band; found {dataset.count}")
            dtype = str(dataset.dtypes[0])
            if not _is_float_dtype(dtype):
                raise DemIngestError(f"DEM raster must use a floating-point dtype; found {dtype}")
            actual_res = _resolution_from_dataset(dataset)
            if not _same_resolution(actual_res, expected_res):
                raise DemIngestError(
                    f"DEM raster resolution does not match expected resolution "
                    f"({actual_res} != {expected_res})"
                )
            geometry = _boundary_geometry(Path(boundary_path), expected_crs_obj)
            raster_bounds = box(*dataset.bounds)
            # Require the registered geometry to be covered by the raster
            # extent.  ``covers`` includes exact edge contact while refusing a
            # materially out-of-bounds polygon; only the tiny coordinate
            # tolerance in the affine/geometry libraries is accepted.
            extent_epsilon = max(actual_res) * 1e-9
            if not raster_bounds.buffer(extent_epsilon).covers(geometry):
                raise DemIngestError("DEM raster bounds do not cover the registered boundary")

            boundary_mapping = [mapping(geometry)]
            for _, window in dataset.block_windows(1):
                if not window_bounds(window, dataset.transform):
                    continue
                block_shape = (int(window.height), int(window.width))
                block_box = box(*window_bounds(window, dataset.transform))
                if not block_box.intersects(geometry):
                    continue
                values = dataset.read(1, window=window, masked=True)
                inside = geometry_mask(
                    boundary_mapping,
                    out_shape=block_shape,
                    transform=window_transform(window, dataset.transform),
                    invert=True,
                    all_touched=False,
                )
                masked = np.ma.getmaskarray(values)
                finite = np.isfinite(np.asarray(values.filled(np.nan), dtype=float))
                if np.any(inside & ~masked & finite):
                    return None
    except DemIngestError:
        raise
    except Exception as exc:
        raise DemIngestError(f"Cannot validate DEM coverage for {raster}: {exc}") from exc
    raise DemIngestError("DEM raster has no valid finite elevation inside the registered boundary")


def _safe_catalog_path(root: Path, value: str | Path, *, description: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts:
        raise DemIngestError(f"{description} must be a repository-relative path")
    try:
        return _safe_repository_path(root, *relative.parts)
    except Exception as exc:
        if isinstance(exc, DemIngestError):
            raise
        raise DemIngestError(f"Unsafe {description}: {value}: {exc}") from exc


def _format_resolution(resolution: tuple[float, float]) -> str:
    if math.isclose(resolution[0], resolution[1], rel_tol=0.0, abs_tol=_PIXEL_EPSILON):
        return f"{resolution[0]:g} m"
    return f"{resolution[0]:g} x {resolution[1]:g} m"


def _registry_resolution(dem: dict[str, Any]) -> tuple[float, float]:
    value = dem.get("native_scale_m")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DemIngestError("Registered DEM native_scale_m must be numeric")
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0:
        raise DemIngestError("Registered DEM native_scale_m must be positive")
    return (scale, scale)


def _dem_observed_metadata(path: Path) -> dict[str, Any]:
    try:
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                raise DemIngestError("Merged DEM is missing a CRS")
            crs_obj, crs_text = _canonical_crs(dataset.crs)
            del crs_obj
            resolution = _resolution_from_dataset(dataset)
            dtype = str(dataset.dtypes[0])
            if dataset.count != 1 or not _is_float_dtype(dtype):
                raise DemIngestError("Merged DEM must contain one floating-point band")
            return {
                "crs": crs_text,
                "spatial_resolution": _format_resolution(resolution),
                "resolution": resolution,
                "dtype": dtype,
                "nodata": _normalise_nodata(dataset.nodata),
                "count": int(dataset.count),
                "width": int(dataset.width),
                "height": int(dataset.height),
                "bounds": tuple(float(value) for value in dataset.bounds),
            }
    except DemIngestError:
        raise
    except Exception as exc:
        raise DemIngestError(f"Cannot inspect merged DEM {path}: {exc}") from exc


def _remove_owned_output(path: Path, installed_stat: tuple[int, int] | None) -> None:
    if installed_stat is None or not os.path.lexists(path):
        return
    try:
        stat_result = path.stat()
        if (stat_result.st_dev, stat_result.st_ino) == installed_stat:
            path.unlink()
    except FileNotFoundError:
        return


def _rollback_ingest(
    *,
    destination: Path,
    installed_stat: tuple[int, int] | None,
    catalog_path: Path,
    original_catalog: bytes,
    written_catalog: bytes | None,
    manifest_path: Path,
    original_manifest: bytes | None,
    manifest_update_started: bool,
    operation_error: Exception,
) -> None:
    failures: list[tuple[str, Exception]] = []
    try:
        _remove_owned_output(destination, installed_stat)
    except Exception as exc:
        failures.append(("destination", exc))
    try:
        if written_catalog is not None:
            _restore_owned_bytes(catalog_path, original_catalog, written_catalog)
    except Exception as exc:
        failures.append(("catalog", exc))
    if manifest_update_started:
        try:
            if _optional_bytes(manifest_path) != original_manifest:
                _restore_bytes(manifest_path, original_manifest)
        except Exception as exc:
            failures.append(("manifest", exc))
    if failures:
        details = "; ".join(f"{name}: {error}" for name, error in failures)
        raise DemIngestError(f"DEM ingest rollback failed: {details}") from operation_error


def _ingest_dem_shards_locked(
    catalog: DataCatalog,
    scenario_id: str,
    tiles_dir: str | Path,
    staging_root: Path,
) -> DataCatalog:
    scenario = catalog.scenario(scenario_id)
    dem_id = scenario.get("dem_dataset_id")
    if dem_id is None:
        raise DemIngestError(f"Scenario {scenario_id!r} does not declare a DEM")
    dem = catalog.dataset(dem_id)
    boundary = catalog.dataset(scenario["boundary_dataset_id"])
    boundary_path = _safe_catalog_path(
        catalog.root, boundary["entrypoint"], description="boundary entrypoint"
    )
    if not boundary_path.is_file():
        raise DemIngestError(f"Registered scenario boundary file does not exist: {boundary_path}")
    prefix = dem.get("export_prefix")
    if not isinstance(prefix, str) or not prefix:
        raise DemIngestError(f"Registered DEM {dem_id!r} has no export_prefix")

    destination = _safe_catalog_path(catalog.root, dem["entrypoint"], description="DEM entrypoint")
    if os.path.lexists(destination):
        raise DemIngestError(f"DEM destination already exists: {destination}")
    if os.path.lexists(destination.parent) and not destination.parent.is_dir():
        raise DemIngestError(f"DEM destination parent is not a directory: {destination.parent}")
    manifest_path = _safe_catalog_path(catalog.root, "data/manifest.json", description="manifest path")
    if os.path.lexists(manifest_path.parent) and not manifest_path.parent.is_dir():
        raise DemIngestError(f"Data manifest parent is not a directory: {manifest_path.parent}")
    if os.path.lexists(manifest_path) and not manifest_path.is_file():
        raise DemIngestError(f"Data manifest path must be a file: {manifest_path}")

    expected_crs = dem.get("export_crs") or dem.get("crs")
    if not isinstance(expected_crs, str) or not expected_crs:
        raise DemIngestError(f"Registered DEM {dem_id!r} has no expected CRS")
    original_catalog = catalog.path.read_bytes()
    original_manifest = _optional_bytes(manifest_path)
    expected_resolution = _registry_resolution(dem)
    shards = inspect_dem_shards(tiles_dir, prefix=prefix)
    expected_crs_obj, _ = _canonical_crs(expected_crs)
    shard_crs_obj, _ = _canonical_crs(shards.crs)
    if shard_crs_obj != expected_crs_obj:
        raise DemIngestError(
            f"DEM shard CRS does not match registered CRS ({shards.crs} != {expected_crs})"
        )
    if not _same_resolution(shards.resolution, expected_resolution):
        raise DemIngestError(
            f"DEM shard resolution does not match registered resolution "
            f"({shards.resolution} != {expected_resolution})"
        )

    destination_parent_existed = os.path.lexists(destination.parent)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DemIngestError(f"Cannot create DEM destination parent: {destination.parent}") from exc
    partial_name: str | None = None
    try:
        partial_fd, partial_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
        )
        os.close(partial_fd)
        partial = Path(partial_name)
        partial.unlink()
    except OSError as exc:
        if partial_name is not None:
            Path(partial_name).unlink(missing_ok=True)
        if not destination_parent_existed and destination.parent.is_dir():
            try:
                destination.parent.rmdir()
            except OSError:
                pass
        raise DemIngestError(f"Cannot allocate hidden DEM merge destination beside {destination}") from exc

    installed_stat: tuple[int, int] | None = None
    written_catalog: bytes | None = None
    manifest_update_started = False
    try:
        merge_dem_shards(shards, partial)
        validate_dem_coverage(
            partial,
            boundary_path,
            expected_crs=expected_crs,
            expected_resolution=expected_resolution,
        )
        observed = _dem_observed_metadata(partial)
        document = copy.deepcopy(catalog.document)
        dem_record = next(
            item for item in document["datasets"] if item.get("dataset_id") == dem_id
        )
        today = datetime.now(timezone.utc).date().isoformat()
        previous_notes = dem_record.get("notes")
        provenance = (
            f"Ingested from {len(shards.paths)} manually downloaded DEM shard(s) "
            f"on {today}; source prefix {prefix!r}."
        )
        dem_record.update(
            {
                "crs": observed["crs"],
                "spatial_resolution": observed["spatial_resolution"],
                "download_date": today,
                "dtype": observed["dtype"],
                "nodata": observed["nodata"],
                "count": observed["count"],
                "width": observed["width"],
                "height": observed["height"],
                "bounds": list(observed["bounds"]),
                "notes": (
                    f"{previous_notes.rstrip()} {provenance}"
                    if isinstance(previous_notes, str) and previous_notes.strip()
                    else provenance
                ),
            }
        )
        written_catalog = yaml.safe_dump(
            document, sort_keys=False, allow_unicode=True
        ).encode("utf-8")

        # Hard-linking then unlinking the partial is an atomic, no-clobber
        # install on the same filesystem.  The parent was created above and
        # destination existence was checked before the operation began.
        try:
            os.link(partial, destination)
        except FileExistsError as exc:
            raise DemIngestError(f"DEM destination appeared during ingest: {destination}") from exc
        partial.unlink()
        installed_stat_result = destination.stat()
        installed_stat = (installed_stat_result.st_dev, installed_stat_result.st_ino)

        saved_catalog = save_data_catalog(catalog, document)
        manifest_update_started = True
        update_data_manifest(saved_catalog, "data/manifest.json", dataset_ids={dem_id})
        return saved_catalog
    except DemIngestError as exc:
        try:
            _rollback_ingest(
                destination=destination,
                installed_stat=installed_stat,
                catalog_path=catalog.path,
                original_catalog=original_catalog,
                written_catalog=written_catalog,
                manifest_path=manifest_path,
                original_manifest=original_manifest,
                manifest_update_started=manifest_update_started,
                operation_error=exc,
            )
        finally:
            partial.unlink(missing_ok=True)
            if not destination_parent_existed and destination.parent.is_dir():
                try:
                    destination.parent.rmdir()
                except OSError:
                    pass
        raise
    except Exception as exc:
        wrapped = DemIngestError(f"DEM ingest failed for scenario {scenario_id!r}: {exc}")
        try:
            _rollback_ingest(
                destination=destination,
                installed_stat=installed_stat,
                catalog_path=catalog.path,
                original_catalog=original_catalog,
                written_catalog=written_catalog,
                manifest_path=manifest_path,
                original_manifest=original_manifest,
                manifest_update_started=manifest_update_started,
                operation_error=wrapped,
            )
        finally:
            partial.unlink(missing_ok=True)
            if not destination_parent_existed and destination.parent.is_dir():
                try:
                    destination.parent.rmdir()
                except OSError:
                    pass
        raise wrapped from exc
    finally:
        partial.unlink(missing_ok=True)


def ingest_dem_shards(
    catalog: DataCatalog,
    scenario_id: str,
    tiles_dir: str | Path,
) -> DataCatalog:
    """Transactionally merge and register manually downloaded DEM shards."""

    if not isinstance(catalog, DataCatalog):
        raise DemIngestError("ingest_dem_shards requires a loaded DataCatalog")
    try:
        with _catalog_transaction_lock(catalog.root) as staging_root:
            # Reload after acquiring the lock: callers may have loaded a stale
            # catalog while another transaction was completing.
            fresh_catalog = load_data_catalog(catalog.path, repo_root=catalog.root)
            return _ingest_dem_shards_locked(
                fresh_catalog,
                scenario_id,
                tiles_dir,
                staging_root,
            )
    except DemIngestError:
        raise
    except Exception as exc:
        raise DemIngestError(f"DEM ingest failed for scenario {scenario_id!r}: {exc}") from exc
