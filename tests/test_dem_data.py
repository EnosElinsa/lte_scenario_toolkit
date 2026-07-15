import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import Affine, from_origin
from shapely.geometry import box

from lte_scenario_toolkit.data_catalog import load_data_catalog


def _write_dem_tile(
    path: Path,
    *,
    left: float,
    top: float,
    values: np.ndarray | None = None,
    crs: str = "EPSG:3857",
    resolution: tuple[float, float] = (1.0, 1.0),
    dtype: str = "float32",
    nodata: float | None = -9999.0,
    count: int = 1,
) -> Path:
    data = np.asarray(values if values is not None else np.ones((2, 2)), dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(left, top, resolution[0], resolution[1])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[-1],
        height=data.shape[-2],
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        if count == 1:
            dst.write(data, 1)
        else:
            dst.write(np.stack([data] * count), tuple(range(1, count + 1)))
    return path


def _write_catalog(tmp_path: Path, *, dem_dataset_id: str | None = "dem") -> Path:
    boundary_path = tmp_path / "boundary_shp" / "sample-city" / "sample-city.shp"
    boundary_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"name": ["west", "east"]},
        geometry=[
            box(0, 0, 1000, 2000),
            box(500, 0, 1500, 2000),
        ],
        crs="EPSG:3857",
    ).to_file(boundary_path)

    document = {
        "schema_version": 2,
        "datasets": [
            {
                "dataset_id": "boundary",
                "role": "boundary",
                "path": "boundary_shp/sample-city",
                "entrypoint": "boundary_shp/sample-city/sample-city.shp",
                "source_url": "https://example.test/boundary.zip",
                "provider": "Example GIS",
                "license": "CC0-1.0",
                "download_date": "2026-07-16",
                "crs": "EPSG:3857",
                "spatial_resolution": "polygon vector",
                "notes": "fixture boundary",
                "geometry_type": "Polygon",
                "feature_count": 2,
                "redistribution_confirmed": True,
            },
            {
                "dataset_id": "dem",
                "role": "dem",
                "path": "dem/sample-city",
                "entrypoint": "dem/sample-city/usgs_3dep_1m_sample-city.tif",
                "source_url": (
                    "https://developers.google.com/earth-engine/datasets/catalog/USGS_3DEP_1m"
                ),
                "provider": "United States Geological Survey",
                "license": "USGS public domain",
                "download_date": None,
                "crs": "EPSG:3857",
                "spatial_resolution": "1 m",
                "notes": "pending export",
                "external": True,
                "earth_engine_collection": "USGS/3DEP/1m",
                "band": "elevation",
                "units": "metres",
                "vertical_datum": "NAVD88",
                "native_scale_m": 1,
                "export_crs": "EPSG:3857",
                "export_prefix": "usgs_3dep_1m_sample-city",
                "drive_folder": "lte-scenario-toolkit-dem",
            },
        ],
        "scenarios": [
            {
                "scenario_id": "sample-city",
                "display_name": "Sample City",
                "boundary_dataset_id": "boundary",
                "dem_dataset_id": dem_dataset_id,
                "config_path": None,
            }
        ],
    }
    catalog_path = tmp_path / "data" / "datasets.yaml"
    catalog_path.parent.mkdir()
    catalog_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return catalog_path


def _build_plan(tmp_path: Path, *, project: str | None = "ee-project"):
    from lte_scenario_toolkit.dem_data import build_dem_export_plan

    catalog = load_data_catalog(_write_catalog(tmp_path), repo_root=tmp_path)
    return build_dem_export_plan(catalog, "sample-city", project=project)


class _Info:
    def __init__(self, value):
        self.value = value

    def getInfo(self):
        return self.value


class _FakeImage:
    def __init__(self):
        self.operations = []

    def select(self, band):
        self.operations.append(("select", band))
        return self

    def clip(self, roi):
        self.operations.append(("clip", roi))
        return self


class _FakeCollection:
    def __init__(self, image_count):
        self.image_count = image_count
        self.image = _FakeImage()
        self.filtered_roi = None
        self.mosaic_calls = 0

    def filterBounds(self, roi):
        self.filtered_roi = roi
        return self

    def size(self):
        return _Info(self.image_count)

    def mosaic(self):
        self.mosaic_calls += 1
        return self.image


class _FakeTask:
    def __init__(self):
        self.id = "task-123"
        self.start_calls = 0

    def start(self):
        self.start_calls += 1


class _FakeToDrive:
    def __init__(self, task):
        self.task = task
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.task


class _FakeEE:
    def __init__(self, image_count=3, *, initialize_error=None):
        self.collection = _FakeCollection(image_count)
        self.task = _FakeTask()
        self.to_drive = _FakeToDrive(self.task)
        self.batch = type(
            "Batch",
            (),
            {
                "Export": type(
                    "Export",
                    (),
                    {"image": type("ImageExport", (), {"toDrive": self.to_drive})()},
                )()
            },
        )()
        self.initialize_error = initialize_error
        self.initialized_projects = []
        self.image_collection_ids = []

    def Initialize(self, *, project):
        self.initialized_projects.append(project)
        if self.initialize_error is not None:
            raise self.initialize_error

    def ImageCollection(self, dataset_id):
        self.image_collection_ids.append(dataset_id)
        return self.collection


class _FakeGeemap:
    def __init__(self):
        self.calls = []
        self.roi = object()

    def gdf_to_ee(self, frame, *, geodesic):
        self.calls.append((frame.copy(), geodesic))
        roi = self.roi

        class FeatureCollection:
            def geometry(self):
                return roi

        return FeatureCollection()


def test_build_dem_export_plan_uses_registered_defaults_and_boundary_area(tmp_path):
    from lte_scenario_toolkit.dem_data import build_dem_export_plan

    catalog = load_data_catalog(_write_catalog(tmp_path), repo_root=tmp_path)
    plan = build_dem_export_plan(catalog, "sample-city", project="ee-project")

    assert plan.repo_root == tmp_path.resolve()
    assert plan.scenario_id == "sample-city"
    assert plan.display_name == "Sample City"
    assert plan.boundary_path == (
        tmp_path / "boundary_shp" / "sample-city" / "sample-city.shp"
    ).resolve()
    assert plan.boundary_sha256 == hashlib.sha256(plan.boundary_path.read_bytes()).hexdigest()
    assert plan.dataset_id == "USGS/3DEP/1m"
    assert plan.band == "elevation"
    assert plan.scale_m == 1
    assert plan.export_crs == "EPSG:3857"
    assert plan.drive_folder == "lte-scenario-toolkit-dem"
    assert plan.export_prefix == "usgs_3dep_1m_sample-city"
    assert plan.file_dimensions == 8192
    assert plan.shard_size == 256
    assert plan.max_pixels == 1e13
    assert plan.estimated_pixels == 3_000_000
    assert plan.project == "ee-project"
    assert json.dumps(plan.json_dict())
    assert plan.json_dict()["repo_root"] == str(tmp_path.resolve())
    assert plan.json_dict()["boundary_path"] == str(plan.boundary_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"scale_m": 0}, "scale"),
        ({"file_dimensions": 0}, "file_dimensions"),
        ({"shard_size": 0}, "shard_size"),
        ({"max_pixels": 0}, "max_pixels"),
        ({"file_dimensions": 1000, "shard_size": 256}, "multiple"),
        ({"max_pixels": 2_999_999}, "estimated"),
    ],
)
def test_build_dem_export_plan_rejects_invalid_export_limits(tmp_path, overrides, message):
    from lte_scenario_toolkit.dem_data import EarthEngineExportError, build_dem_export_plan

    catalog = load_data_catalog(_write_catalog(tmp_path), repo_root=tmp_path)

    with pytest.raises(EarthEngineExportError, match=message) as captured:
        build_dem_export_plan(catalog, "sample-city", project=None, **overrides)

    assert captured.value.__cause__ is not None


def test_build_dem_export_plan_requires_a_registered_dem_and_boundary_file(tmp_path):
    from lte_scenario_toolkit.dem_data import EarthEngineExportError, build_dem_export_plan

    no_dem_root = tmp_path / "no-dem"
    no_dem = load_data_catalog(
        _write_catalog(no_dem_root, dem_dataset_id=None), repo_root=no_dem_root
    )
    with pytest.raises(EarthEngineExportError, match="does not declare a DEM"):
        build_dem_export_plan(no_dem, "sample-city", project=None)

    missing_boundary_root = tmp_path / "missing-boundary"
    missing_boundary = load_data_catalog(
        _write_catalog(missing_boundary_root), repo_root=missing_boundary_root
    )
    missing_boundary.resolve(
        missing_boundary.dataset("boundary")["entrypoint"]
    ).unlink()
    with pytest.raises(EarthEngineExportError, match="boundary") as captured:
        build_dem_export_plan(missing_boundary, "sample-city", project=None)

    assert isinstance(captured.value.__cause__, FileNotFoundError)


def test_execute_dem_export_preflight_builds_image_without_starting_task(tmp_path):
    from lte_scenario_toolkit.dem_data import execute_dem_export

    plan = _build_plan(tmp_path)
    ee = _FakeEE(image_count=3)
    geemap = _FakeGeemap()

    result = execute_dem_export(plan, start=False, ee_module=ee, geemap_module=geemap)

    assert result.image_count == 3
    assert result.task_id is None
    assert ee.initialized_projects == ["ee-project"]
    assert ee.image_collection_ids == ["USGS/3DEP/1m"]
    assert ee.collection.filtered_roi is geemap.roi
    assert ee.collection.image.operations == [
        ("select", "elevation"),
        ("clip", geemap.roi),
    ]
    assert ee.to_drive.calls == []
    assert ee.task.start_calls == 0
    boundary_frame, geodesic = geemap.calls[0]
    assert boundary_frame.crs.to_epsg() == 4326
    assert list(boundary_frame.columns) == ["geometry"]
    assert geodesic is False


def test_execute_dem_export_starts_one_drive_task_with_exact_parameters(tmp_path):
    from lte_scenario_toolkit.dem_data import execute_dem_export

    plan = _build_plan(tmp_path)
    ee = _FakeEE(image_count=7)
    geemap = _FakeGeemap()

    result = execute_dem_export(plan, start=True, ee_module=ee, geemap_module=geemap)

    assert result.image_count == 7
    assert result.task_id == "task-123"
    assert ee.task.start_calls == 1
    assert ee.to_drive.calls == [
        {
            "image": ee.collection.image,
            "description": "usgs_3dep_1m_sample-city",
            "folder": "lte-scenario-toolkit-dem",
            "fileNamePrefix": "usgs_3dep_1m_sample-city",
            "region": geemap.roi,
            "scale": 1,
            "crs": "EPSG:3857",
            "maxPixels": 1e13,
            "fileDimensions": 8192,
            "shardSize": 256,
            "fileFormat": "GeoTIFF",
            "formatOptions": {"cloudOptimized": True},
            "skipEmptyTiles": False,
        }
    ]


def test_execute_dem_export_rejects_zero_coverage_without_starting(tmp_path):
    from lte_scenario_toolkit.dem_data import EarthEngineExportError, execute_dem_export

    ee = _FakeEE(image_count=0)

    with pytest.raises(EarthEngineExportError, match="no images|zero") as captured:
        execute_dem_export(
            _build_plan(tmp_path),
            start=True,
            ee_module=ee,
            geemap_module=_FakeGeemap(),
        )

    assert captured.value.__cause__ is not None
    assert ee.collection.mosaic_calls == 0
    assert ee.to_drive.calls == []
    assert ee.task.start_calls == 0


def test_execute_dem_export_normalizes_earth_engine_errors_with_cause(tmp_path):
    from lte_scenario_toolkit.dem_data import EarthEngineExportError, execute_dem_export

    cause = PermissionError("credentials denied")
    ee = _FakeEE(initialize_error=cause)

    with pytest.raises(EarthEngineExportError, match="credentials denied") as captured:
        execute_dem_export(
            _build_plan(tmp_path),
            start=False,
            ee_module=ee,
            geemap_module=_FakeGeemap(),
        )

    assert captured.value.__cause__ is cause


def test_execute_dem_export_rejects_changed_boundary_before_any_earth_engine_call(
    tmp_path,
):
    from lte_scenario_toolkit.dem_data import EarthEngineExportError, execute_dem_export

    plan = _build_plan(tmp_path)
    plan.boundary_path.write_bytes(plan.boundary_path.read_bytes() + b"tampered")
    ee = _FakeEE()
    geemap = _FakeGeemap()

    with pytest.raises(EarthEngineExportError, match="checksum|SHA256"):
        execute_dem_export(
            plan,
            start=True,
            ee_module=ee,
            geemap_module=geemap,
        )

    assert ee.initialized_projects == []
    assert ee.image_collection_ids == []
    assert geemap.calls == []
    assert ee.to_drive.calls == []
    assert ee.task.start_calls == 0


def test_write_export_run_creates_complete_unique_run_artifacts(tmp_path):
    from lte_scenario_toolkit.dem_data import DemExportResult, write_export_run

    plan = _build_plan(tmp_path)
    result = DemExportResult(image_count=7, task_id="task-123")
    runs_root = tmp_path / "runs"

    first = write_export_run(plan, result, runs_root=runs_root)
    second = write_export_run(plan, result, runs_root=runs_root)

    assert first.parent == runs_root
    assert first != second
    assert first.name.startswith("20")
    assert first.name.endswith("-sample-city-dem-export")
    assert {path.name for path in first.iterdir()} == {
        "RUN.md",
        "DATA_LAYER.md",
        "sources.md",
        "export-plan.json",
    }

    payload = json.loads((first / "export-plan.json").read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "sample-city"
    assert payload["boundary_path"] == str(plan.boundary_path)
    assert payload["boundary_sha256"] == plan.boundary_sha256
    assert payload["timestamp"].endswith("Z")
    assert payload["image_count"] == 7
    assert payload["task_id"] == "task-123"
    assert "git_commit" in payload
    assert "python" in payload["software_versions"]

    run_text = (first / "RUN.md").read_text(encoding="utf-8")
    layer_text = (first / "DATA_LAYER.md").read_text(encoding="utf-8")
    sources_text = (first / "sources.md").read_text(encoding="utf-8")
    assert all(value in run_text for value in ("Sample City", "7", "task-123"))
    assert all(value in layer_text for value in ("USGS/3DEP/1m", "elevation", "EPSG:3857"))
    assert str(plan.boundary_path) in sources_text
    assert plan.boundary_sha256 in sources_text


def test_write_export_run_removes_staging_and_never_publishes_partial_run(
    tmp_path, monkeypatch
):
    import lte_scenario_toolkit.dem_data as dem_data

    plan = _build_plan(tmp_path)
    result = dem_data.DemExportResult(image_count=7, task_id="task-123")
    runs_root = tmp_path / "runs"
    original_write = dem_data._atomic_write_text
    write_calls = 0

    def fail_second_write(path, text):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 2:
            raise OSError("injected write failure")
        original_write(path, text)

    monkeypatch.setattr(dem_data, "_atomic_write_text", fail_second_write)

    with pytest.raises(dem_data.EarthEngineExportError, match="injected write failure"):
        dem_data.write_export_run(plan, result, runs_root=runs_root)

    assert runs_root.is_dir()
    assert list(runs_root.iterdir()) == []


def test_data_cli_dem_export_dry_run_is_json_only_and_never_imports_gee(
    tmp_path, monkeypatch, capsys
):
    import lte_scenario_toolkit.dem_data as dem_data
    from lte_scenario_toolkit.data_cli import main

    catalog_path = _write_catalog(tmp_path)

    def fail_import():
        raise AssertionError("dry-run imported Earth Engine dependencies")

    monkeypatch.setattr(dem_data, "_import_gee", fail_import)

    result = main(
        [
            "--catalog",
            str(catalog_path),
            "dem",
            "export",
            "sample-city",
            "--project",
            "dry-run-project",
            "--dry-run",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scenario_id"] == "sample-city"
    assert payload["project"] == "dry-run-project"
    assert payload["estimated_pixels"] == 3_000_000
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(("mode", "expected_start"), [([], False), (["--export"], True)])
def test_data_cli_dem_export_online_modes_record_results(
    tmp_path, monkeypatch, capsys, mode, expected_start
):
    import lte_scenario_toolkit.data_cli as data_cli
    from lte_scenario_toolkit.dem_data import DemExportResult

    catalog_path = _write_catalog(tmp_path)
    starts = []
    recorded = []
    run_path = tmp_path / "runs" / "recorded-run"

    def fake_execute(plan, *, start):
        starts.append((plan, start))
        return DemExportResult(
            image_count=5,
            task_id="task-123" if start else None,
        )

    def fake_write(plan, result, *, runs_root):
        recorded.append((plan, result, runs_root))
        return run_path

    monkeypatch.setattr(data_cli, "execute_dem_export", fake_execute)
    monkeypatch.setattr(data_cli, "write_export_run", fake_write)

    result = data_cli.main(
        [
            "--catalog",
            str(catalog_path),
            "dem",
            "export",
            "sample-city",
            "--project",
            "ee-project",
            *mode,
        ]
    )

    assert result == 0
    assert starts[0][1] is expected_start
    assert recorded[0][2] == tmp_path / "runs"
    output = capsys.readouterr().out
    assert "Image count: 5" in output
    assert "Task ID:" in output
    assert f"Run path: {run_path}" in output


def test_inspect_dem_shards_returns_sorted_complete_grid_metadata(tmp_path):
    from lte_scenario_toolkit.dem_data import inspect_dem_shards

    tiles = tmp_path / "tiles"
    _write_dem_tile(
        tiles / "dem_01.tif",
        left=0,
        top=4,
        values=np.full((2, 2), 1, dtype=np.float32),
    )
    _write_dem_tile(
        tiles / "dem_00.tif",
        left=0,
        top=2,
        values=np.full((2, 2), 2, dtype=np.float32),
    )
    _write_dem_tile(
        tiles / "dem_11.tif",
        left=2,
        top=2,
        values=np.full((2, 2), 3, dtype=np.float32),
    )
    _write_dem_tile(
        tiles / "dem_10.tif",
        left=2,
        top=4,
        values=np.full((2, 2), 4, dtype=np.float32),
    )

    shards = inspect_dem_shards(tiles, prefix="dem_")

    assert tuple(path.name for path in shards.paths) == (
        "dem_00.tif",
        "dem_01.tif",
        "dem_10.tif",
        "dem_11.tif",
    )
    assert shards.crs == "EPSG:3857"
    assert shards.resolution == (1.0, 1.0)
    assert shards.dtype == "float32"
    assert shards.nodata == -9999.0
    assert shards.count == 1
    assert shards.bounds == (0.0, 0.0, 4.0, 4.0)


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        ("missing", "missing"),
        ("overlap", "overlap"),
    ],
)
def test_inspect_dem_shards_rejects_missing_or_overlapping_grid(tmp_path, layout, message):
    from lte_scenario_toolkit.dem_data import DemIngestError, inspect_dem_shards

    tiles = tmp_path / "tiles"
    _write_dem_tile(tiles / "dem_0.tif", left=0, top=2)
    _write_dem_tile(tiles / "dem_1.tif", left=2 if layout == "missing" else 1, top=2)
    if layout == "missing":
        _write_dem_tile(tiles / "dem_2.tif", left=0, top=0)

    with pytest.raises(DemIngestError, match=message):
        inspect_dem_shards(tiles, prefix="dem_")


def test_inspect_dem_shards_distinguishes_nan_nodata_from_absent_nodata(tmp_path):
    from lte_scenario_toolkit.dem_data import DemIngestError, inspect_dem_shards

    tiles = tmp_path / "tiles"
    _write_dem_tile(tiles / "dem_a.tif", left=0, top=2, nodata=float("nan"))
    _write_dem_tile(tiles / "dem_b.tif", left=2, top=2, nodata=None)

    with pytest.raises(DemIngestError, match="nodata"):
        inspect_dem_shards(tiles, prefix="dem_")


def test_merge_dem_shards_preserves_all_invalid_internal_mask_without_nodata(tmp_path):
    from lte_scenario_toolkit.dem_data import (
        DemIngestError,
        inspect_dem_shards,
        merge_dem_shards,
        validate_dem_coverage,
    )

    tiles = tmp_path / "tiles"
    path = _write_dem_tile(tiles / "dem_masked.tif", left=0, top=2, nodata=None)
    with rasterio.open(path, "r+") as dataset:
        dataset.write_mask(np.zeros((2, 2), dtype=np.uint8))

    shards = inspect_dem_shards(tiles, prefix="dem_")
    output = tmp_path / "merged" / "dem.tif"
    output.parent.mkdir()
    merge_dem_shards(shards, output)
    with rasterio.open(output) as dataset:
        assert dataset.nodata == np.finfo(np.float32).min
        assert np.ma.getmaskarray(dataset.read(1, masked=True)).all()

    boundary = tmp_path / "boundary.geojson"
    gpd.GeoDataFrame({"name": ["city"]}, geometry=[box(0, 0, 2, 2)], crs="EPSG:3857").to_file(boundary, driver="GeoJSON")
    with pytest.raises(DemIngestError, match="no valid"):
        validate_dem_coverage(
            output, boundary, expected_crs="EPSG:3857", expected_resolution=(1.0, 1.0)
        )


def test_merge_dem_shards_preserves_valid_values_next_to_masked_pixels(tmp_path):
    from lte_scenario_toolkit.dem_data import inspect_dem_shards, merge_dem_shards

    tiles = tmp_path / "tiles"
    path = _write_dem_tile(
        tiles / "dem_mixed.tif",
        left=0,
        top=2,
        values=np.array([[1, 2], [3, 4]], dtype=np.float32),
        nodata=None,
    )
    with rasterio.open(path, "r+") as dataset:
        dataset.write_mask(np.array([[0, 255], [255, 255]], dtype=np.uint8))
    shards = inspect_dem_shards(tiles, prefix="dem_")
    output = tmp_path / "merged" / "dem.tif"
    output.parent.mkdir()
    merge_dem_shards(shards, output)
    with rasterio.open(output) as dataset:
        values = dataset.read(1, masked=True)
        assert values.mask.tolist() == [[True, False], [False, False]]
        assert values[0, 1] == pytest.approx(2)


def test_inspect_dem_shards_rejects_even_small_rotation(tmp_path):
    from lte_scenario_toolkit.dem_data import DemIngestError, inspect_dem_shards

    tiles = tmp_path / "tiles"
    path = tiles / "dem_rotated.tif"
    path.parent.mkdir()
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=Affine(1.0, 1e-9, 0.0, 0.0, -1.0, 2.0),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(np.ones((2, 2), dtype=np.float32), 1)

    with pytest.raises(DemIngestError, match="north-up|rotat"):
        inspect_dem_shards(tiles, prefix="dem_")


def test_merge_dem_shards_writes_tiled_compressed_raster_with_overviews(tmp_path):
    from lte_scenario_toolkit.dem_data import inspect_dem_shards, merge_dem_shards

    tiles = tmp_path / "tiles"
    _write_dem_tile(
        tiles / "dem_a.tif", left=0, top=2, values=np.array([[1, 2], [3, 4]], dtype=np.float32)
    )
    _write_dem_tile(
        tiles / "dem_b.tif", left=2, top=2, values=np.array([[5, 6], [7, 8]], dtype=np.float32)
    )
    shards = inspect_dem_shards(tiles, prefix="dem_")
    output = tmp_path / "merged" / "dem.tif"
    output.parent.mkdir()
    merge_dem_shards(shards, output)

    with rasterio.open(output) as dataset:
        assert dataset.shape == (2, 4)
        assert dataset.read(1).tolist() == [[1, 2, 5, 6], [3, 4, 7, 8]]
        assert dataset.compression.name.lower() == "lzw"
        assert dataset.overviews(1) == [2]
        assert dataset.tags(ns="rio_overview")["resampling"] == "average"


def test_validate_dem_coverage_checks_only_intersecting_blocks_and_nodata(tmp_path):
    from lte_scenario_toolkit.dem_data import validate_dem_coverage

    raster = _write_dem_tile(
        tmp_path / "dem.tif",
        left=0,
        top=4,
        values=np.array(
            [[-9999, -9999, -9999, -9999], [-9999, 2, -9999, -9999], [-9999, -9999, -9999, -9999], [-9999, -9999, -9999, -9999]],
            dtype=np.float32,
        ),
    )
    boundary = tmp_path / "boundary.geojson"
    gpd.GeoDataFrame({"name": ["city"]}, geometry=[box(1, 1, 3, 3)], crs="EPSG:3857").to_file(boundary, driver="GeoJSON")

    assert validate_dem_coverage(
        raster, boundary, expected_crs="EPSG:3857", expected_resolution=(1.0, 1.0)
    ) is None


def test_validate_dem_coverage_rejects_boundary_outside_raster_extent(tmp_path):
    from lte_scenario_toolkit.dem_data import DemIngestError, validate_dem_coverage

    raster = _write_dem_tile(
        tmp_path / "dem.tif",
        left=0,
        top=4,
        values=np.ones((4, 4), dtype=np.float32),
    )
    boundary = tmp_path / "boundary.geojson"
    gpd.GeoDataFrame({"name": ["city"]}, geometry=[box(-0.1, 1, 1, 3)], crs="EPSG:3857").to_file(boundary, driver="GeoJSON")

    with pytest.raises(DemIngestError, match="bounds|cover"):
        validate_dem_coverage(
            raster, boundary, expected_crs="EPSG:3857", expected_resolution=(1.0, 1.0)
        )


def test_ingest_dem_shards_merges_updates_catalog_and_manifest(tmp_path):
    from lte_scenario_toolkit.dem_data import ingest_dem_shards

    catalog_path = _write_catalog(tmp_path)
    boundary_path = tmp_path / "boundary_shp" / "sample-city" / "sample-city.shp"
    boundary_path.unlink()
    for sidecar in boundary_path.parent.glob("sample-city.*"):
        sidecar.unlink()
    gpd.GeoDataFrame({"name": ["city"]}, geometry=[box(0, 0, 4, 4)], crs="EPSG:3857").to_file(boundary_path)
    tiles = tmp_path / "downloaded"
    _write_dem_tile(tiles / "usgs_3dep_1m_sample-city_a.tif", left=0, top=4, values=np.ones((4, 2), dtype=np.float32))
    _write_dem_tile(tiles / "usgs_3dep_1m_sample-city_b.tif", left=2, top=4, values=np.full((4, 2), 2, dtype=np.float32))

    catalog = ingest_dem_shards(load_data_catalog(catalog_path, repo_root=tmp_path), "sample-city", tiles)

    dem = catalog.dataset("dem")
    assert catalog.resolve(dem["entrypoint"]).is_file()
    assert dem["external"] is True
    assert dem["dtype"] == "float32"
    assert dem["width"] == 4
    assert dem["height"] == 4
    assert dem["download_date"]
    assert "pending export" in dem["notes"]
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert {item["dataset_id"] for item in manifest["datasets"]} == {"boundary", "dem"}


def test_data_cli_dem_ingest_prints_final_dem_and_maps_ingest_errors(tmp_path, monkeypatch, capsys):
    import lte_scenario_toolkit.data_cli as data_cli
    from lte_scenario_toolkit.dem_data import DemIngestError

    catalog_path = _write_catalog(tmp_path)
    calls = []

    def fake_ingest(catalog, scenario_id, tiles_dir):
        calls.append((catalog, scenario_id, tiles_dir))
        return catalog

    monkeypatch.setattr(data_cli, "ingest_dem_shards", fake_ingest)
    assert data_cli.main(
        [
            "--catalog",
            str(catalog_path),
            "dem",
            "ingest",
            "sample-city",
            "--tiles-dir",
            str(tmp_path / "tiles"),
        ]
    ) == 0
    assert calls[0][1:] == ("sample-city", tmp_path / "tiles")
    assert "Final DEM:" in capsys.readouterr().out

    def fail_ingest(*args):
        raise DemIngestError("invalid shard set")

    monkeypatch.setattr(data_cli, "ingest_dem_shards", fail_ingest)
    assert data_cli.main(
        [
            "--catalog",
            str(catalog_path),
            "dem",
            "ingest",
            "sample-city",
            "--tiles-dir",
            str(tmp_path / "tiles"),
        ]
    ) == 2
    assert "invalid shard set" in capsys.readouterr().err


def test_ingest_dem_shards_rolls_back_owned_files_when_manifest_update_fails(
    tmp_path, monkeypatch
):
    import lte_scenario_toolkit.dem_data as dem_data
    from lte_scenario_toolkit.dem_data import ingest_dem_shards

    catalog_path = _write_catalog(tmp_path)
    boundary_path = tmp_path / "boundary_shp" / "sample-city" / "sample-city.shp"
    for sidecar in boundary_path.parent.glob("sample-city.*"):
        sidecar.unlink()
    gpd.GeoDataFrame({"name": ["city"]}, geometry=[box(0, 0, 4, 4)], crs="EPSG:3857").to_file(boundary_path)
    manifest_path = tmp_path / "data" / "manifest.json"
    original_manifest = b'{"original": true}\n'
    manifest_path.write_bytes(original_manifest)
    original_catalog = catalog_path.read_bytes()
    tiles = tmp_path / "downloaded"
    tile_a = _write_dem_tile(tiles / "usgs_3dep_1m_sample-city_a.tif", left=0, top=4, values=np.ones((4, 2), dtype=np.float32))
    tile_b = _write_dem_tile(tiles / "usgs_3dep_1m_sample-city_b.tif", left=2, top=4, values=np.ones((4, 2), dtype=np.float32))

    def fail_manifest(*args, **kwargs):
        manifest_path.write_text("partial", encoding="utf-8")
        raise RuntimeError("manifest failed")

    monkeypatch.setattr(dem_data, "update_data_manifest", fail_manifest)
    with pytest.raises(dem_data.DemIngestError, match="manifest failed"):
        ingest_dem_shards(load_data_catalog(catalog_path, repo_root=tmp_path), "sample-city", tiles)

    assert catalog_path.read_bytes() == original_catalog
    assert manifest_path.read_bytes() == original_manifest
    assert not (tmp_path / "dem" / "sample-city" / "usgs_3dep_1m_sample-city.tif").exists()
    assert tile_a.is_file() and tile_b.is_file()
    assert not (tmp_path / ".lte-data" / "catalog.lock").exists()


def test_ingest_dem_shards_reloads_catalog_only_after_lock_and_rejects_existing_symlink(
    tmp_path, monkeypatch
):
    import lte_scenario_toolkit.dem_data as dem_data
    from lte_scenario_toolkit.dem_data import DemIngestError, ingest_dem_shards

    catalog_path = _write_catalog(tmp_path)
    loaded = load_data_catalog(catalog_path, repo_root=tmp_path)
    lock_path = tmp_path / ".lte-data" / "catalog.lock"
    real_load = dem_data.load_data_catalog
    calls = []

    def assert_locked(path, **kwargs):
        calls.append(path)
        assert lock_path.is_file()
        return real_load(path, **kwargs)

    monkeypatch.setattr(dem_data, "load_data_catalog", assert_locked)
    destination = tmp_path / "dem" / "sample-city" / "usgs_3dep_1m_sample-city.tif"
    destination.parent.mkdir(parents=True)
    try:
        destination.symlink_to(tmp_path / "missing-target.tif")
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(DemIngestError, match="already exists"):
        ingest_dem_shards(loaded, "sample-city", tmp_path / "tiles")
    assert calls == [catalog_path]
    assert destination.is_symlink()
    assert not lock_path.exists()
