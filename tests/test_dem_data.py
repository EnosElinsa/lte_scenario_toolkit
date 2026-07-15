import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pytest
import yaml
from shapely.geometry import box

from lte_scenario_toolkit.data_catalog import load_data_catalog


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
