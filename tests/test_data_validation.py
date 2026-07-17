"""Focused tests for scenario data validation and the validate CLI command."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from shapely.geometry import Point, Polygon

from lte_scenario_toolkit.data_catalog import load_data_catalog
from lte_scenario_toolkit.data_cli import main
from lte_scenario_toolkit.data_validation import validate_scenario_data


def _write_boundary(root: Path, city: str = "city", *, crs: str = "EPSG:3857") -> Path:
    directory = root / "boundary_shp" / city
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{city}.shp"
    frame = gpd.GeoDataFrame(
        {"name": [city]},
        geometry=[Polygon([(1, 1), (9, 1), (9, 9), (1, 9)])],
        crs=crs,
    )
    frame.to_file(path, driver="ESRI Shapefile")
    return path


def _write_points(root: Path) -> Path:
    directory = root / "points_shp" / "points"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "points.shp"
    frame = gpd.GeoDataFrame(
        {"cell": [1]}, geometry=[Point(2, 2)], crs="EPSG:3857"
    )
    frame.to_file(path, driver="ESRI Shapefile")
    return path


def _write_dem(root: Path, city: str = "city") -> Path:
    path = root / "dem" / city / "elevation.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 10, 1, 1),
    ) as raster:
        raster.write(np.ones((1, 10, 10), dtype=np.float32))
    return path


def _file_record(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _dataset_files(root: Path, directory: Path) -> list[dict[str, object]]:
    return [_file_record(root, path) for path in sorted(directory.rglob("*")) if path.is_file()]


def _catalog_document(
    root: Path,
    *,
    dem: bool = False,
    config_path: str | None = None,
    second_city: bool = False,
) -> dict[str, object]:
    boundary = {
        "dataset_id": "boundary_city",
        "role": "boundary",
        "path": "boundary_shp/city",
        "entrypoint": "boundary_shp/city/city.shp",
        "source_url": None,
        "provider": "test",
        "license": "CC0-1.0",
        "download_date": None,
        "crs": "EPSG:3857",
        "spatial_resolution": "polygon vector",
        "notes": "fixture",
        "geometry_type": "Polygon",
        "feature_count": 1,
        "redistribution_confirmed": True,
    }
    datasets: list[dict[str, object]] = [boundary]
    scenario_dem: str | None = None
    if dem:
        scenario_dem = "dem_city"
        datasets.append(
            {
                "dataset_id": scenario_dem,
                "role": "dem",
                "path": "dem/city",
                "entrypoint": "dem/city/elevation.tif",
                "source_url": None,
                "provider": "test",
                "license": "CC0-1.0",
                "download_date": None,
                "crs": "EPSG:3857",
                "spatial_resolution": "1 m",
                "notes": "fixture",
                "external": True,
                "earth_engine_collection": "TEST/DEM",
                "band": "elevation",
                "units": "metres",
                "vertical_datum": "test",
                "native_scale_m": 1,
                "export_crs": "EPSG:3857",
                "export_prefix": "elevation",
                "drive_folder": "test",
            }
        )
    scenarios: list[dict[str, object]] = [
        {
            "scenario_id": "city",
            "display_name": "City",
            "boundary_dataset_id": "boundary_city",
            "dem_dataset_id": scenario_dem,
            "config_path": config_path,
        }
    ]
    if second_city:
        boundary_two = dict(boundary)
        boundary_two.update(
            {
                "dataset_id": "boundary_other",
                "path": "boundary_shp/other",
                "entrypoint": "boundary_shp/other/other.shp",
            }
        )
        datasets.append(boundary_two)
        scenarios.append(
            {
                "scenario_id": "other",
                "display_name": "Other",
                "boundary_dataset_id": "boundary_other",
                "dem_dataset_id": None,
                "config_path": None,
            }
        )
    return {"schema_version": 2, "datasets": datasets, "scenarios": scenarios}


def _write_catalog(
    tmp_path: Path,
    *,
    dem: bool = False,
    config_path: str | None = None,
    second_city: bool = False,
    manifest: bool = True,
) -> tuple[Path, object]:
    _write_boundary(tmp_path)
    _write_points(tmp_path)
    if second_city:
        _write_boundary(tmp_path, "other")
    if dem:
        _write_dem(tmp_path)
    document = _catalog_document(
        tmp_path,
        dem=dem,
        config_path=config_path,
        second_city=second_city,
    )
    catalog_path = tmp_path / "data" / "datasets.yaml"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    catalog = load_data_catalog(catalog_path, repo_root=tmp_path)
    if manifest:
        records = []
        for dataset in document["datasets"]:
            record = dict(dataset)
            dataset_path = tmp_path / str(dataset["path"])
            record["files"] = (
                _dataset_files(tmp_path, dataset_path) if dataset_path.exists() else []
            )
            records.append(record)
        manifest_payload = {
            "schema_version": 2,
            "generated_at": "2026-01-01T00:00:00Z",
            "datasets": records,
            "scenarios": document["scenarios"],
        }
        (tmp_path / "data" / "manifest.json").write_text(
            json.dumps(manifest_payload), encoding="utf-8"
        )
    return catalog_path, catalog


def _rewrite_manifest(catalog_root: Path, datasets: list[dict[str, object]]) -> None:
    path = catalog_root / "data" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["datasets"] = datasets
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pending_dem_is_warning_and_report_is_ok(tmp_path):
    _, catalog = _write_catalog(tmp_path, dem=True)
    shutil.rmtree(tmp_path / "dem" / "city")
    manifest_path = tmp_path / "data" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    dem_record = next(item for item in payload["datasets"] if item["dataset_id"] == "dem_city")
    # A previously-ready external DEM can leave stale metadata and file
    # records behind after the local raster is removed.  Pending validation
    # must not turn those cached records into missing-file failures.
    dem_record["provider"] = "cached provider"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_scenario_data(catalog, "city")

    assert report.status == "dem-pending"
    assert report.ok
    assert any(message.code == "dem.pending" for message in report.messages)
    assert [message for message in report.messages if message.level == "error"] == []


def test_ready_dem_passes_without_full_checksum(tmp_path):
    _, catalog = _write_catalog(tmp_path, dem=True)
    report = validate_scenario_data(catalog, "city")

    assert report.status == "ready"
    assert report.ok
    assert not [message for message in report.messages if message.level == "error"]


def test_undeclared_dem_is_a_nonblocking_warning(tmp_path):
    _, catalog = _write_catalog(tmp_path, dem=False)

    report = validate_scenario_data(catalog, "city")

    assert report.ok
    assert report.status == "boundary-ready"
    assert any(message.code == "dem.undeclared" for message in report.messages)


def test_fast_mode_reports_size_drift_without_hashing(tmp_path, monkeypatch):
    _, catalog = _write_catalog(tmp_path)
    boundary = catalog.resolve("boundary_shp/city/city.shp")
    boundary.write_bytes(boundary.read_bytes() + b"x")

    def fail_hash(*args, **kwargs):
        raise AssertionError("fast validation must not hash files")

    monkeypatch.setattr("lte_scenario_toolkit.data_validation.sha256_file", fail_hash)
    report = validate_scenario_data(catalog, "city")

    assert not report.ok
    assert any(message.code == "manifest.size" for message in report.messages)


def test_full_mode_reports_same_size_hash_drift(tmp_path):
    _, catalog = _write_catalog(tmp_path)
    boundary = catalog.resolve("boundary_shp/city/city.shp")
    original = boundary.read_bytes()
    boundary.write_bytes(bytes((original[0] ^ 1,)) + original[1:])
    assert boundary.stat().st_size == len(original)

    report = validate_scenario_data(catalog, "city", full_checksum=True)

    assert not report.ok
    assert any(message.code == "manifest.sha256" for message in report.messages)


def test_unrelated_manifest_files_are_not_stat_or_hashed(tmp_path, monkeypatch):
    _, catalog = _write_catalog(tmp_path, second_city=True)
    manifest_path = tmp_path / "data" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    other = next(item for item in payload["datasets"] if item["dataset_id"] == "boundary_other")
    other["files"] = [
        {"path": "boundary_shp/other/not-local.shp", "size_bytes": 1, "sha256": "0" * 64}
    ]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    from lte_scenario_toolkit import data_validation as validation_module

    real_hash = validation_module.sha256_file
    hashed_paths: list[Path] = []

    def record_hash(path, **kwargs):
        resolved = Path(path).resolve()
        hashed_paths.append(resolved)
        if resolved.name == "not-local.shp":
            raise AssertionError("unrelated dataset must not be hashed")
        return real_hash(resolved, **kwargs)

    monkeypatch.setattr("lte_scenario_toolkit.data_validation.sha256_file", record_hash)
    report = validate_scenario_data(catalog, "city", full_checksum=True)

    assert report.ok
    assert not any(message.code.startswith("manifest.missing") for message in report.messages)
    assert all(path.name != "not-local.shp" for path in hashed_paths)


def test_unrelated_known_record_cannot_substitute_another_dataset_file(tmp_path):
    _, catalog = _write_catalog(tmp_path, second_city=True)
    manifest_path = tmp_path / "data" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    city_record = next(item for item in payload["datasets"] if item["dataset_id"] == "boundary_city")
    other = next(item for item in payload["datasets"] if item["dataset_id"] == "boundary_other")
    substituted = next(
        item for item in city_record["files"] if item["path"].endswith(".shp")
    )
    other["files"] = [dict(substituted)]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_scenario_data(catalog, "city")

    assert not report.ok
    assert any(message.code == "manifest.traversal" for message in report.messages)


def test_unrelated_manifest_structure_is_still_checked(tmp_path):
    _, catalog = _write_catalog(tmp_path, second_city=True)
    manifest_path = tmp_path / "data" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    other = next(item for item in payload["datasets"] if item["dataset_id"] == "boundary_other")
    other["files"] = [{"path": "boundary_shp/other/other.shp"}]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_scenario_data(catalog, "city")

    assert not report.ok
    assert any(message.code == "manifest.file.malformed" for message in report.messages)


@pytest.mark.parametrize("dataset_id", ["boundary_city", "dem_city"])
def test_present_selected_dataset_requires_manifest_entrypoint(tmp_path, dataset_id):
    _, catalog = _write_catalog(tmp_path, dem=True)
    manifest_path = tmp_path / "data" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in payload["datasets"] if item["dataset_id"] == dataset_id)
    record["files"] = []
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_scenario_data(catalog, "city")

    assert not report.ok
    assert any(message.code == "manifest.entrypoint" for message in report.messages)


def test_non_path_manifest_metadata_drift_is_reported(tmp_path):
    _, catalog = _write_catalog(tmp_path)
    manifest_path = tmp_path / "data" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in payload["datasets"] if item["dataset_id"] == "boundary_city")
    record["provider"] = "different provider"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_scenario_data(catalog, "city")

    assert not report.ok
    assert any(message.code == "manifest.metadata" for message in report.messages)


def test_manifest_scenario_mapping_drift_is_reported(tmp_path):
    _, catalog = _write_catalog(tmp_path)
    manifest_path = tmp_path / "data" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["scenarios"][0]["display_name"] = "stale name"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_scenario_data(catalog, "city")

    assert not report.ok
    assert any(message.code == "manifest.scenarios" for message in report.messages)


def test_malformed_manifest_scenario_mapping_is_reported(tmp_path):
    _, catalog = _write_catalog(tmp_path)
    manifest_path = tmp_path / "data" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["scenarios"] = {"city": "stale"}
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_scenario_data(catalog, "city")

    assert not report.ok
    assert any(message.code == "manifest.scenarios" for message in report.messages)


def test_present_dem_metadata_drift_is_still_checked(tmp_path):
    _, catalog = _write_catalog(tmp_path, dem=True)
    manifest_path = tmp_path / "data" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in payload["datasets"] if item["dataset_id"] == "dem_city")
    record["provider"] = "different provider"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_scenario_data(catalog, "city")

    assert not report.ok
    assert any(message.code == "manifest.metadata" for message in report.messages)


@pytest.mark.parametrize(
    ("mutator", "code"),
    [
        (lambda payload: payload.update({"schema_version": 1}), "manifest.schema"),
        (
            lambda payload: payload["datasets"].append(dict(payload["datasets"][0])),
            "manifest.duplicate",
        ),
        (
            lambda payload: payload["datasets"][0]["files"].__setitem__(
                0, {"path": "../outside", "size_bytes": 0, "sha256": "0" * 64}
            ),
            "manifest.traversal",
        ),
    ],
)
def test_malformed_duplicate_and_traversal_manifest_errors(tmp_path, mutator, code):
    _, catalog = _write_catalog(tmp_path)
    path = tmp_path / "data" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_scenario_data(catalog, "city")

    assert not report.ok
    assert any(message.code == code for message in report.messages)


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ("crs", "boundary.crs"),
        ("count", "boundary.count"),
        ("type", "boundary.geometry_type"),
        ("sidecar", "boundary.sidecar"),
    ],
)
def test_boundary_contract_errors_are_normalized(tmp_path, change, code):
    _, catalog = _write_catalog(tmp_path)
    if change == "crs":
        _write_boundary(tmp_path, crs="EPSG:4326")
    elif change == "count":
        document = yaml.safe_load(catalog.path.read_text(encoding="utf-8"))
        document["datasets"][0]["feature_count"] = 2
        catalog.path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        catalog = load_data_catalog(catalog.path, repo_root=tmp_path)
    elif change == "type":
        path = catalog.resolve("boundary_shp/city/city.shp")
        frame = gpd.GeoDataFrame({"name": ["city"]}, geometry=[Point(2, 2)], crs="EPSG:3857")
        frame.to_file(path, driver="ESRI Shapefile")
    else:
        catalog.resolve("boundary_shp/city/city.dbf").unlink()

    report = validate_scenario_data(catalog, "city")

    assert not report.ok
    assert any(message.code == code for message in report.messages)


def test_boundary_sidecar_extension_matching_is_case_insensitive(tmp_path):
    _, catalog = _write_catalog(tmp_path)
    cpg = catalog.resolve("boundary_shp/city/city.cpg")
    uppercase_cpg = cpg.with_suffix(".CPG")
    cpg.rename(uppercase_cpg)
    manifest_path = tmp_path / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    boundary_record = next(
        item for item in manifest["datasets"] if item["dataset_id"] == "boundary_city"
    )
    cpg_record = next(
        item for item in boundary_record["files"] if item["path"].endswith("/city.cpg")
    )
    cpg_record["path"] = uppercase_cpg.relative_to(tmp_path).as_posix()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_scenario_data(catalog, "city")

    assert report.ok


def test_config_boundary_and_dem_mismatches_are_reported(tmp_path):
    config_path = "configs/city.yaml"
    _, catalog = _write_catalog(tmp_path, dem=True, config_path=config_path)
    config = {
        "experiment": {"name": "city"},
        "inputs": {
            "points_root": "points_shp",
            "points_layer": "points",
            "boundary_root": "boundary_shp",
            "city": "other",
            "dem_path": "dem/other/elevation.tif",
        },
        "spatial": {
            "target_crs": "EPSG:3857",
            "rectangle_size_m": 10,
            "target_base_station_count": 1,
            "count_tolerance": 0,
        },
        "scan": {
            "strategy": "uniform",
            "step_m": 1,
            "max_rectangles": 1,
            "minimum_center_spacing_m": 1,
        },
        "outputs": {"root": "results/city"},
    }
    (tmp_path / "configs").mkdir()
    (tmp_path / config_path).write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_boundary(tmp_path, "other")

    report = validate_scenario_data(catalog, "city")

    assert not report.ok
    assert {message.code for message in report.messages} >= {
        "config.boundary",
        "config.dem",
    }
    assert not (tmp_path / "results").exists()


def test_schema_v2_linked_profile_uses_catalog_owned_boundary_and_dem(tmp_path):
    config_path = "configs/city/default.yaml"
    catalog_path, _ = _write_catalog(tmp_path, dem=True, config_path=config_path)
    catalog_document = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    points_dataset = {
        "dataset_id": "points",
        "role": "points",
        "path": "points_shp/points",
        "entrypoint": "points_shp/points/points.shp",
        "source_url": None,
        "provider": "test",
        "license": "CC0-1.0",
        "download_date": None,
        "crs": "EPSG:3857",
        "spatial_resolution": "point vector",
        "notes": "fixture",
    }
    catalog_document["datasets"].append(points_dataset)
    catalog_path.write_text(
        yaml.safe_dump(catalog_document, sort_keys=False),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["datasets"].append(
        {
            **points_dataset,
            "files": _dataset_files(tmp_path, tmp_path / "points_shp" / "points"),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    profile = {
        "schema_version": 2,
        "profile": {
            "id": "default",
            "display_name": "Default",
            "scenario_id": "city",
        },
        "inputs": {"points_dataset_id": "points"},
        "experiment": {"random_seed": 7},
        "spatial": {
            "target_crs": "EPSG:3857",
            "rectangle_size_m": 2,
            "target_base_station_count": 1,
            "count_tolerance": 0,
        },
        "scan": {
            "mode": "fast",
            "strategy": "uniform",
            "step_m": 1,
            "max_rectangles": 1,
            "minimum_center_spacing_m": 2,
        },
        "outputs": {"root": "results"},
        "figures": {"preset": "publication"},
    }
    profile_path = tmp_path / config_path
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    catalog = load_data_catalog(catalog_path, repo_root=tmp_path)

    report = validate_scenario_data(catalog, "city")

    assert report.ok
    assert not any(message.code.startswith("config.") for message in report.messages)
    assert not (tmp_path / "results").exists()

    points_dbf = tmp_path / "points_shp" / "points" / "points.dbf"
    points_dbf.write_bytes(points_dbf.read_bytes() + b"drift")
    drifted = validate_scenario_data(catalog, "city", dataset_ids=("points",))
    assert not drifted.ok
    assert any(message.code == "manifest.size" for message in drifted.messages)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    points_record = next(
        record for record in manifest["datasets"] if record["dataset_id"] == "points"
    )
    for file_record in points_record["files"]:
        if file_record["path"].endswith("points.dbf"):
            file_record["size_bytes"] = points_dbf.stat().st_size
    points_shp = tmp_path / "points_shp" / "points" / "points.shp"
    points_record["files"] = [
        file_record
        for file_record in points_record["files"]
        if file_record["path"] != points_shp.relative_to(tmp_path).as_posix()
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    points_shp.unlink()

    missing_entrypoint = validate_scenario_data(
        catalog,
        "city",
        dataset_ids=("points",),
    )
    assert not missing_entrypoint.ok
    assert any(
        message.code == "manifest.entrypoint"
        for message in missing_entrypoint.messages
    )


def test_validate_cli_single_all_and_argument_errors(tmp_path, capsys):
    catalog_path, _ = _write_catalog(tmp_path, second_city=True)

    assert main(["--catalog", str(catalog_path), "validate", "city"]) == 0
    single_output = capsys.readouterr().out
    assert single_output.splitlines()[0] == "city: boundary-ready (ok)"

    assert main(["--catalog", str(catalog_path), "validate", "--all"]) == 0
    all_output = capsys.readouterr().out.splitlines()
    assert [line for line in all_output if ": boundary-ready (ok)" in line] == [
        "city: boundary-ready (ok)",
        "other: boundary-ready (ok)",
    ]

    assert main(["--catalog", str(catalog_path), "validate"]) == 2
    capsys.readouterr()
    assert main(["--catalog", str(catalog_path), "validate", "city", "--all"]) == 2


def test_validate_cli_returns_one_for_failed_report(tmp_path, capsys):
    catalog_path, catalog = _write_catalog(tmp_path)
    boundary = catalog.resolve("boundary_shp/city/city.shp")
    boundary.write_bytes(boundary.read_bytes() + b"drift")

    assert main(["--catalog", str(catalog_path), "validate", "city"]) == 1
    output = capsys.readouterr().out
    assert "city: boundary-ready (failed)" in output
    assert "ERROR manifest.size:" in output
