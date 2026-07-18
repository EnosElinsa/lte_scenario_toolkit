from __future__ import annotations

import hashlib
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import lte_scenario_toolkit.data_catalog as data_catalog_module
from lte_scenario_toolkit.data_catalog import (
    CatalogError,
    ConcurrentCatalogUpdateError,
    catalog_transaction_lock,
    load_data_catalog,
    save_data_catalog,
    update_data_manifest,
)


def _base_dataset(dataset_id: str, role: str, path: str, entrypoint: str) -> dict:
    return {
        "dataset_id": dataset_id,
        "role": role,
        "path": path,
        "entrypoint": entrypoint,
        "source_url": "https://example.test/data",
        "provider": "Example provider",
        "license": "CC0-1.0",
        "download_date": "2026-07-16",
        "crs": "EPSG:4326",
        "spatial_resolution": "fixture",
        "notes": "test fixture",
    }


def catalog_document(*, dem_dataset_id: str | None = "dem") -> dict:
    boundary = _base_dataset(
        "boundary",
        "boundary",
        "inputs/boundary",
        "inputs/boundary/boundary.geojson",
    )
    boundary.update(
        {
            "geometry_type": "Polygon",
            "feature_count": 1,
            "redistribution_confirmed": True,
        }
    )
    dem = _base_dataset("dem", "dem", "inputs/dem", "inputs/dem/elevation.tif")
    dem.update(
        {
            "external": True,
            "earth_engine_collection": "EXAMPLE/DEM",
            "band": "elevation",
            "units": "metres",
            "vertical_datum": "NAVD88",
            "native_scale_m": 1,
            "export_crs": "EPSG:3857",
            "export_prefix": "fixture-dem",
            "drive_folder": "fixture-exports",
        }
    )
    return {
        "datasets": [boundary, dem],
        "scenarios": [
            {
                "scenario_id": "test-city",
                "display_name": "Test City",
                "boundary_dataset_id": "boundary",
                "dem_dataset_id": dem_dataset_id,
                "config_path": "configs/test-city.yaml",
            }
        ],
    }


def write_catalog(tmp_path: Path, document: dict | None = None) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    path = data_dir / "datasets.yaml"
    path.write_text(
        yaml.safe_dump(document or catalog_document(), sort_keys=False),
        encoding="utf-8",
    )
    return path


def create_entrypoints(tmp_path: Path) -> None:
    boundary = tmp_path / "inputs" / "boundary" / "boundary.geojson"
    boundary.parent.mkdir(parents=True)
    boundary.write_text("boundary", encoding="utf-8")
    dem = tmp_path / "inputs" / "dem" / "elevation.tif"
    dem.parent.mkdir(parents=True)
    dem.write_bytes(b"dem")
    config = tmp_path / "configs" / "test-city.yaml"
    config.parent.mkdir()
    config.write_text("experiment: test", encoding="utf-8")


def test_catalog_transaction_lock_uses_shared_staging_root_and_cleans_up(tmp_path):
    lock_path = tmp_path / ".lte-data" / "catalog.lock"

    with catalog_transaction_lock(tmp_path) as staging_root:
        assert staging_root == tmp_path / ".lte-data"
        assert lock_path.is_file()
        with pytest.raises(CatalogError, match="lock|progress"):
            with catalog_transaction_lock(tmp_path):
                pass

    assert not lock_path.exists()
    assert not staging_root.exists()


def test_load_data_catalog_indexes_entries_and_reports_status_transitions(tmp_path):
    create_entrypoints(tmp_path)
    path = write_catalog(tmp_path)

    catalog = load_data_catalog(path)

    assert catalog.path == path.resolve()
    assert catalog.root == tmp_path.resolve()
    assert catalog.dataset("boundary")["role"] == "boundary"
    assert catalog.scenario("test-city")["display_name"] == "Test City"
    assert catalog.resolve("inputs/dem/elevation.tif") == (
        tmp_path / "inputs" / "dem" / "elevation.tif"
    ).resolve()
    assert catalog.scenario_status("test-city") == "ready"

    catalog.resolve("inputs/dem/elevation.tif").unlink()
    assert catalog.scenario_status("test-city") == "dem-pending"

    catalog.resolve("inputs/boundary/boundary.geojson").unlink()
    assert catalog.scenario_status("test-city") == "invalid"


def test_scenario_without_dem_is_boundary_ready(tmp_path):
    create_entrypoints(tmp_path)
    document = deepcopy(catalog_document(dem_dataset_id=None))
    path = write_catalog(tmp_path, document)

    catalog = load_data_catalog(path)

    assert catalog.scenario_status("test-city") == "boundary-ready"


@pytest.mark.parametrize(
    ("collection", "id_key", "message"),
    [
        ("datasets", "dataset_id", "Duplicate dataset ID"),
        ("scenarios", "scenario_id", "Duplicate scenario ID"),
    ],
)
def test_catalog_rejects_duplicate_ids(tmp_path, collection, id_key, message):
    document = catalog_document()
    duplicate = deepcopy(document[collection][0])
    duplicate[id_key] = document[collection][0][id_key]
    document[collection].append(duplicate)
    path = write_catalog(tmp_path, document)

    with pytest.raises(CatalogError, match=message):
        load_data_catalog(path)


@pytest.mark.parametrize(
    ("reference", "value", "message"),
    [
        ("boundary_dataset_id", "missing", "unknown boundary dataset"),
        ("boundary_dataset_id", "dem", "must have role 'boundary'"),
        ("dem_dataset_id", "missing", "unknown DEM dataset"),
        ("dem_dataset_id", "boundary", "must have role 'dem'"),
    ],
)
def test_catalog_rejects_broken_or_wrong_role_references(
    tmp_path,
    reference,
    value,
    message,
):
    document = catalog_document()
    document["scenarios"][0][reference] = value
    path = write_catalog(tmp_path, document)

    with pytest.raises(CatalogError, match=message):
        load_data_catalog(path)


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("dataset", "path"),
        ("dataset", "entrypoint"),
        ("scenario", "config_path"),
    ],
)
def test_catalog_rejects_paths_outside_repository_root(tmp_path, location, field):
    document = catalog_document()
    entry = document["datasets"][0] if location == "dataset" else document["scenarios"][0]
    entry[field] = "../outside.dat"
    path = write_catalog(tmp_path, document)

    with pytest.raises(CatalogError, match="escapes repository root"):
        load_data_catalog(path)


def test_catalog_rejects_absolute_paths_even_when_they_are_inside_root(tmp_path):
    document = catalog_document()
    document["datasets"][0]["entrypoint"] = str(tmp_path / "inputs" / "boundary.geojson")
    path = write_catalog(tmp_path, document)

    with pytest.raises(CatalogError, match="repository-relative"):
        load_data_catalog(path)


def test_catalog_rejects_entrypoint_outside_declared_dataset_path(tmp_path):
    document = catalog_document()
    document["datasets"][0]["entrypoint"] = "configs/test-city.yaml"
    path = write_catalog(tmp_path, document)

    with pytest.raises(CatalogError, match="entrypoint must be within dataset path"):
        load_data_catalog(path)


def test_load_data_catalog_accepts_explicit_root_for_arbitrary_catalog_location(tmp_path):
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    path = catalog_dir / "datasets.yaml"
    path.write_text(
        yaml.safe_dump(catalog_document(), sort_keys=False),
        encoding="utf-8",
    )

    catalog = load_data_catalog(path, repo_root=tmp_path)

    assert catalog.root == tmp_path.resolve()
    assert catalog.resolve("inputs/boundary") == (tmp_path / "inputs" / "boundary").resolve()


def test_load_data_catalog_normalizes_malformed_yaml_to_catalog_error(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = data_dir / "datasets.yaml"
    path.write_text("datasets: [", encoding="utf-8")

    with pytest.raises(CatalogError, match=r"Cannot parse data catalog .*datasets\.yaml"):
        load_data_catalog(path)


@pytest.mark.parametrize(
    ("entry", "field", "message"),
    [
        ("dataset", "provider", "missing required fields"),
        ("boundary", "feature_count", "missing boundary fields"),
        ("dem", "native_scale_m", "missing DEM fields"),
        ("scenario", "display_name", "missing required fields"),
    ],
)
def test_catalog_rejects_missing_required_fields(tmp_path, entry, field, message):
    document = catalog_document()
    if entry in {"dataset", "boundary"}:
        target = document["datasets"][0]
    elif entry == "dem":
        target = document["datasets"][1]
    else:
        target = document["scenarios"][0]
    del target[field]
    path = write_catalog(tmp_path, document)

    with pytest.raises(CatalogError, match=message):
        load_data_catalog(path)


@pytest.mark.parametrize(
    ("dataset_index", "field", "value", "message"),
    [
        (0, "role", [], "role must be one of"),
        (0, "geometry_type", None, "geometry_type"),
        (0, "feature_count", -1, "feature_count"),
        (0, "redistribution_confirmed", "yes", "redistribution_confirmed"),
        (1, "external", "yes", "external"),
        (1, "native_scale_m", 0, "native_scale_m"),
    ],
)
def test_catalog_rejects_invalid_role_specific_metadata(
    tmp_path,
    dataset_index,
    field,
    value,
    message,
):
    document = catalog_document()
    document["datasets"][dataset_index][field] = value
    path = write_catalog(tmp_path, document)

    with pytest.raises(CatalogError, match=message):
        load_data_catalog(path)


@pytest.mark.parametrize("scenario_id", ["Test-City", "1-test", "test_city", ""])
def test_catalog_rejects_invalid_scenario_ids(tmp_path, scenario_id):
    document = catalog_document()
    document["scenarios"][0]["scenario_id"] = scenario_id
    path = write_catalog(tmp_path, document)

    with pytest.raises(CatalogError, match="scenario_id"):
        load_data_catalog(path)


def test_catalog_lookup_and_resolve_report_unknown_or_unsafe_values(tmp_path):
    path = write_catalog(tmp_path)
    catalog = load_data_catalog(path)

    with pytest.raises(CatalogError, match="Unknown dataset ID"):
        catalog.dataset("missing")
    with pytest.raises(CatalogError, match="Unknown scenario ID"):
        catalog.scenario("missing")
    with pytest.raises(CatalogError, match="escapes repository root"):
        catalog.resolve("../outside")


def test_save_data_catalog_validates_and_atomically_replaces_yaml(tmp_path, monkeypatch):
    path = write_catalog(tmp_path)
    catalog = load_data_catalog(path)
    updated = deepcopy(catalog.document)
    updated["scenarios"][0]["display_name"] = "Updated City"
    replace_calls = []
    real_replace = os.replace

    def record_replace(source, destination):
        replace_calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr("lte_scenario_toolkit.data_catalog.os.replace", record_replace)

    saved = save_data_catalog(catalog, updated)

    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert written["scenarios"][0]["display_name"] == "Updated City"
    assert list(written) == ["datasets", "scenarios"]
    assert replace_calls and replace_calls[0][1] == path
    assert replace_calls[0][0].parent == path.parent
    assert not replace_calls[0][0].exists()
    assert saved.document == written
    assert saved.loaded_mtime_ns == path.stat().st_mtime_ns


def test_save_data_catalog_rejects_concurrent_modification(tmp_path):
    path = write_catalog(tmp_path)
    catalog = load_data_catalog(path)
    external_text = path.read_text(encoding="utf-8") + "\n# external edit\n"
    path.write_text(external_text, encoding="utf-8")
    changed_time = catalog.loaded_mtime_ns + 10_000_000_000
    os.utime(path, ns=(changed_time, changed_time))

    with pytest.raises(ConcurrentCatalogUpdateError, match="changed since it was loaded"):
        save_data_catalog(catalog, catalog.document)

    assert path.read_text(encoding="utf-8") == external_text


def test_save_data_catalog_does_not_replace_file_when_document_is_invalid(tmp_path):
    path = write_catalog(tmp_path)
    catalog = load_data_catalog(path)
    invalid = deepcopy(catalog.document)
    invalid["schema_version"] = 2
    original = path.read_text(encoding="utf-8")

    with pytest.raises(CatalogError, match="unexpected.*schema_version"):
        save_data_catalog(catalog, invalid)

    assert path.read_text(encoding="utf-8") == original


def test_update_data_manifest_writes_scenarios_metadata_and_checksums(tmp_path):
    create_entrypoints(tmp_path)
    catalog = load_data_catalog(write_catalog(tmp_path))

    output = update_data_manifest(catalog, "data/manifest.json")

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert output == (tmp_path / "data" / "manifest.json").resolve()
    assert "schema_version" not in payload
    assert payload["generated_at"].endswith("Z")
    assert payload["scenarios"] == catalog.document["scenarios"]
    datasets = {item["dataset_id"]: item for item in payload["datasets"]}
    assert datasets["boundary"]["geometry_type"] == "Polygon"
    assert datasets["boundary"]["files"] == [
        {
            "path": "inputs/boundary/boundary.geojson",
            "size_bytes": 8,
            "sha256": hashlib.sha256(b"boundary").hexdigest(),
        }
    ]


def test_incremental_manifest_reuses_untargeted_file_hashes(tmp_path, monkeypatch):
    create_entrypoints(tmp_path)
    catalog = load_data_catalog(write_catalog(tmp_path))
    output = update_data_manifest(catalog, "data/manifest.json")
    original = json.loads(output.read_text(encoding="utf-8"))
    original_boundary_files = original["datasets"][0]["files"]
    catalog.resolve("inputs/boundary/boundary.geojson").write_text(
        "changed boundary",
        encoding="utf-8",
    )
    hashed_paths = []
    real_sha256_file = data_catalog_module.sha256_file

    def record_hash(path, **kwargs):
        hashed_paths.append(Path(path))
        return real_sha256_file(path, **kwargs)

    monkeypatch.setattr(data_catalog_module, "sha256_file", record_hash)

    update_data_manifest(catalog, output, dataset_ids={"dem"})

    updated = json.loads(output.read_text(encoding="utf-8"))
    updated_datasets = {item["dataset_id"]: item for item in updated["datasets"]}
    assert updated_datasets["boundary"]["files"] == original_boundary_files
    assert hashed_paths == [catalog.resolve("inputs/dem/elevation.tif")]


def test_incremental_manifest_hashes_new_datasets_even_when_not_targeted(tmp_path, monkeypatch):
    create_entrypoints(tmp_path)
    catalog = load_data_catalog(write_catalog(tmp_path))
    output = update_data_manifest(catalog, "data/manifest.json")
    existing = json.loads(output.read_text(encoding="utf-8"))
    existing["datasets"] = [
        item for item in existing["datasets"] if item["dataset_id"] == "boundary"
    ]
    output.write_text(json.dumps(existing), encoding="utf-8")
    hashed_paths = []
    real_sha256_file = data_catalog_module.sha256_file

    def record_hash(path, **kwargs):
        hashed_paths.append(Path(path))
        return real_sha256_file(path, **kwargs)

    monkeypatch.setattr(data_catalog_module, "sha256_file", record_hash)

    update_data_manifest(catalog, output, dataset_ids={"boundary"})

    assert catalog.resolve("inputs/dem/elevation.tif") in hashed_paths


@pytest.mark.parametrize(
    "stale_case",
    ["path", "entrypoint", "file-path", "size", "sha256"],
)
def test_incremental_manifest_recomputes_malformed_or_stale_untargeted_records(
    tmp_path,
    stale_case,
):
    create_entrypoints(tmp_path)
    catalog = load_data_catalog(write_catalog(tmp_path))
    output = update_data_manifest(catalog, "data/manifest.json")
    existing = json.loads(output.read_text(encoding="utf-8"))
    boundary = next(item for item in existing["datasets"] if item["dataset_id"] == "boundary")
    if stale_case == "path":
        boundary["path"] = "inputs/other-boundary"
    elif stale_case == "entrypoint":
        boundary["entrypoint"] = "inputs/boundary/other.geojson"
    elif stale_case == "file-path":
        boundary["files"][0]["path"] = "inputs/dem/elevation.tif"
    elif stale_case == "size":
        boundary["files"][0]["size_bytes"] = -1
    else:
        boundary["files"][0]["sha256"] = "not-a-sha256"
    output.write_text(json.dumps(existing), encoding="utf-8")
    boundary_entrypoint = catalog.resolve("inputs/boundary/boundary.geojson")
    boundary_entrypoint.write_text("changed boundary", encoding="utf-8")

    update_data_manifest(catalog, output, dataset_ids={"dem"})

    updated = json.loads(output.read_text(encoding="utf-8"))
    updated_boundary = next(
        item for item in updated["datasets"] if item["dataset_id"] == "boundary"
    )
    assert updated_boundary["path"] == "inputs/boundary"
    assert updated_boundary["entrypoint"] == "inputs/boundary/boundary.geojson"
    assert updated_boundary["files"] == [
        {
            "path": "inputs/boundary/boundary.geojson",
            "size_bytes": len(b"changed boundary"),
            "sha256": hashlib.sha256(b"changed boundary").hexdigest(),
        }
    ]


def test_manifest_allows_missing_external_data_but_rejects_missing_local_data(tmp_path):
    create_entrypoints(tmp_path)
    catalog = load_data_catalog(write_catalog(tmp_path))
    shutil.rmtree(catalog.resolve("inputs/dem"))

    output = update_data_manifest(catalog, "data/manifest.json", dataset_ids={"dem"})

    payload = json.loads(output.read_text(encoding="utf-8"))
    datasets = {item["dataset_id"]: item for item in payload["datasets"]}
    assert datasets["dem"]["files"] == []

    shutil.rmtree(catalog.resolve("inputs/boundary"))
    with pytest.raises(FileNotFoundError, match="inputs.*boundary"):
        update_data_manifest(catalog, output, dataset_ids={"boundary"})


def test_incremental_manifest_recomputes_when_external_file_semantics_change(tmp_path):
    create_entrypoints(tmp_path)
    shutil.rmtree(tmp_path / "inputs" / "dem")
    catalog = load_data_catalog(write_catalog(tmp_path))
    output = update_data_manifest(catalog, "data/manifest.json")
    updated_document = deepcopy(catalog.document)
    updated_document["datasets"][1]["external"] = False
    catalog = save_data_catalog(catalog, updated_document)

    with pytest.raises(FileNotFoundError, match="inputs.*dem"):
        update_data_manifest(catalog, output, dataset_ids={"boundary"})


def test_manifest_rejects_unknown_target_dataset_ids(tmp_path):
    create_entrypoints(tmp_path)
    catalog = load_data_catalog(write_catalog(tmp_path))

    with pytest.raises(CatalogError, match="Unknown dataset IDs: missing"):
        update_data_manifest(catalog, "data/manifest.json", dataset_ids={"missing"})
