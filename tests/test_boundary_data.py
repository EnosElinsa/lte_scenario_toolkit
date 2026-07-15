import hashlib
import importlib
import io
import json
import stat
import zipfile
from pathlib import Path

import geopandas as gpd
import pytest
import yaml
from shapely.geometry import Polygon, box


def test_boundary_data_module_is_importable():
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")

    assert module is not None


def test_safe_extract_zip_rejects_parent_traversal(tmp_path):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.txt", "nope")

    with pytest.raises(ValueError, match="unsafe|traversal"):
        module.safe_extract_zip(archive, tmp_path / "extract")


def test_safe_extract_zip_rejects_symlink_entries(tmp_path):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    archive = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link.shp")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(info, "target.shp")

    with pytest.raises(ValueError, match="symlink"):
        module.safe_extract_zip(archive, tmp_path / "extract")


def _write_geojson(path: Path, geometries, *, crs="EPSG:4326") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = gpd.GeoDataFrame(
        {"source_id": list(range(1, len(geometries) + 1))},
        geometry=geometries,
        crs=crs,
    )
    frame.to_file(path, driver="GeoJSON")
    return path


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for name, content in files.items():
            handle.writestr(name, content)
    return stream.getvalue()


def _write_empty_catalog(root: Path) -> Path:
    catalog_path = root / "data" / "datasets.yaml"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(
        yaml.safe_dump(
            {"schema_version": 2, "datasets": [], "scenarios": []},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return catalog_path


def test_register_scenario_installs_boundary_and_updates_catalog_and_manifest(
    tmp_path, monkeypatch
):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(
        tmp_path / "source" / "boundary.geojson",
        [box(0, 0, 1, 1), box(1, 0, 2, 1)],
    )
    catalog_path = _write_empty_catalog(tmp_path)
    manifest_calls = []
    real_update_manifest = module.update_data_manifest

    def record_manifest_update(catalog, output_path, dataset_ids=None):
        manifest_calls.append((output_path, dataset_ids))
        return real_update_manifest(catalog, output_path, dataset_ids=dataset_ids)

    monkeypatch.setattr(module, "update_data_manifest", record_manifest_update)

    catalog = module.register_scenario(
        catalog_path,
        scenario_id="sample-city",
        display_name="Sample City",
        boundary_source=source,
        provider="Example GIS Office",
        license_name="CC0-1.0",
        redistribution_confirmed=True,
        download_date="2026-07-16",
        config_path="configs/sample-city.yaml",
    )

    boundary_id = "boundary_sample_city"
    dem_id = "usgs_3dep_1m_dem_sample_city"
    destination = tmp_path / "boundary_shp" / "sample-city"
    assert catalog.scenario("sample-city") == {
        "scenario_id": "sample-city",
        "display_name": "Sample City",
        "boundary_dataset_id": boundary_id,
        "dem_dataset_id": dem_id,
        "config_path": "configs/sample-city.yaml",
    }
    assert set(path.suffix for path in destination.glob("sample-city.*")) >= {
        ".shp",
        ".shx",
        ".dbf",
        ".prj",
        ".cpg",
    }

    boundary = catalog.dataset(boundary_id)
    assert boundary == {
        "dataset_id": boundary_id,
        "role": "boundary",
        "path": "boundary_shp/sample-city",
        "entrypoint": "boundary_shp/sample-city/sample-city.shp",
        "source_url": None,
        "source_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "provider": "Example GIS Office",
        "license": "CC0-1.0",
        "download_date": "2026-07-16",
        "crs": "EPSG:3857",
        "spatial_resolution": "polygon vector",
        "geometry_type": "Polygon",
        "feature_count": 1,
        "redistribution_confirmed": True,
        "notes": "Normalized, dissolved scenario boundary in EPSG:3857.",
    }
    dem = catalog.dataset(dem_id)
    assert dem == {
        "dataset_id": dem_id,
        "role": "dem",
        "path": "dem/sample-city",
        "entrypoint": "dem/sample-city/usgs_3dep_1m_sample-city.tif",
        "source_url": "https://developers.google.com/earth-engine/datasets/catalog/USGS_3DEP_1m",
        "provider": "United States Geological Survey",
        "license": "USGS public-domain data; retain source attribution",
        "download_date": None,
        "crs": "EPSG:3857",
        "spatial_resolution": "1 m",
        "notes": "Pending external Earth Engine export for this scenario.",
        "external": True,
        "earth_engine_collection": "USGS/3DEP/1m",
        "band": "elevation",
        "units": "metres",
        "vertical_datum": "NAVD88",
        "native_scale_m": 1,
        "export_crs": "EPSG:3857",
        "export_prefix": "usgs_3dep_1m_sample-city",
        "drive_folder": "lte-scenario-toolkit-dem",
    }
    assert manifest_calls == [
        ("data/manifest.json", {boundary_id, dem_id}),
    ]

    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text(encoding="utf-8"))
    records = {item["dataset_id"]: item for item in manifest["datasets"]}
    assert records[boundary_id]["files"]
    assert records[dem_id]["files"] == []
    assert manifest["scenarios"] == [catalog.scenario("sample-city")]


def test_register_scenario_requires_redistribution_confirmation_without_side_effects(tmp_path):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(tmp_path / "boundary.geojson", [box(0, 0, 1, 1)])
    catalog_path = _write_empty_catalog(tmp_path)
    original_catalog = catalog_path.read_bytes()

    with pytest.raises(module.BoundaryImportError, match="redistribution"):
        module.register_scenario(
            catalog_path,
            scenario_id="sample-city",
            display_name="Sample City",
            boundary_source=source,
            provider="Example GIS Office",
            license_name="CC0-1.0",
            redistribution_confirmed=False,
        )

    assert catalog_path.read_bytes() == original_catalog
    assert not (tmp_path / "boundary_shp" / "sample-city").exists()
    assert not (tmp_path / "data" / "manifest.json").exists()
    assert not (tmp_path / ".lte-data").exists()


def test_register_scenario_rolls_back_boundary_catalog_and_manifest_when_manifest_fails(
    tmp_path, monkeypatch
):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(tmp_path / "boundary.geojson", [box(0, 0, 1, 1)])
    catalog_path = _write_empty_catalog(tmp_path)
    manifest_path = tmp_path / "data" / "manifest.json"
    manifest_path.write_bytes(b'{"original": true}\n')
    original_catalog = catalog_path.read_bytes()
    original_manifest = manifest_path.read_bytes()

    def fail_manifest(*args, **kwargs):
        manifest_path.write_bytes(b'{"partial": true}\n')
        raise RuntimeError("manifest failed")

    monkeypatch.setattr(module, "update_data_manifest", fail_manifest)

    with pytest.raises(RuntimeError, match="manifest failed"):
        module.register_scenario(
            catalog_path,
            scenario_id="sample-city",
            display_name="Sample City",
            boundary_source=source,
            provider="Example GIS Office",
            license_name="CC0-1.0",
            redistribution_confirmed=True,
        )

    assert catalog_path.read_bytes() == original_catalog
    assert manifest_path.read_bytes() == original_manifest
    assert not (tmp_path / "boundary_shp" / "sample-city").exists()
    assert not (tmp_path / "boundary_shp").exists()
    assert not (tmp_path / ".lte-data").exists()


def test_register_scenario_does_not_overwrite_a_concurrent_catalog_update(
    tmp_path, monkeypatch
):
    from lte_scenario_toolkit.data_catalog import ConcurrentCatalogUpdateError

    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(tmp_path / "boundary.geojson", [box(0, 0, 1, 1)])
    catalog_path = _write_empty_catalog(tmp_path)
    concurrent_bytes = yaml.safe_dump(
        {"schema_version": 2, "datasets": [], "scenarios": []},
        sort_keys=False,
    ).replace("scenarios: []", "scenarios: []\n# concurrent writer\n").encode()

    def concurrent_save(catalog, document):
        catalog.path.write_bytes(concurrent_bytes)
        raise ConcurrentCatalogUpdateError("catalog changed concurrently")

    monkeypatch.setattr(module, "save_data_catalog", concurrent_save)

    with pytest.raises(ConcurrentCatalogUpdateError, match="concurrently"):
        module.register_scenario(
            catalog_path,
            scenario_id="sample-city",
            display_name="Sample City",
            boundary_source=source,
            provider="Example GIS Office",
            license_name="CC0-1.0",
            redistribution_confirmed=True,
        )

    assert catalog_path.read_bytes() == concurrent_bytes
    assert not (tmp_path / "boundary_shp" / "sample-city").exists()
    assert not (tmp_path / "data" / "manifest.json").exists()


def test_register_scenario_restores_catalog_when_save_writes_then_raises(
    tmp_path, monkeypatch
):
    from lte_scenario_toolkit.data_catalog import CatalogError

    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(tmp_path / "boundary.geojson", [box(0, 0, 1, 1)])
    catalog_path = _write_empty_catalog(tmp_path)
    original_catalog = catalog_path.read_bytes()

    def write_then_fail(catalog, document):
        catalog.path.write_bytes(
            yaml.safe_dump(document, sort_keys=False, allow_unicode=True).encode("utf-8")
        )
        raise CatalogError("post-replace load failed")

    monkeypatch.setattr(module, "save_data_catalog", write_then_fail)

    with pytest.raises(CatalogError, match="post-replace"):
        module.register_scenario(
            catalog_path,
            scenario_id="sample-city",
            display_name="Sample City",
            boundary_source=source,
            provider="Example GIS Office",
            license_name="CC0-1.0",
            redistribution_confirmed=True,
        )

    assert catalog_path.read_bytes() == original_catalog
    assert not (tmp_path / "boundary_shp" / "sample-city").exists()
    assert not (tmp_path / "data" / "manifest.json").exists()


def test_register_scenario_surfaces_destination_cleanup_failure_and_restores_registry(
    tmp_path, monkeypatch
):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(tmp_path / "boundary.geojson", [box(0, 0, 1, 1)])
    catalog_path = _write_empty_catalog(tmp_path)
    original_catalog = catalog_path.read_bytes()
    destination = tmp_path / "boundary_shp" / "sample-city"
    real_rmtree = module.shutil.rmtree

    def fail_destination_cleanup(path, *args, **kwargs):
        if Path(path) == destination:
            raise PermissionError("destination is locked")
        return real_rmtree(path, *args, **kwargs)

    def fail_manifest(*args, **kwargs):
        raise RuntimeError("manifest failed")

    monkeypatch.setattr(module, "update_data_manifest", fail_manifest)
    monkeypatch.setattr(module.shutil, "rmtree", fail_destination_cleanup)

    with pytest.raises(module.BoundaryImportError, match="rollback.*destination") as caught:
        module.register_scenario(
            catalog_path,
            scenario_id="sample-city",
            display_name="Sample City",
            boundary_source=source,
            provider="Example GIS Office",
            license_name="CC0-1.0",
            redistribution_confirmed=True,
        )

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert catalog_path.read_bytes() == original_catalog
    assert not (tmp_path / "data" / "manifest.json").exists()


def test_register_scenario_surfaces_staging_cleanup_failure_after_success(
    tmp_path, monkeypatch
):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(tmp_path / "boundary.geojson", [box(0, 0, 1, 1)])
    catalog_path = _write_empty_catalog(tmp_path)
    staging_root = tmp_path / ".lte-data"
    real_rmtree = module.shutil.rmtree

    def fail_staging_cleanup(path, *args, **kwargs):
        if Path(path).parent == staging_root:
            raise PermissionError("staging directory is locked")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(module.shutil, "rmtree", fail_staging_cleanup)

    with pytest.raises(module.BoundaryImportError, match="staging cleanup") as caught:
        module.register_scenario(
            catalog_path,
            scenario_id="sample-city",
            display_name="Sample City",
            boundary_source=source,
            provider="Example GIS Office",
            license_name="CC0-1.0",
            redistribution_confirmed=True,
        )

    assert isinstance(caught.value.__cause__, PermissionError)
    assert (tmp_path / "boundary_shp" / "sample-city").is_dir()


def test_register_scenario_preserves_operation_error_when_staging_cleanup_also_fails(
    tmp_path, monkeypatch
):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(tmp_path / "boundary.geojson", [box(0, 0, 1, 1)])
    catalog_path = _write_empty_catalog(tmp_path)
    staging_root = tmp_path / ".lte-data"
    real_rmtree = module.shutil.rmtree

    def fail_staging_cleanup(path, *args, **kwargs):
        if Path(path).parent == staging_root:
            raise PermissionError("staging directory is locked")
        return real_rmtree(path, *args, **kwargs)

    def fail_manifest(*args, **kwargs):
        raise RuntimeError("manifest failed")

    monkeypatch.setattr(module.shutil, "rmtree", fail_staging_cleanup)
    monkeypatch.setattr(module, "update_data_manifest", fail_manifest)

    with pytest.raises(RuntimeError, match="manifest failed") as caught:
        module.register_scenario(
            catalog_path,
            scenario_id="sample-city",
            display_name="Sample City",
            boundary_source=source,
            provider="Example GIS Office",
            license_name="CC0-1.0",
            redistribution_confirmed=True,
        )

    assert isinstance(caught.value.__cause__, module.BoundaryImportError)
    assert "staging cleanup" in str(caught.value.__cause__)


def _directory_symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")


def test_register_scenario_rejects_boundary_parent_symlink_escape(tmp_path):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(tmp_path / "boundary.geojson", [box(0, 0, 1, 1)])
    catalog_path = _write_empty_catalog(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    _directory_symlink_or_skip(tmp_path / "boundary_shp", outside)

    with pytest.raises(module.BoundaryImportError, match="symlink|repository"):
        module.register_scenario(
            catalog_path,
            scenario_id="sample-city",
            display_name="Sample City",
            boundary_source=source,
            provider="Example GIS Office",
            license_name="CC0-1.0",
            redistribution_confirmed=True,
        )

    assert not (outside / "sample-city").exists()
    assert not (tmp_path / ".lte-data").exists()


def test_register_scenario_rejects_dangling_destination_symlink(tmp_path):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(tmp_path / "boundary.geojson", [box(0, 0, 1, 1)])
    catalog_path = _write_empty_catalog(tmp_path)
    destination = tmp_path / "boundary_shp" / "sample-city"
    destination.parent.mkdir()
    _directory_symlink_or_skip(destination, tmp_path / "missing-target")

    with pytest.raises(module.BoundaryImportError, match="symlink|exists"):
        module.register_scenario(
            catalog_path,
            scenario_id="sample-city",
            display_name="Sample City",
            boundary_source=source,
            provider="Example GIS Office",
            license_name="CC0-1.0",
            redistribution_confirmed=True,
        )

    assert destination.is_symlink()
    assert not (tmp_path / ".lte-data").exists()


def test_register_scenario_rejects_an_existing_symlink_parent_before_staging(
    tmp_path, monkeypatch
):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(tmp_path / "boundary.geojson", [box(0, 0, 1, 1)])
    catalog_path = _write_empty_catalog(tmp_path)
    boundary_parent = tmp_path / "boundary_shp"
    boundary_parent.mkdir()
    real_is_symlink = Path.is_symlink

    def report_boundary_parent_symlink(path):
        return Path(path) == boundary_parent or real_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", report_boundary_parent_symlink)

    with pytest.raises(module.BoundaryImportError, match="symlink|repository"):
        module.register_scenario(
            catalog_path,
            scenario_id="sample-city",
            display_name="Sample City",
            boundary_source=source,
            provider="Example GIS Office",
            license_name="CC0-1.0",
            redistribution_confirmed=True,
        )

    assert not (tmp_path / ".lte-data").exists()


def test_register_scenario_uses_lexical_destination_existence_check(tmp_path, monkeypatch):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(tmp_path / "boundary.geojson", [box(0, 0, 1, 1)])
    catalog_path = _write_empty_catalog(tmp_path)
    destination = tmp_path / "boundary_shp" / "sample-city"
    real_lexists = module.os.path.lexists

    def report_dangling_destination(path):
        return Path(path) == destination or real_lexists(path)

    monkeypatch.setattr(module.os.path, "lexists", report_dangling_destination)

    with pytest.raises(module.BoundaryImportError, match="exists"):
        module.register_scenario(
            catalog_path,
            scenario_id="sample-city",
            display_name="Sample City",
            boundary_source=source,
            provider="Example GIS Office",
            license_name="CC0-1.0",
            redistribution_confirmed=True,
        )

    assert not (tmp_path / ".lte-data").exists()


@pytest.mark.parametrize("malformed_manifest_location", ["data-file", "manifest-directory"])
def test_register_scenario_rejects_malformed_manifest_location_before_staging(
    tmp_path, malformed_manifest_location
):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(tmp_path / "boundary.geojson", [box(0, 0, 1, 1)])
    if malformed_manifest_location == "data-file":
        catalog_path = tmp_path / "datasets.yaml"
        catalog_path.write_text(
            yaml.safe_dump(
                {"schema_version": 2, "datasets": [], "scenarios": []},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (tmp_path / "data").write_text("not a directory", encoding="utf-8")
    else:
        catalog_path = _write_empty_catalog(tmp_path)
        (tmp_path / "data" / "manifest.json").mkdir()
    original_catalog = catalog_path.read_bytes()

    with pytest.raises(module.BoundaryImportError, match="manifest|data.*directory"):
        module.register_scenario(
            catalog_path,
            scenario_id="sample-city",
            display_name="Sample City",
            boundary_source=source,
            provider="Example GIS Office",
            license_name="CC0-1.0",
            redistribution_confirmed=True,
        )

    assert catalog_path.read_bytes() == original_catalog
    assert not (tmp_path / "boundary_shp").exists()
    assert not (tmp_path / ".lte-data").exists()


def test_import_boundary_source_dissolves_reprojects_and_writes_shapefile_sidecars(tmp_path):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(
        tmp_path / "boundary.geojson",
        [box(0, 0, 1, 1), box(1, 0, 2, 1)],
    )

    artifact = module.import_boundary_source(
        source,
        scenario_id="city-1",
        display_name="Test City",
        staging_dir=tmp_path / "stage",
    )

    assert artifact.directory == tmp_path / "stage" / "normalized"
    assert artifact.entrypoint == artifact.directory / "city-1.shp"
    assert artifact.source_url is None
    assert artifact.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert artifact.crs == "EPSG:3857"
    assert artifact.geometry_type == "Polygon"
    assert artifact.feature_count == 1
    assert set(path.suffix for path in artifact.directory.glob("city-1.*")) >= {
        ".shp",
        ".shx",
        ".dbf",
        ".prj",
        ".cpg",
    }

    normalized = gpd.read_file(artifact.entrypoint)
    assert len(normalized) == 1
    assert normalized.crs.to_epsg() == 3857
    assert normalized.geometry.iloc[0].geom_type == "Polygon"
    assert normalized["scenario"].tolist() == ["city-1"]
    assert normalized["name"].tolist() == ["Test City"]


def test_import_boundary_source_requires_layer_when_zip_has_multiple_vectors(tmp_path):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    first = _write_geojson(tmp_path / "first.geojson", [box(0, 0, 1, 1)])
    second = _write_geojson(tmp_path / "second.geojson", [box(2, 2, 3, 3)])
    archive = tmp_path / "boundaries.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(first, arcname="first.geojson")
        handle.write(second, arcname="second.geojson")

    with pytest.raises(ValueError, match="layer"):
        module.import_boundary_source(
            archive,
            scenario_id="multi",
            display_name="Multiple",
            staging_dir=tmp_path / "stage-no-layer",
        )

    artifact = module.import_boundary_source(
        archive,
        scenario_id="multi",
        display_name="Multiple",
        staging_dir=tmp_path / "stage-second",
        layer="second",
    )
    normalized = gpd.read_file(artifact.entrypoint)
    assert normalized.geometry.iloc[0].centroid.x > 200000


def test_import_boundary_source_rejects_invalid_geometry(tmp_path):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    invalid = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    source = _write_geojson(tmp_path / "invalid.geojson", [invalid])

    with pytest.raises(ValueError, match="valid"):
        module.import_boundary_source(
            source,
            scenario_id="invalid",
            display_name="Invalid",
            staging_dir=tmp_path / "stage",
        )


def test_import_boundary_source_fetches_mocked_http_zip_and_preserves_url(tmp_path, monkeypatch):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(tmp_path / "remote.geojson", [box(0, 0, 1, 1)])
    payload = _zip_bytes({"remote.geojson": source.read_bytes()})
    requested = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.close()

    def fake_urlopen(url, *, timeout):
        del timeout
        requested.append(str(url))
        return Response(payload)

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    url = "https://example.test/boundaries/latest.zip"
    artifact = module.import_boundary_source(
        url,
        scenario_id="remote",
        display_name="Remote",
        staging_dir=tmp_path / "stage",
    )

    assert requested == [url]
    assert artifact.source_url == url
    assert artifact.source_sha256 == hashlib.sha256(payload).hexdigest()
    assert artifact.entrypoint.is_file()


def test_import_boundary_source_rejects_shapefile_without_prj(tmp_path):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(tmp_path / "source.geojson", [box(0, 0, 1, 1)])
    shapefile = tmp_path / "source.shp"
    gpd.read_file(source).to_file(shapefile, driver="ESRI Shapefile")
    shapefile.with_suffix(".prj").unlink()

    with pytest.raises(ValueError, match="CRS|crs"):
        module.import_boundary_source(
            shapefile,
            scenario_id="no-crs",
            display_name="No CRS",
            staging_dir=tmp_path / "stage",
        )


def test_safe_extract_zip_rejects_member_count_limit(tmp_path, monkeypatch):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    archive = tmp_path / "many.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("one.txt", "1")
        handle.writestr("two.txt", "2")
    monkeypatch.setattr(module, "MAX_ZIP_MEMBERS", 1)

    with pytest.raises(ValueError, match="member count"):
        module.safe_extract_zip(archive, tmp_path / "extract")


def test_safe_extract_zip_rejects_member_and_aggregate_expanded_size_limits(
    tmp_path, monkeypatch
):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    archive = tmp_path / "large.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("one.txt", "1234")
        handle.writestr("two.txt", "5678")

    monkeypatch.setattr(module, "MAX_ZIP_MEMBER_BYTES", 3)
    with pytest.raises(ValueError, match="member.*expanded|expanded.*member"):
        module.safe_extract_zip(archive, tmp_path / "member-extract")

    monkeypatch.setattr(module, "MAX_ZIP_MEMBER_BYTES", 100)
    monkeypatch.setattr(module, "MAX_ZIP_TOTAL_BYTES", 7)
    with pytest.raises(ValueError, match="aggregate.*expanded|expanded.*aggregate"):
        module.safe_extract_zip(archive, tmp_path / "total-extract")


def test_remote_download_uses_timeout_and_rejects_oversized_response(tmp_path, monkeypatch):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    requested = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.close()

    def fake_urlopen(url, *, timeout):
        requested.append((str(url), timeout))
        return Response(b"12345")

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(module, "MAX_REMOTE_BYTES", 4)
    with pytest.raises(ValueError, match="maximum|limit|bytes"):
        module.import_boundary_source(
            "https://example.test/too-large.zip",
            scenario_id="remote-limit",
            display_name="Remote",
            staging_dir=tmp_path / "stage",
        )

    assert requested == [("https://example.test/too-large.zip", module.REMOTE_TIMEOUT_SECONDS)]


@pytest.mark.parametrize(
    "member_name",
    ["dir/file:stream", "CON.txt", "dir/trailing. ", "dir/trailing./file.txt"],
)
def test_safe_extract_zip_rejects_windows_unsafe_member_names(tmp_path, member_name):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    archive = tmp_path / "windows-unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member_name, "nope")

    with pytest.raises(ValueError, match="Windows|unsafe|reserved|trailing"):
        module.safe_extract_zip(archive, tmp_path / "extract")


def test_safe_extract_zip_rejects_case_insensitive_duplicate_members(tmp_path):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("Boundary.geojson", "one")
        handle.writestr("boundary.GEOJSON", "two")

    with pytest.raises(ValueError, match="duplicate"):
        module.safe_extract_zip(archive, tmp_path / "extract")


@pytest.mark.parametrize("scenario_id", ["Bad", "a_b", "a.b", "../bad", "con", ""])
def test_import_boundary_source_rejects_unsafe_scenario_ids(tmp_path, scenario_id):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")

    with pytest.raises(ValueError, match="scenario_id"):
        module.import_boundary_source(
            tmp_path / "missing.geojson",
            scenario_id=scenario_id,
            display_name="Invalid",
            staging_dir=tmp_path / f"stage-{scenario_id or 'empty'}",
        )


def test_remote_shapefile_sidecars_try_source_extension_case_then_fallback(tmp_path, monkeypatch):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(tmp_path / "source.geojson", [box(0, 0, 1, 1)])
    shapefile = tmp_path / "source.shp"
    gpd.read_file(source).to_file(shapefile, driver="ESRI Shapefile", encoding="UTF-8")
    payloads = {
        f"https://example.test/boundary{suffix.upper()}": shapefile.with_suffix(suffix).read_bytes()
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg")
    }
    requested = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.close()

    def fake_urlopen(url, *, timeout):
        del timeout
        requested.append(str(url))
        try:
            return Response(payloads[str(url)])
        except KeyError as exc:
            raise module.urllib.error.HTTPError(str(url), 404, "missing", None, None) from exc

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    artifact = module.import_boundary_source(
        "https://example.test/boundary.SHP",
        scenario_id="upper-sidecars",
        display_name="Upper",
        staging_dir=tmp_path / "stage",
    )

    assert artifact.entrypoint.is_file()
    assert requested[:4] == [
        "https://example.test/boundary.SHP",
        "https://example.test/boundary.SHX",
        "https://example.test/boundary.SHP".replace(".SHP", ".DBF"),
        "https://example.test/boundary.SHP".replace(".SHP", ".PRJ"),
    ]


def test_zip_metadata_json_is_not_treated_as_a_vector_layer(tmp_path):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    boundary = _write_geojson(tmp_path / "boundary.geojson", [box(0, 0, 1, 1)])
    archive = tmp_path / "with-metadata.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(boundary, arcname="boundary.geojson")
        handle.writestr("metadata.json", '{"title": "not a vector"}')

    artifact = module.import_boundary_source(
        archive,
        scenario_id="metadata-filter",
        display_name="Metadata",
        staging_dir=tmp_path / "stage",
    )

    assert artifact.entrypoint.is_file()


def test_remote_budget_overflow_is_not_retried_with_case_variant_sidecar(
    tmp_path, monkeypatch
):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    requested = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.close()

    def fake_urlopen(url, *, timeout):
        del timeout
        requested.append(str(url))
        if str(url).endswith(".SHP"):
            return Response(b"1")
        if str(url).endswith(".SHX"):
            return Response(b"234567")
        if str(url).endswith(".shx"):
            return Response(b"2")
        raise module.urllib.error.HTTPError(str(url), 404, "missing", None, None)

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(module, "MAX_REMOTE_BYTES", 5)
    with pytest.raises(ValueError, match="maximum|limit|bytes"):
        module.import_boundary_source(
            "https://example.test/boundary.SHP",
            scenario_id="budget-retry",
            display_name="Budget",
            staging_dir=tmp_path / "stage",
        )

    assert requested == [
        "https://example.test/boundary.SHP",
        "https://example.test/boundary.SHX",
    ]


def test_remote_optional_cpg_does_not_swallow_budget_overflow(tmp_path, monkeypatch):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    source = _write_geojson(tmp_path / "source.geojson", [box(0, 0, 1, 1)])
    shapefile = tmp_path / "source.shp"
    gpd.read_file(source).to_file(shapefile, driver="ESRI Shapefile", encoding="UTF-8")
    required = [shapefile.with_suffix(suffix).read_bytes() for suffix in (".shp", ".shx", ".dbf", ".prj")]
    payloads = {
        f"https://example.test/boundary{suffix}": shapefile.with_suffix(suffix).read_bytes()
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg")
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.close()

    def fake_urlopen(url, *, timeout):
        del timeout
        try:
            return Response(payloads[str(url)])
        except KeyError as exc:
            raise module.urllib.error.HTTPError(str(url), 404, "missing", None, None) from exc

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(module, "MAX_REMOTE_BYTES", sum(map(len, required)) + 1)
    with pytest.raises(ValueError, match="maximum|limit|bytes"):
        module.import_boundary_source(
            "https://example.test/boundary.shp",
            scenario_id="budget-cpg",
            display_name="Budget CPG",
            staging_dir=tmp_path / "stage",
        )


def test_empty_unknown_json_is_not_treated_as_a_vector_layer(tmp_path):
    module = importlib.import_module("lte_scenario_toolkit.boundary_data")
    boundary = _write_geojson(tmp_path / "boundary.geojson", [box(0, 0, 1, 1)])
    archive = tmp_path / "with-empty-metadata.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(boundary, arcname="boundary.geojson")
        handle.writestr("metadata.json", '{"type": "FeatureCollection", "features": []}')

    artifact = module.import_boundary_source(
        archive,
        scenario_id="empty-metadata",
        display_name="Empty Metadata",
        staging_dir=tmp_path / "stage",
    )

    assert artifact.entrypoint.is_file()
