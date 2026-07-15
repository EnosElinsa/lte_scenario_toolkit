import hashlib
import importlib
import io
import stat
import zipfile
from pathlib import Path

import geopandas as gpd
import pytest
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
