from __future__ import annotations

import os
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from shapely.geometry import box

import lte_scenario_toolkit.gui.scenario_previews as preview_module
from lte_scenario_toolkit.gui.scenario_previews import (
    PREVIEW_SIZE,
    ScenarioPreviewRequest,
    ScenarioPreviewService,
    build_scenario_previews,
)


def _write_boundary(path: Path) -> Path:
    gpd.GeoDataFrame(
        {"name": ["test"]},
        geometry=[box(0, 0, 100, 50)],
        crs="EPSG:3857",
    ).to_file(path, driver="GeoJSON")
    return path


def _request(root: Path, *, boundary_path: Path, dem_path: Path | None = None):
    return ScenarioPreviewRequest(
        scenario_id="test/scenario",
        scenario_name="Test Scenario",
        boundary_path=boundary_path,
        dem_path=dem_path,
        allowed_root=root,
    )


def _write_dem(path: Path) -> Path:
    values = np.linspace(0, 100, 100 * 50, dtype="float32").reshape(50, 100)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=100,
        height=50,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 50, 1, 1),
    ) as dataset:
        dataset.write(values, 1)
    return path


def test_boundary_only_preview_is_a_fixed_valid_png(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"

    result = ScenarioPreviewService(cache_root).build(_request(tmp_path, boundary_path=boundary))

    assert result.kind == "boundary"
    assert result.cache_hit is False
    assert "DEM" in result.diagnostic
    assert result.path.parent == cache_root
    with Image.open(result.path) as image:
        image.verify()
    with Image.open(result.path) as image:
        assert image.format == "PNG"
        assert image.size == PREVIEW_SIZE == (760, 360)


def test_boundary_only_request_can_omit_dem_positionally(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    request = ScenarioPreviewRequest("scenario", "Scenario", boundary, tmp_path)

    assert request.dem_path is None
    result = ScenarioPreviewService(tmp_path / ".lte-data" / "cache" / "scenario-previews").build(
        request
    )
    assert result.kind == "boundary"


def test_request_preserves_legacy_five_positional_order(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    dem = _write_dem(tmp_path / "terrain.tif")

    request = ScenarioPreviewRequest("scenario", "Scenario", boundary, dem, tmp_path)

    assert request.dem_path == dem
    assert request.allowed_root == tmp_path


def test_request_accepts_dem_positional_with_keyword_root(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    dem = _write_dem(tmp_path / "terrain.tif")

    request = ScenarioPreviewRequest("scenario", "Scenario", boundary, dem, allowed_root=tmp_path)

    assert request.dem_path == dem
    assert request.allowed_root == tmp_path


def test_dem_and_boundary_preview_renders_muted_terrain(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    dem = _write_dem(tmp_path / "terrain.tif")
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"

    result = ScenarioPreviewService(cache_root).build(
        _request(tmp_path, boundary_path=boundary, dem_path=dem)
    )

    assert result.kind == "terrain"
    assert result.cache_hit is False
    assert result.diagnostic is None
    with Image.open(result.path) as image:
        assert image.size == PREVIEW_SIZE
        assert len(image.convert("RGB").getcolors(maxcolors=PREVIEW_SIZE[0] * PREVIEW_SIZE[1])) > 20


def test_terrain_cache_hit_preserves_terrain_kind(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    dem = _write_dem(tmp_path / "terrain.tif")
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"
    service = ScenarioPreviewService(cache_root)

    first = service.build(_request(tmp_path, boundary_path=boundary, dem_path=dem))
    second = service.build(_request(tmp_path, boundary_path=boundary, dem_path=dem))

    assert first.kind == second.kind == "terrain"
    assert second.cache_hit is True


def test_invalid_dem_degrades_to_boundary_only(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    dem = tmp_path / "invalid.tif"
    dem.write_bytes(b"not a raster")
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"

    result = ScenarioPreviewService(cache_root).build(
        _request(tmp_path, boundary_path=boundary, dem_path=dem)
    )

    assert result.kind == "boundary"
    assert result.diagnostic and "DEM preview unavailable" in result.diagnostic


def test_preview_cache_hit_reuses_valid_png(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"
    service = ScenarioPreviewService(cache_root)

    first = service.build(_request(tmp_path, boundary_path=boundary))
    second = service.build(_request(tmp_path, boundary_path=boundary))

    assert first.path == second.path
    assert second.cache_hit is True
    assert second.kind == "boundary"


def test_malformed_cached_png_is_regenerated(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"
    service = ScenarioPreviewService(cache_root)
    first = service.build(_request(tmp_path, boundary_path=boundary))
    first.path.write_bytes(b"not a png")

    second = service.build(_request(tmp_path, boundary_path=boundary))

    assert second.cache_hit is False
    with Image.open(second.path) as image:
        image.verify()


def test_missing_boundary_degrades_to_stable_fallback(tmp_path):
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"
    request = _request(tmp_path, boundary_path=tmp_path / "missing.geojson")

    first = ScenarioPreviewService(cache_root).build(request)
    second = ScenarioPreviewService(cache_root).build(request)

    assert first.kind == second.kind == "fallback"
    assert first.path == second.path
    assert first.diagnostic and "does not exist" in first.diagnostic
    assert second.cache_hit is True
    assert second.diagnostic == first.diagnostic
    with Image.open(first.path) as image:
        assert image.size == PREVIEW_SIZE


def test_batch_isolates_one_bad_request_from_good_requests(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"
    requests = [
        _request(tmp_path, boundary_path=boundary),
        _request(tmp_path, boundary_path=tmp_path / "missing.geojson"),
    ]

    results = build_scenario_previews(requests, cache_root)

    assert [result.kind for result in results] == ["boundary", "fallback"]
    assert all(result.path.is_file() for result in results)


def test_outside_and_traversal_inputs_degrade_without_exposing_source(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    outside = _write_boundary(tmp_path.parent / "outside.geojson")
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"
    service = ScenarioPreviewService(cache_root)

    outside_result = service.build(_request(tmp_path, boundary_path=outside))
    traversal_result = service.build(
        _request(tmp_path, boundary_path=Path("nested") / ".." / boundary.name)
    )

    assert outside_result.kind == traversal_result.kind == "fallback"
    assert "escapes" in outside_result.diagnostic
    assert "traversal" in traversal_result.diagnostic
    assert outside_result.path.suffix == ".png"


def test_cache_root_outside_allowed_root_writes_nowhere_outside(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    outside_cache = tmp_path.parent / "secret-preview-cache"
    outside_cache.mkdir()
    marker = outside_cache / "marker.txt"
    marker.write_text("keep", encoding="utf-8")

    result = ScenarioPreviewService(outside_cache).build(_request(tmp_path, boundary_path=boundary))

    assert result.kind == "fallback"
    assert result.path.is_relative_to(tmp_path / ".lte-data" / "cache" / "scenario-previews")
    assert marker.read_text(encoding="utf-8") == "keep"
    assert list(outside_cache.glob("*.png")) == []


def test_outside_cache_fallback_is_not_reused_by_internal_cache(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    outside_cache = tmp_path.parent / "outside-cache-provenance"
    outside_cache.mkdir()
    outside_result = ScenarioPreviewService(outside_cache).build(
        _request(tmp_path, boundary_path=boundary)
    )
    assert outside_result.kind == "fallback"

    internal = ScenarioPreviewService(tmp_path / ".lte-data" / "cache" / "scenario-previews").build(
        _request(tmp_path, boundary_path=boundary)
    )

    assert internal.kind == "boundary"
    assert internal.cache_hit is False


def test_symlink_boundary_degrades_when_platform_allows_links(tmp_path):
    target = _write_boundary(tmp_path / "real.geojson")
    link = tmp_path / "linked.geojson"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        import pytest

        pytest.skip(f"symlink creation unavailable: {exc}")
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"

    result = ScenarioPreviewService(cache_root).build(_request(tmp_path, boundary_path=link))

    assert result.kind == "fallback"
    assert "redirected" in result.diagnostic or "symlink" in result.diagnostic


def test_diagnostics_and_cache_filename_redact_sensitive_paths(tmp_path):
    secret = tmp_path / "SECRET_DO_NOT_LEAK"
    boundary = secret / "boundary.geojson"
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"
    result = ScenarioPreviewService(cache_root).build(_request(tmp_path, boundary_path=boundary))

    assert "SECRET_DO_NOT_LEAK" not in (result.diagnostic or "")
    assert "SECRET_DO_NOT_LEAK" not in result.path.name


def test_content_change_invalidates_key_even_when_metadata_is_restored(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"
    service = ScenarioPreviewService(cache_root)
    first = service.build(_request(tmp_path, boundary_path=boundary))
    original_stat = boundary.stat()
    raw = boundary.read_bytes()
    replacement = raw.replace(b'"test"', b'"demo"')
    assert len(replacement) == len(raw)
    boundary.write_bytes(replacement)
    os.utime(boundary, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    second = service.build(_request(tmp_path, boundary_path=boundary))

    assert second.cache_hit is False
    assert second.path != first.path


def test_shapefile_sidecar_change_invalidates_preview_key(tmp_path):
    boundary = tmp_path / "boundary.shp"
    gpd.GeoDataFrame(
        {"name": ["test"]},
        geometry=[box(0, 0, 100, 50)],
        crs="EPSG:3857",
    ).to_file(boundary)
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"
    service = ScenarioPreviewService(cache_root)
    first = service.build(_request(tmp_path, boundary_path=boundary))
    sidecar = boundary.with_suffix(".dbf")
    original_stat = sidecar.stat()
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    os.utime(sidecar, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    second = service.build(_request(tmp_path, boundary_path=boundary))

    assert second.cache_hit is False
    assert second.path != first.path


def test_read_time_revalidation_failure_degrades_without_opening_source(tmp_path, monkeypatch):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"

    def reject_recheck(path, root, label):
        raise ValueError("simulated source swap")

    monkeypatch.setattr(preview_module, "_revalidate_input", reject_recheck)
    result = ScenarioPreviewService(cache_root).build(_request(tmp_path, boundary_path=boundary))

    assert result.kind == "fallback"
    assert result.path.is_file()
    assert "simulated" not in (result.diagnostic or "")


def test_temporary_fallback_does_not_follow_existing_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(preview_module.tempfile, "gettempdir", lambda: str(tmp_path))
    fallback_root = tmp_path / "lte-scenario-toolkit-previews"
    fallback_root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"keep")
    symlink = fallback_root / "secret-fallback.png"
    try:
        symlink.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        import pytest

        pytest.skip(f"symlink creation unavailable: {exc}")

    result = preview_module._temporary_fallback("secret", preview_module._fallback_image())

    assert result.is_file()
    assert not result.is_symlink()
    assert outside.read_bytes() == b"keep"
    with Image.open(result) as image:
        assert image.size == PREVIEW_SIZE


def test_boundary_change_invalidates_preview_key(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"
    service = ScenarioPreviewService(cache_root)
    first = service.build(_request(tmp_path, boundary_path=boundary))
    boundary.write_text(boundary.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    second = service.build(_request(tmp_path, boundary_path=boundary))

    assert second.cache_hit is False
    assert second.path != first.path


def test_cache_root_rejects_traversal_and_redirected_component(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="traversal"):
        ScenarioPreviewService(tmp_path / "cache" / ".." / "escape")
    redirected = tmp_path / "redirected"
    try:
        redirected.symlink_to(tmp_path, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="redirected|symlink"):
        ScenarioPreviewService(redirected / "cache")


def test_atomic_replace_failure_cleans_temporary_file(tmp_path, monkeypatch):
    import pytest

    boundary = _write_boundary(tmp_path / "boundary.geojson")
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"
    service = ScenarioPreviewService(cache_root)

    def fail_replace(*args, **kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("lte_scenario_toolkit.gui.scenario_previews.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        service.build(_request(tmp_path, boundary_path=boundary))
    assert not list(cache_root.glob("*.tmp"))


def test_requests_and_batch_worker_are_pickleable(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    request = _request(tmp_path, boundary_path=boundary)

    pickle.loads(pickle.dumps(request))
    pickle.loads(pickle.dumps(build_scenario_previews))


def test_invalid_request_fields_degrade_to_fallback_result(tmp_path):
    request = ScenarioPreviewRequest(
        scenario_id=123,  # type: ignore[arg-type]
        scenario_name="Scenario",
        boundary_path=tmp_path / "missing.geojson",
        dem_path=None,
        allowed_root=tmp_path,
    )

    result = ScenarioPreviewService(tmp_path / "cache").build(request)

    assert result.kind == "fallback"
    assert result.path.is_file()
    with Image.open(result.path) as image:
        assert image.size == PREVIEW_SIZE


def test_invalid_allowed_root_degrades_without_raising(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    cache_root = tmp_path / "cache"
    request = ScenarioPreviewRequest(
        "scenario",
        "Scenario",
        boundary,
        allowed_root=tmp_path / "missing-root",
    )

    result = ScenarioPreviewService(cache_root).build(request)

    assert result.kind == "fallback"
    assert result.path.is_file()
    with Image.open(result.path) as image:
        assert image.size == PREVIEW_SIZE


def test_batch_error_results_always_have_valid_png_path(tmp_path):
    cache_root = tmp_path / "cache"
    request = ScenarioPreviewRequest(
        "scenario",
        "Scenario",
        tmp_path / "missing.geojson",
        allowed_root=tmp_path / "missing-root",
    )

    results = build_scenario_previews([request], cache_root)

    assert len(results) == 1
    assert results[0].path.is_file()
    with Image.open(results[0].path) as image:
        assert image.format == "PNG"
        assert image.size == PREVIEW_SIZE


def test_redirected_metadata_is_regenerated_without_reading_target(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"
    service = ScenarioPreviewService(cache_root)
    first = service.build(_request(tmp_path, boundary_path=boundary))
    metadata = first.path.with_suffix(".json")
    outside = tmp_path.parent / "metadata-secret.json"
    outside.write_text('{"kind":"fallback","diagnostic":"SECRET"}', encoding="utf-8")
    try:
        metadata.unlink()
        metadata.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        import pytest

        pytest.skip(f"symlink creation unavailable: {exc}")

    second = service.build(_request(tmp_path, boundary_path=boundary))

    assert second.cache_hit is False
    assert outside.read_text(encoding="utf-8") == '{"kind":"fallback","diagnostic":"SECRET"}'


def test_untrusted_metadata_diagnostic_is_redacted_on_cache_hit(tmp_path):
    boundary = _write_boundary(tmp_path / "boundary.geojson")
    cache_root = tmp_path / ".lte-data" / "cache" / "scenario-previews"
    service = ScenarioPreviewService(cache_root)
    first = service.build(_request(tmp_path, boundary_path=boundary))
    first.path.with_suffix(".json").write_text(
        '{"kind":"boundary","diagnostic":"SECRET_DIAGNOSTIC"}',
        encoding="utf-8",
    )

    second = service.build(_request(tmp_path, boundary_path=boundary))

    assert second.cache_hit is True
    assert second.diagnostic == "Cached preview diagnostic unavailable."
