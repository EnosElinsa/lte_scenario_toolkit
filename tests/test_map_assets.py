import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio import Affine
from rasterio.transform import from_origin
from shapely import from_wkt
from shapely.geometry import Point, box

import lte_scenario_toolkit.map_assets as map_assets
from lte_scenario_toolkit.map_assets import MapAssetService, MapStyle


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")


@pytest.fixture
def dem_fixture(tmp_path):
    path = tmp_path / "dem.tif"
    values = np.arange(256, dtype="float32").reshape(16, 16)
    values[:4, :4] = -9999
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        nodata=-9999,
        transform=from_origin(0, 16, 1, 1),
    ) as dataset:
        dataset.write(values, 1)
    return path


def test_dem_overview_is_bounded_and_nodata_is_transparent(dem_fixture, tmp_path):
    service = MapAssetService(tmp_path)

    asset = service.dem_overlay(
        dem_fixture,
        fingerprint="dem-a",
        bounds=(0, 0, 16, 16),
        bounds_crs="EPSG:3857",
        style=MapStyle.COMBINED,
        max_dimension=128,
    )

    image = Image.open(asset.path).convert("RGBA")
    assert max(image.size) <= 128
    assert image.getextrema()[3][0] == 0
    assert asset.bounds == (0, 0, 16, 16)


def test_dem_styles_have_distinct_cache_entries(dem_fixture, tmp_path):
    service = MapAssetService(tmp_path)
    paths = {
        service.dem_overlay(
            dem_fixture,
            fingerprint="dem-a",
            bounds=(0, 0, 16, 16),
            bounds_crs="EPSG:3857",
            style=style,
        ).path
        for style in MapStyle
    }

    assert len(paths) == 3
    assert all(path.is_file() for path in paths)


def test_map_asset_is_frozen(dem_fixture, tmp_path):
    asset = MapAssetService(tmp_path).dem_overlay(
        dem_fixture,
        fingerprint="dem-a",
        bounds=(0, 0, 16, 16),
        bounds_crs="EPSG:3857",
        style=MapStyle.ELEVATION,
    )

    with pytest.raises(FrozenInstanceError):
        asset.style = MapStyle.HILLSHADE


def test_dem_overlay_cache_hit_does_not_rewrite(dem_fixture, tmp_path, monkeypatch):
    service = MapAssetService(tmp_path)
    request = {
        "fingerprint": "dem-a",
        "bounds": (0, 0, 16, 16),
        "bounds_crs": "EPSG:3857",
        "style": MapStyle.COMBINED,
    }
    first = service.dem_overlay(dem_fixture, **request)
    modified_ns = first.path.stat().st_mtime_ns

    def fail_save(*args, **kwargs):
        raise AssertionError("a cache hit must not render or rewrite the PNG")

    monkeypatch.setattr(Image.Image, "save", fail_save)
    second = service.dem_overlay(dem_fixture, **request)

    assert second.path == first.path
    assert second.path.stat().st_mtime_ns == modified_ns
    assert second.path.parent == tmp_path / ".lte-data" / "cache" / "maps"


@pytest.mark.parametrize("corruption", ["empty", "bad-png", "rgb", "wrong-size"])
def test_dem_overlay_rebuilds_invalid_cached_png(
    dem_fixture,
    tmp_path,
    corruption,
):
    service = MapAssetService(tmp_path)
    request = {
        "fingerprint": "dem-a",
        "bounds": (0, 0, 16, 16),
        "bounds_crs": "EPSG:3857",
        "style": MapStyle.COMBINED,
    }
    asset = service.dem_overlay(dem_fixture, **request)
    if corruption == "empty":
        asset.path.write_bytes(b"")
    elif corruption == "bad-png":
        asset.path.write_bytes(b"not a png")
    elif corruption == "rgb":
        Image.new("RGB", (16, 16)).save(asset.path)
    else:
        Image.new("RGBA", (1, 1)).save(asset.path)

    rebuilt = service.dem_overlay(dem_fixture, **request)

    with Image.open(rebuilt.path) as image:
        image.load()
        assert image.format == "PNG"
        assert image.mode == "RGBA"
        assert image.size == (16, 16)


def test_dem_overlay_cache_key_covers_render_inputs(
    dem_fixture,
    tmp_path,
    monkeypatch,
):
    service = MapAssetService(tmp_path)
    common = {
        "bounds": (0, 0, 16, 16),
        "bounds_crs": "EPSG:3857",
        "style": MapStyle.COMBINED,
    }
    baseline = service.dem_overlay(dem_fixture, fingerprint="dem-a", **common)

    changed = {
        service.dem_overlay(dem_fixture, fingerprint="dem-b", **common).path,
        service.dem_overlay(
            dem_fixture,
            fingerprint="dem-a",
            **{**common, "bounds": (0, 0, 15, 16)},
        ).path,
        service.dem_overlay(
            dem_fixture,
            fingerprint="dem-a",
            **{**common, "color_limits": (0, 100)},
        ).path,
        service.dem_overlay(
            dem_fixture,
            fingerprint="dem-a",
            **{**common, "light_azimuth": 90},
        ).path,
        service.dem_overlay(
            dem_fixture,
            fingerprint="dem-a",
            **{**common, "light_altitude": 30},
        ).path,
        service.dem_overlay(
            dem_fixture,
            fingerprint="dem-a",
            **{**common, "max_dimension": 8},
        ).path,
    }
    monkeypatch.setattr(map_assets, "MAP_STYLE_VERSION", "map-style-v2")
    changed.add(
        service.dem_overlay(dem_fixture, fingerprint="dem-a", **common).path
    )

    assert baseline.path not in changed
    assert len(changed) == 7
    assert (
        service.dem_overlay(
            dem_fixture,
            fingerprint="dem-a",
            **{**common, "max_dimension": 2048},
        ).path
        == service.dem_overlay(
            dem_fixture,
            fingerprint="dem-a",
            **{**common, "max_dimension": 4096},
        ).path
    )


def test_dem_overlay_atomic_failure_leaves_no_partial_asset(
    dem_fixture,
    tmp_path,
    monkeypatch,
):
    service = MapAssetService(tmp_path)
    real_replace = Path.replace

    def fail_publish(path, target):
        if path.suffix == ".tmp":
            raise OSError("publish failed")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_publish)

    with pytest.raises(OSError, match="publish failed"):
        service.dem_overlay(
            dem_fixture,
            fingerprint="dem-failure",
            bounds=(0, 0, 16, 16),
            bounds_crs="EPSG:3857",
            style=MapStyle.ELEVATION,
        )

    assert not list(service.cache_root.glob("*.png"))
    assert not list(service.cache_root.glob("*.tmp"))


@pytest.mark.parametrize("corruption", ["truncated", "rgb"])
def test_invalid_temporary_png_never_replaces_existing_cache(
    dem_fixture,
    tmp_path,
    monkeypatch,
    corruption,
):
    service = MapAssetService(tmp_path)
    request = {
        "fingerprint": "preserve-existing",
        "bounds": (0, 0, 16, 16),
        "bounds_crs": "EPSG:3857",
        "style": MapStyle.COMBINED,
    }
    asset = service.dem_overlay(dem_fixture, **request)
    original = asset.path.read_bytes()
    actual_valid = map_assets._valid_cached_png
    force_rebuild = True

    def validity(path, *, size):
        nonlocal force_rebuild
        if path == asset.path and force_rebuild:
            force_rebuild = False
            return False
        return actual_valid(path, size=size)

    real_save = Image.Image.save

    def corrupt_save(image, target, *args, **kwargs):
        target = Path(target)
        if corruption == "truncated":
            target.write_bytes(b"truncated")
        else:
            real_save(Image.new("RGB", image.size), target, format="PNG")

    monkeypatch.setattr(map_assets, "_valid_cached_png", validity)
    monkeypatch.setattr(Image.Image, "save", corrupt_save)

    with pytest.raises(ValueError, match="temporary PNG"):
        service.dem_overlay(dem_fixture, **request)

    assert asset.path.read_bytes() == original
    assert actual_valid(asset.path, size=(16, 16))
    assert not list(service.cache_root.glob("*.tmp"))


def test_invalid_published_png_is_rejected_and_removed(
    dem_fixture,
    tmp_path,
    monkeypatch,
):
    service = MapAssetService(tmp_path)
    real_replace = Path.replace

    def corrupt_after_replace(path, target):
        result = real_replace(path, target)
        if path.suffix == ".tmp":
            Path(target).write_bytes(b"corrupted after replace")
        return result

    monkeypatch.setattr(Path, "replace", corrupt_after_replace)

    with pytest.raises(ValueError, match="published PNG"):
        service.dem_overlay(
            dem_fixture,
            fingerprint="corrupt-publish",
            bounds=(0, 0, 16, 16),
            bounds_crs="EPSG:3857",
            style=MapStyle.COMBINED,
        )

    assert not list(service.cache_root.glob("*.png"))


def test_concurrent_same_key_renders_once(dem_fixture, tmp_path, monkeypatch):
    service = MapAssetService(tmp_path)
    real_save = Image.Image.save
    counter_lock = threading.Lock()
    save_count = 0

    def slow_save(image, *args, **kwargs):
        nonlocal save_count
        with counter_lock:
            save_count += 1
        time.sleep(0.05)
        return real_save(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", slow_save)

    def render():
        return service.dem_overlay(
            dem_fixture,
            fingerprint="concurrent",
            bounds=(0, 0, 16, 16),
            bounds_crs="EPSG:3857",
            style=MapStyle.COMBINED,
        ).path

    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(lambda _: render(), range(8)))

    assert len(set(paths)) == 1
    assert save_count == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"fingerprint": " "}, "fingerprint"),
        ({"bounds": [0, 0, 16, 16]}, "bounds"),
        ({"bounds": (0, 0, float("nan"), 16)}, "bounds"),
        ({"bounds": (1, 0, 0, 16)}, "left < right"),
        ({"bounds_crs": "not-a-crs"}, "valid CRS"),
        ({"style": "combined"}, "MapStyle"),
        ({"max_dimension": True}, "max_dimension"),
        ({"max_dimension": 0}, "max_dimension"),
        ({"max_dimension": 4097}, "max_dimension"),
        ({"color_limits": (10, 1)}, "lower < upper"),
        ({"light_azimuth": 360}, "light_azimuth"),
        ({"light_altitude": 0}, "light_altitude"),
    ],
)
def test_dem_overlay_strictly_validates_render_arguments(
    dem_fixture,
    tmp_path,
    changes,
    message,
):
    request = {
        "fingerprint": "dem-a",
        "bounds": (0, 0, 16, 16),
        "bounds_crs": "EPSG:3857",
        "style": MapStyle.COMBINED,
    }
    request.update(changes)

    with pytest.raises(ValueError, match=message):
        MapAssetService(tmp_path).dem_overlay(dem_fixture, **request)


def test_dem_overlay_rejects_network_paths(dem_fixture, tmp_path):
    with pytest.raises(ValueError, match="local filesystem"):
        MapAssetService(tmp_path).dem_overlay(
            "https://example.test/dem.tif",
            fingerprint="remote",
            bounds=(0, 0, 16, 16),
            bounds_crs="EPSG:3857",
            style=MapStyle.ELEVATION,
        )


@pytest.mark.parametrize("redirected_target", ["dem", "cache-parent"])
def test_dem_overlay_rejects_redirected_paths(
    dem_fixture,
    tmp_path,
    monkeypatch,
    redirected_target,
):
    cache_parent = tmp_path / ".lte-data"
    cache_parent.mkdir(exist_ok=True)
    target = dem_fixture if redirected_target == "dem" else cache_parent
    real_is_junction = getattr(Path, "is_junction", lambda path: False)

    def report_redirect(path):
        return path == target or real_is_junction(path)

    monkeypatch.setattr(Path, "is_junction", report_redirect, raising=False)

    with pytest.raises(ValueError, match="redirected"):
        MapAssetService(tmp_path).dem_overlay(
            dem_fixture,
            fingerprint="redirected",
            bounds=(0, 0, 16, 16),
            bounds_crs="EPSG:3857",
            style=MapStyle.ELEVATION,
        )


def test_dem_overlay_rejects_symlinked_cache_parent_without_escape(
    dem_fixture,
    tmp_path,
):
    repo_root = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo_root.mkdir()
    outside.mkdir()
    _symlink_or_skip(repo_root / ".lte-data", outside, directory=True)

    with pytest.raises(ValueError, match="redirected"):
        MapAssetService(repo_root).dem_overlay(
            dem_fixture,
            fingerprint="symlink-parent",
            bounds=(0, 0, 16, 16),
            bounds_crs="EPSG:3857",
            style=MapStyle.ELEVATION,
        )

    assert not list(outside.iterdir())


def test_cache_parent_creation_race_is_revalidated(
    dem_fixture,
    tmp_path,
    monkeypatch,
):
    cache_parent = tmp_path / ".lte-data"
    real_mkdir = Path.mkdir
    real_is_junction = getattr(Path, "is_junction", lambda path: False)
    raced = False

    def race_mkdir(path, *args, **kwargs):
        nonlocal raced
        if path == cache_parent and not raced:
            real_mkdir(path, *args, **kwargs)
            raced = True
            raise FileExistsError(str(path))
        return real_mkdir(path, *args, **kwargs)

    def report_redirect(path):
        return (raced and path == cache_parent) or real_is_junction(path)

    monkeypatch.setattr(Path, "mkdir", race_mkdir)
    monkeypatch.setattr(Path, "is_junction", report_redirect, raising=False)

    with pytest.raises(ValueError, match="redirected"):
        MapAssetService(tmp_path).dem_overlay(
            dem_fixture,
            fingerprint="parent-race",
            bounds=(0, 0, 16, 16),
            bounds_crs="EPSG:3857",
            style=MapStyle.ELEVATION,
        )


def test_dem_overlay_transforms_geographic_bounds_to_web_mercator(tmp_path):
    dem_path = tmp_path / "web-mercator.tif"
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=40,
        height=40,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(-2000, 2000, 100, 100),
    ) as dataset:
        dataset.write(np.arange(1600, dtype="float32").reshape(40, 40), 1)
    requested = (-0.01, -0.01, 0.01, 0.01)

    asset = MapAssetService(tmp_path).dem_overlay(
        dem_path,
        fingerprint="cross-crs",
        bounds=requested,
        bounds_crs="EPSG:4326",
        style=MapStyle.COMBINED,
        max_dimension=32,
    )

    with Image.open(asset.path) as image:
        image.load()
        assert max(image.size) <= 32
        assert image.getextrema()[3] == (255, 255)
    assert asset.bounds == requested
    assert asset.bounds_crs == "EPSG:4326"


def test_partial_boundless_overlay_keeps_extent_and_transparent_edges(
    dem_fixture,
    tmp_path,
):
    requested = (-4, -4, 8, 8)

    asset = MapAssetService(tmp_path).dem_overlay(
        dem_fixture,
        fingerprint="partial",
        bounds=requested,
        bounds_crs="EPSG:3857",
        style=MapStyle.ELEVATION,
    )

    with Image.open(asset.path) as image:
        image.load()
        assert image.size == (12, 12)
        assert image.getextrema()[3] == (0, 255)
    assert asset.bounds == requested


def test_extreme_aspect_ratio_uses_a_bounded_masked_read(tmp_path, monkeypatch):
    dem_path = tmp_path / "wide.tif"
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=10_000,
        height=10,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 10, 1, 1),
    ) as dataset:
        dataset.write(np.ones((10, 10_000), dtype="float32"), 1)
    actual_open = rasterio.open
    read_calls = []

    class TrackedDataset:
        def __init__(self, dataset):
            self.dataset = dataset

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.dataset.close()

        def __getattr__(self, name):
            return getattr(self.dataset, name)

        def read(self, *args, **kwargs):
            read_calls.append(kwargs.copy())
            return self.dataset.read(*args, **kwargs)

    monkeypatch.setattr(
        rasterio,
        "open",
        lambda *args, **kwargs: TrackedDataset(actual_open(*args, **kwargs)),
    )

    asset = MapAssetService(tmp_path).dem_overlay(
        dem_path,
        fingerprint="wide",
        bounds=(0, 0, 10_000, 10),
        bounds_crs="EPSG:3857",
        style=MapStyle.HILLSHADE,
        max_dimension=64,
    )

    assert len(read_calls) == 1
    assert read_calls[0]["out_shape"] == (1, 64)
    assert read_calls[0]["masked"] is True
    assert read_calls[0]["boundless"] is True
    assert Image.open(asset.path).size == (64, 1)


def test_constant_dem_renders_every_style(tmp_path):
    dem_path = tmp_path / "constant.tif"
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 4, 1, 1),
    ) as dataset:
        dataset.write(np.full((4, 4), 12.5, dtype="float32"), 1)
    service = MapAssetService(tmp_path)

    assets = [
        service.dem_overlay(
            dem_path,
            fingerprint="constant",
            bounds=(0, 0, 4, 4),
            bounds_crs="EPSG:3857",
            style=style,
        )
        for style in MapStyle
    ]

    assert len({asset.path for asset in assets}) == 3
    assert all(Image.open(asset.path).getextrema()[3] == (255, 255) for asset in assets)


def test_dem_overlay_rebuilds_a_directory_at_the_cache_leaf(dem_fixture, tmp_path):
    service = MapAssetService(tmp_path)
    request = {
        "fingerprint": "dem-a",
        "bounds": (0, 0, 16, 16),
        "bounds_crs": "EPSG:3857",
        "style": MapStyle.COMBINED,
    }
    asset = service.dem_overlay(dem_fixture, **request)
    asset.path.unlink()
    asset.path.mkdir()

    rebuilt = service.dem_overlay(dem_fixture, **request)

    assert rebuilt.path.is_file()
    assert Image.open(rebuilt.path).mode == "RGBA"


def test_dem_overlay_rejects_non_web_mercator_source(tmp_path):
    dem_path = tmp_path / "geographic.tif"
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
    ) as dataset:
        dataset.write(np.ones((2, 2), dtype="float32"), 1)

    with pytest.raises(ValueError, match="EPSG:3857"):
        MapAssetService(tmp_path).dem_overlay(
            dem_path,
            fingerprint="geographic",
            bounds=(0, 0, 2, 2),
            bounds_crs="EPSG:4326",
            style=MapStyle.ELEVATION,
        )


def test_dem_overlay_rejects_rotated_source_with_clear_error(tmp_path):
    dem_path = tmp_path / "rotated.tif"
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=Affine(1, 0.1, 0, 0.1, -1, 2),
    ) as dataset:
        dataset.write(np.ones((2, 2), dtype="float32"), 1)

    with pytest.raises(ValueError, match="north-up"):
        MapAssetService(tmp_path).dem_overlay(
            dem_path,
            fingerprint="rotated",
            bounds=(0, 0, 2, 2),
            bounds_crs="EPSG:3857",
            style=MapStyle.ELEVATION,
        )


def test_dem_overlay_rejects_all_nodata_without_publishing(tmp_path):
    dem_path = tmp_path / "nodata.tif"
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        nodata=-9999,
        transform=from_origin(0, 2, 1, 1),
    ) as dataset:
        dataset.write(np.full((2, 2), -9999, dtype="float32"), 1)
    service = MapAssetService(tmp_path)

    with pytest.raises(ValueError, match="valid elevation"):
        service.dem_overlay(
            dem_path,
            fingerprint="nodata",
            bounds=(0, 0, 2, 2),
            bounds_crs="EPSG:3857",
            style=MapStyle.COMBINED,
        )

    assert not list(service.cache_root.glob("*.png"))


def test_boundary_display_geojson_does_not_replace_exact_geometry(tmp_path):
    service = MapAssetService(tmp_path)
    exact = box(0, 0, 10, 10).difference(box(4, 4, 6, 6))

    payload = service.boundary_geojson(exact, crs="EPSG:3857", tolerance=0.5)

    assert payload["type"] == "Feature"
    assert exact.area == 96


def test_boundary_geojson_rejects_non_polygon_geometry(tmp_path):
    service = MapAssetService(tmp_path)

    with pytest.raises(ValueError, match="Polygon"):
        service.boundary_geojson(Point(1, 1), crs="EPSG:3857")


def test_boundary_geojson_drops_nan_z_without_mutating_exact_geometry(tmp_path):
    exact = from_wkt(
        "POLYGON Z ((0 0 NaN, 1 0 NaN, 1 1 NaN, 0 1 NaN, 0 0 NaN))"
    )
    original_wkt = exact.wkt

    payload = MapAssetService(tmp_path).boundary_geojson(
        exact,
        crs="EPSG:4326",
    )

    coordinates = payload["geometry"]["coordinates"][0]
    assert all(len(position) == 2 for position in coordinates)
    assert all(np.isfinite(position).all() for position in coordinates)
    json.dumps(payload, allow_nan=False)
    assert exact.wkt == original_wkt
    assert exact.has_z


def test_station_geojson_strictly_filters_and_emits_only_display_fields(tmp_path):
    service = MapAssetService(tmp_path)
    stations = gpd.GeoDataFrame(
        {
            "cell": [1, 2, 3],
            "longitude": [999.0, 999.0, 999.0],
            "latitude": [999.0, 999.0, 999.0],
            "range": [100, 200, 300],
            "samples": [10, 20, 30],
            "created": ["a", "b", "c"],
            "updated": ["d", "e", "f"],
            "secret": ["inside", "edge", "outside"],
        },
        geometry=[Point(5, 5), Point(0, 5), Point(20, 20)],
        crs="EPSG:3857",
    )
    original = stations.copy(deep=True)

    payload = service.station_geojson(
        stations,
        box(0, 0, 10, 10),
        boundary_crs="EPSG:3857",
    )

    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) == 1
    properties = payload["features"][0]["properties"]
    assert set(properties) == {
        "cell",
        "longitude",
        "latitude",
        "range",
        "samples",
        "created",
        "updated",
    }
    assert properties["cell"] == 1
    assert properties["longitude"] != 999.0
    assert properties["latitude"] != 999.0
    assert stations.equals(original)


def test_station_geojson_drops_nan_z_without_mutating_geodataframe(tmp_path):
    stations = gpd.GeoDataFrame(
        {"cell": [7]},
        geometry=[from_wkt("POINT Z (0.5 0.5 NaN)")],
        crs="EPSG:4326",
    )
    original_wkt = stations.geometry.iloc[0].wkt

    payload = MapAssetService(tmp_path).station_geojson(
        stations,
        box(0, 0, 1, 1),
        boundary_crs="EPSG:4326",
    )

    coordinates = payload["features"][0]["geometry"]["coordinates"]
    assert len(coordinates) == 2
    assert np.isfinite(coordinates).all()
    json.dumps(payload, allow_nan=False)
    assert stations.geometry.iloc[0].wkt == original_wkt
    assert stations.geometry.iloc[0].has_z


def test_station_geojson_uses_active_custom_geometry_column(tmp_path):
    stations = gpd.GeoDataFrame(
        {"cell": [7], "site_shape": [from_wkt("POINT Z (0.5 0.5 NaN)")]},
        geometry="site_shape",
        crs="EPSG:4326",
    )

    payload = MapAssetService(tmp_path).station_geojson(
        stations,
        box(0, 0, 1, 1),
        boundary_crs="EPSG:4326",
    )

    assert payload["features"][0]["geometry"]["coordinates"] == (0.5, 0.5)
    json.dumps(payload, allow_nan=False)
    assert stations.active_geometry_name == "site_shape"
    assert stations.site_shape.iloc[0].has_z


@pytest.mark.parametrize("attributes", [{"name": ["missing"]}, {"cell": [None]}])
def test_station_geojson_requires_non_null_cell(tmp_path, attributes):
    stations = gpd.GeoDataFrame(
        attributes,
        geometry=[Point(1, 1)],
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="cell"):
        MapAssetService(tmp_path).station_geojson(
            stations,
            box(0, 0, 2, 2),
            boundary_crs="EPSG:3857",
        )


def test_station_geojson_filters_cross_crs_holes_and_remains_json_safe(tmp_path):
    from pyproj import Transformer

    to_geographic = Transformer.from_crs(
        "EPSG:3857",
        "EPSG:4326",
        always_xy=True,
    )
    projected = [(10, 10), (80, 80), (50, 50), (0, 10), (120, 120)]
    geographic = [Point(*to_geographic.transform(x, y)) for x, y in projected]
    stations = gpd.GeoDataFrame(
        {
            "cell": [7, 7, 8, 9, 10],
            "range": np.asarray([1, 2, 3, 4, 5], dtype="int64"),
            "samples": np.asarray([11, 12, 13, 14, 15], dtype="int64"),
            "created": ["2026-01-01"] * 5,
            "updated": ["2026-01-02"] * 5,
            "secret": ["do-not-emit"] * 5,
        },
        geometry=geographic,
        crs="EPSG:4326",
    )
    boundary = box(0, 0, 100, 100).difference(box(40, 40, 60, 60))

    payload = MapAssetService(tmp_path).station_geojson(
        stations,
        boundary,
        boundary_crs="EPSG:3857",
    )

    assert [feature["properties"]["cell"] for feature in payload["features"]] == [
        7,
        7,
    ]
    for feature in payload["features"]:
        assert set(feature["properties"]) == {
            "cell",
            "longitude",
            "latitude",
            "range",
            "samples",
            "created",
            "updated",
        }
        assert feature["geometry"]["coordinates"] == (
            feature["properties"]["longitude"],
            feature["properties"]["latitude"],
        )
    json.dumps(payload, allow_nan=False)
