from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from lte_scenario_toolkit.candidate_scanner import (
    Candidate,
    ScanCancelled,
)
from lte_scenario_toolkit.profiles import ExperimentProfile, FigureSettings, OutputSettings
from lte_scenario_toolkit.selection_service import (
    SelectionPreflight,
    SelectionPreflightError,
    SelectionProgress,
    SelectionScanError,
    SelectionService,
    SelectionStatisticsError,
    stream_dem_statistics,
)


def _profile(tmp_path: Path, **changes) -> ExperimentProfile:
    profile = ExperimentProfile(
        schema_version=2,
        profile_id="chicago-default",
        display_name="Chicago default",
        scenario_id="chicago",
        points_dataset_id="points",
        random_seed=7,
        target_crs="EPSG:3857",
        rect_size=2,
        target_count=1,
        tolerance=0,
        scan_mode="fast",
        strategy="sequential",
        scan_step=1,
        max_rects=2,
        min_spacing=2,
        output_root=tmp_path / "output",
        outputs=OutputSettings(),
        figure=FigureSettings(),
    )
    return replace(profile, **changes)


def test_preflight_rejects_nonready_scenario_without_output_creation(tmp_path):
    class PendingCatalog:
        root = tmp_path

        @staticmethod
        def scenario_status(scenario_id):
            assert scenario_id == "chicago"
            return "dem-pending"

    profile = SimpleNamespace(scenario_id="chicago")
    output = tmp_path / "output"

    with pytest.raises(SelectionPreflightError, match="ready") as captured:
        SelectionService(PendingCatalog()).preflight(profile, output_root=output)

    assert captured.value.code == "scenario.not_ready"
    assert captured.value.details == {
        "scenario_id": "chicago",
        "status": "dem-pending",
    }
    assert not output.exists()


def test_preflight_validates_directly_constructed_profile_before_io(tmp_path):
    class ReadyCatalog:
        root = tmp_path

        @staticmethod
        def scenario_status(scenario_id):
            assert scenario_id == "chicago"
            return "ready"

    with pytest.raises(SelectionPreflightError) as captured:
        SelectionService(ReadyCatalog()).preflight(
            _profile(tmp_path, rect_size=0),
            output_root=tmp_path / "output",
        )

    assert captured.value.code == "profile.invalid"
    assert "greater than zero" in str(captured.value)


def test_preflight_maps_unknown_points_dataset_to_profile_field(tmp_path):
    class Catalog:
        root = tmp_path

        @staticmethod
        def scenario_status(scenario_id):
            return "ready"

        @staticmethod
        def scenario(scenario_id):
            return {
                "scenario_id": scenario_id,
                "boundary_dataset_id": "boundary",
                "dem_dataset_id": "dem",
            }

        @staticmethod
        def dataset(dataset_id):
            raise KeyError(dataset_id)

    with pytest.raises(SelectionPreflightError) as captured:
        SelectionService(Catalog()).preflight(
            _profile(tmp_path, points_dataset_id="missing-points"),
            output_root=tmp_path / "output",
        )

    assert captured.value.code == "inputs.points_dataset_id"


def test_preflight_resolves_catalog_inputs_and_manifest_fingerprints(
    tmp_path,
    monkeypatch,
):
    points_path = tmp_path / "points.geojson"
    boundary_path = tmp_path / "boundary.geojson"
    dem_path = tmp_path / "dem.tif"
    points_path.write_text("points", encoding="utf-8")
    boundary_path.write_text("boundary", encoding="utf-8")
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=1,
        height=1,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 1, 1, 1),
    ) as dem:
        dem.write(np.ones((1, 1), dtype="float32"), 1)
    datasets = {
        "points": {
            "dataset_id": "points",
            "role": "points",
            "entrypoint": "points.geojson",
        },
        "boundary": {
            "dataset_id": "boundary",
            "role": "boundary",
            "entrypoint": "boundary.geojson",
        },
        "dem": {
            "dataset_id": "dem",
            "role": "dem",
            "entrypoint": "dem.tif",
        },
    }

    class Catalog:
        root = tmp_path

        @staticmethod
        def scenario_status(scenario_id):
            assert scenario_id == "chicago"
            return "ready"

        @staticmethod
        def scenario(scenario_id):
            assert scenario_id == "chicago"
            return {"boundary_dataset_id": "boundary", "dem_dataset_id": "dem"}

        @staticmethod
        def dataset(dataset_id):
            return datasets[dataset_id]

        @staticmethod
        def resolve(path):
            return (tmp_path / path).resolve()

    manifest = {
        "schema_version": 2,
        "datasets": [
            {
                **dataset,
                "files": [
                    {
                        "path": dataset["entrypoint"],
                        "size_bytes": (tmp_path / dataset["entrypoint"]).stat().st_size,
                        "sha256": character * 64,
                    }
                ],
            }
            for character, dataset in zip("abc", datasets.values(), strict=True)
        ],
    }
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "lte_scenario_toolkit.selection_service.validate_scenario_data",
        lambda catalog, scenario_id, **kwargs: SimpleNamespace(
            ok=True,
            status="ready",
            messages=[],
        ),
    )
    output = tmp_path / "output"

    preflight = SelectionService(Catalog()).preflight(
        _profile(tmp_path),
        output_root=output,
    )

    assert preflight.points_path == points_path.resolve()
    assert preflight.boundary_path == boundary_path.resolve()
    assert preflight.dem_path == dem_path.resolve()
    assert len(preflight.points_fingerprint) == 64
    assert len(preflight.boundary_fingerprint) == 64
    assert len(preflight.dem_fingerprint) == 64
    assert not output.exists()

    nested_output = tmp_path / "new" / "nested" / "output"
    nested = SelectionService(Catalog()).preflight(
        _profile(tmp_path),
        output_root=nested_output,
    )
    assert nested.output_root == nested_output.resolve()
    assert not (tmp_path / "new").exists()

    occupied_parent = tmp_path / "occupied"
    occupied_parent.write_text("not a directory", encoding="utf-8")
    with pytest.raises(SelectionPreflightError) as captured:
        SelectionService(Catalog()).preflight(
            _profile(tmp_path),
            output_root=occupied_parent / "output",
        )
    assert captured.value.code == "outputs.root"
    assert not (occupied_parent / "output").exists()


def test_stream_dem_statistics_uses_valid_pixels_only():
    profile = {
        "driver": "GTiff",
        "height": 4,
        "width": 4,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3857",
        "nodata": -9999,
        "transform": from_origin(0, 4, 1, 1),
    }
    values = np.asarray(
        [[1, 2, 3, 4], [5, -9999, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        dtype="float32",
    )
    with MemoryFile() as memory:
        with memory.open(**profile) as dem:
            dem.write(values, 1)
            stats = stream_dem_statistics(
                dem,
                box(0, 0, 4, 4),
                geometry_crs="EPSG:3857",
            )

    assert stats.minimum == 1
    assert stats.maximum == 16
    assert stats.mean == pytest.approx(130 / 15)
    assert stats.elevation_range == 15
    assert stats.valid_pixel_count == 15


def test_stream_dem_statistics_masks_pixels_outside_geometry():
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": from_origin(0, 2, 1, 1),
    }
    with MemoryFile() as memory:
        with memory.open(**profile) as dem:
            dem.write(np.asarray([[1, 100], [3, 200]], dtype="float32"), 1)
            stats = stream_dem_statistics(
                dem,
                box(0, 0, 1, 2),
                geometry_crs="EPSG:3857",
            )

    assert stats.minimum == 1
    assert stats.maximum == 3
    assert stats.mean == 2
    assert stats.valid_pixel_count == 2


def test_stream_dem_statistics_reprojects_geometry_to_dem_crs():
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": from_origin(0, 222640, 111320, 111320),
    }
    with MemoryFile() as memory:
        with memory.open(**profile) as dem:
            dem.write(np.asarray([[1, 2], [3, 4]], dtype="float32"), 1)
            stats = stream_dem_statistics(
                dem,
                box(0, 0, 1, 2),
                geometry_crs="EPSG:4326",
            )

    assert stats.valid_pixel_count == 2
    assert stats.mean == 2


def test_stream_dem_statistics_rejects_geometry_without_valid_pixels():
    profile = {
        "driver": "GTiff",
        "height": 1,
        "width": 1,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3857",
        "nodata": -9999,
        "transform": from_origin(0, 1, 1, 1),
    }
    with MemoryFile() as memory:
        with memory.open(**profile) as dem:
            dem.write(np.asarray([[-9999]], dtype="float32"), 1)
            with pytest.raises(ValueError, match="valid DEM pixels"):
                stream_dem_statistics(
                    dem,
                    box(0, 0, 1, 1),
                    geometry_crs="EPSG:3857",
                )


def test_scan_uses_shared_cache_and_force_bypasses_reads(tmp_path, monkeypatch):
    points_path = tmp_path / "points.geojson"
    boundary_path = tmp_path / "boundary.geojson"
    dem_path = tmp_path / "dem.tif"
    gpd.GeoDataFrame(
        {"cell": [7]},
        geometry=[Point(1.5, 1.5)],
        crs="EPSG:3857",
    ).to_file(points_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"name": ["city"]},
        geometry=[box(0, 0, 5, 5)],
        crs="EPSG:3857",
    ).to_file(boundary_path, driver="GeoJSON")
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=5,
        height=5,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 5, 1, 1),
    ) as dem:
        dem.write(np.ones((5, 5), dtype="float32"), 1)
    profile = _profile(tmp_path)
    preflight = SelectionPreflight(
        scenario_id="chicago",
        profile=profile,
        points_path=points_path,
        boundary_path=boundary_path,
        dem_path=dem_path,
        output_root=tmp_path / "output",
        boundary_fingerprint="boundary-a",
        points_fingerprint="points-a",
        dem_fingerprint="dem-a",
    )
    service = SelectionService(SimpleNamespace(root=tmp_path))

    miss_events: list[SelectionProgress] = []
    first = service.scan(preflight, progress=miss_events.append)
    assert miss_events
    assert {event.cache_status for event in miss_events} == {"miss"}
    assert all(event.cache_key for event in miss_events)
    monkeypatch.setattr(
        "lte_scenario_toolkit.selection_service.scan_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    hit_events: list[SelectionProgress] = []
    second = service.scan(preflight, progress=hit_events.append)

    assert second == first
    assert len(hit_events) == 1
    assert hit_events[0].cache_status == "hit"
    assert hit_events[0].cache_key == miss_events[0].cache_key
    cancel = Event()
    cancel.set()
    with pytest.raises(ScanCancelled) as captured:
        service.scan(preflight, cancel=cancel)
    assert captured.value.code == "scan.cancelled"
    with pytest.raises(AssertionError, match="cache miss"):
        service.scan(preflight, force=True)

    missing_points = replace(preflight, points_path=tmp_path / "missing.geojson")
    with pytest.raises(SelectionScanError) as captured:
        service.scan(missing_points)
    assert captured.value.code == "scan.inputs"


def test_candidate_statistics_uses_candidate_bounds_and_profile_crs(tmp_path):
    dem_path = tmp_path / "dem.tif"
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
    ) as dem:
        dem.write(np.arange(1, 17, dtype="float32").reshape(4, 4), 1)
    profile = _profile(tmp_path)
    preflight = SelectionPreflight(
        scenario_id="chicago",
        profile=profile,
        points_path=tmp_path / "points.geojson",
        boundary_path=tmp_path / "boundary.geojson",
        dem_path=dem_path,
        output_root=tmp_path / "output",
        boundary_fingerprint="boundary-a",
        points_fingerprint="points-a",
        dem_fingerprint="dem-a",
    )

    stats = SelectionService(SimpleNamespace(root=tmp_path)).candidate_statistics(
        preflight,
        Candidate(0, 1, 0, 0, 1, 1),
    )

    assert stats.valid_pixel_count == 4
    assert stats.mean == pytest.approx((9 + 10 + 13 + 14) / 4)

    missing_dem = replace(preflight, dem_path=tmp_path / "missing.tif")
    with pytest.raises(SelectionStatisticsError) as captured:
        SelectionService(SimpleNamespace(root=tmp_path)).candidate_statistics(
            missing_dem,
            Candidate(0, 1, 0, 0, 1, 1),
        )
    assert captured.value.code == "statistics.failed"
