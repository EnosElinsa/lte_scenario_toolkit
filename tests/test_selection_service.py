from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from lte_scenario_toolkit.candidate_scanner import (
    Candidate,
    ScanCancelled,
    ScanResult,
)
from lte_scenario_toolkit.figure_service import FigureService
from lte_scenario_toolkit.profiles import ExperimentProfile, FigureSettings, OutputSettings
from lte_scenario_toolkit.selection_service import (
    PreparedSelection,
    SelectionExportError,
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
    assert preflight.boundary_dataset_id == "boundary"
    assert preflight.dem_dataset_id == "dem"
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
        boundary_dataset_id="boundary",
        dem_dataset_id="dem",
    )
    service = SelectionService(SimpleNamespace(root=tmp_path))

    miss_events: list[SelectionProgress] = []
    first = service.scan(preflight, progress=miss_events.append)
    assert miss_events
    assert {event.cache_status for event in miss_events} == {"miss"}
    assert all(event.cache_key for event in miss_events)
    prepared = service.prepared_selection(preflight)
    assert isinstance(prepared, PreparedSelection)
    assert prepared.preflight is preflight
    assert prepared.points.geometry.x.tolist() == [1.5]
    assert prepared.coordinates.tolist() == [[1.5, 1.5]]
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

    fresh_service = SelectionService(SimpleNamespace(root=tmp_path))
    with monkeypatch.context() as context:
        context.setattr(
            "lte_scenario_toolkit.selection_service.gpd.read_file",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("cache hit must not load vectors")
            ),
        )
        assert fresh_service.scan(preflight) == first
    cancel = Event()
    cancel.set()
    with pytest.raises(ScanCancelled) as captured:
        service.scan(preflight, cancel=cancel)
    assert captured.value.code == "scan.cancelled"
    with pytest.raises(AssertionError, match="cache miss"):
        service.scan(preflight, force=True)

    missing_points = replace(preflight, points_path=tmp_path / "missing.geojson")
    with pytest.raises(SelectionScanError) as captured:
        service.scan(missing_points, force=True)
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


@pytest.fixture
def selection_export_fixture(tmp_path):
    points_path = tmp_path / "points.geojson"
    boundary_path = tmp_path / "boundary.geojson"
    dem_path = tmp_path / "dem.tif"
    gpd.GeoDataFrame(
        {"cell": [7, 8], "range": [100, 200]},
        geometry=[Point(0.0, 1.0), Point(0.5, 1.5)],
        crs="EPSG:3857",
    ).to_file(points_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"name": ["city"]},
        geometry=[box(-1, -1, 3, 3)],
        crs="EPSG:3857",
    ).to_file(boundary_path, driver="GeoJSON")
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
        dem.write(np.arange(16, dtype="float32").reshape(4, 4), 1)
    output_root = tmp_path / "runs"
    profile = _profile(
        tmp_path,
        target_count=2,
        max_rects=1,
        output_root=tmp_path / "stale-output",
    )
    preflight = SelectionPreflight(
        scenario_id="chicago",
        profile=profile,
        points_path=points_path,
        boundary_path=boundary_path,
        dem_path=dem_path,
        output_root=output_root,
        boundary_fingerprint="boundary-a",
        points_fingerprint="points-a",
        dem_fingerprint="dem-a",
        boundary_dataset_id="boundary",
        dem_dataset_id="dem",
    )
    candidate = Candidate(0, 2, 0.0, 0.0, 1.0, 1.0)
    result = ScanResult((candidate,), 1, 1, True, "row-sweep-v1")
    service = SelectionService(SimpleNamespace(root=tmp_path))
    prepared = service.prepared_selection(preflight)
    return service, preflight, result, candidate, prepared


def test_export_publishes_traceable_csv_preview_and_exact_run_schemas(
    selection_export_fixture,
    monkeypatch,
):
    service, preflight, result, candidate, prepared = selection_export_fixture
    monkeypatch.setattr(
        "lte_scenario_toolkit.selection_service.gpd.read_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("export must reuse PreparedSelection")
        ),
    )
    monkeypatch.setattr(
        "lte_scenario_toolkit.io._git_commit",
        lambda repository: "abc123",
    )
    monkeypatch.setattr(
        "lte_scenario_toolkit.io.software_versions",
        lambda: {"python": "3.test"},
    )

    run_dir = service.export(
        preflight,
        result,
        candidate,
        output_root=preflight.output_root,
        artifacts={"csv", "preview_png"},
        entrypoint=["lte-select-sites", "--select-index", "1"],
    )

    assert service.prepared_selection(preflight) is prepared
    base = "chicago_2m_target2_tol0"
    csv_path = run_dir / f"{base}.csv"
    preview_path = run_dir / f"{base}.png"
    assert csv_path.stat().st_size > 0
    assert preview_path.stat().st_size > 0
    frame = pd.read_csv(csv_path)
    assert frame["cell"].tolist() == [7, 8]
    assert frame["run_id"].nunique() == 1
    assert frame["scenario_id"].unique().tolist() == ["chicago"]
    assert frame["profile_id"].unique().tolist() == ["chicago-default"]
    assert frame["candidate_id"].unique().tolist() == ["candidate-0001"]

    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (run_dir / "selection.json").read_text(encoding="utf-8")
    )
    assert frame["run_id"].unique().tolist() == [run_record["run_id"]]
    assert selection.keys() == {
        "schema_version",
        "run_id",
        "scenario_id",
        "profile_id",
        "candidate_id",
        "target_crs",
        "scan",
        "candidates",
    }
    assert selection["run_id"] == run_record["run_id"]
    assert selection["candidate_id"] == "candidate-0001"
    assert selection["target_crs"] == "EPSG:3857"
    assert selection["scan"] == {
        "algorithm_version": "row-sweep-v1",
        "cache_key": run_record["metadata"]["cache"]["key"],
        "checked_positions": 1,
        "total_positions": 1,
        "completed": True,
    }
    assert len(selection["candidates"]) == 1
    selected_candidate = selection["candidates"][0]
    assert selected_candidate.keys() == {
        "candidate_id",
        "candidate_index",
        "flat_grid_id",
        "point_count",
        "left_x",
        "bottom_y",
        "center_x",
        "center_y",
        "bounds",
        "geometry",
        "dem_statistics",
        "selected_station_id_field",
        "selected_station_ids",
    }
    assert selected_candidate["candidate_index"] == 1
    assert selected_candidate["bounds"] == [0.0, 0.0, 2.0, 2.0]
    assert selected_candidate["geometry"]["type"] == "Polygon"
    assert selected_candidate["selected_station_ids"] == [7, 8]
    assert selected_candidate["selected_station_id_field"] == "cell"
    assert selected_candidate["dem_statistics"]["valid_pixel_count"] == 4

    assert run_record["status"] == "completed"
    assert run_record["errors"] == []
    assert run_record["metadata"] == {
        "schema_version": 1,
        "run_kind": "selection",
        "candidate": {
            "candidate_id": "candidate-0001",
            "flat_grid_id": 0,
            "point_count": 2,
            "center_x": 1.0,
            "center_y": 1.0,
        },
        "scanner": {
            "algorithm_version": "row-sweep-v1",
            "checked_positions": 1,
            "total_positions": 1,
        },
        "cache": {
            "schema_version": 1,
            "key": selection["scan"]["cache_key"],
        },
        "inputs": {
            "points": {
                "dataset_id": "points",
                "fingerprint": "points-a",
            },
            "boundary": {
                "dataset_id": "boundary",
                "fingerprint": "boundary-a",
            },
            "dem": {
                "dataset_id": "dem",
                "fingerprint": "dem-a",
                "path": str(preflight.dem_path.resolve()),
            },
        },
        "parameters": {
            "target_crs": "EPSG:3857",
            "rectangle_size_m": 2,
            "target_base_station_count": 2,
            "count_tolerance": 0,
            "scan_mode": "fast",
            "strategy": "sequential",
            "step_m": 1,
            "max_candidates": 1,
            "minimum_center_spacing_m": 2,
            "random_seed": 7,
        },
        "sidecars": {
            "config": "run-config.yaml",
            "selection": "selection.json",
            "compatibility_records": [],
        },
        "requested_artifacts": ["csv", "preview_png"],
        "artifact_paths": {
            "csv": f"{base}.csv",
            "preview_png": f"{base}.png",
        },
        "git_commit": "abc123",
        "software_versions": {"python": "3.test"},
        "entrypoint": ["lte-select-sites", "--select-index", "1"],
    }
    assert run_record["artifacts"] == list(
        run_record["metadata"]["artifact_paths"].values()
    )
    profile_snapshot = yaml.safe_load(
        (run_dir / "run-config.yaml").read_text(encoding="utf-8")
    )
    assert Path(profile_snapshot["outputs"]["root"]).resolve() == (
        preflight.output_root.resolve()
    )
    figure_source = FigureService.load_source(run_dir)
    assert figure_source.dem_path == preflight.dem_path.resolve()
    assert not list(run_dir.glob(".*.tmp-*"))


def test_export_csv_xy_matches_non_web_mercator_profile_crs(tmp_path):
    points_path = tmp_path / "points.geojson"
    boundary_path = tmp_path / "boundary.geojson"
    dem_path = tmp_path / "dem.tif"
    gpd.GeoDataFrame(
        {"cell": [7]},
        geometry=[Point(500_000, 4_500_000)],
        crs="EPSG:32618",
    ).to_file(points_path, driver="GeoJSON")
    gpd.GeoDataFrame(
        {"name": ["city"]},
        geometry=[box(499_999, 4_499_999, 500_003, 4_500_003)],
        crs="EPSG:32618",
    ).to_file(boundary_path, driver="GeoJSON")
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="float32",
        crs="EPSG:32618",
        transform=from_origin(499_999, 4_500_003, 1, 1),
    ) as dem:
        dem.write(np.ones((4, 4), dtype="float32"), 1)
    output_root = tmp_path / "runs"
    profile = _profile(
        tmp_path,
        target_crs="EPSG:32618",
        target_count=1,
        max_rects=1,
        output_root=output_root,
    )
    preflight = SelectionPreflight(
        scenario_id="chicago",
        profile=profile,
        points_path=points_path,
        boundary_path=boundary_path,
        dem_path=dem_path,
        output_root=output_root,
        boundary_fingerprint="boundary-utm",
        points_fingerprint="points-utm",
        dem_fingerprint="dem-utm",
        boundary_dataset_id="boundary",
        dem_dataset_id="dem",
    )
    candidate = Candidate(
        0,
        1,
        500_000,
        4_500_000,
        500_001,
        4_500_001,
    )
    result = ScanResult((candidate,), 1, 1, True, "row-sweep-v1")
    service = SelectionService(SimpleNamespace(root=tmp_path))

    run_dir = service.export(
        preflight,
        result,
        candidate,
        output_root=output_root,
        artifacts={"csv"},
    )

    frame = pd.read_csv(next(run_dir.glob("*.csv")))
    assert frame.loc[0, "X"] == pytest.approx(500_000)
    assert frame.loc[0, "Y"] == pytest.approx(4_500_000)
    assert frame.loc[0, "center_x"] == pytest.approx(500_001)
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))[
        "metadata"
    ]
    assert metadata["parameters"]["target_crs"] == "EPSG:32618"


@pytest.mark.parametrize(
    "case",
    [
        "empty-artifacts",
        "bare-string",
        "duplicate-artifact",
        "unknown-artifact",
        "root-mismatch",
        "incomplete-scan",
        "alien-candidate",
        "duplicate-candidate",
    ],
)
def test_export_rejects_invalid_requests_before_begin(
    selection_export_fixture,
    tmp_path,
    case,
):
    service, preflight, result, candidate, _ = selection_export_fixture
    output_root = preflight.output_root
    artifacts = {"csv"}
    if case == "empty-artifacts":
        artifacts = set()
    elif case == "bare-string":
        artifacts = "csv"
    elif case == "duplicate-artifact":
        artifacts = ["csv", "csv"]
    elif case == "unknown-artifact":
        artifacts = {"pdf"}
    elif case == "root-mismatch":
        output_root = tmp_path / "other-output"
    elif case == "incomplete-scan":
        result = replace(result, completed=False)
    elif case == "alien-candidate":
        candidate = replace(candidate, flat_grid_id=999)
    elif case == "duplicate-candidate":
        result = replace(result, candidates=(candidate, candidate))

    with pytest.raises(SelectionExportError):
        service.export(
            preflight,
            result,
            candidate,
            output_root=output_root,
            artifacts=artifacts,
        )

    assert not preflight.output_root.exists()
    assert not (tmp_path / "other-output").exists()


def test_export_uses_deterministic_dataset_ids_for_a_minimal_manual_preflight(
    selection_export_fixture,
):
    service, preflight, result, candidate, _ = selection_export_fixture
    preflight = replace(
        preflight,
        output_root=preflight.output_root.parent / "manual-runs",
        boundary_dataset_id=None,
        dem_dataset_id=None,
    )

    run_dir = service.export(
        preflight,
        result,
        candidate,
        output_root=preflight.output_root,
        artifacts={"csv"},
    )

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["metadata"]["inputs"]["points"]["dataset_id"] == "points"
    assert record["metadata"]["inputs"]["boundary"]["dataset_id"] == "boundary"
    assert record["metadata"]["inputs"]["dem"]["dataset_id"] == "dem"


def test_export_wraps_vector_snapshot_failure_without_creating_output(
    selection_export_fixture,
):
    service, preflight, result, candidate, _ = selection_export_fixture
    preflight = replace(
        preflight,
        points_path=preflight.points_path.parent / "missing-points.geojson",
        output_root=preflight.output_root.parent / "missing-input-runs",
    )

    with pytest.raises(SelectionExportError) as captured:
        service.export(
            preflight,
            result,
            candidate,
            output_root=preflight.output_root,
            artifacts={"csv"},
        )

    assert captured.value.code == "export.inputs"
    assert not preflight.output_root.exists()


def test_export_samples_points_without_full_band_dem_read(
    selection_export_fixture,
    monkeypatch,
):
    service, preflight, result, candidate, _ = selection_export_fixture
    real_open = rasterio.open

    class GuardedDataset:
        def __init__(self, dataset):
            self.dataset = dataset

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.dataset.close()

        def __getattr__(self, name):
            return getattr(self.dataset, name)

        def read(self, *args, **kwargs):
            assert kwargs.get("window") is not None, "full-band DEM read is forbidden"
            return self.dataset.read(*args, **kwargs)

    monkeypatch.setattr(
        "lte_scenario_toolkit.selection_service.rasterio.open",
        lambda *args, **kwargs: GuardedDataset(real_open(*args, **kwargs)),
    )

    run_dir = service.export(
        preflight,
        result,
        candidate,
        output_root=preflight.output_root,
        artifacts={"csv"},
    )

    assert next(run_dir.glob("*.csv")).stat().st_size > 0


@pytest.mark.parametrize(
    ("artifact", "filename"),
    [
        ("terrain_png", "chicago_2m_target2_tol0_3d.png"),
        ("terrain_eps", "chicago_2m_target2_tol0_3d.eps"),
        ("terrain_html", "chicago_2m_target2_tol0_3d.html"),
    ],
)
def test_export_supports_each_independent_terrain_artifact(
    selection_export_fixture,
    artifact,
    filename,
):
    service, preflight, result, candidate, _ = selection_export_fixture

    run_dir = service.export(
        preflight,
        result,
        candidate,
        output_root=preflight.output_root,
        artifacts={artifact},
    )

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["artifacts"] == [filename]
    assert (run_dir / record["artifacts"][0]).stat().st_size > 0
    if artifact == "terrain_eps":
        assert not list(run_dir.glob("*.png"))
    assert not list(run_dir.glob(".*.tmp*"))


def test_export_allows_a_zero_station_candidate(selection_export_fixture):
    service, preflight, _, _, _ = selection_export_fixture
    profile = replace(preflight.profile, target_count=0)
    preflight = replace(
        preflight,
        profile=profile,
        output_root=preflight.output_root.parent / "empty-runs",
    )
    candidate = Candidate(1, 0, 2.0, 0.0, 3.0, 1.0)
    result = ScanResult((candidate,), 1, 1, True, "row-sweep-v1")

    run_dir = service.export(
        preflight,
        result,
        candidate,
        output_root=preflight.output_root,
        artifacts={"csv"},
    )

    frame = pd.read_csv(next(run_dir.glob("*.csv")))
    selection = json.loads(
        (run_dir / "selection.json").read_text(encoding="utf-8")
    )
    assert frame.empty
    assert selection["candidates"][0]["selected_station_ids"] == []


def test_export_rejects_when_any_selected_station_has_no_valid_elevation(
    selection_export_fixture,
    tmp_path,
):
    service, preflight, result, candidate, _ = selection_export_fixture
    dem_path = tmp_path / "partially-invalid-dem.tif"
    elevation = np.arange(16, dtype="float32").reshape(4, 4)
    elevation[3, 0] = -9999
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
        nodata=-9999,
    ) as dem:
        dem.write(elevation, 1)
    preflight = replace(
        preflight,
        dem_path=dem_path,
        dem_fingerprint="dem-partial",
        output_root=tmp_path / "invalid-elevation-runs",
    )

    with pytest.raises(SelectionExportError) as captured:
        service.export(
            preflight,
            result,
            candidate,
            output_root=preflight.output_root,
            artifacts={"csv"},
        )

    assert captured.value.code == "export.elevation"
    assert not preflight.output_root.exists()


def test_export_publishes_partial_when_one_requested_artifact_fails(
    selection_export_fixture,
    monkeypatch,
):
    service, preflight, result, candidate, _ = selection_export_fixture

    def fail_preview(*args, **kwargs):
        raise RuntimeError("preview boom")

    monkeypatch.setattr(
        "lte_scenario_toolkit.selection_service.visualization.save_preview",
        fail_preview,
    )

    run_dir = service.export(
        preflight,
        result,
        candidate,
        output_root=preflight.output_root,
        artifacts={"csv", "preview_png"},
    )

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "partial"
    assert record["artifacts"] == ["chicago_2m_target2_tol0.csv"]
    assert record["errors"] == [
        {
            "artifact": "preview_png",
            "code": "artifact.preview_png.failed",
            "message": "RuntimeError: preview boom",
        }
    ]
    assert record["metadata"]["requested_artifacts"] == [
        "csv",
        "preview_png",
    ]
    assert record["metadata"]["artifact_paths"] == {
        "csv": "chicago_2m_target2_tol0.csv"
    }
    assert record["artifacts"] == list(
        record["metadata"]["artifact_paths"].values()
    )
    assert not list(run_dir.glob(".*.tmp*"))


def test_export_abandons_when_only_requested_artifact_is_zero_bytes(
    selection_export_fixture,
    monkeypatch,
):
    service, preflight, result, candidate, _ = selection_export_fixture

    def empty_preview(points, boundary, selected, config):
        del points, boundary, selected
        path = Path(config["preview_png"])
        path.write_bytes(b"")
        return path

    monkeypatch.setattr(
        "lte_scenario_toolkit.selection_service.visualization.save_preview",
        empty_preview,
    )

    with pytest.raises(SelectionExportError) as captured:
        service.export(
            preflight,
            result,
            candidate,
            output_root=preflight.output_root,
            artifacts={"preview_png"},
        )

    assert captured.value.code == "export.no_artifacts"
    assert not list(preflight.output_root.glob("*/*/*"))
    assert not list(preflight.output_root.rglob(".staging-*"))
    assert not list(preflight.output_root.rglob(".*.tmp*"))


@pytest.mark.parametrize("sidecar", ["run-config.yaml", "selection.json"])
def test_export_abandons_when_a_mandatory_sidecar_is_empty(
    selection_export_fixture,
    monkeypatch,
    sidecar,
):
    service, preflight, result, candidate, _ = selection_export_fixture

    if sidecar == "run-config.yaml":
        monkeypatch.setattr(
            "lte_scenario_toolkit.selection_service.dump_profile",
            lambda profile, path: Path(path).write_bytes(b""),
        )
    else:
        monkeypatch.setattr(
            "lte_scenario_toolkit.selection_service.io.atomic_write_json",
            lambda path, document: Path(path).write_bytes(b""),
        )

    with pytest.raises(SelectionExportError):
        service.export(
            preflight,
            result,
            candidate,
            output_root=preflight.output_root,
            artifacts={"csv"},
        )

    assert not list(preflight.output_root.glob("*/*/*"))
    assert not list(preflight.output_root.rglob(".staging-*"))


def test_export_revalidates_mandatory_sidecars_after_artifact_rendering(
    selection_export_fixture,
    monkeypatch,
):
    service, preflight, result, candidate, _ = selection_export_fixture

    def damage_selection_sidecar(points, boundary, selected, config):
        del points, boundary, selected
        temporary = Path(config["preview_png"])
        (temporary.parent / "selection.json").unlink()
        temporary.write_bytes(b"preview")
        return temporary

    monkeypatch.setattr(
        "lte_scenario_toolkit.selection_service.visualization.save_preview",
        damage_selection_sidecar,
    )

    with pytest.raises(SelectionExportError):
        service.export(
            preflight,
            result,
            candidate,
            output_root=preflight.output_root,
            artifacts={"preview_png"},
        )

    assert not list(preflight.output_root.glob("*/*/*"))
    assert not list(preflight.output_root.rglob(".staging-*"))


def test_export_allocates_unique_runs_for_two_same_second_publications(
    selection_export_fixture,
    monkeypatch,
):
    service, preflight, result, candidate, _ = selection_export_fixture
    monkeypatch.setattr(
        "lte_scenario_toolkit.run_service._normalise_created_at",
        lambda value: (
            "2026-07-17T00:00:00Z",
            datetime(2026, 7, 17, tzinfo=timezone.utc),
        ),
    )

    first = service.export(
        preflight,
        result,
        candidate,
        output_root=preflight.output_root,
        artifacts={"csv"},
    )
    second = service.export(
        preflight,
        result,
        candidate,
        output_root=preflight.output_root,
        artifacts={"csv"},
    )

    assert first != second
    assert first.is_dir() and second.is_dir()


def test_export_retains_complete_staging_when_final_publication_move_fails(
    selection_export_fixture,
    monkeypatch,
):
    service, preflight, result, candidate, _ = selection_export_fixture
    original_replace = Path.replace

    def fail_final_move(source, destination):
        if source.name.startswith(".staging-"):
            raise OSError("simulated final publication move failure")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_final_move)

    with pytest.raises(SelectionExportError, match="publication move failure"):
        service.export(
            preflight,
            result,
            candidate,
            output_root=preflight.output_root,
            artifacts={"csv"},
        )

    staging = list(preflight.output_root.rglob(".staging-*"))
    assert len(staging) == 1
    assert (staging[0] / "run.json").stat().st_size > 0
    assert (staging[0] / "run-config.yaml").stat().st_size > 0
    assert (staging[0] / "selection.json").stat().st_size > 0
    assert next(staging[0].glob("*.csv")).stat().st_size > 0
