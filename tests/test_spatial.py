from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from lte_scenario_toolkit.spatial import (
    discover_boundary_layers,
    prepare_spatial_data,
    resolve_io_paths,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_resolve_io_paths_selects_city_by_name(tmp_path):
    points_dir = tmp_path / "points" / "stations"
    boundary_dir = tmp_path / "boundaries" / "TestCity"
    points_dir.mkdir(parents=True)
    boundary_dir.mkdir(parents=True)
    (points_dir / "stations.shp").touch()
    (boundary_dir / "city_boundary.shp").touch()

    config = {
        "points_root": tmp_path / "points",
        "points_layer": "stations",
        "boundary_root": tmp_path / "boundaries",
        "city_name": "testcity",
        "dem_path": tmp_path / "dem.tif",
        "output_root": tmp_path / "output",
        "output_dir_is_final": True,
        "rect_size": 1000,
        "target_count": 3,
        "tolerance": 1,
        "scan_step": 50,
        "min_spacing": 500,
        "strategy": "uniform",
        "random_seed": 7,
    }

    paths = resolve_io_paths(config)

    assert paths["city_id"] == 1
    assert paths["boundary_folder"] == "TestCity"
    assert paths["points_shp"] == points_dir / "stations.shp"
    assert paths["boundary_shp"] == boundary_dir / "city_boundary.shp"
    assert paths["output_dir"] == tmp_path / "output"
    assert paths["output_dir"].is_dir()
    assert paths["cache_json"].name.endswith("_uniform_seed7_cache.json")


def test_discover_boundary_layers_rejects_empty_root(tmp_path):
    with pytest.raises(FileNotFoundError, match="boundary"):
        discover_boundary_layers(tmp_path)


def test_prepare_spatial_data_reprojects_and_filters_points():
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    points = gpd.GeoDataFrame(
        {"station": ["inside", "outside"]},
        geometry=[Point(0.5, 0.5), Point(2, 2)],
        crs="EPSG:4326",
    )

    selected, geometry, coordinates = prepare_spatial_data(
        points, boundary, target_crs="EPSG:3857"
    )

    assert selected["station"].tolist() == ["inside"]
    assert selected.crs.to_epsg() == 3857
    assert geometry.contains(selected.geometry.iloc[0])
    assert coordinates.shape == (1, 2)


def test_prepare_spatial_data_rejects_missing_crs():
    boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    points = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)])

    with pytest.raises(ValueError, match="CRS"):
        prepare_spatial_data(points, boundary, target_crs="EPSG:3857")


def test_prepare_spatial_data_rejects_empty_boundary():
    boundary = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    points = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5)], crs="EPSG:4326")

    with pytest.raises(ValueError, match="Boundary data is empty"):
        prepare_spatial_data(points, boundary, target_crs="EPSG:3857")


def test_committed_vector_fixtures_exercise_point_in_boundary_pipeline():
    points = gpd.read_file(FIXTURES / "points.geojson")
    boundary = gpd.read_file(FIXTURES / "boundary.geojson")

    selected, _, coordinates = prepare_spatial_data(
        points, boundary, target_crs="EPSG:3857"
    )

    assert selected["station_id"].tolist() == ["inside"]
    assert coordinates.shape == (1, 2)
