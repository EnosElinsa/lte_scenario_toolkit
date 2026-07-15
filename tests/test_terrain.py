import geopandas as gpd
import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import Point

from src.terrain import extract_elevation, require_valid_elevations, validate_dem_path


def test_extract_elevation_handles_values_nodata_and_outside_points():
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": from_origin(0, 2, 1, 1),
        "nodata": -9999.0,
    }
    points = gpd.GeoDataFrame(
        geometry=[Point(0.5, 1.5), Point(1.5, 0.5), Point(3, 3)],
        crs="EPSG:3857",
    )

    with MemoryFile() as memory_file:
        with memory_file.open(**profile) as dataset:
            dataset.write(np.array([[10, 20], [30, -9999]], dtype="float32"), 1)
            elevations = extract_elevation(points, dataset)

    assert elevations[0] == 10
    assert np.isnan(elevations[1])
    assert np.isnan(elevations[2])


def test_extract_elevation_returns_empty_array_for_no_points():
    points = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

    assert extract_elevation(points, None).size == 0


def test_require_valid_elevations_rejects_all_nodata():
    with pytest.raises(ValueError, match="valid elevation"):
        require_valid_elevations(np.array([np.nan, np.nan]))


def test_validate_dem_path_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="DEM"):
        validate_dem_path(tmp_path / "missing.tif")
