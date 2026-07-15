import geopandas as gpd
import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from lte_scenario_toolkit.select_sites import process_selected_rectangles


def test_process_selected_rectangles_samples_dem_and_builds_csv_rows():
    points = gpd.GeoDataFrame(
        {"cell": [7]},
        geometry=[Point(0.5, 1.5)],
        crs="EPSG:3857",
    )
    rectangle = {
        "geometry": box(0, 0, 2, 2),
        "pt_count": 1,
        "left_x": 0.0,
        "bottom_y": 0.0,
        "center_x": 1.0,
        "center_y": 1.0,
    }
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": from_origin(0, 2, 1, 1),
    }

    with MemoryFile() as memory_file:
        with memory_file.open(**profile) as dem:
            dem.write(np.array([[12, 13], [14, 15]], dtype="float32"), 1)
            frame, selected = process_selected_rectangles(
                [rectangle], points, dem, {"rect_size": 2}
            )

    assert frame["cell"].tolist() == [7]
    assert frame["elevation"].tolist() == [12.0]
    assert frame["rect_id"].tolist() == [1]
    assert selected.crs.to_epsg() == 3857
