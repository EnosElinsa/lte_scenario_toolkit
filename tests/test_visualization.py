import geopandas as gpd
import matplotlib
import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import Point, box

matplotlib.use("Agg")

from lte_scenario_toolkit.visualization import render_3d_terrain, save_preview  # noqa: E402


def test_save_preview_writes_noninteractive_png(tmp_path):
    points = gpd.GeoDataFrame(geometry=[Point(1, 1)], crs="EPSG:3857")
    selected = [
        {
            "left_x": 0,
            "bottom_y": 0,
            "center_x": 1,
            "center_y": 1,
            "pt_count": 1,
        }
    ]
    output = tmp_path / "preview.png"

    save_preview(
        points,
        box(-1, -1, 3, 3),
        selected,
        {"rect_size": 2, "boundary_layer": "fixture", "preview_png": output},
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_render_3d_terrain_writes_png_and_html(tmp_path):
    profile = {
        "driver": "GTiff",
        "height": 4,
        "width": 4,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": from_origin(0, 4, 1, 1),
    }
    selected = gpd.GeoDataFrame(
        {"elevation": [2.0]}, geometry=[Point(1.5, 1.5)], crs="EPSG:3857"
    )
    config = {
        "rect_size": 2,
        "target_crs": "EPSG:3857",
        "output_3d_png": tmp_path / "terrain.png",
        "output_3d_html": tmp_path / "terrain.html",
        "save_terrain_png": True,
        "save_terrain_eps": False,
        "save_terrain_html": True,
    }
    rectangle = {
        "left_x": 1,
        "bottom_y": 1,
        "pt_count": 1,
    }

    with MemoryFile() as memory_file:
        with memory_file.open(**profile) as dataset:
            dataset.write(np.arange(16, dtype="float32").reshape(4, 4), 1)
            outputs = render_3d_terrain(rectangle, selected, dataset, config)

    assert config["output_3d_png"] in outputs
    assert config["output_3d_html"] in outputs
    assert config["output_3d_png"].stat().st_size > 0
    assert config["output_3d_html"].stat().st_size > 0
