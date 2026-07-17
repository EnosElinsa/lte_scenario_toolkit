import builtins
from dataclasses import replace

import geopandas as gpd
import matplotlib
import numpy as np
import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure as MatplotlibFigure
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from lte_scenario_toolkit.figure_service import FigureSpec  # noqa: E402
from lte_scenario_toolkit.visualization import (  # noqa: E402
    prepare_terrain_arrays,
    render_3d_terrain,
    save_preview,
)


def test_save_preview_writes_noninteractive_png_without_pyplot_or_backend_change(
    tmp_path,
    monkeypatch,
):
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
    backend = matplotlib.get_backend()
    actual_import = builtins.__import__

    def reject_pyplot(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib.pyplot" or (
            name == "matplotlib" and fromlist and "pyplot" in fromlist
        ):
            raise AssertionError("file rendering must not import pyplot")
        return actual_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_pyplot)

    save_preview(
        points,
        box(-1, -1, 3, 3),
        selected,
        {"rect_size": 2, "boundary_layer": "fixture", "preview_png": output},
    )

    assert output.is_file()
    assert output.stat().st_size > 0
    assert matplotlib.get_backend() == backend


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


def test_prepare_terrain_arrays_bounds_the_raster_read_shape():
    profile = {
        "driver": "GTiff",
        "height": 80,
        "width": 100,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": from_origin(0, 80, 1, 1),
    }
    selected = gpd.GeoDataFrame(
        {"elevation": [2.0]}, geometry=[Point(10.5, 10.5)], crs="EPSG:3857"
    )

    class RecordingRaster:
        def __init__(self, raster):
            self.raster = raster
            self.out_shape = None

        def __getattr__(self, name):
            return getattr(self.raster, name)

        def read(self, *args, **kwargs):
            self.out_shape = kwargs.get("out_shape")
            return self.raster.read(*args, **kwargs)

    with MemoryFile() as memory_file:
        with memory_file.open(**profile) as dataset:
            dataset.write(np.arange(8000, dtype="float32").reshape(80, 100), 1)
            recording = RecordingRaster(dataset)
            arrays = prepare_terrain_arrays(
                {"left_x": 0, "bottom_y": 0},
                selected,
                recording,
                80,
                "EPSG:3857",
                max_pixels=20,
            )

    assert recording.out_shape is not None
    assert max(recording.out_shape) <= 20
    assert max(arrays["z"].shape) <= 20


def test_prepare_terrain_arrays_keeps_boundless_window_coordinates_and_masks_edges():
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
        {"elevation": [2.0]}, geometry=[Point(0.5, 1.5)], crs="EPSG:3857"
    )

    with MemoryFile() as memory_file:
        with memory_file.open(**profile) as dataset:
            dataset.write(np.arange(16, dtype="float32").reshape(4, 4), 1)
            arrays = prepare_terrain_arrays(
                {"left_x": -1, "bottom_y": 1},
                selected,
                dataset,
                2,
                "EPSG:3857",
                max_pixels=2,
            )

    assert arrays["z"].shape == (2, 2)
    assert np.isnan(arrays["z"][:, 0]).all()
    assert arrays["x"][0].tolist() == pytest.approx([0.5, 1.5])


def test_prepare_terrain_arrays_normalises_nan_and_infinity():
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": from_origin(0, 2, 1, 1),
    }
    selected = gpd.GeoDataFrame(
        {"elevation": [2.0]}, geometry=[Point(0.5, 0.5)], crs="EPSG:3857"
    )

    with MemoryFile() as memory_file:
        with memory_file.open(**profile) as dataset:
            dataset.write(
                np.array([[1.0, np.nan], [np.inf, 4.0]], dtype="float32"),
                1,
            )
            arrays = prepare_terrain_arrays(
                {"left_x": 0, "bottom_y": 0},
                selected,
                dataset,
                2,
                "EPSG:3857",
                max_pixels=2,
            )

    assert np.isfinite(arrays["z"][~np.isnan(arrays["z"])]).all()
    assert np.isnan(arrays["z"]).sum() == 2


def test_render_3d_terrain_accepts_spec_without_pyplot_or_backend_change(
    tmp_path,
    monkeypatch,
):
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
    eps_path = tmp_path / "terrain.eps"
    backend = matplotlib.get_backend()
    actual_import = builtins.__import__

    def reject_pyplot(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib.pyplot" or (
            name == "matplotlib" and fromlist and "pyplot" in fromlist
        ):
            raise AssertionError("file rendering must not import pyplot")
        return actual_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_pyplot)

    with MemoryFile() as memory_file:
        with memory_file.open(**profile) as dataset:
            dataset.write(np.arange(16, dtype="float32").reshape(4, 4), 1)
            outputs = render_3d_terrain(
                {"left_x": 1, "bottom_y": 1, "pt_count": 1},
                selected,
                dataset,
                FigureSpec.from_preset("publication"),
                rectangle_size=2,
                target_crs="EPSG:3857",
                eps_path=eps_path,
            )

    assert outputs == [eps_path]
    assert eps_path.stat().st_size > 0
    assert not (tmp_path / "terrain.png").exists()
    assert matplotlib.get_backend() == backend


def test_render_3d_terrain_closes_static_figure_when_save_fails(
    tmp_path,
    monkeypatch,
):
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
    cleared = []
    actual_clear = MatplotlibFigure.clear
    monkeypatch.setattr(
        MatplotlibFigure,
        "savefig",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("save failed")),
    )

    def record_clear(figure, *args, **kwargs):
        cleared.append(figure)
        return actual_clear(figure, *args, **kwargs)

    monkeypatch.setattr(MatplotlibFigure, "clear", record_clear)

    with MemoryFile() as memory_file:
        with memory_file.open(**profile) as dataset:
            dataset.write(np.arange(16, dtype="float32").reshape(4, 4), 1)
            with pytest.raises(RuntimeError, match="save failed"):
                render_3d_terrain(
                    {"left_x": 1, "bottom_y": 1, "pt_count": 1},
                    selected,
                    dataset,
                    FigureSpec.from_preset("preview"),
                    rectangle_size=2,
                    target_crs="EPSG:3857",
                    png_path=tmp_path / "terrain.png",
                )

    assert cleared
    assert len({id(figure) for figure in cleared}) == 1


def test_html_camera_uses_spec_azimuth_and_elevation(tmp_path, monkeypatch):
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
    captured = {}
    actual_write_html = go.Figure.write_html

    def capture_layout(figure, *args, **kwargs):
        captured.update(figure.layout.scene.camera.to_plotly_json())
        return actual_write_html(figure, *args, **kwargs)

    monkeypatch.setattr(go.Figure, "write_html", capture_layout)
    html_path = tmp_path / "terrain.html"
    spec = replace(
        FigureSpec.from_preset("preview"),
        azimuth=0.0,
        elevation_angle=0.0,
    )

    with MemoryFile() as memory_file:
        with memory_file.open(**profile) as dataset:
            dataset.write(np.arange(16, dtype="float32").reshape(4, 4), 1)
            render_3d_terrain(
                {"left_x": 1, "bottom_y": 1, "pt_count": 1},
                selected,
                dataset,
                spec,
                rectangle_size=2,
                target_crs="EPSG:3857",
                html_path=html_path,
            )

    assert captured["eye"] == pytest.approx({"x": 1.5, "y": 0.0, "z": 0.0})
    html = html_path.read_text(encoding="utf-8")
    assert '"camera"' in html
    assert '"eye":{"x":1.5,"y":0.0,"z":0.0}' in html
