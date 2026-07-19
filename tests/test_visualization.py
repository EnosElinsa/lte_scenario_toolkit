import builtins
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure as MatplotlibFigure
from mpl_toolkits.mplot3d.axes3d import Axes3D
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


def test_render_3d_terrain_writes_explicit_png_and_html(tmp_path):
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
    png_path = tmp_path / "terrain.png"
    html_path = tmp_path / "terrain.html"
    rectangle = {
        "left_x": 1,
        "bottom_y": 1,
        "pt_count": 1,
    }

    with MemoryFile() as memory_file:
        with memory_file.open(**profile) as dataset:
            dataset.write(np.arange(16, dtype="float32").reshape(4, 4), 1)
            outputs = render_3d_terrain(
                rectangle,
                selected,
                dataset,
                FigureSpec.from_preset("preview"),
                rectangle_size=2,
                target_crs="EPSG:3857",
                png_path=png_path,
                html_path=html_path,
            )

    assert png_path in outputs
    assert html_path in outputs
    assert png_path.stat().st_size > 0
    assert html_path.stat().st_size > 0


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


def test_publication_surface_uses_explicit_dense_sampling_and_rasterizes_mesh(
    tmp_path,
    monkeypatch,
):
    captured = []
    actual_plot_surface = Axes3D.plot_surface

    def record_plot_surface(axis, *args, **kwargs):
        captured.append(dict(kwargs))
        return actual_plot_surface(axis, *args, **kwargs)

    monkeypatch.setattr(Axes3D, "plot_surface", record_plot_surface)
    rows, columns = 64, 72
    x, y = np.meshgrid(np.arange(columns), np.arange(rows))
    arrays = {
        "x": x.astype(float),
        "y": y.astype(float),
        "z": (x + y).astype(float),
        "point_x": np.array([10.0]),
        "point_y": np.array([10.0]),
        "point_z": np.array([20.0]),
    }
    selected = gpd.GeoDataFrame(
        {"elevation": [20.0]}, geometry=[Point(10.0, 10.0)], crs="EPSG:3857"
    )

    render_3d_terrain(
        {"left_x": 0.0, "bottom_y": 0.0, "pt_count": 1},
        selected,
        None,
        FigureSpec.from_preset("publication"),
        rectangle_size=3000,
        target_crs="EPSG:3857",
        png_path=tmp_path / "publication.png",
        terrain_arrays=arrays,
    )

    assert len(captured) == 1
    assert captured[0].get("rcount", 0) > 50
    assert captured[0].get("ccount", 0) > 50
    assert captured[0].get("rasterized") is True


def test_publication_station_markers_render_above_the_terrain_surface(
    tmp_path,
    monkeypatch,
):
    captured = {}
    actual_plot_surface = Axes3D.plot_surface
    actual_scatter = Axes3D.scatter

    def record_plot_surface(axis, *args, **kwargs):
        artist = actual_plot_surface(axis, *args, **kwargs)
        captured["axis"] = axis
        captured["surface"] = artist
        return artist

    def record_scatter(axis, *args, **kwargs):
        artist = actual_scatter(axis, *args, **kwargs)
        captured["stations"] = artist
        return artist

    monkeypatch.setattr(Axes3D, "plot_surface", record_plot_surface)
    monkeypatch.setattr(Axes3D, "scatter", record_scatter)
    x, y = np.meshgrid(np.arange(8), np.arange(8))
    arrays = {
        "x": x.astype(float),
        "y": y.astype(float),
        "z": np.linspace(5.0, 55.0, 64).reshape(8, 8),
        "point_x": np.array([2.0, 6.0]),
        "point_y": np.array([2.0, 6.0]),
        "point_z": np.array([20.0, 40.0]),
    }
    selected = gpd.GeoDataFrame(
        {"elevation": [20.0, 40.0]},
        geometry=[Point(2.0, 2.0), Point(6.0, 6.0)],
        crs="EPSG:3857",
    )

    render_3d_terrain(
        {"left_x": 0.0, "bottom_y": 0.0, "pt_count": 2},
        selected,
        None,
        FigureSpec.from_preset("publication"),
        rectangle_size=3000,
        target_crs="EPSG:3857",
        png_path=tmp_path / "publication.png",
        terrain_arrays=arrays,
    )

    assert captured["axis"].computed_zorder is False
    assert captured["stations"].get_zorder() > captured["surface"].get_zorder()


def test_publication_static_layout_uses_paper_font_blank_title_and_readable_z_axis(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def inspect_figure(figure, output, *args, **kwargs):
        axis = figure.axes[0]
        captured["size"] = tuple(figure.get_size_inches())
        captured["title"] = axis.get_title()
        captured["font_families"] = (
            axis.xaxis.label.get_fontfamily(),
            axis.yaxis.label.get_fontfamily(),
            axis.zaxis.label.get_fontfamily(),
            *(label.get_fontfamily() for label in axis.get_xticklabels()),
            *(label.get_fontfamily() for label in axis.get_yticklabels()),
            *(label.get_fontfamily() for label in axis.get_zticklabels()),
        )
        aspect = axis.get_box_aspect()
        captured["relative_z_aspect"] = float(aspect[2] / aspect[0])
        captured["z_tick_labels_visible"] = tuple(
            label.get_visible() for label in axis.get_zticklabels()
        )
        captured["colorbar_label"] = figure.axes[1].get_ylabel()
        Path(output).write_bytes(b"rendered")

    monkeypatch.setattr(MatplotlibFigure, "savefig", inspect_figure)
    x, y = np.meshgrid(np.arange(8), np.arange(8))
    arrays = {
        "x": x.astype(float),
        "y": y.astype(float),
        "z": np.linspace(5.0, 55.0, 64).reshape(8, 8),
        "point_x": np.array([2.0]),
        "point_y": np.array([2.0]),
        "point_z": np.array([20.0]),
    }
    selected = gpd.GeoDataFrame(
        {"elevation": [20.0]}, geometry=[Point(2.0, 2.0)], crs="EPSG:3857"
    )

    render_3d_terrain(
        {"left_x": 0.0, "bottom_y": 0.0, "pt_count": 1},
        selected,
        None,
        FigureSpec.from_preset("publication"),
        rectangle_size=3000,
        target_crs="EPSG:3857",
        png_path=tmp_path / "publication.png",
        terrain_arrays=arrays,
    )

    assert captured["title"] == ""
    assert captured["size"][0] <= 8.0
    assert captured["relative_z_aspect"] == pytest.approx(50.0 / 3000.0)
    assert not any(captured["z_tick_labels_visible"])
    assert captured["colorbar_label"] == "Elevation (m)"
    assert all(
        "Times New Roman" in family for family in captured["font_families"]
    )


def test_publication_static_layout_discloses_non_unit_vertical_exaggeration(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def inspect_figure(figure, output, *args, **kwargs):
        axis = figure.axes[0]
        captured["notes"] = tuple(text.get_text() for text in axis.texts)
        Path(output).write_bytes(b"rendered")

    monkeypatch.setattr(MatplotlibFigure, "savefig", inspect_figure)
    x, y = np.meshgrid(np.arange(8), np.arange(8))
    arrays = {
        "x": x.astype(float),
        "y": y.astype(float),
        "z": np.linspace(5.0, 55.0, 64).reshape(8, 8),
        "point_x": np.array([2.0]),
        "point_y": np.array([2.0]),
        "point_z": np.array([20.0]),
    }
    selected = gpd.GeoDataFrame(
        {"elevation": [20.0]}, geometry=[Point(2.0, 2.0)], crs="EPSG:3857"
    )
    spec = replace(
        FigureSpec.from_preset("publication"),
        vertical_exaggeration=4.0,
    )

    render_3d_terrain(
        {"left_x": 0.0, "bottom_y": 0.0, "pt_count": 1},
        selected,
        None,
        spec,
        rectangle_size=3000,
        target_crs="EPSG:3857",
        png_path=tmp_path / "publication.png",
        terrain_arrays=arrays,
    )

    assert "Vertical exaggeration: 4x" in captured["notes"]


def test_html_uses_true_vertical_scale_colorbar_and_exaggeration_disclosure(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def inspect_html(figure, output, *args, **kwargs):
        scene = figure.layout.scene
        captured["relative_z_aspect"] = float(scene.aspectratio.z)
        captured["z_tick_labels"] = scene.zaxis.showticklabels
        captured["colorbar_label"] = figure.data[0].colorbar.title.text
        captured["notes"] = tuple(
            annotation.text for annotation in figure.layout.annotations
        )
        Path(output).write_text("rendered", encoding="utf-8")

    monkeypatch.setattr(go.Figure, "write_html", inspect_html)
    x, y = np.meshgrid(np.arange(8), np.arange(8))
    arrays = {
        "x": x.astype(float),
        "y": y.astype(float),
        "z": np.linspace(5.0, 55.0, 64).reshape(8, 8),
        "point_x": np.array([2.0]),
        "point_y": np.array([2.0]),
        "point_z": np.array([20.0]),
    }
    selected = gpd.GeoDataFrame(
        {"elevation": [20.0]}, geometry=[Point(2.0, 2.0)], crs="EPSG:3857"
    )
    spec = replace(
        FigureSpec.from_preset("publication"),
        vertical_exaggeration=4.0,
    )

    render_3d_terrain(
        {"left_x": 0.0, "bottom_y": 0.0, "pt_count": 1},
        selected,
        None,
        spec,
        rectangle_size=3000,
        target_crs="EPSG:3857",
        html_path=tmp_path / "publication.html",
        terrain_arrays=arrays,
    )

    assert captured["relative_z_aspect"] == pytest.approx(4.0 * 50.0 / 3000.0)
    assert captured["z_tick_labels"] is False
    assert captured["colorbar_label"] == "Elevation (m)"
    assert "Vertical exaggeration: 4x" in captured["notes"]


def test_publication_eps_does_not_emit_transparency_warning(
    tmp_path,
    caplog,
    capsys,
):
    x, y = np.meshgrid(np.arange(8), np.arange(8))
    arrays = {
        "x": x.astype(float),
        "y": y.astype(float),
        "z": np.linspace(5.0, 55.0, 64).reshape(8, 8),
        "point_x": np.array([2.0]),
        "point_y": np.array([2.0]),
        "point_z": np.array([20.0]),
    }
    selected = gpd.GeoDataFrame(
        {"elevation": [20.0]},
        geometry=[Point(2.0, 2.0)],
        crs="EPSG:3857",
    )

    render_3d_terrain(
        {"left_x": 0.0, "bottom_y": 0.0, "pt_count": 1},
        selected,
        None,
        FigureSpec.from_preset("publication"),
        rectangle_size=3000,
        target_crs="EPSG:3857",
        eps_path=tmp_path / "publication.eps",
        terrain_arrays=arrays,
    )

    captured = capsys.readouterr()
    messages = "\n".join(
        (captured.out, captured.err, *(record.getMessage() for record in caplog.records))
    )
    assert "does not support transparency" not in messages


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
