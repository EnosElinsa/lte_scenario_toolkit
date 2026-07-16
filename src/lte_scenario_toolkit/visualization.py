"""Reusable two-dimensional previews and three-dimensional terrain figures."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np


def save_preview(points_gdf, boundary, selected_rectangles, config) -> Path:
    """Save a non-blocking 2D preview of the chosen scenarios."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rectangle_size = float(config["rect_size"])
    output = Path(config["preview_png"])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 10))
    gpd.GeoSeries([boundary], crs=points_gdf.crs).plot(
        ax=axis, facecolor="none", edgecolor="black", linewidth=1.5
    )
    points_gdf.plot(ax=axis, color="gray", markersize=0.5, alpha=0.3)
    for index, result in enumerate(selected_rectangles, start=1):
        patch = Rectangle(
            (result["left_x"], result["bottom_y"]),
            rectangle_size,
            rectangle_size,
            facecolor="green",
            edgecolor="green",
            alpha=0.35,
            linewidth=1.5,
        )
        axis.add_patch(patch)
        axis.annotate(
            f"{index}\n({result['pt_count']})",
            xy=(result["center_x"], result["center_y"]),
            ha="center",
            va="center",
            fontsize=7,
            color="darkgreen",
        )
    axis.set_title(
        f"{config.get('boundary_layer', 'boundary')} | {len(selected_rectangles)} selected "
        f"({rectangle_size:g}m x {rectangle_size:g}m)"
    )
    axis.set_aspect("equal")
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output


def interactive_select(points_gdf, boundary, results, config):
    """Display the legacy single-selection GUI and return zero or one result."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not results:
        return []
    rectangle_size = float(config["rect_size"])
    selected_index = [None]
    patches = []
    figure, axis = plt.subplots(figsize=(14, 11))
    gpd.GeoSeries([boundary], crs=points_gdf.crs).plot(
        ax=axis, facecolor="none", edgecolor="black", linewidth=1.5
    )
    points_gdf.plot(ax=axis, color="gray", markersize=0.5, alpha=0.3)
    for result in results:
        patch = Rectangle(
            (result["left_x"], result["bottom_y"]),
            rectangle_size,
            rectangle_size,
            facecolor="red",
            edgecolor="red",
            alpha=0.3,
            linewidth=1.5,
        )
        axis.add_patch(patch)
        patches.append(patch)
    axis.set_aspect("equal")
    axis.autoscale_view()

    def on_click(event):
        toolbar = figure.canvas.manager.toolbar
        if toolbar is not None and toolbar.mode != "":
            return
        if event.inaxes != axis or event.xdata is None or event.ydata is None:
            return
        for index in reversed(range(len(results))):
            result = results[index]
            inside = (
                result["left_x"] <= event.xdata <= result["left_x"] + rectangle_size
                and result["bottom_y"]
                <= event.ydata
                <= result["bottom_y"] + rectangle_size
            )
            if not inside:
                continue
            previous = selected_index[0]
            if previous is not None:
                patches[previous].set_facecolor("red")
                patches[previous].set_edgecolor("red")
                patches[previous].set_alpha(0.3)
            if previous == index:
                selected_index[0] = None
            else:
                patches[index].set_facecolor("green")
                patches[index].set_edgecolor("green")
                patches[index].set_alpha(0.45)
                selected_index[0] = index
            figure.canvas.draw_idle()
            return

    figure.canvas.mpl_connect("button_press_event", on_click)
    figure.tight_layout()
    plt.show()
    plt.close(figure)
    return [] if selected_index[0] is None else [results[selected_index[0]]]


def prepare_terrain_arrays(
    rectangle,
    selected_points,
    dem,
    rectangle_size,
    target_crs,
    *,
    max_pixels,
):
    """Read and project one DEM window with its largest dimension bounded."""

    if type(max_pixels) is not int or max_pixels <= 0:
        raise ValueError("max_pixels must be a positive integer")
    from pyproj import Transformer
    from rasterio import Affine
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    left = float(rectangle["left_x"])
    bottom = float(rectangle["bottom_y"])
    right = left + rectangle_size
    top = bottom + rectangle_size
    target_crs = str(target_crs)
    if dem.crs is None:
        raise ValueError("DEM requires a CRS for terrain rendering")

    if str(dem.crs) != target_crs:
        left_dem, bottom_dem, right_dem, top_dem = transform_bounds(
            target_crs,
            dem.crs,
            left,
            bottom,
            right,
            top,
            densify_pts=21,
        )
    else:
        left_dem, bottom_dem, right_dem, top_dem = left, bottom, right, top
    window = from_bounds(left_dem, bottom_dem, right_dem, top_dem, dem.transform)
    source_rows = max(1, int(math.ceil(abs(float(window.height)))))
    source_columns = max(1, int(math.ceil(abs(float(window.width)))))
    scale = min(1.0, max_pixels / max(source_rows, source_columns))
    output_rows = max(1, min(max_pixels, int(math.ceil(source_rows * scale))))
    output_columns = max(
        1,
        min(max_pixels, int(math.ceil(source_columns * scale))),
    )
    elevation = dem.read(
        1,
        window=window,
        out_shape=(output_rows, output_columns),
        resampling=Resampling.bilinear,
        boundless=True,
        masked=True,
    )
    if elevation.size == 0:
        raise ValueError("DEM window is empty for the selected rectangle")
    if np.ma.isMaskedArray(elevation):
        elevation = elevation.astype(float).filled(np.nan)
    else:
        elevation = elevation.astype(float)
    if dem.nodata is not None:
        elevation[np.isclose(elevation, dem.nodata, equal_nan=True)] = np.nan
    elevation[~np.isfinite(elevation)] = np.nan
    if not np.isfinite(elevation).any():
        raise ValueError("DEM window contains no valid elevation values")

    transform = dem.window_transform(window) * Affine.scale(
        float(window.width) / output_columns,
        float(window.height) / output_rows,
    )
    rows, columns = elevation.shape
    column_grid, row_grid = np.meshgrid(np.arange(columns), np.arange(rows))
    x_grid = (
        transform.c
        + (column_grid + 0.5) * transform.a
        + (row_grid + 0.5) * transform.b
    )
    y_grid = (
        transform.f
        + (column_grid + 0.5) * transform.d
        + (row_grid + 0.5) * transform.e
    )
    if str(dem.crs) != target_crs:
        from_dem = Transformer.from_crs(dem.crs, target_crs, always_xy=True)
        transformed_x, transformed_y = from_dem.transform(x_grid.ravel(), y_grid.ravel())
        x_grid = np.asarray(transformed_x).reshape(x_grid.shape)
        y_grid = np.asarray(transformed_y).reshape(y_grid.shape)

    points = selected_points.to_crs(target_crs)
    point_elevation = (
        points["elevation"].to_numpy(dtype=float)
        if "elevation" in points.columns
        else np.full(len(points), np.nan)
    )
    return {
        "x": x_grid - left,
        "y": y_grid - bottom,
        "z": elevation,
        "point_x": points.geometry.x.to_numpy() - left,
        "point_y": points.geometry.y.to_numpy() - bottom,
        "point_z": point_elevation,
    }


def _terrain_arrays(rectangle, selected_points, dem, rectangle_size, target_crs):
    """Compatibility alias for callers that used the former private helper."""

    return prepare_terrain_arrays(
        rectangle,
        selected_points,
        dem,
        rectangle_size,
        target_crs,
        max_pixels=1800,
    )


def _legacy_render_request(config, *, publication_style):
    from .figure_service import FigureSpec

    preset = "publication" if publication_style else "preview"
    spec = FigureSpec.from_preset(preset)
    updates = {
        "colormap": config.get("colormap", spec.colormap),
        "dpi": config.get("dpi", spec.dpi),
        "azimuth": config.get("azimuth", config.get("azimuth_deg", spec.azimuth)),
        "elevation_angle": config.get(
            "elevation_angle",
            config.get("elevation_deg", spec.elevation_angle),
        ),
        "vertical_exaggeration": config.get(
            "vertical_exaggeration",
            spec.vertical_exaggeration,
        ),
        "station_color": config.get("station_color", spec.station_color),
        "station_size": config.get(
            "station_size",
            config.get("station_marker_size", spec.station_size),
        ),
        "title": config.get("title", spec.title),
    }
    spec = replace(spec, **updates).validate()
    png_path = None
    eps_path = None
    html_path = None
    if config.get("save_terrain_png", True):
        png_path = Path(config["output_3d_png"])
        if config.get("save_terrain_eps", False):
            eps_path = png_path.with_suffix(".eps")
    if config.get("save_terrain_html", True):
        html_path = Path(config["output_3d_html"])
    return (
        spec,
        float(config["rect_size"]),
        str(config.get("target_crs", "EPSG:3857")),
        png_path,
        eps_path,
        html_path,
    )


def render_3d_terrain(
    rectangle: dict[str, Any],
    selected_points,
    dem,
    spec,
    *,
    rectangle_size: float | None = None,
    target_crs: str | None = None,
    png_path: str | Path | None = None,
    eps_path: str | Path | None = None,
    html_path: str | Path | None = None,
    terrain_arrays: Mapping[str, np.ndarray] | None = None,
    publication_style: bool = False,
) -> list[Path]:
    """Render requested outputs from validated style and explicit paths.

    A mapping in the fourth positional argument is accepted as a compatibility
    adapter for the pre-service selection exporter.
    """

    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    if isinstance(spec, Mapping):
        (
            spec,
            rectangle_size,
            target_crs,
            png_path,
            eps_path,
            html_path,
        ) = _legacy_render_request(spec, publication_style=publication_style)
    else:
        from .figure_service import FigureSpec

        if not isinstance(spec, FigureSpec):
            raise ValueError("spec must be a FigureSpec")
        spec.validate()
    if rectangle_size is None or not math.isfinite(float(rectangle_size)):
        raise ValueError("rectangle_size must be a finite positive number")
    rectangle_size = float(rectangle_size)
    if rectangle_size <= 0:
        raise ValueError("rectangle_size must be a finite positive number")
    if type(target_crs) is not str or not target_crs:
        raise ValueError("target_crs must be a non-empty string")
    png_path = None if png_path is None else Path(png_path)
    eps_path = None if eps_path is None else Path(eps_path)
    html_path = None if html_path is None else Path(html_path)
    if png_path is None and eps_path is None and html_path is None:
        raise ValueError("At least one explicit terrain output path is required")
    arrays = (
        dict(terrain_arrays)
        if terrain_arrays is not None
        else prepare_terrain_arrays(
            rectangle,
            selected_points,
            dem,
            rectangle_size,
            target_crs,
            max_pixels=spec.max_pixels,
        )
    )
    z_range = max(float(np.nanmax(arrays["z"]) - np.nanmin(arrays["z"])), 0.1)
    z_aspect = max(
        spec.vertical_exaggeration * z_range / rectangle_size,
        0.01,
    )
    valid_points = np.isfinite(arrays["point_z"])
    offset = z_range * 0.02
    title = spec.resolved_title(
        rectangle_size,
        int(rectangle["pt_count"]),
    )
    outputs: list[Path] = []

    if png_path is not None or eps_path is not None:
        static_parent = png_path.parent if png_path is not None else eps_path.parent
        static_parent.mkdir(parents=True, exist_ok=True)
        figure = plt.figure(figsize=(14, 10))
        try:
            axis = figure.add_subplot(111, projection="3d")
            surface = axis.plot_surface(
                arrays["x"],
                arrays["y"],
                arrays["z"],
                cmap=spec.colormap,
                alpha=0.85,
                linewidth=0,
                antialiased=True,
            )
            axis.scatter(
                arrays["point_x"][valid_points],
                arrays["point_y"][valid_points],
                arrays["point_z"][valid_points] + offset,
                c=spec.station_color,
                s=spec.station_size,
                label=f"Stations ({int(valid_points.sum())})",
            )
            axis.set_xlabel("X (m)")
            axis.set_ylabel("Y (m)")
            axis.set_zlabel("Elevation (m)")
            axis.set_title(
                title,
                fontfamily="Times New Roman" if spec.preset == "publication" else None,
            )
            axis.view_init(elev=spec.elevation_angle, azim=spec.azimuth)
            axis.set_box_aspect((1, 1, z_aspect))
            if valid_points.any():
                axis.legend()
            figure.colorbar(surface, ax=axis, shrink=0.5, label="Elevation (m)")
            if png_path is not None:
                png_path.parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(png_path, dpi=spec.dpi, bbox_inches="tight")
                outputs.append(png_path)
            if eps_path is not None:
                eps_path.parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(
                    eps_path,
                    format="eps",
                    dpi=spec.dpi,
                    bbox_inches="tight",
                )
                outputs.append(eps_path)
        finally:
            plt.close(figure)

    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        surface = go.Surface(
            x=arrays["x"],
            y=arrays["y"],
            z=arrays["z"],
            colorscale=_plotly_colorscale(spec.colormap),
            opacity=0.9,
            name="Terrain",
        )
        stations = go.Scatter3d(
            x=arrays["point_x"][valid_points],
            y=arrays["point_y"][valid_points],
            z=arrays["point_z"][valid_points] + offset,
            mode="markers",
            marker={
                "size": max(1.0, spec.station_size / 5.0),
                "color": spec.station_color,
                "symbol": "diamond",
            },
            name=f"Stations ({int(valid_points.sum())})",
        )
        figure = go.Figure(data=[surface, stations])
        figure.update_layout(
            title=title,
            scene={
                "xaxis_title": "X (m)",
                "yaxis_title": "Y (m)",
                "zaxis_title": "Elevation (m)",
                "aspectmode": "manual",
                "aspectratio": {"x": 1, "y": 1, "z": z_aspect},
                "camera": _plotly_camera(
                    spec.azimuth,
                    spec.elevation_angle,
                ),
            },
        )
        figure.write_html(str(html_path), include_plotlyjs=True)
        outputs.append(html_path)
    return outputs


def _plotly_colorscale(colormap: str):
    """Convert a validated Matplotlib colormap to a Plotly colorscale."""

    import matplotlib
    from matplotlib.colors import to_hex

    cmap = matplotlib.colormaps.get_cmap(colormap)
    return [[index / 10, to_hex(cmap(index / 10))] for index in range(11)]


def _plotly_camera(azimuth: float, elevation_angle: float) -> dict[str, Any]:
    """Map Matplotlib-style view angles to a deterministic Plotly camera."""

    radius = 1.5
    azimuth_radians = math.radians(float(azimuth))
    elevation_radians = math.radians(float(elevation_angle))
    horizontal = radius * math.cos(elevation_radians)

    def stable(value: float) -> float:
        return 0.0 if abs(value) < 1e-12 else value

    return {
        "eye": {
            "x": stable(horizontal * math.cos(azimuth_radians)),
            "y": stable(horizontal * math.sin(azimuth_radians)),
            "z": stable(radius * math.sin(elevation_radians)),
        },
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
    }
