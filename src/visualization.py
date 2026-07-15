"""Reusable two-dimensional previews and three-dimensional terrain figures."""

from __future__ import annotations

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


def _terrain_arrays(rectangle, selected_points, dem, rectangle_size, target_crs):
    from pyproj import Transformer
    from rasterio.windows import from_bounds

    left = float(rectangle["left_x"])
    bottom = float(rectangle["bottom_y"])
    right = left + rectangle_size
    top = bottom + rectangle_size
    target_crs = str(target_crs)
    if dem.crs is None:
        raise ValueError("DEM requires a CRS for terrain rendering")

    if str(dem.crs) != target_crs:
        to_dem = Transformer.from_crs(target_crs, dem.crs, always_xy=True)
        left_dem, bottom_dem = to_dem.transform(left, bottom)
        right_dem, top_dem = to_dem.transform(right, top)
    else:
        left_dem, bottom_dem, right_dem, top_dem = left, bottom, right, top
    window = from_bounds(left_dem, bottom_dem, right_dem, top_dem, dem.transform)
    elevation = dem.read(1, window=window)
    if elevation.size == 0:
        raise ValueError("DEM window is empty for the selected rectangle")
    elevation = elevation.astype(float)
    if dem.nodata is not None:
        elevation[np.isclose(elevation, dem.nodata, equal_nan=True)] = np.nan
    if not np.isfinite(elevation).any():
        raise ValueError("DEM window contains no valid elevation values")

    transform = dem.window_transform(window)
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


def render_3d_terrain(
    rectangle: dict[str, Any],
    selected_points,
    dem,
    config: dict[str, Any],
    *,
    publication_style: bool = False,
) -> list[Path]:
    """Render static and interactive terrain outputs from a selected rectangle."""

    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    rectangle_size = float(config["rect_size"])
    arrays = _terrain_arrays(
        rectangle,
        selected_points,
        dem,
        rectangle_size,
        config.get("target_crs", "EPSG:3857"),
    )
    z_range = max(float(np.nanmax(arrays["z"]) - np.nanmin(arrays["z"])), 0.1)
    z_aspect = 5 * z_range / rectangle_size
    valid_points = np.isfinite(arrays["point_z"])
    offset = z_range * 0.02
    title = (
        {1000: "DCMOP1", 2000: "DCMOP2", 3000: "DCMOP3"}.get(
            int(rectangle_size), f"{rectangle_size:g}m"
        )
        if publication_style
        else f"Terrain | {rectangle_size:g}m x {rectangle_size:g}m | {rectangle['pt_count']} stations"
    )
    outputs: list[Path] = []

    if config.get("save_terrain_png", True):
        png_path = Path(config["output_3d_png"])
        png_path.parent.mkdir(parents=True, exist_ok=True)
        figure = plt.figure(figsize=(14, 10))
        axis = figure.add_subplot(111, projection="3d")
        row_step = max(1, arrays["z"].shape[0] // 300)
        column_step = max(1, arrays["z"].shape[1] // 300)
        surface = axis.plot_surface(
            arrays["x"][::row_step, ::column_step],
            arrays["y"][::row_step, ::column_step],
            arrays["z"][::row_step, ::column_step],
            cmap="RdYlGn_r",
            alpha=0.85,
            linewidth=0,
            antialiased=True,
        )
        axis.scatter(
            arrays["point_x"][valid_points],
            arrays["point_y"][valid_points],
            arrays["point_z"][valid_points] + offset,
            c="red",
            s=20,
            label=f"Stations ({int(valid_points.sum())})",
        )
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_zlabel("Elevation (m)")
        axis.set_title(title, fontfamily="Times New Roman" if publication_style else None)
        axis.set_box_aspect((1, 1, z_aspect))
        if valid_points.any():
            axis.legend()
        figure.colorbar(surface, ax=axis, shrink=0.5, label="Elevation (m)")
        figure.savefig(png_path, dpi=300 if publication_style else 150, bbox_inches="tight")
        outputs.append(png_path)
        if config.get("save_terrain_eps", False):
            eps_path = png_path.with_suffix(".eps")
            figure.savefig(eps_path, format="eps", dpi=300, bbox_inches="tight")
            outputs.append(eps_path)
        plt.close(figure)

    if config.get("save_terrain_html", True):
        html_path = Path(config["output_3d_html"])
        html_path.parent.mkdir(parents=True, exist_ok=True)
        surface = go.Surface(
            x=arrays["x"],
            y=arrays["y"],
            z=arrays["z"],
            colorscale="RdYlGn_r",
            opacity=0.9,
            name="Terrain",
        )
        stations = go.Scatter3d(
            x=arrays["point_x"][valid_points],
            y=arrays["point_y"][valid_points],
            z=arrays["point_z"][valid_points] + offset,
            mode="markers",
            marker={"size": 4, "color": "red", "symbol": "diamond"},
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
            },
        )
        figure.write_html(str(html_path), include_plotlyjs=True)
        outputs.append(html_path)
    return outputs
