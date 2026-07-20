"""Reusable two-dimensional previews and three-dimensional terrain figures."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_PUBLICATION_FONT = "Times New Roman"
_STATIC_SURFACE_SAMPLE_LIMITS = {
    "preview": 160,
    "publication": 400,
}


def _surface_sample_counts(z: np.ndarray, preset: str) -> tuple[int, int]:
    """Choose an explicit mesh density instead of Matplotlib's 50x50 default."""

    rows, columns = z.shape
    limit = _STATIC_SURFACE_SAMPLE_LIMITS[preset]
    return min(rows, limit), min(columns, limit)


def _vertical_z_aspect(
    z_range: float,
    rectangle_size: float,
    vertical_exaggeration: float,
) -> float:
    """Preserve the physical vertical scale and apply only the requested factor."""

    return z_range / rectangle_size * vertical_exaggeration


def _vertical_exaggeration_note(vertical_exaggeration: float) -> str | None:
    """Disclose any non-unit vertical scale used by the exported figure."""

    if math.isclose(vertical_exaggeration, 1.0):
        return None
    return f"Vertical exaggeration: {vertical_exaggeration:g}x"


def _geometry_line_segments(geometry) -> list[np.ndarray]:
    """Return two-dimensional line segments without invoking GeoPandas plotting."""

    if geometry is None or geometry.is_empty:
        return []
    geometry_type = geometry.geom_type
    if geometry_type == "Polygon":
        return [
            np.asarray(ring.coords, dtype=float)[:, :2]
            for ring in (geometry.exterior, *geometry.interiors)
        ]
    if geometry_type in {
        "MultiPolygon",
        "MultiLineString",
        "GeometryCollection",
    }:
        return [
            segment
            for child in geometry.geoms
            for segment in _geometry_line_segments(child)
        ]
    if geometry_type in {"LineString", "LinearRing"}:
        return [np.asarray(geometry.coords, dtype=float)[:, :2]]
    raise ValueError(f"Unsupported boundary geometry: {geometry_type}")


def save_preview(points_gdf, boundary, selected_rectangles, config) -> Path:
    """Save a non-blocking 2D preview of the chosen scenarios."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    rectangle_size = float(config["rect_size"])
    output = Path(config["preview_png"])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(12, 10))
    FigureCanvasAgg(figure)
    try:
        axis = figure.add_subplot(111)
        boundary_lines = LineCollection(
            _geometry_line_segments(boundary),
            colors="black",
            linewidths=1.5,
        )
        axis.add_collection(boundary_lines)
        point_geometry = points_gdf.geometry
        axis.scatter(
            point_geometry.x.to_numpy(),
            point_geometry.y.to_numpy(),
            color="gray",
            s=0.5,
            alpha=0.3,
        )
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
        axis.autoscale_view()
        figure.tight_layout()
        figure.savefig(output, dpi=150, bbox_inches="tight")
    finally:
        figure.clear()
    return output


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
) -> list[Path]:
    """Render requested outputs from a validated style and explicit paths."""

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
    z_aspect = _vertical_z_aspect(
        z_range,
        rectangle_size,
        spec.vertical_exaggeration,
    )
    vertical_exaggeration_note = _vertical_exaggeration_note(
        spec.vertical_exaggeration
    )
    valid_points = np.isfinite(arrays["point_z"])
    offset = z_range * 0.02
    title = spec.resolved_title(
        rectangle_size,
        int(rectangle["pt_count"]),
    )
    outputs: list[Path] = []

    if png_path is not None or eps_path is not None:
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.ticker import MaxNLocator

        static_parent = png_path.parent if png_path is not None else eps_path.parent
        static_parent.mkdir(parents=True, exist_ok=True)
        publication = spec.preset == "publication"
        rc_parameters = {
            "font.family": _PUBLICATION_FONT,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        } if publication else {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
        with matplotlib.rc_context(rc_parameters):
            figure = Figure(figsize=(8.0, 6.0), facecolor="white")
            FigureCanvasAgg(figure)
            try:
                axis = figure.add_subplot(
                    111,
                    projection="3d",
                    computed_zorder=False,
                )
                row_count, column_count = _surface_sample_counts(
                    arrays["z"],
                    spec.preset,
                )
                surface = axis.plot_surface(
                    arrays["x"],
                    arrays["y"],
                    arrays["z"],
                    cmap=spec.colormap,
                    alpha=1.0,
                    linewidth=0,
                    antialiased=True,
                    rcount=row_count,
                    ccount=column_count,
                    rasterized=True,
                    zorder=1,
                )
                axis.scatter(
                    arrays["point_x"][valid_points],
                    arrays["point_y"][valid_points],
                    arrays["point_z"][valid_points] + offset,
                    c=spec.station_color,
                    s=spec.station_size,
                    edgecolors="white",
                    linewidths=0.45,
                    depthshade=False,
                    alpha=1.0,
                    label=f"Stations ({int(valid_points.sum())})",
                    zorder=10,
                )
                axis.set_xlabel("X (m)", labelpad=8)
                axis.set_ylabel("Y (m)", labelpad=8)
                axis.set_zlabel("Z (m)", labelpad=7)
                if title is not None:
                    axis.set_title(title, pad=12)
                axis.view_init(elev=spec.elevation_angle, azim=spec.azimuth)
                axis.set_proj_type("ortho")
                axis.set_box_aspect((1, 1, z_aspect), zoom=1.05)
                axis.set_xlim(0.0, rectangle_size)
                axis.set_ylim(0.0, rectangle_size)
                axis.zaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
                axis.tick_params(axis="x", pad=1)
                axis.tick_params(axis="y", pad=1)
                axis.tick_params(axis="z", pad=2)
                for tick_label in axis.get_zticklabels():
                    tick_label.set_visible(False)
                if vertical_exaggeration_note is not None:
                    axis.text2D(
                        0.02,
                        0.98,
                        vertical_exaggeration_note,
                        transform=axis.transAxes,
                        ha="left",
                        va="top",
                        fontsize=7,
                        color="0.25",
                    )
                for pane in (
                    axis.xaxis.pane,
                    axis.yaxis.pane,
                    axis.zaxis.pane,
                ):
                    pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
                    pane.set_edgecolor((0.75, 0.75, 0.75, 1.0))
                    pane.set_alpha(1.0)
                axis.grid(
                    True,
                    linewidth=0.35,
                    color=(0.68, 0.68, 0.68, 1.0),
                    alpha=1.0,
                )
                if valid_points.any():
                    axis.legend(
                        loc="upper right",
                        bbox_to_anchor=(0.98, 0.88),
                        frameon=False,
                        borderaxespad=0.4,
                        handletextpad=0.5,
                    )
                colorbar = figure.colorbar(
                    surface,
                    ax=axis,
                    shrink=0.72,
                    aspect=28,
                    fraction=0.04,
                    pad=0.08,
                )
                colorbar.set_label("Elevation (m)", labelpad=8)
                colorbar.ax.tick_params(labelsize=8, pad=2)
                figure.subplots_adjust(
                    left=0.02,
                    right=0.88,
                    bottom=0.05,
                    top=0.96,
                )
                save_options = {
                    "dpi": spec.dpi,
                    "bbox_inches": "tight",
                    "pad_inches": 0.08,
                    "facecolor": "white",
                }
                if png_path is not None:
                    png_path.parent.mkdir(parents=True, exist_ok=True)
                    figure.savefig(png_path, **save_options)
                    outputs.append(png_path)
                if eps_path is not None:
                    eps_path.parent.mkdir(parents=True, exist_ok=True)
                    figure.savefig(
                        eps_path,
                        format="eps",
                        **save_options,
                    )
                    outputs.append(eps_path)
            finally:
                figure.clear()

    if html_path is not None:
        import plotly.graph_objects as go

        html_path.parent.mkdir(parents=True, exist_ok=True)
        surface = go.Surface(
            x=arrays["x"],
            y=arrays["y"],
            z=arrays["z"],
            colorscale=_plotly_colorscale(spec.colormap),
            colorbar={"title": "Elevation (m)"},
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
            font={
                "family": (
                    "Times New Roman"
                    if spec.preset == "publication"
                    else "Arial"
                ),
                "size": 13,
            },
            template="plotly_white",
            margin={"l": 20, "r": 20, "t": 48 if title else 20, "b": 20},
            annotations=(
                []
                if vertical_exaggeration_note is None
                else [
                    {
                        "text": vertical_exaggeration_note,
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.01,
                        "y": 0.99,
                        "showarrow": False,
                        "xanchor": "left",
                        "yanchor": "top",
                        "font": {"size": 11, "color": "#444444"},
                    }
                ]
            ),
            scene={
                "xaxis_title": "X (m)",
                "yaxis_title": "Y (m)",
                "zaxis": {
                    "title": "Z (m)",
                    "nticks": 5,
                    "showticklabels": False,
                },
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
