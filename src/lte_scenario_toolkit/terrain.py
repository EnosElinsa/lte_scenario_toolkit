"""DEM sampling helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def validate_dem_path(path: str | Path) -> Path:
    """Return an existing DEM file or raise an actionable error."""

    dem_path = Path(path)
    if not dem_path.is_file():
        raise FileNotFoundError(f"DEM file does not exist: {dem_path}")
    return dem_path


def require_valid_elevations(elevations: np.ndarray) -> np.ndarray:
    """Reject a scenario whose points have no usable DEM samples."""

    values = np.asarray(elevations, dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        raise ValueError("No valid elevation samples were found inside the DEM")
    return values


def extract_elevation(points_gdf, dem) -> np.ndarray:
    """Sample band 1 at point locations and return NaN for nodata/outside pixels."""

    if len(points_gdf) == 0:
        return np.array([], dtype=float)
    if points_gdf.crs is None:
        raise ValueError("Point data requires a CRS before DEM sampling")
    if dem.crs is None:
        raise ValueError("DEM requires a CRS before elevation sampling")

    points = points_gdf.to_crs(dem.crs) if points_gdf.crs != dem.crs else points_gdf
    band = dem.read(1)
    nodata = dem.nodata
    elevations: list[float] = []

    for x, y in zip(points.geometry.x, points.geometry.y, strict=True):
        try:
            row, column = dem.index(float(x), float(y))
        except (TypeError, ValueError, OverflowError):
            elevations.append(np.nan)
            continue
        if not (0 <= row < dem.height and 0 <= column < dem.width):
            elevations.append(np.nan)
            continue
        value = band[row, column]
        if nodata is not None and np.isclose(value, nodata, equal_nan=True):
            elevations.append(np.nan)
        else:
            elevations.append(float(value))
    return np.asarray(elevations, dtype=float)
