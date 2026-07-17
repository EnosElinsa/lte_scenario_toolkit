"""Offline, display-only map assets derived from registered local data."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

MAP_STYLE_VERSION = "map-style-v1"
DEFAULT_MAX_DIMENSION = 1024
MAX_MAP_DIMENSION = 4096
STATION_DISPLAY_FIELDS = (
    "cell",
    "longitude",
    "latitude",
    "range",
    "samples",
    "created",
    "updated",
)


class MapStyle(str, Enum):
    """Supported offline DEM display styles."""

    ELEVATION = "elevation"
    HILLSHADE = "hillshade"
    COMBINED = "combined"


@dataclass(frozen=True)
class MapAsset:
    """One cached raster overlay and its requested display extent."""

    path: Path
    bounds: tuple[float, float, float, float]
    bounds_crs: str
    style: MapStyle


_LOCKS_GUARD = threading.Lock()
_KEY_LOCKS: dict[Path, threading.Lock] = {}


def _key_lock(path: Path) -> threading.Lock:
    with _LOCKS_GUARD:
        return _KEY_LOCKS.setdefault(path, threading.Lock())


def _is_redirected_path(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    try:
        attributes = path.lstat().st_file_attributes
    except FileNotFoundError:
        return False
    except AttributeError:
        return False
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_point)


def _is_local_regular_file(path: Path, *, label: str) -> Path:
    if _is_redirected_path(path):
        raise ValueError(f"{label} path must not be redirected: {path}")
    try:
        status = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    return path.resolve(strict=True)


def _normalise_crs(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-empty CRS string")
    from pyproj import CRS

    try:
        return CRS.from_user_input(value.strip()).to_string()
    except Exception as exc:
        raise ValueError(f"{label} is not a valid CRS: {value!r}") from exc


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _normalise_bounds(value: Any) -> tuple[float, float, float, float]:
    if type(value) is not tuple or len(value) != 4:
        raise ValueError("bounds must be a four-item tuple")
    left, bottom, right, top = (
        _finite_number(item, label="bounds value") for item in value
    )
    if left >= right or bottom >= top:
        raise ValueError("bounds must satisfy left < right and bottom < top")
    return left, bottom, right, top


def _normalise_color_limits(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("color_limits must be None or a two-item tuple")
    lower = _finite_number(value[0], label="color limit")
    upper = _finite_number(value[1], label="color limit")
    if lower >= upper:
        raise ValueError("color_limits must satisfy lower < upper")
    return lower, upper


def _output_shape(window: Any, max_dimension: int) -> tuple[int, int]:
    source_rows = max(1, int(math.ceil(abs(float(window.height)))))
    source_columns = max(1, int(math.ceil(abs(float(window.width)))))
    scale = min(1.0, max_dimension / max(source_rows, source_columns))
    rows = max(1, min(max_dimension, int(math.ceil(source_rows * scale))))
    columns = max(1, min(max_dimension, int(math.ceil(source_columns * scale))))
    return rows, columns


def _cache_key(payload: dict[str, Any]) -> str:
    serialised = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    try:
        import pandas as pd

        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _force_finite_2d(geometry: Any, *, label: str) -> Any:
    from shapely import force_2d, get_coordinates

    display = force_2d(geometry)
    coordinates = get_coordinates(display, include_z=False)
    if coordinates.size == 0 or not np.isfinite(coordinates).all():
        raise ValueError(f"{label} must contain finite XY coordinates")
    return display


def _valid_cached_png(path: Path, *, size: tuple[int, int]) -> bool:
    if not os.path.lexists(path) or _is_redirected_path(path):
        return False
    try:
        status = path.stat()
    except OSError:
        return False
    if not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
        return False
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
            return (
                image.format == "PNG"
                and image.mode == "RGBA"
                and image.size == size
            )
    except (OSError, ValueError):
        return False


def _remove_quarantined_leaf(path: Path) -> None:
    if not os.path.lexists(path):
        return
    if path.is_symlink():
        path.unlink()
        return
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        path.rmdir()
        return
    if path.is_file():
        path.unlink()
        return
    try:
        path.rmdir()
    except OSError:
        # A non-empty unexpected directory is retained for manual inspection.
        return


def _rgba_image(
    elevation: np.ndarray,
    valid: np.ndarray,
    *,
    style: MapStyle,
    color_limits: tuple[float, float] | None,
    light_azimuth: float,
    light_altitude: float,
    dx: float,
    dy: float,
):
    from matplotlib import colormaps
    from matplotlib.colors import LightSource, Normalize
    from PIL import Image

    rgba = np.zeros((*elevation.shape, 4), dtype=np.uint8)
    if not bool(valid.any()):
        return Image.fromarray(rgba, mode="RGBA")

    values = elevation[valid]
    if color_limits is None:
        lower = float(values.min())
        upper = float(values.max())
        if lower == upper:
            lower -= 0.5
            upper += 0.5
    else:
        lower, upper = color_limits
    normalised = Normalize(vmin=lower, vmax=upper, clip=True)(elevation)
    elevation_rgb = colormaps["terrain"](normalised)[..., :3]

    fill_value = float(values.mean())
    filled = np.where(valid, elevation, fill_value)
    if min(elevation.shape) < 2 or float(values.min()) == float(values.max()):
        hillshade = np.full(elevation.shape, 0.5, dtype=float)
    else:
        light = LightSource(azdeg=light_azimuth, altdeg=light_altitude)
        hillshade = light.hillshade(filled, dx=dx, dy=dy)
        hillshade = np.clip(np.nan_to_num(hillshade, nan=0.5), 0.0, 1.0)

    if style is MapStyle.ELEVATION:
        rgb = elevation_rgb
    elif style is MapStyle.HILLSHADE:
        rgb = np.repeat(hillshade[..., None], 3, axis=2)
    else:
        rgb = elevation_rgb * (0.4 + 0.6 * hillshade[..., None])
    rgba[..., :3] = np.rint(np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


class MapAssetService:
    """Create bounded local overlays without exposing source GeoTIFFs."""

    def __init__(self, repo_root: str | Path) -> None:
        if not isinstance(repo_root, (str, Path)):
            raise ValueError("repo_root must be a local path")
        raw_root = Path(repo_root)
        if _is_redirected_path(raw_root):
            raise ValueError(f"Repository root must not be redirected: {raw_root}")
        try:
            status = raw_root.stat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Repository root does not exist: {raw_root}"
            ) from exc
        if not stat.S_ISDIR(status.st_mode):
            raise ValueError(f"Repository root is not a directory: {raw_root}")
        self.repo_root = raw_root.resolve(strict=True)
        self.cache_root = self.repo_root / ".lte-data" / "cache" / "maps"

    def _ensure_cache_root(self) -> None:
        current = self.repo_root
        for name in (".lte-data", "cache", "maps"):
            current = current / name
            if not os.path.lexists(current):
                try:
                    current.mkdir()
                except FileExistsError:
                    pass
            if _is_redirected_path(current):
                raise ValueError(
                    f"Map cache directory must not be redirected: {current}"
                )
            try:
                status = current.stat()
            except OSError as exc:
                raise ValueError(
                    f"Map cache directory cannot be inspected: {current}"
                ) from exc
            if not stat.S_ISDIR(status.st_mode):
                raise ValueError(f"Map cache path is not a directory: {current}")
        if self.cache_root.resolve(strict=True).parent.parent.parent != self.repo_root:
            raise ValueError("Map cache path escapes the repository")

    def dem_overlay(
        self,
        dem_path: str | Path,
        *,
        fingerprint: str,
        bounds: tuple[float, float, float, float],
        bounds_crs: str,
        style: MapStyle,
        max_dimension: int = DEFAULT_MAX_DIMENSION,
        color_limits: tuple[float, float] | None = None,
        light_azimuth: float = 315.0,
        light_altitude: float = 45.0,
    ) -> MapAsset:
        """Return a cached, bounded RGBA overlay for one requested extent."""

        if not isinstance(dem_path, (str, Path)) or "://" in str(dem_path):
            raise ValueError("dem_path must be a local filesystem path")
        resolved_dem = _is_local_regular_file(Path(dem_path), label="DEM")
        if type(fingerprint) is not str or not fingerprint.strip():
            raise ValueError("fingerprint must be a non-empty string")
        requested_bounds = _normalise_bounds(bounds)
        requested_crs = _normalise_crs(bounds_crs, label="bounds_crs")
        if not isinstance(style, MapStyle):
            raise ValueError("style must be a MapStyle")
        if (
            type(max_dimension) is not int
            or max_dimension <= 0
            or max_dimension > MAX_MAP_DIMENSION
        ):
            raise ValueError(
                f"max_dimension must be an integer from 1 to {MAX_MAP_DIMENSION}"
            )
        limits = _normalise_color_limits(color_limits)
        azimuth = _finite_number(light_azimuth, label="light_azimuth")
        altitude = _finite_number(light_altitude, label="light_altitude")
        if not 0.0 <= azimuth < 360.0:
            raise ValueError("light_azimuth must be in [0, 360)")
        if not 0.0 < altitude <= 90.0:
            raise ValueError("light_altitude must be in (0, 90]")

        import rasterio
        from rasterio import Affine
        from rasterio.enums import Resampling
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds

        with rasterio.open(resolved_dem) as dataset:
            if dataset.count < 1:
                raise ValueError("DEM must contain at least one raster band")
            if dataset.crs is None:
                raise ValueError("DEM requires a CRS")
            dem_crs = _normalise_crs(dataset.crs.to_string(), label="DEM CRS")
            if dem_crs != "EPSG:3857":
                raise ValueError(
                    "Offline map overlays require a DEM in EPSG:3857; "
                    "reproject the registered DEM before rendering"
                )
            transform = dataset.transform
            if not (
                math.isclose(float(transform.b), 0.0, abs_tol=1e-12)
                and math.isclose(float(transform.d), 0.0, abs_tol=1e-12)
                and float(transform.a) > 0.0
                and float(transform.e) < 0.0
            ):
                raise ValueError(
                    "Offline map overlays require a north-up, unrotated DEM"
                )
            transformed_bounds = (
                requested_bounds
                if requested_crs == dem_crs
                else tuple(
                    float(item)
                    for item in transform_bounds(
                        requested_crs,
                        dem_crs,
                        *requested_bounds,
                        densify_pts=21,
                    )
                )
            )
            if not all(math.isfinite(item) for item in transformed_bounds):
                raise ValueError("Requested bounds could not be transformed to DEM CRS")
            window = from_bounds(*transformed_bounds, transform=dataset.transform)
            rows, columns = _output_shape(window, max_dimension)
            key = _cache_key(
                {
                    "style_version": MAP_STYLE_VERSION,
                    "dem_fingerprint": fingerprint.strip(),
                    "render_crs": dem_crs,
                    "transformed_bounds": transformed_bounds,
                    "output_shape": [rows, columns],
                    "style": style.value,
                    "color_limits": limits,
                    "light_azimuth": azimuth,
                    "light_altitude": altitude,
                }
            )
            self._ensure_cache_root()
            output = self.cache_root / f"{key}.png"
            size = (columns, rows)
            with _key_lock(output):
                if _valid_cached_png(output, size=size):
                    return MapAsset(output, requested_bounds, requested_crs, style)
                elevation = dataset.read(
                    1,
                    window=window,
                    out_shape=(rows, columns),
                    resampling=Resampling.bilinear,
                    boundless=True,
                    masked=True,
                )
                values = np.asarray(np.ma.getdata(elevation), dtype=float)
                valid = ~np.ma.getmaskarray(elevation)
                valid &= np.isfinite(values)
                if dataset.nodata is not None:
                    valid &= ~np.isclose(values, dataset.nodata, equal_nan=True)
                if not bool(valid.any()):
                    raise ValueError(
                        "DEM window contains no valid elevation values"
                    )
                rendered_transform = dataset.window_transform(window) * Affine.scale(
                    float(window.width) / columns,
                    float(window.height) / rows,
                )
                dx = max(math.hypot(rendered_transform.a, rendered_transform.d), 1e-12)
                dy = max(math.hypot(rendered_transform.b, rendered_transform.e), 1e-12)
                image = _rgba_image(
                    values,
                    valid,
                    style=style,
                    color_limits=limits,
                    light_azimuth=azimuth,
                    light_altitude=altitude,
                    dx=dx,
                    dy=dy,
                )
                temporary = output.with_name(
                    f".{output.stem}.{uuid.uuid4().hex}.tmp"
                )
                quarantined: Path | None = None
                published = False
                try:
                    image.save(temporary, format="PNG")
                    if not _valid_cached_png(temporary, size=size):
                        raise ValueError(
                            "Rendered map temporary PNG is invalid"
                        )
                    if _valid_cached_png(output, size=size):
                        return MapAsset(output, requested_bounds, requested_crs, style)
                    if os.path.lexists(output):
                        quarantined = output.with_name(
                            f".{output.name}.invalid-{uuid.uuid4().hex}"
                        )
                        output.replace(quarantined)
                    temporary.replace(output)
                    if not _valid_cached_png(output, size=size):
                        _remove_quarantined_leaf(output)
                        raise ValueError("Rendered map published PNG is invalid")
                    published = True
                finally:
                    if os.path.lexists(temporary):
                        temporary.unlink()
                    if quarantined is not None:
                        if not published and not os.path.lexists(output):
                            quarantined.replace(output)
                        elif published:
                            _remove_quarantined_leaf(quarantined)
            return MapAsset(output, requested_bounds, requested_crs, style)

    def boundary_geojson(
        self,
        geometry: Any,
        *,
        crs: str,
        tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Return a reprojected display copy of an exact boundary geometry."""

        from pyproj import Transformer
        from shapely.geometry import mapping
        from shapely.geometry.base import BaseGeometry
        from shapely.ops import transform

        if not isinstance(geometry, BaseGeometry):
            raise ValueError("geometry must be a Shapely geometry")
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("geometry must be non-empty and valid")
        if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError("geometry must be a Polygon or MultiPolygon")
        source_crs = _normalise_crs(crs, label="crs")
        display = _force_finite_2d(geometry, label="display boundary")
        if tolerance is not None:
            simplify_tolerance = _finite_number(tolerance, label="tolerance")
            if simplify_tolerance < 0:
                raise ValueError("tolerance must be non-negative")
            display = geometry.simplify(
                simplify_tolerance,
                preserve_topology=True,
            )
        if source_crs != "EPSG:4326":
            transformer = Transformer.from_crs(
                source_crs,
                "EPSG:4326",
                always_xy=True,
            )
            display = transform(transformer.transform, display)
        display = _force_finite_2d(display, label="display boundary")
        if display.is_empty or not display.is_valid:
            raise ValueError("display boundary is empty or invalid")
        return {
            "type": "Feature",
            "properties": {},
            "geometry": mapping(display),
        }

    def station_geojson(
        self,
        stations: Any,
        boundary: Any,
        *,
        boundary_crs: str,
    ) -> dict[str, Any]:
        """Return the strict city subset with approved display attributes only."""

        import geopandas as gpd
        from pyproj import Transformer
        from shapely.geometry import mapping
        from shapely.geometry.base import BaseGeometry
        from shapely.ops import transform

        if not isinstance(stations, gpd.GeoDataFrame):
            raise ValueError("stations must be a GeoDataFrame")
        if stations.crs is None:
            raise ValueError("stations require a CRS")
        if "cell" not in stations.columns:
            raise ValueError("stations require a cell display field")
        if not isinstance(boundary, BaseGeometry):
            raise ValueError("boundary must be a Shapely geometry")
        if boundary.is_empty or not boundary.is_valid:
            raise ValueError("boundary must be non-empty and valid")
        station_crs = _normalise_crs(stations.crs.to_string(), label="station CRS")
        source_boundary_crs = _normalise_crs(boundary_crs, label="boundary_crs")
        filter_boundary = _force_finite_2d(boundary, label="station boundary")
        if source_boundary_crs != station_crs:
            transformer = Transformer.from_crs(
                source_boundary_crs,
                station_crs,
                always_xy=True,
            )
            filter_boundary = transform(transformer.transform, filter_boundary)
        filter_boundary = _force_finite_2d(
            filter_boundary,
            label="station boundary",
        )

        working = stations.copy(deep=True)
        if working.geometry.isna().any() or working.geometry.is_empty.any():
            raise ValueError("station geometries must be non-empty points")
        if not working.geom_type.eq("Point").all():
            raise ValueError("station geometries must all be points")
        subset = working[working.geometry.within(filter_boundary)].copy()
        if subset["cell"].isna().any():
            raise ValueError("station cell values must not be null")
        display = subset.to_crs("EPSG:4326")
        geometry_column = display.geometry.name
        features: list[dict[str, Any]] = []
        available = set(display.columns)
        for _, row in display.iterrows():
            geometry = _force_finite_2d(
                row[geometry_column],
                label="station geometry",
            )
            properties: dict[str, Any] = {}
            for field in STATION_DISPLAY_FIELDS:
                if field == "longitude":
                    properties[field] = float(geometry.x)
                elif field == "latitude":
                    properties[field] = float(geometry.y)
                elif field in available:
                    properties[field] = _json_value(row[field])
            features.append(
                {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": mapping(geometry),
                }
            )
        return {"type": "FeatureCollection", "features": features}
