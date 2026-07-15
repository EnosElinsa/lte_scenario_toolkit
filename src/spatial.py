"""Spatial data discovery, CRS normalization, and point filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def resolve_root_dir(root_dir: str | Path, repo_root: str | Path | None = None) -> Path:
    """Resolve a data path against the repository instead of the current directory."""

    root = Path(root_dir)
    base = Path(repo_root).resolve() if repo_root is not None else REPOSITORY_ROOT
    return root.resolve() if root.is_absolute() else (base / root).resolve()


def build_layer_shp_path(root_dir: str | Path, layer_name: str) -> Path:
    """Find the named Shapefile, accepting a directory with one alternate name."""

    layer_dir = Path(root_dir) / layer_name
    exact = layer_dir / f"{layer_name}.shp"
    if exact.exists():
        return exact
    candidates = sorted(layer_dir.glob("*.shp"))
    return candidates[0] if len(candidates) == 1 else exact


def discover_boundary_layers(boundary_root: str | Path) -> list[dict[str, Any]]:
    """Discover one usable Shapefile per city directory."""

    root = Path(boundary_root)
    if not root.exists():
        raise FileNotFoundError(f"Boundary root does not exist: {root}")

    layers: list[dict[str, Any]] = []
    for folder in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda p: p.name):
        exact = folder / f"{folder.name}.shp"
        candidates = sorted(folder.glob("*.shp"))
        if exact.exists():
            shapefile = exact
        elif len(candidates) == 1:
            shapefile = candidates[0]
        else:
            continue
        layers.append(
            {
                "folder_name": folder.name,
                "layer_name": shapefile.stem,
                "shp_path": shapefile,
            }
        )

    if not layers:
        raise FileNotFoundError(f"No usable boundary Shapefile found below: {root}")
    return layers


def _choose_boundary(layers: list[dict[str, Any]], config: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    city_name = config.get("city_name")
    if city_name:
        wanted = str(city_name).casefold()
        for index, item in enumerate(layers, start=1):
            if wanted in {item["folder_name"].casefold(), item["layer_name"].casefold()}:
                return index, item
        choices = ", ".join(item["folder_name"] for item in layers)
        raise ValueError(f"Unknown city {city_name!r}. Available cities: {choices}")

    city_id = int(config.get("city_id", 1))
    if city_id < 1 or city_id > len(layers):
        choices = " | ".join(
            f"{index}:{item['folder_name']}" for index, item in enumerate(layers, start=1)
        )
        raise ValueError(f"city_id out of range: {city_id}. Available cities: {choices}")
    return city_id, layers[city_id - 1]


def resolve_io_paths(config: dict[str, Any], *, create_output: bool = True) -> dict[str, Any]:
    """Resolve input paths and deterministic output names for an experiment."""

    boundary_root = resolve_root_dir(config["boundary_root"], config.get("repo_root"))
    layers = discover_boundary_layers(boundary_root)
    city_id, boundary_item = _choose_boundary(layers, config)
    points_root = resolve_root_dir(config["points_root"], config.get("repo_root"))
    points_path = build_layer_shp_path(points_root, config["points_layer"])
    boundary_path = boundary_item["shp_path"]
    if not points_path.exists():
        raise FileNotFoundError(f"Base-station Shapefile does not exist: {points_path}")
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary Shapefile does not exist: {boundary_path}")

    city_tag = boundary_item["folder_name"]
    base_name = (
        f"{city_tag}_{config['rect_size']}m_"
        f"target{config['target_count']}_tol{config['tolerance']}"
    )
    cache_name = (
        f"{base_name}_step{config['scan_step']}_"
        f"sp{config['min_spacing']}_{config['strategy']}"
    )
    output_root = resolve_root_dir(config["output_root"], config.get("repo_root"))
    output_dir = output_root if config.get("output_dir_is_final") else output_root / city_tag
    if create_output:
        output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "city_id": city_id,
        "city_options": [item["folder_name"] for item in layers],
        "boundary_folder": city_tag,
        "boundary_layer": boundary_item["layer_name"],
        "points_shp": points_path,
        "boundary_shp": boundary_path,
        "dem_path": resolve_root_dir(config["dem_path"], config.get("repo_root")),
        "output_dir": output_dir,
        "output_csv": output_dir / f"{base_name}.csv",
        "output_3d_png": output_dir / f"{base_name}_3d.png",
        "output_3d_html": output_dir / f"{base_name}_3d.html",
        "preview_png": output_dir / f"{base_name}.png",
        "cache_json": output_dir / f"{cache_name}_cache.json",
    }


def prepare_spatial_data(
    points_gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    *,
    target_crs: str = "EPSG:3857",
) -> tuple[gpd.GeoDataFrame, Any, np.ndarray]:
    """Project inputs, dissolve the boundary, and retain points strictly inside it."""

    if points_gdf.crs is None or boundary_gdf.crs is None:
        raise ValueError("Both base-station points and boundary data require a CRS")
    if points_gdf.empty:
        raise ValueError("Base-station point data is empty")
    if boundary_gdf.empty:
        raise ValueError("Boundary data is empty")

    points = points_gdf.to_crs(target_crs)
    boundaries = boundary_gdf.to_crs(target_crs)
    boundary = boundaries.geometry.union_all()
    if boundary.is_empty:
        raise ValueError("Boundary geometry is empty")

    selected = points[points.geometry.within(boundary)].copy().reset_index(drop=True)
    coordinates = np.column_stack(
        (
            np.asarray(selected.geometry.x.to_numpy(), dtype=float),
            np.asarray(selected.geometry.y.to_numpy(), dtype=float),
        )
    )
    return selected, boundary, coordinates


def load_and_prepare(config: dict[str, Any]):
    """Read configured vector inputs and prepare them for scenario scanning."""

    points = gpd.read_file(config["points_shp"])
    boundaries = gpd.read_file(config["boundary_shp"])
    return prepare_spatial_data(points, boundaries, target_crs=config.get("target_crs", "EPSG:3857"))

