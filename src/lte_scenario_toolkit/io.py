"""Tabular output, dataset manifests, and reproducible run records."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_output_dataframe(
    selected,
    points_crs,
    *,
    rect_id,
    pt_count,
    left_x,
    bottom_y,
    center_x,
    center_y,
    rect_size,
) -> pd.DataFrame:
    """Build the stable CSV schema used by downstream figure generation."""

    del rect_size  # retained in the public signature for compatibility
    frame = pd.DataFrame(index=range(len(selected)))
    for column in ("cell", "Cell", "CELL"):
        if column in selected.columns:
            frame["cell"] = selected[column].to_numpy()
            break
    else:
        frame["cell"] = range(1, len(selected) + 1)

    wgs84 = selected if points_crs.to_epsg() == 4326 else selected.to_crs(epsg=4326)
    frame["lon"] = wgs84.geometry.x.to_numpy()
    frame["lat"] = wgs84.geometry.y.to_numpy()

    for column in ("range", "Range", "RANGE"):
        if column in selected.columns:
            frame["range"] = selected[column].to_numpy()
            break
    else:
        frame["range"] = np.nan

    projected = selected if points_crs.to_epsg() == 3857 else selected.to_crs(epsg=3857)
    frame["X"] = projected.geometry.x.to_numpy()
    frame["Y"] = projected.geometry.y.to_numpy()
    frame["rect_id"] = rect_id
    frame["pt_count"] = pt_count
    frame["left_x"] = left_x
    frame["bottom_y"] = bottom_y
    frame["center_x"] = center_x
    frame["center_y"] = center_y
    if "elevation" in selected.columns:
        frame["elevation"] = selected["elevation"].to_numpy()
    return frame.reset_index(drop=True)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a streaming SHA256 without loading large datasets into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_record(
    path: str | Path,
    *,
    name: str,
    source_url: str,
    license_name: str,
    download_date: str | None = None,
    crs: str | None = None,
    resolution_m: float | None = None,
) -> dict[str, Any]:
    """Describe one local input using provenance and integrity metadata."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    record: dict[str, Any] = {
        "name": name,
        "path": str(source),
        "source_url": source_url,
        "license": license_name,
        "download_date": download_date,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }
    if crs is not None:
        record["crs"] = crs
    if resolution_m is not None:
        record["resolution_m"] = resolution_m
    return record


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime,)) or (
        hasattr(value, "isoformat") and value.__class__.__module__ == "datetime"
    ):
        return value.isoformat()
    return value


def create_data_manifest(
    metadata_path: str | Path,
    output_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    dataset_ids: Iterable[str] | None = None,
) -> Path:
    """Validate a schema-v2 catalog and update its checksummed JSON manifest."""

    from .data_catalog import _load_data_catalog, update_data_manifest

    catalog = _load_data_catalog(metadata_path, repo_root)
    return update_data_manifest(catalog, output_path, dataset_ids=dataset_ids)


def software_versions() -> dict[str, str]:
    """Return the interpreter and principal geospatial package versions."""

    versions = {"python": platform.python_version()}
    for package in ("geopandas", "numpy", "pandas", "rasterio", "shapely"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _git_commit(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_run_record(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    inputs: Iterable[dict[str, Any]],
    outputs: Iterable[str | Path],
    command: Iterable[str],
    timestamp: str | None = None,
    filename: str = "run.json",
) -> Path:
    """Write a compact machine-readable record for one experiment run."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    repository = Path(config.get("repo_root", Path.cwd()))
    payload = {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": list(command),
        "git_commit": _git_commit(repository),
        "config": _json_safe(config),
        "inputs": _json_safe(list(inputs)),
        "software": software_versions(),
        "outputs": [str(Path(path)) for path in outputs],
    }
    path = directory / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
