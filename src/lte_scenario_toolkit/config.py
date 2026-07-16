"""Load reproducible experiment configuration from YAML."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .profiles import load_profile

VALID_SCAN_STRATEGIES = {"sequential", "uniform"}


def _required(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"Missing required configuration value: {section}.{key}") from exc


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _infer_project_root(config_path: Path) -> Path:
    parent = config_path.parent
    return parent.parent if parent.name.casefold() == "configs" else parent


def load_experiment_config(
    config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    city: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return the nested YAML configuration in the legacy flat script shape.

    Relative data paths are resolved against the repository root, not the
    current working directory, so the same file behaves consistently in CI and
    when invoked from another directory.
    """

    path = Path(config_path).resolve()
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        raise ValueError("The experiment configuration must be a YAML mapping")

    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else _infer_project_root(path)
    )
    if document.get("schema_version") == 2:
        profile = load_profile(path, repo_root=root)
        if city is not None:
            resolved_city = city
            catalog_path = root / "data" / "datasets.yaml"
            if catalog_path.is_file():
                from .data_catalog import load_data_catalog

                catalog = load_data_catalog(catalog_path, repo_root=root)
                if city in catalog.scenarios_by_id:
                    resolved_city = city
                else:
                    matches = [
                        scenario_id
                        for scenario_id, scenario in catalog.scenarios_by_id.items()
                        if str(scenario["display_name"]).casefold() == city.casefold()
                    ]
                    if len(matches) == 1:
                        resolved_city = matches[0]
            if resolved_city != profile.scenario_id:
                raise ValueError(
                    f"--city {city!r} does not match profile scenario_id "
                    f"{profile.scenario_id!r}"
                )

        config = profile.runtime_values()
        config["repo_root"] = root
        if output_dir is not None:
            config["output_root"] = _resolve_path(output_dir, root)
            config["output_dir_is_final"] = True
        return config

    experiment = document.get("experiment", {})
    inputs = document.get("inputs", {})
    spatial = document.get("spatial", {})
    scan = document.get("scan", {})
    outputs = document.get("outputs", {})
    strategy = str(_required(scan, "strategy", "scan"))
    if strategy not in VALID_SCAN_STRATEGIES:
        choices = ", ".join(sorted(VALID_SCAN_STRATEGIES))
        raise ValueError(f"scan.strategy must be one of: {choices}")

    configured_city = city or str(_required(inputs, "city", "inputs"))
    configured_output = output_dir or _required(outputs, "root", "outputs")

    config: dict[str, Any] = {
        "experiment_name": str(experiment.get("name", path.stem)),
        "random_seed": int(experiment.get("random_seed", 42)),
        "points_root": _resolve_path(_required(inputs, "points_root", "inputs"), root),
        "points_layer": str(_required(inputs, "points_layer", "inputs")),
        "boundary_root": _resolve_path(_required(inputs, "boundary_root", "inputs"), root),
        "city_name": configured_city,
        "dem_path": _resolve_path(_required(inputs, "dem_path", "inputs"), root),
        "target_crs": str(_required(spatial, "target_crs", "spatial")),
        "rect_size": int(_required(spatial, "rectangle_size_m", "spatial")),
        "target_count": int(
            _required(spatial, "target_base_station_count", "spatial")
        ),
        "tolerance": int(_required(spatial, "count_tolerance", "spatial")),
        "strategy": strategy,
        "scan_step": int(_required(scan, "step_m", "scan")),
        "max_rects": int(_required(scan, "max_rectangles", "scan")),
        "min_spacing": int(_required(scan, "minimum_center_spacing_m", "scan")),
        "output_root": _resolve_path(configured_output, root),
        "output_dir_is_final": True,
        "save_csv": bool(outputs.get("save_csv", True)),
        "save_preview_png": bool(outputs.get("save_preview_png", True)),
        "save_terrain_png": bool(outputs.get("save_terrain_png", True)),
        "save_terrain_eps": bool(outputs.get("save_terrain_eps", True)),
        "save_terrain_html": bool(outputs.get("save_terrain_html", True)),
        "config_path": path,
        "repo_root": root,
    }

    positive_keys = ("rect_size", "scan_step", "max_rects", "min_spacing")
    for key in positive_keys:
        if config[key] <= 0:
            raise ValueError(f"{key} must be greater than zero")
    if config["target_count"] < 0 or config["tolerance"] < 0:
        raise ValueError("target_count and tolerance must be non-negative")

    return config
