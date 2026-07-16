"""Select an LTE scenario and enrich its stations with DEM elevations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from . import io, scenario, spatial, terrain, visualization
from .candidate_scanner import Candidate, ScanResult
from .config import load_experiment_config
from .data_catalog import CatalogError, DataCatalog, load_data_catalog
from .profiles import ExperimentProfile, FigureSettings, OutputSettings, load_profile
from .selection_service import SelectionPreflight, SelectionProgress, SelectionService


def _points_dataset_id(catalog: DataCatalog, points_path: Path) -> str:
    matches = [
        dataset_id
        for dataset_id, dataset in catalog.datasets_by_id.items()
        if dataset.get("role") == "points"
        and catalog.resolve(dataset["entrypoint"]) == points_path.resolve()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one registered points dataset for {points_path}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _legacy_profile_id(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).casefold()).strip("-")
    return slug or "legacy-profile"


def _selection_profile(
    config: dict[str, Any],
    catalog: DataCatalog,
    scenario_id: str,
) -> ExperimentProfile:
    """Build the typed service profile used by legacy and schema-v2 CLIs."""

    if config.get("profile_id") is not None:
        profile = getattr(config, "profile_snapshot", None)
        if not isinstance(profile, ExperimentProfile):
            profile = load_profile(config["config_path"], repo_root=config["repo_root"])
        return replace(
            profile,
            rect_size=config["rect_size"],
            target_count=config["target_count"],
            output_root=Path(config["output_root"]),
        )

    return ExperimentProfile(
        schema_version=2,
        profile_id=_legacy_profile_id(
            config.get("experiment_name", Path(config["config_path"]).stem)
        ),
        display_name=str(config.get("experiment_name", scenario_id)),
        scenario_id=scenario_id,
        points_dataset_id=_points_dataset_id(catalog, Path(config["points_shp"])),
        random_seed=config.get("random_seed", 42),
        target_crs=config["target_crs"],
        rect_size=config["rect_size"],
        target_count=config["target_count"],
        tolerance=config["tolerance"],
        scan_mode=config.get("scan_mode", "fast"),
        strategy=config["strategy"],
        scan_step=config["scan_step"],
        max_rects=config["max_rects"],
        min_spacing=config["min_spacing"],
        output_root=Path(config["output_root"]),
        outputs=OutputSettings(
            save_csv=config.get("save_csv", True),
            save_preview_png=config.get("save_preview_png", True),
            save_terrain_png=config.get("save_terrain_png", True),
            save_terrain_eps=config.get("save_terrain_eps", True),
            save_terrain_html=config.get("save_terrain_html", True),
        ),
        figure=FigureSettings(),
        source_path=Path(config["config_path"]),
    )


def _legacy_result(candidate: Candidate, rectangle_size: int) -> dict[str, Any]:
    return {
        "geometry": box(
            candidate.left_x,
            candidate.bottom_y,
            candidate.left_x + rectangle_size,
            candidate.bottom_y + rectangle_size,
        ),
        "flat_grid_id": candidate.flat_grid_id,
        "pt_count": candidate.point_count,
        "left_x": candidate.left_x,
        "bottom_y": candidate.bottom_y,
        "center_x": candidate.center_x,
        "center_y": candidate.center_y,
    }


def _profile_selection_io_paths(
    config: dict[str, Any],
    catalog: DataCatalog,
    preflight: SelectionPreflight,
) -> dict[str, Any]:
    """Build temporary legacy renderer paths from catalog-owned profile inputs."""

    registered = catalog.scenario(preflight.scenario_id)
    city_tag = registered["scenario_id"]
    output_dir = Path(preflight.output_root)
    base_name = (
        f"{city_tag}_{config['rect_size']}m_"
        f"target{config['target_count']}_tol{config['tolerance']}"
    )
    return {
        "boundary_folder": city_tag,
        "boundary_layer": Path(preflight.boundary_path).stem,
        "points_shp": Path(preflight.points_path),
        "boundary_shp": Path(preflight.boundary_path),
        "dem_path": Path(preflight.dem_path),
        "output_dir": output_dir,
        "output_csv": output_dir / f"{base_name}.csv",
        "output_3d_png": output_dir / f"{base_name}_3d.png",
        "output_3d_html": output_dir / f"{base_name}_3d.html",
        "preview_png": output_dir / f"{base_name}.png",
    }


def _shared_cache_message(result: ScanResult, progress: SelectionProgress) -> str:
    """Return the legacy hit/miss message for the shared cache location."""

    if progress.cache_status not in {"hit", "miss", "forced"}:
        raise ValueError("Selection scan did not report a cache status")
    if progress.cache_key is None:
        raise ValueError("Selection scan did not report a cache key")
    cache_name = f"{progress.cache_key}.json"
    if progress.cache_status == "hit":
        return f"Loaded {len(result.candidates)} cached candidates: {cache_name}"
    return f"Saved {len(result.candidates)} candidates: {cache_name}"


def _chosen_candidate(
    chosen: list[dict[str, Any]],
    scan_result: ScanResult,
) -> Candidate:
    """Map one legacy selector result back to one scanned candidate."""

    if len(chosen) != 1 or not isinstance(chosen[0], dict):
        raise ValueError("Selection must contain exactly one legacy candidate")
    flat_grid_id = chosen[0].get("flat_grid_id")
    matches = [
        candidate
        for candidate in scan_result.candidates
        if candidate.flat_grid_id == flat_grid_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "Selected flat_grid_id must match exactly one completed scan candidate"
        )
    return matches[0]


def _export_artifacts(config: dict[str, Any]) -> tuple[str, ...]:
    """Translate legacy output flags to the service's stable artifact tokens."""

    flags = (
        ("save_csv", "csv"),
        ("save_preview_png", "preview_png"),
        ("save_terrain_png", "terrain_png"),
        ("save_terrain_eps", "terrain_eps"),
        ("save_terrain_html", "terrain_html"),
    )
    return tuple(token for flag, token in flags if config.get(flag, True))


def _report_published_run(run_dir: Path) -> None:
    """Print legacy output labels from the authoritative published run record."""

    run_record_path = run_dir / "run.json"
    record = json.loads(run_record_path.read_text(encoding="utf-8"))
    for artifact in record.get("artifacts", []):
        path = run_dir / artifact
        if artifact.endswith(".csv"):
            print(f"Scenario CSV: {path}")
        elif artifact.endswith(".png") and not artifact.endswith("_3d.png"):
            print(f"Preview: {path}")
    for error in record.get("errors", []):
        print(
            f"WARNING: {error.get('artifact', 'artifact')}: "
            f"{error.get('message', 'export failed')}",
            file=sys.stderr,
        )
    print(f"Run record: {run_record_path}")


def _linked_catalog_scenario(
    config: dict[str, Any],
) -> tuple[DataCatalog, dict[str, Any]] | None:
    """Return the one catalog scenario linked to this experiment config."""

    repo_root = config.get("repo_root")
    config_path = config.get("config_path")
    if repo_root is None or config_path is None:
        return None
    repository = Path(repo_root).resolve()
    catalog_path = repository / "data" / "datasets.yaml"
    if not catalog_path.is_file():
        return None

    catalog = load_data_catalog(catalog_path, repo_root=repository)
    experiment_path = Path(config_path)
    if not experiment_path.is_absolute():
        experiment_path = repository / experiment_path
    experiment_path = experiment_path.resolve()
    matches: list[dict[str, Any]] = []
    for scenario_id in sorted(catalog.scenarios_by_id):
        registered_scenario = catalog.scenario(scenario_id)
        registered_config = registered_scenario.get("config_path")
        if (
            registered_config is not None
            and catalog.resolve(registered_config) == experiment_path
        ):
            matches.append(registered_scenario)
    if not matches:
        return None
    if len(matches) > 1:
        scenario_ids = ", ".join(item["scenario_id"] for item in matches)
        raise CatalogError(
            f"Experiment config {experiment_path} is linked by multiple scenarios: "
            f"{scenario_ids}"
        )
    return catalog, matches[0]


def resolve_selection_io_paths(
    config: dict[str, Any],
    *,
    create_output: bool = True,
) -> dict[str, Any]:
    """Resolve experiment paths and enforce linked catalog entrypoints."""

    paths = spatial.resolve_io_paths(config, create_output=False)
    linked = _linked_catalog_scenario(config)
    if linked is not None:
        catalog, registered_scenario = linked
        scenario_id = registered_scenario["scenario_id"]
        boundary = catalog.dataset(registered_scenario["boundary_dataset_id"])
        registered_boundary = catalog.resolve(boundary["entrypoint"])
        resolved_boundary = Path(paths["boundary_shp"]).resolve()
        if resolved_boundary != registered_boundary:
            raise ValueError(
                f"Linked scenario {scenario_id!r} boundary does not match config "
                f"({resolved_boundary} != {registered_boundary})"
            )

        dem_id = registered_scenario.get("dem_dataset_id")
        if dem_id is None:
            raise ValueError(f"Linked scenario {scenario_id!r} does not declare a DEM")
        dem = catalog.dataset(dem_id)
        registered_dem = catalog.resolve(dem["entrypoint"])
        resolved_dem = Path(paths["dem_path"]).resolve()
        if resolved_dem != registered_dem:
            raise ValueError(
                f"Linked scenario {scenario_id!r} DEM does not match config "
                f"({resolved_dem} != {registered_dem})"
            )

        paths["boundary_shp"] = registered_boundary
        paths["dem_path"] = registered_dem
        paths["registered_scenario_id"] = scenario_id

    if create_output:
        Path(paths["output_dir"]).mkdir(parents=True, exist_ok=True)
    return paths


def process_selected_rectangles(chosen, points_gdf, dem, config):
    """Extract stations and elevations for the selected rectangles."""

    if not chosen:
        return None, None

    rectangle_size = config["rect_size"]
    point_crs = points_gdf.crs
    frames = []
    selected_groups = []

    for index, result in enumerate(chosen, start=1):
        rectangle = result["geometry"]
        minimum_x, minimum_y, maximum_x, maximum_y = rectangle.bounds
        rough_mask = (
            (points_gdf.geometry.x >= minimum_x)
            & (points_gdf.geometry.x <= maximum_x)
            & (points_gdf.geometry.y >= minimum_y)
            & (points_gdf.geometry.y <= maximum_y)
        )
        candidates = points_gdf[rough_mask]
        selected = candidates[candidates.geometry.within(rectangle)].copy().reset_index(drop=True)
        if selected.empty:
            continue

        elevations = terrain.extract_elevation(selected, dem)
        terrain.require_valid_elevations(elevations)
        selected["elevation"] = elevations
        selected_groups.append(selected)
        frames.append(
            io.build_output_dataframe(
                selected,
                point_crs,
                rect_id=index,
                pt_count=result["pt_count"],
                left_x=result["left_x"],
                bottom_y=result["bottom_y"],
                center_x=result["center_x"],
                center_y=result["center_y"],
                rect_size=rectangle_size,
            )
        )

    if not frames:
        return None, None
    frame = pd.concat(frames, ignore_index=True)
    selected_points = gpd.GeoDataFrame(pd.concat(selected_groups, ignore_index=True), crs=point_crs)
    return frame, selected_points


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Select an LTE scenario using a reproducible YAML configuration"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="experiment YAML; paths are repository-relative",
    )
    parser.add_argument("--city", help="boundary directory or layer name")
    parser.add_argument("--output-dir", type=Path, help="write outputs directly to this directory")
    parser.add_argument("--size", type=int, help="override rectangle size in metres")
    parser.add_argument("--target", type=int, help="override target base-station count")
    parser.add_argument(
        "--select-index",
        type=int,
        help="choose a one-based candidate index without opening the interactive selector",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Run the configured scenario-selection workflow."""

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    args = _parse_args(argv)
    try:
        config = load_experiment_config(
            args.config,
            city=args.city,
            output_dir=args.output_dir,
        )
        if args.size is not None:
            config["rect_size"] = args.size
        if args.target is not None:
            config["target_count"] = args.target
        catalog = load_data_catalog(
            Path(config["repo_root"]) / "data" / "datasets.yaml",
            repo_root=config["repo_root"],
        )
        is_versioned_profile = config.get("profile_id") is not None
        if is_versioned_profile:
            scenario_id = config["scenario_id"]
        else:
            config.update(resolve_selection_io_paths(config, create_output=False))
            scenario_id = config.get("registered_scenario_id")
            if scenario_id is None:
                raise ValueError(
                    "Selection requires a scenario registered in data/datasets.yaml"
                )
        profile = _selection_profile(config, catalog, scenario_id)
        selection_service = SelectionService(catalog)
        preflight = selection_service.preflight(
            profile,
            output_root=config["output_root"],
        )
        if is_versioned_profile:
            config.update(_profile_selection_io_paths(config, catalog, preflight))
        else:
            config["points_shp"] = preflight.points_path
            config["boundary_shp"] = preflight.boundary_path
            config["dem_path"] = preflight.dem_path
    except (CatalogError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"City: {config['boundary_folder']} ({config['boundary_layer']})")
    print(f"Points: {config['points_shp']}")
    print(f"Boundary: {config['boundary_shp']}")
    print(f"DEM: {config['dem_path']}")
    print(f"Output: {config['output_dir']}")
    try:
        cache_progress: SelectionProgress | None = None

        def capture_progress(event: SelectionProgress) -> None:
            nonlocal cache_progress
            if event.cache_status is not None:
                cache_progress = event

        scan_result = selection_service.scan(preflight, progress=capture_progress)
        if cache_progress is None:
            raise ValueError("Selection scan did not report cache status")
        prepared = selection_service.prepared_selection(preflight)
        points_gdf = prepared.points
        boundary = prepared.boundary
        coordinates = prepared.coordinates
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    results = [
        _legacy_result(candidate, config["rect_size"])
        for candidate in scan_result.candidates
    ]
    print(_shared_cache_message(scan_result, cache_progress))

    try:
        scenario.verify_results(results, coordinates, config["rect_size"])
        if args.select_index is None:
            chosen = visualization.interactive_select(
                points_gdf,
                boundary,
                results,
                config,
            )
        else:
            chosen = scenario.choose_result(results, args.select_index)
        if not chosen:
            print("No rectangle selected", file=sys.stderr)
            return 2
        selected_candidate = _chosen_candidate(chosen, scan_result)
        run_dir = selection_service.export(
            preflight,
            scan_result,
            selected_candidate,
            output_root=preflight.output_root,
            artifacts=_export_artifacts(config),
            entrypoint=(
                sys.argv if argv is None else ["lte-select-sites", *argv]
            ),
        )
        _report_published_run(run_dir)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0
