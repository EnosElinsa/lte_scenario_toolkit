"""Select an LTE scenario and enrich its stations with DEM elevations."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .candidate_scanner import Candidate, ScanResult
from .config import load_experiment_config
from .data_catalog import CatalogError, DataCatalog, load_data_catalog
from .profiles import ExperimentProfile, load_profile
from .selection_service import SelectionPreflight, SelectionProgress, SelectionService


class SelectorError(ValueError):
    """Raised when an explicitly requested interactive selector cannot run."""


def _selection_profile(
    config: dict[str, Any],
    catalog: DataCatalog,
    scenario_id: str,
) -> ExperimentProfile:
    """Return the current typed profile with explicit CLI overrides."""

    del catalog
    profile = getattr(config, "profile_snapshot", None)
    if not isinstance(profile, ExperimentProfile):
        profile = load_profile(config["config_path"], repo_root=config["repo_root"])
    if profile.scenario_id != scenario_id:
        raise ValueError("Profile scenario does not match the selected scenario")
    return replace(
        profile,
        rect_size=config["rect_size"],
        target_count=config["target_count"],
        output_root=Path(config["output_root"]),
    )


def _selection_io_paths(
    config: dict[str, Any],
    catalog: DataCatalog,
    preflight: SelectionPreflight,
) -> dict[str, Any]:
    """Build display paths from catalog-owned profile inputs."""

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
    """Return the hit/miss message for the shared cache location."""

    if progress.cache_status not in {"hit", "miss", "forced"}:
        raise ValueError("Selection scan did not report a cache status")
    if progress.cache_key is None:
        raise ValueError("Selection scan did not report a cache key")
    cache_name = f"{progress.cache_key}.json"
    if progress.cache_status == "hit":
        return f"Loaded {len(result.candidates)} cached candidates: {cache_name}"
    return f"Saved {len(result.candidates)} candidates: {cache_name}"


def _scanned_candidate(value: Any, scan_result: ScanResult) -> Candidate:
    """Resolve one selector value to exactly one completed-scan candidate."""

    if not isinstance(value, Candidate):
        raise ValueError("Selector must return one scanned Candidate")
    matches = [candidate for candidate in scan_result.candidates if candidate == value]
    if len(matches) != 1:
        raise ValueError("Selected candidate must match exactly one completed scan candidate")
    return matches[0]


def _web_selected_candidate(
    scan_result: ScanResult,
    *,
    preflight: SelectionPreflight,
    selection_service: SelectionService,
    repo_root: str | Path,
) -> Candidate | None:
    """Run the optional local web selector without importing it for other modes."""

    guidance = "Use --select-index in headless environments."
    try:
        from .web_selector import (
            WebSelectorError,
            WebSelectorPayload,
            select_candidate,
        )
    except ImportError as exc:
        raise SelectorError(f"Web selector is unavailable: {exc}. {guidance}") from exc

    try:
        payload = WebSelectorPayload(
            preflight=preflight,
            selection_service=selection_service,
            scan_result=scan_result,
            repo_root=Path(repo_root).expanduser().resolve(strict=False),
        )
        return select_candidate(scan_result.candidates, map_payload=payload)
    except (ImportError, WebSelectorError) as exc:
        raise SelectorError(f"Web selector could not start: {exc}. {guidance}") from exc


def _select_candidate(
    scan_result: ScanResult,
    *,
    select_index: int | None,
    config: dict[str, Any],
    preflight: SelectionPreflight,
    selection_service: SelectionService,
) -> Candidate | None:
    """Select one candidate by explicit index or the local web UI."""

    if select_index is not None:
        if type(select_index) is not int or not 1 <= select_index <= len(
            scan_result.candidates
        ):
            raise ValueError(
                f"select_index must be between 1 and {len(scan_result.candidates)}"
            )
        return scan_result.candidates[select_index - 1]
    selected = _web_selected_candidate(
        scan_result,
        preflight=preflight,
        selection_service=selection_service,
        repo_root=config["repo_root"],
    )
    return None if selected is None else _scanned_candidate(selected, scan_result)


def _export_artifacts(config: dict[str, Any]) -> tuple[str, ...]:
    """Translate profile output flags to stable artifact tokens."""

    flags = (
        ("save_csv", "csv"),
        ("save_preview_png", "preview_png"),
        ("save_terrain_png", "terrain_png"),
        ("save_terrain_eps", "terrain_eps"),
        ("save_terrain_html", "terrain_html"),
    )
    return tuple(token for flag, token in flags if config.get(flag, True))


def _publish_candidate(
    selection_service: SelectionService,
    preflight: SelectionPreflight,
    scan_result: ScanResult,
    candidate: Candidate,
    *,
    artifacts: tuple[str, ...],
    entrypoint: tuple[str, ...] | list[str],
) -> Path:
    """Publish one unique run through the shared service."""

    return Path(selection_service.export(
        preflight,
        scan_result,
        candidate,
        output_root=preflight.output_root,
        artifacts=artifacts,
        entrypoint=entrypoint,
    ))


def _report_published_run(run_dir: Path) -> None:
    """Print published artifacts from the authoritative run record."""

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
    parser.add_argument(
        "--output-root",
        type=Path,
        help="root for a new unique scenario/profile/run directory",
    )
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
            output_dir=args.output_root,
        )
        if args.size is not None:
            config["rect_size"] = args.size
        if args.target is not None:
            config["target_count"] = args.target
        catalog = load_data_catalog(
            Path(config["repo_root"]) / "data" / "datasets.yaml",
            repo_root=config["repo_root"],
        )
        scenario_id = config["scenario_id"]
        profile = _selection_profile(config, catalog, scenario_id)
        selection_service = SelectionService(catalog)
        preflight = selection_service.preflight(
            profile,
            output_root=config["output_root"],
        )
        config.update(_selection_io_paths(config, catalog, preflight))
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
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(_shared_cache_message(scan_result, cache_progress))

    try:
        selected_candidate = _select_candidate(
            scan_result,
            select_index=args.select_index,
            config=config,
            preflight=preflight,
            selection_service=selection_service,
        )
        if selected_candidate is None:
            print("No rectangle selected", file=sys.stderr)
            return 2
        run_dir = _publish_candidate(
            selection_service,
            preflight,
            scan_result,
            selected_candidate,
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
