"""Opt-in, output-free benchmarks for the production candidate scanner."""

from __future__ import annotations

import argparse
import json
import sys
import tracemalloc
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml

from .candidate_scanner import ScanRequest, grid_axes, scan_candidates
from .config import load_experiment_config
from .data_catalog import load_data_catalog
from .select_sites import _selection_profile, resolve_selection_io_paths
from .selection_service import SelectionService

BenchmarkMetrics = dict[str, int | float | str]
BenchmarkInputs = tuple[ScanRequest, Any, np.ndarray, str]


def _load_benchmark_inputs(config_path: str | Path) -> BenchmarkInputs:
    """Load the same frozen profile, preflight, and spatial snapshot as selection."""

    config = load_experiment_config(config_path)
    repository = Path(config["repo_root"]).resolve()
    catalog = load_data_catalog(
        repository / "data" / "datasets.yaml",
        repo_root=repository,
    )
    if config.get("profile_id") is not None:
        scenario_id = config["scenario_id"]
    else:
        config.update(resolve_selection_io_paths(config, create_output=False))
        scenario_id = config.get("registered_scenario_id")
        if type(scenario_id) is not str or not scenario_id:
            raise ValueError(
                "Candidate benchmark requires a scenario registered in "
                "data/datasets.yaml"
            )

    profile = _selection_profile(config, catalog, scenario_id)
    service = SelectionService(catalog)
    preflight = service.preflight(profile, output_root=config["output_root"])
    prepared = service.prepared_selection(preflight)
    return (
        service._request(profile),
        prepared.boundary,
        prepared.coordinates,
        preflight.scenario_id,
    )


def benchmark_profile(config_path: str | Path) -> BenchmarkMetrics:
    """Benchmark one real profile without reading or writing candidate caches."""

    request, boundary, coordinates, scenario_id = _load_benchmark_inputs(config_path)
    x_origins, y_origins = grid_axes(
        boundary,
        request.rectangle_size,
        request.step,
    )
    grid_x_positions = int(len(x_origins))
    grid_y_positions = int(len(y_origins))
    del x_origins, y_origins

    tracing_was_active = tracemalloc.is_tracing()
    if tracing_was_active:
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()
    started_at = perf_counter()
    try:
        result = scan_candidates(request, boundary, coordinates)
    finally:
        elapsed_seconds = perf_counter() - started_at
        _, peak_python_bytes = tracemalloc.get_traced_memory()
        if not tracing_was_active:
            tracemalloc.stop()

    return {
        "candidate_count": len(result.candidates),
        "checked_positions": result.checked_positions,
        "elapsed_seconds": elapsed_seconds,
        "grid_positions": grid_x_positions * grid_y_positions,
        "grid_x_positions": grid_x_positions,
        "grid_y_positions": grid_y_positions,
        "peak_python_bytes": peak_python_bytes,
        "scenario": scenario_id,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the production candidate scanner without cache or outputs"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="registered experiment YAML or schema-version-2 profile",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print sorted machine-readable metrics for one benchmark profile."""

    args = _parse_args(argv)
    try:
        metrics = benchmark_profile(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(metrics, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
