"""Candidate-grid generation and LTE scenario constraints."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from shapely.geometry import box
from shapely.prepared import prep

from .candidate_scanner import Candidate


def generate_scan_positions(
    boundary: Any,
    rectangle_size: float,
    step: float,
    strategy: str = "sequential",
    *,
    random_seed: int = 42,
) -> np.ndarray:
    """Compatibility helper that materializes the legacy Cartesian scan grid."""

    if rectangle_size <= 0 or step <= 0:
        raise ValueError("rectangle_size and step must be greater than zero")
    if strategy not in {"sequential", "uniform"}:
        raise ValueError("strategy must be 'sequential' or 'uniform'")

    minimum_x, minimum_y, maximum_x, maximum_y = boundary.bounds
    xs = np.arange(minimum_x, maximum_x - rectangle_size, step)
    ys = np.arange(minimum_y, maximum_y - rectangle_size, step)
    if xs.size == 0 or ys.size == 0:
        return np.empty((0, 2), dtype=float)

    xx, yy = np.meshgrid(xs, ys)
    positions = np.column_stack((xx.ravel(), yy.ravel()))
    if strategy == "uniform":
        positions = np.random.default_rng(random_seed).permutation(positions)
    return positions


def _point_count(coordinates: np.ndarray, x: float, y: float, size: float) -> int:
    if coordinates.size == 0:
        return 0
    mask = (
        (coordinates[:, 0] >= x)
        & (coordinates[:, 0] <= x + size)
        & (coordinates[:, 1] >= y)
        & (coordinates[:, 1] <= y + size)
    )
    return int(mask.sum())


def scan_rectangles(
    coordinates: np.ndarray,
    boundary: Any,
    positions: np.ndarray,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compatibility scanner for callers that still supply materialized positions."""

    rectangle_size = float(config["rect_size"])
    target_min = int(config["target_count"]) - int(config["tolerance"])
    target_max = int(config["target_count"]) + int(config["tolerance"])
    maximum_results = int(config["max_rects"])
    minimum_spacing = float(config["min_spacing"])
    prepared_boundary = prep(boundary)
    results: list[dict[str, Any]] = []
    centers: list[tuple[float, float]] = []

    for x_value, y_value in positions:
        x = float(x_value)
        y = float(y_value)
        point_count = _point_count(coordinates, x, y, rectangle_size)
        if not target_min <= point_count <= target_max:
            continue

        center_x = x + rectangle_size / 2
        center_y = y + rectangle_size / 2
        if centers:
            center_array = np.asarray(centers)
            distances = np.hypot(center_array[:, 0] - center_x, center_array[:, 1] - center_y)
            if np.any(distances < minimum_spacing):
                continue

        geometry = box(x, y, x + rectangle_size, y + rectangle_size)
        if not prepared_boundary.contains(geometry):
            continue

        results.append(
            {
                "geometry": geometry,
                "pt_count": point_count,
                "left_x": round(x, 2),
                "bottom_y": round(y, 2),
                "center_x": round(center_x, 2),
                "center_y": round(center_y, 2),
            }
        )
        centers.append((center_x, center_y))
        if len(results) >= maximum_results:
            break
    return results


def validate_results(
    results: Iterable[dict[str, Any]],
    coordinates: np.ndarray,
    rectangle_size: float,
) -> list[dict[str, int]]:
    """Return count mismatches instead of relying only on console output."""

    mismatches: list[dict[str, int]] = []
    for index, result in enumerate(results):
        actual = _point_count(
            coordinates,
            float(result["left_x"]),
            float(result["bottom_y"]),
            rectangle_size,
        )
        recorded = int(result["pt_count"])
        if actual != recorded:
            mismatches.append({"index": index, "recorded": recorded, "actual": actual})
    return mismatches


def choose_result(results: list[dict[str, Any]], one_based_index: int) -> list[dict[str, Any]]:
    """Select one cached/scanned result without opening a desktop GUI."""

    if one_based_index < 1 or one_based_index > len(results):
        raise ValueError(
            f"--select-index must be between 1 and {len(results)}, got {one_based_index}"
        )
    return [results[one_based_index - 1]]


def candidate_to_legacy(
    candidate: Candidate,
    rectangle_size: float,
) -> dict[str, Any]:
    """Convert one row-sweep candidate to the exact legacy result mapping."""

    return {
        "geometry": box(
            candidate.left_x,
            candidate.bottom_y,
            candidate.left_x + rectangle_size,
            candidate.bottom_y + rectangle_size,
        ),
        "pt_count": candidate.point_count,
        "left_x": candidate.left_x,
        "bottom_y": candidate.bottom_y,
        "center_x": candidate.center_x,
        "center_y": candidate.center_y,
    }


def verify_results(results, coordinates, rectangle_size):
    """Compatibility wrapper that raises when a cached result is inconsistent."""

    mismatches = validate_results(results, coordinates, rectangle_size)
    if mismatches:
        raise ValueError(f"Rectangle count verification failed: {mismatches}")
    return mismatches
