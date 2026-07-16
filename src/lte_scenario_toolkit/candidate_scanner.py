"""Memory-bounded candidate-grid axes and exact row counting."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np


def _positive_finite_dimension(value: Any, *, field: str) -> float:
    message = f"{field} must be a finite number greater than zero"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(message)
    return result


def grid_axes(
    boundary: Any,
    rectangle_size: float,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return separate scan axes using NumPy's exclusive-stop semantics."""

    rectangle_size = _positive_finite_dimension(
        rectangle_size,
        field="rectangle_size",
    )
    step = _positive_finite_dimension(step, field="step")
    min_x, min_y, max_x, max_y = boundary.bounds
    return (
        np.arange(min_x, max_x - rectangle_size, step, dtype=float),
        np.arange(min_y, max_y - rectangle_size, step, dtype=float),
    )


def count_row(
    coordinates: np.ndarray,
    x_origins: np.ndarray,
    *,
    y: float,
    rectangle_size: float,
) -> np.ndarray:
    """Count inclusive rectangle hits for one y row without a Cartesian grid."""

    points = np.asarray(coordinates)
    origins = np.asarray(x_origins)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("coordinates must be an N-by-at-least-2 array")
    if origins.ndim != 1:
        raise ValueError("x_origins must be a one-dimensional array")
    rectangle_size = _positive_finite_dimension(
        rectangle_size,
        field="rectangle_size",
    )
    if points.size == 0:
        return np.zeros(origins.shape, dtype=np.int64)

    y_values = points[:, 1]
    active_x = np.sort(
        points[
            (y_values >= y) & (y_values <= y + rectangle_size),
            0,
        ]
    )
    left = np.searchsorted(active_x, origins, side="left")
    right = np.searchsorted(
        active_x,
        origins + rectangle_size,
        side="right",
    )
    return (right - left).astype(np.int64, copy=False)
