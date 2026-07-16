"""Memory-bounded candidate-grid axes and exact row counting."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter
from typing import Any, Protocol

import numpy as np
import shapely


class _CancellationSignal(Protocol):
    def is_set(self) -> bool: ...


@dataclass(frozen=True)
class ScanRequest:
    rectangle_size: float
    target_count: int
    tolerance: int
    step: float
    max_candidates: int
    minimum_spacing: float
    strategy: str
    mode: str
    random_seed: int
    algorithm_version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rectangle_size",
            _positive_finite_dimension(
                self.rectangle_size,
                field="rectangle_size",
            ),
        )
        object.__setattr__(self, "step", _positive_finite_dimension(self.step, field="step"))
        object.__setattr__(
            self,
            "minimum_spacing",
            _positive_finite_dimension(
                self.minimum_spacing,
                field="minimum_spacing",
            ),
        )
        object.__setattr__(
            self,
            "target_count",
            _integer_value(
                self.target_count,
                field="target_count",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "tolerance",
            _integer_value(
                self.tolerance,
                field="tolerance",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "max_candidates",
            _integer_value(
                self.max_candidates,
                field="max_candidates",
                minimum=1,
            ),
        )
        if self.strategy not in {"sequential", "uniform"}:
            raise ValueError("strategy must be one of: sequential, uniform")
        if self.mode not in {"fast", "complete"}:
            raise ValueError("mode must be one of: complete, fast")
        object.__setattr__(
            self,
            "random_seed",
            _integer_value(self.random_seed, field="random_seed"),
        )
        if type(self.algorithm_version) is not str or not self.algorithm_version.strip():
            raise ValueError("algorithm_version must be a non-empty string")


@dataclass(frozen=True)
class Candidate:
    flat_grid_id: int
    point_count: int
    left_x: float
    bottom_y: float
    center_x: float
    center_y: float


@dataclass(frozen=True)
class ScanProgress:
    phase: str
    checked_positions: int
    total_positions: int
    candidate_count: int
    elapsed_seconds: float
    added_candidates: tuple[Candidate, ...] = ()
    removed_flat_grid_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ScanResult:
    candidates: tuple[Candidate, ...]
    checked_positions: int
    total_positions: int
    completed: bool
    algorithm_version: str


class ScanCancelled(RuntimeError):
    """Raised instead of returning a partial result when cancellation is requested."""

    code = "scan.cancelled"

    def __init__(self, message: str = "Candidate scan was cancelled") -> None:
        super().__init__(message)
        self.details: dict[str, Any] = {}


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


def _integer_value(
    value: Any,
    *,
    field: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        qualifier = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{field} must be {qualifier}")
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


class _SpacingIndex:
    def __init__(self, minimum_spacing: float) -> None:
        self.minimum_spacing = minimum_spacing
        self.cell_size = minimum_spacing / math.sqrt(2.0)
        self.buckets: dict[tuple[int, int], list[Candidate]] = {}

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            math.floor(x / self.cell_size),
            math.floor(y / self.cell_size),
        )

    def accepts(self, candidate: Candidate) -> bool:
        cell_x, cell_y = self._cell(candidate.center_x, candidate.center_y)
        for offset_x in range(-2, 3):
            for offset_y in range(-2, 3):
                for selected in self.buckets.get(
                    (cell_x + offset_x, cell_y + offset_y),
                    (),
                ):
                    if (
                        math.hypot(
                            selected.center_x - candidate.center_x,
                            selected.center_y - candidate.center_y,
                        )
                        < self.minimum_spacing
                    ):
                        return False
        self.buckets.setdefault((cell_x, cell_y), []).append(candidate)
        return True


def _priority(seed: int, flat_grid_id: int) -> int:
    payload = f"{seed}:{flat_grid_id}".encode("ascii")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        "big",
    )


def _rank(seed: int, candidate: Candidate) -> tuple[int, int]:
    return (
        _priority(seed, candidate.flat_grid_id),
        candidate.flat_grid_id,
    )


class _CompleteSelection:
    def __init__(
        self,
        *,
        random_seed: int,
        maximum_candidates: int,
        minimum_spacing: float,
    ) -> None:
        self.random_seed = random_seed
        self.maximum_candidates = maximum_candidates
        self.minimum_spacing = minimum_spacing
        self.selected: list[tuple[tuple[int, int], Candidate]] = []

    def consider(self, candidate: Candidate) -> tuple[bool, tuple[int, ...]]:
        candidate_rank = _rank(self.random_seed, candidate)
        conflicts = [
            ranked
            for ranked in self.selected
            if math.hypot(
                ranked[1].center_x - candidate.center_x,
                ranked[1].center_y - candidate.center_y,
            )
            < self.minimum_spacing
        ]
        if conflicts:
            if not all(candidate_rank < rank for rank, _ in conflicts):
                return False, ()
            conflict_ids = {selected.flat_grid_id for _, selected in conflicts}
            removed = tuple(
                selected.flat_grid_id
                for _, selected in self.selected
                if selected.flat_grid_id in conflict_ids
            )
            self.selected = [
                ranked
                for ranked in self.selected
                if ranked[1].flat_grid_id not in conflict_ids
            ]
        elif len(self.selected) < self.maximum_candidates:
            removed = ()
        else:
            worst = max(self.selected, key=lambda ranked: ranked[0])
            if candidate_rank >= worst[0]:
                return False, ()
            self.selected.remove(worst)
            removed = (worst[1].flat_grid_id,)

        self.selected.append((candidate_rank, candidate))
        return True, removed

    def candidates(self) -> tuple[Candidate, ...]:
        return tuple(
            candidate
            for _, candidate in sorted(
                self.selected,
                key=lambda ranked: ranked[0],
            )
        )

    def __len__(self) -> int:
        return len(self.selected)


def _cancelled(cancel: _CancellationSignal | None) -> bool:
    return cancel is not None and cancel.is_set()


def _raise_if_cancelled(cancel: _CancellationSignal | None) -> None:
    if _cancelled(cancel):
        raise ScanCancelled("Candidate scan was cancelled")


def _validate_scan_inputs(
    boundary: Any,
    coordinates: Any,
    progress: Callable[[ScanProgress], None] | None,
    cancel: _CancellationSignal | None,
) -> np.ndarray:
    points = np.asarray(coordinates)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("coordinates must be an N-by-at-least-2 array")
    if not np.issubdtype(points.dtype, np.number) or np.issubdtype(
        points.dtype,
        np.complexfloating,
    ):
        raise ValueError("coordinates must contain real numeric values")
    if not bool(np.isfinite(points[:, :2]).all()):
        raise ValueError("coordinates must contain finite real values")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    if cancel is not None and not callable(getattr(cancel, "is_set", None)):
        raise ValueError("cancel must provide an is_set() method")
    try:
        bounds = tuple(boundary.bounds)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("boundary must provide four numeric bounds") from exc
    if len(bounds) != 4:
        raise ValueError("boundary must provide four numeric bounds")
    try:
        finite_bounds = all(math.isfinite(float(value)) for value in bounds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("boundary bounds must be finite numbers") from exc
    if not finite_bounds:
        raise ValueError("boundary bounds must be finite numbers")
    return points


def _axis_orders(
    request: ScanRequest,
    x_count: int,
    y_count: int,
) -> tuple[range | np.ndarray, range | np.ndarray]:
    if request.strategy == "sequential":
        return range(x_count), range(y_count)
    generator = np.random.default_rng(request.random_seed % (2**64))
    return generator.permutation(x_count), generator.permutation(y_count)


def _strict_containment_by_x_index(
    boundary: Any,
    x_origins: np.ndarray,
    x_indices: list[int],
    y: float,
    rectangle_size: float,
) -> dict[int, bool]:
    if not x_indices:
        return {}
    index_array = np.asarray(x_indices, dtype=np.intp)
    left_values = x_origins[index_array]
    rectangles = shapely.box(
        left_values,
        y,
        left_values + rectangle_size,
        y + rectangle_size,
    )
    contained = np.asarray(shapely.contains(boundary, rectangles), dtype=bool)
    touches_boundary = np.asarray(
        shapely.intersects(boundary.boundary, rectangles),
        dtype=bool,
    )
    strict = np.atleast_1d(contained & ~touches_boundary)
    return {
        x_index: bool(is_inside)
        for x_index, is_inside in zip(x_indices, strict, strict=True)
    }


def scan_candidates(
    request: ScanRequest,
    boundary: Any,
    coordinates: np.ndarray,
    progress: Callable[[ScanProgress], None] | None = None,
    cancel: _CancellationSignal | None = None,
) -> ScanResult:
    """Run cancellable fast or bounded-complete scans without Cartesian positions."""

    if not isinstance(request, ScanRequest):
        raise ValueError("request must be a ScanRequest")
    points = _validate_scan_inputs(boundary, coordinates, progress, cancel)
    started_at = perf_counter()
    _raise_if_cancelled(cancel)
    x_origins, y_origins = grid_axes(
        boundary,
        request.rectangle_size,
        request.step,
    )
    total_positions = int(len(x_origins) * len(y_origins))
    if total_positions == 0:
        return ScanResult(
            candidates=(),
            checked_positions=0,
            total_positions=0,
            completed=True,
            algorithm_version=request.algorithm_version,
        )

    x_order, y_order = _axis_orders(
        request,
        len(x_origins),
        len(y_origins),
    )
    target_minimum = request.target_count - request.tolerance
    target_maximum = request.target_count + request.tolerance
    fast_selected: list[Candidate] = []
    fast_spacing = _SpacingIndex(request.minimum_spacing)
    complete_selection = (
        _CompleteSelection(
            random_seed=request.random_seed,
            maximum_candidates=request.max_candidates,
            minimum_spacing=request.minimum_spacing,
        )
        if request.mode == "complete"
        else None
    )
    checked_positions = 0

    for traversal_y_index, original_y_index_value in enumerate(y_order):
        _raise_if_cancelled(cancel)
        original_y_index = int(original_y_index_value)
        y = float(y_origins[original_y_index])
        counts = count_row(
            points,
            x_origins,
            y=y,
            rectangle_size=request.rectangle_size,
        )
        _raise_if_cancelled(cancel)
        matching_x_indices = [
            int(index)
            for index in x_order
            if target_minimum <= int(counts[int(index)]) <= target_maximum
        ]
        strict_containment = _strict_containment_by_x_index(
            boundary,
            x_origins,
            matching_x_indices,
            y,
            request.rectangle_size,
        )
        added_this_row: list[Candidate] = []
        removed_this_row: list[int] = []

        for original_x_index_value in x_order:
            original_x_index = int(original_x_index_value)
            checked_positions += 1
            point_count = int(counts[original_x_index])
            if not target_minimum <= point_count <= target_maximum:
                continue
            if not strict_containment[original_x_index]:
                continue
            left_x = float(x_origins[original_x_index])
            candidate = Candidate(
                flat_grid_id=(
                    original_y_index * len(x_origins) + original_x_index
                ),
                point_count=point_count,
                left_x=left_x,
                bottom_y=y,
                center_x=left_x + request.rectangle_size / 2.0,
                center_y=y + request.rectangle_size / 2.0,
            )
            if complete_selection is not None:
                accepted, removed_ids = complete_selection.consider(candidate)
                if not accepted:
                    continue
                added_this_row.append(candidate)
                removed_this_row.extend(removed_ids)
                continue
            if not fast_spacing.accepts(candidate):
                continue
            fast_selected.append(candidate)
            added_this_row.append(candidate)
            if len(fast_selected) >= request.max_candidates:
                _raise_if_cancelled(cancel)
                event = ScanProgress(
                    phase="completed",
                    checked_positions=checked_positions,
                    total_positions=total_positions,
                    candidate_count=len(fast_selected),
                    elapsed_seconds=perf_counter() - started_at,
                    added_candidates=tuple(added_this_row),
                )
                if progress is not None:
                    progress(event)
                return ScanResult(
                    candidates=tuple(fast_selected),
                    checked_positions=checked_positions,
                    total_positions=total_positions,
                    completed=True,
                    algorithm_version=request.algorithm_version,
                )

        final_row = traversal_y_index == len(y_origins) - 1
        if final_row:
            _raise_if_cancelled(cancel)
        event = ScanProgress(
            phase="completed" if final_row else "scanning",
            checked_positions=checked_positions,
            total_positions=total_positions,
            candidate_count=(
                len(complete_selection)
                if complete_selection is not None
                else len(fast_selected)
            ),
            elapsed_seconds=perf_counter() - started_at,
            added_candidates=tuple(added_this_row),
            removed_flat_grid_ids=tuple(removed_this_row),
        )
        if progress is not None:
            progress(event)
        if not final_row:
            _raise_if_cancelled(cancel)

    return ScanResult(
        candidates=(
            complete_selection.candidates()
            if complete_selection is not None
            else tuple(fast_selected)
        ),
        checked_positions=checked_positions,
        total_positions=total_positions,
        completed=True,
        algorithm_version=request.algorithm_version,
    )
