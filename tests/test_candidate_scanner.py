import json
from dataclasses import FrozenInstanceError, fields, replace
from itertools import combinations
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box

import lte_scenario_toolkit.candidate_scanner as scanner_module
from lte_scenario_toolkit.candidate_scanner import (
    Candidate,
    ScanCancelled,
    ScanProgress,
    ScanRequest,
    ScanResult,
    count_row,
    grid_axes,
    scan_candidates,
)


def make_request(**overrides):
    values = {
        "rectangle_size": 4,
        "target_count": 2,
        "tolerance": 0,
        "step": 1,
        "max_candidates": 2,
        "minimum_spacing": 1,
        "strategy": "sequential",
        "mode": "fast",
        "random_seed": 7,
        "algorithm_version": "row-sweep-v1",
    }
    values.update(overrides)
    return ScanRequest(**values)


def brute_counts(points, xs, y, size):
    return np.asarray(
        [
            np.count_nonzero(
                (points[:, 0] >= x)
                & (points[:, 0] <= x + size)
                & (points[:, 1] >= y)
                & (points[:, 1] <= y + size)
            )
            for x in xs
        ]
    )


def test_count_row_includes_points_on_all_rectangle_edges():
    coordinates = np.asarray(
        [[0, 0], [2, 2], [4, 4], [2, 0], [4, 2]],
        dtype=float,
    )
    x_origins = np.asarray([0, 1, 2], dtype=float)

    actual = count_row(
        coordinates,
        x_origins,
        y=0,
        rectangle_size=2,
    )

    np.testing.assert_array_equal(
        actual,
        brute_counts(coordinates, x_origins, 0, 2),
    )
    assert actual.dtype == np.int64


def test_grid_axes_returns_separate_one_dimensional_axes_without_meshgrid(
    monkeypatch,
):
    def reject_meshgrid(*args, **kwargs):
        del args, kwargs
        raise AssertionError("grid_axes must not allocate Cartesian positions")

    monkeypatch.setattr(np, "meshgrid", reject_meshgrid)

    x_origins, y_origins = grid_axes(
        box(0, 0, 11, 11),
        rectangle_size=4,
        step=3,
    )

    np.testing.assert_array_equal(x_origins, np.asarray([0.0, 3.0, 6.0]))
    np.testing.assert_array_equal(y_origins, np.asarray([0.0, 3.0, 6.0]))
    assert x_origins.ndim == 1
    assert y_origins.ndim == 1


def test_count_row_matches_seeded_randomized_brute_force_oracle():
    generator = np.random.default_rng(20260716)
    coordinates = generator.integers(-20, 61, size=(200, 2))
    x_origins = generator.integers(-20, 51, size=25)
    y_origins = generator.integers(-20, 51, size=5)
    rectangle_size = 9

    for y_origin in y_origins:
        np.testing.assert_array_equal(
            count_row(
                coordinates,
                x_origins,
                y=y_origin,
                rectangle_size=rectangle_size,
            ),
            brute_counts(
                coordinates,
                x_origins,
                y_origin,
                rectangle_size,
            ),
        )


def test_count_row_empty_coordinates_returns_int64_zeros_with_origin_shape():
    x_origins = np.asarray([-1.5, 0.0, 4.5, 9.0])

    counts = count_row(
        np.empty((0, 2)),
        x_origins,
        y=3,
        rectangle_size=2,
    )

    np.testing.assert_array_equal(counts, np.zeros(x_origins.shape, dtype=np.int64))
    assert counts.dtype == np.int64


@pytest.mark.parametrize(
    "coordinates",
    [
        np.asarray([1.0, 2.0]),
        np.ones((2, 1)),
        np.ones((1, 2, 1)),
    ],
)
def test_count_row_rejects_coordinates_that_are_not_n_by_at_least_two(
    coordinates,
):
    with pytest.raises(ValueError, match="coordinates"):
        count_row(
            coordinates,
            np.asarray([0.0]),
            y=0,
            rectangle_size=1,
        )


def test_count_row_rejects_non_one_dimensional_x_origins():
    with pytest.raises(ValueError, match="x_origins"):
        count_row(
            np.asarray([[0.0, 0.0]]),
            np.asarray([[0.0, 1.0]]),
            y=0,
            rectangle_size=1,
        )


@pytest.mark.parametrize("rectangle_size", [0, -1])
def test_count_row_rejects_non_positive_rectangle_size(rectangle_size):
    with pytest.raises(ValueError, match="rectangle_size"):
        count_row(
            np.asarray([[0.0, 0.0]]),
            np.asarray([0.0]),
            y=0,
            rectangle_size=rectangle_size,
        )


@pytest.mark.parametrize(
    "rectangle_size",
    [np.nan, np.inf, -np.inf, True, np.bool_(True), "1"],
)
def test_count_row_rejects_non_finite_boolean_and_non_numeric_size(
    rectangle_size,
):
    with pytest.raises(ValueError, match="rectangle_size"):
        count_row(
            np.asarray([[0.0, 0.0]]),
            np.asarray([0.0]),
            y=0,
            rectangle_size=rectangle_size,
        )


@pytest.mark.parametrize(
    ("rectangle_size", "step", "field"),
    [(0, 1, "rectangle_size"), (-1, 1, "rectangle_size"), (1, 0, "step"), (1, -1, "step")],
)
def test_grid_axes_rejects_non_positive_parameters(rectangle_size, step, field):
    with pytest.raises(ValueError, match=field):
        grid_axes(
            box(0, 0, 10, 10),
            rectangle_size=rectangle_size,
            step=step,
        )


@pytest.mark.parametrize("field", ["rectangle_size", "step"])
@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf, True, np.bool_(True), "1"],
)
def test_grid_axes_rejects_non_finite_boolean_and_non_numeric_dimensions(
    field,
    invalid_value,
):
    dimensions = {"rectangle_size": 1, "step": 1}
    dimensions[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        grid_axes(box(0, 0, 10, 10), **dimensions)


def test_grid_axes_excludes_origins_whose_window_touches_maximum_boundary():
    x_origins, y_origins = grid_axes(
        box(1, 2, 5, 6),
        rectangle_size=4,
        step=1,
    )

    np.testing.assert_array_equal(x_origins, np.asarray([], dtype=float))
    np.testing.assert_array_equal(y_origins, np.asarray([], dtype=float))


def test_grid_axes_uses_numpy_arange_exclusive_stop_for_float_steps():
    x_origins, y_origins = grid_axes(
        box(0, 0, 1, 1),
        rectangle_size=0.25,
        step=0.2,
    )

    expected = np.asarray([0.0, 0.2, 0.4, 0.6], dtype=float)
    np.testing.assert_allclose(x_origins, expected)
    np.testing.assert_allclose(y_origins, expected)
    assert x_origins.dtype == float
    assert y_origins.dtype == float


def test_scan_models_are_frozen_with_the_exact_public_fields():
    expected_fields = {
        ScanRequest: (
            "rectangle_size",
            "target_count",
            "tolerance",
            "step",
            "max_candidates",
            "minimum_spacing",
            "strategy",
            "mode",
            "random_seed",
            "algorithm_version",
        ),
        Candidate: (
            "flat_grid_id",
            "point_count",
            "left_x",
            "bottom_y",
            "center_x",
            "center_y",
        ),
        ScanProgress: (
            "phase",
            "checked_positions",
            "total_positions",
            "candidate_count",
            "elapsed_seconds",
            "added_candidates",
            "removed_flat_grid_ids",
        ),
        ScanResult: (
            "candidates",
            "checked_positions",
            "total_positions",
            "completed",
            "algorithm_version",
        ),
    }
    for model, expected in expected_fields.items():
        assert tuple(field.name for field in fields(model)) == expected

    request = make_request()
    assert request == ScanRequest(4, 2, 0, 1, 2, 1, "sequential", "fast", 7, "row-sweep-v1")
    with pytest.raises(FrozenInstanceError):
        request.step = 2


def test_fast_scan_returns_early_and_reports_monotonic_progress():
    coordinates = np.asarray([[1, 1], [2, 2], [5, 1], [6, 2]], dtype=float)
    boundary = box(-1, -1, 10, 8)
    progress_events = []

    result = scan_candidates(
        make_request(),
        boundary,
        coordinates,
        progress=progress_events.append,
    )

    assert result.completed is True
    assert len(result.candidates) == 2
    assert result.checked_positions < result.total_positions
    assert result.algorithm_version == "row-sweep-v1"
    assert [event.checked_positions for event in progress_events] == sorted(
        event.checked_positions for event in progress_events
    )
    assert progress_events[-1].phase == "completed"
    assert progress_events[-1].candidate_count == len(result.candidates)
    assert progress_events[-1].checked_positions == result.checked_positions
    assert all(event.total_positions == result.total_positions for event in progress_events)
    assert all(event.elapsed_seconds >= 0 for event in progress_events)
    assert all(event.removed_flat_grid_ids == () for event in progress_events)
    assert tuple(
        candidate
        for event in progress_events
        for candidate in event.added_candidates
    ) == result.candidates


def test_uniform_scan_is_seeded_and_reproducible():
    coordinates = np.asarray(
        [(x + 0.5, y + 0.5) for x in range(8) for y in range(8)],
        dtype=float,
    )
    request = make_request(
        target_count=16,
        max_candidates=6,
        minimum_spacing=0.5,
        strategy="uniform",
    )
    boundary = box(-1, -1, 9, 9)

    first = scan_candidates(request, boundary, coordinates)
    second = scan_candidates(request, boundary, coordinates)

    assert first.candidates == second.candidates
    assert len(first.candidates) == request.max_candidates


def test_scan_raises_when_cancel_is_already_set():
    cancel = Event()
    cancel.set()

    with pytest.raises(ScanCancelled):
        scan_candidates(
            make_request(),
            box(-1, -1, 10, 8),
            np.asarray([[1.0, 1.0]]),
            cancel=cancel,
        )


def test_progress_callback_can_cancel_after_one_completed_row():
    cancel = Event()
    progress_events = []

    def cancel_after_first_row(event):
        progress_events.append(event)
        cancel.set()

    with pytest.raises(ScanCancelled):
        scan_candidates(
            make_request(target_count=999),
            box(-1, -1, 10, 8),
            np.asarray([[1.0, 1.0]]),
            progress=cancel_after_first_row,
            cancel=cancel,
        )

    assert len(progress_events) == 1
    assert progress_events[0].phase == "scanning"
    assert progress_events[0].checked_positions > 0
    assert all(event.phase != "completed" for event in progress_events)


def test_fast_limit_completed_progress_is_terminal_before_callback_cancel():
    cancel = Event()
    progress_events = []

    def cancel_on_completed(event):
        progress_events.append(event)
        if event.phase == "completed":
            cancel.set()

    result = scan_candidates(
        make_request(),
        box(-1, -1, 10, 8),
        np.asarray([[1, 1], [2, 2], [5, 1], [6, 2]], dtype=float),
        progress=cancel_on_completed,
        cancel=cancel,
    )

    assert result.completed is True
    assert len(result.candidates) == 2
    assert cancel.is_set()
    assert sum(event.phase == "completed" for event in progress_events) == 1
    assert progress_events[-1].phase == "completed"


def test_exhausted_scan_completed_progress_is_terminal_before_callback_cancel():
    cancel = Event()
    progress_events = []

    def cancel_on_completed(event):
        progress_events.append(event)
        if event.phase == "completed":
            cancel.set()

    result = scan_candidates(
        make_request(target_count=999),
        box(-1, -1, 10, 8),
        np.asarray([[1.0, 1.0]]),
        progress=cancel_on_completed,
        cancel=cancel,
    )

    assert result.completed is True
    assert result.candidates == ()
    assert result.checked_positions == result.total_positions
    assert cancel.is_set()
    assert sum(event.phase == "completed" for event in progress_events) == 1
    assert progress_events[-1].phase == "completed"


def test_uniform_flat_grid_id_uses_original_axis_indices_and_coordinates():
    coordinates = np.asarray([[3.5, 1.5], [4.0, 2.0]])
    request = make_request(
        rectangle_size=2,
        target_count=2,
        step=2,
        max_candidates=10,
        minimum_spacing=0.5,
        strategy="uniform",
    )
    boundary = box(-1, -1, 8, 8)
    x_origins, _ = grid_axes(boundary, request.rectangle_size, request.step)

    result = scan_candidates(request, boundary, coordinates)

    assert result.candidates == (
        Candidate(
            flat_grid_id=1 * len(x_origins) + 2,
            point_count=2,
            left_x=3.0,
            bottom_y=1.0,
            center_x=4.0,
            center_y=2.0,
        ),
    )


def test_scan_count_tolerance_accepts_both_edges_only():
    coordinates = np.asarray(
        [
            [2.5, 2.5],
            [5.2, 2.5],
            [5.5, 2.5],
            [5.8, 2.5],
            [11.1, 2.5],
            [11.3, 2.5],
            [11.5, 2.5],
            [11.7, 2.5],
        ]
    )

    result = scan_candidates(
        make_request(
            rectangle_size=2,
            target_count=2,
            tolerance=1,
            step=3,
            max_candidates=10,
            minimum_spacing=1,
        ),
        box(-1, -1, 20, 5),
        coordinates,
    )

    assert {candidate.point_count for candidate in result.candidates} == {1, 3}


def test_scan_rejects_rectangles_that_touch_boundary():
    result = scan_candidates(
        make_request(
            rectangle_size=2,
            target_count=1,
            max_candidates=10,
        ),
        box(0, 0, 5, 5),
        np.asarray([[0.5, 0.5]]),
    )

    assert result.candidates == ()
    assert result.completed is True
    assert result.checked_positions == result.total_positions


def test_minimum_spacing_rejects_below_threshold_and_allows_exact_distance():
    coordinates = np.asarray([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]])

    result = scan_candidates(
        make_request(
            rectangle_size=1,
            target_count=1,
            max_candidates=10,
            minimum_spacing=2,
        ),
        box(-1, -1, 6, 3),
        coordinates,
    )

    assert [(item.center_x, item.center_y) for item in result.candidates] == [
        (0.5, 0.5),
        (2.5, 0.5),
    ]


def test_random_scan_results_match_brute_minimum_spacing_constraint():
    generator = np.random.default_rng(105)
    coordinates = generator.uniform(0.1, 11.9, size=(120, 2))
    request = make_request(
        rectangle_size=2,
        target_count=3,
        tolerance=3,
        max_candidates=30,
        minimum_spacing=2.3,
        strategy="uniform",
    )

    result = scan_candidates(request, box(0, 0, 12, 12), coordinates)

    for left, right in combinations(result.candidates, 2):
        assert np.hypot(left.center_x - right.center_x, left.center_y - right.center_y) >= (
            request.minimum_spacing
        )


def test_progress_events_are_frozen_and_only_report_new_row_candidates():
    events = []
    result = scan_candidates(
        make_request(max_candidates=2),
        box(-1, -1, 10, 8),
        np.asarray([[1, 1], [2, 2], [5, 1], [6, 2]], dtype=float),
        progress=events.append,
    )

    assert all(isinstance(event, ScanProgress) for event in events)
    assert all(event.phase in {"scanning", "completed"} for event in events)
    assert sum(len(event.added_candidates) for event in events) == len(result.candidates)
    assert len({item.flat_grid_id for event in events for item in event.added_candidates}) == len(
        result.candidates
    )
    with pytest.raises(FrozenInstanceError):
        events[0].phase = "changed"


def test_empty_axes_and_no_matches_return_completed_empty_results():
    empty_axes = scan_candidates(
        make_request(rectangle_size=4),
        box(0, 0, 4, 4),
        np.empty((0, 2)),
    )
    no_matches = scan_candidates(
        make_request(target_count=99),
        box(-1, -1, 10, 8),
        np.asarray([[1.0, 1.0]]),
    )

    assert empty_axes == ScanResult((), 0, 0, True, "row-sweep-v1")
    assert no_matches.candidates == ()
    assert no_matches.completed is True
    assert no_matches.checked_positions == no_matches.total_positions


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("rectangle_size", 0),
        ("rectangle_size", np.nan),
        ("step", np.inf),
        ("step", True),
        ("minimum_spacing", -1),
        ("minimum_spacing", np.nan),
        ("target_count", -1),
        ("target_count", 1.5),
        ("target_count", True),
        ("tolerance", -1),
        ("tolerance", np.nan),
        ("max_candidates", 0),
        ("max_candidates", 1.0),
        ("strategy", "random"),
        ("mode", "ranked"),
        ("random_seed", True),
        ("random_seed", 1.5),
        ("algorithm_version", ""),
        ("algorithm_version", 1),
    ],
)
def test_scan_request_rejects_invalid_values(field, invalid_value):
    with pytest.raises(ValueError, match=field):
        replace(make_request(), **{field: invalid_value})


def test_scan_validates_coordinates_progress_cancel_and_boundary_inputs():
    request = make_request()
    boundary = box(-1, -1, 10, 8)

    with pytest.raises(ValueError, match="coordinates"):
        scan_candidates(request, boundary, np.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="progress"):
        scan_candidates(request, boundary, np.empty((0, 2)), progress=object())
    with pytest.raises(ValueError, match="cancel"):
        scan_candidates(request, boundary, np.empty((0, 2)), cancel=object())
    with pytest.raises(ValueError, match="boundary"):
        scan_candidates(request, object(), np.empty((0, 2)))


@pytest.mark.parametrize(
    "coordinates",
    [
        np.asarray([[np.nan, 1.0]]),
        np.asarray([[1.0, np.inf]]),
        np.asarray([[-np.inf, 1.0]]),
        np.asarray([[1.0 + 0.0j, 2.0 + 0.0j]]),
    ],
)
def test_scan_rejects_non_finite_and_complex_real_world_coordinates(coordinates):
    with pytest.raises(ValueError, match="coordinates|finite|real"):
        scan_candidates(
            make_request(target_count=0),
            box(-1, -1, 10, 8),
            coordinates,
        )


def test_scan_accepts_empty_real_coordinate_array_on_non_empty_grid():
    result = scan_candidates(
        make_request(target_count=0, max_candidates=1),
        box(-1, -1, 10, 8),
        np.empty((0, 2), dtype=float),
    )

    assert result.completed is True
    assert len(result.candidates) == 1


def test_complete_uniform_scan_is_deterministic_and_checks_every_position():
    coordinates = np.asarray(
        [(x, y) for x in range(8) for y in range(8)],
        dtype=float,
    )
    request = make_request(
        rectangle_size=3,
        target_count=16,
        mode="complete",
        strategy="uniform",
        max_candidates=3,
        minimum_spacing=1,
    )
    boundary = box(-1, -1, 9, 9)

    first = scan_candidates(request, boundary, coordinates)
    second = scan_candidates(request, boundary, coordinates)

    assert first == second
    assert first.completed is True
    assert first.checked_positions == first.total_positions
    assert len(first.candidates) <= request.max_candidates


def test_complete_scan_enforces_minimum_spacing_on_ranked_result():
    coordinates = np.asarray(
        [(x, y) for x in range(12) for y in range(12)],
        dtype=float,
    )
    request = make_request(
        rectangle_size=3,
        target_count=16,
        mode="complete",
        max_candidates=10,
        minimum_spacing=4,
    )

    result = scan_candidates(request, box(-1, -1, 13, 13), coordinates)

    assert result.checked_positions == result.total_positions
    for left, right in combinations(result.candidates, 2):
        assert np.hypot(left.center_x - right.center_x, left.center_y - right.center_y) >= 4


def test_complete_rank_ties_select_smallest_strictly_contained_flat_id(
    monkeypatch,
):
    monkeypatch.setattr(scanner_module, "_priority", lambda seed, flat_grid_id: 5)
    request = make_request(
        target_count=0,
        mode="complete",
        strategy="uniform",
        max_candidates=1,
    )

    result = scan_candidates(
        request,
        box(0, 0, 10, 10),
        np.empty((0, 2), dtype=float),
    )

    assert result.checked_positions == result.total_positions
    assert result.candidates[0].flat_grid_id == 7


def test_complete_progress_replays_provisional_selection_and_reaches_total():
    events = []
    request = make_request(
        target_count=0,
        mode="complete",
        strategy="uniform",
        max_candidates=2,
        minimum_spacing=1,
    )

    result = scan_candidates(
        request,
        box(0, 0, 10, 10),
        np.empty((0, 2), dtype=float),
        progress=events.append,
    )

    replayed = {}
    for event in events:
        for candidate in event.added_candidates:
            replayed[candidate.flat_grid_id] = candidate
        for flat_grid_id in event.removed_flat_grid_ids:
            replayed.pop(flat_grid_id, None)
        assert len(replayed) == event.candidate_count
        assert event.candidate_count <= request.max_candidates
    assert events[-1].phase == "completed"
    assert events[-1].checked_positions == result.total_positions
    assert result.checked_positions == result.total_positions
    assert set(replayed) == {candidate.flat_grid_id for candidate in result.candidates}
    assert any(event.removed_flat_grid_ids for event in events)


def test_complete_capacity_replaces_worst_with_later_better_rank(monkeypatch):
    priorities = {9: 50, 10: 40, 11: 10, 12: 100}
    monkeypatch.setattr(
        scanner_module,
        "_priority",
        lambda seed, flat_grid_id: priorities.get(flat_grid_id, 1000 + flat_grid_id),
    )
    events = []
    request = make_request(
        rectangle_size=2,
        target_count=0,
        mode="complete",
        max_candidates=2,
        minimum_spacing=0.5,
    )

    result = scan_candidates(
        request,
        box(0, 0, 10, 6),
        np.empty((0, 2), dtype=float),
        progress=events.append,
    )

    final_ids = {candidate.flat_grid_id for candidate in result.candidates}
    assert 11 in final_ids
    assert 9 not in final_ids
    assert any(
        11 in {item.flat_grid_id for item in event.added_candidates}
        and 9 in event.removed_flat_grid_ids
        for event in events
    )


@pytest.mark.parametrize(
    ("new_priority", "replacement_expected"),
    [(10, True), (25, False)],
)
def test_complete_spacing_conflicts_require_better_rank_than_every_conflict(
    monkeypatch,
    new_priority,
    replacement_expected,
):
    priorities = {6: 20, 7: 100, 8: 30, 11: 100, 12: new_priority}
    monkeypatch.setattr(
        scanner_module,
        "_priority",
        lambda seed, flat_grid_id: priorities.get(flat_grid_id, 1000 + flat_grid_id),
    )
    events = []
    request = make_request(
        rectangle_size=1,
        target_count=0,
        mode="complete",
        max_candidates=2,
        minimum_spacing=2,
    )

    result = scan_candidates(
        request,
        box(0, 0, 6, 6),
        np.empty((0, 2), dtype=float),
        progress=events.append,
    )

    added_ids = {
        candidate.flat_grid_id
        for event in events
        for candidate in event.added_candidates
    }
    removed_ids = {
        flat_grid_id
        for event in events
        for flat_grid_id in event.removed_flat_grid_ids
    }
    final_ids = {candidate.flat_grid_id for candidate in result.candidates}
    if replacement_expected:
        assert 12 in added_ids
        assert {6, 8}.issubset(removed_ids)
        assert 12 in final_ids
        assert not {6, 8} & final_ids
    else:
        assert 12 not in added_ids
        assert not {6, 8} & removed_ids
        assert {6, 8}.issubset(final_ids)


def test_complete_result_is_rank_sorted_and_repeatable(monkeypatch):
    monkeypatch.setattr(
        scanner_module,
        "_priority",
        lambda seed, flat_grid_id: (flat_grid_id * 17) % 31,
    )
    request = make_request(
        target_count=0,
        mode="complete",
        strategy="uniform",
        max_candidates=5,
        minimum_spacing=1,
    )
    boundary = box(0, 0, 10, 10)
    coordinates = np.empty((0, 2), dtype=float)

    first = scan_candidates(request, boundary, coordinates)
    second = scan_candidates(request, boundary, coordinates)

    assert first == second
    ranks = [scanner_module._rank(request.random_seed, item) for item in first.candidates]
    assert ranks == sorted(ranks)


def test_complete_scan_cancels_before_full_coverage_without_completed_event():
    cancel = Event()
    events = []

    def cancel_after_first_row(event):
        events.append(event)
        cancel.set()

    with pytest.raises(ScanCancelled):
        scan_candidates(
            make_request(target_count=0, mode="complete", max_candidates=1),
            box(0, 0, 10, 10),
            np.empty((0, 2), dtype=float),
            progress=cancel_after_first_row,
            cancel=cancel,
        )

    assert events
    assert events[-1].checked_positions < events[-1].total_positions
    assert all(event.phase != "completed" for event in events)


def fake_benchmark_inputs(_config_path):
    request = ScanRequest(
        rectangle_size=4,
        target_count=1,
        tolerance=1,
        step=3,
        max_candidates=2,
        minimum_spacing=1,
        strategy="sequential",
        mode="fast",
        random_seed=7,
        algorithm_version="row-sweep-v1",
    )
    return request, box(0, 0, 11, 11), np.asarray([[1, 1]], dtype=float), "fixture"


def test_benchmark_reports_metrics_without_writing_outputs(tmp_path, monkeypatch):
    from lte_scenario_toolkit.benchmark import benchmark_profile

    config = tmp_path / "profile.yaml"
    config.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(
        "lte_scenario_toolkit.benchmark._load_benchmark_inputs",
        fake_benchmark_inputs,
    )

    result = benchmark_profile(config)

    assert result["scenario"] == "fixture"
    assert result["grid_x_positions"] == 3
    assert result["grid_y_positions"] == 3
    assert result["grid_positions"] == 9
    assert result["checked_positions"] <= 9
    assert result["candidate_count"] >= 0
    assert result["peak_python_bytes"] > 0
    assert result["elapsed_seconds"] >= 0
    assert list(tmp_path.iterdir()) == [config]


def test_benchmark_input_loader_uses_preflight_without_touching_cache_or_outputs(
    tmp_path,
    monkeypatch,
):
    from lte_scenario_toolkit import benchmark
    from lte_scenario_toolkit.candidate_cache import CandidateCache
    from lte_scenario_toolkit.selection_service import SelectionService

    repository = tmp_path / "repository"
    repository.mkdir()
    config = tmp_path / "profile.yaml"
    config.write_text("fixture", encoding="utf-8")
    output_root = repository / "results"
    profile = object()
    catalog = SimpleNamespace(root=repository)
    request, boundary, coordinates, _ = fake_benchmark_inputs(config)
    preflight = SimpleNamespace(scenario_id="fixture")
    calls: list[str] = []

    monkeypatch.setattr(
        benchmark,
        "load_experiment_config",
        lambda path: {
            "repo_root": repository,
            "profile_id": "fixture",
            "scenario_id": "fixture",
            "output_root": output_root,
        },
    )
    monkeypatch.setattr(benchmark, "load_data_catalog", lambda *args, **kwargs: catalog)
    monkeypatch.setattr(
        benchmark,
        "_selection_profile",
        lambda config_values, loaded_catalog, scenario_id: profile,
    )

    def fail_cache(*_args, **_kwargs):
        pytest.fail("benchmark input loading must not read or write candidate caches")

    monkeypatch.setattr(CandidateCache, "_ensure_cache_root", fail_cache)
    monkeypatch.setattr(CandidateCache, "load", fail_cache)
    monkeypatch.setattr(CandidateCache, "store", fail_cache)

    def prepare_preflight(self, selected_profile, *, output_root):
        calls.append("preflight")
        assert selected_profile is profile
        assert output_root == repository / "results"
        return preflight

    def prepare_spatial(self, selected_preflight):
        calls.append("prepared_selection")
        assert selected_preflight is preflight
        return SimpleNamespace(boundary=boundary, coordinates=coordinates)

    monkeypatch.setattr(SelectionService, "preflight", prepare_preflight)
    monkeypatch.setattr(SelectionService, "prepared_selection", prepare_spatial)
    monkeypatch.setattr(
        SelectionService,
        "_request",
        staticmethod(lambda selected_profile: request),
    )

    loaded_request, loaded_boundary, loaded_coordinates, loaded_scenario = (
        benchmark._load_benchmark_inputs(config)
    )

    assert loaded_request is request
    assert loaded_boundary is boundary
    assert loaded_coordinates is coordinates
    assert loaded_scenario == "fixture"
    assert calls == ["preflight", "prepared_selection"]
    assert not (repository / ".lte-data").exists()
    assert not output_root.exists()


def test_benchmark_main_prints_sorted_json(tmp_path, monkeypatch, capsys):
    from lte_scenario_toolkit import benchmark

    config = tmp_path / "profile.yaml"
    monkeypatch.setattr(
        benchmark,
        "benchmark_profile",
        lambda path: {"z_metric": str(path), "a_metric": 1},
    )

    exit_code = benchmark.main(["--config", str(config)])

    assert exit_code == 0
    assert capsys.readouterr().out == (
        json.dumps({"z_metric": str(config), "a_metric": 1}, sort_keys=True)
        + "\n"
    )


def test_benchmark_main_returns_nonzero_for_domain_error(tmp_path, monkeypatch, capsys):
    from lte_scenario_toolkit import benchmark

    def fail(_path):
        raise ValueError("invalid benchmark profile")

    monkeypatch.setattr(benchmark, "benchmark_profile", fail)

    exit_code = benchmark.main(["--config", str(tmp_path / "profile.yaml")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "ERROR: invalid benchmark profile\n"
