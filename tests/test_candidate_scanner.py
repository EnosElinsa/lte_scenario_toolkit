import numpy as np
import pytest
from shapely.geometry import box

from lte_scenario_toolkit.candidate_scanner import count_row, grid_axes


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
