import numpy as np
import pytest
from shapely.geometry import box

from src.scenario import choose_result, generate_scan_positions, scan_rectangles, validate_results


def test_generate_scan_positions_is_deterministic_for_uniform_strategy():
    boundary = box(0, 0, 11, 11)

    first = generate_scan_positions(boundary, rectangle_size=4, step=3, strategy="uniform")
    second = generate_scan_positions(boundary, rectangle_size=4, step=3, strategy="uniform")

    assert first.shape == (9, 2)
    np.testing.assert_array_equal(first, second)


def test_scan_rectangles_enforces_count_boundary_and_limit():
    coordinates = np.array([[1.0, 1.0], [2.0, 2.0], [8.0, 8.0]])
    boundary = box(-1, -1, 11, 11)
    positions = np.array([[0.0, 0.0], [6.0, 6.0]])
    config = {
        "rect_size": 4,
        "target_count": 2,
        "tolerance": 0,
        "max_rects": 1,
        "min_spacing": 1,
    }

    results = scan_rectangles(coordinates, boundary, positions, config)

    assert len(results) == 1
    assert results[0]["pt_count"] == 2
    assert results[0]["center_x"] == 2
    assert validate_results(results, coordinates, rectangle_size=4) == []


def test_validate_results_reports_changed_point_count():
    coordinates = np.array([[1.0, 1.0]])
    results = [{"left_x": 0, "bottom_y": 0, "pt_count": 2}]

    mismatches = validate_results(results, coordinates, rectangle_size=4)

    assert mismatches == [{"index": 0, "recorded": 2, "actual": 1}]


def test_choose_result_uses_one_based_index_and_rejects_invalid_value():
    results = [{"pt_count": 2}, {"pt_count": 3}]

    assert choose_result(results, 2) == [results[1]]
    with pytest.raises(ValueError, match="select-index"):
        choose_result(results, 0)
