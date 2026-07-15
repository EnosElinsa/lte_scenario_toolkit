from pathlib import Path

import pandas as pd
import pytest

from lte_scenario_toolkit.generate_figures import load_scenario_csv

REQUIRED_ROW = {
    "rect_id": 1,
    "pt_count": 1,
    "left_x": 0.0,
    "bottom_y": 0.0,
    "center_x": 500.0,
    "center_y": 500.0,
    "X": 100.0,
    "Y": 200.0,
    "elevation": 12.5,
}


def write_csv(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(path, index=False)


def test_load_scenario_csv_builds_rectangle_and_projected_points(tmp_path):
    csv_path = tmp_path / "scenario.csv"
    write_csv(csv_path, REQUIRED_ROW)

    frame, rectangle, points = load_scenario_csv(csv_path)

    assert len(frame) == 1
    assert rectangle["pt_count"] == 1
    assert rectangle["center_x"] == 500.0
    assert points.crs.to_epsg() == 3857
    assert points.geometry.iloc[0].x == 100.0


def test_load_scenario_csv_rejects_missing_required_columns(tmp_path):
    csv_path = tmp_path / "scenario.csv"
    write_csv(csv_path, {"X": 100.0, "Y": 200.0})

    with pytest.raises(ValueError, match="missing required columns"):
        load_scenario_csv(csv_path)
