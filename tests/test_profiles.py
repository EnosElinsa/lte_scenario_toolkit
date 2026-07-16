from pathlib import Path

import pytest

from lte_scenario_toolkit.profiles import (
    DEFAULT_PROFILE_VALUES,
    ExperimentProfile,
    load_profile,
)


def write_profile(path: Path, *, profile_id: str = "chicago-default") -> None:
    path.write_text(
        f"""
schema_version: 2
profile:
  id: {profile_id}
  display_name: Chicago default
  scenario_id: chicago
inputs:
  points_dataset_id: points
experiment:
  random_seed: 7
spatial:
  target_crs: EPSG:3857
  rectangle_size_m: 2000
  target_base_station_count: 20
  count_tolerance: 1
scan:
  mode: complete
  strategy: uniform
  step_m: 25
  max_rectangles: 40
  minimum_center_spacing_m: 1500
outputs:
  root: results
  save_csv: true
figures:
  preset: publication
  dpi: 300
""".strip(),
        encoding="utf-8",
    )


def test_load_profile_maps_schema_version_2_to_runtime_values(tmp_path):
    profile_path = tmp_path / "profile.yaml"
    write_profile(profile_path)

    profile = load_profile(profile_path, repo_root=tmp_path)

    assert isinstance(profile, ExperimentProfile)
    assert profile.schema_version == 2
    assert profile.profile_id == "chicago-default"
    assert profile.display_name == "Chicago default"
    assert profile.scenario_id == "chicago"
    assert profile.points_dataset_id == "points"
    assert profile.random_seed == 7
    assert profile.target_crs == "EPSG:3857"
    assert profile.rect_size == 2000
    assert profile.target_count == 20
    assert profile.tolerance == 1
    assert profile.scan_mode == "complete"
    assert profile.strategy == "uniform"
    assert profile.scan_step == 25
    assert profile.max_rects == 40
    assert profile.min_spacing == 1500
    assert profile.output_root == tmp_path / "results"
    assert profile.outputs.save_csv is True
    assert profile.figure.preset == "publication"
    assert profile.figure.dpi == 300
    assert profile.source_path == profile_path.resolve()

    runtime = profile.runtime_values()
    assert runtime == {
        "profile_id": "chicago-default",
        "scenario_id": "chicago",
        "points_dataset_id": "points",
        "random_seed": 7,
        "target_crs": "EPSG:3857",
        "rect_size": 2000,
        "target_count": 20,
        "tolerance": 1,
        "scan_mode": "complete",
        "strategy": "uniform",
        "scan_step": 25,
        "max_rects": 40,
        "min_spacing": 1500,
        "output_root": tmp_path / "results",
        "save_csv": True,
        "save_preview_png": True,
        "save_terrain_png": True,
        "save_terrain_eps": True,
        "save_terrain_html": True,
        "config_path": profile_path.resolve(),
    }


def test_default_profile_values_are_explicit_and_stable():
    assert DEFAULT_PROFILE_VALUES["rect_size"] == 3000
    assert DEFAULT_PROFILE_VALUES["target_count"] == 30
    assert DEFAULT_PROFILE_VALUES["tolerance"] == 0
    assert DEFAULT_PROFILE_VALUES["scan_mode"] == "fast"
    assert DEFAULT_PROFILE_VALUES["max_rects"] == 100


@pytest.mark.parametrize("bad_id", ["Chicago Default", "../escape", "con"])
def test_load_profile_rejects_unsafe_profile_id(tmp_path, bad_id):
    profile_path = tmp_path / "profile.yaml"
    write_profile(profile_path, profile_id=bad_id)

    with pytest.raises(ValueError, match=r"profile\.id"):
        load_profile(profile_path, repo_root=tmp_path)
