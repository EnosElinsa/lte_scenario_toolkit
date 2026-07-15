from pathlib import Path

import pytest

from src.config import load_experiment_config


def write_config(path: Path, *, strategy: str = "uniform") -> None:
    path.write_text(
        f"""
experiment:
  name: fixture_run
  random_seed: 7
inputs:
  points_root: points
  points_layer: stations
  boundary_root: boundaries
  city: TestCity
  dem_path: dem/test.tif
spatial:
  target_crs: EPSG:3857
  rectangle_size_m: 1000
  target_base_station_count: 3
  count_tolerance: 1
scan:
  strategy: {strategy}
  step_m: 50
  max_rectangles: 4
  minimum_center_spacing_m: 500
outputs:
  root: results/fixture
  save_csv: true
  save_preview_png: false
  save_terrain_png: true
  save_terrain_eps: false
""".strip(),
        encoding="utf-8",
    )


def test_load_experiment_config_maps_yaml_and_cli_overrides(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    write_config(config_path)

    config = load_experiment_config(
        config_path,
        repo_root=tmp_path,
        city="OverrideCity",
        output_dir=tmp_path / "custom-output",
    )

    assert config["experiment_name"] == "fixture_run"
    assert config["random_seed"] == 7
    assert config["city_name"] == "OverrideCity"
    assert config["rect_size"] == 1000
    assert config["target_count"] == 3
    assert config["strategy"] == "uniform"
    assert config["points_root"] == tmp_path / "points"
    assert config["output_root"] == tmp_path / "custom-output"
    assert config["output_dir_is_final"] is True


def test_load_experiment_config_rejects_unknown_scan_strategy(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    write_config(config_path, strategy="random")

    with pytest.raises(ValueError, match="strategy"):
        load_experiment_config(config_path, repo_root=tmp_path)
