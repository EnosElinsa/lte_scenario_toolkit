from pathlib import Path

import pytest

from lte_scenario_toolkit.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]


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


def test_repository_config_resolves_inputs_from_repository_root():
    config = load_experiment_config(ROOT / "configs" / "example.yaml")

    assert config["repo_root"] == ROOT
    assert config["points_root"] == ROOT / "points_shp"
    assert config["boundary_root"] == ROOT / "boundary_shp"


def test_config_in_configs_directory_infers_external_project_root(tmp_path):
    project = tmp_path / "external-study"
    config_dir = project / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "experiment.yaml"
    write_config(config_path)

    config = load_experiment_config(config_path)

    assert config["repo_root"] == project
    assert config["points_root"] == project / "points"


def test_public_loader_supports_schema_version_2_profiles(tmp_path):
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        """
schema_version: 2
profile:
  id: chicago-default
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

    config = load_experiment_config(profile_path, repo_root=tmp_path)

    assert config["profile_id"] == "chicago-default"
    assert config["scenario_id"] == "chicago"
    assert config["rect_size"] == 2000
    assert config["scan_mode"] == "complete"
