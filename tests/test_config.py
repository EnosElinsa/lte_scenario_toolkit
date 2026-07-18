from pathlib import Path

import pytest
import yaml

from lte_scenario_toolkit.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]


def write_config(path: Path, *, strategy: str = "uniform") -> None:
    path.write_text(
        f"""
profile:
  id: fixture
  display_name: Fixture
  scenario_id: test-city
inputs:
  points_dataset_id: points
experiment:
  random_seed: 7
spatial:
  target_crs: EPSG:3857
  rectangle_size_m: 1000
  target_base_station_count: 3
  count_tolerance: 1
scan:
  mode: fast
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
figures:
  preset: publication
""".strip(),
        encoding="utf-8",
    )


def test_load_experiment_config_maps_current_profile_and_output_override(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    write_config(config_path)

    config = load_experiment_config(
        config_path,
        repo_root=tmp_path,
        output_dir=tmp_path / "custom-output",
    )

    assert config["profile_id"] == "fixture"
    assert config["scenario_id"] == "test-city"
    assert config["random_seed"] == 7
    assert config["rect_size"] == 1000
    assert config["target_count"] == 3
    assert config["strategy"] == "uniform"
    assert config["output_root"] == tmp_path / "custom-output"
    assert config.profile_snapshot.profile_id == "fixture"


def test_load_experiment_config_rejects_unknown_scan_strategy(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    write_config(config_path, strategy="random")

    with pytest.raises(ValueError, match="strategy"):
        load_experiment_config(config_path, repo_root=tmp_path)


def test_load_experiment_config_rejects_removed_schema_discriminator(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    write_config(config_path)
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["schema_version"] = 2
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected.*schema_version"):
        load_experiment_config(config_path, repo_root=tmp_path)


def test_repository_config_resolves_from_repository_root():
    config = load_experiment_config(ROOT / "configs" / "example.yaml")

    assert config["repo_root"] == ROOT
    assert config["profile_id"] == "chicago-default"
    assert config["scenario_id"] == "chicago"
    assert config["output_root"] == ROOT / "results"


def test_config_in_configs_directory_infers_external_project_root(tmp_path):
    project = tmp_path / "external-study"
    config_dir = project / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "experiment.yaml"
    write_config(config_path)

    config = load_experiment_config(config_path)

    assert config["repo_root"] == project
    assert config["output_root"] == project / "results" / "fixture"
