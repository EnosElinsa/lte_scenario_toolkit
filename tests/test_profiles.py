import re
from pathlib import Path

import pytest
import yaml

from lte_scenario_toolkit.profiles import (
    DEFAULT_PROFILE_VALUES,
    ExperimentProfile,
    FigureSettings,
    OutputSettings,
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


def test_experiment_profile_uses_fresh_default_output_and_figure_settings():
    profile = ExperimentProfile(
        schema_version=2,
        profile_id="chicago-default",
        display_name="Chicago default",
        scenario_id="chicago",
        points_dataset_id="points",
        random_seed=42,
        target_crs="EPSG:3857",
        rect_size=3000,
        target_count=30,
        tolerance=0,
        scan_mode="fast",
        strategy="uniform",
        scan_step=10,
        max_rects=100,
        min_spacing=3000,
        output_root=Path("results"),
    )

    assert profile.outputs == OutputSettings()
    assert profile.figure == FigureSettings()


@pytest.mark.parametrize("outputs_section", [{}, None])
def test_load_profile_defaults_outputs_when_root_is_omitted(
    tmp_path,
    outputs_section,
):
    profile_path = tmp_path / "profile.yaml"
    write_profile(profile_path)
    document = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if outputs_section is None:
        document.pop("outputs")
    else:
        document["outputs"] = outputs_section
    profile_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    profile = load_profile(profile_path, repo_root=tmp_path)

    assert profile.output_root == tmp_path / "results"
    assert profile.outputs == OutputSettings()


@pytest.mark.parametrize(
    ("section", "key", "path"),
    [
        ("spatial", "target_crs", "spatial.target_crs"),
        ("spatial", "rectangle_size_m", "spatial.rectangle_size_m"),
        (
            "spatial",
            "target_base_station_count",
            "spatial.target_base_station_count",
        ),
        ("spatial", "count_tolerance", "spatial.count_tolerance"),
        ("scan", "strategy", "scan.strategy"),
        ("scan", "step_m", "scan.step_m"),
        ("scan", "max_rectangles", "scan.max_rectangles"),
        (
            "scan",
            "minimum_center_spacing_m",
            "scan.minimum_center_spacing_m",
        ),
    ],
)
def test_load_profile_requires_explicit_spatial_and_scan_values(
    tmp_path,
    section,
    key,
    path,
):
    profile_path = tmp_path / "profile.yaml"
    write_profile(profile_path)
    document = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    document[section].pop(key)
    profile_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=rf"^Missing required configuration value: {path}$",
    ):
        load_profile(profile_path, repo_root=tmp_path)


@pytest.mark.parametrize("bad_id", ["Chicago Default", "../escape", "con"])
def test_load_profile_rejects_unsafe_profile_id(tmp_path, bad_id):
    profile_path = tmp_path / "profile.yaml"
    write_profile(profile_path, profile_id=bad_id)

    with pytest.raises(ValueError, match=r"profile\.id"):
        load_profile(profile_path, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    [
        ("schema_version", 2.0),
        ("schema_version", True),
        ("profile.display_name", None),
        ("profile.scenario_id", 123),
        ("inputs.points_dataset_id", None),
        ("spatial.rectangle_size_m", 1.9),
        ("spatial.target_base_station_count", True),
        ("spatial.count_tolerance", "1"),
        ("experiment.random_seed", 7.5),
        ("scan.step_m", True),
        ("scan.mode", 1),
        ("outputs.root", ["results"]),
        ("outputs.save_csv", "false"),
        ("figures.dpi", True),
        ("figures.vertical_exaggeration", float("nan")),
        ("figures.title", 123),
    ],
)
def test_load_profile_rejects_invalid_value_types_with_field_location(
    tmp_path,
    field_path,
    invalid_value,
):
    profile_path = tmp_path / "profile.yaml"
    write_profile(profile_path)
    document = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    path_parts = field_path.split(".")
    target = document
    for part in path_parts[:-1]:
        target = target[part]
    target[path_parts[-1]] = invalid_value
    profile_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(field_path)):
        load_profile(profile_path, repo_root=tmp_path)


def test_load_profile_accepts_finite_figure_numbers_and_real_booleans(tmp_path):
    profile_path = tmp_path / "profile.yaml"
    write_profile(profile_path)
    document = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    document["figures"].update(
        {
            "azimuth_deg": -45,
            "elevation_deg": 22.5,
            "vertical_exaggeration": 2,
            "station_marker_size": 12.25,
        }
    )
    document["outputs"].update(
        {
            "save_csv": False,
            "save_preview_png": True,
            "save_terrain_png": False,
            "save_terrain_eps": True,
            "save_terrain_html": False,
        }
    )
    profile_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    profile = load_profile(profile_path, repo_root=tmp_path)

    assert profile.figure.azimuth_deg == -45.0
    assert profile.figure.elevation_deg == 22.5
    assert profile.figure.vertical_exaggeration == 2.0
    assert profile.figure.station_marker_size == 12.25
    assert type(profile.figure.azimuth_deg) is float
    assert type(profile.figure.elevation_deg) is float
    assert type(profile.figure.vertical_exaggeration) is float
    assert type(profile.figure.station_marker_size) is float
    assert profile.outputs == OutputSettings(
        save_csv=False,
        save_preview_png=True,
        save_terrain_png=False,
        save_terrain_eps=True,
        save_terrain_html=False,
    )


def test_load_profile_normalizes_numeric_overflow_with_field_location(tmp_path):
    profile_path = tmp_path / "profile.yaml"
    write_profile(profile_path)
    document = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    document["figures"]["vertical_exaggeration"] = 10**1000
    profile_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"^figures\.vertical_exaggeration must be a finite number$",
    ):
        load_profile(profile_path, repo_root=tmp_path)
