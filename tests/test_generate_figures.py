import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from lte_scenario_toolkit import generate_figures
from lte_scenario_toolkit.figure_service import FigureService
from lte_scenario_toolkit.generate_figures import load_scenario_csv
from lte_scenario_toolkit.run_service import RunService

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


def write_dem(path: Path) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 1000, 250, 250),
    ) as dem:
        dem.write(np.arange(16, dtype="float32").reshape(4, 4), 1)


def test_cli_rejects_mutually_exclusive_or_missing_sources():
    with pytest.raises(SystemExit) as missing:
        generate_figures.main([])
    assert missing.value.code == 2

    with pytest.raises(SystemExit) as multiple:
        generate_figures.main(["--run-dir", "run", "--csv", "scenario.csv"])
    assert multiple.value.code == 2

    with pytest.raises(SystemExit) as run_with_config:
        generate_figures.main(["--run-dir", "run", "--config", "profile.yaml"])
    assert run_with_config.value.code == 2

    with pytest.raises(SystemExit) as multiple_outputs:
        generate_figures.main(
            [
                "--csv",
                "scenario.csv",
                "--output-root",
                "runs",
                "--output-dir",
                "exact",
            ]
        )
    assert multiple_outputs.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--preset", "invalid"],
        ["--format", "pdf"],
    ],
)
def test_cli_parser_rejects_invalid_choices_with_exit_2(arguments):
    with pytest.raises(SystemExit) as captured:
        generate_figures.main(["--csv", "scenario.csv", *arguments])
    assert captured.value.code == 2


def test_cli_maps_invalid_rect_and_style_to_exit_2(tmp_path, capsys):
    csv_path = tmp_path / "scenario.csv"
    write_csv(csv_path, REQUIRED_ROW)

    assert (
        generate_figures.main(
            ["--csv", str(csv_path), "--rect-id", "9", "--output-dir", str(tmp_path)]
        )
        == 2
    )
    assert "rect_id" in capsys.readouterr().err

    assert (
        generate_figures.main(
            ["--csv", str(csv_path), "--dpi", "0", "--output-dir", str(tmp_path)]
        )
        == 2
    )
    assert "DPI" in capsys.readouterr().err


def test_bare_csv_cli_reports_actionable_missing_dem(tmp_path, capsys):
    csv_path = tmp_path / "scenario.csv"
    write_csv(csv_path, REQUIRED_ROW)

    exit_code = generate_figures.main(
        ["--csv", str(csv_path), "--output-dir", str(tmp_path / "runs")]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "DEM" in captured.err
    assert "legacy config" in captured.err


def test_run_cli_maps_style_formats_and_parent_to_service(
    tmp_path,
    monkeypatch,
    capsys,
):
    csv_path = tmp_path / "scenario.csv"
    dem_path = tmp_path / "dem.tif"
    write_csv(csv_path, REQUIRED_ROW)
    write_dem(dem_path)
    source = replace(
        FigureService.load_source(csv_path),
        dem_path=dem_path,
        run_id="a" * 32,
        scenario_id="city",
        profile_id="profile",
    )
    monkeypatch.setattr(
        generate_figures.FigureService,
        "load_source",
        lambda path, rect_id=None: source,
    )
    calls = []
    published = tmp_path / "published"

    def render(
        source_value,
        spec,
        service,
        formats,
        parent_run_id=None,
        **provenance,
    ):
        calls.append(
            (
                source_value,
                spec,
                service.output_root,
                formats,
                parent_run_id,
                provenance,
            )
        )
        return published

    monkeypatch.setattr(generate_figures.FigureService, "render", render)

    exit_code = generate_figures.main(
        [
            "--run-dir",
            str(tmp_path / "source-run"),
            "--output-root",
            str(tmp_path / "runs"),
            "--preset",
            "preview",
            "--dpi",
            "144",
            "--azimuth",
            "-45",
            "--elevation-angle",
            "20",
            "--vertical-exaggeration",
            "2",
            "--colormap",
            "viridis",
            "--station-color",
            "navy",
            "--station-size",
            "33",
            "--title",
            "Terrain detail",
            "--format",
            "png",
            "--format",
            "eps",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    _, spec, output_root, formats, parent_run_id, provenance = calls[0]
    assert spec.preset == "preview"
    assert spec.dpi == 144
    assert spec.azimuth == -45
    assert spec.elevation_angle == 20
    assert spec.vertical_exaggeration == 2
    assert spec.colormap == "viridis"
    assert spec.station_color == "navy"
    assert spec.station_size == 33
    assert spec.title == "Terrain detail"
    assert output_root == (tmp_path / "runs").resolve()
    assert formats == ("png", "eps")
    assert parent_run_id == "a" * 32
    assert provenance["entrypoint"][0] == "lte-generate-figures"
    assert provenance["repository"] == Path.cwd().resolve()
    assert str(published) in capsys.readouterr().out


def test_v2_config_uses_profile_figure_settings_before_cli_overrides(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "scenario.csv"
    dem_path = tmp_path / "dem.tif"
    write_csv(csv_path, REQUIRED_ROW)
    write_dem(dem_path)
    source = replace(
        FigureService.load_source(csv_path),
        dem_path=dem_path,
        scenario_id="city",
        profile_id="profile",
    )
    from lte_scenario_toolkit.profiles import FigureSettings

    profile_figure = FigureSettings(
        preset="preview",
        colormap="viridis",
        dpi=144,
        azimuth_deg=-25.0,
        elevation_deg=18.0,
        vertical_exaggeration=2.5,
        station_color="navy",
        station_marker_size=31.0,
        title="Configured title",
    )
    configured_spec = generate_figures._profile_figure_spec(profile_figure)
    monkeypatch.setattr(
        generate_figures,
        "_source_and_output",
        lambda args: (source, tmp_path / "runs", ("png",), configured_spec),
    )
    calls = []
    monkeypatch.setattr(
        generate_figures.FigureService,
        "render",
        lambda source, spec, service, formats, parent_run_id=None, **kwargs: (
            calls.append(spec) or tmp_path / "published"
        ),
    )

    assert generate_figures.main(
        [
            "--config",
            str(tmp_path / "profile.yaml"),
            "--dpi",
            "200",
        ]
    ) == 0

    spec = calls[0]
    assert spec.preset == "preview"
    assert spec.colormap == "viridis"
    assert spec.dpi == 200
    assert spec.azimuth == -25.0
    assert spec.elevation_angle == 18.0
    assert spec.vertical_exaggeration == 2.5
    assert spec.station_color == "navy"
    assert spec.station_size == 31.0
    assert spec.title == "Configured title"


def test_legacy_config_cli_injects_validated_dem_and_config_crs(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "scenario.csv"
    dem_path = tmp_path / "dem.tif"
    write_csv(csv_path, REQUIRED_ROW)
    write_dem(dem_path)
    config = {
        "target_crs": "EPSG:32616",
        "rect_size": 750,
        "target_count": 1,
        "output_root": tmp_path / "runs",
        "dem_path": dem_path,
        "save_terrain_png": True,
        "save_terrain_eps": False,
        "save_terrain_html": False,
    }
    monkeypatch.setattr(
        generate_figures,
        "load_experiment_config",
        lambda *args, **kwargs: config.copy(),
    )
    monkeypatch.setattr(
        generate_figures,
        "resolve_io_paths",
        lambda value, create_output=False: {
            "output_csv": csv_path,
            "dem_path": dem_path,
        },
    )
    calls = []

    def render(source, spec, service, formats, parent_run_id=None, **kwargs):
        calls.append((source, formats))
        return tmp_path / "published"

    monkeypatch.setattr(generate_figures.FigureService, "render", render)

    exit_code = generate_figures.main(["--config", str(tmp_path / "config.yaml")])

    assert exit_code == 0
    source, formats = calls[0]
    assert source.dem_path == dem_path.resolve()
    assert source.target_crs == "EPSG:32616"
    assert source.rectangle_size_m == 750
    assert formats == ("png",)


def test_config_context_does_not_override_authoritative_run_snapshot(tmp_path):
    csv_path = tmp_path / "scenario.csv"
    run_dem = tmp_path / "run-dem.tif"
    context_dem = tmp_path / "context-dem.tif"
    write_csv(csv_path, REQUIRED_ROW)
    write_dem(run_dem)
    write_dem(context_dem)
    source = replace(
        FigureService.load_source(csv_path),
        run_id="a" * 32,
        target_crs="EPSG:32616",
        rectangle_size_m=750,
        dem_path=run_dem,
        scenario_id="run-city",
        profile_id="run-profile",
    )

    contextual = generate_figures._contextual_source(
        source,
        dem_path=context_dem,
        target_crs="EPSG:3857",
        rectangle_size=1000,
        scenario_id="context-city",
        profile_id="context-profile",
    )

    assert contextual.target_crs == "EPSG:32616"
    assert contextual.rectangle_size_m == 750
    assert contextual.dem_path == run_dem
    assert contextual.scenario_id == "run-city"
    assert contextual.profile_id == "run-profile"


def test_explicit_legacy_csv_with_config_does_not_load_vector_inputs(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "scenario.csv"
    dem_path = tmp_path / "dem.tif"
    write_csv(csv_path, REQUIRED_ROW)
    write_dem(dem_path)
    config = {
        "target_crs": "EPSG:3857",
        "rect_size": 1000,
        "target_count": 1,
        "output_root": tmp_path / "runs",
        "dem_path": dem_path,
        "save_terrain_png": True,
        "save_terrain_eps": False,
        "save_terrain_html": False,
    }
    monkeypatch.setattr(
        generate_figures,
        "load_experiment_config",
        lambda *args, **kwargs: config.copy(),
    )
    monkeypatch.setattr(
        generate_figures,
        "resolve_io_paths",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("explicit figure CSV must not load vectors")
        ),
    )
    calls = []
    monkeypatch.setattr(
        generate_figures.FigureService,
        "render",
        lambda source, *args, **kwargs: calls.append(source) or tmp_path / "published",
    )

    exit_code = generate_figures.main(
        ["--config", str(tmp_path / "config.yaml"), "--csv", str(csv_path)]
    )

    assert exit_code == 0
    assert calls[0].dem_path == dem_path.resolve()


def test_v2_config_supplies_catalog_dem_context_to_explicit_csv_without_vectors(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "scenario.csv"
    write_csv(csv_path, REQUIRED_ROW)
    dem_path = tmp_path / "inputs" / "dem" / "elevation.tif"
    dem_path.parent.mkdir(parents=True)
    write_dem(dem_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    base = {
        "source_url": "https://example.test/data",
        "provider": "Fixture",
        "license": "CC0-1.0",
        "download_date": "2026-07-17",
        "crs": "EPSG:3857",
        "spatial_resolution": "fixture",
        "notes": "fixture",
    }
    catalog = {
        "schema_version": 2,
        "datasets": [
            {
                **base,
                "dataset_id": "boundary",
                "role": "boundary",
                "path": "inputs/boundary",
                "entrypoint": "inputs/boundary/boundary.geojson",
                "geometry_type": "Polygon",
                "feature_count": 1,
                "redistribution_confirmed": True,
            },
            {
                **base,
                "dataset_id": "dem",
                "role": "dem",
                "path": "inputs/dem",
                "entrypoint": "inputs/dem/elevation.tif",
                "external": True,
                "earth_engine_collection": "EXAMPLE/DEM",
                "band": "elevation",
                "units": "metres",
                "vertical_datum": "NAVD88",
                "native_scale_m": 1,
                "export_crs": "EPSG:3857",
                "export_prefix": "fixture-dem",
                "drive_folder": "fixture-exports",
            },
        ],
        "scenarios": [
            {
                "scenario_id": "test-city",
                "display_name": "Test City",
                "boundary_dataset_id": "boundary",
                "dem_dataset_id": "dem",
                "config_path": "configs/test-city.yaml",
            }
        ],
    }
    (data_dir / "datasets.yaml").write_text(
        yaml.safe_dump(catalog, sort_keys=False),
        encoding="utf-8",
    )
    configs = tmp_path / "configs"
    configs.mkdir()
    profile_path = configs / "test-city.yaml"
    profile_path.write_text(
        """
schema_version: 2
profile:
  id: test-default
  display_name: Test default
  scenario_id: test-city
inputs:
  points_dataset_id: points
experiment:
  random_seed: 7
spatial:
  target_crs: EPSG:3857
  rectangle_size_m: 1000
  target_base_station_count: 1
  count_tolerance: 0
scan:
  mode: fast
  strategy: uniform
  step_m: 10
  max_rectangles: 1
  minimum_center_spacing_m: 1000
outputs:
  root: runs
  save_terrain_png: true
  save_terrain_eps: false
  save_terrain_html: false
figures:
  preset: preview
  colormap: viridis
  dpi: 144
  azimuth_deg: -25
  elevation_deg: 18
  vertical_exaggeration: 2.5
  station_color: navy
  station_marker_size: 31
  title: Configured title
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        generate_figures,
        "resolve_io_paths",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("v2 figure CSV must not load boundary or points")
        ),
    )
    calls = []

    def render(source, spec, service, formats, parent_run_id=None, **kwargs):
        calls.append((source, spec, service.output_root, formats))
        return tmp_path / "published"

    monkeypatch.setattr(generate_figures.FigureService, "render", render)

    exit_code = generate_figures.main(
        [
            "--config",
            str(profile_path),
            "--csv",
            str(csv_path),
            "--output-root",
            str(tmp_path / "published-runs"),
            "--dpi",
            "200",
        ]
    )

    assert exit_code == 0
    source, spec, output_root, formats = calls[0]
    assert source.dem_path == dem_path.resolve()
    assert source.scenario_id == "test-city"
    assert source.profile_id == "test-default"
    assert spec.preset == "preview"
    assert spec.colormap == "viridis"
    assert spec.dpi == 200
    assert spec.azimuth == -25
    assert spec.elevation_angle == 18
    assert spec.vertical_exaggeration == 2.5
    assert spec.station_color == "navy"
    assert spec.station_size == 31
    assert spec.title == "Configured title"
    assert output_root == (tmp_path / "published-runs").resolve()
    assert formats == ("png",)


def test_latest_profile_run_uses_public_entries_and_published_created_at(
    tmp_path,
    monkeypatch,
):
    service = RunService(tmp_path / "runs")

    def publish(created_at: str, center_x: float):
        run = service.begin(
            "city",
            "profile",
            created_at=created_at,
        )
        pd.DataFrame([{**REQUIRED_ROW, "center_x": center_x}]).to_csv(
            run.path / "scenario.csv",
            index=False,
        )
        (run.path / "run-config.yaml").write_text(
            yaml.safe_dump(
                {
                    "profile": {"id": "profile", "scenario_id": "city"},
                    "spatial": {
                        "target_crs": "EPSG:3857",
                        "rectangle_size_m": 1000,
                    },
                }
            ),
            encoding="utf-8",
        )
        path = service.publish(
            run,
            status="completed",
            artifacts=("scenario.csv", "run-config.yaml"),
            metadata={"run_kind": "selection"},
        )
        return run, path

    old_run, old_path = publish("2026-01-01T00:00:00Z", 500.0)
    new_run, _ = publish("2026-02-01T00:00:00Z", 600.0)
    future = 4_102_444_800_000_000_000
    os.utime(old_path / "run.json", ns=(future, future))

    staging = service.begin(
        "city",
        "profile",
        created_at="2027-01-01T00:00:00Z",
    )
    pd.DataFrame([{**REQUIRED_ROW, "center_x": 700.0}]).to_csv(
        staging.path / "scenario.csv",
        index=False,
    )
    (staging.path / "run-config.yaml").write_text(
        yaml.safe_dump(
            {
                "profile": {"id": "profile", "scenario_id": "city"},
                "spatial": {
                    "target_crs": "EPSG:3857",
                    "rectangle_size_m": 1000,
                },
            }
        ),
        encoding="utf-8",
    )
    (staging.path / "run.json").write_text(
        json.dumps(
            {
                "run_id": staging.run_id,
                "scenario_id": "city",
                "profile_id": "profile",
                "artifacts": ["scenario.csv", "run-config.yaml"],
                "metadata": {"run_kind": "selection"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        RunService,
        "discover",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("figure lookup must consume public RunEntry objects")
        ),
    )

    source = generate_figures._latest_profile_run(
        service.output_root,
        "city",
        "profile",
    )

    assert source.run_id == new_run.run_id
    assert source.run_id != old_run.run_id


def test_legacy_output_dir_is_exact_and_preserves_unrelated_files(
    tmp_path,
    monkeypatch,
    capsys,
):
    csv_path = tmp_path / "scenario.csv"
    dem_path = tmp_path / "dem.tif"
    write_csv(csv_path, REQUIRED_ROW)
    write_dem(dem_path)
    source = replace(
        FigureService.load_source(csv_path),
        dem_path=dem_path,
        scenario_id="city",
        profile_id="profile",
    )
    monkeypatch.setattr(
        generate_figures.FigureService,
        "load_source",
        lambda path, rect_id=None: source,
    )

    def render(
        source_value,
        spec,
        service,
        formats,
        parent_run_id=None,
        **kwargs,
    ):
        run = service.begin("city", "profile")
        artifacts = ["source.csv", *(f"terrain.{value}" for value in formats)]
        for artifact in artifacts:
            (run.path / artifact).write_text(artifact, encoding="utf-8")
        return service.publish(
            run,
            status="completed",
            artifacts=artifacts,
            metadata={"run_kind": "figure"},
        )

    monkeypatch.setattr(generate_figures.FigureService, "render", render)
    exact = tmp_path / "exact"
    exact.mkdir()
    (exact / "keep.txt").write_text("keep", encoding="utf-8")

    exit_code = generate_figures.main(
        [
            "--run-dir",
            str(tmp_path / "selection-run"),
            "--output-dir",
            str(exact),
            "--format",
            "png",
        ]
    )

    assert exit_code == 0
    assert (exact / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert (exact / "source.csv").is_file()
    assert (exact / "terrain.png").is_file()
    assert (exact / "run.json").is_file()
    assert not (exact / "city").exists()
    assert not list(tmp_path.glob(".lte-figure-staging-*"))
    assert f"Figure run: {exact.resolve()}" in capsys.readouterr().out


def test_legacy_output_dir_rejects_artifact_conflicts_before_rendering(
    tmp_path,
    monkeypatch,
    capsys,
):
    csv_path = tmp_path / "scenario.csv"
    dem_path = tmp_path / "dem.tif"
    write_csv(csv_path, REQUIRED_ROW)
    write_dem(dem_path)
    source = replace(
        FigureService.load_source(csv_path),
        dem_path=dem_path,
        scenario_id="city",
        profile_id="profile",
    )
    monkeypatch.setattr(
        generate_figures.FigureService,
        "load_source",
        lambda path, rect_id=None: source,
    )
    monkeypatch.setattr(
        generate_figures.FigureService,
        "render",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("render must not start when exact outputs conflict")
        ),
    )
    exact = tmp_path / "exact"
    exact.mkdir()
    conflict = exact / "terrain.png"
    conflict.write_text("original", encoding="utf-8")

    exit_code = generate_figures.main(
        [
            "--run-dir",
            str(tmp_path / "selection-run"),
            "--output-dir",
            str(exact),
            "--format",
            "png",
        ]
    )

    assert exit_code == 2
    assert conflict.read_text(encoding="utf-8") == "original"
    assert "conflict" in capsys.readouterr().err.lower()
