import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from lte_scenario_toolkit import select_sites
from lte_scenario_toolkit.candidate_scanner import Candidate, ScanResult
from lte_scenario_toolkit.config import load_experiment_config
from lte_scenario_toolkit.data_catalog import load_data_catalog
from lte_scenario_toolkit.select_sites import process_selected_rectangles
from lte_scenario_toolkit.selection_service import SelectionProgress

ROOT = Path(__file__).resolve().parents[1]


def _write_shapefile(path: Path, geometry) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"name": [path.stem]}, geometry=[geometry], crs="EPSG:3857").to_file(
        path,
        driver="ESRI Shapefile",
    )
    return path


def _catalog_dataset(
    dataset_id: str,
    role: str,
    path: str,
    entrypoint: str,
) -> dict[str, object]:
    dataset: dict[str, object] = {
        "dataset_id": dataset_id,
        "role": role,
        "path": path,
        "entrypoint": entrypoint,
        "source_url": None,
        "provider": "test",
        "license": "test",
        "download_date": None,
        "crs": "EPSG:3857",
        "spatial_resolution": "polygon vector" if role == "boundary" else "1 m",
        "notes": "fixture",
    }
    if role == "boundary":
        dataset.update(
            {
                "geometry_type": "Polygon",
                "feature_count": 1,
                "redistribution_confirmed": True,
            }
        )
    else:
        dataset.update(
            {
                "external": True,
                "earth_engine_collection": "TEST/DEM",
                "band": "elevation",
                "units": "metres",
                "vertical_datum": "test",
                "native_scale_m": 1,
                "export_crs": "EPSG:3857",
                "export_prefix": "elevation",
                "drive_folder": "test",
            }
        )
    return dataset


def _linked_config_fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path, Path]:
    registered_boundary = _write_shapefile(
        tmp_path / "boundary_shp" / "registered" / "registered.shp",
        box(0, 0, 10, 10),
    )
    alternate_boundary = _write_shapefile(
        tmp_path / "boundary_shp" / "alternate" / "alternate.shp",
        box(20, 20, 30, 30),
    )
    _write_shapefile(
        tmp_path / "points_shp" / "points" / "points.shp",
        Point(1, 1),
    )
    registered_dem = tmp_path / "dem" / "registered" / "elevation.tif"
    linked_config = tmp_path / "configs" / "registered.yaml"
    linked_config.parent.mkdir(parents=True)
    linked_config.write_text("# linked fixture\n", encoding="utf-8")
    catalog = {
        "schema_version": 2,
        "datasets": [
            _catalog_dataset(
                "boundary_registered",
                "boundary",
                "boundary_shp/registered",
                "boundary_shp/registered/registered.shp",
            ),
            _catalog_dataset(
                "dem_registered",
                "dem",
                "dem/registered",
                "dem/registered/elevation.tif",
            ),
        ],
        "scenarios": [
            {
                "scenario_id": "registered",
                "display_name": "Registered",
                "boundary_dataset_id": "boundary_registered",
                "dem_dataset_id": "dem_registered",
                "config_path": "configs/registered.yaml",
            }
        ],
    }
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "datasets.yaml").write_text(
        yaml.safe_dump(catalog, sort_keys=False),
        encoding="utf-8",
    )
    config: dict[str, object] = {
        "repo_root": tmp_path.resolve(),
        "config_path": linked_config.resolve(),
        "points_root": tmp_path / "points_shp",
        "points_layer": "points",
        "boundary_root": tmp_path / "boundary_shp",
        "city_name": "registered",
        "dem_path": registered_dem,
        "rect_size": 10,
        "target_count": 1,
        "tolerance": 0,
        "scan_step": 1,
        "min_spacing": 1,
        "strategy": "uniform",
        "random_seed": 42,
        "output_root": tmp_path / "outputs",
        "output_dir_is_final": True,
    }
    return config, registered_boundary, alternate_boundary, registered_dem


def test_linked_config_uses_exact_registered_boundary_and_dem(tmp_path):
    config, registered_boundary, _, registered_dem = _linked_config_fixture(tmp_path)

    paths = select_sites.resolve_selection_io_paths(config, create_output=False)

    assert paths["boundary_shp"] == registered_boundary.resolve()
    assert paths["dem_path"] == registered_dem.resolve()
    assert paths["registered_scenario_id"] == "registered"
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    ("config_name", "scenario_id"),
    [
        ("example.yaml", "chicago"),
        ("newyork.yaml", "new-york-city"),
    ],
)
def test_repository_linked_configs_resolve_exact_catalog_entrypoints(
    config_name,
    scenario_id,
):
    config = load_experiment_config(ROOT / "configs" / config_name)
    catalog = load_data_catalog(ROOT / "data" / "datasets.yaml", repo_root=ROOT)
    scenario = catalog.scenario(scenario_id)
    boundary = catalog.dataset(scenario["boundary_dataset_id"])
    dem = catalog.dataset(scenario["dem_dataset_id"])

    paths = select_sites.resolve_selection_io_paths(config, create_output=False)

    assert paths["registered_scenario_id"] == scenario_id
    assert paths["boundary_shp"] == catalog.resolve(boundary["entrypoint"])
    assert paths["dem_path"] == catalog.resolve(dem["entrypoint"])
    assert not Path(paths["output_dir"]).exists()


def test_linked_config_rejects_discovered_boundary_mismatch(tmp_path):
    config, _, _, _ = _linked_config_fixture(tmp_path)
    config["city_name"] = "alternate"
    output_root = Path(config["output_root"])
    assert not output_root.exists()

    with pytest.raises(ValueError, match="boundary.*does not match"):
        select_sites.resolve_selection_io_paths(config)

    assert not output_root.exists()


def test_linked_config_rejects_dem_mismatch(tmp_path):
    config, _, _, _ = _linked_config_fixture(tmp_path)
    config["dem_path"] = tmp_path / "dem" / "other" / "elevation.tif"
    output_root = Path(config["output_root"])
    assert not output_root.exists()

    with pytest.raises(ValueError, match="DEM.*does not match"):
        select_sites.resolve_selection_io_paths(config)

    assert not output_root.exists()


def test_main_reports_linked_boundary_mismatch_without_creating_output(
    tmp_path,
    capsys,
):
    config, _, _, _ = _linked_config_fixture(tmp_path)
    config_path = Path(config["config_path"])
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "mismatched-linked-config"},
                "inputs": {
                    "points_root": "points_shp",
                    "points_layer": "points",
                    "boundary_root": "boundary_shp",
                    "city": "alternate",
                    "dem_path": "dem/registered/elevation.tif",
                },
                "spatial": {
                    "target_crs": "EPSG:3857",
                    "rectangle_size_m": 10,
                    "target_base_station_count": 1,
                    "count_tolerance": 0,
                },
                "scan": {
                    "strategy": "uniform",
                    "step_m": 1,
                    "max_rectangles": 1,
                    "minimum_center_spacing_m": 1,
                },
                "outputs": {"root": "outputs"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_root = Path(config["output_root"])
    assert not output_root.exists()

    exit_code = select_sites.main(["--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "ERROR:" in captured.err
    assert "boundary does not match" in captured.err
    assert not output_root.exists()


def test_unlinked_config_keeps_standalone_path_resolution(tmp_path):
    config, _, alternate_boundary, _ = _linked_config_fixture(tmp_path)
    config["config_path"] = tmp_path / "configs" / "standalone.yaml"
    config["city_name"] = "alternate"
    standalone_dem = tmp_path / "dem" / "standalone.tif"
    config["dem_path"] = standalone_dem

    paths = select_sites.resolve_selection_io_paths(config, create_output=False)

    assert paths["boundary_shp"] == alternate_boundary.resolve()
    assert paths["dem_path"] == standalone_dem.resolve()
    assert "registered_scenario_id" not in paths


@pytest.mark.parametrize("catalog_mode", ["missing", "null-config-link"])
def test_missing_catalog_or_null_config_link_keeps_standalone_behavior(
    tmp_path,
    catalog_mode,
):
    config, _, alternate_boundary, _ = _linked_config_fixture(tmp_path)
    catalog_path = tmp_path / "data" / "datasets.yaml"
    if catalog_mode == "missing":
        catalog_path.unlink()
    else:
        catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
        catalog["scenarios"][0]["config_path"] = None
        catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    config["city_name"] = "alternate"
    config["dem_path"] = tmp_path / "dem" / "standalone.tif"

    paths = select_sites.resolve_selection_io_paths(config, create_output=False)

    assert paths["boundary_shp"] == alternate_boundary.resolve()
    assert "registered_scenario_id" not in paths


def test_process_selected_rectangles_samples_dem_and_builds_csv_rows():
    points = gpd.GeoDataFrame(
        {"cell": [7]},
        geometry=[Point(0.5, 1.5)],
        crs="EPSG:3857",
    )
    rectangle = {
        "geometry": box(0, 0, 2, 2),
        "pt_count": 1,
        "left_x": 0.0,
        "bottom_y": 0.0,
        "center_x": 1.0,
        "center_y": 1.0,
    }
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": from_origin(0, 2, 1, 1),
    }

    with MemoryFile() as memory_file:
        with memory_file.open(**profile) as dem:
            dem.write(np.array([[12, 13], [14, 15]], dtype="float32"), 1)
            frame, selected = process_selected_rectangles(
                [rectangle], points, dem, {"rect_size": 2}
            )

    assert frame["cell"].tolist() == [7]
    assert frame["elevation"].tolist() == [12.0]
    assert frame["rect_id"].tolist() == [1]
    assert selected.crs.to_epsg() == 3857


@pytest.mark.parametrize(
    ("cache_status", "prefix"),
    [("hit", "Loaded 1 cached candidates:"), ("miss", "Saved 1 candidates:")],
)
def test_shared_cache_message_preserves_legacy_loaded_and_saved_text(
    cache_status,
    prefix,
):
    candidate = Candidate(0, 1, 0, 0, 1, 1)
    result = ScanResult((candidate,), 1, 1, True, "row-sweep-v1")
    progress = SelectionProgress(
        phase="completed",
        checked_positions=1,
        total_positions=1,
        candidate_count=1,
        elapsed_seconds=0,
        added_candidates=(candidate,),
        removed_flat_grid_ids=(),
        cache_status=cache_status,
        cache_key="a" * 64,
    )

    message = select_sites._shared_cache_message(result, progress)

    assert message.startswith(prefix)
    assert message.endswith(f"{'a' * 64}.json")


def test_cli_maps_one_legacy_choice_and_builds_exact_artifact_tokens():
    selected = Candidate(4, 1, 0, 0, 1, 1)
    other = Candidate(7, 2, 2, 0, 3, 1)
    result = ScanResult((selected, other), 2, 2, True, "row-sweep-v1")

    assert select_sites._chosen_candidate(
        [{"flat_grid_id": 4}],
        result,
    ) is selected
    assert select_sites._export_artifacts(
        {
            "save_csv": True,
            "save_preview_png": False,
            "save_terrain_png": True,
            "save_terrain_eps": False,
            "save_terrain_html": True,
        }
    ) == ("csv", "terrain_png", "terrain_html")

    with pytest.raises(ValueError, match="exactly one"):
        select_sites._chosen_candidate(
            [{"flat_grid_id": 4}],
            replace(
                result,
                candidates=(selected, replace(selected, point_count=9)),
            ),
        )


def test_cli_reports_published_artifacts_partial_errors_and_run_record(
    tmp_path,
    capsys,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    csv_name = "chicago_2m_target1_tol0.csv"
    (run_dir / csv_name).write_text("cell\n1\n", encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "partial",
                "artifacts": [csv_name],
                "errors": [
                    {
                        "artifact": "preview_png",
                        "code": "artifact.preview_png.failed",
                        "message": "RuntimeError: preview boom",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    select_sites._report_published_run(run_dir)

    captured = capsys.readouterr()
    assert f"Scenario CSV: {run_dir / csv_name}" in captured.out
    assert f"Run record: {run_dir / 'run.json'}" in captured.out
    assert (
        "WARNING: preview_png: RuntimeError: preview boom" in captured.err
    )


@pytest.mark.parametrize(
    ("cache_status", "expected"),
    [("hit", "Loaded 1 cached candidates:"), ("miss", "Saved 1 candidates:")],
)
def test_main_reports_the_legacy_shared_cache_message_on_success(
    tmp_path,
    monkeypatch,
    capsys,
    cache_status,
    expected,
):
    output = tmp_path / "output"
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile", encoding="utf-8")
    dem_path = tmp_path / "dem.tif"
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 2, 1, 1),
    ) as dem:
        dem.write(np.ones((2, 2), dtype="float32"), 1)
    config = {
        "repo_root": tmp_path,
        "config_path": config_path,
        "output_root": output,
        "rect_size": 2,
        "target_count": 1,
        "tolerance": 0,
        "scan_step": 1,
        "max_rects": 1,
        "min_spacing": 2,
        "strategy": "sequential",
        "random_seed": 7,
        "target_crs": "EPSG:3857",
        "save_csv": True,
        "save_preview_png": False,
        "save_terrain_png": False,
        "save_terrain_eps": False,
        "save_terrain_html": False,
    }
    points_path = tmp_path / "points.geojson"
    boundary_path = tmp_path / "boundary.geojson"
    paths = {
        "registered_scenario_id": "chicago",
        "output_dir": output,
        "output_csv": output / "scenario.csv",
        "output_3d_png": output / "terrain.png",
        "output_3d_html": output / "terrain.html",
        "preview_png": output / "preview.png",
        "points_shp": points_path,
        "boundary_shp": boundary_path,
        "dem_path": dem_path,
        "boundary_folder": "Chicago",
        "boundary_layer": "Chicago_Boundary",
    }
    candidate = Candidate(0, 1, 0, 0, 1, 1)
    scan_result = ScanResult((candidate,), 1, 1, True, "row-sweep-v1")
    preflight = SimpleNamespace(
        points_path=points_path,
        boundary_path=boundary_path,
        dem_path=dem_path,
        output_root=output,
    )
    export_calls = []

    class Service:
        @staticmethod
        def preflight(profile, output_root):
            del profile
            assert output_root == output
            return preflight

        @staticmethod
        def scan(received, *, progress):
            assert received is preflight
            progress(
                SelectionProgress(
                    phase="completed",
                    checked_positions=1,
                    total_positions=1,
                    candidate_count=1,
                    elapsed_seconds=0,
                    added_candidates=(candidate,),
                    removed_flat_grid_ids=(),
                    cache_status=cache_status,
                    cache_key="a" * 64,
                )
            )
            return scan_result

        @staticmethod
        def prepared_selection(received):
            assert received is preflight
            return SimpleNamespace(
                points=points,
                boundary=box(0, 0, 2, 2),
                coordinates=np.asarray([[1.0, 1.0]]),
            )

        @staticmethod
        def export(
            received,
            completed,
            selected_candidate,
            *,
            output_root,
            artifacts,
            entrypoint,
        ):
            assert received is preflight
            assert completed is scan_result
            assert selected_candidate is candidate
            assert output_root == output
            assert set(artifacts) == {"csv"}
            assert entrypoint == [
                "lte-select-sites",
                "--config",
                str(config_path),
                "--select-index",
                "1",
            ]
            export_calls.append(selected_candidate)
            run_dir = output / "chicago" / "default" / "run-1"
            run_dir.mkdir(parents=True)
            artifact = "chicago_2m_target1_tol0.csv"
            (run_dir / artifact).write_text("cell\n1\n", encoding="utf-8")
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "artifacts": [artifact],
                        "errors": [],
                    }
                ),
                encoding="utf-8",
            )
            return run_dir

    points = gpd.GeoDataFrame(
        {"cell": [1]},
        geometry=[Point(1, 1)],
        crs="EPSG:3857",
    )
    monkeypatch.setattr(select_sites, "load_experiment_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        select_sites,
        "resolve_selection_io_paths",
        lambda received, *, create_output: paths,
    )
    monkeypatch.setattr(select_sites, "load_data_catalog", lambda *args, **kwargs: object())
    monkeypatch.setattr(select_sites, "_selection_profile", lambda *args: object())
    monkeypatch.setattr(select_sites, "SelectionService", lambda catalog: Service())
    monkeypatch.setattr(
        select_sites.spatial,
        "load_and_prepare",
        lambda received: (_ for _ in ()).throw(
            AssertionError("CLI must reuse the service vector snapshot")
        ),
    )
    monkeypatch.setattr(select_sites.scenario, "verify_results", lambda *args: None)
    monkeypatch.setattr(
        select_sites.scenario,
        "choose_result",
        lambda results, index: [results[index - 1]],
    )
    monkeypatch.setattr(
        select_sites.terrain,
        "validate_dem_path",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("CLI export must not validate the DEM a second time")
        ),
    )
    monkeypatch.setattr(
        select_sites,
        "process_selected_rectangles",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("CLI must delegate extraction to SelectionService.export")
        ),
    )
    monkeypatch.setattr(
        select_sites.visualization,
        "save_preview",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("CLI must delegate preview export")
        ),
    )
    monkeypatch.setattr(
        select_sites.visualization,
        "render_3d_terrain",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("CLI must delegate terrain export")
        ),
    )
    monkeypatch.setattr(
        select_sites.io,
        "write_run_record",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("CLI must not write a second run record")
        ),
    )

    exit_code = select_sites.main(
        ["--config", str(config_path), "--select-index", "1"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert export_calls == [candidate]
    assert expected in captured.out
    assert f"{'a' * 64}.json" in captured.out
    assert "Scenario CSV:" in captured.out
    assert "Run record:" in captured.out


def test_main_maps_out_of_range_select_index_to_exit_code_two(
    tmp_path,
    monkeypatch,
    capsys,
):
    output = tmp_path / "output"
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile", encoding="utf-8")
    config = {
        "repo_root": tmp_path,
        "config_path": config_path,
        "output_root": output,
        "rect_size": 2,
        "target_count": 1,
        "tolerance": 0,
        "scan_step": 1,
        "max_rects": 1,
        "min_spacing": 2,
        "strategy": "sequential",
        "random_seed": 7,
        "target_crs": "EPSG:3857",
    }
    points_path = tmp_path / "points.geojson"
    boundary_path = tmp_path / "boundary.geojson"
    dem_path = tmp_path / "dem.tif"
    paths = {
        "registered_scenario_id": "chicago",
        "output_dir": output,
        "output_csv": output / "scenario.csv",
        "output_3d_png": output / "terrain.png",
        "output_3d_html": output / "terrain.html",
        "preview_png": output / "preview.png",
        "points_shp": points_path,
        "boundary_shp": boundary_path,
        "dem_path": dem_path,
        "boundary_folder": "Chicago",
        "boundary_layer": "Chicago_Boundary",
    }
    candidate = Candidate(0, 1, 0, 0, 1, 1)
    scan_result = ScanResult((candidate,), 1, 1, True, "row-sweep-v1")
    preflight = SimpleNamespace(
        points_path=points_path,
        boundary_path=boundary_path,
        dem_path=dem_path,
        output_root=output,
    )
    points = gpd.GeoDataFrame(
        {"cell": [1]},
        geometry=[Point(1, 1)],
        crs="EPSG:3857",
    )

    class Service:
        @staticmethod
        def preflight(profile, output_root):
            del profile
            assert output_root == output
            return preflight

        @staticmethod
        def scan(received, *, progress):
            assert received is preflight
            progress(
                SelectionProgress(
                    phase="completed",
                    checked_positions=1,
                    total_positions=1,
                    candidate_count=1,
                    elapsed_seconds=0,
                    added_candidates=(candidate,),
                    removed_flat_grid_ids=(),
                    cache_status="hit",
                    cache_key="a" * 64,
                )
            )
            return scan_result

        @staticmethod
        def prepared_selection(received):
            assert received is preflight
            return SimpleNamespace(
                points=points,
                boundary=box(0, 0, 2, 2),
                coordinates=np.asarray([[1.0, 1.0]]),
            )

    monkeypatch.setattr(select_sites, "load_experiment_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        select_sites,
        "resolve_selection_io_paths",
        lambda received, *, create_output: paths,
    )
    monkeypatch.setattr(select_sites, "load_data_catalog", lambda *args, **kwargs: object())
    monkeypatch.setattr(select_sites, "_selection_profile", lambda *args: object())
    monkeypatch.setattr(select_sites, "SelectionService", lambda catalog: Service())

    exit_code = select_sites.main(
        ["--config", str(config_path), "--select-index", "2"]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "ERROR:" in captured.err
    assert "--select-index must be between 1 and 1" in captured.err
    assert not output.exists()


def test_main_runs_shared_preflight_before_creating_output(tmp_path, monkeypatch, capsys):
    output = tmp_path / "output"
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile", encoding="utf-8")
    config = {
        "repo_root": tmp_path,
        "config_path": config_path,
        "output_root": output,
        "rect_size": 2,
        "target_count": 1,
        "tolerance": 0,
        "scan_step": 1,
        "max_rects": 1,
        "min_spacing": 2,
        "strategy": "sequential",
        "random_seed": 7,
        "target_crs": "EPSG:3857",
    }
    paths = {
        "registered_scenario_id": "chicago",
        "output_dir": output,
        "output_csv": output / "scenario.csv",
        "output_3d_png": output / "terrain.png",
        "output_3d_html": output / "terrain.html",
        "preview_png": output / "preview.png",
        "points_shp": tmp_path / "points.shp",
        "boundary_shp": tmp_path / "boundary.shp",
        "dem_path": tmp_path / "dem.tif",
        "boundary_folder": "Chicago",
        "boundary_layer": "Chicago_Boundary",
    }
    monkeypatch.setattr(select_sites, "load_experiment_config", lambda *args, **kwargs: config)

    def resolve_paths(received, *, create_output=True):
        assert received is config
        assert create_output is False
        return paths

    monkeypatch.setattr(select_sites, "resolve_selection_io_paths", resolve_paths)
    monkeypatch.setattr(select_sites, "load_data_catalog", lambda *args, **kwargs: object())
    monkeypatch.setattr(select_sites, "_selection_profile", lambda *args: object())

    class RejectingService:
        def __init__(self, catalog):
            assert catalog is not None

        @staticmethod
        def preflight(profile, output_root):
            assert profile is not None
            assert output_root == output
            raise ValueError("preflight rejected")

    monkeypatch.setattr(select_sites, "SelectionService", RejectingService)

    exit_code = select_sites.main(["--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "preflight rejected" in captured.err
    assert not output.exists()


def test_main_schema_v2_profile_resolves_inputs_only_through_preflight(
    tmp_path,
    monkeypatch,
    capsys,
):
    output = tmp_path / "output"
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile", encoding="utf-8")
    config = {
        "repo_root": tmp_path,
        "config_path": config_path,
        "profile_id": "default",
        "scenario_id": "chicago",
        "points_dataset_id": "points",
        "output_root": output,
        "rect_size": 2,
        "target_count": 1,
        "tolerance": 0,
        "scan_mode": "fast",
        "scan_step": 1,
        "max_rects": 1,
        "min_spacing": 2,
        "strategy": "sequential",
        "random_seed": 7,
        "target_crs": "EPSG:3857",
    }
    monkeypatch.setattr(select_sites, "load_experiment_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        select_sites,
        "resolve_selection_io_paths",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("schema-v2 must not resolve legacy path fields")
        ),
    )
    monkeypatch.setattr(select_sites, "load_data_catalog", lambda *args, **kwargs: object())
    monkeypatch.setattr(select_sites, "_selection_profile", lambda *args: object())

    class RejectingService:
        def __init__(self, catalog):
            assert catalog is not None

        @staticmethod
        def preflight(profile, output_root):
            assert profile is not None
            assert output_root == output
            raise ValueError("profile preflight rejected")

    monkeypatch.setattr(select_sites, "SelectionService", RejectingService)

    exit_code = select_sites.main(["--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "profile preflight rejected" in captured.err
    assert not output.exists()


def test_selection_profile_reuses_the_loader_snapshot(tmp_path, monkeypatch):
    profile_path = tmp_path / "configs" / "default.yaml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "profile": {
                    "id": "default",
                    "display_name": "Default",
                    "scenario_id": "chicago",
                },
                "inputs": {"points_dataset_id": "points"},
                "experiment": {"random_seed": 7},
                "spatial": {
                    "target_crs": "EPSG:3857",
                    "rectangle_size_m": 2,
                    "target_base_station_count": 1,
                    "count_tolerance": 0,
                },
                "scan": {
                    "mode": "fast",
                    "strategy": "sequential",
                    "step_m": 1,
                    "max_rectangles": 1,
                    "minimum_center_spacing_m": 2,
                },
                "outputs": {"root": "results"},
                "figures": {"preset": "publication"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = load_experiment_config(profile_path, repo_root=tmp_path)
    monkeypatch.setattr(
        select_sites,
        "load_profile",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("profile must not be parsed twice")
        ),
    )

    profile = select_sites._selection_profile(config, object(), "chicago")

    assert profile.profile_id == "default"
    assert profile.source_path == profile_path
