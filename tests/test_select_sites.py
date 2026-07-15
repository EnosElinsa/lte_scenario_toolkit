from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import yaml
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from lte_scenario_toolkit import select_sites
from lte_scenario_toolkit.config import load_experiment_config
from lte_scenario_toolkit.data_catalog import load_data_catalog
from lte_scenario_toolkit.select_sites import process_selected_rectangles

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
