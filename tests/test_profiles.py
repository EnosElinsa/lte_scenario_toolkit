import os
import re
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

import lte_scenario_toolkit.profiles as profiles_module
from lte_scenario_toolkit.config import load_experiment_config
from lte_scenario_toolkit.data_catalog import ConcurrentCatalogUpdateError
from lte_scenario_toolkit.profiles import (
    DEFAULT_PROFILE_VALUES,
    ConcurrentProfileUpdateError,
    ExperimentProfile,
    FigureSettings,
    OutputSettings,
    ProfileStore,
    dump_profile,
    load_profile,
)

LEGACY_FIXTURES = Path(__file__).parent / "fixtures" / "legacy"


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


def test_legacy_profile_conversion_preserves_effective_values_and_source(tmp_path):
    source = LEGACY_FIXTURES / "experiment-v1.yaml"
    original = source.read_bytes()

    legacy = profiles_module.load_legacy_profile(source, repo_root=tmp_path)

    with pytest.raises(TypeError):
        legacy["rect_size"] = 1
    assert legacy.source_path == source.resolve()
    assert legacy.source_sha256 == sha256(original).hexdigest()
    assert legacy.source_revision == f"sha256:{sha256(original).hexdigest()}"
    converted = profiles_module.convert_legacy_profile(
        legacy,
        profile_id="legacy",
        scenario_id="chicago",
        points_dataset_id="points",
    )

    assert converted is profiles_module.validate_profile(converted)
    assert converted.display_name == "legacy_chicago_2400m_target24"
    assert converted.random_seed == legacy["random_seed"] == 17
    assert converted.target_crs == legacy["target_crs"] == "EPSG:3857"
    assert converted.rect_size == legacy["rect_size"] == 2400
    assert converted.target_count == legacy["target_count"] == 24
    assert converted.tolerance == legacy["tolerance"] == 2
    assert converted.scan_mode == DEFAULT_PROFILE_VALUES["scan_mode"]
    assert converted.strategy == legacy["strategy"] == "sequential"
    assert converted.scan_step == legacy["scan_step"] == 30
    assert converted.max_rects == legacy["max_rects"] == 12
    assert converted.min_spacing == legacy["min_spacing"] == 1800
    assert converted.output_root == legacy["output_root"]
    assert converted.outputs == OutputSettings(
        save_csv=True,
        save_preview_png=False,
        save_terrain_png=True,
        save_terrain_eps=False,
        save_terrain_html=True,
    )
    assert converted.source_path == source.resolve()
    assert converted.is_legacy_preview is False
    assert converted.legacy_source is None
    assert source.read_bytes() == original


def test_explicit_legacy_save_writes_v2_without_rewriting_source(
    profile_repository,
):
    repo_root, catalog_path = profile_repository
    source = LEGACY_FIXTURES / "experiment-v1.yaml"
    original = source.read_bytes()
    legacy = profiles_module.load_legacy_profile(source, repo_root=repo_root)
    converted = profiles_module.convert_legacy_profile(
        legacy,
        profile_id="legacy",
        scenario_id="chicago",
        points_dataset_id="points",
    )

    destination = ProfileStore(repo_root, catalog_path).save(
        converted,
        set_default=True,
    )

    assert source.read_bytes() == original
    assert destination == repo_root / "configs" / "chicago" / "legacy.yaml"
    document = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert document["schema_version"] == 2
    reloaded = load_profile(destination, repo_root=repo_root)
    assert reloaded.rect_size == legacy["rect_size"]
    assert reloaded.target_count == legacy["target_count"]
    assert reloaded.tolerance == legacy["tolerance"]
    assert reloaded.scan_step == legacy["scan_step"]
    assert reloaded.min_spacing == legacy["min_spacing"]
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert catalog["scenarios"][0]["config_path"] == (
        "configs/chicago/legacy.yaml"
    )


def test_profile_store_save_closes_catalog_legacy_migration(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    source = repo_root / "configs" / "example.yaml"
    source.parent.mkdir()
    source.write_bytes((LEGACY_FIXTURES / "experiment-v1.yaml").read_bytes())
    original = source.read_bytes()
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    catalog["scenarios"][0]["config_path"] = "configs/example.yaml"
    catalog_path.write_text(
        yaml.safe_dump(catalog, sort_keys=False),
        encoding="utf-8",
    )
    store = ProfileStore(repo_root, catalog_path)
    real_convert = profiles_module.convert_legacy_profile
    conversions = []

    def record_conversion(legacy, **kwargs):
        conversions.append((legacy, kwargs))
        return real_convert(legacy, **kwargs)

    monkeypatch.setattr(profiles_module, "convert_legacy_profile", record_conversion)
    legacy_view = store.discover("chicago")[0]

    assert conversions == []
    assert legacy_view.is_legacy_preview is True
    assert legacy_view.legacy_source is not None
    assert legacy_view.legacy_source.source_path == source.resolve()
    assert legacy_view.legacy_source.source_sha256 == sha256(original).hexdigest()
    assert legacy_view.source_path == source.resolve()
    assert source.read_bytes() == original
    destination = store.save(legacy_view, overwrite=True)

    assert len(conversions) == 1
    assert conversions[0][0].source_sha256 == sha256(original).hexdigest()
    assert destination == repo_root / "configs" / "chicago" / "example.yaml"
    assert source.read_bytes() == original
    assert load_profile(destination, repo_root=repo_root).profile_id == "example"
    migrated_catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert migrated_catalog["scenarios"][0]["config_path"] == (
        "configs/chicago/example.yaml"
    )
    discovered = store.discover("chicago")
    assert [profile.profile_id for profile in discovered] == ["example"]
    assert discovered[0].source_path == destination.resolve()
    assert discovered[0].is_legacy_preview is False


def test_catalog_legacy_migration_rejects_external_change_since_discovery(
    profile_repository,
):
    repo_root, catalog_path = profile_repository
    source = repo_root / "configs" / "legacy-default.yaml"
    source.parent.mkdir()
    source.write_bytes((LEGACY_FIXTURES / "experiment-v1.yaml").read_bytes())
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    catalog["scenarios"][0]["config_path"] = "configs/legacy-default.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    original_catalog = catalog_path.read_bytes()
    store = ProfileStore(repo_root, catalog_path)
    legacy_view = store.discover("chicago")[0]
    destination = repo_root / "configs" / "chicago" / "legacy-default.yaml"

    source.write_text(
        source.read_text(encoding="utf-8") + "\n# edited after opening GUI\n",
        encoding="utf-8",
    )

    with pytest.raises(ConcurrentProfileUpdateError, match="changed since discovery"):
        store.save(legacy_view, overwrite=True)

    assert not destination.exists()
    assert source.read_text(encoding="utf-8").endswith("# edited after opening GUI\n")
    assert catalog_path.read_bytes() == original_catalog


def test_catalog_legacy_migration_in_place_preserves_revision_and_effective_values(
    profile_repository,
):
    repo_root, catalog_path = profile_repository
    source = repo_root / "configs" / "chicago" / "legacy-default.yaml"
    source.parent.mkdir(parents=True)
    source.write_bytes((LEGACY_FIXTURES / "experiment-v1.yaml").read_bytes())
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    catalog["scenarios"][0]["config_path"] = (
        "configs/chicago/legacy-default.yaml"
    )
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    original_catalog = catalog_path.read_bytes()
    store = ProfileStore(repo_root, catalog_path)
    legacy_view = store.discover("chicago")[0]
    revision = legacy_view.legacy_source.source_revision

    destination = store.save(legacy_view, overwrite=True)

    assert destination == source.resolve()
    migrated = load_profile(source, repo_root=repo_root)
    assert migrated.rect_size == 2400
    assert migrated.target_count == 24
    assert migrated.tolerance == 2
    assert migrated.scan_step == 30
    assert migrated.min_spacing == 1800
    assert migrated.is_legacy_preview is False
    assert revision.startswith("sha256:")
    assert catalog_path.read_bytes() == original_catalog


def test_catalog_legacy_migration_in_place_rejects_change_since_discovery(
    profile_repository,
):
    repo_root, catalog_path = profile_repository
    source = repo_root / "configs" / "chicago" / "legacy-default.yaml"
    source.parent.mkdir(parents=True)
    source.write_bytes((LEGACY_FIXTURES / "experiment-v1.yaml").read_bytes())
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    catalog["scenarios"][0]["config_path"] = (
        "configs/chicago/legacy-default.yaml"
    )
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    original_catalog = catalog_path.read_bytes()
    store = ProfileStore(repo_root, catalog_path)
    legacy_view = store.discover("chicago")[0]

    source.write_text(
        source.read_text(encoding="utf-8") + "\n# edited after opening GUI\n",
        encoding="utf-8",
    )

    with pytest.raises(ConcurrentProfileUpdateError, match="changed since discovery"):
        store.save(legacy_view, overwrite=True)

    assert source.read_text(encoding="utf-8").endswith("# edited after opening GUI\n")
    assert catalog_path.read_bytes() == original_catalog


def test_catalog_legacy_migration_in_place_detects_change_during_save(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    source = repo_root / "configs" / "chicago" / "legacy-default.yaml"
    source.parent.mkdir(parents=True)
    source.write_bytes((LEGACY_FIXTURES / "experiment-v1.yaml").read_bytes())
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    catalog["scenarios"][0]["config_path"] = (
        "configs/chicago/legacy-default.yaml"
    )
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    original_catalog = catalog_path.read_bytes()
    store = ProfileStore(repo_root, catalog_path)
    legacy_view = store.discover("chicago")[0]
    real_dump = profiles_module.dump_profile

    def dump_then_change_source(profile, path):
        result = real_dump(profile, path)
        source.write_text(
            source.read_text(encoding="utf-8") + "\n# concurrent writer\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(profiles_module, "dump_profile", dump_then_change_source)

    with pytest.raises(ConcurrentProfileUpdateError, match="during migration"):
        store.save(legacy_view, overwrite=True)

    assert source.read_text(encoding="utf-8").endswith("# concurrent writer\n")
    assert catalog_path.read_bytes() == original_catalog


def test_profile_store_legacy_migration_rolls_back_if_source_changes(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    source = repo_root / "configs" / "example.yaml"
    source.parent.mkdir()
    source.write_bytes((LEGACY_FIXTURES / "experiment-v1.yaml").read_bytes())
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    catalog["scenarios"][0]["config_path"] = "configs/example.yaml"
    catalog_path.write_text(
        yaml.safe_dump(catalog, sort_keys=False),
        encoding="utf-8",
    )
    store = ProfileStore(repo_root, catalog_path)
    legacy_view = store.discover("chicago")[0]
    destination = repo_root / "configs" / "chicago" / "example.yaml"
    real_dump = profiles_module.dump_profile

    def dump_then_change_source(profile, path):
        real_dump(profile, path)
        source.write_text(
            source.read_text(encoding="utf-8") + "\n# external change\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(profiles_module, "dump_profile", dump_then_change_source)

    with pytest.raises(ConcurrentProfileUpdateError, match="during migration"):
        store.save(legacy_view, overwrite=True)

    assert not destination.exists()
    assert source.read_text(encoding="utf-8").endswith("# external change\n")
    unchanged_catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert unchanged_catalog["scenarios"][0]["config_path"] == (
        "configs/example.yaml"
    )


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


def _catalog_dataset(
    dataset_id: str,
    role: str,
    path: str,
    entrypoint: str,
) -> dict:
    dataset = {
        "dataset_id": dataset_id,
        "role": role,
        "path": path,
        "entrypoint": entrypoint,
        "source_url": "https://example.test/data",
        "provider": "Example provider",
        "license": "CC0-1.0",
        "download_date": "2026-07-16",
        "crs": "EPSG:3857",
        "spatial_resolution": "fixture",
        "notes": "profile store fixture",
    }
    if role == "boundary":
        dataset.update(
            {
                "geometry_type": "Polygon",
                "feature_count": 1,
                "redistribution_confirmed": True,
            }
        )
    elif role == "dem":
        dataset.update(
            {
                "external": False,
                "earth_engine_collection": "EXAMPLE/DEM",
                "band": "elevation",
                "units": "metres",
                "vertical_datum": "NAVD88",
                "native_scale_m": 1,
                "export_crs": "EPSG:3857",
                "export_prefix": "fixture-dem",
                "drive_folder": "fixture-exports",
            }
        )
    return dataset


@pytest.fixture
def profile_repository(tmp_path):
    entrypoints = {
        "points": (
            tmp_path
            / "points_shp"
            / "USA_Clear_LTE_Base_Station"
            / "USA_Clear_LTE_Base_Station.shp"
        ),
        "boundary": tmp_path / "boundary_shp" / "Chicago" / "Chicago.shp",
        "dem": tmp_path / "dem" / "Chicago" / "elevation.tif",
    }
    for dataset_id, path in entrypoints.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dataset_id, encoding="utf-8")

    document = {
        "schema_version": 2,
        "datasets": [
            _catalog_dataset(
                "points",
                "points",
                "points_shp",
                "points_shp/USA_Clear_LTE_Base_Station/USA_Clear_LTE_Base_Station.shp",
            ),
            _catalog_dataset(
                "boundary",
                "boundary",
                "boundary_shp",
                "boundary_shp/Chicago/Chicago.shp",
            ),
            _catalog_dataset(
                "dem",
                "dem",
                "dem",
                "dem/Chicago/elevation.tif",
            ),
        ],
        "scenarios": [
            {
                "scenario_id": "chicago",
                "display_name": "Chicago",
                "boundary_dataset_id": "boundary",
                "dem_dataset_id": "dem",
                "config_path": None,
            },
            {
                "scenario_id": "new-york-city",
                "display_name": "New York City",
                "boundary_dataset_id": "boundary",
                "dem_dataset_id": "dem",
                "config_path": None,
            },
        ],
    }
    catalog_path = tmp_path / "data" / "datasets.yaml"
    catalog_path.parent.mkdir()
    catalog_path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return tmp_path, catalog_path


def make_profile(
    repo_root: Path,
    *,
    profile_id: str = "chicago-default",
    display_name: str = "Chicago default",
    scenario_id: str = "chicago",
    points_dataset_id: str = "points",
) -> ExperimentProfile:
    return ExperimentProfile(
        schema_version=2,
        profile_id=profile_id,
        display_name=display_name,
        scenario_id=scenario_id,
        points_dataset_id=points_dataset_id,
        random_seed=7,
        target_crs="EPSG:3857",
        rect_size=2000,
        target_count=20,
        tolerance=1,
        scan_mode="complete",
        strategy="uniform",
        scan_step=25,
        max_rects=40,
        min_spacing=1500,
        output_root=repo_root / "results",
        outputs=OutputSettings(
            save_csv=True,
            save_preview_png=False,
            save_terrain_png=True,
            save_terrain_eps=False,
            save_terrain_html=True,
        ),
        figure=FigureSettings(
            preset="preview",
            colormap="viridis",
            dpi=144,
            azimuth_deg=-45.0,
            elevation_deg=25.0,
            vertical_exaggeration=1.5,
            station_color="blue",
            station_marker_size=12.5,
            title="Candidate terrain",
        ),
    )


def _write_external_catalog(
    catalog,
    save_catalog,
    *,
    scenario_id: str,
    config_path: str,
    note: str,
) -> bytes:
    document = deepcopy(catalog.document)
    document["datasets"][0]["notes"] = note
    for scenario in document["scenarios"]:
        if scenario["scenario_id"] == scenario_id:
            scenario["config_path"] = config_path
            break
    save_catalog(catalog, document)
    stat = catalog.path.stat()
    if stat.st_mtime_ns == catalog.loaded_mtime_ns:
        os.utime(
            catalog.path,
            ns=(stat.st_atime_ns, catalog.loaded_mtime_ns + 1_000_000_000),
        )
    return catalog.path.read_bytes()


def test_dump_profile_is_deterministic_atomic_and_round_trips(profile_repository):
    repo_root, _ = profile_repository
    profile = make_profile(repo_root)
    path = repo_root / "configs" / "chicago" / "chicago-default.yaml"

    assert dump_profile(profile, path) == path.resolve()
    first = path.read_bytes()
    assert dump_profile(profile, path) == path.resolve()
    second = path.read_bytes()

    assert first == second
    assert first.endswith(b"\n")
    assert list(yaml.safe_load(first).keys()) == [
        "schema_version",
        "profile",
        "inputs",
        "experiment",
        "spatial",
        "scan",
        "outputs",
        "figures",
    ]
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))
    loaded = load_profile(path, repo_root=repo_root)
    assert replace(loaded, source_path=None) == profile


def test_repository_profile_loaders_infer_root_from_nested_configs(
    profile_repository,
):
    repo_root, catalog_path = profile_repository
    saved = ProfileStore(repo_root, catalog_path).save(make_profile(repo_root))

    loaded = load_profile(saved)
    public_config = load_experiment_config(saved)

    assert loaded.output_root == repo_root / "results"
    assert public_config["output_root"] == repo_root / "results"
    assert public_config["repo_root"] == repo_root


def test_profile_root_inference_preserves_direct_configs_and_standalone_files(
    tmp_path,
):
    repository = tmp_path / "repository"
    direct = repository / "configs" / "example.yaml"
    direct.parent.mkdir(parents=True)
    write_profile(direct)
    standalone = tmp_path / "standalone" / "profile.yaml"
    standalone.parent.mkdir()
    write_profile(standalone)

    assert load_profile(direct).output_root == repository / "results"
    assert load_profile(standalone).output_root == standalone.parent / "results"


def test_profile_store_save_sets_posix_default_and_refuses_overwrite(
    profile_repository,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    profile = make_profile(repo_root)

    saved = store.save(profile, set_default=True)

    assert saved == (repo_root / "configs/chicago/chicago-default.yaml").resolve()
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert catalog["scenarios"][0]["config_path"] == (
        "configs/chicago/chicago-default.yaml"
    )
    with pytest.raises(FileExistsError):
        store.save(profile)
    with pytest.raises(ValueError, match="default profile"):
        store.delete(saved)
    assert saved.is_file()


def test_profile_store_rolls_back_new_profile_when_catalog_save_fails(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    original_catalog = catalog_path.read_bytes()
    store = ProfileStore(repo_root, catalog_path)
    destination = repo_root / "configs/chicago/chicago-default.yaml"

    def fail_save(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(profiles_module, "save_data_catalog", fail_save)
    with pytest.raises(OSError, match="boom"):
        store.save(make_profile(repo_root), set_default=True)

    assert not destination.exists()
    assert catalog_path.read_bytes() == original_catalog


def test_profile_store_restores_catalog_if_save_replaces_it_then_raises(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    original_catalog = catalog_path.read_bytes()
    store = ProfileStore(repo_root, catalog_path)
    destination = repo_root / "configs/chicago/chicago-default.yaml"

    def replace_then_fail(catalog, document):
        catalog.path.write_text(
            yaml.safe_dump(document, sort_keys=False),
            encoding="utf-8",
        )
        raise OSError("boom after replace")

    monkeypatch.setattr(profiles_module, "save_data_catalog", replace_then_fail)
    with pytest.raises(OSError, match="boom after replace"):
        store.save(make_profile(repo_root), set_default=True)

    assert not destination.exists()
    assert catalog_path.read_bytes() == original_catalog


def test_profile_store_restores_catalog_if_save_deletes_it_then_raises(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    original_catalog = catalog_path.read_bytes()
    store = ProfileStore(repo_root, catalog_path)
    destination = repo_root / "configs/chicago/chicago-default.yaml"

    def delete_then_fail(catalog, document):
        catalog.path.unlink()
        raise OSError("boom")

    monkeypatch.setattr(profiles_module, "save_data_catalog", delete_then_fail)
    with pytest.raises(OSError, match="boom"):
        store.save(make_profile(repo_root), set_default=True)

    assert not destination.exists()
    assert catalog_path.read_bytes() == original_catalog
    restored = profiles_module.load_data_catalog(catalog_path, repo_root=repo_root)
    assert restored.scenario("chicago")["config_path"] is None


def test_profile_store_preserves_external_catalog_on_concurrent_update(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    destination = repo_root / "configs/chicago/chicago-default.yaml"
    real_save = profiles_module.save_data_catalog
    external_state = {}

    def write_external_then_attempt_stale_save(catalog, document):
        external_document = deepcopy(catalog.document)
        external_document["datasets"][0]["notes"] = "external concurrent update"
        real_save(catalog, external_document)
        external_state["bytes"] = catalog.path.read_bytes()
        return real_save(catalog, document)

    monkeypatch.setattr(
        profiles_module,
        "save_data_catalog",
        write_external_then_attempt_stale_save,
    )

    with pytest.raises(ConcurrentCatalogUpdateError, match="changed since"):
        store.save(make_profile(repo_root), set_default=True)

    assert destination.is_file()
    assert load_profile(destination, repo_root=repo_root).profile_id == (
        "chicago-default"
    )
    assert catalog_path.read_bytes() == external_state["bytes"]
    external_catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert external_catalog["datasets"][0]["notes"] == "external concurrent update"
    assert external_catalog["scenarios"][0]["config_path"] is None


def test_profile_store_keeps_new_target_referenced_by_concurrent_catalog(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    destination = repo_root / "configs/chicago/chicago-default.yaml"
    relative_destination = "configs/chicago/chicago-default.yaml"
    real_save = profiles_module.save_data_catalog
    external_state = {}

    def reference_target_then_attempt_stale_save(catalog, document):
        external_state["bytes"] = _write_external_catalog(
            catalog,
            real_save,
            scenario_id="chicago",
            config_path=relative_destination,
            note="external owner references new target",
        )
        return real_save(catalog, document)

    monkeypatch.setattr(
        profiles_module,
        "save_data_catalog",
        reference_target_then_attempt_stale_save,
    )

    with pytest.raises(ConcurrentCatalogUpdateError, match="changed since"):
        store.save(make_profile(repo_root), set_default=True)

    assert catalog_path.read_bytes() == external_state["bytes"]
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert catalog["scenarios"][0]["config_path"] == relative_destination
    assert destination.is_file()
    assert load_profile(destination, repo_root=repo_root).profile_id == (
        "chicago-default"
    )


def test_profile_store_preserves_external_catalog_after_save_failure(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    destination = repo_root / "configs/chicago/chicago-default.yaml"
    real_save = profiles_module.save_data_catalog
    external_state = {}

    def write_external_then_fail(catalog, document):
        external_document = deepcopy(catalog.document)
        external_document["datasets"][0]["notes"] = "external replacement"
        real_save(catalog, external_document)
        external_state["bytes"] = catalog.path.read_bytes()
        raise OSError("failure after external replacement")

    monkeypatch.setattr(profiles_module, "save_data_catalog", write_external_then_fail)

    with pytest.raises(OSError, match="external replacement") as error:
        store.save(make_profile(repo_root), set_default=True)

    assert destination.is_file()
    assert load_profile(destination, repo_root=repo_root).profile_id == (
        "chicago-default"
    )
    assert catalog_path.read_bytes() == external_state["bytes"]
    assert any("rollback skipped" in note for note in error.value.__notes__)


def test_profile_store_keeps_new_profile_when_catalog_rollback_fails(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    destination = repo_root / "configs/chicago/chicago-default.yaml"
    real_save = profiles_module.save_data_catalog
    real_restore = profiles_module._atomic_restore_bytes

    def save_then_fail(catalog, document):
        real_save(catalog, document)
        raise OSError("after write")

    def fail_catalog_restore(path, content):
        if Path(path).resolve() == catalog_path.resolve():
            raise OSError("rollback failed")
        return real_restore(path, content)

    monkeypatch.setattr(profiles_module, "save_data_catalog", save_then_fail)
    monkeypatch.setattr(
        profiles_module,
        "_atomic_restore_bytes",
        fail_catalog_restore,
    )

    with pytest.raises(OSError, match="after write") as error:
        store.save(make_profile(repo_root), set_default=True)

    assert any("rollback failed" in note for note in error.value.__notes__)
    assert any("target was retained" in note for note in error.value.__notes__)
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    configured = repo_root / catalog["scenarios"][0]["config_path"]
    assert configured == destination
    assert configured.is_file()
    assert load_profile(configured, repo_root=repo_root).profile_id == (
        "chicago-default"
    )


def test_profile_store_recognizes_catalog_restored_before_restore_error(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    original_catalog = catalog_path.read_bytes()
    store = ProfileStore(repo_root, catalog_path)
    destination = repo_root / "configs/chicago/chicago-default.yaml"
    real_save = profiles_module.save_data_catalog
    real_restore = profiles_module._atomic_restore_bytes

    def save_then_fail(catalog, document):
        real_save(catalog, document)
        raise OSError("after write")

    def restore_then_fail(path, content):
        real_restore(path, content)
        if Path(path).resolve() == catalog_path.resolve():
            raise OSError("after restore")

    monkeypatch.setattr(profiles_module, "save_data_catalog", save_then_fail)
    monkeypatch.setattr(
        profiles_module,
        "_atomic_restore_bytes",
        restore_then_fail,
    )

    with pytest.raises(OSError, match="after write") as error:
        store.save(make_profile(repo_root), set_default=True)

    assert any("after restore" in note for note in error.value.__notes__)
    assert catalog_path.read_bytes() == original_catalog
    assert not destination.exists()


def test_profile_store_keeps_overwrite_when_catalog_rollback_is_uncertain(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    destination = store.save(make_profile(repo_root))
    real_save = profiles_module.save_data_catalog
    real_restore = profiles_module._atomic_restore_bytes

    def save_then_fail(catalog, document):
        real_save(catalog, document)
        raise OSError("after write")

    def fail_catalog_restore(path, content):
        if Path(path).resolve() == catalog_path.resolve():
            raise OSError("rollback failed")
        return real_restore(path, content)

    monkeypatch.setattr(profiles_module, "save_data_catalog", save_then_fail)
    monkeypatch.setattr(
        profiles_module,
        "_atomic_restore_bytes",
        fail_catalog_restore,
    )

    with pytest.raises(OSError, match="after write") as error:
        store.save(
            make_profile(repo_root, display_name="Changed"),
            overwrite=True,
            set_default=True,
        )

    assert any("target was retained" in note for note in error.value.__notes__)
    assert destination.is_file()
    assert load_profile(destination, repo_root=repo_root).display_name == "Changed"


def test_profile_store_removes_new_profile_if_dump_writes_then_raises(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    destination = repo_root / "configs/chicago/chicago-default.yaml"
    original_dump = profiles_module.dump_profile

    def write_then_fail(profile, path):
        original_dump(profile, path)
        raise OSError("write boom")

    monkeypatch.setattr(profiles_module, "dump_profile", write_then_fail)
    with pytest.raises(OSError, match="write boom"):
        store.save(make_profile(repo_root))

    assert not destination.exists()


def test_profile_store_restores_overwrite_if_dump_writes_then_raises(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    destination = store.save(make_profile(repo_root))
    original_bytes = destination.read_bytes()
    original_dump = profiles_module.dump_profile

    def write_then_fail(profile, path):
        original_dump(profile, path)
        raise OSError("write boom")

    monkeypatch.setattr(profiles_module, "dump_profile", write_then_fail)
    with pytest.raises(OSError, match="write boom"):
        store.save(
            make_profile(repo_root, display_name="Changed"),
            overwrite=True,
        )

    assert destination.read_bytes() == original_bytes
    assert load_profile(destination, repo_root=repo_root).display_name == (
        "Chicago default"
    )


def test_profile_store_removes_copy_if_dump_writes_then_raises(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    source = store.save(make_profile(repo_root))
    source_bytes = source.read_bytes()
    destination = repo_root / "configs/chicago/chicago-copy.yaml"
    original_dump = profiles_module.dump_profile

    def write_then_fail(profile, path):
        original_dump(profile, path)
        raise OSError("write boom")

    monkeypatch.setattr(profiles_module, "dump_profile", write_then_fail)
    with pytest.raises(OSError, match="write boom"):
        store.copy(source, "chicago-copy", "Chicago copy")

    assert source.read_bytes() == source_bytes
    assert not destination.exists()


def test_profile_store_removes_rename_target_if_dump_writes_then_raises(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    source = store.save(make_profile(repo_root), set_default=True)
    source_bytes = source.read_bytes()
    original_catalog = catalog_path.read_bytes()
    destination = repo_root / "configs/chicago/chicago-renamed.yaml"
    original_dump = profiles_module.dump_profile

    def write_then_fail(profile, path):
        original_dump(profile, path)
        raise OSError("write boom")

    monkeypatch.setattr(profiles_module, "dump_profile", write_then_fail)
    with pytest.raises(OSError, match="write boom"):
        store.rename(source, "chicago-renamed", "Chicago renamed")

    assert source.read_bytes() == source_bytes
    assert not destination.exists()
    assert catalog_path.read_bytes() == original_catalog


def test_profile_store_preserves_external_change_after_dump_failure(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    destination = repo_root / "configs/chicago/chicago-default.yaml"
    sentinel = b"external owner\n"
    original_dump = profiles_module.dump_profile

    def write_change_then_fail(profile, path):
        original_dump(profile, path)
        Path(path).write_bytes(sentinel)
        raise OSError("write boom")

    monkeypatch.setattr(profiles_module, "dump_profile", write_change_then_fail)
    with pytest.raises(OSError, match="write boom") as error:
        store.save(make_profile(repo_root))

    assert destination.read_bytes() == sentinel
    assert any("changed" in note for note in error.value.__notes__)


@pytest.mark.parametrize("overwrite", [False, True])
def test_profile_store_handles_deleted_target_after_dump_failure(
    profile_repository,
    monkeypatch,
    overwrite,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    destination = repo_root / "configs/chicago/chicago-default.yaml"
    original_bytes = None
    if overwrite:
        destination = store.save(make_profile(repo_root))
        original_bytes = destination.read_bytes()
    original_dump = profiles_module.dump_profile

    def write_delete_then_fail(profile, path):
        original_dump(profile, path)
        Path(path).unlink()
        raise OSError("write boom")

    monkeypatch.setattr(profiles_module, "dump_profile", write_delete_then_fail)
    with pytest.raises(OSError, match="write boom"):
        store.save(
            make_profile(repo_root, display_name="Changed"),
            overwrite=overwrite,
        )

    if overwrite:
        assert destination.read_bytes() == original_bytes
    else:
        assert not destination.exists()


def test_profile_store_discover_is_recursive_sorted_and_filterable(
    profile_repository,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    store.save(make_profile(repo_root, profile_id="zeta", display_name="Zeta"))
    store.save(make_profile(repo_root, profile_id="alpha", display_name="Alpha"))
    store.save(
        make_profile(
            repo_root,
            profile_id="metro",
            display_name="Metro",
            scenario_id="new-york-city",
        )
    )

    discovered = store.discover()

    assert [profile.profile_id for profile in discovered] == ["alpha", "zeta", "metro"]
    assert all(isinstance(profile, ExperimentProfile) for profile in discovered)
    assert [profile.profile_id for profile in store.discover("chicago")] == [
        "alpha",
        "zeta",
    ]


def test_profile_store_discovers_catalog_legacy_among_mixed_yaml(
    profile_repository,
):
    repo_root, catalog_path = profile_repository
    configs = repo_root / "configs"
    configs.mkdir()
    legacy_path = configs / "chicago-default.yaml"
    legacy_path.write_bytes(
        (LEGACY_FIXTURES / "experiment-v1.yaml").read_bytes()
    )
    unrelated = configs / "unrelated.yaml"
    unrelated.write_text("this: legacy file is unrelated\n", encoding="utf-8")
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    catalog["scenarios"][0]["config_path"] = "configs/chicago-default.yaml"
    catalog_path.write_text(
        yaml.safe_dump(catalog, sort_keys=False),
        encoding="utf-8",
    )
    store = ProfileStore(repo_root, catalog_path)
    store.save(
        make_profile(
            repo_root,
            profile_id="metro",
            display_name="Metro",
            scenario_id="new-york-city",
        )
    )
    original = legacy_path.read_bytes()

    discovered = store.discover()

    assert [profile.profile_id for profile in discovered] == [
        "chicago-default",
        "metro",
    ]
    legacy = discovered[0]
    assert legacy.scenario_id == "chicago"
    assert legacy.points_dataset_id == "points"
    assert legacy.rect_size == 2400
    assert legacy.source_path == legacy_path.resolve()
    assert legacy.legacy_source.catalog_owner_scenario_id == "chicago"
    assert [profile.profile_id for profile in store.discover("chicago")] == [
        "chicago-default"
    ]
    assert [profile.profile_id for profile in store.discover()] == [
        "chicago-default",
        "metro",
    ]
    assert legacy_path.read_bytes() == original
    assert unrelated.read_text(encoding="utf-8") == (
        "this: legacy file is unrelated\n"
    )


def test_profile_store_catalog_owned_legacy_validates_registered_paths(
    profile_repository,
):
    repo_root, catalog_path = profile_repository
    source = repo_root / "configs" / "legacy-default.yaml"
    source.parent.mkdir()
    source.write_bytes((LEGACY_FIXTURES / "experiment-v1.yaml").read_bytes())
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    catalog["scenarios"][0]["config_path"] = "configs/legacy-default.yaml"
    dem = next(item for item in catalog["datasets"] if item["role"] == "dem")
    dem["entrypoint"] = "dem/Chicago/other.tif"
    other_dem = repo_root / dem["entrypoint"]
    other_dem.parent.mkdir(parents=True, exist_ok=True)
    other_dem.write_text("dem", encoding="utf-8")
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="catalog-owned legacy.*registered paths"):
        ProfileStore(repo_root, catalog_path).discover()


def _nondefault_legacy_fixture(profile_repository):
    repo_root, catalog_path = profile_repository
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    catalog["scenarios"] = [catalog["scenarios"][0]]
    catalog["scenarios"][0]["config_path"] = None
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    source = repo_root / "configs" / "chicago" / "legacy-secondary.yaml"
    source.parent.mkdir(parents=True)
    source.write_bytes((LEGACY_FIXTURES / "experiment-v1.yaml").read_bytes())
    return repo_root, catalog_path, source


def test_profile_store_discovers_and_migrates_matching_nondefault_legacy(
    profile_repository,
):
    repo_root, catalog_path, source = _nondefault_legacy_fixture(profile_repository)
    store = ProfileStore(repo_root, catalog_path)

    discovered = store.discover()

    assert [profile.profile_id for profile in discovered] == ["legacy-secondary"]
    preview = discovered[0]
    assert preview.scenario_id == "chicago"
    assert preview.points_dataset_id == "points"
    assert preview.is_legacy_preview is True
    assert preview.legacy_source.source_path == source.resolve()
    assert preview.legacy_source.catalog_owner_scenario_id is None
    assert store.discover("chicago") == [preview]
    assert store.discover("new-york-city") == []

    original = source.read_bytes()
    original_catalog = catalog_path.read_bytes()
    source.write_bytes(original + b"\n# external change\n")
    with pytest.raises(ConcurrentProfileUpdateError, match="changed since discovery"):
        store.save(preview, overwrite=True)
    assert source.read_bytes().endswith(b"# external change\n")
    assert catalog_path.read_bytes() == original_catalog

    source.write_bytes(original)
    refreshed = store.discover("chicago")[0]
    destination = store.save(refreshed, overwrite=True)

    assert destination == source.resolve()
    migrated = load_profile(source, repo_root=repo_root)
    assert migrated.is_legacy_preview is False
    assert migrated.rect_size == 2400
    assert migrated.target_count == 24
    assert yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["scenarios"][0][
        "config_path"
    ] is None


def test_profile_store_rejects_valid_unowned_legacy_without_catalog_path_match(
    profile_repository,
):
    repo_root, catalog_path, _ = _nondefault_legacy_fixture(profile_repository)
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    points = next(item for item in catalog["datasets"] if item["role"] == "points")
    points["path"] = "other-points"
    points["entrypoint"] = "other-points/Other/Other.shp"
    unmatched = repo_root / points["entrypoint"]
    unmatched.parent.mkdir(parents=True)
    unmatched.write_text("points", encoding="utf-8")
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy.*points.*does not match|no.*points"):
        ProfileStore(repo_root, catalog_path).discover()


def test_profile_store_rejects_valid_unowned_legacy_with_ambiguous_scenario_match(
    profile_repository,
):
    repo_root, catalog_path, _ = _nondefault_legacy_fixture(profile_repository)
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    duplicate = dict(catalog["scenarios"][0])
    duplicate.update(
        scenario_id="chicago-copy",
        display_name="Chicago Copy",
        config_path=None,
    )
    catalog["scenarios"].append(duplicate)
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy.*multiple scenarios|ambiguous"):
        ProfileStore(repo_root, catalog_path).discover()


def _assert_duplicate_profile_identity(error: Exception, *paths: Path) -> None:
    message = str(error)
    assert "duplicate profile identity" in message
    for path in paths:
        assert str(path.resolve()) in message


def test_profile_store_rejects_v2_and_unowned_legacy_with_same_identity(
    profile_repository,
):
    repo_root, catalog_path, legacy = _nondefault_legacy_fixture(profile_repository)
    legacy_target = legacy.with_name("foo.yaml")
    legacy.replace(legacy_target)
    modern = repo_root / "configs" / "modern" / "foo.yaml"
    modern.parent.mkdir()
    dump_profile(
        make_profile(repo_root, profile_id="foo", scenario_id="chicago"),
        modern,
    )

    with pytest.raises(ValueError) as exc_info:
        ProfileStore(repo_root, catalog_path).discover()

    _assert_duplicate_profile_identity(exc_info.value, legacy_target, modern)


def test_profile_store_rejects_two_unowned_legacy_profiles_with_same_identity(
    profile_repository,
):
    repo_root, catalog_path, legacy = _nondefault_legacy_fixture(profile_repository)
    first = repo_root / "configs" / "legacy-a" / "foo.yaml"
    second = repo_root / "configs" / "legacy-b" / "foo.yaml"
    first.parent.mkdir()
    second.parent.mkdir()
    legacy.replace(first)
    second.write_bytes(first.read_bytes())

    with pytest.raises(ValueError) as exc_info:
        ProfileStore(repo_root, catalog_path).discover()

    _assert_duplicate_profile_identity(exc_info.value, first, second)


def test_profile_store_rejects_two_v2_profiles_with_same_identity(
    profile_repository,
):
    repo_root, catalog_path = profile_repository
    first = repo_root / "configs" / "modern-a" / "foo.yaml"
    second = repo_root / "configs" / "modern-b" / "foo.yaml"
    profile = make_profile(repo_root, profile_id="foo", scenario_id="chicago")
    dump_profile(profile, first)
    dump_profile(profile, second)

    with pytest.raises(ValueError) as exc_info:
        ProfileStore(repo_root, catalog_path).discover()

    _assert_duplicate_profile_identity(exc_info.value, first, second)


@pytest.mark.parametrize(
    "content",
    (
        "just-a-scalar\n",
        "- one\n- two\n",
    ),
)
def test_profile_store_discover_skips_unowned_schema_less_non_mapping_yaml(
    profile_repository,
    content,
):
    repo_root, catalog_path = profile_repository
    unowned = repo_root / "configs" / "notes.yaml"
    unowned.parent.mkdir()
    unowned.write_text(content, encoding="utf-8")

    assert ProfileStore(repo_root, catalog_path).discover() == []


def test_profile_store_discover_keeps_unowned_explicit_v2_strict(
    profile_repository,
):
    repo_root, catalog_path = profile_repository
    explicit = repo_root / "configs" / "broken-v2.yaml"
    explicit.parent.mkdir()
    explicit.write_text("schema_version: 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required configuration value"):
        ProfileStore(repo_root, catalog_path).discover()


@pytest.mark.parametrize(
    "content",
    (
        "just-a-scalar\n",
        "- one\n- two\n",
    ),
)
def test_profile_store_discover_keeps_catalog_owned_legacy_strict(
    profile_repository,
    content,
):
    repo_root, catalog_path = profile_repository
    owned = repo_root / "configs" / "owned.yaml"
    owned.parent.mkdir()
    owned.write_text(content, encoding="utf-8")
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    catalog["scenarios"][0]["config_path"] = "configs/owned.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy.*mapping|mapping.*legacy"):
        ProfileStore(repo_root, catalog_path).discover()


def test_profile_store_discover_rejects_catalog_profile_outside_configs(
    profile_repository,
):
    repo_root, catalog_path = profile_repository
    outside = repo_root / "legacy-outside.yaml"
    outside.write_bytes((LEGACY_FIXTURES / "experiment-v1.yaml").read_bytes())
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    catalog["scenarios"][0]["config_path"] = "legacy-outside.yaml"
    catalog_path.write_text(
        yaml.safe_dump(catalog, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="configs|inside"):
        ProfileStore(repo_root, catalog_path).discover()


@pytest.mark.parametrize(
    ("profile_id", "scenario_id"),
    [
        ("../escape", "chicago"),
        ("con", "chicago"),
        ("legacy", "../escape"),
    ],
)
def test_convert_legacy_profile_rejects_unsafe_identity(
    tmp_path,
    profile_id,
    scenario_id,
):
    legacy = profiles_module.load_legacy_profile(
        LEGACY_FIXTURES / "experiment-v1.yaml",
        repo_root=tmp_path,
    )

    with pytest.raises(ValueError, match=r"profile\.(id|scenario_id)"):
        profiles_module.convert_legacy_profile(
            legacy,
            profile_id=profile_id,
            scenario_id=scenario_id,
            points_dataset_id="points",
        )


def test_profile_store_discover_rejects_symlink_escape(profile_repository):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    outside = repo_root.parent / f"{repo_root.name}-outside-profile.yaml"
    dump_profile(make_profile(repo_root), outside)
    link = repo_root / "configs" / "chicago" / "escape.yaml"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(outside)
    except OSError as exc:
        outside.unlink(missing_ok=True)
        pytest.skip(f"symlink creation unavailable: {exc}")
    try:
        with pytest.raises(ValueError, match="configs|symlink|outside"):
            store.discover()
    finally:
        link.unlink(missing_ok=True)
        outside.unlink(missing_ok=True)


def test_profile_store_copy_and_default_rename_preserve_source_until_commit(
    profile_repository,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    source = store.save(make_profile(repo_root), set_default=True)
    source_bytes = source.read_bytes()

    copied = store.copy(source, "chicago-copy", "Chicago copy")

    assert source.read_bytes() == source_bytes
    assert load_profile(copied, repo_root=repo_root).profile_id == "chicago-copy"
    renamed = store.rename(source, "chicago-renamed", "Chicago renamed")
    assert renamed == (repo_root / "configs/chicago/chicago-renamed.yaml").resolve()
    assert not source.exists()
    renamed_profile = load_profile(renamed, repo_root=repo_root)
    assert renamed_profile.profile_id == "chicago-renamed"
    assert renamed_profile.display_name == "Chicago renamed"
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert catalog["scenarios"][0]["config_path"] == (
        "configs/chicago/chicago-renamed.yaml"
    )


def test_profile_store_rename_rolls_back_new_target_when_catalog_save_fails(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    source = store.save(make_profile(repo_root), set_default=True)
    original_source = source.read_bytes()
    destination = repo_root / "configs/chicago/chicago-renamed.yaml"

    def fail_save(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(profiles_module, "save_data_catalog", fail_save)
    with pytest.raises(OSError, match="boom"):
        store.rename(source, "chicago-renamed", "Chicago renamed")

    assert source.read_bytes() == original_source
    assert not destination.exists()


def test_profile_store_default_rename_preserves_concurrently_changed_source(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    source = store.save(make_profile(repo_root), set_default=True)
    destination = repo_root / "configs/chicago/chicago-renamed.yaml"
    sentinel = b"external profile owner\n"
    real_save = profiles_module.save_data_catalog
    save_calls = 0

    def switch_default_then_change_source(catalog, document):
        nonlocal save_calls
        saved = real_save(catalog, document)
        save_calls += 1
        if save_calls == 1:
            source.write_bytes(sentinel)
        return saved

    monkeypatch.setattr(
        profiles_module,
        "save_data_catalog",
        switch_default_then_change_source,
    )

    with pytest.raises(ConcurrentProfileUpdateError, match="changed"):
        store.rename(source, "chicago-renamed", "Chicago renamed")

    assert source.read_bytes() == sentinel
    assert not destination.exists()
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert catalog["scenarios"][0]["config_path"] == (
        "configs/chicago/chicago-default.yaml"
    )


def test_profile_store_keeps_rename_target_referenced_by_concurrent_catalog(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    source = store.save(make_profile(repo_root), set_default=True)
    destination = repo_root / "configs/chicago/chicago-renamed.yaml"
    relative_destination = "configs/chicago/chicago-renamed.yaml"
    real_save = profiles_module.save_data_catalog
    external_state = {}

    def reference_target_then_attempt_stale_save(catalog, document):
        external_state["bytes"] = _write_external_catalog(
            catalog,
            real_save,
            scenario_id="chicago",
            config_path=relative_destination,
            note="external owner references rename target",
        )
        return real_save(catalog, document)

    monkeypatch.setattr(
        profiles_module,
        "save_data_catalog",
        reference_target_then_attempt_stale_save,
    )

    with pytest.raises(ConcurrentCatalogUpdateError, match="changed since"):
        store.rename(source, "chicago-renamed", "Chicago renamed")

    assert catalog_path.read_bytes() == external_state["bytes"]
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert catalog["scenarios"][0]["config_path"] == relative_destination
    assert source.is_file()
    assert destination.is_file()
    assert load_profile(destination, repo_root=repo_root).profile_id == (
        "chicago-renamed"
    )


def test_profile_store_non_default_rename_guards_catalog_before_unlink(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    source = store.save(make_profile(repo_root))
    destination = repo_root / "configs/chicago/chicago-renamed.yaml"
    relative_source = "configs/chicago/chicago-default.yaml"
    real_save = profiles_module.save_data_catalog
    real_guard = getattr(
        profiles_module,
        "_ensure_catalog_unchanged",
        lambda catalog: None,
    )
    changed = False

    def make_source_default_then_guard(catalog):
        nonlocal changed
        if not changed:
            _write_external_catalog(
                catalog,
                real_save,
                scenario_id="chicago",
                config_path=relative_source,
                note="external owner made rename source default",
            )
            changed = True
        return real_guard(catalog)

    monkeypatch.setattr(
        profiles_module,
        "_ensure_catalog_unchanged",
        make_source_default_then_guard,
        raising=False,
    )

    with pytest.raises(ConcurrentCatalogUpdateError, match="changed since"):
        store.rename(source, "chicago-renamed", "Chicago renamed")

    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    configured = repo_root / catalog["scenarios"][0]["config_path"]
    assert configured == source
    assert source.is_file()
    assert not destination.exists()


def test_profile_store_default_rename_guards_saved_catalog_before_unlink(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    source = store.save(make_profile(repo_root), set_default=True)
    destination = repo_root / "configs/chicago/chicago-renamed.yaml"
    relative_source = "configs/chicago/chicago-default.yaml"
    real_save = profiles_module.save_data_catalog
    real_guard = getattr(
        profiles_module,
        "_ensure_catalog_unchanged",
        lambda catalog: None,
    )
    changed = False

    def restore_source_default_then_guard(catalog):
        nonlocal changed
        if not changed:
            _write_external_catalog(
                catalog,
                real_save,
                scenario_id="chicago",
                config_path=relative_source,
                note="external owner restored rename source",
            )
            changed = True
        return real_guard(catalog)

    monkeypatch.setattr(
        profiles_module,
        "_ensure_catalog_unchanged",
        restore_source_default_then_guard,
        raising=False,
    )

    with pytest.raises(ConcurrentCatalogUpdateError, match="changed since"):
        store.rename(source, "chicago-renamed", "Chicago renamed")

    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    configured = repo_root / catalog["scenarios"][0]["config_path"]
    assert configured == source
    assert source.is_file()
    assert destination.is_file()
    assert load_profile(destination, repo_root=repo_root).profile_id == (
        "chicago-renamed"
    )


def test_profile_store_keeps_rename_target_when_catalog_rollback_fails(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    source = store.save(make_profile(repo_root), set_default=True)
    destination = repo_root / "configs/chicago/chicago-renamed.yaml"
    real_save = profiles_module.save_data_catalog
    real_restore = profiles_module._atomic_restore_bytes

    def save_then_fail(catalog, document):
        real_save(catalog, document)
        raise OSError("after write")

    def fail_catalog_restore(path, content):
        if Path(path).resolve() == catalog_path.resolve():
            raise OSError("rollback failed")
        return real_restore(path, content)

    monkeypatch.setattr(profiles_module, "save_data_catalog", save_then_fail)
    monkeypatch.setattr(
        profiles_module,
        "_atomic_restore_bytes",
        fail_catalog_restore,
    )

    with pytest.raises(OSError, match="after write") as error:
        store.rename(source, "chicago-renamed", "Chicago renamed")

    assert any("rollback failed" in note for note in error.value.__notes__)
    assert any("target was retained" in note for note in error.value.__notes__)
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    configured = repo_root / catalog["scenarios"][0]["config_path"]
    assert configured == destination
    assert source.is_file()
    assert destination.is_file()
    assert load_profile(source, repo_root=repo_root).profile_id == "chicago-default"
    assert load_profile(destination, repo_root=repo_root).profile_id == (
        "chicago-renamed"
    )


def test_profile_store_set_default_validates_location_and_scenario(
    profile_repository,
    tmp_path,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    chicago = store.save(make_profile(repo_root))
    new_york = store.save(
        make_profile(
            repo_root,
            profile_id="nyc-default",
            display_name="NYC default",
            scenario_id="new-york-city",
        )
    )

    assert store.set_default("chicago", chicago) == chicago
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert catalog["scenarios"][0]["config_path"] == (
        "configs/chicago/chicago-default.yaml"
    )
    outside = tmp_path / "outside.yaml"
    dump_profile(make_profile(repo_root), outside)
    with pytest.raises(ValueError, match="configs|outside"):
        store.set_default("chicago", outside)
    with pytest.raises(ValueError, match="configs|outside"):
        store.set_default("chicago", "../outside.yaml")
    with pytest.raises(ValueError, match="scenario"):
        store.set_default("chicago", new_york)


def test_profile_store_delete_switches_default_before_removing_source(
    profile_repository,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    source = store.save(make_profile(repo_root), set_default=True)
    replacement = store.save(
        make_profile(
            repo_root,
            profile_id="chicago-replacement",
            display_name="Replacement",
        )
    )
    disposable = store.save(
        make_profile(repo_root, profile_id="disposable", display_name="Disposable")
    )

    assert store.delete(disposable) is None
    assert not disposable.exists()
    assert store.delete(source, replacement_default=replacement) is None
    assert not source.exists()
    assert replacement.is_file()
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert catalog["scenarios"][0]["config_path"] == (
        "configs/chicago/chicago-replacement.yaml"
    )


def test_profile_store_default_delete_preserves_concurrently_changed_source(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    source = store.save(make_profile(repo_root), set_default=True)
    replacement = store.save(
        make_profile(
            repo_root,
            profile_id="chicago-replacement",
            display_name="Replacement",
        )
    )
    sentinel = b"external profile owner\n"
    real_save = profiles_module.save_data_catalog
    save_calls = 0

    def switch_default_then_change_source(catalog, document):
        nonlocal save_calls
        saved = real_save(catalog, document)
        save_calls += 1
        if save_calls == 1:
            source.write_bytes(sentinel)
        return saved

    monkeypatch.setattr(
        profiles_module,
        "save_data_catalog",
        switch_default_then_change_source,
    )

    with pytest.raises(ConcurrentProfileUpdateError, match="changed"):
        store.delete(source, replacement_default=replacement)

    assert source.read_bytes() == sentinel
    assert replacement.is_file()
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assert catalog["scenarios"][0]["config_path"] == (
        "configs/chicago/chicago-default.yaml"
    )


def test_profile_store_non_default_delete_preserves_change_during_load(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    source = store.save(
        make_profile(repo_root, profile_id="disposable", display_name="Disposable")
    )
    sentinel = b"external profile owner\n"
    real_load = profiles_module.load_profile

    def load_then_change_source(path, **kwargs):
        loaded = real_load(path, **kwargs)
        if Path(path).resolve() == source:
            source.write_bytes(sentinel)
        return loaded

    monkeypatch.setattr(profiles_module, "load_profile", load_then_change_source)

    with pytest.raises(ConcurrentProfileUpdateError, match="changed"):
        store.delete(source)

    assert source.read_bytes() == sentinel


def test_profile_store_non_default_delete_guards_catalog_before_unlink(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    source = store.save(
        make_profile(repo_root, profile_id="disposable", display_name="Disposable")
    )
    relative_source = "configs/chicago/disposable.yaml"
    real_save = profiles_module.save_data_catalog
    real_guard = getattr(
        profiles_module,
        "_ensure_catalog_unchanged",
        lambda catalog: None,
    )
    changed = False

    def make_source_default_then_guard(catalog):
        nonlocal changed
        if not changed:
            _write_external_catalog(
                catalog,
                real_save,
                scenario_id="chicago",
                config_path=relative_source,
                note="external owner made delete source default",
            )
            changed = True
        return real_guard(catalog)

    monkeypatch.setattr(
        profiles_module,
        "_ensure_catalog_unchanged",
        make_source_default_then_guard,
        raising=False,
    )

    with pytest.raises(ConcurrentCatalogUpdateError, match="changed since"):
        store.delete(source)

    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    configured = repo_root / catalog["scenarios"][0]["config_path"]
    assert configured == source
    assert source.is_file()


def test_profile_store_default_delete_guards_saved_catalog_before_unlink(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    source = store.save(make_profile(repo_root), set_default=True)
    replacement = store.save(
        make_profile(
            repo_root,
            profile_id="chicago-replacement",
            display_name="Replacement",
        )
    )
    relative_source = "configs/chicago/chicago-default.yaml"
    real_save = profiles_module.save_data_catalog
    real_guard = getattr(
        profiles_module,
        "_ensure_catalog_unchanged",
        lambda catalog: None,
    )
    changed = False

    def restore_source_default_then_guard(catalog):
        nonlocal changed
        if not changed:
            _write_external_catalog(
                catalog,
                real_save,
                scenario_id="chicago",
                config_path=relative_source,
                note="external owner restored delete source",
            )
            changed = True
        return real_guard(catalog)

    monkeypatch.setattr(
        profiles_module,
        "_ensure_catalog_unchanged",
        restore_source_default_then_guard,
        raising=False,
    )

    with pytest.raises(ConcurrentCatalogUpdateError, match="changed since"):
        store.delete(source, replacement_default=replacement)

    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    configured = repo_root / catalog["scenarios"][0]["config_path"]
    assert configured == source
    assert source.is_file()
    assert replacement.is_file()


def test_profile_store_rejects_invalid_default_replacement(profile_repository):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    source = store.save(make_profile(repo_root), set_default=True)
    wrong_scenario = store.save(
        make_profile(
            repo_root,
            profile_id="nyc-default",
            display_name="NYC default",
            scenario_id="new-york-city",
        )
    )

    with pytest.raises(ValueError, match="scenario"):
        store.delete(source, replacement_default=wrong_scenario)
    with pytest.raises((FileNotFoundError, ValueError)):
        store.delete(
            source,
            replacement_default=repo_root / "configs/chicago/missing.yaml",
        )
    assert source.is_file()


def test_profile_store_rejects_external_and_symlink_escape_without_deleting(
    profile_repository,
    tmp_path,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    outside = tmp_path / "outside.yaml"
    dump_profile(make_profile(repo_root), outside)

    for unsafe in (outside, "../outside.yaml"):
        with pytest.raises(ValueError, match="configs|outside"):
            store.delete(unsafe)
        assert outside.is_file()

    link = repo_root / "configs" / "chicago" / "escape.yaml"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="configs|outside"):
        store.delete(link)
    assert link.is_symlink()
    assert outside.is_file()


def test_profile_store_overwrite_restores_existing_profile_on_catalog_failure(
    profile_repository,
    monkeypatch,
):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)
    original = store.save(make_profile(repo_root))
    original_bytes = original.read_bytes()
    replacement_profile = make_profile(repo_root, display_name="Changed")

    def fail_save(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(profiles_module, "save_data_catalog", fail_save)
    with pytest.raises(OSError, match="boom"):
        store.save(replacement_profile, overwrite=True, set_default=True)

    assert original.read_bytes() == original_bytes
    assert load_profile(original, repo_root=repo_root).display_name == "Chicago default"


def test_profile_store_validates_points_dataset_role(profile_repository):
    repo_root, catalog_path = profile_repository
    store = ProfileStore(repo_root, catalog_path)

    with pytest.raises(ValueError, match="points|role"):
        store.save(make_profile(repo_root, points_dataset_id="boundary"))
    with pytest.raises(ValueError, match="points|Unknown"):
        store.save(make_profile(repo_root, points_dataset_id="missing"))
