import re
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import lte_scenario_toolkit.profiles as profiles_module
from lte_scenario_toolkit.profiles import (
    DEFAULT_PROFILE_VALUES,
    ExperimentProfile,
    FigureSettings,
    OutputSettings,
    ProfileStore,
    dump_profile,
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
        "points": tmp_path / "inputs" / "points" / "stations.csv",
        "boundary": tmp_path / "inputs" / "boundary" / "boundary.geojson",
        "dem": tmp_path / "inputs" / "dem" / "elevation.tif",
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
                "inputs/points",
                "inputs/points/stations.csv",
            ),
            _catalog_dataset(
                "boundary",
                "boundary",
                "inputs/boundary",
                "inputs/boundary/boundary.geojson",
            ),
            _catalog_dataset(
                "dem",
                "dem",
                "inputs/dem",
                "inputs/dem/elevation.tif",
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
