import json
import py_compile
import re
import subprocess
from pathlib import Path

import geopandas as gpd
import tomllib
import yaml

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_READMES = (
    "README.md",
    "boundary_shp/README.md",
    "configs/README.md",
    "data/README.md",
    "dem/README.md",
    "runs/README.md",
    "scripts/README.md",
    "src/README.md",
    "tests/README.md",
)
PUBLIC_SURFACES = (
    ROOT / "README.md",
    ROOT / "pyproject.toml",
    ROOT / "boundary_shp",
    ROOT / "configs",
    ROOT / "data",
    ROOT / "dem",
    ROOT / "gee",
    # Generated runs/** are ignored traceability artifacts and may record the
    # user's selected Earth Engine project. Only the tracked public contract is scanned.
    ROOT / "runs" / "README.md",
    ROOT / "scripts",
    ROOT / "src",
)
TEXT_SUFFIXES = {"", ".md", ".py", ".js", ".yaml", ".yml", ".json", ".toml", ".txt"}
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
SCENARIOS = {
    "phoenix": ("Phoenix", "boundary_phoenix", "usgs_3dep_1m_dem_phoenix", None),
    "chicago": (
        "Chicago",
        "boundary_chicago",
        "usgs_3dep_1m_dem_chicago",
        "configs/example.yaml",
    ),
    "chicago-cbd": (
        "Chicago CBD",
        "boundary_chicago_cbd",
        "usgs_3dep_1m_dem_chicago_cbd",
        None,
    ),
    "cambridge": (
        "Cambridge",
        "boundary_cambridge",
        "usgs_3dep_1m_dem_cambridge",
        None,
    ),
    "new-york-city": (
        "New York City",
        "boundary_new_york_city",
        "usgs_3dep_1m_dem_new_york_city",
        "configs/newyork.yaml",
    ),
}
BOUNDARIES = {
    "boundary_phoenix": (
        "boundary_shp/Arizona_Maricopa_Phoenix",
        "boundary_shp/Arizona_Maricopa_Phoenix/Arizona_Maricopa_Phoenix.shp",
        "EPSG:3857",
        "MultiPolygon",
    ),
    "boundary_chicago": (
        "boundary_shp/Chicago",
        "boundary_shp/Chicago/Chicago_Boundary.shp",
        "EPSG:3857",
        "MultiPolygon",
    ),
    "boundary_chicago_cbd": (
        "boundary_shp/Chicago_CBD",
        "boundary_shp/Chicago_CBD/Chicago_CBD.shp",
        "EPSG:4326",
        "Polygon",
    ),
    "boundary_cambridge": (
        "boundary_shp/Massachusetts_Middlesex_Cambridge",
        "boundary_shp/Massachusetts_Middlesex_Cambridge/Massachusetts_Middlesex_Cambridge.shp",
        "EPSG:3857",
        "Polygon",
    ),
    "boundary_new_york_city": (
        "boundary_shp/NewYorkState_NewYork",
        "boundary_shp/NewYorkState_NewYork/NewYorkState_NewYork.shp",
        "EPSG:3857",
        "MultiPolygon",
    ),
}
DEMS = {
    "usgs_3dep_1m_dem_phoenix": (
        "dem/phoenix",
        "dem/phoenix/usgs_3dep_1m_phoenix.tif",
        "usgs_3dep_1m_phoenix",
        False,
    ),
    "usgs_3dep_1m_dem_chicago": (
        "dem/USGS_1M_DEM_Chicago",
        "dem/USGS_1M_DEM_Chicago/USGS_1M_DEM_Chicago.tif",
        "USGS_1M_DEM_Chicago",
        True,
    ),
    "usgs_3dep_1m_dem_chicago_cbd": (
        "dem/chicago-cbd",
        "dem/chicago-cbd/usgs_3dep_1m_chicago-cbd.tif",
        "usgs_3dep_1m_chicago-cbd",
        False,
    ),
    "usgs_3dep_1m_dem_cambridge": (
        "dem/cambridge",
        "dem/cambridge/usgs_3dep_1m_cambridge.tif",
        "usgs_3dep_1m_cambridge",
        False,
    ),
    "usgs_3dep_1m_dem_new_york_city": (
        "dem/USGS_1M_DEM_NewYorkState_NewYork",
        "dem/USGS_1M_DEM_NewYorkState_NewYork/USGS_1M_DEM_NewYorkState_NewYork.tif",
        "USGS_1M_DEM_NewYorkState_NewYork",
        True,
    ),
}


def _is_public_text_file(path: Path) -> bool:
    """Exclude ignored build metadata from tracked public-surface checks."""

    return path.suffix.casefold() in TEXT_SUFFIXES and not any(
        part.casefold().endswith(".egg-info") for part in path.parts
    )


def test_repository_metadata_files_exist():
    for relative_path in (
        "README.md",
        "LICENSE",
        "pyproject.toml",
        ".gitignore",
        ".gitattributes",
        "data/datasets.yaml",
        "data/manifest.json",
        *PUBLIC_READMES,
    ):
        assert (ROOT / relative_path).is_file(), relative_path


def test_generated_package_metadata_is_not_a_public_source_surface():
    generated = ROOT / "src" / "lte_scenario_toolkit.egg-info" / "PKG-INFO"

    assert not _is_public_text_file(generated)
    assert _is_public_text_file(ROOT / "src" / "lte_scenario_toolkit" / "data_cli.py")


def test_public_surfaces_have_no_removed_city_exporter_or_project_id():
    removed_terms = (
        "lte-" + "download-newyork-dem",
        "download_newyork_" + "1m_dem.py",
        "newyork_" + "dem.py",
        "gee/" + "newyork_1m_dem.js",
    )
    project_id_pattern = re.compile(r"gen-lang-client-\d+")
    paths = []
    for surface in PUBLIC_SURFACES:
        if surface.is_file():
            if _is_public_text_file(surface):
                paths.append(surface)
        elif surface.is_dir():
            paths.extend(
                path
                for path in surface.rglob("*")
                if path.is_file() and _is_public_text_file(path)
            )

    offenders: list[str] = []
    for path in sorted(set(paths)):
        text = path.read_text(encoding="utf-8")
        if any(term in text for term in removed_terms) or project_id_pattern.search(text):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, f"stale city-specific or project-secret references: {offenders}"


def test_example_configuration_declares_epsg3857():
    config = yaml.safe_load((ROOT / "configs" / "example.yaml").read_text(encoding="utf-8"))
    assert config["spatial"]["target_crs"] == "EPSG:3857"
    assert config["spatial"]["rectangle_size_m"] == 3000
    assert config["scan"]["strategy"] == "uniform"


def test_large_and_generated_paths_are_protected():
    ignore_rules = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "dem/**" in ignore_rules
    assert "results/**" in ignore_rules
    assert "runs/**" in ignore_rules
    assert "docs/superpowers/" in ignore_rules
    assert ".superpowers/" in ignore_rules

    tracked = subprocess.run(
        ["git", "ls-files", "--", ".superpowers", "docs/superpowers"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tracked.stdout == ""

    for ignored_path in (
        ".superpowers/session.json",
        "docs/superpowers/plan.md",
    ):
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", ignored_path],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert ignored.returncode == 0

    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "points_shp/**/*.dbf filter=lfs" in attributes
    assert "points_shp/**/*.shp filter=lfs" in attributes


def test_packaged_python_modules_and_scripts_compile():
    python_files = sorted((ROOT / "src" / "lte_scenario_toolkit").glob("*.py"))
    python_files.extend(sorted((ROOT / "scripts").glob("*.py")))

    assert python_files
    for path in python_files:
        py_compile.compile(str(path), doraise=True)


def test_packaging_declares_research_cli_commands():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = metadata["project"]["scripts"]

    assert metadata["project"]["name"] == "lte_scenario_toolkit"
    assert scripts["lte-select-sites"] == "lte_scenario_toolkit.select_sites:main"
    assert scripts["lte-generate-figures"] == "lte_scenario_toolkit.generate_figures:main"
    assert scripts["lte-data"] == "lte_scenario_toolkit.data_cli:main"
    assert "lte-download-newyork-dem" not in scripts


def test_generic_data_cli_replaces_city_specific_exporters():
    assert (ROOT / "scripts" / "manage_data.py").is_file()
    assert (ROOT / "src" / "lte_scenario_toolkit" / "data_cli.py").is_file()
    assert (ROOT / "src" / "lte_scenario_toolkit" / "dem_data.py").is_file()
    assert not (ROOT / "scripts" / "download_newyork_1m_dem.py").exists()
    assert not (ROOT / "src" / "lte_scenario_toolkit" / "newyork_dem.py").exists()
    assert not (ROOT / "gee" / "newyork_1m_dem.js").exists()


def test_public_readmes_and_source_files_are_english_only():
    checked_paths = [ROOT / relative_path for relative_path in PUBLIC_READMES]
    for source_root in (ROOT / "src", ROOT / "scripts", ROOT / "gee", ROOT / "tests"):
        checked_paths.extend(source_root.rglob("*.py"))
        checked_paths.extend(source_root.rglob("*.js"))

    offenders = []
    for path in sorted(set(checked_paths)):
        text = path.read_text(encoding="utf-8")
        if CJK_PATTERN.search(text):
            offenders.append(path.relative_to(ROOT).as_posix())

    assert not offenders, f"CJK characters found in public English files: {offenders}"


def test_repository_catalog_registers_every_boundary_and_scenario():
    metadata = yaml.safe_load((ROOT / "data" / "datasets.yaml").read_text(encoding="utf-8"))
    datasets = {item["dataset_id"]: item for item in metadata["datasets"]}
    scenarios = {item["scenario_id"]: item for item in metadata["scenarios"]}

    assert set(metadata) == {"datasets", "scenarios"}
    assert set(scenarios) == set(SCENARIOS)
    assert "administrative_boundaries" not in datasets
    assert datasets["usa_clear_lte_base_stations"]["role"] == "points"
    assert datasets["usa_clear_lte_base_stations"]["entrypoint"].endswith(
        "/USA_Clear_LTE_Base_Station.shp"
    )

    for boundary_id, (path, entrypoint, crs, geometry_type) in BOUNDARIES.items():
        boundary = datasets[boundary_id]
        assert boundary["role"] == "boundary"
        assert boundary["path"] == path
        assert boundary["entrypoint"] == entrypoint
        assert boundary["crs"] == crs
        assert boundary["geometry_type"] == geometry_type
        assert boundary["provider"] == "Repository owner supplied dataset"
        assert boundary["license"] == "Public redistribution confirmed by repository owner"
        assert boundary["source_url"] is None
        assert boundary["download_date"] is None
        assert boundary["feature_count"] == 1
        assert boundary["redistribution_confirmed"] is True

    for dem_id, (path, entrypoint, export_prefix, _available) in DEMS.items():
        dem = datasets[dem_id]
        assert dem["role"] == "dem"
        assert dem["path"] == path
        assert dem["entrypoint"] == entrypoint
        assert dem["source_url"].endswith("/USGS_3DEP_1m")
        assert dem["provider"] == "United States Geological Survey"
        assert "public-domain" in dem["license"]
        assert dem["earth_engine_collection"] == "USGS/3DEP/1m"
        assert dem["band"] == "elevation"
        assert dem["crs"] == "EPSG:3857"
        assert dem["spatial_resolution"] == "1 m"
        assert dem["units"] == "metres"
        assert dem["vertical_datum"] == "NAVD88"
        assert dem["native_scale_m"] == 1
        assert dem["export_crs"] == "EPSG:3857"
        assert dem["export_prefix"] == export_prefix
        assert dem["drive_folder"] == "lte-scenario-toolkit-dem"
        assert dem["external"] is True

    for scenario_id, (display_name, boundary_id, dem_id, config_path) in SCENARIOS.items():
        scenario = scenarios[scenario_id]
        assert scenario["display_name"] == display_name
        assert scenario["boundary_dataset_id"] == boundary_id
        assert scenario["dem_dataset_id"] == dem_id
        assert scenario["config_path"] == config_path


def test_registered_boundary_entrypoints_match_declared_geometry():
    metadata = yaml.safe_load((ROOT / "data" / "datasets.yaml").read_text(encoding="utf-8"))
    boundaries = {
        item["dataset_id"]: item
        for item in metadata["datasets"]
        if item["role"] == "boundary"
    }

    assert set(boundaries) == set(BOUNDARIES)
    for boundary_id, boundary in boundaries.items():
        frame = gpd.read_file(ROOT / boundary["entrypoint"])

        assert not frame.empty, boundary_id
        assert len(frame) == boundary["feature_count"], boundary_id
        assert frame.crs is not None, boundary_id
        assert frame.crs.to_string() == boundary["crs"], boundary_id
        assert frame.geometry.notna().all(), boundary_id
        assert not frame.geometry.is_empty.any(), boundary_id
        assert set(frame.geometry.geom_type) == {boundary["geometry_type"]}, boundary_id
        assert frame.geometry.is_valid.all(), boundary_id


def test_manifest_preserves_current_scenario_links():
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    datasets = {item["dataset_id"]: item for item in manifest["datasets"]}
    scenarios = {item["scenario_id"]: item for item in manifest["scenarios"]}

    assert set(manifest) == {"generated_at", "datasets", "scenarios"}
    assert set(scenarios) == set(SCENARIOS)
    assert "administrative_boundaries" not in datasets
    assert set(BOUNDARIES) <= set(datasets)
    assert set(DEMS) <= set(datasets)
    for boundary_id in BOUNDARIES:
        assert datasets[boundary_id]["files"]
    for dem_id, (_path, _entrypoint, _prefix, available) in DEMS.items():
        assert bool(datasets[dem_id]["files"]) is available
