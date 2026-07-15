import py_compile
from pathlib import Path

import tomllib
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_repository_metadata_files_exist():
    for relative_path in ("README.md", "LICENSE", "pyproject.toml", ".gitignore", ".gitattributes"):
        assert (ROOT / relative_path).is_file(), relative_path


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

    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "points_shp/**/*.dbf filter=lfs" in attributes
    assert "points_shp/**/*.shp filter=lfs" in attributes


def test_existing_python_entry_points_compile():
    for relative_path in (
        "select_sites.py",
        "generate_scenario_figures.py",
        "download_newyork_1m_dem.py",
    ):
        py_compile.compile(str(ROOT / relative_path), doraise=True)


def test_packaging_declares_research_cli_commands():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = metadata["project"]["scripts"]

    assert metadata["project"]["name"] == "lte_scenario_toolkit"
    assert scripts["lte-select-sites"] == "lte_scenario_toolkit.select_sites:main"
    assert scripts["lte-generate-figures"] == "lte_scenario_toolkit.generate_figures:main"
    assert scripts["lte-download-newyork-dem"] == "lte_scenario_toolkit.newyork_dem:main"
