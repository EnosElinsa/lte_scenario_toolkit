import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command_help(script: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_select_sites_exposes_reproducible_configuration_arguments():
    help_text = command_help("select_sites.py")

    assert "--config" in help_text
    assert "--city" in help_text
    assert "--output-dir" in help_text
    assert "--select-index" in help_text


def test_figure_generator_exposes_reproducible_configuration_arguments():
    help_text = command_help("generate_scenario_figures.py")

    assert "--config" in help_text
    assert "--city" in help_text
    assert "--output-dir" in help_text


def test_data_manifest_generator_exposes_metadata_and_output_arguments():
    help_text = command_help("scripts/create_data_manifest.py")

    assert "--metadata" in help_text
    assert "--output" in help_text


def test_select_sites_thin_script_forwards_help():
    assert "--config" in command_help("scripts/select_sites.py")


def test_generate_figures_thin_script_forwards_help():
    assert "--config" in command_help("scripts/generate_scenario_figures.py")


def test_newyork_dem_thin_script_forwards_help():
    dem_help = command_help("scripts/download_newyork_1m_dem.py")
    assert "--project" in dem_help
    assert "--dry-run" in dem_help
