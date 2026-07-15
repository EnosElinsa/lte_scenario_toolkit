import subprocess
import sys
from pathlib import Path

import yaml

import scripts.create_data_manifest as manifest_script

ROOT = Path(__file__).resolve().parents[1]


def command_help(script: str, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / script), *args, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_select_sites_exposes_reproducible_configuration_arguments():
    help_text = command_help("scripts/select_sites.py")

    assert "--config" in help_text
    assert "--city" in help_text
    assert "--output-dir" in help_text
    assert "--select-index" in help_text
    assert not (ROOT / "select_sites.py").exists()


def test_figure_generator_exposes_reproducible_configuration_arguments():
    help_text = command_help("scripts/generate_scenario_figures.py")

    assert "--config" in help_text
    assert "--city" in help_text
    assert "--output-dir" in help_text
    assert not (ROOT / "generate_scenario_figures.py").exists()


def test_data_manifest_generator_exposes_metadata_and_output_arguments():
    help_text = command_help("scripts/create_data_manifest.py")

    assert "--metadata" in help_text
    assert "--output" in help_text
    assert "--dataset-id" in help_text


def test_data_manifest_generator_passes_repeated_dataset_ids_as_a_set(tmp_path, monkeypatch):
    calls = []

    def fake_create_data_manifest(metadata, output, *, repo_root, dataset_ids):
        calls.append((metadata, output, repo_root, dataset_ids))
        return output

    monkeypatch.setattr(manifest_script, "create_data_manifest", fake_create_data_manifest)

    result = manifest_script.main(
        [
            "--metadata",
            str(tmp_path / "datasets.yaml"),
            "--output",
            str(tmp_path / "manifest.json"),
            "--dataset-id",
            "boundary",
            "--dataset-id",
            "dem",
        ]
    )

    assert result == 0
    assert calls[0][3] == {"boundary", "dem"}


def test_select_sites_thin_script_forwards_help():
    assert "--config" in command_help("scripts/select_sites.py")


def test_generate_figures_thin_script_forwards_help():
    assert "--config" in command_help("scripts/generate_scenario_figures.py")


def test_newyork_dem_thin_script_forwards_help():
    dem_help = command_help("scripts/download_newyork_1m_dem.py")
    assert "--project" in dem_help
    assert "--dry-run" in dem_help


def _write_cli_catalog(tmp_path: Path) -> Path:
    for scenario_id in ("alpha", "zulu"):
        boundary = tmp_path / "boundary_shp" / scenario_id / f"{scenario_id}.shp"
        boundary.parent.mkdir(parents=True)
        boundary.write_bytes(b"boundary")

    document = {
        "schema_version": 2,
        "datasets": [
            {
                "dataset_id": "boundary_zulu",
                "role": "boundary",
                "path": "boundary_shp/zulu",
                "entrypoint": "boundary_shp/zulu/zulu.shp",
                "source_url": None,
                "provider": "Test",
                "license": "CC0-1.0",
                "download_date": None,
                "crs": "EPSG:3857",
                "spatial_resolution": "polygon vector",
                "notes": "fixture",
                "geometry_type": "Polygon",
                "feature_count": 1,
                "redistribution_confirmed": True,
            },
            {
                "dataset_id": "boundary_alpha",
                "role": "boundary",
                "path": "boundary_shp/alpha",
                "entrypoint": "boundary_shp/alpha/alpha.shp",
                "source_url": None,
                "provider": "Test",
                "license": "CC0-1.0",
                "download_date": None,
                "crs": "EPSG:3857",
                "spatial_resolution": "polygon vector",
                "notes": "fixture",
                "geometry_type": "Polygon",
                "feature_count": 1,
                "redistribution_confirmed": True,
            },
        ],
        "scenarios": [
            {
                "scenario_id": "zulu",
                "display_name": "Zulu City",
                "boundary_dataset_id": "boundary_zulu",
                "dem_dataset_id": None,
                "config_path": None,
            },
            {
                "scenario_id": "alpha",
                "display_name": "Alpha City",
                "boundary_dataset_id": "boundary_alpha",
                "dem_dataset_id": None,
                "config_path": None,
            },
        ],
    }
    path = tmp_path / "data" / "datasets.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_data_manager_help_exposes_extensible_scenario_commands():
    root_help = command_help("scripts/manage_data.py")
    scenario_help = command_help("scripts/manage_data.py", "scenario")
    add_help = command_help("scripts/manage_data.py", "scenario", "add")
    dem_help = command_help("scripts/manage_data.py", "dem")
    dem_export_help = command_help("scripts/manage_data.py", "dem", "export")

    assert "--catalog" in root_help
    assert "scenario" in root_help
    assert "dem" in root_help
    assert all(command in scenario_help for command in ("add", "list", "show"))
    assert "export" in dem_help
    assert all(
        option in add_help
        for option in (
            "--display-name",
            "--boundary-source",
            "--provider",
            "--license",
            "--layer",
            "--download-date",
            "--config-path",
            "--redistribution-confirmed",
        )
    )
    assert all(
        option in dem_export_help
        for option in (
            "--project",
            "--scale",
            "--file-dimensions",
            "--shard-size",
            "--max-pixels",
            "--drive-folder",
            "--dry-run",
            "--export",
        )
    )


def test_data_cli_preserves_remote_boundary_source_urls():
    from lte_scenario_toolkit.data_cli import build_parser

    args = build_parser().parse_args(
        [
            "scenario",
            "add",
            "sample-city",
            "--boundary-source",
            "https://example.test/boundary.zip",
            "--provider",
            "Example GIS Office",
            "--license",
            "CC0-1.0",
        ]
    )

    assert args.boundary_source == "https://example.test/boundary.zip"


def test_data_cli_scenario_list_is_stable_and_show_reports_paths(tmp_path, capsys):
    from lte_scenario_toolkit.data_cli import main

    catalog_path = _write_cli_catalog(tmp_path)

    assert main(["--catalog", str(catalog_path), "scenario", "list"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "alpha\tboundary-ready\tAlpha City",
        "zulu\tboundary-ready\tZulu City",
    ]

    assert main(["--catalog", str(catalog_path), "scenario", "show", "alpha"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "scenario_id: alpha",
        "display_name: Alpha City",
        "status: boundary-ready",
        "boundary: boundary_shp/alpha/alpha.shp",
        "dem: <not declared>",
    ]
