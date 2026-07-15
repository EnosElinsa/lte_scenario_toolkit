import hashlib
import json

import geopandas as gpd
from shapely.geometry import Point

from src.io import (
    build_dataset_record,
    build_output_dataframe,
    create_data_manifest,
    write_run_record,
)


def test_build_output_dataframe_contains_reproducible_scenario_fields():
    selected = gpd.GeoDataFrame(
        {"cell": [10], "elevation": [12.5]},
        geometry=[Point(100, 200)],
        crs="EPSG:3857",
    )

    frame = build_output_dataframe(
        selected,
        selected.crs,
        rect_id=1,
        pt_count=1,
        left_x=0,
        bottom_y=0,
        center_x=500,
        center_y=500,
        rect_size=1000,
    )

    required = {
        "X",
        "Y",
        "rect_id",
        "pt_count",
        "left_x",
        "bottom_y",
        "center_x",
        "center_y",
        "elevation",
    }
    assert required.issubset(frame.columns)
    assert frame.loc[0, "X"] == 100
    assert frame.loc[0, "elevation"] == 12.5


def test_dataset_record_includes_sha256_and_spatial_metadata(tmp_path):
    source = tmp_path / "data.bin"
    source.write_bytes(b"abc")

    record = build_dataset_record(
        source,
        name="fixture",
        source_url="https://example.test/data",
        license_name="public-domain",
        crs="EPSG:3857",
        resolution_m=1,
    )

    assert record["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert record["size_bytes"] == 3
    assert record["crs"] == "EPSG:3857"
    assert record["resolution_m"] == 1


def test_write_run_record_captures_config_inputs_outputs_and_versions(tmp_path):
    output = write_run_record(
        tmp_path,
        config={"experiment_name": "fixture", "repo_root": tmp_path},
        inputs=[{"name": "stations", "sha256": "123"}],
        outputs=[tmp_path / "result.csv"],
        command=["python", "select_sites.py"],
        timestamp="2026-07-15T12:00:00Z",
        filename="run-select-sites.json",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["timestamp"] == "2026-07-15T12:00:00Z"
    assert payload["config"]["repo_root"] == str(tmp_path)
    assert payload["inputs"][0]["name"] == "stations"
    assert payload["outputs"] == [str(tmp_path / "result.csv")]
    assert "python" in payload["software"]
    assert output.name == "run-select-sites.json"


def test_create_data_manifest_combines_provenance_with_file_checksums(tmp_path):
    dataset_dir = tmp_path / "inputs"
    dataset_dir.mkdir()
    (dataset_dir / "a.txt").write_text("alpha", encoding="utf-8")
    metadata = tmp_path / "datasets.yaml"
    metadata.write_text(
        """
schema_version: 1
datasets:
  - dataset_id: fixture
    path: inputs
    source_url: https://example.test/fixture
    provider: Example
    license: CC0-1.0
    download_date: 2026-07-15
    crs: EPSG:3857
    spatial_resolution: vector
    notes: test data
""".strip(),
        encoding="utf-8",
    )

    output = create_data_manifest(metadata, tmp_path / "manifest.json", repo_root=tmp_path)

    payload = json.loads(output.read_text(encoding="utf-8"))
    dataset = payload["datasets"][0]
    assert dataset["dataset_id"] == "fixture"
    assert dataset["source_url"] == "https://example.test/fixture"
    assert dataset["files"][0]["path"] == "inputs/a.txt"
    assert dataset["files"][0]["size_bytes"] == 5
    assert len(dataset["files"][0]["sha256"]) == 64
