import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

import lte_scenario_toolkit.io as io_module
from lte_scenario_toolkit.io import (
    atomic_write_json,
    build_dataset_record,
    build_output_dataframe,
    create_data_manifest,
    software_versions,
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


def test_build_output_dataframe_adds_optional_traceability_without_legacy_regression():
    selected = gpd.GeoDataFrame(
        {"cell": [10], "elevation": [12.5]},
        geometry=[Point(100, 200)],
        crs="EPSG:3857",
    )
    arguments = {
        "rect_id": 1,
        "pt_count": 1,
        "left_x": 0,
        "bottom_y": 0,
        "center_x": 500,
        "center_y": 500,
        "rect_size": 1000,
    }

    legacy = build_output_dataframe(selected, selected.crs, **arguments)
    traced = build_output_dataframe(
        selected,
        selected.crs,
        **arguments,
        run_id="a" * 32,
        scenario_id="chicago",
        profile_id="chicago-default",
        candidate_id="candidate-0001",
    )

    assert traced.columns.tolist() == legacy.columns.tolist() + [
        "run_id",
        "scenario_id",
        "profile_id",
        "candidate_id",
    ]
    assert traced.loc[0, [
        "run_id",
        "scenario_id",
        "profile_id",
        "candidate_id",
    ]].tolist() == [
        "a" * 32,
        "chicago",
        "chicago-default",
        "candidate-0001",
    ]


def test_build_output_dataframe_uses_authoritative_target_crs_for_xy():
    selected = gpd.GeoDataFrame(
        {"cell": [10]},
        geometry=[Point(500_000, 4_500_000)],
        crs="EPSG:32618",
    )

    frame = build_output_dataframe(
        selected,
        selected.crs,
        rect_id=1,
        pt_count=1,
        left_x=499_000,
        bottom_y=4_499_000,
        center_x=500_000,
        center_y=4_500_000,
        rect_size=2000,
        target_crs="EPSG:32618",
    )

    assert frame.loc[0, "X"] == 500_000
    assert frame.loc[0, "Y"] == 4_500_000
    assert frame.loc[0, "center_x"] == 500_000


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


def test_git_commit_is_none_when_git_executable_is_unavailable(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        io_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("git")),
    )

    assert io_module._git_commit(tmp_path) is None


def test_write_run_record_captures_config_inputs_outputs_and_versions(tmp_path):
    output = write_run_record(
        tmp_path,
        config={"experiment_name": "fixture", "repo_root": tmp_path},
        inputs=[{"name": "stations", "sha256": "123"}],
        outputs=[tmp_path / "result.csv"],
        command=["lte-select-sites"],
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


def test_atomic_write_json_replaces_from_sibling_temp_and_serializes_json_safe_values(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "record.json"
    destination.write_text("old content\n", encoding="utf-8")
    replacements = []
    original_replace = Path.replace

    def tracking_replace(path, target):
        replacements.append((path, Path(target)))
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", tracking_replace)

    returned = atomic_write_json(
        destination,
        {
            "label": "Café",
            "path": tmp_path / "résumé.csv",
            "date": date(2026, 7, 16),
            "timestamp": datetime(2026, 7, 16, 10, tzinfo=timezone.utc),
            "count": np.int64(3),
        },
    )

    assert returned == destination
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload == {
        "label": "Café",
        "path": str(tmp_path / "résumé.csv"),
        "date": "2026-07-16",
        "timestamp": "2026-07-16T10:00:00+00:00",
        "count": 3,
    }
    assert destination.read_bytes().endswith(b"\n")
    temporary, target = replacements[-1]
    assert target == destination
    assert temporary.parent == destination.parent
    assert temporary.name.startswith(f".{destination.name}.")
    assert not temporary.exists()


def test_write_run_record_preserves_payload_contract_and_uses_atomic_helper(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_atomic_write_json(path, payload):
        captured["path"] = Path(path)
        captured["payload"] = payload
        return Path(path)

    monkeypatch.setattr(io_module, "atomic_write_json", fake_atomic_write_json)
    monkeypatch.setattr(io_module, "_git_commit", lambda repository: "abc123")
    monkeypatch.setattr(io_module, "software_versions", lambda: {"python": "3.test"})
    config = {"experiment_name": "fixture", "repo_root": tmp_path}
    inputs = [{"name": "stations", "path": tmp_path / "stations.geojson"}]
    outputs = [tmp_path / "scenario.csv"]

    returned = write_run_record(
        tmp_path / "run-output",
        config=config,
        inputs=inputs,
        outputs=outputs,
        command=["lte-select-sites", "--city", "chicago"],
        timestamp="2026-07-16T10:00:00Z",
        filename="run-select-sites.json",
    )

    expected_path = tmp_path / "run-output" / "run-select-sites.json"
    assert returned == expected_path
    assert captured == {
        "path": expected_path,
        "payload": {
            "timestamp": "2026-07-16T10:00:00Z",
            "command": ["lte-select-sites", "--city", "chicago"],
            "git_commit": "abc123",
            "config": {
                "experiment_name": "fixture",
                "repo_root": str(tmp_path),
            },
            "inputs": [
                {
                    "name": "stations",
                    "path": str(tmp_path / "stations.geojson"),
                }
            ],
            "software": {"python": "3.test"},
            "outputs": [str(tmp_path / "scenario.csv")],
        },
    }


def test_create_data_manifest_combines_provenance_with_file_checksums(tmp_path):
    dataset_dir = tmp_path / "inputs"
    dataset_dir.mkdir()
    (dataset_dir / "a.txt").write_text("alpha", encoding="utf-8")
    metadata = tmp_path / "datasets.yaml"
    metadata.write_text(
        """
schema_version: 2
datasets:
  - dataset_id: fixture
    role: points
    path: inputs
    entrypoint: inputs/a.txt
    source_url: https://example.test/fixture
    provider: Example
    license: CC0-1.0
    download_date: 2026-07-15
    crs: EPSG:3857
    spatial_resolution: vector
    notes: test data
scenarios: []
""".strip(),
        encoding="utf-8",
    )

    output = create_data_manifest(
        metadata,
        tmp_path / "manifest.json",
        repo_root=tmp_path,
        dataset_ids={"fixture"},
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    dataset = payload["datasets"][0]
    assert payload["schema_version"] == 2
    assert payload["scenarios"] == []
    assert dataset["dataset_id"] == "fixture"
    assert dataset["source_url"] == "https://example.test/fixture"
    assert dataset["files"][0]["path"] == "inputs/a.txt"
    assert dataset["files"][0]["size_bytes"] == 5
    assert len(dataset["files"][0]["sha256"]) == 64


def test_software_versions_includes_earth_engine_dependencies(monkeypatch):
    requested = []

    def fake_version(package):
        requested.append(package)
        return f"{package}-version"

    monkeypatch.setattr("lte_scenario_toolkit.io.metadata.version", fake_version)

    versions = software_versions()

    assert versions["earthengine-api"] == "earthengine-api-version"
    assert versions["geemap"] == "geemap-version"
    assert {"earthengine-api", "geemap"}.issubset(requested)
