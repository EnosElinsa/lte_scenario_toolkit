from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from lte_scenario_toolkit import io
from lte_scenario_toolkit.figure_service import (
    FigureResult,
    FigureService,
    FigureSpec,
    SelectionFigureIdentity,
)
from lte_scenario_toolkit.run_service import RunService

REQUIRED_ROW = {
    "rect_id": 1,
    "pt_count": 1,
    "left_x": 0.0,
    "bottom_y": 0.0,
    "center_x": 500.0,
    "center_y": 500.0,
    "X": 100.0,
    "Y": 200.0,
    "elevation": 12.5,
}


def write_single_csv(tmp_path: Path, *, row: dict | None = None) -> Path:
    path = tmp_path / "scenario.csv"
    pd.DataFrame([REQUIRED_ROW if row is None else row]).to_csv(path, index=False)
    return path


def write_dem(tmp_path: Path) -> Path:
    path = tmp_path / "dem.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 1000, 250, 250),
    ) as dem:
        dem.write(np.arange(16, dtype="float32").reshape(4, 4), 1)
    return path


def write_figure_run(
    tmp_path: Path,
    *,
    target_crs: str,
    rectangle_size_m: int = 1000,
    artifacts: list[str] | None = None,
    dem_path: Path | None = None,
) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    pd.DataFrame([REQUIRED_ROW]).to_csv(run_dir / "scenario.csv", index=False)
    (run_dir / "run-config.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "profile": {
                    "id": "fixture",
                    "display_name": "Fixture",
                    "scenario_id": "city",
                },
                "inputs": {"points_dataset_id": "points"},
                "experiment": {"random_seed": 7},
                "spatial": {
                    "target_crs": target_crs,
                    "rectangle_size_m": rectangle_size_m,
                    "target_base_station_count": 1,
                    "count_tolerance": 0,
                },
                "scan": {
                    "mode": "fast",
                    "strategy": "uniform",
                    "step_m": 10,
                    "max_rectangles": 1,
                    "minimum_center_spacing_m": 1000,
                },
                "outputs": {"root": "results"},
                "figures": {"preset": "publication"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    metadata = {
        "inputs": {
            "dem": {
                "dataset_id": "dem",
                "fingerprint": "fixture",
            }
        }
    }
    if dem_path is not None:
        metadata["inputs"]["dem"]["path"] = str(dem_path.resolve())
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "a" * 32,
                "scenario_id": "city",
                "profile_id": "fixture",
                "artifacts": artifacts or ["scenario.csv"],
                "metadata": metadata,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def write_current_selection_run(tmp_path: Path) -> tuple[Path, Path]:
    dem_path = write_dem(tmp_path)
    run_dir = write_figure_run(
        tmp_path,
        target_crs="EPSG:3857",
        dem_path=dem_path,
    )
    run_id = "a" * 32
    frame = pd.DataFrame(
        [
            {
                **REQUIRED_ROW,
                "run_id": run_id,
                "scenario_id": "city",
                "profile_id": "fixture",
                "candidate_id": "candidate-0001",
            }
        ]
    )
    frame.to_csv(run_dir / "scenario.csv", index=False)
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    record["metadata"].update(
        {
            "run_kind": "selection",
            "candidate": {
                "candidate_id": "candidate-0001",
                "point_count": 1,
                "center_x": 500.0,
                "center_y": 500.0,
            },
        }
    )
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir, dem_path


def write_in_memory_selection_source(tmp_path: Path):
    source = FigureService.load_source(write_single_csv(tmp_path))
    return replace(
        source,
        path=None,
        csv_path=None,
        csv_identity=None,
        source_kind="selection",
        dem_path=write_dem(tmp_path),
        dem_fingerprint="dem-selection",
        scenario_id="city",
        profile_id="fixture",
        selection_identity=SelectionFigureIdentity(
            scenario_id="city",
            profile_id="fixture",
            profile_fingerprint="profile-selection",
            points_fingerprint="points-selection",
            boundary_fingerprint="boundary-selection",
            dem_fingerprint="dem-selection",
            scan_algorithm_version="row-sweep-v1",
            scan_checked_positions=10,
            scan_total_positions=10,
            candidate_index=1,
            candidate_flat_grid_id=9,
            candidate_point_count=1,
            candidate_left_x=0.0,
            candidate_bottom_y=0.0,
            candidate_center_x=500.0,
            candidate_center_y=500.0,
        ),
    )


def test_legacy_multi_rectangle_csv_requires_rect_id(tmp_path):
    path = tmp_path / "multi.csv"
    pd.DataFrame(
        [
            {**REQUIRED_ROW, "rect_id": 1},
            {
                **REQUIRED_ROW,
                "rect_id": 2,
                "left_x": 1000.0,
                "center_x": 1500.0,
                "X": 1100.0,
            },
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="rect_id"):
        FigureService.load_source(path)
    inspection = FigureService.inspect_source(path)
    assert inspection.rectangle_ids == (1, 2)
    assert inspection.requires_rectangle is True
    assert inspection.source_kind == "csv"
    assert any("EPSG:3857" in warning for warning in inspection.warnings)
    loaded = FigureService.load_source(path, rect_id=2)
    assert loaded.rectangle["rect_id"] == 2
    assert loaded.frame["rect_id"].unique().tolist() == [2]


def test_run_snapshot_is_authoritative_crs_and_rectangle_size(tmp_path):
    run_dir = write_figure_run(
        tmp_path,
        target_crs="EPSG:32616",
        rectangle_size_m=750,
    )

    source = FigureService.load_source(run_dir)

    assert source.target_crs == "EPSG:32616"
    assert source.rectangle_size_m == 750
    assert source.warnings == ()


def test_run_json_is_an_accepted_source_and_dem_path_is_validated(tmp_path):
    dem_path = write_dem(tmp_path)
    run_dir = write_figure_run(
        tmp_path,
        target_crs="EPSG:3857",
        dem_path=dem_path,
    )

    source = FigureService.load_source(run_dir / "run.json")

    assert source.path == run_dir.resolve()
    assert source.dem_path == dem_path.resolve()
    assert source.run_id == "a" * 32


def test_csv_inside_run_uses_its_verified_snapshot_and_dem_context(tmp_path):
    dem_path = write_dem(tmp_path)
    run_dir = write_figure_run(
        tmp_path,
        target_crs="EPSG:3857",
        dem_path=dem_path,
    )

    source = FigureService.load_source(run_dir / "scenario.csv")

    assert source.path == run_dir.resolve()
    assert source.dem_path == dem_path.resolve()
    assert source.warnings == ()


def test_run_source_rejects_ambiguous_and_escaping_csv_artifacts(tmp_path):
    ambiguous = write_figure_run(
        tmp_path / "ambiguous",
        target_crs="EPSG:3857",
        artifacts=["scenario.csv", "second.csv"],
    )
    (ambiguous / "second.csv").write_text(
        (ambiguous / "scenario.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one scenario CSV"):
        FigureService.load_source(ambiguous)

    outside_root = tmp_path / "escaping"
    outside_root.mkdir()
    outside = outside_root / "outside.csv"
    pd.DataFrame([REQUIRED_ROW]).to_csv(outside, index=False)
    run_dir = write_figure_run(
        outside_root,
        target_crs="EPSG:3857",
        artifacts=["../outside.csv"],
    )
    with pytest.raises(ValueError, match="contained relative"):
        FigureService.load_source(run_dir)


def test_legacy_csv_falls_back_with_warning(tmp_path):
    path = write_single_csv(tmp_path)

    source = FigureService.load_source(path)

    assert source.target_crs == "EPSG:3857"
    assert "EPSG:3857" in source.warnings[0]
    assert source.dem_path is None


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("X", np.nan),
        ("center_y", np.inf),
        ("pt_count", -np.inf),
        ("elevation", np.nan),
    ],
)
def test_source_rejects_non_finite_required_numeric_values(tmp_path, column, value):
    path = write_single_csv(tmp_path, row={**REQUIRED_ROW, column: value})

    with pytest.raises(ValueError, match="finite"):
        FigureService.load_source(path)


@pytest.mark.parametrize("point_count", [-1, 1.5, 2, True])
def test_source_requires_integer_nonnegative_matching_point_count(tmp_path, point_count):
    path = write_single_csv(
        tmp_path,
        row={**REQUIRED_ROW, "pt_count": point_count},
    )

    with pytest.raises(ValueError, match="pt_count"):
        FigureService.load_source(path)


def test_current_selection_run_cross_checks_matching_snapshots_and_trace_columns(tmp_path):
    run_dir, dem_path = write_current_selection_run(tmp_path)

    source = FigureService.load_source(run_dir)

    assert source.run_id == "a" * 32
    assert source.dem_path == dem_path.resolve()
    assert source.dem_fingerprint == "fixture"


def test_selection_trace_ids_preserve_leading_zeroes(tmp_path):
    run_dir, _ = write_current_selection_run(tmp_path)
    run_id = "0" * 31 + "1"
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    record["run_id"] = run_id
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    frame = pd.read_csv(run_dir / "scenario.csv", dtype={"run_id": "string"})
    frame.loc[:, "run_id"] = run_id
    frame.to_csv(run_dir / "scenario.csv", index=False)

    source = FigureService.load_source(run_dir)

    assert source.run_id == run_id


@pytest.mark.parametrize(
    "case",
    [
        "snapshot-profile",
        "snapshot-profile-section",
        "csv-run-id",
        "csv-scenario-id",
        "candidate-id",
        "candidate-count",
        "candidate-center",
        "candidate-section",
    ],
)
def test_current_selection_run_rejects_cross_source_drift(tmp_path, case):
    run_dir, _ = write_current_selection_run(tmp_path)
    if case == "snapshot-profile":
        snapshot = yaml.safe_load(
            (run_dir / "run-config.yaml").read_text(encoding="utf-8")
        )
        snapshot["profile"]["id"] = "other-profile"
        (run_dir / "run-config.yaml").write_text(
            yaml.safe_dump(snapshot),
            encoding="utf-8",
        )
    elif case == "snapshot-profile-section":
        snapshot = yaml.safe_load(
            (run_dir / "run-config.yaml").read_text(encoding="utf-8")
        )
        snapshot["profile"] = None
        (run_dir / "run-config.yaml").write_text(
            yaml.safe_dump(snapshot),
            encoding="utf-8",
        )
    elif case.startswith("csv-") or case == "candidate-id":
        frame = pd.read_csv(run_dir / "scenario.csv")
        column = {
            "csv-run-id": "run_id",
            "csv-scenario-id": "scenario_id",
            "candidate-id": "candidate_id",
        }[case]
        frame.loc[:, column] = "mismatch"
        frame.to_csv(run_dir / "scenario.csv", index=False)
    else:
        record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if case == "candidate-section":
            record["metadata"]["candidate"] = "invalid"
        else:
            candidate = record["metadata"]["candidate"]
            if case == "candidate-count":
                candidate["point_count"] = 2
            else:
                candidate["center_x"] = 999.0
        (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="match|mismatch|mapping"):
        FigureService.load_source(run_dir)


def test_figure_models_are_immutable_and_presets_are_validated(tmp_path):
    preview = FigureSpec.from_preset("preview")
    publication = FigureSpec.from_preset("publication")
    source = FigureService.load_source(write_single_csv(tmp_path))
    result = FigureResult(path=tmp_path, artifacts=())

    assert preview.dpi == 120
    assert preview.max_pixels == 600
    assert publication.dpi == 300
    assert publication.max_pixels == 1800
    assert preview.dpi < publication.dpi
    with pytest.raises(ValueError, match="DPI"):
        replace(publication, dpi=0).validate()
    with pytest.raises(ValueError, match="preset"):
        FigureSpec.from_preset("unknown")
    with pytest.raises(FrozenInstanceError):
        source.target_crs = "EPSG:4326"
    with pytest.raises(FrozenInstanceError):
        result.path = tmp_path / "changed"


def test_preview_writes_only_requested_path_and_requires_resolved_dem(tmp_path):
    source = FigureService.load_source(write_single_csv(tmp_path))
    output = tmp_path / "preview" / "terrain.png"

    with pytest.raises(ValueError, match="DEM.*resolved"):
        FigureService.preview(source, FigureSpec.from_preset("preview"), output)
    assert not output.exists()

    dem_path = write_dem(tmp_path)
    source = replace(source, dem_path=dem_path)
    returned = FigureService.preview(
        source,
        FigureSpec.from_preset("preview"),
        output,
    )

    assert returned == output.resolve()
    assert output.stat().st_size > 0
    assert sorted(path.name for path in output.parent.iterdir()) == ["terrain.png"]


def test_render_publishes_full_spec_parent_and_self_contained_outputs(tmp_path):
    dem_path = write_dem(tmp_path)
    source = replace(
        FigureService.load_source(write_single_csv(tmp_path)),
        dem_path=dem_path,
    )
    spec = replace(
        FigureSpec.from_preset("publication"),
        title="Fixture terrain",
        azimuth=-45.0,
    )
    service = RunService(tmp_path / "runs")
    parent_run_id = "b" * 32

    run_dir = FigureService.render(
        source,
        spec,
        service,
        ("png", "html"),
        parent_run_id=parent_run_id,
    )

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["parent_run_id"] == parent_run_id
    assert record["metadata"]["figure_spec"] == spec.as_dict()
    assert record["metadata"]["requested_formats"] == ["png", "html"]
    assert record["metadata"]["run_kind"] == "figure"
    inputs = record["metadata"]["inputs"]
    assert inputs["csv"] == {
        "path": str(source.csv_path.resolve()),
        "size_bytes": source.csv_path.stat().st_size,
        "sha256": hashlib.sha256(source.csv_path.read_bytes()).hexdigest(),
    }
    assert inputs["dem"] == {
        "path": str(dem_path.resolve()),
        "size_bytes": dem_path.stat().st_size,
        "fingerprint": hashlib.sha256(dem_path.read_bytes()).hexdigest(),
        "fingerprint_source": "sha256",
    }
    assert set(record["artifacts"]) == {
        "source.csv",
        "terrain.png",
        "terrain.html",
    }
    html = (run_dir / "terrain.html").read_text(encoding="utf-8")
    assert "plotly.js" in html.lower()
    assert '<script src="http' not in html.lower()


def test_render_prepares_terrain_once_for_all_requested_formats(
    tmp_path,
    monkeypatch,
):
    dem_path = write_dem(tmp_path)
    source = replace(
        FigureService.load_source(write_single_csv(tmp_path)),
        dem_path=dem_path,
    )
    from lte_scenario_toolkit import figure_service

    calls = []
    actual = figure_service.visualization.prepare_terrain_arrays

    def record_prepare(*args, **kwargs):
        calls.append(kwargs["max_pixels"])
        return actual(*args, **kwargs)

    monkeypatch.setattr(
        figure_service.visualization,
        "prepare_terrain_arrays",
        record_prepare,
    )

    FigureService.render(
        source,
        FigureSpec.from_preset("preview"),
        RunService(tmp_path / "runs"),
        ("png", "html"),
    )

    assert calls == [600]


def test_render_publishes_partial_run_for_per_format_failure(
    tmp_path,
    monkeypatch,
):
    dem_path = write_dem(tmp_path)
    source = replace(
        FigureService.load_source(write_single_csv(tmp_path)),
        dem_path=dem_path,
    )
    from lte_scenario_toolkit import figure_service

    actual = figure_service.visualization.render_3d_terrain

    def fail_html(*args, **kwargs):
        if kwargs.get("html_path") is not None:
            raise RuntimeError("HTML failed")
        return actual(*args, **kwargs)

    monkeypatch.setattr(figure_service.visualization, "render_3d_terrain", fail_html)

    run_dir = FigureService.render(
        source,
        FigureSpec.from_preset("preview"),
        RunService(tmp_path / "runs"),
        ("png", "html"),
    )

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "partial"
    assert record["artifacts"] == ["source.csv", "terrain.png"]
    assert record["errors"] == [
        {
            "artifact": "html",
            "code": "figure.html.failed",
            "message": "RuntimeError: HTML failed",
        }
    ]
    reloaded = FigureService.load_source(run_dir)
    assert reloaded.rectangle == source.rectangle
    assert reloaded.dem_path == dem_path.resolve()


def test_rendered_selection_run_contains_reloadable_validated_source_snapshot(
    tmp_path,
):
    source = write_in_memory_selection_source(tmp_path)

    run_dir = FigureService.render(
        source,
        FigureSpec.from_preset("preview"),
        RunService(tmp_path / "runs"),
        ("png",),
    )

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["artifacts"] == ["source.csv", "terrain.png"]
    assert record["metadata"]["source"]["artifact"] == "source.csv"
    assert record["metadata"]["inputs"]["selection"][
        "candidate_flat_grid_id"
    ] == 9
    loaded = FigureService.load_source(run_dir)
    assert loaded.source_kind == "run"
    assert loaded.rectangle == source.rectangle
    assert loaded.frame["X"].tolist() == source.frame["X"].tolist()
    assert loaded.dem_path == source.dem_path.resolve()
    assert loaded.target_crs == source.target_crs

    (run_dir / "source.csv").unlink()
    with pytest.raises(ValueError, match="artifact|source|CSV"):
        FigureService.load_source(run_dir)


def test_zero_station_selection_figure_snapshot_is_reloadable(tmp_path):
    source = write_in_memory_selection_source(tmp_path)
    rectangle = {**source.rectangle, "pt_count": 0}
    identity = replace(source.selection_identity, candidate_point_count=0)
    zero_source = replace(
        source,
        frame=source.frame.iloc[:0].copy(),
        points=source.points.iloc[:0].copy(),
        rectangle=rectangle,
        selection_identity=identity,
    )

    run_dir = FigureService.render(
        zero_source,
        FigureSpec.from_preset("preview"),
        RunService(tmp_path / "zero-runs"),
        ("png",),
    )
    loaded = FigureService.load_source(run_dir)

    assert loaded.frame.empty
    assert loaded.points.empty
    assert loaded.rectangle == rectangle


def test_render_abandons_run_when_no_requested_format_succeeds(tmp_path, monkeypatch):
    dem_path = write_dem(tmp_path)
    source = replace(
        FigureService.load_source(write_single_csv(tmp_path)),
        dem_path=dem_path,
    )
    from lte_scenario_toolkit import figure_service

    monkeypatch.setattr(
        figure_service.visualization,
        "render_3d_terrain",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    output_root = tmp_path / "runs"

    with pytest.raises(ValueError, match="No requested figure"):
        FigureService.render(
            source,
            FigureSpec.from_preset("preview"),
            RunService(output_root),
            ("png",),
        )

    assert not list(output_root.glob("*/*/.staging-*"))
    assert RunService(output_root).discover().records == ()


@pytest.mark.parametrize(
    "formats",
    [(), "png", ("pdf",), ("png", "PNG")],
)
def test_render_rejects_invalid_formats_before_creating_run(tmp_path, formats):
    dem_path = write_dem(tmp_path)
    source = replace(
        FigureService.load_source(write_single_csv(tmp_path)),
        dem_path=dem_path,
    )
    output_root = tmp_path / "runs"

    with pytest.raises(ValueError, match="format"):
        FigureService.render(
            source,
            FigureSpec.from_preset("preview"),
            RunService(output_root),
            formats,
        )

    assert not output_root.exists()


def test_render_abandons_incomplete_staging_when_publish_fails_early(
    tmp_path,
    monkeypatch,
):
    dem_path = write_dem(tmp_path)
    source = replace(
        FigureService.load_source(write_single_csv(tmp_path)),
        dem_path=dem_path,
    )
    output_root = tmp_path / "runs"
    service = RunService(output_root)
    monkeypatch.setattr(
        service,
        "publish",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("publish failed")),
    )

    with pytest.raises(RuntimeError, match="publish failed"):
        FigureService.render(
            source,
            FigureSpec.from_preset("preview"),
            service,
            ("png",),
        )

    assert not list(output_root.glob("*/*/.staging-*"))


def test_render_reuses_recorded_dem_fingerprint_and_hashes_csv(tmp_path, monkeypatch):
    dem_path = write_dem(tmp_path)
    run_dir = write_figure_run(
        tmp_path,
        target_crs="EPSG:3857",
        dem_path=dem_path,
    )
    source = FigureService.load_source(run_dir)
    calls = []
    actual_sha256 = io.sha256_file

    def record_hash(path):
        calls.append(Path(path).resolve())
        return actual_sha256(path)

    monkeypatch.setattr(io, "sha256_file", record_hash)

    figure_run = FigureService.render(
        source,
        FigureSpec.from_preset("preview"),
        RunService(tmp_path / "figure-runs"),
        ("png",),
    )

    metadata = json.loads((figure_run / "run.json").read_text(encoding="utf-8"))[
        "metadata"
    ]
    assert metadata["inputs"]["dem"]["fingerprint"] == "fixture"
    assert calls == [source.csv_path.resolve()]


def test_input_identity_failure_happens_before_run_begin(tmp_path, monkeypatch):
    dem_path = write_dem(tmp_path)
    source = replace(
        FigureService.load_source(write_single_csv(tmp_path)),
        dem_path=dem_path,
    )
    output_root = tmp_path / "runs"
    monkeypatch.setattr(
        io,
        "sha256_file",
        lambda path: (_ for _ in ()).throw(OSError("identity failed")),
    )

    with pytest.raises(OSError, match="identity failed"):
        FigureService.render(
            source,
            FigureSpec.from_preset("preview"),
            RunService(output_root),
            ("png",),
        )

    assert not output_root.exists()
