from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import geopandas as gpd
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
    FigureSource,
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
    status: str = "completed",
) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    pd.DataFrame([REQUIRED_ROW]).to_csv(run_dir / "scenario.csv", index=False)
    (run_dir / "run-config.yaml").write_text(
        yaml.safe_dump(
            {
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
                "status": status,
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
    frame = pd.DataFrame([REQUIRED_ROW])
    points = gpd.GeoDataFrame(
        frame.copy(),
        geometry=gpd.points_from_xy(frame["X"], frame["Y"]),
        crs="EPSG:3857",
    )
    return FigureSource(
        path=None,
        csv_path=None,
        csv_identity=None,
        frame=frame,
        rectangle={key: REQUIRED_ROW[key] for key in (
            "rect_id",
            "pt_count",
            "left_x",
            "bottom_y",
            "center_x",
            "center_y",
        )},
        points=points,
        target_crs="EPSG:3857",
        rectangle_size_m=1000.0,
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


def make_run_backed_figure_source(
    tmp_path: Path,
    *,
    root_name: str,
) -> tuple[object, FigureSource]:
    """Publish one real run and attach its identity to a validated source."""

    service = RunService(tmp_path / root_name)
    run = service.begin("city", "default")
    (run.path / "source.csv").write_text("X,Y\n1,2\n", encoding="utf-8")
    run_path = service.publish(
        run,
        status="completed",
        artifacts=["source.csv"],
        metadata={"run_kind": "selection"},
    )
    entry = service.entry_for_path(run_path, run_id=run.run_id)
    source = replace(
        write_in_memory_selection_source(tmp_path),
        path=run_path,
        source_kind="run",
        run_id=run.run_id,
        selection_identity=None,
    )
    return entry, source


def test_figure_controller_releases_run_lease_on_source_invalidation(tmp_path):
    from lte_scenario_toolkit.gui.pages.figures import FigureController
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.run_trash import (
        RunIdentity,
        RunUsageLeaseRegistry,
    )

    entry, source = make_run_backed_figure_source(tmp_path, root_name="source-runs")
    leases = RunUsageLeaseRegistry()
    coordinator = JobCoordinator()
    controller = FigureController(
        tmp_path,
        coordinator,
        usage_leases=leases,
        run_roots=lambda: (entry.root,),
        lease_owner="figures:invalidate",
    )
    identity = RunIdentity.from_entry(entry)
    try:
        controller.set_source(source)
        assert leases.conflicts((identity,)) == ("figures:invalidate",)

        controller.invalidate_source()

        assert leases.conflicts((identity,)) == ()
    finally:
        controller.close()
        coordinator.shutdown()


def test_figure_controller_replaces_run_source_lease_atomically(tmp_path):
    from lte_scenario_toolkit.gui.pages.figures import FigureController
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.run_trash import (
        RunIdentity,
        RunUsageLeaseRegistry,
    )

    entry_a, source_a = make_run_backed_figure_source(tmp_path, root_name="runs-a")
    entry_b, source_b = make_run_backed_figure_source(tmp_path, root_name="runs-b")
    leases = RunUsageLeaseRegistry()
    coordinator = JobCoordinator()
    controller = FigureController(
        tmp_path,
        coordinator,
        usage_leases=leases,
        run_roots=lambda: (entry_a.root, entry_b.root),
        lease_owner="figures:replace",
    )
    identity_a = RunIdentity.from_entry(entry_a)
    identity_b = RunIdentity.from_entry(entry_b)
    try:
        controller.set_source(source_a)
        assert leases.conflicts((identity_a,)) == ("figures:replace",)

        controller.set_source(source_b)

        assert leases.conflicts((identity_a,)) == ()
        assert leases.conflicts((identity_b,)) == ("figures:replace",)
    finally:
        controller.close()
        coordinator.shutdown()


def test_figure_controller_leases_cross_root_source_and_explicit_parent(tmp_path):
    from lte_scenario_toolkit.gui.pages.figures import FigureController
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.run_trash import (
        RunIdentity,
        RunUsageLeaseRegistry,
    )

    source_entry, source = make_run_backed_figure_source(
        tmp_path,
        root_name="source-runs",
    )
    parent_entry, _unused = make_run_backed_figure_source(
        tmp_path,
        root_name="parent-runs",
    )
    leases = RunUsageLeaseRegistry()
    coordinator = JobCoordinator()
    controller = FigureController(
        tmp_path,
        coordinator,
        usage_leases=leases,
        run_roots=lambda: (
            source_entry.root,
            Path(str(source_entry.root)),
            parent_entry.root,
        ),
        lease_owner="figures:cross-root",
    )
    source_identity = RunIdentity.from_entry(source_entry)
    parent_identity = RunIdentity.from_entry(parent_entry)
    try:
        controller.set_source(
            source,
            output_root=parent_entry.root,
            parent_run_id=parent_entry.run_id,
            parent_run_path=parent_entry.run_dir,
        )

        assert leases.conflicts((source_identity,)) == ("figures:cross-root",)
        assert leases.conflicts((parent_identity,)) == ("figures:cross-root",)
    finally:
        controller.close()
        controller.close()
        coordinator.shutdown()
    assert leases.conflicts((source_identity, parent_identity)) == ()


@pytest.mark.parametrize("failure_kind", ["resolution", "lease"])
def test_figure_controller_failed_lease_replacement_is_fail_closed(
    tmp_path,
    failure_kind,
):
    from lte_scenario_toolkit.gui.pages.figures import FigureController
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.run_trash import (
        RunIdentity,
        RunLeaseConflictError,
        RunUsageLeaseRegistry,
    )

    entry_a, source_a = make_run_backed_figure_source(tmp_path, root_name="runs-a")
    entry_b, source_b = make_run_backed_figure_source(tmp_path, root_name="runs-b")
    leases = RunUsageLeaseRegistry()
    identity_b = RunIdentity.from_entry(entry_b)
    mutation = (
        leases.reserve_mutation((identity_b,))
        if failure_kind == "lease"
        else None
    )
    coordinator = JobCoordinator()
    controller = FigureController(
        tmp_path,
        coordinator,
        source=source_a,
        usage_leases=leases,
        run_roots=lambda: (entry_a.root, entry_b.root),
        lease_owner="figures:fail-closed",
    )
    identity_a = RunIdentity.from_entry(entry_a)
    try:
        assert controller.state.source is source_a
        assert controller.state.revision == 0
        assert leases.conflicts((identity_a,)) == ("figures:fail-closed",)
        replacement = (
            replace(source_b, path=source_b.path / "missing")
            if failure_kind == "resolution"
            else source_b
        )
        expected_error = (
            ValueError
            if failure_kind == "resolution"
            else RunLeaseConflictError
        )
        with pytest.raises(expected_error):
            controller.set_source(replacement)

        assert controller.state.source is None
        assert controller.state.source_dirty is True
        assert leases.conflicts((identity_a,)) == ()
        assert leases.conflicts((identity_b,)) == ()
    finally:
        if mutation is not None:
            leases.release_mutation(mutation)
        controller.close()
        coordinator.shutdown()


def test_figure_controller_discards_source_result_after_invalidation(
    tmp_path,
):
    from threading import Event, Thread

    from lte_scenario_toolkit.gui.pages.figures import (
        FigureController,
        _FigureJobResult,
    )
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.run_trash import (
        RunIdentity,
        RunUsageLeaseRegistry,
    )

    entry, source_a = make_run_backed_figure_source(
        tmp_path,
        root_name="source-runs",
    )
    source_b = write_in_memory_selection_source(tmp_path)
    leases = RunUsageLeaseRegistry()
    coordinator = JobCoordinator()
    reached_state_read = Event()
    release_state_read = Event()

    class PausingController(FigureController):
        pause_next_state_read = False

        @property
        def state(self):
            state = super().state
            if self.pause_next_state_read:
                self.pause_next_state_read = False
                reached_state_read.set()
                assert release_state_read.wait(2)
            return state

    controller = PausingController(
        tmp_path,
        coordinator,
        source=source_a,
        usage_leases=leases,
        run_roots=lambda: (entry.root,),
        lease_owner="figures:race",
    )
    identity = RunIdentity.from_entry(entry)
    job = coordinator.start("figure-source")
    controller._job = job
    result = _FigureJobResult(
        "source",
        controller.state.revision,
        source=source_b,
        output_root=tmp_path / "selection-output",
    )
    controller.pause_next_state_read = True
    applied: list[object] = []
    worker = Thread(target=lambda: applied.append(controller._apply_result(job, result)))
    try:
        worker.start()
        assert reached_state_read.wait(2)
        controller.invalidate_source("source changed while loading")
        release_state_read.set()
        worker.join(timeout=2)

        assert not worker.is_alive()
        assert applied and applied[0] is controller.state
        assert controller.state.source is None
        assert controller.state.source_error == "source changed while loading"
        assert leases.conflicts((identity,)) == ()
    finally:
        release_state_read.set()
        worker.join(timeout=2)
        controller.close()
        coordinator.shutdown()


def test_figure_controller_source_result_acquires_run_lease(tmp_path):
    from lte_scenario_toolkit.gui.pages.figures import (
        FigureController,
        _FigureJobResult,
    )
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.run_trash import (
        RunIdentity,
        RunUsageLeaseRegistry,
    )

    entry, source = make_run_backed_figure_source(tmp_path, root_name="runs-b")
    leases = RunUsageLeaseRegistry()
    coordinator = JobCoordinator()
    controller = FigureController(
        tmp_path,
        coordinator,
        usage_leases=leases,
        run_roots=lambda: (entry.root,),
        lease_owner="figures:source-result",
    )
    job = coordinator.start("figure-source")
    controller._job = job
    result = _FigureJobResult(
        "source",
        controller.state.revision,
        source=source,
        output_root=entry.root,
    )
    identity = RunIdentity.from_entry(entry)
    try:
        state = controller._apply_result(job, result)

        assert state.source is source
        assert leases.conflicts((identity,)) == ("figures:source-result",)
    finally:
        controller.close()
        coordinator.shutdown()




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


@pytest.mark.parametrize("source_name", [None, "run.json"])
@pytest.mark.parametrize("status", ["partial", "current"])
def test_run_source_requires_completed_status(tmp_path, source_name, status):
    run_dir = write_figure_run(
        tmp_path,
        target_crs="EPSG:3857",
        status=status,
    )
    source_path = run_dir if source_name is None else run_dir / source_name

    with pytest.raises(ValueError, match="completed"):
        FigureService.load_source(source_path)




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
    path = write_figure_run(tmp_path, target_crs="EPSG:3857")
    pd.DataFrame([{**REQUIRED_ROW, column: value}]).to_csv(
        path / "scenario.csv", index=False
    )

    with pytest.raises(ValueError, match="finite"):
        FigureService.load_source(path)


@pytest.mark.parametrize("point_count", [-1, 1.5, 2, True])
def test_source_requires_integer_nonnegative_matching_point_count(tmp_path, point_count):
    path = write_figure_run(tmp_path, target_crs="EPSG:3857")
    pd.DataFrame([{**REQUIRED_ROW, "pt_count": point_count}]).to_csv(
        path / "scenario.csv", index=False
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
    source = write_in_memory_selection_source(tmp_path)
    result = FigureResult(path=tmp_path, artifacts=())

    assert preview.dpi == 120
    assert preview.max_pixels == 600
    assert publication.dpi == 300
    assert publication.max_pixels == 1800
    assert preview.dpi < publication.dpi
    assert preview.resolved_title(3000, 30) is None
    assert publication.resolved_title(3000, 30) is None
    assert replace(publication, title="  ").resolved_title(3000, 30) is None
    assert (
        replace(publication, title="New York terrain").resolved_title(3000, 30)
        == "New York terrain"
    )
    with pytest.raises(ValueError, match="DPI"):
        replace(publication, dpi=0).validate()
    with pytest.raises(ValueError, match="preset"):
        FigureSpec.from_preset("unknown")
    with pytest.raises(FrozenInstanceError):
        source.target_crs = "EPSG:4326"
    with pytest.raises(FrozenInstanceError):
        result.path = tmp_path / "changed"


def test_preview_writes_only_requested_path_and_requires_recorded_dem(tmp_path):
    source = replace(write_in_memory_selection_source(tmp_path), dem_path=None)
    output = tmp_path / "preview" / "terrain.png"

    with pytest.raises(ValueError, match="DEM provenance"):
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


def test_render_publishes_full_spec_parent_and_self_contained_outputs(
    tmp_path,
    monkeypatch,
):
    dem_path = write_dem(tmp_path)
    source = replace(
        write_in_memory_selection_source(tmp_path),
        dem_path=dem_path,
    )
    spec = replace(
        FigureSpec.from_preset("publication"),
        title="Fixture terrain",
        azimuth=-45.0,
    )
    service = RunService(tmp_path / "runs")
    parent_run_id = "b" * 32
    monkeypatch.setattr(io, "_git_commit", lambda repository: "abc123")
    monkeypatch.setattr(io, "software_versions", lambda: {"python": "3.test"})

    run_dir = FigureService.render(
        source,
        spec,
        service,
        ("png", "html"),
        parent_run_id=parent_run_id,
        entrypoint=("lte-generate-figures", "--format", "png"),
        repository=tmp_path,
    )

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["parent_run_id"] == parent_run_id
    assert record["metadata"]["figure_spec"] == spec.as_dict()
    assert record["metadata"]["requested_formats"] == ["png", "html"]
    assert record["metadata"]["run_kind"] == "figure"
    assert record["metadata"]["entrypoint"] == [
        "lte-generate-figures",
        "--format",
        "png",
    ]
    assert record["metadata"]["git_commit"] == "abc123"
    assert record["metadata"]["software_versions"] == {"python": "3.test"}
    inputs = record["metadata"]["inputs"]
    assert inputs["selection"] == source.selection_identity.as_dict()
    assert inputs["dem"] == {
        "path": str(dem_path.resolve()),
        "size_bytes": dem_path.stat().st_size,
        "fingerprint": "dem-selection",
        "fingerprint_source": "run",
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
        write_in_memory_selection_source(tmp_path),
        dem_path=dem_path,
        dem_fingerprint=None,
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
        write_in_memory_selection_source(tmp_path),
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
    with pytest.raises(ValueError, match="completed"):
        FigureService.load_source(run_dir)


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
        write_in_memory_selection_source(tmp_path),
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
        write_in_memory_selection_source(tmp_path),
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
        write_in_memory_selection_source(tmp_path),
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
        write_in_memory_selection_source(tmp_path),
        dem_path=dem_path,
        dem_fingerprint=None,
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
