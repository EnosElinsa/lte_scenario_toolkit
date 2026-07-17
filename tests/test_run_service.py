import errno
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lte_scenario_toolkit.run_service import RunEntry, RunService, StagingRun

CREATED_AT = "2026-07-16T10:00:00Z"


def _write_artifact(run: StagingRun, name: str = "scenario.csv") -> Path:
    artifact = run.path / name
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("cell,lon,lat\n1,0,0\n", encoding="utf-8")
    return artifact


def _begin_with_artifact(
    service: RunService,
    *,
    created_at: str = CREATED_AT,
    parent_run_id: str | None = None,
) -> StagingRun:
    run = service.begin(
        "chicago",
        "default",
        created_at=created_at,
        parent_run_id=parent_run_id,
    )
    _write_artifact(run)
    return run


def _record_payload(run_id: str, **overrides):
    payload = {
        "run_id": run_id,
        "scenario_id": "chicago",
        "profile_id": "default",
        "created_at": CREATED_AT,
        "parent_run_id": None,
        "status": "completed",
        "artifacts": ["scenario.csv"],
        "metadata": {},
        "errors": [],
    }
    payload.update(overrides)
    return payload


def _manual_record(
    root: Path,
    index: int,
    payload,
    *,
    raw_text: str | None = None,
    create_artifact: bool = True,
) -> Path:
    run_id = f"{index:08x}" + "0" * 24
    run_dir = root / "chicago" / "default" / f"20260716-100000-{run_id[:8]}"
    run_dir.mkdir(parents=True)
    if create_artifact:
        (run_dir / "scenario.csv").write_text("ok\n", encoding="utf-8")
    record_path = run_dir / "run.json"
    if raw_text is None:
        record_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        record_path.write_text(raw_text, encoding="utf-8")
    return record_path


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")


def test_begin_publish_writes_completed_run_record_and_atomically_moves_directory(
    tmp_path,
    monkeypatch,
):
    service = RunService(tmp_path)
    run = service.begin("chicago", "default", created_at=CREATED_AT)
    _write_artifact(run)
    moves = []
    original_replace = Path.replace

    def tracking_replace(path, target):
        moves.append((path, Path(target)))
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", tracking_replace)

    published = service.publish(
        run,
        status="completed",
        artifacts=["scenario.csv"],
    )

    assert published == run.final_path
    assert published.is_dir()
    assert not run.path.exists()
    assert published.name.startswith("20260716-100000-")
    assert (run.path, run.final_path) in moves
    payload = json.loads((published / "run.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == run.run_id
    assert payload["status"] == "completed"
    assert payload["artifacts"] == ["scenario.csv"]


def test_begin_twice_in_same_second_allocates_unique_run_and_paths(tmp_path):
    service = RunService(tmp_path)

    first = service.begin("chicago", "default", created_at=CREATED_AT)
    second = service.begin("chicago", "default", created_at=CREATED_AT)

    assert first.run_id != second.run_id
    assert first.path != second.path
    assert first.final_path != second.final_path
    assert first.path.name == f".staging-{first.run_id}"
    assert second.path.name == f".staging-{second.run_id}"
    assert first.path.is_dir() and second.path.is_dir()
    assert not first.final_path.exists() and not second.final_path.exists()


def test_begin_rejects_symlinked_scenario_before_creating_outside_profile(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink_or_skip(root / "chicago", outside, directory=True)

    with pytest.raises(ValueError, match="parent|symlink|path"):
        RunService(root).begin("chicago", "default", created_at=CREATED_AT)

    assert not (outside / "default").exists()


@pytest.mark.parametrize("status", ["completed", "partial"])
def test_publish_requires_at_least_one_requested_artifact(tmp_path, status):
    service = RunService(tmp_path)
    run = service.begin("chicago", "default", created_at=CREATED_AT)

    with pytest.raises(ValueError, match="artifact"):
        service.publish(
            run,
            status=status,
            artifacts=[],
            errors=[{"artifact": "figure", "error": "figure.failed"}],
        )

    assert run.path.is_dir()
    assert not run.final_path.exists()


def test_parent_run_is_preserved_and_discovery_is_stably_sorted(tmp_path):
    service = RunService(tmp_path)
    parent = _begin_with_artifact(service, created_at="2026-07-16T10:00:01Z")
    service.publish(parent, status="completed", artifacts=["scenario.csv"])
    child = _begin_with_artifact(
        service,
        created_at="2026-07-16T10:00:02Z",
        parent_run_id=parent.run_id,
    )
    service.publish(child, status="partial", artifacts=["scenario.csv"])

    discovered = service.discover()

    assert len(discovered.records) == 2
    assert list(discovered.records) == sorted(
        discovered.records,
        key=lambda record: (record["created_at"], record["run_id"]),
    )
    by_id = {record["run_id"]: record for record in discovered.records}
    assert by_id[parent.run_id]["parent_run_id"] is None
    assert by_id[child.run_id]["parent_run_id"] == parent.run_id
    assert discovered.diagnostics == ()


def test_public_run_entry_discovery_returns_validated_paths_and_frozen_records(
    tmp_path,
):
    service = RunService(tmp_path)
    run = _begin_with_artifact(service)
    final = service.publish(
        run,
        status="completed",
        artifacts=["scenario.csv"],
        metadata={"nested": {"value": 1}},
    )

    discovery = service.discover_entries()

    assert discovery.diagnostics == ()
    assert len(discovery.entries) == 1
    entry = discovery.entries[0]
    assert isinstance(entry, RunEntry)
    assert entry.root == tmp_path.resolve()
    assert entry.run_dir == final.resolve()
    assert entry.record["run_id"] == run.run_id
    with pytest.raises(TypeError):
        entry.record["status"] = "partial"
    with pytest.raises(TypeError):
        entry.record["metadata"]["nested"]["value"] = 2
    assert service.entry_for_path(final) == entry


def test_public_run_entry_resolution_revalidates_live_manifest(tmp_path):
    service = RunService(tmp_path / "runs")
    run = _begin_with_artifact(service)
    final = service.publish(run, status="completed", artifacts=["scenario.csv"])
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="root|run|path"):
        service.entry_for_path(outside)

    (final / "scenario.csv").unlink()
    with pytest.raises(ValueError, match="available|valid|run"):
        service.entry_for_path(final)


def test_relocate_to_exact_directory_preserves_unrelated_files_and_moves_record_last(
    tmp_path,
    monkeypatch,
):
    exact = tmp_path / "exact"
    exact.mkdir()
    unrelated = exact / "notes.txt"
    unrelated.write_text("keep\n", encoding="utf-8")
    service = RunService(exact)
    run = _begin_with_artifact(service)
    (run.path / "run-config.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    final = service.publish(
        run,
        status="completed",
        artifacts=["scenario.csv"],
        metadata={
            "entrypoint": ["lte-select-sites", "--output-dir", str(exact)],
            "git_commit": "abc123",
            "software_versions": {"python": "3.test"},
            "parameters": {"rectangle_size_m": 1000},
            "inputs": {"points": {"dataset_id": "points"}},
        },
    )
    moved_to_exact = []
    original_link = os.link

    def tracking_link(path, target, *args, **kwargs):
        destination = Path(target)
        if destination.parent == exact:
            moved_to_exact.append(destination.name)
        return original_link(path, target, *args, **kwargs)

    monkeypatch.setattr(os, "link", tracking_link)

    relocated = service.relocate_to_exact_directory(
        final,
        exact,
        compatibility_record="run-select-sites.json",
    )

    assert relocated == exact.resolve()
    assert unrelated.read_text(encoding="utf-8") == "keep\n"
    assert (exact / "scenario.csv").is_file()
    assert (exact / "run-config.yaml").is_file()
    assert (exact / "run.json").is_file()
    assert (exact / "run-select-sites.json").is_file()
    assert moved_to_exact[-1] == "run.json"
    assert not final.exists()
    assert not (exact / "chicago").exists()
    compatibility = json.loads(
        (exact / "run-select-sites.json").read_text(encoding="utf-8")
    )
    assert compatibility["timestamp"] == CREATED_AT
    assert compatibility["command"][0] == "lte-select-sites"
    assert compatibility["outputs"] == [str(exact / "scenario.csv")]
    assert compatibility["run_id"] == run.run_id


def test_figure_exact_compatibility_record_keeps_cli_provenance_and_figure_outputs(
    tmp_path,
):
    exact = tmp_path / "exact"
    service = RunService(exact)
    run = service.begin("city", "profile", created_at=CREATED_AT)
    _write_artifact(run, "source.csv")
    _write_artifact(run, "terrain.png")
    figure_spec = {"preset": "publication", "dpi": 300}
    source = {"kind": "run", "path": "selection-run", "run_id": "a" * 32}
    final = service.publish(
        run,
        status="completed",
        artifacts=["source.csv", "terrain.png"],
        metadata={
            "run_kind": "figure",
            "entrypoint": ["lte-generate-figures", "--format", "png"],
            "git_commit": "abc123",
            "software_versions": {"python": "3.test"},
            "target_crs": "EPSG:3857",
            "rectangle_size_m": 1000,
            "figure_spec": figure_spec,
            "source": source,
            "inputs": {"csv": {"path": "selection.csv"}},
        },
    )

    service.relocate_to_exact_directory(
        final,
        exact,
        compatibility_record="run-generate-figures.json",
    )

    compatibility = json.loads(
        (exact / "run-generate-figures.json").read_text(encoding="utf-8")
    )
    assert compatibility["command"] == [
        "lte-generate-figures",
        "--format",
        "png",
    ]
    assert compatibility["git_commit"] == "abc123"
    assert compatibility["software"] == {"python": "3.test"}
    assert compatibility["outputs"] == [str(exact / "terrain.png")]
    assert compatibility["config"]["figure_spec"] == figure_spec
    assert compatibility["config"]["source"] == source
    assert compatibility["config"]["target_crs"] == "EPSG:3857"
    assert compatibility["config"]["rectangle_size_m"] == 1000


def test_relocate_to_exact_directory_rejects_all_conflicts_before_writing(tmp_path):
    exact = tmp_path / "exact"
    exact.mkdir()
    existing = exact / "scenario.csv"
    existing.write_text("user-owned\n", encoding="utf-8")
    service = RunService(exact)
    run = _begin_with_artifact(service)
    final = service.publish(
        run,
        status="completed",
        artifacts=["scenario.csv"],
    )

    with pytest.raises(FileExistsError, match="conflict|scenario.csv"):
        service.relocate_to_exact_directory(
            final,
            exact,
            compatibility_record="run-select-sites.json",
        )

    assert existing.read_text(encoding="utf-8") == "user-owned\n"
    assert not (exact / "run.json").exists()
    assert not (exact / "run-select-sites.json").exists()
    assert service.entry_for_path(final).run_id == run.run_id
    assert not (final / "run-select-sites.json").exists()


def test_relocate_to_exact_directory_rolls_back_when_manifest_move_fails(
    tmp_path,
    monkeypatch,
):
    exact = tmp_path / "exact"
    service = RunService(exact)
    run = _begin_with_artifact(service)
    (run.path / "selection.json").write_text("{}\n", encoding="utf-8")
    final = service.publish(
        run,
        status="completed",
        artifacts=["scenario.csv"],
    )
    original_link = os.link
    failed = False

    def fail_manifest_once(path, target, *args, **kwargs):
        nonlocal failed
        destination = Path(target)
        if (
            not failed
            and path.name == "run.json"
            and destination.parent == exact
        ):
            failed = True
            raise OSError("simulated manifest publication failure")
        return original_link(path, target, *args, **kwargs)

    monkeypatch.setattr(os, "link", fail_manifest_once)

    with pytest.raises(OSError, match="manifest publication"):
        service.relocate_to_exact_directory(
            final,
            exact,
            compatibility_record="run-select-sites.json",
        )

    assert failed is True
    assert not (exact / "scenario.csv").exists()
    assert not (exact / "selection.json").exists()
    assert not (exact / "run.json").exists()
    assert not (exact / "run-select-sites.json").exists()
    assert service.entry_for_path(final).run_id == run.run_id
    assert (final / "scenario.csv").is_file()
    assert (final / "selection.json").is_file()
    assert (final / "run.json").is_file()
    assert not (final / "run-select-sites.json").exists()


def test_relocate_to_exact_directory_never_overwrites_a_racing_file(
    tmp_path,
    monkeypatch,
):
    exact = tmp_path / "exact"
    service = RunService(exact)
    run = _begin_with_artifact(service)
    final = service.publish(
        run,
        status="completed",
        artifacts=["scenario.csv"],
    )
    original_link = os.link
    injected = False

    def inject_conflict(path, target, *args, **kwargs):
        nonlocal injected
        destination = Path(target)
        if not injected and destination == exact / "scenario.csv":
            destination.write_text("racing-owner\n", encoding="utf-8")
            injected = True
        return original_link(path, target, *args, **kwargs)

    monkeypatch.setattr(os, "link", inject_conflict)

    with pytest.raises(FileExistsError, match="conflict|scenario.csv"):
        service.relocate_to_exact_directory(
            final,
            exact,
            compatibility_record="run-select-sites.json",
        )

    assert injected is True
    assert (exact / "scenario.csv").read_text(encoding="utf-8") == "racing-owner\n"
    assert not (exact / "run.json").exists()
    assert not (exact / "run-select-sites.json").exists()
    assert service.entry_for_path(final).run_id == run.run_id
    assert (final / "scenario.csv").is_file()
    assert (final / "run.json").is_file()


def test_relocate_to_exact_directory_falls_back_when_hardlinks_are_unsupported(
    tmp_path,
    monkeypatch,
):
    exact = tmp_path / "exact"
    service = RunService(exact)
    run = _begin_with_artifact(service)
    final = service.publish(
        run,
        status="completed",
        artifacts=["scenario.csv"],
    )

    def unsupported_link(*args, **kwargs):
        raise OSError(errno.ENOTSUP, "hardlinks are unavailable")

    monkeypatch.setattr(os, "link", unsupported_link)

    relocated = service.relocate_to_exact_directory(
        final,
        exact,
        compatibility_record="run-select-sites.json",
    )

    assert relocated == exact.resolve()
    assert (exact / "scenario.csv").read_text(encoding="utf-8").startswith(
        "cell,lon,lat"
    )
    assert (exact / "run.json").is_file()
    assert (exact / "run-select-sites.json").is_file()
    assert not final.exists()
    assert not (exact / "chicago").exists()


def test_relocate_to_exact_directory_rolls_back_when_claim_cleanup_fails(
    tmp_path,
    monkeypatch,
):
    exact = tmp_path / "exact"
    service = RunService(exact)
    run = _begin_with_artifact(service)
    final = service.publish(
        run,
        status="completed",
        artifacts=["scenario.csv"],
    )
    claim = exact / ".publish-.exact-directory.lock"
    original_unlink = Path.unlink
    failed = False

    def fail_claim_cleanup_once(path, *args, **kwargs):
        nonlocal failed
        if not failed and path == claim:
            failed = True
            raise OSError("simulated claim cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_claim_cleanup_once)

    with pytest.raises(OSError, match="claim cleanup"):
        service.relocate_to_exact_directory(
            final,
            exact,
            compatibility_record="run-select-sites.json",
        )

    assert failed is True
    assert claim.is_file()
    assert not (exact / "scenario.csv").exists()
    assert not (exact / "run.json").exists()
    assert not (exact / "run-select-sites.json").exists()
    assert service.entry_for_path(final).run_id == run.run_id
    assert (final / "scenario.csv").is_file()
    assert (final / "run.json").is_file()
    assert not (final / "run-select-sites.json").exists()


def test_publish_rejects_unknown_status(tmp_path):
    service = RunService(tmp_path)
    run = _begin_with_artifact(service)

    with pytest.raises(ValueError, match="status"):
        service.publish(run, status="failed", artifacts=["scenario.csv"])


@pytest.mark.parametrize("artifact", ["missing.csv", "directory", "../outside.csv"])
def test_publish_rejects_missing_directory_and_traversal_artifacts(tmp_path, artifact):
    service = RunService(tmp_path / "runs")
    run = service.begin("chicago", "default", created_at=CREATED_AT)
    (run.path / "directory").mkdir()
    (run.path.parent / "outside.csv").write_text("outside\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact"):
        service.publish(run, status="completed", artifacts=[artifact])

    assert run.path.is_dir()
    assert not run.final_path.exists()


def test_publish_rejects_absolute_and_duplicate_artifacts(tmp_path):
    service = RunService(tmp_path / "runs")
    run = _begin_with_artifact(service)
    outside = tmp_path / "outside.csv"
    outside.write_text("outside\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact"):
        service.publish(run, status="completed", artifacts=[outside])
    with pytest.raises(ValueError, match="duplicate"):
        service.publish(
            run,
            status="completed",
            artifacts=["scenario.csv", "scenario.csv"],
        )


@pytest.mark.parametrize("manifest_alias", ["RUN.JSON", "Run.Json"])
def test_publish_rejects_case_insensitive_run_manifest_alias(
    tmp_path,
    manifest_alias,
):
    service = RunService(tmp_path)
    run = service.begin("chicago", "default", created_at=CREATED_AT)
    _write_artifact(run, manifest_alias)

    with pytest.raises(ValueError, match="run.json|reserved"):
        service.publish(run, status="completed", artifacts=[manifest_alias])

    assert run.path.is_dir()
    assert not run.final_path.exists()


def test_publish_rejects_artifact_symlink_escape(tmp_path):
    service = RunService(tmp_path / "runs")
    run = service.begin("chicago", "default", created_at=CREATED_AT)
    outside = tmp_path / "outside.csv"
    outside.write_text("outside\n", encoding="utf-8")
    _symlink_or_skip(run.path / "linked.csv", outside)

    with pytest.raises(ValueError, match="artifact|symlink|outside"):
        service.publish(run, status="completed", artifacts=["linked.csv"])

    assert outside.read_text(encoding="utf-8") == "outside\n"
    assert not run.final_path.exists()


def test_publish_serializes_safe_copies_of_metadata_and_errors(tmp_path):
    service = RunService(tmp_path)
    run = _begin_with_artifact(service)
    marker = tmp_path / "metadata.txt"
    metadata = {"path": marker, "nested": [{"value": 1}]}
    errors = [{"artifact": marker, "failed_at": datetime(2026, 7, 16, tzinfo=timezone.utc)}]

    final = service.publish(
        run,
        status="partial",
        artifacts=["scenario.csv"],
        metadata=metadata,
        errors=errors,
    )
    metadata["nested"][0]["value"] = 2
    errors[0]["artifact"] = "changed"

    payload = json.loads((final / "run.json").read_text(encoding="utf-8"))
    assert payload["metadata"] == {"path": str(marker), "nested": [{"value": 1}]}
    assert payload["errors"][0]["artifact"] == str(marker)
    assert payload["errors"][0]["failed_at"] == "2026-07-16T00:00:00+00:00"


def test_publish_refuses_existing_final_without_overwriting_or_losing_staging(tmp_path):
    service = RunService(tmp_path)
    run = _begin_with_artifact(service)
    run.final_path.mkdir()
    sentinel = run.final_path / "sentinel.txt"
    sentinel.write_text("external owner\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        service.publish(run, status="completed", artifacts=["scenario.csv"])

    assert sentinel.read_text(encoding="utf-8") == "external owner\n"
    assert (run.path / "scenario.csv").is_file()


def test_publish_refuses_existing_cooperative_claim_without_removing_it(tmp_path):
    service = RunService(tmp_path)
    run = _begin_with_artifact(service)
    claim = run.final_path.parent / f".publish-{run.final_path.name}.lock"
    claim.write_text("external owner\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="publish|claim|lock|exists"):
        service.publish(run, status="completed", artifacts=["scenario.csv"])

    assert claim.read_text(encoding="utf-8") == "external owner\n"
    assert run.path.is_dir()
    assert (run.path / "scenario.csv").is_file()
    assert not run.final_path.exists()


def test_publish_normalizes_directory_collision_without_losing_staging(
    tmp_path,
    monkeypatch,
):
    service = RunService(tmp_path)
    run = _begin_with_artifact(service)
    original_replace = Path.replace

    def collide_on_publication(path, target):
        if path == run.path:
            raise OSError(errno.ENOTEMPTY, "directory collision")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", collide_on_publication)

    with pytest.raises(FileExistsError, match="publish|final|exists"):
        service.publish(run, status="completed", artifacts=["scenario.csv"])

    claim = run.final_path.parent / f".publish-{run.final_path.name}.lock"
    assert run.path.is_dir()
    assert not run.final_path.exists()
    assert not claim.exists()


def test_publish_move_failure_keeps_retryable_staging_and_no_final(tmp_path, monkeypatch):
    service = RunService(tmp_path)
    run = _begin_with_artifact(service)
    original_replace = Path.replace

    def fail_publication(path, target):
        if path == run.path:
            raise OSError("publication failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_publication)

    with pytest.raises(OSError, match="publication failed"):
        service.publish(run, status="completed", artifacts=["scenario.csv"])

    assert run.path.is_dir()
    assert (run.path / "run.json").is_file()
    assert not run.final_path.exists()
    claim = run.final_path.parent / f".publish-{run.final_path.name}.lock"
    assert not claim.exists()

    monkeypatch.setattr(Path, "replace", original_replace)
    published = service.publish(
        run,
        status="completed",
        artifacts=["scenario.csv"],
    )
    assert published == run.final_path
    assert published.is_dir()


def test_publish_and_abandon_reject_equal_but_unowned_staging_value(tmp_path):
    service = RunService(tmp_path)
    run = _begin_with_artifact(service)
    forged = replace(run)

    with pytest.raises(ValueError, match="owned|service|staging"):
        service.publish(forged, status="completed", artifacts=["scenario.csv"])
    with pytest.raises(ValueError, match="owned|service|staging"):
        service.abandon(forged)

    assert run.path.is_dir()
    assert (run.path / "scenario.csv").is_file()
    assert not run.final_path.exists()


def test_other_service_cannot_publish_or_abandon_staging(tmp_path):
    owner = RunService(tmp_path)
    other = RunService(tmp_path)
    run = _begin_with_artifact(owner)

    with pytest.raises(ValueError, match="owned|service|staging"):
        other.publish(run, status="completed", artifacts=["scenario.csv"])
    with pytest.raises(ValueError, match="owned|service|staging"):
        other.abandon(run)

    assert run.path.is_dir()
    assert (run.path / "scenario.csv").is_file()
    assert not run.final_path.exists()


@pytest.mark.parametrize(
    ("metadata", "errors", "field"),
    [
        ([], None, "metadata"),
        (None, {}, "errors"),
    ],
)
def test_publish_rejects_invalid_metadata_and_error_container_types(
    tmp_path,
    metadata,
    errors,
    field,
):
    service = RunService(tmp_path)
    run = _begin_with_artifact(service)

    with pytest.raises(ValueError, match=field):
        service.publish(
            run,
            status="completed",
            artifacts=["scenario.csv"],
            metadata=metadata,
            errors=errors,
        )

    assert run.path.is_dir()
    assert not (run.path / "run.json").exists()
    assert not run.final_path.exists()


def test_publish_and_abandon_reject_forged_paths_outside_service_root(tmp_path):
    root = tmp_path / "runs"
    service = RunService(root)
    run = _begin_with_artifact(service)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    forged = replace(run, path=outside, final_path=outside / "final")

    with pytest.raises(ValueError, match="staging|service|path"):
        service.publish(forged, status="completed", artifacts=["sentinel.txt"])
    with pytest.raises(ValueError, match="staging|service|path"):
        service.abandon(forged)

    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert run.path.is_dir()


def test_abandon_is_idempotent_and_never_removes_published_final(tmp_path):
    service = RunService(tmp_path)
    abandoned = service.begin("chicago", "default", created_at=CREATED_AT)

    assert service.abandon(abandoned) is None
    assert service.abandon(abandoned) is None
    assert not abandoned.path.exists()

    published = _begin_with_artifact(
        service,
        created_at="2026-07-16T10:00:01Z",
    )
    final = service.publish(published, status="completed", artifacts=["scenario.csv"])
    service.abandon(published)

    assert final.is_dir()
    assert (final / "scenario.csv").is_file()


def test_abandon_rejects_staging_symlink_without_deleting_target(tmp_path):
    service = RunService(tmp_path / "runs")
    run = service.begin("chicago", "default", created_at=CREATED_AT)
    run.path.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    _symlink_or_skip(run.path, outside, directory=True)

    with pytest.raises(ValueError, match="staging|symlink|path"):
        service.abandon(run)

    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_discover_reports_bad_records_and_keeps_valid_history_unchanged(tmp_path):
    service = RunService(tmp_path)
    valid = _begin_with_artifact(service, created_at="2026-07-16T10:00:03Z")
    valid_final = service.publish(valid, status="completed", artifacts=["scenario.csv"])

    bad_paths = []
    bad_paths.append(
        _manual_record(
            tmp_path,
            1,
            None,
            raw_text="{not json",
        )
    )
    bad_paths.append(_manual_record(tmp_path, 2, ["not", "a", "mapping"]))
    missing = _record_payload(f"{3:08x}" + "0" * 24)
    missing.pop("status")
    bad_paths.append(_manual_record(tmp_path, 3, missing))
    bad_paths.append(
        _manual_record(
            tmp_path,
            4,
            _record_payload(f"{4:08x}" + "0" * 24, status="failed"),
        )
    )
    bad_paths.append(
        _manual_record(
            tmp_path,
            5,
            _record_payload(
                f"{5:08x}" + "0" * 24,
                artifacts=["../outside.csv"],
            ),
        )
    )
    bad_paths.append(
        _manual_record(
            tmp_path,
            6,
            _record_payload(f"{6:08x}" + "0" * 24, artifacts=["missing.csv"]),
            create_artifact=False,
        )
    )
    all_records = bad_paths + [valid_final / "run.json"]
    before = {path: path.read_bytes() for path in all_records}

    discovered = service.discover()

    assert [record["run_id"] for record in discovered.records] == [valid.run_id]
    assert len(discovered.diagnostics) == 6
    assert [item["path"] for item in discovered.diagnostics] == sorted(
        item["path"] for item in discovered.diagnostics
    )
    assert all(item["error"] for item in discovered.diagnostics)
    assert {path: path.read_bytes() for path in all_records} == before


def test_begin_normalizes_aware_times_to_utc_and_defaults_to_utc_z(tmp_path):
    service = RunService(tmp_path)

    converted = service.begin(
        "chicago",
        "default",
        created_at="2026-07-16T18:00:00+08:00",
    )
    generated = service.begin("chicago", "default")

    assert converted.created_at == CREATED_AT
    assert converted.final_path.name.startswith("20260716-100000-")
    assert generated.created_at.endswith("Z")
    parsed = datetime.fromisoformat(generated.created_at.replace("Z", "+00:00"))
    assert parsed.utcoffset() == timezone.utc.utcoffset(parsed)


@pytest.mark.parametrize(
    "created_at",
    ["2026-07-16T10:00:00", "not-a-time", "2026-07-16"],
)
def test_begin_rejects_naive_and_invalid_times(tmp_path, created_at):
    with pytest.raises(ValueError, match="created_at|timezone"):
        RunService(tmp_path).begin(
            "chicago",
            "default",
            created_at=created_at,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scenario_id", "../escape"),
        ("scenario_id", "con"),
        ("scenario_id", "Chicago"),
        ("profile_id", ".."),
        ("profile_id", "lpt1"),
        ("profile_id", "Default Profile"),
    ],
)
def test_begin_rejects_unsafe_scenario_and_profile_slugs(tmp_path, field, value):
    values = {"scenario_id": "chicago", "profile_id": "default"}
    values[field] = value

    with pytest.raises(ValueError, match=field):
        RunService(tmp_path).begin(**values, created_at=CREATED_AT)
