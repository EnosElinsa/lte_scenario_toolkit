import errno
import json
import shutil
import stat
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import lte_scenario_toolkit.run_service as run_module
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
        "published_at": CREATED_AT,
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


def test_live_discovery_skips_reserved_trash_payload(tmp_path):
    service = RunService(tmp_path / "results")
    run = _begin_with_artifact(service)
    live_path = service.publish(
        run,
        status="completed",
        artifacts=["scenario.csv"],
    )
    trash_run = (
        service.output_root
        / ".trash"
        / ("f" * 32)
        / "runs"
        / "chicago"
        / "default"
        / live_path.name
    )
    trash_run.parent.mkdir(parents=True)
    shutil.copytree(live_path, trash_run)

    assert [
        entry.run_id for entry in service.discover_entries().entries
    ] == [run.run_id]


def test_live_discovery_prunes_reserved_trash_before_nested_redirect(
    tmp_path,
    monkeypatch,
):
    service = RunService(tmp_path / "results")
    run = _begin_with_artifact(service)
    service.publish(run, status="completed", artifacts=["scenario.csv"])
    nested_redirect = (
        service.output_root / ".trash" / ("f" * 32) / "nested-redirect"
    )
    nested_redirect.mkdir(parents=True)
    original = run_module._is_redirected_path
    monkeypatch.setattr(
        run_module,
        "_is_redirected_path",
        lambda path: Path(path) == nested_redirect or original(Path(path)),
    )

    discovered = service.discover_entries()

    assert [entry.run_id for entry in discovered.entries] == [run.run_id]
    assert discovered.diagnostics == ()


def test_transaction_root_is_exactly_below_reserved_trash(tmp_path):
    service = RunService(tmp_path / "results")

    transaction = service.prepare_trash_transaction("f" * 32)

    assert transaction == service.output_root / ".trash" / ("f" * 32)


@pytest.mark.parametrize(
    "transaction_id",
    [
        "F" * 32,
        "f" * 31,
        "f" * 33,
        "../" + "f" * 32,
        "f" * 16 + "/" + "f" * 16,
        Path("f" * 32),
        1,
        True,
        None,
    ],
)
def test_prepare_trash_transaction_rejects_untrusted_ids(
    tmp_path,
    transaction_id,
):
    service = RunService(tmp_path / "results")

    with pytest.raises(ValueError, match="32 lowercase hexadecimal"):
        service.prepare_trash_transaction(transaction_id)


def _published_entry(service: RunService) -> RunEntry:
    run = _begin_with_artifact(service)
    final_path = service.publish(
        run,
        status="completed",
        artifacts=["scenario.csv"],
        metadata={"run_kind": "selection"},
    )
    return service.entry_for_path(final_path, run_id=run.run_id)


def _trash_destination(transaction: Path, entry: RunEntry) -> Path:
    return (
        transaction
        / "runs"
        / entry.record["scenario_id"]
        / entry.record["profile_id"]
        / entry.run_dir.name
    )


def test_prepare_trash_transaction_rejects_reserved_trash_symlink(tmp_path):
    service = RunService(tmp_path / "results")
    service.output_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink_or_skip(service.output_root / ".trash", outside, directory=True)

    with pytest.raises(ValueError, match="redirected|real directory|trash"):
        service.prepare_trash_transaction("f" * 32)

    assert not (outside / ("f" * 32)).exists()


def test_prepare_trash_transaction_rejects_redirected_transaction(tmp_path):
    service = RunService(tmp_path / "results")
    trash_root = service.output_root / ".trash"
    trash_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    transaction = trash_root / ("f" * 32)
    _symlink_or_skip(transaction, outside, directory=True)

    with pytest.raises(ValueError, match="redirected|real directory|transaction"):
        service.prepare_trash_transaction("f" * 32)

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("redirected_component", ["trash", "transaction"])
def test_prepare_trash_transaction_rejects_junction_or_reparse_component(
    tmp_path,
    monkeypatch,
    redirected_component,
):
    service = RunService(tmp_path / "results")
    service.output_root.mkdir(parents=True)
    trash_root = service.output_root / ".trash"
    trash_root.mkdir()
    transaction = trash_root / ("f" * 32)
    if redirected_component == "transaction":
        transaction.mkdir()
    redirected = trash_root if redirected_component == "trash" else transaction
    original = run_module._is_redirected_path
    monkeypatch.setattr(
        run_module,
        "_is_redirected_path",
        lambda path: Path(path) == redirected or original(Path(path)),
    )

    with pytest.raises(ValueError, match="redirected|real directory|trash"):
        service.prepare_trash_transaction("f" * 32)


def test_move_entry_to_uses_one_atomic_replace_into_exact_transaction(
    tmp_path,
    monkeypatch,
):
    service = RunService(tmp_path / "results")
    entry = _published_entry(service)
    transaction = service.prepare_trash_transaction("f" * 32)
    destination = _trash_destination(transaction, entry)
    original_bytes = {
        path.relative_to(entry.run_dir): path.read_bytes()
        for path in entry.run_dir.rglob("*")
        if path.is_file()
    }
    calls: list[tuple[Path, Path]] = []
    original_replace = Path.replace

    def tracking_replace(path, target):
        calls.append((Path(path), Path(target)))
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", tracking_replace)

    moved = service.move_entry_to(entry, destination)

    assert moved == destination
    assert calls == [(entry.run_dir, destination)]
    assert not entry.run_dir.exists()
    assert {
        path.relative_to(destination): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    } == original_bytes


def test_move_entry_to_rejects_malformed_or_unprepared_destinations(tmp_path):
    service = RunService(tmp_path / "results")
    entry = _published_entry(service)
    transaction = service.prepare_trash_transaction("f" * 32)
    unprepared = service.output_root / ".trash" / ("e" * 32)
    destinations = [
        tmp_path / "browser-selected-path",
        transaction,
        transaction / "payload" / "chicago" / "default" / entry.run_dir.name,
        transaction / "runs" / "other" / "default" / entry.run_dir.name,
        transaction / "runs" / "chicago" / "other" / entry.run_dir.name,
        transaction / "runs" / "chicago" / "default" / "wrong-run",
        _trash_destination(transaction, entry) / "extra",
        unprepared / "runs" / "chicago" / "default" / entry.run_dir.name,
        Path(".trash") / ("f" * 32) / "runs" / "chicago" / "default" / entry.run_dir.name,
        transaction
        / "runs"
        / "chicago"
        / "default"
        / "alias"
        / ".."
        / entry.run_dir.name,
    ]

    for destination in destinations:
        with pytest.raises(
            (FileNotFoundError, ValueError),
            match="destination|transaction|absolute|outside|shape|traversal|prepared",
        ):
            service.move_entry_to(entry, destination)
        assert entry.run_dir.is_dir()


def test_move_entry_to_rejects_existing_destination_without_overwrite(tmp_path):
    service = RunService(tmp_path / "results")
    entry = _published_entry(service)
    transaction = service.prepare_trash_transaction("f" * 32)
    destination = _trash_destination(transaction, entry)
    destination.mkdir(parents=True)
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="destination"):
        service.move_entry_to(entry, destination)

    assert entry.run_dir.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_move_entry_to_rejects_redirected_destination_parent(tmp_path):
    service = RunService(tmp_path / "results")
    entry = _published_entry(service)
    transaction = service.prepare_trash_transaction("f" * 32)
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink_or_skip(transaction / "runs", outside, directory=True)
    destination = _trash_destination(transaction, entry)

    with pytest.raises(ValueError, match="redirected|destination|path"):
        service.move_entry_to(entry, destination)

    assert entry.run_dir.is_dir()
    assert list(outside.iterdir()) == []


def test_move_entry_to_revalidates_immutable_manifest_snapshot(tmp_path):
    service = RunService(tmp_path / "results")
    entry = _published_entry(service)
    transaction = service.prepare_trash_transaction("f" * 32)
    destination = _trash_destination(transaction, entry)
    manifest_path = entry.run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["label"] = "changed after selection"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest|snapshot|changed"):
        service.move_entry_to(entry, destination)

    assert entry.run_dir.is_dir()
    assert not destination.exists()


def test_move_entry_to_rejects_entry_from_another_root(tmp_path):
    source_service = RunService(tmp_path / "source")
    entry = _published_entry(source_service)
    other_service = RunService(tmp_path / "other")
    transaction = other_service.prepare_trash_transaction("f" * 32)
    destination = (
        transaction / "runs" / "chicago" / "default" / entry.run_dir.name
    )

    with pytest.raises(ValueError, match="root|service|entry"):
        other_service.move_entry_to(entry, destination)

    assert entry.run_dir.is_dir()


def test_restore_entry_from_reconstructs_exact_live_path(tmp_path):
    service = RunService(tmp_path / "results")
    entry = _published_entry(service)
    transaction = service.prepare_trash_transaction("f" * 32)
    source = _trash_destination(transaction, entry)
    service.move_entry_to(entry, source)

    restored = service.restore_entry_from(source, entry)

    assert restored == entry.run_dir
    assert entry.run_dir.is_dir()
    assert not source.exists()
    assert service.entry_for_path(entry.run_dir, run_id=entry.run_id).record == entry.record


def test_restore_entry_from_never_overwrites_live_destination(tmp_path):
    service = RunService(tmp_path / "results")
    entry = _published_entry(service)
    transaction = service.prepare_trash_transaction("f" * 32)
    source = _trash_destination(transaction, entry)
    service.move_entry_to(entry, source)
    entry.run_dir.mkdir()
    sentinel = entry.run_dir / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="restore|destination"):
        service.restore_entry_from(source, entry)

    assert source.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_restore_entry_from_rejects_untrusted_source_paths(tmp_path):
    service = RunService(tmp_path / "results")
    entry = _published_entry(service)
    transaction = service.prepare_trash_transaction("f" * 32)
    source = _trash_destination(transaction, entry)
    service.move_entry_to(entry, source)
    outside = tmp_path / "browser-selected-path"
    shutil.copytree(source, outside)
    candidates = [
        outside,
        transaction,
        source / "extra",
        transaction / "runs" / "other" / "default" / source.name,
        source.parent / "alias" / ".." / source.name,
        Path(".trash") / ("f" * 32) / "runs" / "chicago" / "default" / source.name,
    ]

    for candidate in candidates:
        with pytest.raises(
            (FileNotFoundError, ValueError),
            match="source|transaction|absolute|outside|shape|traversal",
        ):
            service.restore_entry_from(candidate, entry)
        assert source.is_dir()
        assert not entry.run_dir.exists()


def test_restore_entry_from_rejects_manifest_mismatch(tmp_path):
    service = RunService(tmp_path / "results")
    entry = _published_entry(service)
    transaction = service.prepare_trash_transaction("f" * 32)
    source = _trash_destination(transaction, entry)
    service.move_entry_to(entry, source)
    manifest_path = source / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["label"] = "tampered in trash"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest|snapshot|changed"):
        service.restore_entry_from(source, entry)

    assert source.is_dir()
    assert not entry.run_dir.exists()


def test_restore_entry_from_rejects_untrusted_snapshot_destination(tmp_path):
    service = RunService(tmp_path / "results")
    entry = _published_entry(service)
    transaction = service.prepare_trash_transaction("f" * 32)
    source = _trash_destination(transaction, entry)
    service.move_entry_to(entry, source)
    forged = object.__new__(RunEntry)
    object.__setattr__(forged, "root", entry.root)
    object.__setattr__(forged, "run_dir", tmp_path / "browser-restore-target")
    object.__setattr__(forged, "record", entry.record)

    with pytest.raises(ValueError, match="destination|path|snapshot|entry"):
        service.restore_entry_from(source, forged)

    assert source.is_dir()
    assert not (tmp_path / "browser-restore-target").exists()


def test_restore_entry_from_rejects_redirected_destination_ancestor(
    tmp_path,
    monkeypatch,
):
    service = RunService(tmp_path / "results")
    entry = _published_entry(service)
    transaction = service.prepare_trash_transaction("f" * 32)
    source = _trash_destination(transaction, entry)
    service.move_entry_to(entry, source)
    redirected = entry.run_dir.parent
    original = run_module._is_redirected_path
    monkeypatch.setattr(
        run_module,
        "_is_redirected_path",
        lambda path: Path(path) == redirected or original(Path(path)),
    )

    with pytest.raises(ValueError, match="redirected|destination|path"):
        service.restore_entry_from(source, entry)

    assert source.is_dir()
    assert not entry.run_dir.exists()


def test_remove_trash_transaction_removes_only_exact_transaction(tmp_path):
    service = RunService(tmp_path / "results")
    target = service.prepare_trash_transaction("f" * 32)
    other = service.prepare_trash_transaction("e" * 32)
    (target / "payload.txt").write_text("remove\n", encoding="utf-8")
    sentinel = other / "keep.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    service.remove_trash_transaction("f" * 32)

    assert not target.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert service.output_root.is_dir()
    assert (service.output_root / ".trash").is_dir()


@pytest.mark.parametrize(
    "untrusted_target",
    [
        ".",
        ".trash",
        "../results",
        "f" * 32 + "/runs",
        "C:/Users/example/Documents",
        Path("f" * 32),
        None,
    ],
)
def test_remove_trash_transaction_rejects_recursive_user_targets(
    tmp_path,
    monkeypatch,
    untrusted_target,
):
    service = RunService(tmp_path / "results")
    service.prepare_trash_transaction("f" * 32)
    calls = []
    monkeypatch.setattr(shutil, "rmtree", lambda path: calls.append(Path(path)))

    with pytest.raises(ValueError, match="32 lowercase hexadecimal"):
        service.remove_trash_transaction(untrusted_target)

    assert calls == []


def test_remove_trash_transaction_rejects_absent_exact_target(tmp_path):
    service = RunService(tmp_path / "results")
    service.prepare_trash_transaction("f" * 32)

    with pytest.raises(FileNotFoundError, match="trash transaction"):
        service.remove_trash_transaction("e" * 32)

    assert (service.output_root / ".trash").is_dir()


def test_remove_trash_transaction_rejects_redirect_to_another_transaction(
    tmp_path,
):
    service = RunService(tmp_path / "results")
    trash_root = service.output_root / ".trash"
    trash_root.mkdir(parents=True)
    other = trash_root / ("e" * 32)
    other.mkdir()
    sentinel = other / "keep.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    _symlink_or_skip(trash_root / ("f" * 32), other, directory=True)

    with pytest.raises(ValueError, match="redirected|transaction|path"):
        service.remove_trash_transaction("f" * 32)

    assert sentinel.read_text(encoding="utf-8") == "keep\n"






@pytest.mark.parametrize("kind", ["junction", "reparse"])
def test_redirected_path_detector_covers_junctions_and_reparse_points(
    monkeypatch,
    kind,
):
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    monkeypatch.setattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", reparse_flag, raising=False)

    class FakePath:
        def is_symlink(self):
            return False

        def is_junction(self):
            return kind == "junction"

        def lstat(self):
            attributes = reparse_flag if kind == "reparse" else 0
            return SimpleNamespace(st_file_attributes=attributes)

    assert run_module._is_redirected_path(FakePath()) is True


def test_entry_and_run_entry_reject_lexical_traversal_alias(tmp_path):
    service = RunService(tmp_path / "results")
    run = _begin_with_artifact(service)
    final = service.publish(run, status="completed", artifacts=["scenario.csv"])
    alias_parent = final.parent / "alias-parent"
    alias_parent.mkdir()
    alias = alias_parent / ".." / final.name
    record = json.loads((final / "run.json").read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="traversal|redirected|path"):
        service.entry_for_path(alias)
    with pytest.raises(ValueError, match="traversal|redirected|path"):
        RunEntry(root=service.output_root, run_dir=alias, record=record)


def test_artifact_rejects_redirected_ancestor_chain(tmp_path, monkeypatch):
    service = RunService(tmp_path / "results")
    run = service.begin("chicago", "default", created_at=CREATED_AT)
    artifact = _write_artifact(run, "nested/scenario.csv")
    redirected = artifact.parent
    original = run_module._is_redirected_path
    monkeypatch.setattr(
        run_module,
        "_is_redirected_path",
        lambda path: Path(path) == redirected or original(Path(path)),
    )

    with pytest.raises(ValueError, match="redirected|artifact|path"):
        service.publish(
            run,
            status="completed",
            artifacts=["nested/scenario.csv"],
        )

    assert artifact.is_file()


def test_entry_for_path_rejects_redirected_ancestor_chain(tmp_path, monkeypatch):
    service = RunService(tmp_path / "results")
    run = _begin_with_artifact(service)
    final = service.publish(run, status="completed", artifacts=["scenario.csv"])
    redirected = final.parent.parent
    original = run_module._is_redirected_path
    monkeypatch.setattr(
        run_module,
        "_is_redirected_path",
        lambda path: Path(path) == redirected or original(Path(path)),
    )

    with pytest.raises(ValueError, match="redirected|run path|path"):
        service.entry_for_path(final)




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
