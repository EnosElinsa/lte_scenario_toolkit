from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest

import lte_scenario_toolkit.run_service as run_service_module
from lte_scenario_toolkit.run_service import RunEntry, RunService
from lte_scenario_toolkit.run_trash import (
    RunDependencyError,
    RunDependencyGraph,
    RunIdentity,
    RunLeaseConflictError,
    RunUsageLeaseRegistry,
)

CREATED_AT = "2026-07-20T10:00:00Z"


def _publish_run(
    root: Path,
    *,
    run_id: str,
    created_at: str = CREATED_AT,
    parent_run_id: str | None = None,
    run_kind: str,
    source: object | None = None,
) -> RunEntry:
    service = RunService(root)
    with patch.object(
        run_service_module.uuid,
        "uuid4",
        return_value=UUID(hex=run_id),
    ):
        run = service.begin(
            "chicago",
            "default",
            created_at=created_at,
            parent_run_id=parent_run_id,
        )
    artifact_name = "selection.csv" if run_kind == "selection" else "figure.png"
    (run.path / artifact_name).write_bytes(b"artifact\n")
    metadata: dict[str, object] = {"run_kind": run_kind}
    if source is not None:
        metadata["source"] = source
    final_path = service.publish(
        run,
        status="completed",
        artifacts=[artifact_name],
        metadata=metadata,
    )
    return service.entry_for_path(final_path, run_id=run_id)


def publish_selection(
    root: Path,
    *,
    run_id: str,
    created_at: str = CREATED_AT,
) -> RunEntry:
    return _publish_run(
        root,
        run_id=run_id,
        created_at=created_at,
        run_kind="selection",
    )


def publish_figure(
    root: Path,
    *,
    run_id: str,
    parent: RunEntry,
    created_at: str = CREATED_AT,
) -> RunEntry:
    return _publish_run(
        root,
        run_id=run_id,
        created_at=created_at,
        parent_run_id=parent.run_id,
        run_kind="figure",
        source={"path": str(parent.run_dir), "run_id": parent.run_id},
    )


def test_run_identity_normalizes_paths_without_requiring_live_entry(tmp_path):
    root = tmp_path / "results"
    expected = root / "chicago" / "default" / "missing-run"

    identity = RunIdentity(root, "a" * 32, expected)

    assert identity.root == root.resolve()
    assert identity.expected_path == expected.resolve()
    assert not identity.expected_path.exists()


@pytest.mark.parametrize("run_id", ["A" * 32, "a" * 31, "g" * 32, 1, None])
def test_run_identity_rejects_invalid_run_ids(tmp_path, run_id):
    with pytest.raises(
        ValueError,
        match="32-character lowercase hexadecimal ID",
    ):
        RunIdentity(tmp_path, run_id, tmp_path / "run")


def test_run_identity_requires_a_run_path_inside_the_root(tmp_path):
    with pytest.raises(ValueError, match="inside its root"):
        RunIdentity(tmp_path / "root", "a" * 32, tmp_path / "outside")
    with pytest.raises(ValueError, match="below its root"):
        RunIdentity(tmp_path, "a" * 32, tmp_path)


def test_run_identity_does_not_collide_across_roots(tmp_path):
    left = publish_selection(tmp_path / "left", run_id="a" * 32)
    right = publish_selection(tmp_path / "right", run_id="a" * 32)

    assert RunIdentity.from_entry(left) != RunIdentity.from_entry(right)


def test_dependency_graph_collects_transitive_family(tmp_path):
    root = tmp_path / "results"
    parent = publish_selection(root, run_id="a" * 32)
    child = publish_figure(root, run_id="b" * 32, parent=parent)
    grandchild = publish_figure(root, run_id="c" * 32, parent=child)

    graph = RunDependencyGraph.from_roots((root,))
    selected = RunIdentity.from_entry(parent)

    assert graph.entry(selected) == parent
    assert graph.children_of(selected) == frozenset({RunIdentity.from_entry(child)})
    assert graph.family(selected) == frozenset(
        {
            RunIdentity.from_entry(parent),
            RunIdentity.from_entry(child),
            RunIdentity.from_entry(grandchild),
        }
    )


def test_source_edge_requires_matching_run_id_and_path(tmp_path):
    root = tmp_path / "results"
    parent = publish_selection(root, run_id="a" * 32)
    child = publish_figure(root, run_id="b" * 32, parent=parent)
    graph = RunDependencyGraph.from_roots((root,))

    assert graph.parent_of(RunIdentity.from_entry(child)) == RunIdentity.from_entry(
        parent
    )


def test_parent_run_id_creates_same_root_edge_without_source_metadata(tmp_path):
    root = tmp_path / "results"
    parent = publish_selection(root, run_id="a" * 32)
    child = _publish_run(
        root,
        run_id="b" * 32,
        parent_run_id=parent.run_id,
        run_kind="figure",
    )

    graph = RunDependencyGraph.from_roots((root,))

    assert graph.parent_of(RunIdentity.from_entry(child)) == RunIdentity.from_entry(
        parent
    )


def test_metadata_source_creates_edge_without_parent_run_id(tmp_path):
    root = tmp_path / "results"
    parent = publish_selection(root, run_id="a" * 32)
    child = _publish_run(
        root,
        run_id="b" * 32,
        run_kind="figure",
        source={"path": str(parent.run_dir), "run_id": parent.run_id},
    )

    graph = RunDependencyGraph.from_roots((root,))

    assert graph.parent_of(RunIdentity.from_entry(child)) == RunIdentity.from_entry(
        parent
    )


def test_cross_root_source_path_disambiguates_copied_run_id(tmp_path):
    source_root = tmp_path / "source"
    duplicate_root = tmp_path / "duplicate"
    child_root = tmp_path / "figures"
    source = publish_selection(source_root, run_id="a" * 32)
    duplicate = publish_selection(duplicate_root, run_id="a" * 32)
    child = _publish_run(
        child_root,
        run_id="b" * 32,
        run_kind="figure",
        source={"path": str(source.run_dir), "run_id": source.run_id},
    )

    graph = RunDependencyGraph.from_roots(
        (duplicate_root, child_root, source_root)
    )

    assert graph.parent_of(RunIdentity.from_entry(child)) == RunIdentity.from_entry(
        source
    )
    assert graph.parent_of(RunIdentity.from_entry(child)) != RunIdentity.from_entry(
        duplicate
    )


def test_graph_rejects_conflicting_resolved_provenance_parents(tmp_path):
    root = tmp_path / "results"
    declared_parent = publish_selection(root, run_id="a" * 32)
    source_parent = publish_selection(root, run_id="d" * 32)
    _publish_run(
        root,
        run_id="b" * 32,
        parent_run_id=declared_parent.run_id,
        run_kind="figure",
        source={
            "path": str(source_parent.run_dir),
            "run_id": source_parent.run_id,
        },
    )

    with pytest.raises(RunDependencyError, match="conflicting provenance parents"):
        RunDependencyGraph.from_roots((root,))


def test_pre_existing_orphan_source_allows_its_own_discovered_family(tmp_path):
    root = tmp_path / "results"
    absent_id = "d" * 32
    orphan = _publish_run(
        root,
        run_id="b" * 32,
        parent_run_id=absent_id,
        run_kind="figure",
        source={"path": str(root / "removed-run"), "run_id": absent_id},
    )
    child = publish_figure(root, run_id="c" * 32, parent=orphan)

    graph = RunDependencyGraph.from_roots((root,))
    orphan_identity = RunIdentity.from_entry(orphan)

    assert graph.diagnostics
    assert graph.family(orphan_identity) == frozenset(
        {orphan_identity, RunIdentity.from_entry(child)}
    )


@pytest.mark.parametrize("matching_component", ["path", "run_id"])
def test_unresolved_source_component_blocks_referenced_parent_family(
    tmp_path,
    matching_component,
):
    root = tmp_path / "results"
    parent = publish_selection(root, run_id="a" * 32)
    source = {
        "path": str(parent.run_dir if matching_component == "path" else root / "gone"),
        "run_id": parent.run_id if matching_component == "run_id" else "d" * 32,
    }
    orphan = _publish_run(
        root,
        run_id="b" * 32,
        run_kind="figure",
        source=source,
    )
    graph = RunDependencyGraph.from_roots((root,))

    with pytest.raises(RunDependencyError, match="unresolved provenance"):
        graph.family(RunIdentity.from_entry(parent))
    assert graph.family(RunIdentity.from_entry(orphan)) == frozenset(
        {RunIdentity.from_entry(orphan)}
    )


def test_graph_rejects_ambiguous_same_root_parent_run_id(tmp_path):
    root = tmp_path / "results"
    duplicate_id = "a" * 32
    publish_selection(
        root,
        run_id=duplicate_id,
        created_at="2026-07-20T10:00:00Z",
    )
    publish_selection(
        root,
        run_id=duplicate_id,
        created_at="2026-07-20T10:00:01Z",
    )
    _publish_run(
        root,
        run_id="b" * 32,
        parent_run_id=duplicate_id,
        run_kind="figure",
    )

    with pytest.raises(RunDependencyError, match="ambiguous"):
        RunDependencyGraph.from_roots((root,))


def test_graph_rejects_source_cycle(tmp_path):
    root = tmp_path / "results"
    service = RunService(root)
    with patch.object(
        run_service_module.uuid,
        "uuid4",
        return_value=UUID(hex="a" * 32),
    ):
        first = service.begin(
            "chicago",
            "default",
            created_at="2026-07-20T10:00:00Z",
        )
    with patch.object(
        run_service_module.uuid,
        "uuid4",
        return_value=UUID(hex="b" * 32),
    ):
        second = service.begin(
            "chicago",
            "default",
            created_at="2026-07-20T10:00:01Z",
        )
    for run in (first, second):
        (run.path / "figure.png").write_bytes(b"figure\n")
    first_path = service.publish(
        first,
        status="completed",
        artifacts=["figure.png"],
        metadata={
            "run_kind": "figure",
            "source": {"path": str(second.final_path), "run_id": second.run_id},
        },
    )
    service.publish(
        second,
        status="completed",
        artifacts=["figure.png"],
        metadata={
            "run_kind": "figure",
            "source": {"path": str(first_path), "run_id": first.run_id},
        },
    )

    with pytest.raises(RunDependencyError, match="cycle"):
        RunDependencyGraph.from_roots((root,))


def test_fingerprint_changes_when_immutable_manifest_snapshot_changes(tmp_path):
    root = tmp_path / "results"
    entry = publish_selection(root, run_id="a" * 32)
    selected = RunIdentity.from_entry(entry)
    original_graph = RunDependencyGraph.from_roots((root,))
    original = original_graph.fingerprint(
        selected,
        original_graph.family(selected),
    )

    manifest_path = entry.run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["label"] = "tampered"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    refreshed_graph = RunDependencyGraph.from_roots((root,))
    refreshed = refreshed_graph.fingerprint(
        selected,
        refreshed_graph.family(selected),
    )

    assert refreshed != original
    assert len(refreshed) == 64


def test_newly_published_child_changes_graph_fingerprint(tmp_path):
    root = tmp_path / "results"
    parent = publish_selection(root, run_id="a" * 32)
    selected = RunIdentity.from_entry(parent)
    original_graph = RunDependencyGraph.from_roots((root,))
    original = original_graph.fingerprint(
        selected,
        original_graph.family(selected),
    )

    publish_figure(root, run_id="b" * 32, parent=parent)
    refreshed_graph = RunDependencyGraph.from_roots((root,))
    refreshed = refreshed_graph.fingerprint(
        selected,
        refreshed_graph.family(selected),
    )

    assert refreshed != original


def test_fingerprint_is_root_order_independent_and_includes_root_universe(tmp_path):
    root = tmp_path / "results"
    other = tmp_path / "other"
    empty = tmp_path / "empty"
    parent = publish_selection(root, run_id="a" * 32)
    publish_selection(other, run_id="d" * 32)
    selected = RunIdentity.from_entry(parent)

    first_graph = RunDependencyGraph.from_roots((root, other))
    second_graph = RunDependencyGraph.from_roots((other, root))
    expanded_graph = RunDependencyGraph.from_roots((empty, other, root))
    first = first_graph.fingerprint(selected, first_graph.family(selected))
    second = second_graph.fingerprint(selected, second_graph.family(selected))
    expanded = expanded_graph.fingerprint(selected, expanded_graph.family(selected))

    assert first == second
    assert expanded != first


def _identity(tmp_path: Path, name: str = "run") -> RunIdentity:
    return RunIdentity(tmp_path, "a" * 32, tmp_path / name)


def test_usage_registry_allows_multiple_read_leases_and_reports_owners(tmp_path):
    identity = _identity(tmp_path)
    registry = RunUsageLeaseRegistry()

    first = registry.acquire((identity,), "Figures B")
    second = registry.acquire((identity,), "Figures A")

    assert first != second
    assert registry.conflicts((identity,)) == ("Figures A", "Figures B")


def test_usage_registry_release_is_idempotent(tmp_path):
    identity = _identity(tmp_path)
    registry = RunUsageLeaseRegistry()
    lease_id = registry.acquire((identity,), "Figures")

    registry.release(lease_id)
    registry.release(lease_id)

    assert registry.conflicts((identity,)) == ()


@pytest.mark.parametrize("owner", ["", "   ", 1, None])
def test_usage_registry_rejects_invalid_owner(tmp_path, owner):
    with pytest.raises(ValueError, match="identities and an owner"):
        RunUsageLeaseRegistry().acquire((_identity(tmp_path),), owner)


def test_usage_registry_rejects_empty_identities():
    with pytest.raises(ValueError, match="identities and an owner"):
        RunUsageLeaseRegistry().acquire((), "Figures")


def test_run_trash_errors_expose_stable_codes():
    assert RunDependencyError.code == "run.dependency_invalid"
    assert RunLeaseConflictError.code == "run.in_use"
