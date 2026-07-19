from __future__ import annotations

import json
import os
from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path, PurePosixPath
from unittest.mock import patch
from uuid import UUID

import pytest

import lte_scenario_toolkit.run_service as run_service_module
import lte_scenario_toolkit.run_trash as run_trash_module
from lte_scenario_toolkit.run_service import RunEntry, RunService
from lte_scenario_toolkit.run_trash import (
    RunDependencyError,
    RunDependencyGraph,
    RunEdge,
    RunIdentity,
    RunLeaseConflictError,
    RunUsageLeaseRegistry,
)

CREATED_AT = "2026-07-20T10:00:00Z"


def build_trash_plan(*args, **kwargs):
    return run_trash_module.build_trash_plan(*args, **kwargs)


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


def test_graph_rejects_source_path_and_run_id_identifying_different_entries(
    tmp_path,
):
    root = tmp_path / "results"
    path_parent = publish_selection(root, run_id="a" * 32)
    id_parent = publish_selection(root, run_id="d" * 32)
    _publish_run(
        root,
        run_id="b" * 32,
        run_kind="figure",
        source={"path": str(path_parent.run_dir), "run_id": id_parent.run_id},
    )

    with pytest.raises(RunDependencyError, match="ambiguous"):
        RunDependencyGraph.from_roots((root,))


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


def test_family_fails_closed_when_discovery_diagnostic_may_hide_dependent(tmp_path):
    root = tmp_path / "results"
    parent = publish_selection(root, run_id="a" * 32)
    child = publish_figure(root, run_id="b" * 32, parent=parent)
    (child.run_dir / "figure.png").unlink()

    graph = RunDependencyGraph.from_roots((root,))

    assert any(
        Path(item["path"]) == child.run_dir / "run.json"
        for item in graph.diagnostics
    )
    with pytest.raises(RunDependencyError, match="discovery diagnostic"):
        graph.family(RunIdentity.from_entry(parent))


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


def test_source_path_runtime_error_is_retained_as_unresolved_diagnostic(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "results"
    source_path = tmp_path / "symlink-loop"
    child = _publish_run(
        root,
        run_id="b" * 32,
        run_kind="figure",
        source={"path": str(source_path), "run_id": "d" * 32},
    )
    original_resolve = Path.resolve

    def resolve(candidate, strict=False):
        if candidate == source_path:
            raise RuntimeError("symlink loop")
        return original_resolve(candidate, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve)

    graph = RunDependencyGraph.from_roots((root,))

    assert any(item.get("source") == "metadata.source" for item in graph.diagnostics)
    assert graph.family(RunIdentity.from_entry(child)) == frozenset(
        {RunIdentity.from_entry(child)}
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


@pytest.mark.skipif(os.name != "nt", reason="Windows path equivalence is required")
def test_windows_equivalent_missing_roots_have_stable_universe_and_fingerprint(
    tmp_path,
):
    live_root = tmp_path / "results"
    parent = publish_selection(live_root, run_id="a" * 32)
    upper_missing = tmp_path / "MISSING"
    lower_missing = tmp_path / "missing"
    first_graph = RunDependencyGraph.from_roots(
        (live_root, upper_missing, lower_missing)
    )
    second_graph = RunDependencyGraph.from_roots(
        (lower_missing, upper_missing, live_root)
    )
    selected = RunIdentity.from_entry(parent)

    assert tuple(map(str, first_graph.roots)) == tuple(map(str, second_graph.roots))
    assert first_graph.fingerprint(
        selected,
        first_graph.family(selected),
    ) == second_graph.fingerprint(selected, second_graph.family(selected))


def _canonical_manifest_digest(entry: RunEntry) -> str:
    document = json.loads((entry.run_dir / "run.json").read_text(encoding="utf-8"))
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _unique_regular_file_size(root: Path) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        details = path.stat()
        key = (details.st_dev, details.st_ino)
        if key in seen:
            continue
        seen.add(key)
        total += details.st_size
    return total


def test_build_trash_plan_populates_immutable_members_and_exact_paths(tmp_path):
    root = tmp_path / "results"
    parent = publish_selection(root, run_id="a" * 32)
    child = publish_figure(
        root,
        run_id="b" * 32,
        parent=parent,
        created_at="2026-07-20T10:00:01Z",
    )
    (child.run_dir / "nested").mkdir()
    (child.run_dir / "nested" / "notes.txt").write_bytes(b"notes\n")
    graph = RunDependencyGraph.from_roots((root,))
    selected = RunIdentity.from_entry(parent)
    family = graph.family(selected)

    with patch.object(
        run_trash_module,
        "uuid4",
        return_value=UUID(hex="f" * 32),
    ):
        plan = build_trash_plan(graph, selected, family)

    assert isinstance(plan, run_trash_module.TrashPlan)
    assert plan.transaction_id == "f" * 32
    assert plan.selected == selected
    assert [member.identity.run_id for member in plan.members] == [
        child.run_id,
        parent.run_id,
    ]
    assert all(
        isinstance(member, run_trash_module.TrashMember) for member in plan.members
    )
    child_member = plan.members[0]
    assert child_member.original_relative_path == PurePosixPath(
        "chicago",
        "default",
        child.run_dir.name,
    )
    assert child_member.trash_relative_path == PurePosixPath(
        "runs",
        "chicago",
        "default",
        child.run_dir.name,
    )
    assert child_member.scenario_id == "chicago"
    assert child_member.profile_id == "default"
    assert child_member.created_at == "2026-07-20T10:00:01Z"
    assert child_member.run_kind == "figure"
    assert child_member.status == "completed"
    assert child_member.parent_run_id == parent.run_id
    assert child_member.artifact_count == 1
    assert child_member.manifest_digest == _canonical_manifest_digest(child)
    assert child_member.size_bytes == _unique_regular_file_size(child.run_dir)
    assert plan.roots == (parent.root,)
    assert plan.fingerprint == graph.fingerprint(selected, family)
    assert plan.total_size_bytes == sum(
        member.size_bytes for member in plan.members
    )
    assert [edge.source for edge in plan.edges] == [
        "metadata.source",
        "parent_run_id",
    ]
    assert all(
        edge.parent in family and edge.child in family for edge in plan.edges
    )
    with pytest.raises(FrozenInstanceError):
        plan.transaction_id = "e" * 32
    with pytest.raises(FrozenInstanceError):
        child_member.size_bytes = 0


def test_build_trash_plan_orders_descendants_first_deterministically(tmp_path):
    root = tmp_path / "results"
    parent = publish_selection(root, run_id="a" * 32)
    later_sibling = publish_figure(root, run_id="c" * 32, parent=parent)
    earlier_sibling = publish_figure(root, run_id="b" * 32, parent=parent)
    grandchild = publish_figure(root, run_id="d" * 32, parent=later_sibling)
    graph = RunDependencyGraph.from_roots((root,))
    selected = RunIdentity.from_entry(parent)
    family = graph.family(selected)

    first = build_trash_plan(graph, selected, tuple(reversed(tuple(family))))
    second = build_trash_plan(graph, selected, family)

    expected = [
        grandchild.run_id,
        earlier_sibling.run_id,
        later_sibling.run_id,
        parent.run_id,
    ]
    assert [member.identity.run_id for member in first.members] == expected
    assert [member.identity.run_id for member in second.members] == expected
    assert first.edges == second.edges


def test_build_trash_plan_records_incoming_edges_for_selected_leaf(tmp_path):
    root = tmp_path / "results"
    parent = publish_selection(root, run_id="a" * 32)
    child = publish_figure(root, run_id="b" * 32, parent=parent)
    graph = RunDependencyGraph.from_roots((root,))
    selected = RunIdentity.from_entry(child)

    plan = build_trash_plan(graph, selected, graph.family(selected))

    assert tuple((edge.parent.run_id, edge.child.run_id, edge.source) for edge in plan.edges) == (
        (parent.run_id, child.run_id, "metadata.source"),
        (parent.run_id, child.run_id, "parent_run_id"),
    )


def test_build_trash_plan_manifest_digest_changes_with_manifest_snapshot(tmp_path):
    root = tmp_path / "results"
    entry = publish_selection(root, run_id="a" * 32)
    selected = RunIdentity.from_entry(entry)
    first_graph = RunDependencyGraph.from_roots((root,))
    first = build_trash_plan(
        first_graph,
        selected,
        first_graph.family(selected),
    )
    manifest_path = entry.run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["label"] = "changed"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    second_graph = RunDependencyGraph.from_roots((root,))
    second = build_trash_plan(
        second_graph,
        selected,
        second_graph.family(selected),
    )

    assert first.members[0].manifest_digest != second.members[0].manifest_digest
    assert first.fingerprint != second.fingerprint


@pytest.mark.parametrize(
    "run_kind",
    [None, "unknown", ["figure"], {"kind": "figure"}],
)
def test_build_trash_plan_uses_safe_selection_default_for_run_kind(
    tmp_path,
    run_kind,
):
    root = tmp_path / "results"
    entry = publish_selection(root, run_id="a" * 32)
    manifest_path = entry.run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["run_kind"] = run_kind
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    graph = RunDependencyGraph.from_roots((root,))
    selected = RunIdentity.from_entry(entry)

    plan = build_trash_plan(graph, selected, graph.family(selected))

    assert plan.members[0].run_kind == "selection"


def test_build_trash_plan_size_counts_hard_link_content_once(tmp_path):
    root = tmp_path / "results"
    entry = publish_selection(root, run_id="a" * 32)
    artifact = entry.run_dir / "selection.csv"
    alias = entry.run_dir / "selection-copy.csv"
    try:
        os.link(artifact, alias)
    except OSError as exc:
        pytest.skip(f"hard-link creation is unavailable: {exc}")
    graph = RunDependencyGraph.from_roots((root,))
    selected = RunIdentity.from_entry(entry)

    plan = build_trash_plan(graph, selected, graph.family(selected))

    expected = _unique_regular_file_size(entry.run_dir)
    naive = sum(path.stat().st_size for path in entry.run_dir.rglob("*") if path.is_file())
    assert naive > expected
    assert plan.members[0].size_bytes == expected
    assert plan.total_size_bytes == expected


def test_build_trash_plan_rejects_nested_redirected_content(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "results"
    entry = publish_selection(root, run_id="a" * 32)
    nested = entry.run_dir / "nested"
    nested.mkdir()
    (nested / "data.bin").write_bytes(b"data")
    graph = RunDependencyGraph.from_roots((root,))
    selected = RunIdentity.from_entry(entry)
    original = run_trash_module._is_redirected_path
    monkeypatch.setattr(
        run_trash_module,
        "_is_redirected_path",
        lambda path: Path(path) == nested or original(Path(path)),
    )

    with pytest.raises(ValueError, match="redirected|size|contents"):
        build_trash_plan(graph, selected, graph.family(selected))


def test_build_trash_plan_size_fails_closed_on_oserror(tmp_path, monkeypatch):
    root = tmp_path / "results"
    entry = publish_selection(root, run_id="a" * 32)
    nested = entry.run_dir / "nested"
    nested.mkdir()
    graph = RunDependencyGraph.from_roots((root,))
    selected = RunIdentity.from_entry(entry)
    original = run_trash_module._is_redirected_path

    def failing_redirect_check(path):
        if Path(path) == nested:
            raise OSError("simulated filesystem failure")
        return original(Path(path))

    monkeypatch.setattr(
        run_trash_module,
        "_is_redirected_path",
        failing_redirect_check,
    )

    with pytest.raises(ValueError, match="size|safely|contents"):
        build_trash_plan(graph, selected, graph.family(selected))


def test_build_trash_plan_rejects_missing_extraneous_and_duplicate_family_members(
    tmp_path,
):
    root = tmp_path / "results"
    parent = publish_selection(root, run_id="a" * 32)
    child = publish_figure(root, run_id="b" * 32, parent=parent)
    other = publish_selection(root, run_id="c" * 32)
    graph = RunDependencyGraph.from_roots((root,))
    selected = RunIdentity.from_entry(parent)
    child_identity = RunIdentity.from_entry(child)
    other_identity = RunIdentity.from_entry(other)

    invalid_member_sets = [
        (selected,),
        (selected, child_identity, other_identity),
        (selected, child_identity, child_identity),
        (selected, child_identity, _identity(tmp_path, "not-in-graph")),
    ]
    for supplied in invalid_member_sets:
        with pytest.raises(ValueError, match="family|member|graph|duplicate"):
            build_trash_plan(graph, selected, supplied)


def test_build_trash_plan_rejects_selected_identity_outside_graph(tmp_path):
    root = tmp_path / "results"
    entry = publish_selection(root, run_id="a" * 32)
    graph = RunDependencyGraph.from_roots((root,))
    member = RunIdentity.from_entry(entry)
    selected = RunIdentity(root, "f" * 32, root / "missing")

    with pytest.raises(ValueError, match="selected|graph|family"):
        build_trash_plan(graph, selected, (member,))


def test_build_trash_plan_records_cross_root_affected_roots(tmp_path):
    source_root = tmp_path / "source-results"
    child_root = tmp_path / "figure-results"
    parent = publish_selection(source_root, run_id="a" * 32)
    child = _publish_run(
        child_root,
        run_id="b" * 32,
        run_kind="figure",
        source={"path": str(parent.run_dir), "run_id": parent.run_id},
    )
    graph = RunDependencyGraph.from_roots((child_root, source_root))
    selected = RunIdentity.from_entry(parent)

    plan = build_trash_plan(graph, selected, graph.family(selected))

    assert plan.roots == graph.roots
    assert {member.identity.root for member in plan.members} == {
        parent.root,
        child.root,
    }
    assert all(
        edge.parent in {member.identity for member in plan.members}
        and edge.child in {member.identity for member in plan.members}
        for edge in plan.edges
    )


def _identity(tmp_path: Path, name: str = "run") -> RunIdentity:
    return RunIdentity(tmp_path, "a" * 32, tmp_path / name)


def test_run_edge_rejects_unknown_source(tmp_path):
    identity = _identity(tmp_path)

    with pytest.raises(ValueError, match="source"):
        RunEdge(parent=identity, child=identity, source="bogus")


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
