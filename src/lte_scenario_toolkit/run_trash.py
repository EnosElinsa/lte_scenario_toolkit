"""Dependency identities, graph fingerprints, and process-local run leases."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from .io import atomic_write_json
from .run_service import (
    RunEntry,
    RunService,
    _assert_unredirected_chain,
    _is_redirected_path,
)


@dataclass(frozen=True, slots=True)
class RunIdentity:
    root: Path
    run_id: str
    expected_path: Path

    def __post_init__(self) -> None:
        if type(self.run_id) is not str or re.fullmatch(r"[0-9a-f]{32}", self.run_id) is None:
            raise ValueError("run identity requires a 32-character lowercase hexadecimal ID")
        root = Path(self.root).expanduser().resolve(strict=False)
        expected = Path(self.expected_path).expanduser().resolve(strict=False)
        try:
            expected.relative_to(root)
        except ValueError as exc:
            raise ValueError("run identity path must remain inside its root") from exc
        if expected == root:
            raise ValueError("run identity path must name a run below its root")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "expected_path", expected)

    @classmethod
    def from_entry(cls, entry: RunEntry) -> RunIdentity:
        return cls(entry.root, entry.run_id, entry.run_dir)


@dataclass(frozen=True, slots=True)
class RunEdge:
    parent: RunIdentity
    child: RunIdentity
    source: Literal["parent_run_id", "metadata.source"]

    def __post_init__(self) -> None:
        if self.source not in ("parent_run_id", "metadata.source"):
            raise ValueError("run dependency edge source must be parent_run_id or metadata.source")


@dataclass(frozen=True, slots=True)
class TrashMember:
    identity: RunIdentity
    original_relative_path: PurePosixPath
    trash_relative_path: PurePosixPath
    scenario_id: str
    profile_id: str
    created_at: str
    run_kind: str
    status: str
    parent_run_id: str | None
    artifact_count: int
    manifest_digest: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class TrashPlan:
    transaction_id: str
    selected: RunIdentity
    members: tuple[TrashMember, ...]
    edges: tuple[RunEdge, ...]
    roots: tuple[Path, ...]
    fingerprint: str
    total_size_bytes: int


class TrashState(str, Enum):
    MOVING = "moving"
    TRASHED = "trashed"
    RESTORING = "restoring"
    PURGING = "purging"
    RECOVERY_REQUIRED = "recovery_required"
    PURGE_FAILED = "purge_failed"


class TrashTransactionError(RuntimeError):
    code = "trash.transaction_failed"


class TrashPlanStaleError(TrashTransactionError):
    code = "trash.plan_stale"


@dataclass(frozen=True, slots=True)
class TrashTransaction:
    transaction_id: str
    selected: RunIdentity
    members: tuple[TrashMember, ...]
    roots: tuple[Path, ...]
    state: TrashState
    deleted_at: str
    completed_move_ids: tuple[str, ...]
    completed_restore_ids: tuple[str, ...]
    total_size_bytes: int
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TrashDiagnostic:
    root_digest: str
    transaction_id: str | None
    code: str
    error: str


@dataclass(frozen=True, slots=True)
class TrashDiscovery:
    transactions: tuple[TrashTransaction, ...]
    diagnostics: tuple[TrashDiagnostic, ...]


class RunDependencyError(ValueError):
    code = "run.dependency_invalid"


class RunLeaseConflictError(RuntimeError):
    code = "run.in_use"


@dataclass(frozen=True, slots=True)
class _UnresolvedProvenance:
    child: RunIdentity
    source: Literal["parent_run_id", "metadata.source"]
    run_id: str | None
    expected_path: Path | None
    reason: str

    def could_refer_to(self, identity: RunIdentity) -> bool:
        if self.source == "parent_run_id":
            return self.child.root == identity.root and self.run_id == identity.run_id
        return self.run_id == identity.run_id or self.expected_path == identity.expected_path


def _canonical_path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))


def _identity_key(identity: RunIdentity) -> tuple[str, str, str]:
    return (
        _canonical_path_key(identity.root),
        identity.run_id,
        _canonical_path_key(identity.expected_path),
    )


def _identity_document(identity: RunIdentity) -> dict[str, str]:
    return {
        "root": str(identity.root),
        "run_id": identity.run_id,
        "expected_path": str(identity.expected_path),
    }


def _json_document(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_document(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_document(item) for item in value]
    return value


def _normalise_source_path(value: object) -> Path | None:
    if type(value) is not str or not value.strip():
        return None
    try:
        return Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


class RunDependencyGraph:
    """Fresh cross-root dependency graph built from validated live run entries."""

    def __init__(
        self,
        *,
        roots: tuple[Path, ...],
        entries: dict[RunIdentity, RunEntry],
        edges: frozenset[RunEdge],
        unresolved: tuple[_UnresolvedProvenance, ...],
        discovery_diagnostics: tuple[dict[str, Any], ...],
    ) -> None:
        self._roots = roots
        self._entries = entries
        self._edges = edges
        self._unresolved = unresolved
        self._discovery_diagnostics = discovery_diagnostics

    @classmethod
    def from_roots(cls, roots: Iterable[str | Path]) -> RunDependencyGraph:
        services: dict[str, RunService] = {}
        for value in roots:
            service = RunService(value)
            services.setdefault(_canonical_path_key(service.output_root), service)
        root_keys = tuple(sorted(services))
        canonical_roots = tuple(Path(key) for key in root_keys)

        entries: dict[RunIdentity, RunEntry] = {}
        discovery_diagnostics: list[dict[str, Any]] = []
        for root_key in root_keys:
            discovered = services[root_key].discover_entries()
            entries.update((RunIdentity.from_entry(entry), entry) for entry in discovered.entries)
            discovery_diagnostics.extend(
                {"root": root_key, **dict(item)} for item in discovered.diagnostics
            )

        by_root_id: dict[tuple[Path, str], list[RunIdentity]] = {}
        by_path_id: dict[tuple[Path, str], list[RunIdentity]] = {}
        by_path: dict[Path, list[RunIdentity]] = {}
        by_id: dict[str, list[RunIdentity]] = {}
        for identity in entries:
            by_root_id.setdefault((identity.root, identity.run_id), []).append(identity)
            by_path_id.setdefault((identity.expected_path, identity.run_id), []).append(identity)
            by_path.setdefault(identity.expected_path, []).append(identity)
            by_id.setdefault(identity.run_id, []).append(identity)

        edges: set[RunEdge] = set()
        unresolved: list[_UnresolvedProvenance] = []
        for child, entry in entries.items():
            resolved: dict[Literal["parent_run_id", "metadata.source"], RunIdentity] = {}
            parent_run_id = entry.record.get("parent_run_id")
            if type(parent_run_id) is str:
                candidates = by_root_id.get((child.root, parent_run_id), ())
                if len(candidates) > 1:
                    raise RunDependencyError(
                        "parent_run_id identifies an ambiguous same-root parent"
                    )
                if candidates:
                    resolved["parent_run_id"] = candidates[0]
                else:
                    unresolved.append(
                        _UnresolvedProvenance(
                            child=child,
                            source="parent_run_id",
                            run_id=parent_run_id,
                            expected_path=None,
                            reason="same-root parent was not discovered",
                        )
                    )

            metadata = entry.record.get("metadata")
            source_present = isinstance(metadata, Mapping) and "source" in metadata
            if source_present:
                source = metadata["source"]
                source_mapping = source if isinstance(source, Mapping) else {}
                source_run_id_value = source_mapping.get("run_id")
                source_run_id = source_run_id_value if type(source_run_id_value) is str else None
                source_path = _normalise_source_path(source_mapping.get("path"))
                candidates = (
                    by_path_id.get((source_path, source_run_id), ())
                    if source_path is not None and source_run_id is not None
                    else ()
                )
                if len(candidates) > 1:
                    raise RunDependencyError("metadata.source identifies an ambiguous parent")
                if candidates:
                    resolved["metadata.source"] = candidates[0]
                else:
                    path_candidates = (
                        by_path.get(source_path, ()) if source_path is not None else ()
                    )
                    id_candidates = (
                        by_id.get(source_run_id, ()) if source_run_id is not None else ()
                    )
                    if (
                        len(path_candidates) > 1
                        or len(id_candidates) > 1
                        or (path_candidates and id_candidates)
                    ):
                        raise RunDependencyError("metadata.source path or run ID is ambiguous")
                    unresolved.append(
                        _UnresolvedProvenance(
                            child=child,
                            source="metadata.source",
                            run_id=source_run_id,
                            expected_path=source_path,
                            reason="source path and run ID did not match one live run",
                        )
                    )

            if len(set(resolved.values())) > 1:
                raise RunDependencyError("run has conflicting provenance parents")
            edges.update(
                RunEdge(parent=parent, child=child, source=source)
                for source, parent in resolved.items()
            )

        frozen_edges = frozenset(edges)
        cls._assert_acyclic(entries, frozen_edges)
        unresolved.sort(
            key=lambda item: (
                _identity_key(item.child),
                item.source,
                item.run_id or "",
                ("" if item.expected_path is None else _canonical_path_key(item.expected_path)),
            )
        )
        discovery_diagnostics.sort(
            key=lambda item: (str(item.get("root", "")), str(item.get("path", "")))
        )
        return cls(
            roots=canonical_roots,
            entries=entries,
            edges=frozen_edges,
            unresolved=tuple(unresolved),
            discovery_diagnostics=tuple(discovery_diagnostics),
        )

    @staticmethod
    def _assert_acyclic(
        entries: Mapping[RunIdentity, RunEntry],
        edges: frozenset[RunEdge],
    ) -> None:
        children: dict[RunIdentity, set[RunIdentity]] = {identity: set() for identity in entries}
        for edge in edges:
            children[edge.parent].add(edge.child)

        state: dict[RunIdentity, int] = {}
        for start in entries:
            if state.get(start, 0) != 0:
                continue
            state[start] = 1
            stack: list[tuple[RunIdentity, Any]] = [(start, iter(children[start]))]
            while stack:
                current, descendants = stack[-1]
                try:
                    child = next(descendants)
                except StopIteration:
                    state[current] = 2
                    stack.pop()
                    continue
                child_state = state.get(child, 0)
                if child_state == 1:
                    raise RunDependencyError("run dependency graph contains a cycle")
                if child_state == 0:
                    state[child] = 1
                    stack.append((child, iter(children[child])))

    @property
    def roots(self) -> tuple[Path, ...]:
        return self._roots

    @property
    def edges(self) -> frozenset[RunEdge]:
        return self._edges

    @property
    def diagnostics(self) -> tuple[dict[str, Any], ...]:
        unresolved = tuple(
            {
                "path": str(item.child.expected_path / "run.json"),
                "error": f"unresolved {item.source}: {item.reason}",
                "source": item.source,
                "run_id": item.run_id,
                "source_path": (None if item.expected_path is None else str(item.expected_path)),
            }
            for item in self._unresolved
        )
        return tuple(dict(item) for item in self._discovery_diagnostics) + unresolved

    def entry(self, identity: RunIdentity) -> RunEntry:
        return self._entries[identity]

    def children_of(self, identity: RunIdentity) -> frozenset[RunIdentity]:
        return frozenset(edge.child for edge in self._edges if edge.parent == identity)

    def parent_of(self, identity: RunIdentity) -> RunIdentity | None:
        values = {edge.parent for edge in self._edges if edge.child == identity}
        if len(values) > 1:
            raise RunDependencyError("run has conflicting provenance parents")
        return next(iter(values), None)

    def family(self, selected: RunIdentity) -> frozenset[RunIdentity]:
        if self._discovery_diagnostics:
            raise RunDependencyError(
                "run family cannot be evaluated while discovery diagnostics exist"
            )
        pending = [selected]
        found: set[RunIdentity] = set()
        while pending:
            current = pending.pop()
            if current in found:
                continue
            found.add(current)
            pending.extend(self.children_of(current))

        for diagnostic in self._unresolved:
            if any(diagnostic.could_refer_to(identity) for identity in found):
                raise RunDependencyError(
                    "run family has unresolved provenance that could name a member"
                )
        return frozenset(found)

    def fingerprint(
        self,
        selected: RunIdentity,
        members: Iterable[RunIdentity],
    ) -> str:
        family = frozenset(members)
        if selected not in family:
            raise ValueError("run fingerprint members must include the selected run")
        family_identities = sorted(family, key=_identity_key)
        relevant_edges = sorted(
            (edge for edge in self._edges if edge.parent in family or edge.child in family),
            key=lambda edge: (
                _identity_key(edge.parent),
                _identity_key(edge.child),
                edge.source,
            ),
        )
        payload = {
            "selected": _identity_document(selected),
            "members": [_identity_document(identity) for identity in family_identities],
            "edges": [
                {
                    "parent": _identity_document(edge.parent),
                    "child": _identity_document(edge.child),
                    "source": edge.source,
                }
                for edge in relevant_edges
            ],
            "manifests": [
                {
                    "identity": _identity_document(identity),
                    "record": _json_document(self._entries[identity].record),
                }
                for identity in family_identities
            ],
            "roots": [_canonical_path_key(root) for root in self._roots],
        }
        document = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(document).hexdigest()


def _manifest_digest(record: Mapping[str, Any]) -> str:
    document = json.dumps(
        _json_document(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(document).hexdigest()


def _filesystem_identity(path: Path, details: os.stat_result) -> tuple[Any, ...]:
    if details.st_ino:
        return (
            "inode",
            stat.S_IFMT(details.st_mode),
            details.st_dev,
            details.st_ino,
        )
    return ("path", _canonical_path_key(path))


def _validated_tree_size(
    root: Path,
    seen: set[tuple[Any, ...]],
) -> int:
    try:
        candidate = _assert_unredirected_chain(
            root,
            description="run contents size root",
        )
        if candidate.resolve(strict=True) != candidate:
            raise ValueError("run contents size root must not be redirected")
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("run contents could not be safely sized") from exc

    total = 0
    pending = [candidate]
    while pending:
        current = pending.pop()
        try:
            if _is_redirected_path(current):
                raise ValueError("run contents contain a redirected path")
            details = current.lstat()
            identity = _filesystem_identity(current, details)
            if identity in seen:
                continue
            seen.add(identity)
            if stat.S_ISDIR(details.st_mode):
                children = sorted(
                    current.iterdir(),
                    key=lambda path: _canonical_path_key(path),
                    reverse=True,
                )
                pending.extend(children)
            elif stat.S_ISREG(details.st_mode):
                total += details.st_size
            else:
                raise ValueError("run contents contain a non-regular filesystem entry")
        except ValueError:
            raise
        except (OSError, RuntimeError) as exc:
            raise ValueError("run contents could not be safely sized") from exc
    return total


def _family_depths(
    graph: RunDependencyGraph,
    selected: RunIdentity,
    family: frozenset[RunIdentity],
) -> dict[RunIdentity, int]:
    depths = {selected: 0}
    pending = [selected]
    while pending:
        current = pending.pop()
        next_depth = depths[current] + 1
        for child in graph.children_of(current):
            if child not in family or next_depth <= depths.get(child, -1):
                continue
            depths[child] = next_depth
            pending.append(child)
    if set(depths) != set(family):
        raise ValueError("trash plan family is not reachable from the selected run")
    return depths


def _member_sort_key(
    identity: RunIdentity,
    depths: Mapping[RunIdentity, int],
) -> tuple[int, str, str, str]:
    return (
        -depths[identity],
        identity.run_id,
        _canonical_path_key(identity.expected_path),
        _canonical_path_key(identity.root),
    )


def _validate_plan_entry(identity: RunIdentity, entry: RunEntry) -> PurePosixPath:
    if RunIdentity.from_entry(entry) != identity:
        raise ValueError("trash plan member identity does not match its graph entry")
    try:
        relative = entry.run_dir.relative_to(entry.root)
    except ValueError as exc:
        raise ValueError("trash plan member path is outside its root") from exc
    scenario_id = entry.record["scenario_id"]
    profile_id = entry.record["profile_id"]
    if (
        len(relative.parts) != 3
        or relative.parts[0] != scenario_id
        or relative.parts[1] != profile_id
        or relative.parts[2] != entry.run_dir.name
        or ".." in relative.parts
    ):
        raise ValueError("trash plan member path does not have the live run shape")
    return PurePosixPath(*relative.parts)


def build_trash_plan(
    graph: RunDependencyGraph,
    selected: RunIdentity,
    members: Iterable[RunIdentity],
) -> TrashPlan:
    if type(graph) is not RunDependencyGraph:
        raise ValueError("trash plan requires a RunDependencyGraph")
    if type(selected) is not RunIdentity:
        raise ValueError("trash plan selected run must be a RunIdentity")
    try:
        supplied = tuple(members)
    except TypeError as exc:
        raise ValueError("trash plan members must be an iterable of identities") from exc
    if any(type(identity) is not RunIdentity for identity in supplied):
        raise ValueError("trash plan members must be RunIdentity values")
    supplied_family = frozenset(supplied)
    if len(supplied_family) != len(supplied):
        raise ValueError("trash plan members must not contain duplicates")

    try:
        graph.entry(selected)
    except KeyError as exc:
        raise ValueError("trash plan selected run is not present in the graph") from exc
    for identity in supplied:
        try:
            graph.entry(identity)
        except KeyError as exc:
            raise ValueError("trash plan member is not present in the graph") from exc
    expected_family = graph.family(selected)
    if supplied_family != expected_family:
        raise ValueError("trash plan members must equal the selected run family")

    depths = _family_depths(graph, selected, expected_family)
    ordered_identities = tuple(
        sorted(
            expected_family,
            key=lambda identity: _member_sort_key(identity, depths),
        )
    )
    relevant_edges = tuple(
        sorted(
            (
                edge
                for edge in graph.edges
                if edge.parent in expected_family or edge.child in expected_family
            ),
            key=lambda edge: (
                _identity_key(edge.parent),
                _identity_key(edge.child),
                edge.source,
            ),
        )
    )

    seen_filesystem_entries: set[tuple[Any, ...]] = set()
    planned_members: list[TrashMember] = []
    affected_roots: dict[str, Path] = {}
    for identity in ordered_identities:
        entry = graph.entry(identity)
        original_relative_path = _validate_plan_entry(identity, entry)
        record = entry.record
        scenario_id = str(record["scenario_id"])
        profile_id = str(record["profile_id"])
        metadata = record["metadata"]
        run_kind_value = metadata.get("run_kind")
        run_kind = (
            run_kind_value
            if type(run_kind_value) is str and run_kind_value in {"figure", "selection"}
            else "selection"
        )
        artifacts = record["artifacts"]
        member_size = _validated_tree_size(
            entry.run_dir,
            seen_filesystem_entries,
        )
        planned_members.append(
            TrashMember(
                identity=identity,
                original_relative_path=original_relative_path,
                trash_relative_path=PurePosixPath(
                    "runs",
                    scenario_id,
                    profile_id,
                    entry.run_dir.name,
                ),
                scenario_id=scenario_id,
                profile_id=profile_id,
                created_at=str(record["created_at"]),
                run_kind=run_kind,
                status=str(record["status"]),
                parent_run_id=record["parent_run_id"],
                artifact_count=len(artifacts),
                manifest_digest=_manifest_digest(record),
                size_bytes=member_size,
            )
        )
        affected_roots.setdefault(_canonical_path_key(identity.root), identity.root)

    frozen_members = tuple(planned_members)
    return TrashPlan(
        transaction_id=uuid4().hex,
        selected=selected,
        members=frozen_members,
        edges=relevant_edges,
        roots=tuple(affected_roots[key] for key in sorted(affected_roots)),
        fingerprint=graph.fingerprint(selected, expected_family),
        total_size_bytes=sum(member.size_bytes for member in frozen_members),
    )


_TRANSACTION_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
_ROOT_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_MANIFEST_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_DURABLE_ERROR_PATTERN = re.compile(r"[a-z][a-z0-9_]*:[A-Za-z_][A-Za-z0-9_]*")
_PORTION_FIELDS = frozenset(
    {
        "schema_version",
        "transaction_id",
        "deleted_at",
        "state",
        "selected",
        "expected_roots",
        "members",
        "edges",
        "completed_move_ids",
        "completed_restore_ids",
        "total_size_bytes",
        "errors",
    }
)
_IDENTITY_FIELDS = frozenset({"root_digest", "run_id", "relative_path"})
_MEMBER_FIELDS = frozenset(
    {
        "identity",
        "original_relative_path",
        "trash_relative_path",
        "scenario_id",
        "profile_id",
        "created_at",
        "run_kind",
        "status",
        "parent_run_id",
        "artifact_count",
        "manifest_digest",
        "size_bytes",
    }
)
_EDGE_FIELDS = frozenset({"parent", "child", "source"})
_RAW_JOURNAL_STATES = frozenset(
    {
        TrashState.MOVING,
        TrashState.TRASHED,
        TrashState.RESTORING,
        TrashState.PURGING,
        TrashState.PURGE_FAILED,
    }
)


@dataclass(frozen=True, slots=True)
class _TrashPortion:
    root: Path
    transaction_path: Path
    transaction_id: str
    deleted_at: str
    state: TrashState
    selected: RunIdentity
    expected_root_digests: tuple[str, ...]
    members: tuple[TrashMember, ...]
    edges: tuple[RunEdge, ...]
    completed_move_ids: tuple[str, ...]
    completed_restore_ids: tuple[str, ...]
    total_size_bytes: int
    errors: tuple[str, ...]

    def immutable_contract(self) -> tuple[Any, ...]:
        return (
            self.transaction_id,
            self.deleted_at,
            self.selected,
            self.expected_root_digests,
            self.edges,
            self.total_size_bytes,
        )


def _root_digest(root: Path) -> str:
    return sha256(_canonical_path_key(root).encode("utf-8")).hexdigest()


def _identity_relative_path(identity: RunIdentity) -> PurePosixPath:
    try:
        relative = identity.expected_path.relative_to(identity.root)
    except ValueError as exc:
        raise ValueError("run identity path is outside its root") from exc
    if not relative.parts or relative == Path(".") or ".." in relative.parts:
        raise ValueError("run identity relative path is invalid")
    return PurePosixPath(*relative.parts)


def _serialise_identity(identity: RunIdentity) -> dict[str, str]:
    return {
        "root_digest": _root_digest(identity.root),
        "run_id": identity.run_id,
        "relative_path": _identity_relative_path(identity).as_posix(),
    }


def _serialise_member(member: TrashMember) -> dict[str, Any]:
    return {
        "identity": _serialise_identity(member.identity),
        "original_relative_path": member.original_relative_path.as_posix(),
        "trash_relative_path": member.trash_relative_path.as_posix(),
        "scenario_id": member.scenario_id,
        "profile_id": member.profile_id,
        "created_at": member.created_at,
        "run_kind": member.run_kind,
        "status": member.status,
        "parent_run_id": member.parent_run_id,
        "artifact_count": member.artifact_count,
        "manifest_digest": member.manifest_digest,
        "size_bytes": member.size_bytes,
    }


def _serialise_edge(edge: RunEdge) -> dict[str, Any]:
    return {
        "parent": _serialise_identity(edge.parent),
        "child": _serialise_identity(edge.child),
        "source": edge.source,
    }


def _require_exact_mapping(
    value: object,
    fields: frozenset[str],
    *,
    description: str,
) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{description} fields do not match schema version 1")
    return value


def _validated_relative_path(value: object, *, description: str) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"{description} must be a POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or path == PurePosixPath(".")
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(f"{description} must be a canonical POSIX relative path")
    return path


def _validated_timestamp(value: object, *, description: str) -> str:
    if type(value) is not str or not value or "T" not in value:
        raise ValueError(f"{description} must be a timezone-aware timestamp")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{description} must be a valid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{description} must include a timezone")
    canonical = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if canonical != value:
        raise ValueError(f"{description} must use canonical UTC form")
    return value


def _deserialise_identity(
    value: object,
    roots_by_digest: Mapping[str, Path],
    *,
    description: str,
) -> RunIdentity:
    document = _require_exact_mapping(
        value,
        _IDENTITY_FIELDS,
        description=description,
    )
    digest = document["root_digest"]
    run_id = document["run_id"]
    if type(digest) is not str or _ROOT_DIGEST_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{description} has an invalid root digest")
    try:
        root = roots_by_digest[digest]
    except KeyError as exc:
        raise ValueError(f"{description} names an unconfigured root") from exc
    relative = _validated_relative_path(
        document["relative_path"],
        description=f"{description} path",
    )
    if len(relative.parts) != 3:
        raise ValueError(f"{description} path does not have the run shape")
    return RunIdentity(root, run_id, root.joinpath(*relative.parts))


def _deserialise_member(
    value: object,
    roots_by_digest: Mapping[str, Path],
    *,
    portion_root: Path,
) -> TrashMember:
    document = _require_exact_mapping(
        value,
        _MEMBER_FIELDS,
        description="trash member",
    )
    identity = _deserialise_identity(
        document["identity"],
        roots_by_digest,
        description="trash member identity",
    )
    if identity.root != portion_root:
        raise ValueError("trash portion contains a member from another root")
    original = _validated_relative_path(
        document["original_relative_path"],
        description="trash member original path",
    )
    trash = _validated_relative_path(
        document["trash_relative_path"],
        description="trash member payload path",
    )
    scenario_id = document["scenario_id"]
    profile_id = document["profile_id"]
    created_at = _validated_timestamp(
        document["created_at"],
        description="trash member created_at",
    )
    run_kind = document["run_kind"]
    status = document["status"]
    parent_run_id = document["parent_run_id"]
    artifact_count = document["artifact_count"]
    manifest_digest = document["manifest_digest"]
    size_bytes = document["size_bytes"]
    if (
        type(scenario_id) is not str
        or re.fullmatch(r"[a-z][a-z0-9-]*", scenario_id) is None
        or type(profile_id) is not str
        or re.fullmatch(r"[a-z][a-z0-9-]*", profile_id) is None
    ):
        raise ValueError("trash member has invalid scenario or profile IDs")
    if run_kind not in {"selection", "figure"} or type(run_kind) is not str:
        raise ValueError("trash member has an invalid run kind")
    if status not in RunService.VALID_STATUSES or type(status) is not str:
        raise ValueError("trash member has an invalid status")
    if parent_run_id is not None and (
        type(parent_run_id) is not str or _TRANSACTION_ID_PATTERN.fullmatch(parent_run_id) is None
    ):
        raise ValueError("trash member has an invalid parent run ID")
    if type(artifact_count) is not int or artifact_count < 0:
        raise ValueError("trash member has an invalid artifact count")
    if (
        type(manifest_digest) is not str
        or _MANIFEST_DIGEST_PATTERN.fullmatch(manifest_digest) is None
    ):
        raise ValueError("trash member has an invalid manifest digest")
    if type(size_bytes) is not int or size_bytes < 0:
        raise ValueError("trash member has an invalid size")
    if original != _identity_relative_path(identity):
        raise ValueError("trash member original path disagrees with its identity")
    expected_trash = PurePosixPath(
        "runs",
        scenario_id,
        profile_id,
        identity.expected_path.name,
    )
    if trash != expected_trash:
        raise ValueError("trash member payload path disagrees with its identity")
    if original.parts[:2] != (scenario_id, profile_id):
        raise ValueError("trash member original path disagrees with its metadata")
    try:
        _, expected_path = RunService(identity.root)._expected_paths(
            scenario_id=scenario_id,
            profile_id=profile_id,
            created_at=created_at,
            run_id=identity.run_id,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("trash member identity could not be reconstructed") from exc
    if expected_path != identity.expected_path:
        raise ValueError("trash member path does not match its timestamp and run ID")
    return TrashMember(
        identity=identity,
        original_relative_path=original,
        trash_relative_path=trash,
        scenario_id=scenario_id,
        profile_id=profile_id,
        created_at=created_at,
        run_kind=run_kind,
        status=status,
        parent_run_id=parent_run_id,
        artifact_count=artifact_count,
        manifest_digest=manifest_digest,
        size_bytes=size_bytes,
    )


def _deserialise_edge(
    value: object,
    roots_by_digest: Mapping[str, Path],
) -> RunEdge:
    document = _require_exact_mapping(
        value,
        _EDGE_FIELDS,
        description="trash edge",
    )
    return RunEdge(
        parent=_deserialise_identity(
            document["parent"],
            roots_by_digest,
            description="trash edge parent",
        ),
        child=_deserialise_identity(
            document["child"],
            roots_by_digest,
            description="trash edge child",
        ),
        source=document["source"],
    )


def _validated_id_progress(
    value: object,
    *,
    description: str,
) -> tuple[str, ...]:
    if type(value) is not list:
        raise ValueError(f"{description} must be a list")
    values = tuple(value)
    if any(
        type(item) is not str or _TRANSACTION_ID_PATTERN.fullmatch(item) is None for item in values
    ):
        raise ValueError(f"{description} contains an invalid run ID")
    if len(set(values)) != len(values):
        raise ValueError(f"{description} contains duplicate run IDs")
    return values


def _deserialise_portion(
    document: object,
    *,
    root: Path,
    transaction_path: Path,
    directory_transaction_id: str,
    roots_by_digest: Mapping[str, Path],
) -> _TrashPortion:
    payload = _require_exact_mapping(
        document,
        _PORTION_FIELDS,
        description="trash journal",
    )
    if payload["schema_version"] != 1 or type(payload["schema_version"]) is not int:
        raise ValueError("trash journal schema version is unsupported")
    transaction_id = payload["transaction_id"]
    if (
        type(transaction_id) is not str
        or _TRANSACTION_ID_PATTERN.fullmatch(transaction_id) is None
        or transaction_id != directory_transaction_id
    ):
        raise ValueError("trash journal transaction ID disagrees with its directory")
    deleted_at = _validated_timestamp(
        payload["deleted_at"],
        description="trash journal deleted_at",
    )
    try:
        state = TrashState(payload["state"])
    except (TypeError, ValueError) as exc:
        raise ValueError("trash journal state is invalid") from exc
    if state not in _RAW_JOURNAL_STATES:
        raise ValueError("trash journal state has no durable recovery phase")

    expected_roots_value = payload["expected_roots"]
    if type(expected_roots_value) is not list or not expected_roots_value:
        raise ValueError("trash journal expected roots must be a nonempty list")
    expected_roots = tuple(expected_roots_value)
    if (
        any(
            type(item) is not str or _ROOT_DIGEST_PATTERN.fullmatch(item) is None
            for item in expected_roots
        )
        or tuple(sorted(expected_roots)) != expected_roots
        or len(set(expected_roots)) != len(expected_roots)
    ):
        raise ValueError("trash journal expected roots are invalid or unordered")
    if any(item not in roots_by_digest for item in expected_roots):
        raise ValueError("trash journal expects an unconfigured root")
    if _root_digest(root) not in expected_roots:
        raise ValueError("trash journal is stored below an unexpected root")

    selected = _deserialise_identity(
        payload["selected"],
        roots_by_digest,
        description="trash selected identity",
    )
    members_value = payload["members"]
    if type(members_value) is not list or not members_value:
        raise ValueError("trash journal members must be a nonempty list")
    members = tuple(
        _deserialise_member(
            item,
            roots_by_digest,
            portion_root=root,
        )
        for item in members_value
    )
    member_keys = tuple(_identity_key(member.identity) for member in members)
    if len(set(member_keys)) != len(member_keys):
        raise ValueError("trash journal contains duplicate members")

    edges_value = payload["edges"]
    if type(edges_value) is not list:
        raise ValueError("trash journal edges must be a list")
    edges = tuple(_deserialise_edge(item, roots_by_digest) for item in edges_value)
    edge_keys = tuple(
        (_identity_key(edge.parent), _identity_key(edge.child), edge.source) for edge in edges
    )
    if len(set(edge_keys)) != len(edge_keys) or edge_keys != tuple(sorted(edge_keys)):
        raise ValueError("trash journal edges are duplicate or unordered")

    completed_move_ids = _validated_id_progress(
        payload["completed_move_ids"],
        description="trash completed move IDs",
    )
    completed_restore_ids = _validated_id_progress(
        payload["completed_restore_ids"],
        description="trash completed restore IDs",
    )
    total_size = payload["total_size_bytes"]
    if type(total_size) is not int or total_size < 0:
        raise ValueError("trash journal total size is invalid")
    errors_value = payload["errors"]
    if type(errors_value) is not list or any(
        type(item) is not str or _DURABLE_ERROR_PATTERN.fullmatch(item) is None
        for item in errors_value
    ):
        raise ValueError("trash journal errors must be machine-safe phase and type codes")
    return _TrashPortion(
        root=root,
        transaction_path=transaction_path,
        transaction_id=transaction_id,
        deleted_at=deleted_at,
        state=state,
        selected=selected,
        expected_root_digests=expected_roots,
        members=members,
        edges=edges,
        completed_move_ids=completed_move_ids,
        completed_restore_ids=completed_restore_ids,
        total_size_bytes=total_size,
        errors=tuple(errors_value),
    )


def _ordered_transaction_members(
    selected: RunIdentity,
    members: Iterable[TrashMember],
    edges: tuple[RunEdge, ...],
) -> tuple[TrashMember, ...]:
    members_by_identity: dict[RunIdentity, TrashMember] = {}
    run_ids: set[str] = set()
    for member in members:
        if member.identity in members_by_identity:
            raise ValueError("trash transaction contains duplicate members")
        if member.identity.run_id in run_ids:
            raise ValueError("trash transaction contains duplicate run IDs")
        members_by_identity[member.identity] = member
        run_ids.add(member.identity.run_id)
    if selected not in members_by_identity:
        raise ValueError("trash selected identity is not a transaction member")

    children: dict[RunIdentity, set[RunIdentity]] = {
        identity: set() for identity in members_by_identity
    }
    for edge in edges:
        if edge.parent in members_by_identity and edge.child in members_by_identity:
            children[edge.parent].add(edge.child)
        if edge.parent not in members_by_identity and edge.child not in members_by_identity:
            raise ValueError("trash edge is unrelated to the transaction family")
    RunDependencyGraph._assert_acyclic(
        members_by_identity,
        frozenset(
            edge
            for edge in edges
            if edge.parent in members_by_identity and edge.child in members_by_identity
        ),
    )

    depths = {selected: 0}
    pending = [selected]
    while pending:
        parent = pending.pop()
        next_depth = depths[parent] + 1
        for child in children[parent]:
            if child == selected:
                raise ValueError("trash transaction edges contain a cycle")
            previous = depths.get(child)
            if previous is not None and next_depth <= previous:
                continue
            depths[child] = next_depth
            pending.append(child)
    if set(depths) != set(members_by_identity):
        raise ValueError("trash transaction family is incomplete or disconnected")

    for member in members_by_identity.values():
        if member.parent_run_id is None:
            continue
        matches = {
            edge.parent
            for edge in edges
            if edge.child == member.identity
            and edge.source == "parent_run_id"
            and edge.parent.root == member.identity.root
            and edge.parent.run_id == member.parent_run_id
        }
        if len(matches) != 1:
            raise ValueError("trash member parent edge disagrees with its snapshot")

    return tuple(
        members_by_identity[identity]
        for identity in sorted(
            members_by_identity,
            key=lambda identity: _member_sort_key(identity, depths),
        )
    )


def _validate_transaction_progress(
    state: TrashState,
    members: tuple[TrashMember, ...],
    completed_move_ids: tuple[str, ...],
    completed_restore_ids: tuple[str, ...],
) -> None:
    move_order = tuple(member.identity.run_id for member in members)
    if completed_move_ids != move_order[: len(completed_move_ids)]:
        raise ValueError("trash completed move IDs are not an ordered prefix")
    restore_order = tuple(reversed(completed_move_ids))
    if completed_restore_ids != restore_order[: len(completed_restore_ids)]:
        raise ValueError("trash completed restore IDs are not an ordered prefix")
    if (
        state
        in {
            TrashState.TRASHED,
            TrashState.RESTORING,
            TrashState.PURGING,
            TrashState.PURGE_FAILED,
        }
        and completed_move_ids != move_order
    ):
        raise ValueError("trash state requires every family member to have moved")
    if (
        state
        in {
            TrashState.TRASHED,
            TrashState.PURGING,
            TrashState.PURGE_FAILED,
        }
        and completed_restore_ids
    ):
        raise ValueError("trash state has invalid completed restore IDs")


def _reconcile_mutable_portions(
    portions: tuple[_TrashPortion, ...],
    members: tuple[TrashMember, ...],
) -> tuple[TrashState, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    for portion in portions:
        _validate_transaction_progress(
            portion.state,
            members,
            portion.completed_move_ids,
            portion.completed_restore_ids,
        )
    move_ids = max(
        (portion.completed_move_ids for portion in portions),
        key=len,
    )
    if any(ids != move_ids[: len(ids)] for ids in (p.completed_move_ids for p in portions)):
        raise ValueError("trash completed move progress diverges across portions")
    restore_order = tuple(reversed(move_ids))
    restore_ids = max(
        (portion.completed_restore_ids for portion in portions),
        key=len,
    )
    if any(
        ids != restore_order[: len(ids)]
        for ids in (portion.completed_restore_ids for portion in portions)
    ):
        raise ValueError("trash completed restore progress diverges across portions")

    states = frozenset(portion.state for portion in portions)
    if len(states) == 1:
        state = next(iter(states))
    elif states == {TrashState.MOVING, TrashState.TRASHED}:
        state = TrashState.MOVING
    elif states == {TrashState.RESTORING, TrashState.TRASHED}:
        state = TrashState.RESTORING
    else:
        raise ValueError("trash transaction states diverge across portions")
    errors = tuple(sorted({error for portion in portions for error in portion.errors}))
    return state, move_ids, restore_ids, errors


def _merge_portions(portions: Iterable[_TrashPortion]) -> TrashTransaction:
    values = tuple(portions)
    if not values:
        raise ValueError("trash transaction has no portions")
    first = values[0]
    if any(portion.immutable_contract() != first.immutable_contract() for portion in values[1:]):
        raise ValueError("trash transaction portions disagree")
    roots_by_digest = {_root_digest(portion.root): portion.root for portion in values}
    if len(roots_by_digest) != len(values):
        raise ValueError("trash transaction contains duplicate portions")
    if set(roots_by_digest) != set(first.expected_root_digests):
        raise ValueError("trash transaction is missing expected root portions")
    members = _ordered_transaction_members(
        first.selected,
        (member for portion in values for member in portion.members),
        first.edges,
    )
    for portion in values:
        expected_local = tuple(member for member in members if member.identity.root == portion.root)
        if portion.members != expected_local:
            raise ValueError("trash root-local members are unordered or incomplete")
    if sum(member.size_bytes for member in members) != first.total_size_bytes:
        raise ValueError("trash transaction member sizes disagree with the total")
    state, completed_move_ids, completed_restore_ids, errors = _reconcile_mutable_portions(
        values, members
    )
    public_state = (
        TrashState.RECOVERY_REQUIRED
        if state in {TrashState.MOVING, TrashState.RESTORING}
        else state
    )
    roots = tuple(sorted((portion.root for portion in values), key=_canonical_path_key))
    return TrashTransaction(
        transaction_id=first.transaction_id,
        selected=first.selected,
        members=members,
        roots=roots,
        state=public_state,
        deleted_at=first.deleted_at,
        completed_move_ids=completed_move_ids,
        completed_restore_ids=completed_restore_ids,
        total_size_bytes=first.total_size_bytes,
        errors=errors,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _transaction_from_plan(
    plan: TrashPlan,
    *,
    state: TrashState,
    deleted_at: str,
    completed_move_ids: Iterable[str] = (),
    completed_restore_ids: Iterable[str] = (),
    errors: Iterable[str] = (),
) -> TrashTransaction:
    return TrashTransaction(
        transaction_id=plan.transaction_id,
        selected=plan.selected,
        members=plan.members,
        roots=plan.roots,
        state=state,
        deleted_at=deleted_at,
        completed_move_ids=tuple(completed_move_ids),
        completed_restore_ids=tuple(completed_restore_ids),
        total_size_bytes=plan.total_size_bytes,
        errors=tuple(errors),
    )


class TrashManager:
    """Coordinate journaled whole-family moves across configured roots."""

    def __init__(
        self,
        roots: Callable[[], Iterable[Path]],
        leases: RunUsageLeaseRegistry,
    ) -> None:
        if not callable(roots):
            raise ValueError("trash manager roots must be callable")
        if type(leases) is not RunUsageLeaseRegistry:
            raise ValueError("trash manager requires a run usage lease registry")
        self._roots_provider = roots
        self._leases = leases
        self._operation_entries: dict[RunIdentity, RunEntry] = {}
        self._recovered_transactions: dict[str, TrashTransaction] = {}

    def _current_roots(self) -> tuple[Path, ...]:
        try:
            supplied = tuple(self._roots_provider())
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise TrashTransactionError("configured trash roots are unavailable") from exc
        roots: dict[str, Path] = {}
        for value in supplied:
            try:
                root = RunService(value).output_root
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise TrashTransactionError("configured trash root is unsafe") from exc
            key = _canonical_path_key(root)
            if key in roots:
                raise TrashTransactionError("configured trash roots contain duplicates")
            roots[key] = root
        return tuple(roots[key] for key in sorted(roots))

    def _fresh_plan(
        self,
        selected: RunIdentity,
    ) -> tuple[TrashPlan, RunDependencyGraph]:
        roots = self._current_roots()
        graph = RunDependencyGraph.from_roots(roots)
        family = graph.family(selected)
        return build_trash_plan(graph, selected, family), graph

    @staticmethod
    def _member_identities(plan: TrashPlan) -> tuple[RunIdentity, ...]:
        return tuple(member.identity for member in plan.members)

    def _assert_unleased(self, plan: TrashPlan) -> None:
        conflicts = self._leases.conflicts(self._member_identities(plan))
        if conflicts:
            raise RunLeaseConflictError("run family is in use by " + ", ".join(conflicts))

    def plan(self, selected: RunIdentity) -> TrashPlan:
        if type(selected) is not RunIdentity:
            raise ValueError("trash plan selection must be a RunIdentity")
        plan, _ = self._fresh_plan(selected)
        self._assert_unleased(plan)
        return plan

    @staticmethod
    def _same_plan_contract(displayed: TrashPlan, current: TrashPlan) -> bool:
        return (
            displayed.selected == current.selected
            and displayed.members == current.members
            and displayed.edges == current.edges
            and displayed.roots == current.roots
            and displayed.fingerprint == current.fingerprint
            and displayed.total_size_bytes == current.total_size_bytes
        )

    def _preflight_root(self, root: Path, plan: TrashPlan) -> None:
        if _TRANSACTION_ID_PATTERN.fullmatch(plan.transaction_id) is None:
            raise TrashTransactionError("trash transaction ID is invalid")
        if not os.path.lexists(root) or _is_redirected_path(root) or not root.is_dir():
            raise TrashTransactionError("an affected output root is unavailable")
        try:
            if root.resolve(strict=True) != root:
                raise TrashTransactionError("an affected output root is redirected")
            details = root.stat()
        except OSError as exc:
            raise TrashTransactionError("an affected output root is unavailable") from exc
        write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        if not details.st_mode & write_bits or not os.access(root, os.W_OK | os.X_OK):
            raise TrashTransactionError("an affected output root is not writable")

        trash_root = root / ".trash"
        if os.path.lexists(trash_root):
            if _is_redirected_path(trash_root) or not trash_root.is_dir():
                raise TrashTransactionError("the trash root is unsafe")
            try:
                trash_details = trash_root.stat()
            except OSError as exc:
                raise TrashTransactionError("the trash root is unavailable") from exc
            if not trash_details.st_mode & write_bits or not os.access(
                trash_root, os.W_OK | os.X_OK
            ):
                raise TrashTransactionError("the trash root is not writable")
        transaction = trash_root / plan.transaction_id
        if os.path.lexists(transaction):
            raise TrashTransactionError("trash transaction destination already exists")

    def _preflight(self, plan: TrashPlan) -> None:
        run_ids = [member.identity.run_id for member in plan.members]
        if len(set(run_ids)) != len(run_ids):
            raise TrashTransactionError("trash family has duplicate run IDs")
        for root in plan.roots:
            self._preflight_root(root, plan)

    @staticmethod
    def _journal_document(
        plan: TrashPlan,
        *,
        portion_root: Path,
        state: TrashState,
        deleted_at: str,
        completed_move_ids: tuple[str, ...],
        completed_restore_ids: tuple[str, ...],
        errors: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "transaction_id": plan.transaction_id,
            "deleted_at": deleted_at,
            "state": state.value,
            "selected": _serialise_identity(plan.selected),
            "expected_roots": sorted(_root_digest(root) for root in plan.roots),
            "members": [
                _serialise_member(member)
                for member in plan.members
                if member.identity.root == portion_root
            ],
            "edges": [_serialise_edge(edge) for edge in plan.edges],
            "completed_move_ids": list(completed_move_ids),
            "completed_restore_ids": list(completed_restore_ids),
            "total_size_bytes": plan.total_size_bytes,
            "errors": list(errors),
        }

    def _write_portions(
        self,
        plan: TrashPlan,
        portions: Mapping[Path, Path],
        *,
        state: TrashState,
        deleted_at: str,
        completed_move_ids: Iterable[str] = (),
        completed_restore_ids: Iterable[str] = (),
        errors: Iterable[str] = (),
    ) -> None:
        moved = tuple(completed_move_ids)
        restored = tuple(completed_restore_ids)
        failure_details = tuple(errors)
        for root in plan.roots:
            transaction = portions[root]
            atomic_write_json(
                transaction / "trash.json",
                self._journal_document(
                    plan,
                    portion_root=root,
                    state=state,
                    deleted_at=deleted_at,
                    completed_move_ids=moved,
                    completed_restore_ids=restored,
                    errors=failure_details,
                ),
            )

    @staticmethod
    def _remove_transaction_portions(portions: Mapping[Path, Path]) -> None:
        for transaction in portions.values():
            manifest = transaction / "trash.json"
            if os.path.lexists(manifest):
                if _is_redirected_path(manifest) or not manifest.is_file():
                    raise TrashTransactionError("trash transaction cleanup found an unsafe journal")
                manifest.unlink()
            directories = sorted(
                (path for path in transaction.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            )
            for directory in directories:
                directory.rmdir()
            transaction.rmdir()

    def _move_member(
        self,
        member: TrashMember,
        portions: Mapping[Path, Path],
    ) -> Path:
        entry = self._operation_entries[member.identity]
        destination = portions[member.identity.root].joinpath(*member.trash_relative_path.parts)
        return RunService(member.identity.root).move_entry_to(entry, destination)

    def _restore_member(
        self,
        member: TrashMember,
        portions: Mapping[Path, Path],
    ) -> Path:
        entry = self._operation_entries[member.identity]
        source = portions[member.identity.root].joinpath(*member.trash_relative_path.parts)
        return RunService(member.identity.root).restore_entry_from(source, entry)

    @staticmethod
    def _error_detail(prefix: str, error: BaseException) -> str:
        phase = re.sub(r"[^a-z0-9]+", "_", prefix.casefold()).strip("_")
        return f"{phase}:{type(error).__name__}"

    @staticmethod
    def _cleanup_live_parents(members: Iterable[TrashMember]) -> None:
        parents: set[Path] = set()
        for member in members:
            profile = member.identity.expected_path.parent
            parents.add(profile)
            parents.add(profile.parent)
        for parent in sorted(parents, key=lambda path: len(path.parts), reverse=True):
            try:
                parent.rmdir()
            except OSError:
                pass

    def move(self, displayed: TrashPlan) -> TrashTransaction:
        if type(displayed) is not TrashPlan:
            raise ValueError("trash move requires a displayed TrashPlan")
        try:
            current, graph = self._fresh_plan(displayed.selected)
        except (KeyError, RunDependencyError, TrashTransactionError, ValueError) as exc:
            raise TrashPlanStaleError(
                "the displayed trash plan is stale; refresh and review it"
            ) from exc
        if not self._same_plan_contract(displayed, current):
            raise TrashPlanStaleError("the displayed trash plan is stale; refresh and review it")
        current = TrashPlan(
            transaction_id=displayed.transaction_id,
            selected=current.selected,
            members=current.members,
            edges=current.edges,
            roots=current.roots,
            fingerprint=current.fingerprint,
            total_size_bytes=current.total_size_bytes,
        )
        self._assert_unleased(current)
        reservation = self._leases.reserve_mutation(self._member_identities(current))
        try:
            self._preflight(current)
            self._operation_entries = {
                member.identity: graph.entry(member.identity) for member in current.members
            }
            deleted_at = _utc_now()
            portions: dict[Path, Path] = {}
            completed: list[TrashMember] = []
            restored: list[TrashMember] = []
        except BaseException:
            self._operation_entries = {}
            self._leases.release_mutation(reservation)
            raise
        try:
            try:
                for root in current.roots:
                    portions[root] = RunService(root).prepare_trash_transaction(
                        current.transaction_id
                    )
                self._write_portions(
                    current,
                    portions,
                    state=TrashState.MOVING,
                    deleted_at=deleted_at,
                )
            except BaseException as exc:
                try:
                    self._remove_transaction_portions(portions)
                except BaseException:
                    pass
                raise TrashTransactionError(
                    "trash move failed before mutation; the family remains live"
                ) from exc

            try:
                for member in current.members:
                    self._move_member(member, portions)
                    completed.append(member)
                    self._write_portions(
                        current,
                        portions,
                        state=TrashState.MOVING,
                        deleted_at=deleted_at,
                        completed_move_ids=(item.identity.run_id for item in completed),
                    )
                self._write_portions(
                    current,
                    portions,
                    state=TrashState.TRASHED,
                    deleted_at=deleted_at,
                    completed_move_ids=(item.identity.run_id for item in completed),
                )
            except BaseException as original_error:
                rollback_errors: list[str] = []
                for member in reversed(completed):
                    try:
                        self._restore_member(member, portions)
                        restored.append(member)
                    except BaseException as rollback_error:
                        rollback_errors.append(self._error_detail("rollback", rollback_error))
                        break
                    try:
                        self._write_portions(
                            current,
                            portions,
                            state=TrashState.MOVING,
                            deleted_at=deleted_at,
                            completed_move_ids=(item.identity.run_id for item in completed),
                            completed_restore_ids=(item.identity.run_id for item in restored),
                            errors=(self._error_detail("move", original_error),),
                        )
                    except BaseException as journal_error:
                        rollback_errors.append(
                            self._error_detail("rollback journal", journal_error)
                        )
                        break

                if not rollback_errors and len(restored) == len(completed):
                    try:
                        self._remove_transaction_portions(portions)
                    except BaseException as cleanup_error:
                        rollback_errors.append(
                            self._error_detail("rollback cleanup", cleanup_error)
                        )
                    else:
                        raise TrashTransactionError(
                            "trash move failed and was rolled back"
                        ) from original_error

                errors = (
                    self._error_detail("move", original_error),
                    *rollback_errors,
                )
                try:
                    self._write_portions(
                        current,
                        portions,
                        state=TrashState.MOVING,
                        deleted_at=deleted_at,
                        completed_move_ids=(item.identity.run_id for item in completed),
                        completed_restore_ids=(item.identity.run_id for item in restored),
                        errors=errors,
                    )
                except BaseException as journal_error:
                    errors = (
                        *errors,
                        self._error_detail("recovery journal", journal_error),
                    )
                raise TrashTransactionError(
                    "trash move failed and requires recovery: " + "; ".join(errors)
                ) from original_error

            self._cleanup_live_parents(current.members)
            return _transaction_from_plan(
                current,
                state=TrashState.TRASHED,
                deleted_at=deleted_at,
                completed_move_ids=(item.identity.run_id for item in completed),
            )
        finally:
            self._operation_entries = {}
            self._leases.release_mutation(reservation)

    def _load_transaction_portions(
        self,
        transaction_id: str,
    ) -> tuple[tuple[_TrashPortion, ...], TrashTransaction]:
        if (
            type(transaction_id) is not str
            or _TRANSACTION_ID_PATTERN.fullmatch(transaction_id) is None
        ):
            raise TrashTransactionError("trash recovery transaction ID is invalid")
        roots = self._current_roots()
        roots_by_digest = {_root_digest(root): root for root in roots}
        portions: list[_TrashPortion] = []
        for root in roots:
            transaction_path = root / ".trash" / transaction_id
            if not os.path.lexists(transaction_path):
                continue
            try:
                trash_root = root / ".trash"
                if (
                    _is_redirected_path(trash_root)
                    or not trash_root.is_dir()
                    or trash_root.resolve(strict=True) != trash_root
                    or _is_redirected_path(transaction_path)
                    or not transaction_path.is_dir()
                    or transaction_path.parent != trash_root
                    or transaction_path.resolve(strict=True) != transaction_path
                ):
                    raise ValueError("trash transaction path is unsafe")
                journal_path = transaction_path / "trash.json"
                if _is_redirected_path(journal_path) or not journal_path.is_file():
                    raise ValueError("trash transaction journal is unsafe")
                document = json.loads(journal_path.read_text(encoding="utf-8"))
                portions.append(
                    _deserialise_portion(
                        document,
                        root=root,
                        transaction_path=transaction_path,
                        directory_transaction_id=transaction_id,
                        roots_by_digest=roots_by_digest,
                    )
                )
            except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise TrashTransactionError("trash recovery journal is non-actionable") from exc
        if not portions:
            try:
                return (), self._recovered_transactions[transaction_id]
            except KeyError as exc:
                raise TrashTransactionError("trash recovery transaction was not found") from exc
        try:
            transaction = _merge_portions(portions)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise TrashTransactionError(
                "trash recovery transaction portions are incomplete or inconsistent"
            ) from exc
        return tuple(portions), transaction

    @staticmethod
    def _entry_proxy(
        member: TrashMember,
        record: Mapping[str, Any],
    ) -> RunEntry:
        entry = object.__new__(RunEntry)
        object.__setattr__(entry, "root", member.identity.root)
        object.__setattr__(entry, "run_dir", member.identity.expected_path)
        object.__setattr__(entry, "record", record)
        return entry

    @staticmethod
    def _assert_member_record(
        member: TrashMember,
        record: Mapping[str, Any],
    ) -> None:
        metadata = record.get("metadata")
        run_kind_value = metadata.get("run_kind") if isinstance(metadata, Mapping) else None
        run_kind = (
            run_kind_value
            if type(run_kind_value) is str and run_kind_value in {"selection", "figure"}
            else "selection"
        )
        artifacts = record.get("artifacts")
        if (
            record.get("run_id") != member.identity.run_id
            or record.get("scenario_id") != member.scenario_id
            or record.get("profile_id") != member.profile_id
            or record.get("created_at") != member.created_at
            or record.get("status") != member.status
            or record.get("parent_run_id") != member.parent_run_id
            or not isinstance(artifacts, (list, tuple))
            or len(artifacts) != member.artifact_count
            or run_kind != member.run_kind
            or _manifest_digest(record) != member.manifest_digest
        ):
            raise TrashTransactionError("trash recovery run manifest changed from its snapshot")

    def _entry_from_physical_copy(
        self,
        member: TrashMember,
        path: Path,
        *,
        live: bool,
    ) -> RunEntry:
        service = RunService(member.identity.root)
        try:
            if live:
                entry = service.entry_for_path(
                    member.identity.expected_path,
                    run_id=member.identity.run_id,
                )
                self._assert_member_record(member, entry.record)
                return entry
            record_path = path / "run.json"
            if _is_redirected_path(record_path) or not record_path.is_file():
                raise ValueError("trash recovery manifest is not a regular file")
            document = json.loads(record_path.read_text(encoding="utf-8"))
            record, _ = service._validate_discovered_record(
                record_path,
                document,
                expected_live_path=member.identity.expected_path,
            )
            self._assert_member_record(member, record)
            entry = self._entry_proxy(member, record)
            service._validated_trash_source_record(
                path,
                entry,
                member.identity.expected_path,
            )
            return entry
        except TrashTransactionError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TrashTransactionError(
                "trash recovery payload failed manifest validation"
            ) from exc

    def _preflight_recovery_pairs(
        self,
        transaction: TrashTransaction,
        portions: Mapping[Path, Path],
    ) -> tuple[dict[RunIdentity, RunEntry], dict[RunIdentity, bool]]:
        live_by_identity: dict[RunIdentity, bool] = {}
        paths: dict[RunIdentity, Path] = {}
        for member in transaction.members:
            original = member.identity.expected_path
            trash = portions[member.identity.root].joinpath(*member.trash_relative_path.parts)
            live_exists = os.path.lexists(original)
            trash_exists = os.path.lexists(trash)
            if live_exists == trash_exists:
                copies = "both" if live_exists else "neither"
                raise TrashTransactionError(
                    f"trash recovery found {copies} original and trash copies"
                )
            live_by_identity[member.identity] = live_exists
            paths[member.identity] = original if live_exists else trash

        parents_first = tuple(reversed(transaction.members))
        live_pattern = tuple(live_by_identity[member.identity] for member in parents_first)
        live_count = sum(live_pattern)
        if live_pattern != (True,) * live_count + (False,) * (len(live_pattern) - live_count):
            raise TrashTransactionError(
                "trash recovery path pairs do not form a deterministic phase prefix"
            )

        entries: dict[RunIdentity, RunEntry] = {}
        seen: set[tuple[Any, ...]] = set()
        for member in transaction.members:
            path = paths[member.identity]
            entries[member.identity] = self._entry_from_physical_copy(
                member,
                path,
                live=live_by_identity[member.identity],
            )
            try:
                size = _validated_tree_size(path, seen)
            except (OSError, RuntimeError, ValueError) as exc:
                raise TrashTransactionError(
                    "trash recovery payload failed size validation"
                ) from exc
            if size != member.size_bytes:
                raise TrashTransactionError("trash recovery payload size changed from its snapshot")
        return entries, live_by_identity

    @staticmethod
    def _plan_from_portions(
        transaction: TrashTransaction,
        portions: tuple[_TrashPortion, ...],
    ) -> TrashPlan:
        return TrashPlan(
            transaction_id=transaction.transaction_id,
            selected=transaction.selected,
            members=transaction.members,
            edges=portions[0].edges,
            roots=transaction.roots,
            fingerprint="",
            total_size_bytes=transaction.total_size_bytes,
        )

    def recover(self, transaction_id: str) -> TrashTransaction:
        portions, transaction = self._load_transaction_portions(transaction_id)
        if not portions:
            return transaction
        raw_state, _, _, _ = _reconcile_mutable_portions(portions, transaction.members)
        if raw_state is TrashState.TRASHED:
            return transaction
        if raw_state not in {TrashState.MOVING, TrashState.RESTORING}:
            raise TrashTransactionError("trash transaction state is not eligible for move recovery")

        portion_paths = {portion.root: portion.transaction_path for portion in portions}
        entries, live_by_identity = self._preflight_recovery_pairs(
            transaction,
            portion_paths,
        )
        self._operation_entries = entries
        plan = self._plan_from_portions(transaction, portions)
        move_ids = tuple(member.identity.run_id for member in transaction.members)
        parents_first = tuple(reversed(transaction.members))
        live_restore_ids = [
            member.identity.run_id for member in parents_first if live_by_identity[member.identity]
        ]
        errors = list(transaction.errors)
        try:
            if raw_state is TrashState.MOVING:
                self._write_portions(
                    plan,
                    portion_paths,
                    state=TrashState.MOVING,
                    deleted_at=transaction.deleted_at,
                    completed_move_ids=move_ids,
                    completed_restore_ids=live_restore_ids,
                    errors=errors,
                )
                try:
                    for member in parents_first[len(live_restore_ids) :]:
                        self._restore_member(member, portion_paths)
                        live_restore_ids.append(member.identity.run_id)
                        self._write_portions(
                            plan,
                            portion_paths,
                            state=TrashState.MOVING,
                            deleted_at=transaction.deleted_at,
                            completed_move_ids=move_ids,
                            completed_restore_ids=live_restore_ids,
                            errors=errors,
                        )
                    self._preflight_recovery_pairs(transaction, portion_paths)
                    self._remove_transaction_portions(portion_paths)
                except BaseException as exc:
                    errors.append(self._error_detail("move recovery", exc))
                    try:
                        self._write_portions(
                            plan,
                            portion_paths,
                            state=TrashState.MOVING,
                            deleted_at=transaction.deleted_at,
                            completed_move_ids=move_ids,
                            completed_restore_ids=live_restore_ids,
                            errors=errors,
                        )
                    except BaseException:
                        pass
                    raise TrashTransactionError("trash move recovery stopped safely") from exc
                recovered = TrashTransaction(
                    transaction_id=transaction.transaction_id,
                    selected=transaction.selected,
                    members=transaction.members,
                    roots=transaction.roots,
                    state=TrashState.RECOVERY_REQUIRED,
                    deleted_at=transaction.deleted_at,
                    completed_move_ids=move_ids,
                    completed_restore_ids=tuple(live_restore_ids),
                    total_size_bytes=transaction.total_size_bytes,
                    errors=tuple(errors),
                )
                self._recovered_transactions[transaction_id] = recovered
                return recovered

            self._write_portions(
                plan,
                portion_paths,
                state=TrashState.RESTORING,
                deleted_at=transaction.deleted_at,
                completed_move_ids=move_ids,
                completed_restore_ids=live_restore_ids,
                errors=errors,
            )
            try:
                for member in transaction.members:
                    if not live_by_identity[member.identity]:
                        continue
                    self._move_member(member, portion_paths)
                    live_by_identity[member.identity] = False
                    live_restore_ids = [
                        candidate.identity.run_id
                        for candidate in parents_first
                        if live_by_identity[candidate.identity]
                    ]
                    self._write_portions(
                        plan,
                        portion_paths,
                        state=TrashState.RESTORING,
                        deleted_at=transaction.deleted_at,
                        completed_move_ids=move_ids,
                        completed_restore_ids=live_restore_ids,
                        errors=errors,
                    )
                self._preflight_recovery_pairs(transaction, portion_paths)
                self._write_portions(
                    plan,
                    portion_paths,
                    state=TrashState.TRASHED,
                    deleted_at=transaction.deleted_at,
                    completed_move_ids=move_ids,
                )
            except BaseException as exc:
                errors.append(self._error_detail("restore recovery", exc))
                try:
                    self._write_portions(
                        plan,
                        portion_paths,
                        state=TrashState.RESTORING,
                        deleted_at=transaction.deleted_at,
                        completed_move_ids=move_ids,
                        completed_restore_ids=live_restore_ids,
                        errors=errors,
                    )
                except BaseException:
                    pass
                raise TrashTransactionError("trash restore recovery stopped safely") from exc
            return TrashTransaction(
                transaction_id=transaction.transaction_id,
                selected=transaction.selected,
                members=transaction.members,
                roots=transaction.roots,
                state=TrashState.TRASHED,
                deleted_at=transaction.deleted_at,
                completed_move_ids=move_ids,
                completed_restore_ids=(),
                total_size_bytes=transaction.total_size_bytes,
                errors=(),
            )
        finally:
            self._operation_entries = {}

    def _discovery_roots(
        self,
    ) -> tuple[tuple[Path, ...], list[TrashDiagnostic]]:
        diagnostics: list[TrashDiagnostic] = []
        try:
            supplied = tuple(self._roots_provider())
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return (), [
                TrashDiagnostic(
                    root_digest=sha256(b"unavailable-root-provider").hexdigest(),
                    transaction_id=None,
                    code="trash.root_provider_unavailable",
                    error=f"configured trash roots are unavailable: {type(exc).__name__}",
                )
            ]
        roots: dict[str, Path] = {}
        digests: set[str] = set()
        for value in supplied:
            try:
                root = RunService(value).output_root
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                try:
                    candidate = Path(value).expanduser().absolute()
                    digest = sha256(_canonical_path_key(candidate).encode("utf-8")).hexdigest()
                except (OSError, RuntimeError, TypeError, ValueError):
                    digest = sha256(repr(type(value)).encode("utf-8")).hexdigest()
                diagnostics.append(
                    TrashDiagnostic(
                        root_digest=digest,
                        transaction_id=None,
                        code="trash.root_invalid",
                        error=f"configured trash root is unsafe: {type(exc).__name__}",
                    )
                )
                continue
            key = _canonical_path_key(root)
            digest = _root_digest(root)
            if key in roots or digest in digests:
                diagnostics.append(
                    TrashDiagnostic(
                        root_digest=digest,
                        transaction_id=None,
                        code="trash.root_duplicate",
                        error="configured trash roots contain duplicate identities",
                    )
                )
                continue
            roots[key] = root
            digests.add(digest)
        return tuple(roots[key] for key in sorted(roots)), diagnostics

    def snapshot(self) -> TrashDiscovery:
        roots, diagnostics = self._discovery_roots()
        roots_by_digest = {_root_digest(root): root for root in roots}
        grouped: dict[str, list[_TrashPortion]] = {}
        for root in roots:
            digest = _root_digest(root)
            trash_root = root / ".trash"
            if not os.path.lexists(trash_root):
                continue
            try:
                if _is_redirected_path(trash_root) or not trash_root.is_dir():
                    raise ValueError("trash root must be a real directory")
                if trash_root.resolve(strict=True) != trash_root:
                    raise ValueError("trash root must not be redirected")
                children = tuple(sorted(trash_root.iterdir(), key=lambda path: path.name))
            except (OSError, RuntimeError, ValueError) as exc:
                diagnostics.append(
                    TrashDiagnostic(
                        root_digest=digest,
                        transaction_id=None,
                        code="trash.root_unreadable",
                        error=f"trash root is non-actionable: {type(exc).__name__}: {exc}",
                    )
                )
                continue

            for transaction_path in children:
                transaction_id = (
                    transaction_path.name
                    if _TRANSACTION_ID_PATTERN.fullmatch(transaction_path.name)
                    else None
                )
                if transaction_id is None:
                    diagnostics.append(
                        TrashDiagnostic(
                            root_digest=digest,
                            transaction_id=None,
                            code="trash.transaction_invalid",
                            error="trash transaction directory has an invalid ID",
                        )
                    )
                    continue
                try:
                    if (
                        _is_redirected_path(transaction_path)
                        or not transaction_path.is_dir()
                        or transaction_path.parent != trash_root
                        or transaction_path.resolve(strict=True) != transaction_path
                    ):
                        raise ValueError("trash transaction must be an exact real directory")
                    journal_path = transaction_path / "trash.json"
                    if (
                        _is_redirected_path(journal_path)
                        or not journal_path.is_file()
                        or journal_path.parent != transaction_path
                    ):
                        raise ValueError("trash journal must be a regular file")
                    try:
                        document = json.loads(journal_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as exc:
                        raise ValueError("trash journal contains malformed JSON") from exc
                    portion = _deserialise_portion(
                        document,
                        root=root,
                        transaction_path=transaction_path,
                        directory_transaction_id=transaction_id,
                        roots_by_digest=roots_by_digest,
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    diagnostics.append(
                        TrashDiagnostic(
                            root_digest=digest,
                            transaction_id=transaction_id,
                            code="trash.portion_invalid",
                            error=f"trash portion is non-actionable: {type(exc).__name__}: {exc}",
                        )
                    )
                    continue
                grouped.setdefault(transaction_id, []).append(portion)

        transactions: list[TrashTransaction] = []
        for transaction_id, portions in sorted(grouped.items()):
            try:
                transactions.append(_merge_portions(portions))
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                diagnostics.append(
                    TrashDiagnostic(
                        root_digest=min(_root_digest(portion.root) for portion in portions),
                        transaction_id=transaction_id,
                        code="trash.transaction_inconsistent",
                        error=(f"trash transaction is non-actionable: {type(exc).__name__}: {exc}"),
                    )
                )
        transactions.sort(key=lambda transaction: transaction.transaction_id)
        diagnostics.sort(
            key=lambda item: (
                item.transaction_id or "",
                item.root_digest,
                item.code,
                item.error,
            )
        )
        return TrashDiscovery(tuple(transactions), tuple(diagnostics))

    def list_transactions(self) -> tuple[TrashTransaction, ...]:
        return self.snapshot().transactions


class RunUsageLeaseRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._leases: dict[str, tuple[str, frozenset[RunIdentity]]] = {}
        self._mutations: dict[str, frozenset[RunIdentity]] = {}

    def acquire(self, identities: Iterable[RunIdentity], owner: str) -> str:
        members = frozenset(identities)
        if not members or type(owner) is not str or not owner.strip():
            raise ValueError("run usage lease requires identities and an owner")
        with self._lock:
            if any(members.intersection(reserved) for reserved in self._mutations.values()):
                raise RunLeaseConflictError("run family is reserved for mutation")
            lease_id = uuid4().hex
            self._leases[lease_id] = (owner, members)
        return lease_id

    def release(self, lease_id: str) -> None:
        with self._lock:
            self._leases.pop(lease_id, None)

    def reserve_mutation(self, identities: Iterable[RunIdentity]) -> str:
        members = frozenset(identities)
        if not members:
            raise ValueError("run mutation reservation requires identities")
        with self._lock:
            owners = tuple(
                sorted(
                    owner for owner, leased in self._leases.values() if members.intersection(leased)
                )
            )
            if owners:
                raise RunLeaseConflictError("run family is in use by " + ", ".join(owners))
            if any(members.intersection(reserved) for reserved in self._mutations.values()):
                raise RunLeaseConflictError("run family already has a mutation reservation")
            token = uuid4().hex
            self._mutations[token] = members
            return token

    def release_mutation(self, token: str) -> None:
        with self._lock:
            self._mutations.pop(token, None)

    def conflicts(self, identities: Iterable[RunIdentity]) -> tuple[str, ...]:
        requested = frozenset(identities)
        with self._lock:
            return tuple(
                sorted(
                    owner
                    for owner, members in self._leases.values()
                    if requested.intersection(members)
                )
            )
