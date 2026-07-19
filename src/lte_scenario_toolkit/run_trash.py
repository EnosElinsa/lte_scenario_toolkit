"""Dependency identities, graph fingerprints, and process-local run leases."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

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
        if type(self.run_id) is not str or re.fullmatch(
            r"[0-9a-f]{32}", self.run_id
        ) is None:
            raise ValueError(
                "run identity requires a 32-character lowercase hexadecimal ID"
            )
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
            raise ValueError(
                "run dependency edge source must be parent_run_id or metadata.source"
            )


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
        return (
            self.run_id == identity.run_id
            or self.expected_path == identity.expected_path
        )


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
            entries.update(
                (RunIdentity.from_entry(entry), entry) for entry in discovered.entries
            )
            discovery_diagnostics.extend(
                {"root": root_key, **dict(item)} for item in discovered.diagnostics
            )

        by_root_id: dict[tuple[Path, str], list[RunIdentity]] = {}
        by_path_id: dict[tuple[Path, str], list[RunIdentity]] = {}
        by_path: dict[Path, list[RunIdentity]] = {}
        by_id: dict[str, list[RunIdentity]] = {}
        for identity in entries:
            by_root_id.setdefault((identity.root, identity.run_id), []).append(
                identity
            )
            by_path_id.setdefault(
                (identity.expected_path, identity.run_id), []
            ).append(identity)
            by_path.setdefault(identity.expected_path, []).append(identity)
            by_id.setdefault(identity.run_id, []).append(identity)

        edges: set[RunEdge] = set()
        unresolved: list[_UnresolvedProvenance] = []
        for child, entry in entries.items():
            resolved: dict[
                Literal["parent_run_id", "metadata.source"], RunIdentity
            ] = {}
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
                source_run_id = (
                    source_run_id_value
                    if type(source_run_id_value) is str
                    else None
                )
                source_path = _normalise_source_path(source_mapping.get("path"))
                candidates = (
                    by_path_id.get((source_path, source_run_id), ())
                    if source_path is not None and source_run_id is not None
                    else ()
                )
                if len(candidates) > 1:
                    raise RunDependencyError(
                        "metadata.source identifies an ambiguous parent"
                    )
                if candidates:
                    resolved["metadata.source"] = candidates[0]
                else:
                    path_candidates = (
                        by_path.get(source_path, ())
                        if source_path is not None
                        else ()
                    )
                    id_candidates = (
                        by_id.get(source_run_id, ())
                        if source_run_id is not None
                        else ()
                    )
                    if (
                        len(path_candidates) > 1
                        or len(id_candidates) > 1
                        or (path_candidates and id_candidates)
                    ):
                        raise RunDependencyError(
                            "metadata.source path or run ID is ambiguous"
                        )
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
                (
                    ""
                    if item.expected_path is None
                    else _canonical_path_key(item.expected_path)
                ),
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
        children: dict[RunIdentity, set[RunIdentity]] = {
            identity: set() for identity in entries
        }
        for edge in edges:
            children[edge.parent].add(edge.child)

        state: dict[RunIdentity, int] = {}
        for start in entries:
            if state.get(start, 0) != 0:
                continue
            state[start] = 1
            stack: list[tuple[RunIdentity, Any]] = [
                (start, iter(children[start]))
            ]
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
                "source_path": (
                    None if item.expected_path is None else str(item.expected_path)
                ),
            }
            for item in self._unresolved
        )
        return tuple(dict(item) for item in self._discovery_diagnostics) + unresolved

    def entry(self, identity: RunIdentity) -> RunEntry:
        return self._entries[identity]

    def children_of(self, identity: RunIdentity) -> frozenset[RunIdentity]:
        return frozenset(
            edge.child for edge in self._edges if edge.parent == identity
        )

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
            (
                edge
                for edge in self._edges
                if edge.parent in family or edge.child in family
            ),
            key=lambda edge: (
                _identity_key(edge.parent),
                _identity_key(edge.child),
                edge.source,
            ),
        )
        payload = {
            "selected": _identity_document(selected),
            "members": [
                _identity_document(identity) for identity in family_identities
            ],
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
                raise ValueError(
                    "run contents contain a non-regular filesystem entry"
                )
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
            if type(run_kind_value) is str
            and run_kind_value in {"figure", "selection"}
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


class RunUsageLeaseRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._leases: dict[str, tuple[str, frozenset[RunIdentity]]] = {}

    def acquire(self, identities: Iterable[RunIdentity], owner: str) -> str:
        members = frozenset(identities)
        if not members or type(owner) is not str or not owner.strip():
            raise ValueError("run usage lease requires identities and an owner")
        lease_id = uuid4().hex
        with self._lock:
            self._leases[lease_id] = (owner, members)
        return lease_id

    def release(self, lease_id: str) -> None:
        with self._lock:
            self._leases.pop(lease_id, None)

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
