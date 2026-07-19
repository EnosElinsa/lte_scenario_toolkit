"""Dependency identities, graph fingerprints, and process-local run leases."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from .run_service import RunEntry, RunService


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


def _path_key(path: Path) -> str:
    return path.as_posix()


def _identity_key(identity: RunIdentity) -> tuple[str, str, str]:
    return (
        _path_key(identity.root),
        identity.run_id,
        _path_key(identity.expected_path),
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
    except (OSError, ValueError):
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
        services: dict[Path, RunService] = {}
        for value in roots:
            service = RunService(value)
            services.setdefault(service.output_root, service)
        canonical_roots = tuple(sorted(services, key=_path_key))

        entries: dict[RunIdentity, RunEntry] = {}
        discovery_diagnostics: list[dict[str, Any]] = []
        for root in canonical_roots:
            discovered = services[root].discover_entries()
            entries.update(
                (RunIdentity.from_entry(entry), entry) for entry in discovered.entries
            )
            discovery_diagnostics.extend(
                {"root": str(root), **dict(item)} for item in discovered.diagnostics
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
                "" if item.expected_path is None else _path_key(item.expected_path),
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
            "roots": [str(root) for root in self._roots],
        }
        document = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(document).hexdigest()


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
