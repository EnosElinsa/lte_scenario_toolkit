"""Rebuildable, non-authoritative run history models and rendering helpers."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ...figure_service import FigureSpec
from ...io import atomic_write_json
from ...run_service import RunEntry, RunService
from ...run_trash import (
    RunDependencyError,
    RunIdentity,
    RunLeaseConflictError,
    TrashDiagnostic,
    TrashDiscovery,
    TrashManager,
    TrashPlan,
    TrashPlanStaleError,
    TrashState,
    TrashTransaction,
)
from ..presentation import (
    TRASH_ACTIONS_BY_STATE,
    ActionSpec,
    PresentationSpec,
    TrashAction,
    render_action_bar,
    render_empty_state,
    render_loading_state,
    render_page_header,
    render_status_badge,
    render_technical_details,
    run_state_presentation,
    trash_action_presentation,
    trash_state_presentation,
)

HISTORY_INDEX_RELATIVE_PATH = Path(".lte-data") / "cache" / "history-index.json"


class HistoryAction(str, Enum):
    """Actions exposed for a published History row.

    ``MOVE_TO_TRASH`` is reversible at this layer; the Trash surface owns the
    separate, explicitly confirmed permanent-delete action.
    """

    REVEAL_DIRECTORY = "reveal_directory"
    INSPECT = "inspect"
    OPEN_FIGURES = "open_figures"
    RETRY_MISSING = "retry_missing"
    MOVE_TO_TRASH = "move_to_trash"
    REFRESH = "refresh"


class HistoryActionError(ValueError):
    """Raised when an action target is no longer a live, discovered run."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return deepcopy(value)


def _canonical_root(value: str | os.PathLike[str]) -> Path:
    try:
        return Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid history output root: {value!r}") from exc


def _path_identity(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


@dataclass(frozen=True, slots=True)
class HistoryRunReference:
    """Opaque-enough run identity that is revalidated before every action."""

    root: Path
    run_id: str
    scenario_id: str
    profile_id: str
    created_at: str
    expected_path: Path

    def __post_init__(self) -> None:
        root = _canonical_root(self.root)
        path = Path(self.expected_path).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("history run path must remain inside its output root") from exc
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "expected_path", path)


@dataclass(frozen=True, slots=True)
class HistoryDiagnostic:
    """One invalid run record or non-authoritative index write diagnostic."""

    root: Path
    path: Path
    error: str


@dataclass(frozen=True, slots=True)
class HistoryRow:
    """Immutable display summary derived from one currently valid run manifest."""

    root: Path
    path: Path
    run_id: str
    scenario_id: str
    profile_id: str
    created_at: str
    local_created_at: str
    status: str
    parent_run_id: str | None
    artifacts: tuple[str, ...]
    parameters: Mapping[str, Any]
    candidate: Mapping[str, Any]
    errors: tuple[Mapping[str, Any], ...]
    record: Mapping[str, Any]
    reference: HistoryRunReference
    figure_source_reference: HistoryRunReference | None = None
    figure_source_path: Path | None = None
    missing_artifacts: tuple[str, ...] = ()
    retry_formats: tuple[str, ...] = ()
    discovery_roots: tuple[Path, ...] = ()

    @property
    def can_open_figures(self) -> bool:
        return self.figure_source_reference is not None

    @property
    def can_retry_missing(self) -> bool:
        return bool(self.figure_source_reference is not None and self.retry_formats)

    @property
    def available_actions(self) -> tuple[HistoryAction, ...]:
        actions = [HistoryAction.REVEAL_DIRECTORY, HistoryAction.INSPECT]
        if self.can_open_figures:
            actions.append(HistoryAction.OPEN_FIGURES)
        if self.can_retry_missing:
            actions.append(HistoryAction.RETRY_MISSING)
        if self.status in {"completed", "partial", "published"}:
            actions.append(HistoryAction.MOVE_TO_TRASH)
        return tuple(actions)


@dataclass(frozen=True, slots=True)
class HistorySnapshot:
    """A fresh multi-root discovery plus its non-actionable diagnostics."""

    roots: tuple[Path, ...]
    rows: tuple[HistoryRow, ...]
    diagnostics: tuple[HistoryDiagnostic, ...]
    index_path: Path


@dataclass(frozen=True, slots=True)
class TrashImpactRow:
    """Path-free display data for one member of a proposed trash move."""

    run_id: str
    scenario_id: str
    profile_id: str
    local_created_at: str
    run_kind: str
    status: str
    artifact_count: int
    size_bytes: int
    root: Path
    root_digest: str

    @property
    def run_id_prefix(self) -> str:
        return self.run_id[:8]

    @property
    def kind(self) -> str:
        return self.run_kind

    @property
    def size(self) -> int:
        return self.size_bytes


@dataclass(frozen=True, slots=True)
class HistoryTrashPlan:
    """Immutable UI adapter around one freshly validated ``TrashPlan``.

    The object intentionally carries the service plan as an opaque value. UI
    callbacks receive this adapter (or a transaction ID for Trash actions),
    never a browser-supplied filesystem path.
    """

    reference: HistoryRunReference
    selected_identity: RunIdentity
    run_ids: tuple[str, ...]
    run_count: int
    total_size_bytes: int
    roots: tuple[Path, ...]
    has_descendants: bool
    direct_descendant_count: int
    indirect_descendant_count: int
    graph_fingerprint: str
    plan_fingerprint: str
    impact_rows: tuple[TrashImpactRow, ...]
    actions: tuple[TrashAction, ...] = ()
    trash_plan: TrashPlan | object | None = None

    @property
    def fingerprint(self) -> str:
        """Compatibility alias used by stale-confirmation callers."""

        return self.graph_fingerprint

    @property
    def size_bytes(self) -> int:
        return self.total_size_bytes

    @property
    def plan(self) -> TrashPlan | object | None:
        return self.trash_plan

    @property
    def selected(self) -> RunIdentity:
        return self.selected_identity


@dataclass(frozen=True, slots=True)
class TrashSnapshot:
    """Read-only GUI snapshot sourced exclusively from ``TrashManager``."""

    transactions: tuple[TrashTransaction, ...]
    diagnostics: tuple[TrashDiagnostic, ...]
    restore_blockers_by_transaction: tuple[
        tuple[str, tuple[str, ...]], ...
    ] = ()

    @classmethod
    def from_manager(cls, manager: TrashManager) -> TrashSnapshot:
        if not isinstance(manager, TrashManager):
            raise ValueError("TrashSnapshot requires a TrashManager")
        discovery = manager.snapshot()
        if not isinstance(discovery, TrashDiscovery):
            raise ValueError("TrashManager.snapshot() returned an invalid discovery")
        blockers = tuple(
            (
                transaction.transaction_id,
                manager.restore_blockers(transaction.transaction_id),
            )
            for transaction in discovery.transactions
        )
        return cls(discovery.transactions, discovery.diagnostics, blockers)

    @property
    def count(self) -> int:
        return len(self.transactions)

    @property
    def cards(self) -> tuple[TrashCard, ...]:
        return build_trash_cards(self)


@dataclass(frozen=True, slots=True)
class TrashCard:
    """Localized-ready summary of one logical Trash transaction."""

    transaction_id: str
    state: TrashState | str
    deleted_at: str
    run_count: int
    size_bytes: int
    artifact_count: int
    scenario_profiles: tuple[str, ...]
    roots: tuple[Path, ...]
    blockers: tuple[str, ...]
    enabled_actions: tuple[TrashAction, ...]
    transaction: TrashTransaction | object | None = None
    diagnostics: tuple[str, ...] = ()

    @property
    def available_actions(self) -> tuple[TrashAction, ...]:
        return trash_card_actions(self.state)

    @property
    def id_prefix(self) -> str:
        return self.transaction_id[:8]


def _trash_state(value: TrashState | str | object) -> TrashState | None:
    if isinstance(value, TrashState):
        return value
    if type(value) is str:
        try:
            return TrashState(value)
        except ValueError:
            return None
    return None


def trash_card_actions(state: TrashState | str | object) -> tuple[TrashAction, ...]:
    """Return potential actions for a state, failing closed for unknown data."""

    resolved = _trash_state(state)
    return TRASH_ACTIONS_BY_STATE.get(resolved, ()) if resolved is not None else ()


def build_trash_snapshot(manager: TrashManager) -> TrashSnapshot:
    """Build a UI snapshot from the manager's authoritative discovery."""

    return TrashSnapshot.from_manager(manager)


def build_trash_cards(
    snapshot: TrashSnapshot,
    *,
    restore_blockers: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[TrashCard, ...]:
    """Adapt one authoritative snapshot without inventing filesystem state."""

    if not isinstance(snapshot, TrashSnapshot):
        raise ValueError("trash cards require a TrashSnapshot")
    blocker_map = (
        dict(snapshot.restore_blockers_by_transaction)
        if restore_blockers is None
        else restore_blockers
    )
    cards: list[TrashCard] = []
    for transaction in snapshot.transactions:
        state = _trash_state(transaction.state)
        potential = trash_card_actions(state)
        blockers = tuple(blocker_map.get(transaction.transaction_id, ()))
        has_lease = "trash.restore.lease_conflict" in blockers
        unsafe_storage = bool(
            {
                "trash.restore.root_unavailable",
                "trash.restore.journal_invalid",
            }.intersection(blockers)
        )
        only_destination_collision = bool(blockers) and set(blockers) <= {
            "trash.restore.destination_occupied"
        }
        if state is TrashState.TRASHED:
            if has_lease or unsafe_storage:
                enabled = ()
            elif only_destination_collision:
                enabled = (TrashAction.PURGE,)
            else:
                enabled = potential
        elif state is TrashState.PURGE_FAILED:
            # A failed purge remains retryable even when a surviving root or
            # journal needs attention; a live Figures lease still gates the
            # mutation reservation.
            enabled = () if has_lease else (TrashAction.PURGE,)
        elif state is TrashState.RECOVERY_REQUIRED:
            recovery_blocked = has_lease or unsafe_storage
            enabled = () if recovery_blocked else potential
        else:
            enabled = ()
        profiles = tuple(
            dict.fromkeys(
                f"{member.scenario_id} / {member.profile_id}"
                for member in transaction.members
            )
        )
        cards.append(
            TrashCard(
                transaction_id=transaction.transaction_id,
                state=transaction.state,
                deleted_at=transaction.deleted_at,
                run_count=max(
                    len(transaction.members),
                    len(transaction.completed_move_ids),
                ),
                size_bytes=transaction.total_size_bytes,
                artifact_count=sum(member.artifact_count for member in transaction.members),
                scenario_profiles=profiles,
                roots=transaction.roots,
                blockers=blockers,
                enabled_actions=enabled,
                transaction=transaction,
                diagnostics=transaction.errors,
            )
        )
    return tuple(cards)


def figure_source_options(snapshot: HistorySnapshot) -> dict[str, str]:
    """Return one labeled dropdown option per authoritative selection run."""

    if not isinstance(snapshot, HistorySnapshot):
        raise ValueError("snapshot must be a HistorySnapshot")
    options: dict[str, str] = {}
    for row in snapshot.rows:
        source_path = row.figure_source_path
        if source_path is None or _path_identity(source_path) != _path_identity(row.path):
            continue
        resolved = str(source_path.resolve(strict=False))
        options.setdefault(
            resolved,
            f"{row.scenario_id} / {row.profile_id} · {row.local_created_at}",
        )
    return options


@dataclass(frozen=True, slots=True)
class ResolvedHistoryAction:
    """A live-validated action target safe to pass to a local UI callback."""

    action: HistoryAction
    path: Path
    run_id: str
    record: Mapping[str, Any]
    retry_formats: tuple[str, ...] = ()
    figure_spec: FigureSpec | None = None
    destination_root: Path | None = None
    derived_parent_run_id: str | None = None
    derived_parent_path: Path | None = None


def _local_timestamp(created_at: str) -> str:
    candidate = created_at[:-1] + "+00:00" if created_at.endswith("Z") else created_at
    try:
        return datetime.fromisoformat(candidate).astimezone().isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return created_at


def _mapping_field(metadata: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = metadata.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _figure_source_path(path: Path, record: Mapping[str, Any]) -> Path:
    """Return the current run directory used as a Figures source."""

    if record.get("status") != "completed":
        raise ValueError("figure source run must be completed")
    artifacts = record.get("artifacts", ())
    if not isinstance(artifacts, (list, tuple)) or not any(
        type(artifact) is str and Path(artifact).suffix.casefold() == ".csv"
        for artifact in artifacts
    ):
        raise ValueError("figure source run must contain a current CSV artifact")
    return path


def _requested_and_missing(
    metadata: Mapping[str, Any],
    artifacts: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    run_kind = metadata.get("run_kind")
    requested_name = "requested_formats" if run_kind == "figure" else "requested_artifacts"
    raw_requested = metadata.get(requested_name, ())
    if not isinstance(raw_requested, (list, tuple)):
        return (), ()
    requested = tuple(item for item in raw_requested if type(item) is str and item)
    raw_paths = metadata.get("artifact_paths", {})
    artifact_set = set(artifacts)
    published = (
        {
            token
            for token, path in raw_paths.items()
            if type(token) is str and type(path) is str and path in artifact_set
        }
        if isinstance(raw_paths, Mapping)
        else set()
    )
    missing = tuple(token for token in requested if token not in published)
    return requested, missing


def _retry_formats(metadata: Mapping[str, Any], missing: tuple[str, ...]) -> tuple[str, ...]:
    if metadata.get("run_kind") == "figure":
        return tuple(token for token in missing if token in {"png", "eps", "html"})
    mapping = {
        "terrain_png": "png",
        "terrain_eps": "eps",
        "terrain_html": "html",
    }
    return tuple(mapping[token] for token in missing if token in mapping)


def _retry_figure_spec(
    metadata: Mapping[str, Any],
    *,
    required: bool = False,
) -> FigureSpec | None:
    run_kind = metadata.get("run_kind")
    if run_kind not in {"figure", "selection"}:
        return None
    value = metadata.get("figure_spec")
    if value is None and not required:
        return None
    if not isinstance(value, Mapping):
        raise HistoryActionError("Figure run is missing a retryable style specification")
    try:
        return FigureSpec(**dict(value)).validate()
    except (TypeError, ValueError) as exc:
        raise HistoryActionError("Figure run style specification is invalid") from exc


def _row_from_entry(entry: RunEntry) -> HistoryRow:
    record = entry.record
    path = entry.run_dir
    metadata_value = record.get("metadata", {})
    metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
    artifacts_value = record.get("artifacts", ())
    artifacts = (
        tuple(item for item in artifacts_value if type(item) is str)
        if isinstance(artifacts_value, (list, tuple))
        else ()
    )
    errors_value = record.get("errors", ())
    errors = (
        tuple(_freeze(item) for item in errors_value if isinstance(item, Mapping))
        if isinstance(errors_value, (list, tuple))
        else ()
    )
    reference = HistoryRunReference(
        root=entry.root,
        run_id=record["run_id"],
        scenario_id=record["scenario_id"],
        profile_id=record["profile_id"],
        created_at=record["created_at"],
        expected_path=path,
    )
    _, missing = _requested_and_missing(metadata, artifacts)
    has_csv = any(Path(artifact).suffix.casefold() == ".csv" for artifact in artifacts)
    is_completed_source = has_csv and record.get("status") == "completed"
    source_path = _figure_source_path(path, record) if is_completed_source else None
    retry_formats = _retry_formats(metadata, missing)
    if retry_formats:
        try:
            _retry_figure_spec(metadata, required=True)
        except HistoryActionError:
            retry_formats = ()
    return HistoryRow(
        root=reference.root,
        path=path,
        run_id=reference.run_id,
        scenario_id=reference.scenario_id,
        profile_id=reference.profile_id,
        created_at=reference.created_at,
        local_created_at=_local_timestamp(reference.created_at),
        status=record["status"],
        parent_run_id=record.get("parent_run_id"),
        artifacts=artifacts,
        parameters=_freeze(_mapping_field(metadata, "parameters")),
        candidate=_freeze(_mapping_field(metadata, "candidate")),
        errors=errors,
        record=_freeze(record),
        reference=reference,
        figure_source_reference=reference if is_completed_source else None,
        figure_source_path=source_path,
        missing_artifacts=missing,
        retry_formats=retry_formats,
        discovery_roots=(reference.root,),
    )


def _created_at_sort_key(row: HistoryRow) -> tuple[datetime, str, str]:
    created_at = row.created_at
    candidate = created_at[:-1] + "+00:00" if created_at.endswith("Z") else created_at
    return (
        datetime.fromisoformat(candidate),
        row.run_id,
        _path_identity(row.root),
    )


def _link_figure_sources(
    rows: Iterable[HistoryRow],
    *,
    discovery_roots: Iterable[Path] | None = None,
) -> tuple[HistoryRow, ...]:
    values = tuple(rows)
    linked_roots = tuple(
        dict.fromkeys(
            (row.root for row in values)
            if discovery_roots is None
            else discovery_roots
        )
    )
    by_identity = {
        (_path_identity(row.root), row.run_id): row for row in values
    }
    by_id: dict[str, list[HistoryRow]] = {}
    for row in values:
        by_id.setdefault(row.run_id, []).append(row)
    cache: dict[tuple[str, str], HistoryRunReference | None] = {}

    def source_for(
        row: HistoryRow,
        seen: frozenset[tuple[str, str]],
    ) -> HistoryRunReference | None:
        if row.figure_source_reference is not None:
            return row.figure_source_reference
        identity = (_path_identity(row.root), row.run_id)
        if identity in seen:
            cache[identity] = None
            return None
        if identity in cache:
            return cache[identity]
        metadata_value = row.record.get("metadata", {})
        metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
        source_value = metadata.get("source", {})
        source = source_value if isinstance(source_value, Mapping) else {}
        source_run_id = source.get("run_id")
        source_path = source.get("path")
        if type(source_run_id) is str and type(source_path) is str:
            candidates = [
                candidate
                for candidate in by_id.get(source_run_id, ())
                if _path_identity(candidate.path)
                == _path_identity(Path(source_path).resolve(strict=False))
            ]
            if len(candidates) == 1:
                linked = source_for(candidates[0], seen | {identity})
                cache[identity] = linked
                return linked
        parent_id = row.parent_run_id
        parent_identity = (_path_identity(row.root), parent_id or "")
        if parent_id is None or parent_identity in seen:
            cache[identity] = None
            return None
        parent = by_identity.get(parent_identity)
        if parent is None:
            cache[identity] = None
            return None
        linked = source_for(parent, seen | {identity})
        cache[identity] = linked
        return linked

    linked = []
    for row in values:
        source = source_for(row, frozenset())
        source_row = (
            None
            if source is None
            else by_identity.get((_path_identity(source.root), source.run_id))
        )
        linked.append(
            replace(
                row,
                figure_source_reference=source,
                figure_source_path=(
                    None
                    if source is None
                    else (
                        source_row.figure_source_path
                        if source_row is not None
                        else source.expected_path
                    )
                ),
                discovery_roots=linked_roots,
            )
        )
    return tuple(linked)


def _discover_rows(
    service: RunService,
) -> tuple[tuple[HistoryRow, ...], tuple[HistoryDiagnostic, ...]]:
    discovery = service.discover_entries()
    diagnostics = [
        HistoryDiagnostic(
            root=service.output_root.resolve(strict=False),
            path=Path(item.get("path", service.output_root)),
            error=str(item.get("error", "Unknown run discovery error")),
        )
        for item in discovery.diagnostics
    ]
    rows = []
    for entry in discovery.entries:
        try:
            rows.append(_row_from_entry(entry))
        except Exception as exc:
            diagnostics.append(
                HistoryDiagnostic(
                    root=service.output_root.resolve(strict=False),
                    path=service.output_root.resolve(strict=False),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return (
        tuple(sorted(rows, key=_created_at_sort_key, reverse=True)),
        tuple(sorted(diagnostics, key=lambda item: str(item.path))),
    )


def history_rows(service: RunService) -> tuple[HistoryRow, ...]:
    """Return fresh, newest-first display rows for one RunService root."""

    if not isinstance(service, RunService):
        raise ValueError("service must be a RunService")
    rows, _ = _discover_rows(service)
    return tuple(
        sorted(
            _link_figure_sources(
                rows,
                discovery_roots=(service.output_root.resolve(strict=False),),
            ),
            key=_created_at_sort_key,
            reverse=True,
        )
    )


def history_roots(
    repo_root: str | os.PathLike[str],
    output_roots: Iterable[str | os.PathLike[str]],
) -> tuple[Path, ...]:
    """Return the repository results root plus canonical, de-duplicated GUI roots."""

    repository = RunService(repo_root).output_root
    if isinstance(output_roots, (str, bytes, os.PathLike)):
        raise ValueError("history output roots must be a path collection")
    candidates = [repository / "results", *list(output_roots)]
    unique: list[Path] = []
    seen: set[str] = set()
    for value in candidates:
        root = RunService(value).output_root
        identity = _path_identity(root)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(root)
    return tuple(unique)


def _validated_index_path(repository: Path, value: str | os.PathLike[str] | None) -> Path:
    candidate = _requested_index_path(repository, value)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(repository)
    except ValueError as exc:
        raise ValueError("history index must remain inside the repository") from exc
    current = repository
    try:
        relative_parts = candidate.relative_to(repository).parts[:-1]
    except ValueError as exc:
        raise ValueError("history index must remain inside the repository") from exc
    for part in relative_parts:
        current = current / part
        is_junction = getattr(current, "is_junction", None)
        redirected = current.is_symlink() or bool(
            is_junction is not None and is_junction()
        )
        if os.path.lexists(current) and redirected:
            raise ValueError(
                f"history index parent must not be redirected: {current}"
            )
        if os.path.lexists(current) and not current.is_dir():
            raise ValueError(f"history index parent must be a directory: {current}")
    if os.path.lexists(candidate):
        is_junction = getattr(candidate, "is_junction", None)
        if candidate.is_symlink() or bool(
            is_junction is not None and is_junction()
        ):
            raise ValueError("history index must not be redirected")
    return resolved


def _requested_index_path(
    repository: Path,
    value: str | os.PathLike[str] | None,
) -> Path:
    candidate = repository / HISTORY_INDEX_RELATIVE_PATH if value is None else Path(value)
    if not candidate.is_absolute():
        candidate = repository / candidate
    return Path(os.path.abspath(candidate))


def _index_row(row: HistoryRow) -> dict[str, Any]:
    return {
        "root": str(row.root),
        "path": str(row.path),
        "run_id": row.run_id,
        "scenario_id": row.scenario_id,
        "profile_id": row.profile_id,
        "created_at": row.created_at,
        "local_created_at": row.local_created_at,
        "status": row.status,
        "parent_run_id": row.parent_run_id,
        "parameters": _thaw(row.parameters),
        "candidate": _thaw(row.candidate),
        "artifacts": list(row.artifacts),
        "errors": _thaw(row.errors),
        "can_open_figures": row.can_open_figures,
        "figure_source_path": (
            None if row.figure_source_path is None else str(row.figure_source_path)
        ),
        "missing_artifacts": list(row.missing_artifacts),
        "retry_formats": list(row.retry_formats),
    }


def rebuild_history(
    repo_root: str | os.PathLike[str],
    output_roots: Iterable[str | os.PathLike[str]],
    *,
    index_path: str | os.PathLike[str] | None = None,
) -> HistorySnapshot:
    """Discover live manifests first, then replace a derived summary index."""

    repository = _canonical_root(repo_root)
    roots = history_roots(repository, output_roots)
    rows: list[HistoryRow] = []
    diagnostics: list[HistoryDiagnostic] = []
    seen: set[tuple[str, str]] = set()
    for root in roots:
        try:
            discovered_rows, discovered_diagnostics = _discover_rows(
                RunService(root)
            )
        except Exception as exc:
            diagnostics.append(
                HistoryDiagnostic(
                    root=root,
                    path=root,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        diagnostics.extend(discovered_diagnostics)
        for row in discovered_rows:
            identity = (_path_identity(row.path), row.run_id)
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(row)
    rows = list(_link_figure_sources(rows, discovery_roots=roots))
    rows.sort(key=_created_at_sort_key, reverse=True)
    target = _requested_index_path(repository, index_path)
    index_enabled = True
    try:
        target = _validated_index_path(repository, index_path)
    except (OSError, RuntimeError, ValueError) as exc:
        index_enabled = False
        diagnostics.append(
            HistoryDiagnostic(
                root=repository,
                path=target,
                error=f"History index was not updated: {type(exc).__name__}: {exc}",
            )
        )
    payload = {
        "rebuilt_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "roots": [str(root) for root in roots],
        "rows": [_index_row(row) for row in rows],
        "diagnostics": [
            {"root": str(item.root), "path": str(item.path), "error": item.error}
            for item in diagnostics
        ],
    }
    if index_enabled:
        try:
            atomic_write_json(target, payload)
        except OSError as exc:
            diagnostics.append(
                HistoryDiagnostic(
                    root=repository,
                    path=target,
                    error=f"History index was not updated: {type(exc).__name__}: {exc}",
                )
            )
    return HistorySnapshot(
        roots=roots,
        rows=tuple(rows),
        diagnostics=tuple(diagnostics),
        index_path=target,
    )


def resolve_history_reference(
    reference: HistoryRunReference,
) -> tuple[Path, Mapping[str, Any]]:
    """Re-discover and resolve a run reference without consulting the index."""

    if not isinstance(reference, HistoryRunReference):
        raise HistoryActionError("history action requires a run reference")
    service = RunService(reference.root)
    discovery = service.discover_entries()
    matches = [
        entry
        for entry in discovery.entries
        if entry.record.get("run_id") == reference.run_id
        and entry.record.get("scenario_id") == reference.scenario_id
        and entry.record.get("profile_id") == reference.profile_id
        and entry.record.get("created_at") == reference.created_at
    ]
    if len(matches) != 1:
        raise HistoryActionError(
            "The selected run is no longer available; refresh History and try again"
        )
    entry = matches[0]
    path = entry.run_dir
    if _path_identity(path) != _path_identity(reference.expected_path):
        raise HistoryActionError("The selected run path changed; refresh History")
    return path, _freeze(entry.record)


def _same_history_reference(
    left: HistoryRunReference,
    right: HistoryRunReference,
) -> bool:
    return (
        _path_identity(left.root) == _path_identity(right.root)
        and left.run_id == right.run_id
        and left.scenario_id == right.scenario_id
        and left.profile_id == right.profile_id
        and left.created_at == right.created_at
        and _path_identity(left.expected_path) == _path_identity(right.expected_path)
    )


def _display_member_order(plan: TrashPlan) -> tuple[Any, ...]:
    """Order family members parent-first for human review.

    ``TrashPlan.members`` is deliberately descendants-first for safe mutation;
    the dialog presents the lineage in the opposite, reader-friendly order.
    """

    members = {member.identity: member for member in plan.members}
    children: dict[RunIdentity, list[RunIdentity]] = {
        identity: [] for identity in members
    }
    for edge in plan.edges:
        if edge.parent in members and edge.child in members:
            children[edge.parent].append(edge.child)
    for values in children.values():
        values.sort(key=lambda identity: (identity.run_id, str(identity.expected_path)))
    ordered: list[RunIdentity] = []
    pending = [plan.selected]
    while pending:
        identity = pending.pop(0)
        if identity in ordered or identity not in members:
            continue
        ordered.append(identity)
        pending.extend(children[identity])
    ordered.extend(
        sorted(
            (identity for identity in members if identity not in ordered),
            key=lambda identity: (identity.run_id, str(identity.expected_path)),
        )
    )
    return tuple(members[identity] for identity in ordered)


def build_history_trash_plan(
    row: HistoryRow,
    manager: TrashManager,
) -> HistoryTrashPlan:
    """Freshly resolve a published row and adapt its complete family.

    This function performs no filesystem mutation. It is safe to call when an
    old History tab is stale; the reference and manager plan are rebuilt from
    current discovery every time.
    """

    if not isinstance(row, HistoryRow):
        raise HistoryActionError("trash move requires a published HistoryRow")
    if row.status not in {"completed", "partial", "published"}:
        raise HistoryActionError("only published runs can move to Trash")
    if not isinstance(manager, TrashManager):
        raise HistoryActionError("Trash service is unavailable")
    try:
        clicked_path, clicked_record = resolve_history_reference(row.reference)
        if _path_identity(clicked_path) != _path_identity(row.path):
            raise HistoryActionError("The selected run path changed; refresh History")
        selected = RunIdentity(row.root, row.run_id, clicked_path)
        plan = manager.plan(selected)
    except RunLeaseConflictError as exc:
        raise HistoryActionError(
            "This run is in use. Close it in Figures before moving it to Trash."
        ) from exc
    except (RunDependencyError, TrashPlanStaleError, ValueError, OSError) as exc:
        if isinstance(exc, HistoryActionError):
            raise
        raise HistoryActionError(
            "The run family cannot be planned safely; refresh History and try again"
        ) from exc
    if not isinstance(plan, TrashPlan):
        raise HistoryActionError("Trash service returned an invalid plan")
    if plan.selected != selected:
        raise HistoryActionError("The selected run identity changed; refresh History")
    # The resolver above is deliberately retained so a future implementation
    # cannot silently trust a stale row's in-memory record.
    _ = clicked_record
    ordered_members = _display_member_order(plan)
    member_by_identity = {member.identity: member for member in plan.members}
    children = {
        edge.child
        for edge in plan.edges
        if edge.parent == plan.selected
        and edge.child in member_by_identity
    }
    descendant_count = len(plan.members) - 1
    impact_rows = tuple(
        TrashImpactRow(
            run_id=member.identity.run_id,
            scenario_id=member.scenario_id,
            profile_id=member.profile_id,
            local_created_at=_local_timestamp(member.created_at),
            run_kind=member.run_kind,
            status=member.status,
            artifact_count=member.artifact_count,
            size_bytes=member.size_bytes,
            root=member.identity.root,
            root_digest=hashlib.sha256(
                _path_identity(member.identity.root).encode("utf-8")
            ).hexdigest(),
        )
        for member in ordered_members
    )
    return HistoryTrashPlan(
        reference=row.reference,
        selected_identity=selected,
        run_ids=tuple(row.run_id for row in impact_rows),
        run_count=len(impact_rows),
        total_size_bytes=plan.total_size_bytes,
        roots=plan.roots,
        has_descendants=descendant_count > 0,
        direct_descendant_count=len(children),
        indirect_descendant_count=max(0, descendant_count - len(children)),
        graph_fingerprint=plan.fingerprint,
        plan_fingerprint=plan.fingerprint,
        impact_rows=impact_rows,
        trash_plan=plan,
    )


def validate_history_trash_plan(
    displayed: HistoryTrashPlan,
    row: HistoryRow,
    manager: TrashManager,
) -> HistoryTrashPlan:
    """Rebuild and compare a confirmation plan without mutating anything."""

    if not isinstance(displayed, HistoryTrashPlan):
        raise HistoryActionError("trash confirmation requires a displayed plan")
    current = build_history_trash_plan(row, manager)
    if (
        not _same_history_reference(displayed.reference, current.reference)
        or displayed.graph_fingerprint != current.graph_fingerprint
        or displayed.run_ids != current.run_ids
        or displayed.total_size_bytes != current.total_size_bytes
    ):
        raise HistoryActionError(
            "This run family changed. Refresh History and review the updated impact."
        )
    return current


def confirm_history_trash_plan(
    displayed: HistoryTrashPlan,
    row: HistoryRow,
    manager: TrashManager,
) -> TrashTransaction:
    """Validate a displayed plan and execute the service move atomically."""

    current = validate_history_trash_plan(displayed, row, manager)
    if not isinstance(current.trash_plan, TrashPlan):
        raise HistoryActionError("Trash service returned an invalid plan")
    try:
        return manager.move(current.trash_plan)
    except (TrashPlanStaleError, RunLeaseConflictError) as exc:
        raise HistoryActionError(
            "This run family changed or is in use. Refresh History and review the impact."
        ) from exc


# Short aliases used by the application orchestration layer.
resolve_history_trash_plan = build_history_trash_plan
move_history_to_trash = confirm_history_trash_plan


def _fresh_linked_action_row(row: HistoryRow) -> HistoryRow:
    """Re-discover configured roots and rebuild the source graph for one click."""

    roots = row.discovery_roots or (row.root,)
    rows: list[HistoryRow] = []
    seen: set[tuple[str, str]] = set()
    for root in roots:
        try:
            discovered, _diagnostics = _discover_rows(RunService(root))
        except Exception as exc:
            raise HistoryActionError(
                "Run history changed; refresh History and try again"
            ) from exc
        for candidate in discovered:
            identity = (_path_identity(candidate.path), candidate.run_id)
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(candidate)
    matches = [
        candidate
        for candidate in _link_figure_sources(rows, discovery_roots=roots)
        if _same_history_reference(candidate.reference, row.reference)
    ]
    if len(matches) != 1:
        raise HistoryActionError(
            "The selected run is no longer available; refresh History and try again"
        )
    return matches[0]


def resolve_history_action(
    row: HistoryRow,
    action: HistoryAction | str,
) -> ResolvedHistoryAction:
    """Resolve one supported action against fresh manifests and safe paths."""

    if not isinstance(row, HistoryRow):
        raise HistoryActionError("history action requires a HistoryRow")
    try:
        selected = action if isinstance(action, HistoryAction) else HistoryAction(action)
    except ValueError as exc:
        raise HistoryActionError(f"Unsupported history action: {action!r}") from exc
    if selected is HistoryAction.REFRESH:
        raise HistoryActionError("Refresh is a page action and has no run target")
    if selected is HistoryAction.MOVE_TO_TRASH:
        raise HistoryActionError(
            "Move to Trash requires a fresh HistoryTrashPlan; use build_history_trash_plan"
        )
    clicked_path, clicked_record = resolve_history_reference(row.reference)
    metadata_value = clicked_record.get("metadata", {})
    metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
    artifacts_value = clicked_record.get("artifacts", ())
    artifacts = (
        tuple(item for item in artifacts_value if type(item) is str)
        if isinstance(artifacts_value, (list, tuple))
        else ()
    )
    _, fresh_missing = _requested_and_missing(metadata, artifacts)
    fresh_retry_formats = _retry_formats(metadata, fresh_missing)
    requested_spec = (
        _retry_figure_spec(
            metadata,
            required=selected is HistoryAction.RETRY_MISSING,
        )
        if selected in {HistoryAction.OPEN_FIGURES, HistoryAction.RETRY_MISSING}
        else None
    )
    if selected is HistoryAction.RETRY_MISSING and not fresh_retry_formats:
        raise HistoryActionError("This run has no retryable missing figure artifacts")
    if selected in {HistoryAction.OPEN_FIGURES, HistoryAction.RETRY_MISSING}:
        if clicked_record.get("parent_run_id") != row.parent_run_id:
            raise HistoryActionError("The selected run lineage changed; refresh History")
        old_metadata_value = row.record.get("metadata", {})
        old_metadata = (
            old_metadata_value if isinstance(old_metadata_value, Mapping) else {}
        )
        if metadata.get("source") != old_metadata.get("source"):
            raise HistoryActionError("The selected run source changed; refresh History")
        if row.figure_source_reference is None:
            raise HistoryActionError("This run has no compatible figure source")
        fresh_row = _fresh_linked_action_row(row)
        fresh_metadata_value = fresh_row.record.get("metadata", {})
        fresh_metadata = (
            fresh_metadata_value
            if isinstance(fresh_metadata_value, Mapping)
            else {}
        )
        if fresh_row.parent_run_id != clicked_record.get("parent_run_id"):
            raise HistoryActionError("The selected run lineage changed; refresh History")
        if fresh_metadata.get("source") != metadata.get("source"):
            raise HistoryActionError("The selected run source changed; refresh History")
        reference = fresh_row.figure_source_reference
        if reference is None:
            raise HistoryActionError(
                "This run no longer has one safe compatible figure source"
            )
        path, record = resolve_history_reference(reference)
        try:
            path = _figure_source_path(path, record)
        except ValueError as exc:
            raise HistoryActionError(
                "This run no longer has one safe compatible figure source"
            ) from exc
    else:
        reference = row.reference
        path, record = clicked_path, clicked_record
    return ResolvedHistoryAction(
        action=selected,
        path=path,
        run_id=reference.run_id,
        record=record,
        retry_formats=fresh_retry_formats,
        figure_spec=requested_spec,
        destination_root=row.root,
        derived_parent_run_id=row.run_id,
        derived_parent_path=clicked_path,
    )


def _translated(translator: Any, key: str, fallback: str, **values: Any) -> str:
    try:
        return translator.text(key, **values)
    except (KeyError, ValueError):
        return fallback.format(**values)


def _format_bytes(value: int) -> str:
    size = max(0, int(value))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(size)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{int(amount)} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{size} B"


def _invoke_opaque(callback: Callable[..., Any], value: object) -> Any:
    """Call a UI callback with its opaque model, supporting legacy no-arg fakes."""

    try:
        parameters = inspect.signature(callback)
    except (TypeError, ValueError):
        # Some extension/builtin callables have no inspectable signature. Call
        # them exactly once; a callback-body TypeError must never be retried.
        return callback(value)
    try:
        parameters.bind(value)
    except TypeError:
        parameters.bind()
        return callback()
    return callback(value)


def render_move_to_trash_dialog(
    ui: Any,
    translator: Any,
    plan: HistoryTrashPlan,
    *,
    on_confirm: Callable[..., Any],
    blocker: str | None = None,
) -> Any | None:
    """Render one fresh family-impact review without exposing path arguments."""

    if blocker is not None:
        ui.label(blocker).classes("lte-callout lte-callout--warning").mark(
            "history-trash-move-blocked"
        )
        return None
    if not isinstance(plan, HistoryTrashPlan):
        raise ValueError("move dialog requires a HistoryTrashPlan")
    with ui.dialog() as dialog, ui.card().classes(
        "lte-confirmation-dialog lte-trash-impact-dialog"
    ):
        ui.label(
            _translated(
                translator,
                "history.trash_impact",
                "Trash impact",
            )
        ).classes("lte-card-title")
        with ui.column().classes("lte-trash-impact full-width").mark(
            "history-trash-impact"
        ):
            ui.label(
                _translated(
                    translator,
                    "history.trash_impact_summary",
                    (
                        "This action moves {count} runs ({size}) across {roots} "
                        "output roots. The family remains restorable."
                    ),
                    count=plan.run_count,
                    size=_format_bytes(plan.total_size_bytes),
                    roots=len(plan.roots),
                )
            ).classes("lte-page-subtitle")
            ui.label(
                _translated(
                    translator,
                    "history.trash_descendants",
                    "Direct dependents: {direct} · Indirect dependents: {indirect}",
                    direct=plan.direct_descendant_count,
                    indirect=plan.indirect_descendant_count,
                )
            ).classes("lte-history-summary-item")
            ui.label(
                _translated(translator, "history.trash_roots", "Output roots")
            ).classes("lte-section-title")
            for root in plan.roots:
                ui.label(str(root)).classes("lte-technical-copy")
            with ui.column().classes("lte-trash-impact-runs full-width"):
                for impact in plan.impact_rows:
                    ui.label(
                        _translated(
                            translator,
                            "history.trash_run_summary",
                            (
                                "{scenario} / {profile} · {kind} · {status} · "
                                "{artifacts} artifacts · {size}"
                            ),
                            scenario=impact.scenario_id,
                            profile=impact.profile_id,
                            kind=impact.run_kind,
                            status=impact.status,
                            artifacts=impact.artifact_count,
                            size=_format_bytes(impact.size_bytes),
                        )
                    ).classes("lte-history-summary-item").mark(
                        f"history-trash-impact-run-{impact.run_id}"
                    )
                    ui.label(
                        f"{impact.local_created_at} · {impact.run_id_prefix}"
                    ).classes("lte-history-time")
            ui.label(
                _translated(
                    translator,
                    "history.trash_no_orphans",
                    "All derived runs move together; orphaned runs are not available.",
                )
            ).classes("lte-callout lte-callout--warning")
        with ui.row().classes("lte-action-bar"):
            ui.button(
                _translated(translator, "action.cancel", "Cancel"),
                on_click=getattr(dialog, "close", lambda: None),
            ).props("outline")
            confirm_label = _translated(
                translator,
                (
                    "history.move_family_to_trash"
                    if plan.has_descendants
                    else "history.move_run_to_trash"
                ),
                (
                    "Move {count} related runs to Trash"
                    if plan.has_descendants
                    else "Move run to Trash"
                ),
                count=plan.run_count,
            )
            ui.button(
                confirm_label,
                on_click=lambda: _invoke_opaque(on_confirm, plan),
            ).props("unelevated color=negative").mark("history-trash-confirm")
    open_dialog = getattr(dialog, "open", None)
    if callable(open_dialog):
        open_dialog()
    return dialog


def permanent_delete_matches(card: TrashCard, value: object) -> bool:
    """Return whether typed confirmation exactly matches the ID prefix."""

    return type(value) is str and value == card.transaction_id[:8]


def render_permanent_delete_dialog(
    ui: Any,
    translator: Any,
    card: TrashCard,
    *,
    on_confirm: Callable[..., Any],
) -> Any:
    """Render exact-prefix, irreversible permanent-deletion confirmation."""

    if not isinstance(card, TrashCard):
        raise ValueError("permanent-delete dialog requires a TrashCard")
    holder: dict[str, Any] = {}

    def update_confirmation(event: Any) -> None:
        value = getattr(event, "value", None)
        if value is None:
            sender = getattr(event, "sender", None)
            value = getattr(sender, "value", None)
        holder["confirm"].set_enabled(permanent_delete_matches(card, value))

    with ui.dialog() as dialog, ui.card().classes(
        "lte-confirmation-dialog lte-trash-purge-dialog"
    ):
        ui.label(
            _translated(
                translator,
                "history.trash_permanent_title",
                "Delete permanently",
            )
        ).classes("lte-card-title")
        ui.label(
            _translated(
                translator,
                "history.trash_permanent_body",
                (
                    "This permanently deletes {count} runs ({size}). Restoration "
                    "becomes impossible once deletion starts."
                ),
                count=card.run_count,
                size=_format_bytes(card.size_bytes),
            )
        ).classes("lte-callout lte-callout--error").mark(
            "history-trash-permanent-consequence"
        )
        ui.label(
            _translated(
                translator,
                "history.trash_permanent_prompt",
                "Type the first eight characters of the transaction ID to continue.",
            )
        ).classes("lte-page-subtitle")
        ui.input(
            _translated(
                translator,
                "history.trash_permanent_confirmation",
                "Transaction ID prefix",
            ),
            on_change=update_confirmation,
        ).props("autocomplete=off spellcheck=false").mark(
            "history-trash-permanent-input"
        )
        with ui.row().classes("lte-action-bar"):
            ui.button(
                _translated(translator, "action.cancel", "Cancel"),
                on_click=getattr(dialog, "close", lambda: None),
            ).props("outline")
            confirm = ui.button(
                _translated(
                    translator,
                    "history.trash_permanent_confirm",
                    "Delete permanently",
                ),
                on_click=lambda: _invoke_opaque(on_confirm, card.transaction_id),
            ).props("unelevated color=negative").mark(
                "history-trash-permanent-confirm"
            )
            confirm.set_enabled(False)
            holder["confirm"] = confirm
    open_dialog = getattr(dialog, "open", None)
    if callable(open_dialog):
        open_dialog()
    return dialog


_RESTORE_BLOCKER_COPY: dict[str, tuple[str, str]] = {
    "trash.restore.root_unavailable": (
        "history.trash_root_unavailable",
        "An expected output root is unavailable.",
    ),
    "trash.restore.destination_occupied": (
        "history.trash_destination_occupied",
        "An original destination is occupied.",
    ),
    "trash.restore.lease_conflict": (
        "history.trash_lease_conflict",
        "A family member is open in Figures.",
    ),
    "trash.restore.journal_invalid": (
        "history.trash_journal_invalid",
        "The Trash journal or payload is not actionable.",
    ),
    "trash.restore.state": (
        "history.trash_recovery_required",
        "Recovery is required before this transaction can continue.",
    ),
}


def _trash_action_label(translator: Any, action: TrashAction) -> str:
    presentation = trash_action_presentation(action)
    return _translated(
        translator,
        presentation.label_key,
        {
            TrashAction.RESTORE: "Restore",
            TrashAction.PURGE: "Delete permanently",
            TrashAction.RECOVER: "Recover transaction",
        }[action],
    )


def render_trash_card(
    ui: Any,
    translator: Any,
    card: TrashCard,
    *,
    on_restore: Callable[[str], Any] | None = None,
    on_purge: Callable[[str], Any] | None = None,
    on_recover: Callable[[str], Any] | None = None,
) -> Any:
    """Render one whole-family transaction card with state-gated actions."""

    if not isinstance(card, TrashCard):
        raise ValueError("trash card renderer requires a TrashCard")
    prefix = card.id_prefix
    with ui.card().classes("lte-history-card lte-trash-card").mark(
        f"trash-card-{prefix}"
    ) as container:
        with ui.row().classes("lte-history-card-heading"):
            with ui.column().classes("lte-history-card-identity"):
                ui.label(" · ".join(card.scenario_profiles) or prefix).classes(
                    "lte-card-title"
                )
                ui.label(card.deleted_at).classes("lte-history-time")
            render_status_badge(
                ui,
                translator,
                trash_state_presentation(card.state),
                marker=f"trash-state-{prefix}",
            )
        with ui.row().classes("lte-history-summary-grid"):
            ui.label(f"{card.run_count} runs").classes("lte-history-summary-item")
            ui.label(_format_bytes(card.size_bytes)).classes(
                "lte-history-summary-item"
            )
            ui.label(f"{card.artifact_count} artifacts").classes(
                "lte-history-summary-item"
            )
            ui.label(prefix).classes("lte-history-summary-item")
        for root in card.roots:
            ui.label(str(root)).classes("lte-history-summary-item")

        state = _trash_state(card.state)
        if state is TrashState.RECOVERY_REQUIRED:
            ui.label(
                _translated(
                    translator,
                    "history.trash_recovery_required",
                    "Recovery is required before this transaction can continue.",
                )
            ).classes("lte-callout lte-callout--warning")
        elif state is TrashState.PURGE_FAILED:
            ui.label(
                _translated(
                    translator,
                    "history.trash_purge_failed",
                    (
                        "Permanent deletion stopped after data was removed. Restore "
                        "is no longer available; retry deletion."
                    ),
                )
            ).classes("lte-callout lte-callout--error")

        if card.blockers:
            with ui.column().classes("lte-callout lte-callout--warning").mark(
                f"trash-restore-blockers-{prefix}"
            ):
                for blocker in card.blockers:
                    key, fallback = _RESTORE_BLOCKER_COPY.get(
                        blocker,
                        (
                            "history.trash_journal_invalid",
                            "The Trash transaction is not currently actionable.",
                        ),
                    )
                    ui.label(_translated(translator, key, fallback))

        actions: list[ActionSpec] = []
        for action in card.available_actions:
            callback: Callable[..., Any] | None
            if action is TrashAction.RESTORE:
                callback = on_restore

                def restore_click(
                    handler: Callable[[str], Any] | None = callback,
                    value: str = card.transaction_id,
                ) -> Any:
                    return None if handler is None else handler(value)

                click = restore_click
                role = "primary"
            elif action is TrashAction.RECOVER:
                callback = on_recover

                def recover_click(
                    handler: Callable[[str], Any] | None = callback,
                    value: str = card.transaction_id,
                ) -> Any:
                    return None if handler is None else handler(value)

                click = recover_click
                role = "primary"
            else:
                callback = on_purge

                def purge_click(
                    current: TrashCard = card,
                    handler: Callable[[str], Any] | None = callback,
                ) -> Any:
                    return render_permanent_delete_dialog(
                        ui,
                        translator,
                        current,
                        on_confirm=handler or (lambda _value: None),
                    )

                click = purge_click
                role = "danger"
            actions.append(
                ActionSpec(
                    action.value,
                    _trash_action_label(translator, action),
                    click,
                    role=role,
                    enabled=(
                        action in card.enabled_actions and callback is not None
                    ),
                    marker=f"trash-action-{action.value}-{prefix}",
                )
            )
        if actions:
            render_action_bar(
                ui,
                actions,
                marker=f"trash-actions-{prefix}",
            )

        def render_details() -> None:
            transaction = card.transaction
            if isinstance(transaction, TrashTransaction):
                for member in transaction.members:
                    ui.label(
                        f"{member.identity.run_id} · "
                        f"{member.original_relative_path.as_posix()}"
                    ).classes("lte-technical-copy")
            for diagnostic in card.diagnostics:
                ui.label(diagnostic).classes("lte-technical-copy")

        render_technical_details(
            ui,
            _translated(
                translator,
                "history.trash_technical",
                "Technical details",
            ),
            render_details,
            marker=f"trash-technical-{prefix}",
        )
    return container


def render_trash_frame(
    ui: Any,
    translator: Any,
    count: int,
    *,
    on_back: Callable[[], Any] | None = None,
) -> Any:
    """Render the stable Trash header and return its replaceable holder."""

    actions = (
        ActionSpec(
            "back_history",
            _translated(translator, "action.back_history", "Back to History"),
            on_back,
            marker="trash-back-history",
        ),
    ) if on_back is not None else ()
    with ui.column().classes("lte-page lte-history-page lte-trash-page"):
        render_page_header(
            ui,
            _translated(
                translator,
                "history.trash",
                "Trash ({count})",
                count=count,
            ),
            _translated(
                translator,
                "history.trash_subtitle",
                (
                    "Run families moved here remain restorable until you permanently "
                    "delete them."
                ),
            ),
            actions,
        )
        return ui.column().classes("lte-history-content full-width").mark(
            "trash-content"
        )


def render_trash_content(
    ui: Any,
    translator: Any,
    holder: Any,
    snapshot: TrashSnapshot,
    *,
    on_restore: Callable[[str], Any] | None = None,
    on_purge: Callable[[str], Any] | None = None,
    on_recover: Callable[[str], Any] | None = None,
) -> None:
    """Render authoritative transaction cards and collapsed diagnostics."""

    if not isinstance(snapshot, TrashSnapshot):
        raise ValueError("trash content requires a TrashSnapshot")
    holder.clear()
    with holder:
        if not snapshot.transactions:
            render_empty_state(
                ui,
                _translated(translator, "history.trash_empty", "Trash is empty."),
                _translated(
                    translator,
                    "history.trash_empty_body",
                    "Moved run families will appear here.",
                ),
                marker="trash-empty",
            )
        for card in snapshot.cards:
            render_trash_card(
                ui,
                translator,
                card,
                on_restore=on_restore,
                on_purge=on_purge,
                on_recover=on_recover,
            )
        if snapshot.diagnostics:
            def render_diagnostics() -> None:
                for diagnostic in snapshot.diagnostics:
                    ui.label(
                        f"{diagnostic.code}: {diagnostic.error}"
                    ).classes("lte-technical-copy")

            render_technical_details(
                ui,
                _translated(
                    translator,
                    "history.trash_technical",
                    "Technical details",
                ),
                render_diagnostics,
                marker="trash-diagnostics-technical",
            )


def render_trash_page(
    ui: Any,
    translator: Any,
    snapshot: TrashSnapshot,
    *,
    on_back: Callable[[], Any] | None = None,
    on_restore: Callable[[str], Any] | None = None,
    on_purge: Callable[[str], Any] | None = None,
    on_recover: Callable[[str], Any] | None = None,
) -> TrashSnapshot:
    """Render one complete Trash snapshot; route orchestration remains external."""

    holder = render_trash_frame(
        ui,
        translator,
        snapshot.count,
        on_back=on_back,
    )
    render_trash_content(
        ui,
        translator,
        holder,
        snapshot,
        on_restore=on_restore,
        on_purge=on_purge,
        on_recover=on_recover,
    )
    return snapshot


def render_history_frame(
    ui: Any,
    translator: Any,
    *,
    trash_count: int | None = None,
    on_open_trash: Callable[[], Any] | None = None,
) -> Any:
    """Render the stable History heading and return its replaceable content holder."""

    actions = [
        ActionSpec(
            "refresh",
            _translated(translator, "action.refresh", "Refresh"),
            ui.navigate.reload,
            marker="history-refresh",
        )
    ]
    if trash_count is not None:
        actions.append(
            ActionSpec(
                "trash",
                _translated(
                    translator,
                    "history.trash",
                    "Trash ({count})",
                    count=max(0, int(trash_count)),
                ),
                on_open_trash or (lambda: None),
                enabled=on_open_trash is not None,
                marker="history-trash-link",
            )
        )
    with ui.column().classes("lte-page lte-history-page"):
        render_page_header(
            ui,
            _translated(translator, "history.title", "Run History"),
            _translated(
                translator,
                "history.subtitle",
                "Published selection and derived figure runs from local output roots.",
            ),
            actions,
        )
        return ui.column().classes("lte-history-content full-width").mark(
            "history-content"
        )


def render_history_loading(ui: Any, translator: Any) -> Any:
    """Render History immediately with a durable loading surface."""

    holder = render_history_frame(ui, translator)
    with holder:
        render_loading_state(
            ui,
            _translated(translator, "history.loading", "Loading run history"),
            marker="history-loading",
        )
    return holder


def render_history_error(
    ui: Any,
    translator: Any,
    holder: Any,
    error: Exception,
) -> None:
    """Replace only History content with human recovery and collapsed diagnostics."""

    holder.clear()
    with holder:
        with ui.column().classes("lte-history-load-error lte-callout lte-callout--error"):
            ui.label(
                _translated(
                    translator,
                    "history.load_failed",
                    "Run history could not be loaded.",
                )
            ).classes("lte-section-title").mark("history-load-error")
            ui.label(
                _translated(
                    translator,
                    "history.load_failed_recovery",
                    "Check the configured output folders, then refresh this page.",
                )
            ).classes("lte-page-subtitle")

        def render_error_detail() -> None:
            ui.label(f"{type(error).__name__}: {error}").classes(
                "lte-technical-copy"
            ).mark("history-load-error-technical-copy")

        render_technical_details(
            ui,
            _translated(translator, "history.technical_details", "Technical details"),
            render_error_detail,
            marker="history-load-error-technical",
        )


def render_history_content(
    ui: Any,
    translator: Any,
    holder: Any,
    current_snapshot: HistorySnapshot,
    *,
    on_reveal: Callable[[Path], None] | None = None,
    on_open_figures: Callable[[Path, Path, str, Path, FigureSpec | None], None]
    | None = None,
    on_retry_missing: Callable[
        [Path, Path, str, Path, tuple[str, ...], FigureSpec | None], None
    ]
    | None = None,
    pending_selections: Iterable[Any] = (),
    on_open_pending_figures: Callable[[str], None] | None = None,
    on_continue_pending: Callable[[str], None] | None = None,
    trash_manager: TrashManager | None = None,
    on_move_to_trash: Callable[[HistoryTrashPlan], None] | None = None,
) -> None:
    """Replace History content with one validated snapshot."""

    def notify_action_error(exc: Exception) -> None:
        ui.notify(str(exc), type="negative")

    def reveal(row: HistoryRow) -> None:
        try:
            target = resolve_history_action(row, HistoryAction.REVEAL_DIRECTORY)
            if on_reveal is None:
                raise HistoryActionError("Directory reveal is unavailable in this host")
            on_reveal(target.path)
        except Exception as exc:
            notify_action_error(exc)

    def inspect(row: HistoryRow) -> None:
        try:
            target = resolve_history_action(row, HistoryAction.INSPECT)
        except Exception as exc:
            notify_action_error(exc)
            return
        with ui.dialog() as dialog, ui.card().classes("lte-confirmation-dialog"):
            ui.label(
                _translated(translator, "history.inspect_title", "Run record")
            ).classes("lte-card-title")
            ui.code(
                json.dumps(_thaw(target.record), ensure_ascii=False, indent=2),
                language="json",
            ).classes("w-full lte-technical-copy")
            ui.button(
                _translated(translator, "action.close", "Close"),
                on_click=dialog.close,
            ).props("outline")
        dialog.open()

    def open_figures(row: HistoryRow) -> None:
        try:
            target = resolve_history_action(row, HistoryAction.OPEN_FIGURES)
            if on_open_figures is None:
                raise HistoryActionError("Figure navigation is unavailable")
            if (
                target.destination_root is None
                or target.derived_parent_run_id is None
                or target.derived_parent_path is None
            ):
                raise HistoryActionError("Figure destination lineage is unavailable")
            on_open_figures(
                target.path,
                target.destination_root,
                target.derived_parent_run_id,
                target.derived_parent_path,
                target.figure_spec,
            )
        except Exception as exc:
            notify_action_error(exc)

    def retry_missing(row: HistoryRow) -> None:
        try:
            target = resolve_history_action(row, HistoryAction.RETRY_MISSING)
            if on_retry_missing is None:
                raise HistoryActionError("Artifact retry is unavailable")
            if (
                target.destination_root is None
                or target.derived_parent_run_id is None
                or target.derived_parent_path is None
            ):
                raise HistoryActionError("Figure destination lineage is unavailable")
            on_retry_missing(
                target.path,
                target.destination_root,
                target.derived_parent_run_id,
                target.derived_parent_path,
                target.retry_formats,
                target.figure_spec,
            )
        except Exception as exc:
            notify_action_error(exc)

    def move_to_trash(row: HistoryRow) -> None:
        try:
            if trash_manager is None or on_move_to_trash is None:
                raise HistoryActionError("Trash is unavailable in this host")
            plan = build_history_trash_plan(row, trash_manager)
        except Exception as exc:
            notify_action_error(exc)
            return
        render_move_to_trash_dialog(
            ui,
            translator,
            plan,
            on_confirm=on_move_to_trash,
        )

    pending = tuple(pending_selections)
    holder.clear()
    with holder:
        if pending:
            ui.label(
                _translated(
                    translator,
                    "history.pending_section",
                    "Current confirmed selections",
                )
            ).classes("lte-section-title").mark("history-pending-section")
            with ui.column().classes("lte-history-list lte-history-pending-list"):
                for session in pending:
                    session_id = getattr(session, "session_id", None)
                    profile = getattr(session, "profile_snapshot", None)
                    candidate = getattr(session, "locked_candidate", None)
                    scenario_id = getattr(profile, "scenario_id", None)
                    profile_id = getattr(profile, "profile_id", None)
                    if (
                        type(session_id) is not str
                        or not session_id
                        or type(scenario_id) is not str
                        or not scenario_id
                        or type(profile_id) is not str
                        or not profile_id
                        or candidate is None
                    ):
                        continue
                    with ui.card().classes(
                        "lte-history-card lte-history-card--pending"
                    ).mark(f"history-pending-{session_id}"):
                        with ui.column().classes("lte-history-primary"):
                            with ui.row().classes("lte-history-card-heading"):
                                with ui.column().classes("lte-history-card-identity"):
                                    ui.label(
                                        f"{scenario_id} / {profile_id}"
                                    ).classes("lte-card-title")
                                    ui.label(
                                        _translated(
                                            translator,
                                            "history.pending_candidate",
                                            "Candidate grid {grid_id} · {count} stations",
                                            grid_id=getattr(
                                                candidate,
                                                "flat_grid_id",
                                                "?",
                                            ),
                                            count=getattr(candidate, "point_count", "?"),
                                        )
                                    ).classes("lte-history-time")
                                render_status_badge(
                                    ui,
                                    translator,
                                    PresentationSpec(
                                        "history.pending_status",
                                        "warning",
                                    ),
                                    marker=f"history-pending-status-{session_id}",
                                )
                            ui.label(
                                _translated(
                                    translator,
                                    "history.pending_body",
                                    (
                                        "This selection is available in the current app "
                                        "session. Publish artifacts to make it permanent."
                                    ),
                                )
                            ).classes("lte-history-summary-item")
                            actions: list[ActionSpec] = []
                            if on_open_pending_figures is not None:
                                actions.append(
                                    ActionSpec(
                                        "open",
                                        _translated(
                                            translator,
                                            "action.open_figures",
                                            "Open in Figures",
                                        ),
                                        lambda value=session_id: (
                                            on_open_pending_figures(value)
                                        ),
                                        role="primary",
                                        marker=f"history-pending-open-{session_id}",
                                    )
                                )
                            if on_continue_pending is not None:
                                actions.append(
                                    ActionSpec(
                                        "continue",
                                        _translated(
                                            translator,
                                            "action.continue_generation",
                                            "Continue generation",
                                        ),
                                        lambda value=session_id: (
                                            on_continue_pending(value)
                                        ),
                                        marker=f"history-pending-continue-{session_id}",
                                    )
                                )
                            if actions:
                                render_action_bar(
                                    ui,
                                    actions,
                                    marker=f"history-pending-actions-{session_id}",
                                )

        if current_snapshot.diagnostics:
            ui.label(
                _translated(
                    translator,
                    "history.diagnostic_summary",
                    "{count} run folders need attention.",
                    count=len(current_snapshot.diagnostics),
                )
            ).classes("lte-callout lte-callout--warning")

            def render_diagnostics() -> None:
                for diagnostic in current_snapshot.diagnostics:
                    ui.label(f"{diagnostic.path}: {diagnostic.error}").classes(
                        "lte-technical-copy"
                    )

            render_technical_details(
                ui,
                _translated(
                    translator,
                    "history.diagnostics",
                    "Discovery diagnostics ({count})",
                    count=len(current_snapshot.diagnostics),
                ),
                render_diagnostics,
                marker="history-diagnostics-technical",
            )

        if not current_snapshot.rows and not pending:
            render_empty_state(
                ui,
                _translated(translator, "history.empty", "No published runs were found."),
                _translated(
                    translator,
                    "history.empty_body",
                    "Generated scenario and figure runs will appear here.",
                ),
                marker="history-empty",
            )

        with ui.column().classes("lte-history-list"):
            for row in current_snapshot.rows:
                root_digest = hashlib.sha256(
                    _path_identity(row.root).encode("utf-8")
                ).hexdigest()[:8]
                with ui.card().classes("lte-history-card").mark(
                    f"history-row-{root_digest}-{row.run_id}"
                ):
                    with ui.column().classes("lte-history-primary"):
                        with ui.row().classes("lte-history-card-heading"):
                            with ui.column().classes("lte-history-card-identity"):
                                ui.label(
                                    f"{row.scenario_id} / {row.profile_id}"
                                ).classes("lte-card-title").mark(
                                    f"history-primary-{row.run_id}"
                                )
                                ui.label(row.local_created_at).classes(
                                    "lte-history-time"
                                )
                            render_status_badge(
                                ui,
                                translator,
                                run_state_presentation(row.status),
                                marker=f"history-status-{row.run_id}",
                            )
                        with ui.row().classes("lte-history-summary-grid"):
                            count = len(row.artifacts)
                            ui.label(
                                _translated(
                                    translator,
                                    (
                                        "history.artifact_count.one"
                                        if count == 1
                                        else "history.artifact_count.many"
                                    ),
                                    "{count} artifact" if count == 1 else "{count} artifacts",
                                    count=count,
                                )
                            ).classes("lte-history-summary-item").mark(
                                f"history-artifact-count-{row.run_id}"
                            )
                            ui.label(
                                _translated(
                                    translator,
                                    (
                                        "history.parent_relationship.derived"
                                        if row.parent_run_id
                                        else "history.parent_relationship.original"
                                    ),
                                    (
                                        "Derived from an earlier run"
                                        if row.parent_run_id
                                        else "Original scenario run"
                                    ),
                                )
                            ).classes("lte-history-summary-item").mark(
                                f"history-lineage-{row.run_id}"
                            )
                            if row.errors:
                                ui.label(
                                    _translated(
                                        translator,
                                        (
                                            "history.error_count.one"
                                            if len(row.errors) == 1
                                            else "history.error_count.many"
                                        ),
                                        (
                                            "1 issue recorded"
                                            if len(row.errors) == 1
                                            else "{count} issues recorded"
                                        ),
                                        count=len(row.errors),
                                    )
                                ).classes(
                                    "lte-history-summary-item lte-history-summary-item--warning"
                                ).mark(f"history-error-summary-{row.run_id}")
                        actions = [
                            ActionSpec(
                                "inspect",
                                _translated(translator, "action.inspect", "Inspect"),
                                lambda current=row: inspect(current),
                                role="primary",
                                marker=f"history-inspect-{row.run_id}",
                            )
                        ]
                        if row.can_open_figures:
                            actions.append(
                                ActionSpec(
                                    "open",
                                    _translated(
                                        translator,
                                        "action.open_figures",
                                        "Open in Figures",
                                    ),
                                    lambda current=row: open_figures(current),
                                    role="primary",
                                    marker=f"history-open-{row.run_id}",
                                )
                            )
                        if row.can_retry_missing:
                            actions.append(
                                ActionSpec(
                                    "retry",
                                    _translated(
                                        translator,
                                        "action.retry_missing",
                                        "Retry Missing Artifacts",
                                    ),
                                    lambda current=row: retry_missing(current),
                                    marker=f"history-retry-{row.run_id}",
                                )
                            )
                        actions.append(
                            ActionSpec(
                                "reveal",
                                _translated(
                                    translator,
                                    "action.reveal_directory",
                                    "Reveal Directory",
                                ),
                                lambda current=row: reveal(current),
                                marker=f"history-reveal-{row.run_id}",
                            )
                        )
                        render_action_bar(
                            ui,
                            actions,
                            marker=f"history-actions-{row.run_id}",
                        )
                        if HistoryAction.MOVE_TO_TRASH in row.available_actions:
                            render_action_bar(
                                ui,
                                (
                                    ActionSpec(
                                        "move_to_trash",
                                        _translated(
                                            translator,
                                            "action.move_to_trash",
                                            "Move to Trash",
                                        ),
                                        lambda current=row: move_to_trash(current),
                                        role="danger",
                                        marker=f"history-trash-move-{row.run_id}",
                                    ),
                                ),
                                marker=f"history-trash-overflow-{row.run_id}",
                            )

                    technical_payload = {
                        "run_id": row.run_id,
                        "parent_run_id": row.parent_run_id,
                        "root": str(row.root),
                        "path": str(row.path),
                        "artifacts": row.artifacts,
                        "missing_artifacts": row.missing_artifacts,
                        "parameters": _thaw(row.parameters),
                        "candidate": _thaw(row.candidate),
                        "errors": _thaw(row.errors),
                        "record": _thaw(row.record),
                    }

                    def render_row_details(
                        payload: dict[str, Any] = technical_payload,
                        run_id: str = row.run_id,
                    ) -> None:
                        ui.code(
                            json.dumps(payload, ensure_ascii=False, indent=2),
                            language="json",
                        ).classes("lte-technical-copy").mark(
                            f"history-technical-copy-{run_id}"
                        )

                    render_technical_details(
                        ui,
                        _translated(
                            translator,
                            "history.technical_details",
                            "Technical details",
                        ),
                        render_row_details,
                        marker=f"history-technical-{row.run_id}",
                    )


def render_history_page(
    ui: Any,
    translator: Any,
    repo_root: str | os.PathLike[str],
    output_roots: Iterable[str | os.PathLike[str]],
    *,
    index_path: str | os.PathLike[str] | None = None,
    snapshot: HistorySnapshot | None = None,
    on_reveal: Callable[[Path], None] | None = None,
    on_open_figures: Callable[[Path, Path, str, Path, FigureSpec | None], None]
    | None = None,
    on_retry_missing: Callable[
        [Path, Path, str, Path, tuple[str, ...], FigureSpec | None], None
    ]
    | None = None,
    pending_selections: Iterable[Any] = (),
    on_open_pending_figures: Callable[[str], None] | None = None,
    on_continue_pending: Callable[[str], None] | None = None,
    trash_manager: TrashManager | None = None,
    on_move_to_trash: Callable[[HistoryTrashPlan], None] | None = None,
    trash_count: int | None = None,
    on_open_trash: Callable[[], Any] | None = None,
) -> HistorySnapshot:
    """Synchronously rebuild and render History from live run manifests."""

    if snapshot is None:
        current_snapshot = rebuild_history(
            repo_root,
            output_roots,
            index_path=index_path,
        )
    else:
        repository = _canonical_root(repo_root)
        expected_roots = history_roots(repository, output_roots)
        expected_index = _requested_index_path(repository, index_path)
        if snapshot.roots != expected_roots or snapshot.index_path != expected_index:
            raise ValueError("history snapshot does not match this page request")
        current_snapshot = snapshot

    holder = render_history_frame(
        ui,
        translator,
        trash_count=trash_count,
        on_open_trash=on_open_trash,
    )
    render_history_content(
        ui,
        translator,
        holder,
        current_snapshot,
        on_reveal=on_reveal,
        on_open_figures=on_open_figures,
        on_retry_missing=on_retry_missing,
        pending_selections=pending_selections,
        on_open_pending_figures=on_open_pending_figures,
        on_continue_pending=on_continue_pending,
        trash_manager=trash_manager,
        on_move_to_trash=on_move_to_trash,
    )
    return current_snapshot


__all__ = [
    "HISTORY_INDEX_RELATIVE_PATH",
    "HistoryAction",
    "HistoryActionError",
    "HistoryDiagnostic",
    "HistoryRow",
    "HistoryRunReference",
    "HistorySnapshot",
    "HistoryTrashPlan",
    "ResolvedHistoryAction",
    "TRASH_ACTIONS_BY_STATE",
    "TrashAction",
    "TrashCard",
    "TrashImpactRow",
    "TrashSnapshot",
    "build_history_trash_plan",
    "build_trash_cards",
    "build_trash_snapshot",
    "confirm_history_trash_plan",
    "figure_source_options",
    "history_roots",
    "history_rows",
    "move_history_to_trash",
    "permanent_delete_matches",
    "rebuild_history",
    "render_history_content",
    "render_history_error",
    "render_history_frame",
    "render_history_loading",
    "render_history_page",
    "render_move_to_trash_dialog",
    "render_permanent_delete_dialog",
    "render_trash_card",
    "render_trash_content",
    "render_trash_frame",
    "render_trash_page",
    "resolve_history_action",
    "resolve_history_reference",
    "resolve_history_trash_plan",
    "trash_card_actions",
    "validate_history_trash_plan",
]
