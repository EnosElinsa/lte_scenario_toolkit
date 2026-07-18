"""Rebuildable, non-authoritative run history models and rendering helpers."""

from __future__ import annotations

import hashlib
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
from ..presentation import (
    ActionSpec,
    render_action_bar,
    render_empty_state,
    render_loading_state,
    render_page_header,
    render_status_badge,
    render_technical_details,
    run_state_presentation,
)

HISTORY_INDEX_RELATIVE_PATH = Path(".lte-data") / "cache" / "history-index.json"


class HistoryAction(str, Enum):
    """The deliberately non-destructive actions exposed by the first GUI release."""

    REVEAL_DIRECTORY = "reveal_directory"
    INSPECT = "inspect"
    OPEN_FIGURES = "open_figures"
    RETRY_MISSING = "retry_missing"
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
        return tuple(actions)


@dataclass(frozen=True, slots=True)
class HistorySnapshot:
    """A fresh multi-root discovery plus its non-actionable diagnostics."""

    roots: tuple[Path, ...]
    rows: tuple[HistoryRow, ...]
    diagnostics: tuple[HistoryDiagnostic, ...]
    index_path: Path


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


def render_history_frame(ui: Any, translator: Any) -> Any:
    """Render the stable History heading and return its replaceable content holder."""

    with ui.column().classes("lte-page lte-history-page"):
        render_page_header(
            ui,
            _translated(translator, "history.title", "Run History"),
            _translated(
                translator,
                "history.subtitle",
                "Published selection and derived figure runs from local output roots.",
            ),
            (
                ActionSpec(
                    "refresh",
                    _translated(translator, "action.refresh", "Refresh"),
                    ui.navigate.reload,
                    marker="history-refresh",
                ),
            ),
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

    holder.clear()
    with holder:
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

        if not current_snapshot.rows:
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

    holder = render_history_frame(ui, translator)
    render_history_content(
        ui,
        translator,
        holder,
        current_snapshot,
        on_reveal=on_reveal,
        on_open_figures=on_open_figures,
        on_retry_missing=on_retry_missing,
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
    "ResolvedHistoryAction",
    "history_roots",
    "history_rows",
    "rebuild_history",
    "render_history_content",
    "render_history_error",
    "render_history_frame",
    "render_history_loading",
    "render_history_page",
    "resolve_history_action",
    "resolve_history_reference",
]
