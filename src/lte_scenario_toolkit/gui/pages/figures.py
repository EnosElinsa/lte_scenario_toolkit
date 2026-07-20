"""Explicit preview and final terrain-figure workflows for the local GUI."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from matplotlib import colormaps

from ... import io
from ...figure_service import (
    FigureService,
    FigureSource,
    FigureSpec,
    validate_csv_identity,
)
from ...jobs import Job, JobBusyError, JobCoordinator
from ...run_service import RunService
from ...run_trash import RunIdentity, RunUsageLeaseRegistry
from ..presentation import (
    ActionSpec,
    MenuActionSpec,
    render_action_bar,
    render_overflow_menu,
    render_technical_details,
)

PREVIEW_CACHE_VERSION = "figure-preview-v3"
PREVIEW_LIMITS = {
    "preview": {"dpi": 120, "max_pixels": 600},
    "publication": {"dpi": 180, "max_pixels": 900},
}
FIGURE_FORMAT_ORDER = ("png", "eps", "html")
STATION_COLOR_OPTIONS = {
    "red": "Red",
    "crimson": "Crimson",
    "darkorange": "Orange",
    "gold": "Gold",
    "forestgreen": "Green",
    "teal": "Teal",
    "royalblue": "Blue",
    "navy": "Navy",
    "purple": "Purple",
    "black": "Black",
    "white": "White",
}


def _colormap_options(current: str) -> dict[str, str]:
    options = {name: name for name in sorted(colormaps)}
    options.setdefault(current, current)
    return options


def _station_color_options(current: str) -> dict[str, str]:
    options = dict(STATION_COLOR_OPTIONS)
    options.setdefault(current, current)
    return options


def _source_options(
    values: Mapping[str, str] | None,
    initial_source: str | Path | None,
) -> dict[str, str]:
    options: dict[str, str] = {}
    for raw_path, raw_label in (values or {}).items():
        path = str(Path(raw_path).expanduser().resolve(strict=False))
        label = str(raw_label).strip()
        options[path] = label or path
    if initial_source is not None:
        path = str(Path(initial_source).expanduser().resolve(strict=False))
        options.setdefault(path, path)
    return options


def load_figure_source(path: str | Path) -> FigureSource:
    """Load one completed current run as a figure source."""

    return FigureService.load_source(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _source_cache_identity(source: FigureSource) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "kind": source.source_kind,
        "rect_id": _jsonable(source.rectangle.get("rect_id")),
        "target_crs": source.target_crs,
        "rectangle_size_m": source.rectangle_size_m,
    }
    if source.selection_identity is not None:
        identity["selection"] = source.selection_identity.as_dict()
    elif source.csv_path is not None:
        csv_identity = validate_csv_identity(source)
        identity["csv"] = {
            "path": str(csv_identity.path),
            "size_bytes": csv_identity.size_bytes,
            "mtime_ns": csv_identity.mtime_ns,
            "sha256": csv_identity.sha256,
        }
    else:
        raise ValueError("figure source has no stable identity")
    if source.dem_path is None:
        identity["dem"] = None
    else:
        dem_path = source.dem_path
        if dem_path.is_symlink() or not dem_path.is_file():
            raise ValueError("figure source DEM must be a regular file")
        resolved_dem = dem_path.resolve(strict=True)
        dem_stat = resolved_dem.stat()
        identity["dem"] = {
            "path": str(resolved_dem),
            "size_bytes": dem_stat.st_size,
            "mtime_ns": dem_stat.st_mtime_ns,
            "fingerprint": source.dem_fingerprint or io.sha256_file(resolved_dem),
        }
    return identity


def preview_cache_path(
    repo_root: str | os.PathLike[str],
    source: FigureSource,
    spec: FigureSpec,
) -> Path:
    """Return a deterministic cache filename without creating its parent."""

    if not isinstance(source, FigureSource):
        raise ValueError("source must be a FigureSource")
    if not isinstance(spec, FigureSpec):
        raise ValueError("spec must be a FigureSpec")
    spec.validate()
    payload = {
        "version": PREVIEW_CACHE_VERSION,
        "source": _source_cache_identity(source),
        "spec": spec.as_dict(),
    }
    encoded = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()
    raw_root = Path(repo_root).expanduser()
    root_is_junction = getattr(raw_root, "is_junction", None)
    if raw_root.is_symlink() or bool(
        root_is_junction is not None and root_is_junction()
    ):
        raise ValueError("preview cache repository root must not be redirected")
    root = raw_root.resolve(strict=False)
    destination = root / ".lte-data" / "cache" / "previews" / f"{key}.png"
    current = root
    for part in destination.relative_to(root).parts[:-1]:
        current /= part
        is_junction = getattr(current, "is_junction", None)
        redirected = current.is_symlink() or bool(
            is_junction is not None and is_junction()
        )
        if os.path.lexists(current) and redirected:
            raise ValueError(f"preview cache parent must not be redirected: {current}")
        if os.path.lexists(current) and not current.is_dir():
            raise ValueError(f"preview cache parent must be a directory: {current}")
    return destination


def _valid_preview(path: Path) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 8:
        return False
    try:
        if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            return False
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError):
        return False
    return True


def preview_spec(spec: FigureSpec) -> FigureSpec:
    """Bound every GUI preview while preserving the selected visual style."""

    spec.validate()
    limits = PREVIEW_LIMITS[spec.preset]
    return replace(
        spec,
        dpi=min(spec.dpi, limits["dpi"]),
        max_pixels=min(spec.max_pixels, limits["max_pixels"]),
    )


def _positive_integer_control(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        raise ValueError(f"{field} must be a positive integer")
    return int(numeric)


@dataclass(frozen=True, slots=True)
class FigurePageState:
    """Immutable source, style, preview, and publication state."""

    source: Any | None = None
    spec: FigureSpec = FigureSpec.from_preset("preview")
    preview_path: Path | None = None
    preview_stale: bool = True
    revision: int = 0
    phase: str = "empty"
    run_path: Path | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[dict[str, Any], ...] = ()
    source_dirty: bool = False
    source_error: str | None = None

    @classmethod
    def for_source(cls, source: Any) -> FigurePageState:
        return cls(
            source=source,
            phase="ready",
            warnings=tuple(getattr(source, "warnings", ())),
        )

    def with_source(self, source: Any) -> FigurePageState:
        return replace(
            self,
            source=source,
            preview_path=None,
            preview_stale=True,
            revision=self.revision + 1,
            phase="ready",
            run_path=None,
            warnings=tuple(getattr(source, "warnings", ())),
            errors=(),
            source_dirty=False,
            source_error=None,
        )

    def with_spec(self, spec: FigureSpec) -> FigurePageState:
        if not isinstance(spec, FigureSpec):
            raise ValueError("spec must be a FigureSpec")
        spec.validate()
        return replace(
            self,
            spec=spec,
            preview_stale=True,
            revision=self.revision + 1,
            phase=(
                "ready"
                if self.source is not None and not self.source_dirty
                else "error"
                if self.source_error is not None
                else "empty"
            ),
            errors=(),
        )

    def with_dpi(self, dpi: int) -> FigurePageState:
        return self.with_spec(replace(self.spec, dpi=dpi))


@dataclass(frozen=True, slots=True)
class _FigureJobResult:
    kind: str
    revision: int
    source: FigureSource | None = None
    path: Path | None = None
    phase: str = "ready"
    warnings: tuple[str, ...] = ()
    errors: tuple[dict[str, Any], ...] = ()
    message: str | None = None
    output_root: Path | None = None


@dataclass(frozen=True, slots=True)
class _FigureExportRequest:
    source: FigureSource
    spec: FigureSpec
    output_root: Path
    requested: tuple[str, ...]
    parent_run_id: str | None
    revision: int
    repository: Path


def _render_figure_export(request: _FigureExportRequest) -> _FigureJobResult:
    """Render and validate one export in a process-safe, UI-free function."""

    source = request.source
    try:
        path = FigureService.render(
            source,
            request.spec,
            RunService(request.output_root),
            request.requested,
            parent_run_id=request.parent_run_id,
            entrypoint=("lte-gui", "figures"),
            repository=request.repository,
        )
        entry = RunService(request.output_root).entry_for_path(path)
        record = entry.record
        metadata = record.get("metadata")
        if record.get("scenario_id") != (source.scenario_id or "figures"):
            raise ValueError("figure run scenario does not match its source")
        if record.get("profile_id") != (source.profile_id or "figures"):
            raise ValueError("figure run profile does not match its source")
        if record.get("parent_run_id") != request.parent_run_id:
            raise ValueError("figure run parent does not match its source")
        if not isinstance(metadata, Mapping) or metadata.get("run_kind") != "figure":
            raise ValueError("published run is not a figure run")
        recorded_formats = metadata.get("requested_formats")
        if not isinstance(recorded_formats, (list, tuple)) or tuple(
            recorded_formats
        ) != request.requested:
            raise ValueError("figure run formats do not match the request")
        recorded_spec = metadata.get("figure_spec")
        if not isinstance(recorded_spec, Mapping) or dict(
            recorded_spec
        ) != request.spec.as_dict():
            raise ValueError("figure run style does not match the request")
        source_metadata = metadata.get("source")
        if not isinstance(source_metadata, Mapping):
            raise ValueError("figure run is missing source metadata")
        if source_metadata.get("run_id") != source.run_id:
            raise ValueError("figure run source ID does not match")
        if source_metadata.get("kind") != source.source_kind:
            raise ValueError("figure run source kind does not match")
        if source_metadata.get("path") != (
            str(source.path) if source.path is not None else None
        ):
            raise ValueError("figure run source path does not match")
        if source_metadata.get("csv") != (
            str(source.csv_path) if source.csv_path is not None else None
        ):
            raise ValueError("figure run source CSV does not match")
        expected_selection = (
            source.selection_identity.as_dict()
            if source.selection_identity is not None
            else None
        )
        if source_metadata.get("selection") != expected_selection:
            raise ValueError("figure run selection identity does not match")
        artifact_paths = metadata.get("artifact_paths")
        if not isinstance(artifact_paths, Mapping):
            raise ValueError("figure run artifact paths are missing")
        artifacts = tuple(record.get("artifacts", ()))
        if source_metadata.get("artifact") != "source.csv":
            raise ValueError("figure run source snapshot is not recorded")
        if "source.csv" not in artifacts:
            raise ValueError("figure run source snapshot is missing")
        if any(relative not in artifacts for relative in artifact_paths.values()):
            raise ValueError("figure run artifact paths are inconsistent")
        if len(set(artifact_paths.values())) != len(artifact_paths):
            raise ValueError("figure run artifact paths must be unique")
        successful = set(artifact_paths)
        if not successful or not successful.issubset(set(request.requested)):
            raise ValueError("figure run has invalid successful formats")
        phase = str(entry.record["status"])
        if phase == "completed" and successful != set(request.requested):
            raise ValueError("completed figure run is missing formats")
        failed_tokens = {
            item.get("artifact")
            for item in record.get("errors", ())
            if isinstance(item, Mapping)
        }
        if phase == "partial" and failed_tokens != set(request.requested) - successful:
            raise ValueError("partial figure run errors do not match missing formats")
        if phase == "completed" and failed_tokens:
            raise ValueError("completed figure run must not contain errors")
        if phase == "partial" and successful == set(request.requested):
            raise ValueError("partial figure run must have a failed format")
        errors = tuple(
            dict(item)
            for item in entry.record["errors"]
            if isinstance(item, Mapping)
        )
        return _FigureJobResult(
            "export",
            request.revision,
            path=entry.run_dir,
            phase=phase,
            warnings=source.warnings,
            errors=errors,
        )
    except Exception as exc:
        return _FigureJobResult(
            "export",
            request.revision,
            phase="error",
            errors=(
                {"code": "figure.export.failed", "message": str(exc)},
            ),
            message=str(exc),
        )


class FigureController:
    """Coordinate source preparation, explicit preview, and final publication."""

    def __init__(
        self,
        repo_root: str | os.PathLike[str],
        coordinator: JobCoordinator,
        *,
        source: FigureSource | None = None,
        spec: FigureSpec | None = None,
        output_root: str | os.PathLike[str] | None = None,
        parent_run_id: str | None = None,
        parent_run_path: str | os.PathLike[str] | None = None,
        on_published: Callable[[Path], None] | None = None,
        usage_leases: RunUsageLeaseRegistry | None = None,
        run_roots: Callable[[], Iterable[Path]] | None = None,
        lease_owner: str | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve(strict=False)
        self.coordinator = coordinator
        if usage_leases is not None and not isinstance(
            usage_leases,
            RunUsageLeaseRegistry,
        ):
            raise ValueError("usage_leases must be a RunUsageLeaseRegistry")
        if run_roots is not None and not callable(run_roots):
            raise ValueError("run_roots must be callable")
        if lease_owner is not None and (
            type(lease_owner) is not str or not lease_owner.strip()
        ):
            raise ValueError("lease_owner must be non-empty text")
        self.usage_leases = usage_leases
        self.run_roots = run_roots
        self.lease_owner = lease_owner or f"figures:{uuid4().hex}"
        self._usage_lease_id: str | None = None
        self.output_root = (
            None
            if output_root is None
            else Path(output_root).expanduser().resolve(strict=False)
        )
        self.on_published = on_published
        self.parent_run_id = parent_run_id
        self.parent_run_path = (
            None
            if parent_run_path is None
            else Path(parent_run_path).resolve(strict=False)
        )
        self._lock = Lock()
        initial_spec = FigureSpec.from_preset("preview") if spec is None else spec.validate()
        initial_state = (
            FigurePageState(revision=-1)
            if source is not None
            else FigurePageState()
        )
        self._state = replace(initial_state, spec=initial_spec)
        self._job: Job | None = None
        self._closed = False
        self._close_requested = False
        if source is not None:
            self.set_source(
                source,
                output_root=output_root,
                parent_run_id=parent_run_id,
                parent_run_path=parent_run_path,
            )

    @property
    def state(self) -> FigurePageState:
        with self._lock:
            return self._state

    @property
    def job(self) -> Job | None:
        with self._lock:
            return self._job

    def _job_unfinished_locked(self) -> bool:
        return (
            self._job is not None
            and (
                self._job.future is None
                or not self._job.future.done()
            )
        )

    @property
    def has_unfinished_job(self) -> bool:
        """Return whether this controller still owns active background work."""

        with self._lock:
            return self._job_unfinished_locked()

    def _set_state(self, state: FigurePageState) -> FigurePageState:
        with self._lock:
            self._state = state
            return state

    def _configured_run_roots(self) -> tuple[Path, ...]:
        if self.run_roots is None:
            raise ValueError(
                "figure run roots are unavailable; reopen the source from History"
            )
        try:
            supplied = tuple(self.run_roots())
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                "figure run roots are unavailable; reopen the source from History"
            ) from exc
        canonical: dict[str, Path] = {}
        for value in supplied:
            try:
                root = RunService(value).output_root
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "figure run roots are unavailable; reopen the source from History"
                ) from exc
            canonical.setdefault(os.path.normcase(str(root)), root)
        return tuple(canonical[key] for key in sorted(canonical))

    def _source_usage_identities(
        self,
        source: FigureSource,
        *,
        output_root: Path | None,
        parent_run_id: str | None,
        parent_run_path: Path | None,
    ) -> tuple[RunIdentity, ...]:
        if self.usage_leases is None:
            return ()
        needs_roots = (
            source.source_kind == "run"
            or parent_run_id is not None
            or parent_run_path is not None
        )
        roots = self._configured_run_roots() if needs_roots else ()
        identities: list[RunIdentity] = []
        if source.source_kind == "run":
            if source.path is None or source.run_id is None:
                raise ValueError(
                    "figure source is no longer available; reopen it from History"
                )
            matches = []
            for root in roots:
                try:
                    entry = RunService(root).entry_for_path(
                        source.path,
                        run_id=source.run_id,
                    )
                except (OSError, RuntimeError, ValueError):
                    continue
                matches.append(entry)
            if len(matches) != 1:
                raise ValueError(
                    "figure source is no longer uniquely available; "
                    "reopen it from History"
                )
            identities.append(RunIdentity.from_entry(matches[0]))

        if parent_run_path is not None and parent_run_id is None:
            raise ValueError("figure parent path requires a parent run id")
        if parent_run_id is not None:
            if output_root is None or parent_run_path is None:
                raise ValueError(
                    "figure parent run is no longer available; reopen from History"
                )
            parent_root = RunService(output_root).output_root
            if parent_root not in roots:
                raise ValueError(
                    "figure parent run root is no longer configured; reopen from History"
                )
            try:
                parent_entry = RunService(parent_root).entry_for_path(
                    parent_run_path,
                    run_id=parent_run_id,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise ValueError(
                    "figure parent run is no longer available; reopen from History"
                ) from exc
            identities.append(RunIdentity.from_entry(parent_entry))

        return tuple(dict.fromkeys(identities))

    def _replace_usage_lease(self, identities: tuple[RunIdentity, ...]) -> None:
        old = self._usage_lease_id
        new = None
        if self.usage_leases is not None and identities:
            new = self.usage_leases.acquire(identities, self.lease_owner)
        self._usage_lease_id = new
        if old is not None and self.usage_leases is not None:
            self.usage_leases.release(old)

    def _discard_source_locked(self, error: str | None = None) -> FigurePageState:
        self._replace_usage_lease(())
        self.output_root = None
        self.parent_run_id = None
        self.parent_run_path = None
        self._state = FigurePageState(
            spec=self._state.spec,
            revision=self._state.revision + 1,
            phase="error" if error is not None else "empty",
            source_dirty=error is not None,
            source_error=error,
        )
        return self._state

    def set_source(
        self,
        source: FigureSource,
        *,
        output_root: str | os.PathLike[str] | None = None,
        parent_run_id: str | None = None,
        parent_run_path: str | os.PathLike[str] | None = None,
    ) -> FigurePageState:
        if not isinstance(source, FigureSource):
            raise ValueError("source must be a FigureSource")
        next_output_root = (
            None
            if output_root is None
            else Path(output_root).expanduser().resolve(strict=False)
        )
        next_parent_path = (
            None
            if parent_run_path is None
            else Path(parent_run_path).expanduser().resolve(strict=False)
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("figure controller is closed")
            if self._job_unfinished_locked():
                raise RuntimeError("figure source cannot change while a job is running")
        try:
            identities = self._source_usage_identities(
                source,
                output_root=next_output_root,
                parent_run_id=parent_run_id,
                parent_run_path=next_parent_path,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            with self._lock:
                if not self._closed and not self._job_unfinished_locked():
                    self._discard_source_locked(str(exc))
            raise
        with self._lock:
            if self._closed:
                raise RuntimeError("figure controller is closed")
            if self._job_unfinished_locked():
                raise RuntimeError("figure source cannot change while a job is running")
            try:
                self._replace_usage_lease(identities)
            except (RuntimeError, ValueError) as exc:
                self._discard_source_locked(str(exc))
                raise
            self.output_root = next_output_root
            self.parent_run_id = parent_run_id
            self.parent_run_path = next_parent_path
            self._state = self._state.with_source(source)
            return self._state

    def update_spec(self, spec: FigureSpec) -> FigurePageState:
        with self._lock:
            if self._closed:
                raise RuntimeError("figure controller is closed")
            if self._job_unfinished_locked():
                raise RuntimeError("figure style cannot change while a job is running")
            self._state = self._state.with_spec(spec)
            return self._state

    def clear_source(self) -> FigurePageState:
        with self._lock:
            if self._closed:
                raise RuntimeError("figure controller is closed")
            if self._job_unfinished_locked():
                raise RuntimeError("figure source cannot change while a job is running")
            return self._discard_source_locked()

    def invalidate_source(self, error: str | None = None) -> FigurePageState:
        """Atomically discard every value derived from an unconfirmed source."""

        if error is not None and (type(error) is not str or not error.strip()):
            raise ValueError("source error must be non-empty text")
        with self._lock:
            if self._closed:
                raise RuntimeError("figure controller is closed")
            if self._job_unfinished_locked():
                raise RuntimeError("figure source cannot change while a job is running")
            state = self._discard_source_locked(error)
            if error is None:
                state = replace(state, source_dirty=True)
                self._state = state
            return state

    def _submit(
        self,
        kind: str,
        worker: Callable[[Any, Callable[[Any], None]], _FigureJobResult],
    ) -> Job:
        with self._lock:
            if self._closed:
                raise RuntimeError("figure controller is closed")
        job = self.coordinator.submit(kind, worker)
        with self._lock:
            self._job = job
        assert job.future is not None

        def finish_background_job(_future: Any) -> None:
            self.coordinator.finish(job.job_id)
            self._finalize_deferred_close()

        job.future.add_done_callback(finish_background_job)
        return job

    def _finalize_deferred_close(self) -> None:
        """Release page-owned leases only after background work is finished."""

        with self._lock:
            if not self._close_requested or self._job_unfinished_locked():
                return
            self._closed = True
            self._close_requested = False
            self._replace_usage_lease(())
            self._job = None

    def _reserve(self, kind: str) -> Job:
        with self._lock:
            if self._closed:
                raise RuntimeError("figure controller is closed")
        job = self.coordinator.start(kind)
        with self._lock:
            self._job = job
        return job

    def prepare_selection(self, session: Any) -> Job:
        revision = self.state.revision + 1
        service = session.selection_service
        preflight = session.preflight
        scan_result = session.scan_result
        candidate = session.locked_candidate
        output_root = Path(preflight.output_root).resolve(strict=False)

        def worker(_cancel: Any, _emit: Callable[[Any], None]) -> _FigureJobResult:
            try:
                source = service.prepare_figure_source(
                    preflight,
                    scan_result,
                    candidate,
                )
                return _FigureJobResult(
                    "source",
                    revision,
                    source=source,
                    warnings=source.warnings,
                    output_root=output_root,
                )
            except Exception as exc:
                return _FigureJobResult(
                    "source",
                    revision,
                    phase="error",
                    errors=(
                        {
                            "code": getattr(exc, "code", "figure.source.failed"),
                            "message": str(exc),
                        },
                    ),
                    message=str(exc),
                    output_root=output_root,
                )

        job = self._submit("figure-source", worker)
        self._set_state(replace(self.state, phase="loading", revision=revision))
        return job

    def refresh_preview(self) -> Job | None:
        state = self.state
        source = state.source
        if not isinstance(source, FigureSource):
            raise ValueError("Choose a figure source before refreshing preview")
        if source.dem_path is None:
            raise ValueError("Choose a DEM before refreshing preview")
        source_snapshot = source.snapshot()
        spec = preview_spec(state.spec)
        revision = state.revision

        def worker(_cancel: Any, _emit: Callable[[Any], None]) -> _FigureJobResult:
            try:
                destination = preview_cache_path(
                    self.repo_root,
                    source_snapshot,
                    spec,
                )
                if destination.is_symlink():
                    raise ValueError("preview cache path must not be a symlink")
                if _valid_preview(destination):
                    return _FigureJobResult("preview", revision, path=destination)
                if destination.exists():
                    if not destination.is_file():
                        raise ValueError("preview cache path must be a regular file")
                    destination.unlink()
                rendered = FigureService.preview(source_snapshot, spec, destination)
                if not _valid_preview(rendered):
                    raise ValueError("rendered preview is not a valid PNG")
                return _FigureJobResult("preview", revision, path=rendered)
            except Exception as exc:
                return _FigureJobResult(
                    "preview",
                    revision,
                    phase="error",
                    errors=(
                        {"code": "figure.preview.failed", "message": str(exc)},
                    ),
                    message=str(exc),
                )

        job = self._submit("figure-preview", worker)
        self._set_state(replace(state, phase="previewing", errors=()))
        return job

    @staticmethod
    def _normalise_formats(formats: Iterable[str]) -> tuple[str, ...]:
        if isinstance(formats, (str, bytes, os.PathLike)):
            raise ValueError("formats must be a collection")
        values = tuple(formats)
        if any(type(value) is not str or value not in FIGURE_FORMAT_ORDER for value in values):
            raise ValueError("figure format must be png, eps, or html")
        if len(set(values)) != len(values):
            raise ValueError("figure formats must not contain duplicates")
        selected = set(values)
        ordered = tuple(token for token in FIGURE_FORMAT_ORDER if token in selected)
        if not ordered:
            raise ValueError("At least one figure format must be selected")
        return ordered

    def _target(self, source: FigureSource) -> tuple[Path, str | None]:
        output_root = self.output_root
        parent_run_id = self.parent_run_id
        if source.source_kind == "run" and source.path is not None and source.run_id:
            try:
                inferred = source.path.parents[2]
                entry = RunService(inferred).entry_for_path(
                    source.path,
                    run_id=source.run_id,
                )
            except (IndexError, OSError, ValueError):
                entry = None
            if entry is not None and entry.run_id == source.run_id:
                if output_root is None:
                    output_root = entry.root
                if parent_run_id is None and output_root.resolve(strict=False) == entry.root:
                    parent_run_id = entry.run_id
        if output_root is None:
            if source.csv_path is None:
                raise ValueError("Choose an output root for the current selection")
            output_root = source.csv_path.parent / "figure-runs"
        if self.parent_run_path is not None and parent_run_id is None:
            raise ValueError("figure parent path requires a parent run id")
        if parent_run_id is not None and self.parent_run_path is not None:
            try:
                parent_entry = RunService(output_root).entry_for_path(
                    self.parent_run_path,
                    run_id=parent_run_id,
                )
            except (OSError, ValueError) as exc:
                raise ValueError(
                    "figure parent run is no longer available; reopen from History"
                ) from exc
            if parent_entry.run_id != parent_run_id:
                raise ValueError("figure parent run changed; reopen from History")
        return output_root.resolve(strict=False), parent_run_id

    def _export_request(self, formats: Iterable[str]) -> _FigureExportRequest:
        state = self.state
        source = state.source
        if not isinstance(source, FigureSource):
            raise ValueError("Choose a figure source before final export")
        if source.dem_path is None:
            raise ValueError("Choose a DEM before final export")
        requested = self._normalise_formats(formats)
        source_snapshot = source.snapshot()
        spec = state.spec.validate()
        output_root, parent_run_id = self._target(source_snapshot)
        return _FigureExportRequest(
            source=source_snapshot,
            spec=spec,
            output_root=output_root,
            requested=requested,
            parent_run_id=parent_run_id,
            revision=state.revision,
            repository=self.repo_root,
        )

    def export(self, formats: Iterable[str]) -> Job:
        request = self._export_request(formats)

        def worker(_cancel: Any, _emit: Callable[[Any], None]) -> _FigureJobResult:
            return _render_figure_export(request)

        job = self._submit("figure-export", worker)
        self._set_state(replace(self.state, phase="exporting", errors=()))
        return job

    def reserve_cpu_export(
        self,
        formats: Iterable[str],
    ) -> tuple[Job, _FigureExportRequest]:
        request = self._export_request(formats)
        job = self._reserve("figure-export")
        self._set_state(replace(self.state, phase="exporting", errors=()))
        return job, request

    def drain(self, job: Job | None = None) -> FigurePageState:
        active = self.job if job is None else job
        if active is None or active.future is None or not active.future.done():
            return self.state
        result = active.future.result()
        return self._apply_result(active, result)

    def complete_cpu_export(
        self,
        job: Job,
        result: _FigureJobResult,
    ) -> FigurePageState:
        if job.future is not None or job.kind != "figure-export":
            raise ValueError("job must be a reserved CPU figure export")
        if result.kind != "export":
            raise ValueError("CPU figure export returned the wrong result kind")
        state = self._apply_result(job, result)
        self._finalize_deferred_close()
        return state

    def abandon_cpu_export(self, job: Job) -> FigurePageState:
        with self._lock:
            if self._job is not None and self._job.job_id == job.job_id:
                self._job = None
            state = self._state
            phase = (
                "ready"
                if state.source is not None and not state.source_dirty
                else "error"
                if state.source_error is not None
                else "empty"
            )
            self._state = replace(state, phase=phase)
            updated = self._state
        self.coordinator.finish(job.job_id)
        self._finalize_deferred_close()
        return updated

    def _apply_result(
        self,
        active: Job,
        result: _FigureJobResult,
    ) -> FigurePageState:
        with self._lock:
            if self._job is not None and self._job.job_id == active.job_id:
                self._job = None
        if (
            result.kind == "export"
            and result.phase != "error"
            and result.path is not None
            and self.on_published is not None
        ):
            callback_warnings = list(result.warnings)
            try:
                self.on_published(result.path)
            except Exception as exc:
                callback_warnings.append(f"figure.settings_failed:{exc}")
            result = replace(result, warnings=tuple(callback_warnings))
        # Read once outside the commit lock so source discovery or callbacks can
        # yield; the revision and closed checks are repeated atomically below.
        observed_revision = self.state.revision
        source_identities: tuple[RunIdentity, ...] = ()
        # RunService discovery may touch the filesystem; acquire only after the
        # final revision check below.
        if (
            result.kind == "source"
            and result.phase != "error"
            and result.source is not None
        ):
            try:
                source_identities = self._source_usage_identities(
                    result.source,
                    output_root=result.output_root,
                    parent_run_id=None,
                    parent_run_path=None,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                result = replace(
                    result,
                    source=None,
                    phase="error",
                    errors=(
                        {
                            "code": "figure.source.failed",
                            "message": str(exc),
                        },
                    ),
                    message=str(exc),
                )
        with self._lock:
            state = self._state
            if self._closed or (
                result.kind == "source"
                and result.revision == observed_revision
                and state.revision != observed_revision
            ):
                updated = state
            elif result.kind == "source":
                if result.revision != state.revision:
                    updated = state
                elif result.phase == "error" or result.source is None:
                    failed = self._discard_source_locked(
                        result.message or "Figure source loading failed"
                    )
                    updated = replace(
                        failed,
                        phase="error",
                        warnings=result.warnings,
                        errors=result.errors,
                        source_dirty=True,
                        source_error=result.message or "Figure source loading failed",
                    )
                else:
                    try:
                        self._replace_usage_lease(source_identities)
                    except (RuntimeError, ValueError) as exc:
                        updated = self._discard_source_locked(str(exc))
                    else:
                        self.output_root = result.output_root
                        self.parent_run_id = None
                        self.parent_run_path = None
                        updated = state.with_source(result.source)
            elif result.kind == "preview":
                if result.revision != state.revision:
                    updated = replace(state, preview_stale=True)
                elif result.phase == "error":
                    updated = replace(state, phase="error", errors=result.errors)
                else:
                    updated = replace(
                        state,
                        preview_path=result.path,
                        preview_stale=False,
                        phase="ready",
                        errors=(),
                    )
            elif result.kind == "export":
                if result.revision != state.revision:
                    if result.phase == "error":
                        updated = replace(
                            state,
                            phase="error",
                            errors=result.errors,
                            warnings=(*state.warnings, *result.warnings),
                        )
                    else:
                        updated = replace(
                            state,
                            phase=result.phase,
                            run_path=result.path,
                            errors=result.errors,
                            warnings=(
                                *state.warnings,
                                *result.warnings,
                                "figure.stale_published",
                            ),
                        )
                else:
                    updated = replace(
                        state,
                        phase=result.phase,
                        run_path=result.path,
                        warnings=result.warnings,
                        errors=result.errors,
                    )
            else:
                updated = state
            self._state = updated
        self.coordinator.finish(active.job_id)
        return updated

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._job_unfinished_locked():
                self._close_requested = True
                return
            self._closed = True
            self._replace_usage_lease(())
            self._job = None


@dataclass(frozen=True, slots=True)
class FigurePageView:
    controller: FigureController
    timer: Any


def render_figures_unavailable(ui: Any, translator: Any) -> None:
    """Render an invalid opaque handoff without exposing filesystem details."""

    with ui.column().classes("lte-page lte-figures-page"):
        ui.label(translator.text("figures.unavailable")).classes(
            "lte-page-title"
        ).props("role=heading aria-level=1")
        ui.label(translator.text("figures.unavailable_body")).classes(
            "lte-callout lte-callout--warning"
        )
        ui.button(
            translator.text("nav.figures"),
            on_click=lambda: ui.navigate.to("/figures"),
        ).props("outline")


def render_figures_page(
    ui: Any,
    translator: Any,
    repo_root: str | os.PathLike[str],
    coordinator: JobCoordinator,
    *,
    initial_source: str | Path | None = None,
    source_options: Mapping[str, str] | None = None,
    current_session: Any | None = None,
    output_root: str | os.PathLike[str] | None = None,
    on_published: Callable[[Path], None] | None = None,
    preview_url_builder: Callable[[Path], str] | None = None,
    initial_formats: Iterable[str] | None = None,
    initial_spec: FigureSpec | None = None,
    parent_run_id: str | None = None,
    parent_run_path: str | os.PathLike[str] | None = None,
    usage_leases: RunUsageLeaseRegistry | None = None,
    run_roots: Callable[[], Iterable[Path]] | None = None,
    lease_owner: str | None = None,
    refresh_source_options: Callable[[], Mapping[str, str]] | None = None,
    on_open_source: Callable[[Path], None] | None = None,
) -> FigurePageView:
    """Render an offline form whose expensive actions are always explicit."""

    controller = FigureController(
        repo_root,
        coordinator,
        spec=initial_spec,
        output_root=output_root,
        parent_run_id=parent_run_id,
        parent_run_path=parent_run_path,
        on_published=on_published,
        usage_leases=usage_leases,
        run_roots=run_roots,
        lease_owner=lease_owner,
    )
    page_client = ui.context.client
    route_output_root = controller.output_root
    route_parent_run_id = controller.parent_run_id
    route_parent_run_path = controller.parent_run_path
    form_spec = controller.state.spec
    format_boxes: dict[str, Any] = {}
    requested_initial_formats = (
        ("png", "html")
        if initial_formats is None
        else FigureController._normalise_formats(initial_formats)
    )
    available_source_options = _source_options(source_options, initial_source)
    initial_source_value = (
        None
        if initial_source is None
        else str(Path(initial_source).expanduser().resolve(strict=False))
    )

    with ui.column().classes("lte-page lte-figures-page"):
        ui.label(translator.text("figures.title")).classes("lte-page-title").props(
            "role=heading aria-level=1"
        )
        ui.label(translator.text("figures.subtitle")).classes("lte-page-subtitle")
        with ui.card().classes("lte-figure-source lte-figure-source-bar full-width"):
            with ui.row().classes("lte-figure-panel-header full-width"):
                ui.label(translator.text("figures.source")).classes(
                    "lte-section-title"
                )
                ui.label(translator.text("figures.source_contract")).classes(
                    "lte-figure-section-note"
                )
            with ui.row().classes("lte-figure-source-current full-width"):
                source_current_label = ui.label(
                    translator.text("figures.no_source")
                ).classes("lte-figure-source-current__value").mark(
                    "figure-source-current"
                )
                source_current_status = ui.label(
                    translator.text("status.pending")
                ).classes("lte-figure-source-current__status").mark(
                    "figure-source-current-status"
                )
                ui.space()
                source_menu_items: dict[str, Any] = {}
                render_overflow_menu(
                    ui,
                    (
                        MenuActionSpec(
                            translator.text("figures.refresh_local"),
                            "refresh",
                            lambda: refresh_local(),
                            marker="figure-refresh-local-menu",
                        ),
                        MenuActionSpec(
                            translator.text("figures.open_source"),
                            "open_in_new",
                            lambda: open_source(),
                            marker="figure-open-source-menu",
                        ),
                        MenuActionSpec(
                            translator.text("figures.current_selection"),
                            "my_location",
                            lambda: prepare_current_selection(),
                            enabled=current_session is not None,
                            marker="figure-current-selection",
                        ),
                    ),
                    label=translator.text("figures.source_menu"),
                    marker="figure-source-overflow",
                    item_sink=source_menu_items,
                )
                refresh_local_button = ui.button(
                    translator.text("figures.refresh_local"),
                    on_click=lambda: refresh_local(),
                ).props("flat").mark("figure-refresh-local")
                open_source_button = ui.button(
                    translator.text("figures.open_source"),
                    on_click=lambda: open_source(),
                ).props("flat").mark("figure-open-source")
            with ui.row().classes("lte-figure-source-actions items-end full-width"):
                # Keep the source selector as a stable public control/marker. It
                # is compact in the source bar and remains the authoritative
                # selection model even when the overflow menu is used.
                source_input = ui.select(
                    available_source_options,
                    value=initial_source_value,
                    label=translator.text("figures.source_path"),
                    with_input=True,
                ).props("clearable options-dense").classes(
                    "grow lte-figure-source-picker"
                ).mark("figure-source-path")
                load_button = ui.button(
                    translator.text("figures.load_source")
                ).props("outline").mark("figure-load-source")
                current_button = None
            source_status = ui.label("").classes(
                "lte-figure-source-status"
            ).mark("figure-source-ready")
            source_status.set_visibility(False)
            source_identity = ui.label("").classes(
                "lte-figure-source-identity"
            ).mark("figure-source-identity")
            source_identity.set_visibility(False)
            dirty_label = ui.label(
                translator.text("figures.source_changed")
            ).classes("lte-callout lte-callout--warning").mark(
                "figure-source-dirty"
            )
            dirty_label.set_visibility(False)
            source_error_label = ui.label("").classes(
                "lte-callout lte-callout--error"
            ).props("role=alert").mark("figure-source-error")
            source_error_label.set_visibility(False)
            terrain_label = ui.label(
                translator.text("figures.terrain_unavailable")
            ).classes("lte-callout lte-callout--warning").mark(
                "figure-terrain-unavailable"
            )
            terrain_label.set_visibility(False)
            empty_source_label = ui.label(
                translator.text("figures.no_source")
            ).classes("lte-figure-empty-copy").mark("figure-source-empty")
            warning_box = ui.column().classes("lte-figure-warnings")

        with ui.element("section").classes("lte-figure-workspace full-width"):
            with ui.card().classes("lte-figure-preview-card"):
                with ui.row().classes("lte-figure-panel-header full-width"):
                    ui.label(translator.text("figures.preview_surface")).classes(
                        "lte-section-title"
                    )
                    ui.space()
                    stale_label = ui.label(
                        translator.text("figures.preview_waiting")
                    ).classes("lte-figure-preview-state")
                with ui.element("div").classes("lte-figure-preview-surface"):
                    preview = ui.image().classes("lte-figure-preview").props(
                        f'alt="{translator.text("figures.preview_accessible_name")}"'
                    ).mark(
                        "figure-preview-surface"
                    )
                    preview.set_visibility(False)
                    preview_empty = ui.label(
                        translator.text("figures.preview_waiting")
                    ).classes("lte-figure-preview-empty")
                refresh_button = ui.button(
                    translator.text("figures.refresh_preview")
                ).mark("figure-refresh-preview")

            with ui.card().classes("lte-figure-style"):
                ui.label(translator.text("figures.style")).classes(
                    "lte-section-title"
                )
                with ui.column().classes("lte-figure-style-grid full-width"):
                    preset = ui.select(
                        {
                            "preview": translator.text("figures.preset_preview"),
                            "publication": translator.text(
                                "figures.preset_publication"
                            ),
                        },
                        value=form_spec.preset,
                        label=translator.text("figures.preset"),
                    ).mark("figure-preset")
                    dpi = ui.number(
                        translator.text("figures.dpi"),
                        value=form_spec.dpi,
                        precision=0,
                    ).mark("figure-dpi")
                    colormap = ui.select(
                        _colormap_options(form_spec.colormap),
                        value=form_spec.colormap,
                        label=translator.text("figures.colormap"),
                        with_input=True,
                    ).props("options-dense").mark("figure-colormap")
                    with ui.row().classes("lte-figure-style-pair full-width"):
                        azimuth = ui.number(
                            translator.text("figures.azimuth"),
                            value=form_spec.azimuth,
                        ).mark("figure-azimuth")
                        elevation = ui.number(
                            translator.text("figures.elevation"),
                            value=form_spec.elevation_angle,
                        ).mark("figure-elevation")
                    vertical = ui.number(
                        translator.text("figures.vertical_exaggeration"),
                        value=form_spec.vertical_exaggeration,
                    ).mark("figure-vertical-exaggeration")
                    with ui.row().classes("lte-figure-style-pair full-width"):
                        station_color = ui.select(
                            _station_color_options(form_spec.station_color),
                            value=form_spec.station_color,
                            label=translator.text("figures.station_color"),
                            with_input=True,
                        ).props("options-dense").mark("figure-station-color")
                        station_size = ui.number(
                            translator.text("figures.station_size"),
                            value=form_spec.station_size,
                        ).mark("figure-station-size")
                    title = ui.input(
                        translator.text("figures.figure_title"),
                        value=form_spec.title or "",
                    ).mark("figure-title")
                    max_pixels = ui.number(
                        translator.text("figures.max_pixels"),
                        value=form_spec.max_pixels,
                        precision=0,
                    ).mark("figure-max-pixels")

        with ui.element("footer").classes(
            "lte-figure-export lte-figure-export-dock full-width"
        ).props('role=region aria-label="Figure export actions"').mark(
            "figure-export-dock"
        ):
            with ui.row().classes("lte-figure-panel-header full-width"):
                ui.label(translator.text("figures.final_export")).classes(
                    "lte-section-title"
                )
                ui.label(translator.text("figures.export_summary")).classes(
                    "lte-figure-section-note"
                )
            with ui.row().classes("lte-figure-export-controls full-width"):
                with ui.row().classes("lte-figure-format-list"):
                    for token in FIGURE_FORMAT_ORDER:
                        format_boxes[token] = ui.checkbox(
                            token.upper(),
                            value=token in requested_initial_formats,
                        ).mark(f"figure-format-{token}")
                ui.space()
            destination_label = ui.label(
                translator.text(
                    "figures.destination",
                    path=translator.text("value.none"),
                )
            ).classes("lte-figure-path")
            result_label = ui.label("").classes("lte-figure-path")
            error_box = ui.column().classes("lte-figure-errors")

            action_refs = render_action_bar(
                ui,
                (
                    ActionSpec(
                        "export",
                        translator.text("figures.export"),
                        lambda: final_export(),
                        role="primary",
                        marker="figure-export",
                    ),
                    ActionSpec(
                        "reset",
                        translator.text("figures.reset"),
                        lambda: reset_workspace(),
                        role="secondary",
                        marker="figure-reset",
                    ),
                    ActionSpec(
                        "back",
                        translator.text("figures.back"),
                        lambda: ui.navigate.to("/history"),
                        role="tertiary",
                        marker="figure-back",
                    ),
                ),
                sticky=True,
                marker="figure-export-action-bar",
            )
            export_button = action_refs["export"]
            reset_button = action_refs["reset"]
            back_button = action_refs["back"]

        technical_refs: dict[str, Any] = {}

        def render_figure_technical_copy() -> None:
            technical_refs["copy"] = ui.label("").classes(
                "lte-technical-copy"
            ).mark("figure-technical-copy")

        technical_expansion = render_technical_details(
            ui,
            translator.text("figures.technical_details"),
            render_figure_technical_copy,
            marker="figure-technical-details",
        )
        technical_expansion.set_visibility(False)
        technical_copy = technical_refs["copy"]

    style_controls = (
        preset,
        dpi,
        colormap,
        azimuth,
        elevation,
        vertical,
        station_color,
        station_size,
        title,
        max_pixels,
    )
    restoring_source_control = False
    restoring_spec_controls = False
    restoring_format_controls = False
    confirmed_formats = {
        token: bool(format_boxes[token].value) for token in FIGURE_FORMAT_ORDER
    }
    initial_workspace_spec = form_spec
    initial_workspace_formats = dict(confirmed_formats)
    ui_validation_diagnostic: str | None = None

    def restore_spec_controls(spec: FigureSpec) -> None:
        nonlocal restoring_spec_controls
        restoring_spec_controls = True
        try:
            preset.value = spec.preset
            dpi.value = spec.dpi
            colormap.value = spec.colormap
            azimuth.value = spec.azimuth
            elevation.value = spec.elevation_angle
            vertical.value = spec.vertical_exaggeration
            station_color.value = spec.station_color
            station_size.value = spec.station_size
            title.value = spec.title or ""
            max_pixels.value = spec.max_pixels
        finally:
            restoring_spec_controls = False

    def restore_confirmed_source_input() -> None:
        nonlocal restoring_source_control
        state = controller.state
        confirmed = (
            str(state.source.path)
            if isinstance(state.source, FigureSource)
            and not state.source_dirty
            and state.source.path is not None
            else None
        )
        if source_input.value != confirmed:
            restoring_source_control = True
            try:
                source_input.value = confirmed
            finally:
                restoring_source_control = False

    def render_messages() -> None:
        warning_box.clear()
        error_box.clear()
        state = controller.state
        with warning_box:
            for index, warning in enumerate(state.warnings):
                if warning == "figure.stale_published":
                    text = translator.text("figures.warning.stale_published")
                elif warning.startswith("figure.settings_failed:"):
                    text = translator.text("figures.warning.settings_failed")
                else:
                    text = translator.text("figures.warning.unknown")
                ui.label(text).classes("lte-callout lte-callout--warning").mark(
                    f"figure-warning-summary-{index}"
                )
        if state.source_error is None and state.errors:
            with error_box:
                ui.label(translator.text("figures.error.summary")).classes(
                    "lte-callout lte-callout--error"
                ).props("role=alert").mark("figure-error-summary")

        diagnostics: list[str] = []
        if state.source_error is not None:
            diagnostics.append(state.source_error)
        diagnostics.extend(state.warnings)
        diagnostics.extend(
            json.dumps(_jsonable(error), ensure_ascii=False, sort_keys=True)
            for error in state.errors
        )
        if ui_validation_diagnostic is not None:
            diagnostics.append(ui_validation_diagnostic)
        technical_copy.set_text("\n".join(diagnostics))
        technical_expansion.set_visibility(bool(diagnostics))

    def set_ui_diagnostic(detail: str | None) -> None:
        nonlocal ui_validation_diagnostic
        ui_validation_diagnostic = (
            None if detail is None else detail.strip() or type(detail).__name__
        )
        render_messages()

    def refresh_local() -> None:
        """Refresh the locally discovered source list without changing source state."""

        if refresh_source_options is None:
            ui.notify(translator.text("figures.refresh_local_unavailable"), type="warning")
            return
        try:
            refreshed = _source_options(refresh_source_options(), None)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            set_ui_diagnostic(str(exc))
            ui.notify(translator.text("figures.refresh_local_failed"), type="warning")
            return
        source_input.options = refreshed
        source_input.update()
        ui.notify(translator.text("figures.refresh_local_done"), type="positive")

    def open_source() -> None:
        current = controller.state.source
        if not isinstance(current, FigureSource) or current.path is None:
            ui.notify(translator.text("figures.open_source_unavailable"), type="warning")
            return
        if on_open_source is not None:
            on_open_source(current.path)
            return
        ui.navigate.to("/history")

    def reset_workspace() -> None:
        if controller.has_unfinished_job:
            ui.notify(translator.text("error.job_busy"), type="warning")
            return
        restore_spec_controls(initial_workspace_spec)
        try:
            controller.update_spec(initial_workspace_spec)
        except RuntimeError as exc:
            set_ui_diagnostic(str(exc))
            return
        for token, control in format_boxes.items():
            control.value = initial_workspace_formats[token]
            confirmed_formats[token] = initial_workspace_formats[token]
        set_ui_diagnostic(None)
        sync_controls()

    def sync_controls(state: FigurePageState | None = None) -> None:
        current = controller.state if state is None else state
        running = controller.has_unfinished_job
        valid_source = (
            isinstance(current.source, FigureSource) and not current.source_dirty
        )
        has_dem = valid_source and current.source.dem_path is not None

        for control in (
            source_input,
            load_button,
            current_button,
            reset_button,
            back_button,
            refresh_local_button,
            open_source_button,
            *style_controls,
            *format_boxes.values(),
        ):
            if control is None:
                continue
            if running:
                control.disable()
            else:
                control.enable()
        for control in (refresh_button, export_button):
            if has_dem and not running:
                control.enable()
            else:
                control.disable()
        export_button.set_text(
            translator.text("figures.export_running")
            if running
            else translator.text("figures.export")
        )
        if not running and not source_input.value:
            load_button.disable()

        source_status.set_visibility(valid_source)
        source_identity.set_visibility(valid_source)
        source_current_label.set_text(
            translator.text("figures.no_source")
            if not valid_source
            else (
                str(current.source.path)
                if isinstance(current.source, FigureSource)
                and current.source.path is not None
                else translator.text("figures.current_selection_loaded")
            )
        )
        source_current_status.set_text(
            translator.text("status.running")
            if running
            else translator.text("status.ready")
            if valid_source
            else translator.text("status.pending")
        )
        dirty_label.set_visibility(current.source_dirty and current.source_error is None)
        source_error_label.set_visibility(current.source_error is not None)
        terrain_label.set_visibility(valid_source and not has_dem)
        empty_source_label.set_visibility(
            not valid_source
            and not current.source_dirty
            and current.source_error is None
        )
        if current.source_error is not None:
            source_error_label.set_text(translator.text("figures.source_load_failed"))

        if valid_source:
            assert isinstance(current.source, FigureSource)
            source_status.set_text(
                translator.text(
                    "figures.source_ready",
                    rect_id=current.source.rectangle["rect_id"],
                )
            )
            identity = (
                translator.text("figures.current_selection_loaded")
                if current.source.path is None
                else translator.text(
                    "figures.source_loaded_path",
                    path=str(current.source.path),
                )
            )
            source_identity.set_text(identity)
            try:
                destination, _parent = controller._target(current.source)
            except ValueError:
                destination_text = translator.text("value.none")
            else:
                destination_text = str(destination)
            destination_label.set_text(
                translator.text("figures.destination", path=destination_text)
            )
        else:
            destination_label.set_text(
                translator.text(
                    "figures.destination",
                    path=translator.text("value.none"),
                )
            )
            result_label.set_text("")

        if not valid_source:
            preview.set_visibility(False)
            preview_empty.set_visibility(True)
            waiting_key = (
                "figures.preview_load_to_continue"
                if current.source_dirty or current.source_error is not None
                else "figures.preview_waiting"
            )
            waiting_text = translator.text(waiting_key)
            preview_empty.set_text(waiting_text)
            stale_label.set_text(waiting_text)
        elif current.preview_path is None:
            preview.set_visibility(False)
            preview_empty.set_visibility(True)
            preview_empty.set_text(translator.text("figures.preview_not_generated"))
            stale_label.set_text(translator.text("figures.preview_stale"))
        else:
            preview_empty.set_visibility(False)
            if preview_url_builder is not None:
                preview.set_source(preview_url_builder(current.preview_path))
                preview.set_visibility(True)
            stale_label.set_text(
                translator.text(
                    "figures.preview_stale"
                    if current.preview_stale
                    else "figures.preview_fresh"
                )
            )
        if current.run_path is not None:
            result_label.set_text(
                translator.text("figures.exported", path=str(current.run_path))
            )

    def invalidate_visible_source(error: str | None = None) -> bool:
        detail = (
            None
            if error is None
            else error.strip() or translator.text("figures.source_load_unknown")
        )
        try:
            state = controller.invalidate_source(detail)
        except RuntimeError as exc:
            restore_confirmed_source_input()
            sync_controls()
            set_ui_diagnostic(str(exc))
            ui.notify(translator.text("error.job_busy"), type="warning")
            return False
        sync_controls(state)
        render_messages()
        return True

    def apply_source(
        source: FigureSource,
        *,
        source_output_root: Path | None = None,
        source_parent_run_id: str | None = None,
        source_parent_run_path: Path | None = None,
    ) -> None:
        set_ui_diagnostic(None)
        state = controller.set_source(
            source,
            output_root=source_output_root,
            parent_run_id=source_parent_run_id,
            parent_run_path=source_parent_run_path,
        )
        sync_controls(state)
        render_messages()

    def load_path(
        *,
        source_output_root: Path | None = None,
        source_parent_run_id: str | None = None,
        source_parent_run_path: Path | None = None,
    ) -> None:
        if not invalidate_visible_source():
            return
        if not source_input.value:
            invalidate_visible_source(
                translator.text("figures.no_source")
            )
            return
        try:
            source = load_figure_source(source_input.value)
        except Exception as exc:
            invalidate_visible_source(str(exc))
            return
        apply_source(
            source,
            source_output_root=source_output_root,
            source_parent_run_id=source_parent_run_id,
            source_parent_run_path=source_parent_run_path,
        )

    def changed_spec() -> bool:
        if restoring_spec_controls:
            return False
        try:
            base = FigureSpec.from_preset(str(preset.value))
            spec = replace(
                base,
                dpi=_positive_integer_control(dpi.value, field="DPI"),
                colormap=str(colormap.value),
                azimuth=float(azimuth.value),
                elevation_angle=float(elevation.value),
                vertical_exaggeration=float(vertical.value),
                station_color=str(station_color.value),
                station_size=float(station_size.value),
                title=str(title.value).strip() or None,
                max_pixels=_positive_integer_control(
                    max_pixels.value,
                    field="Maximum terrain pixels",
                ),
            )
            controller.update_spec(spec)
        except RuntimeError as exc:
            restore_spec_controls(controller.state.spec)
            sync_controls()
            set_ui_diagnostic(str(exc))
            ui.notify(translator.text("error.job_busy"), type="warning")
            return False
        except (TypeError, ValueError) as exc:
            set_ui_diagnostic(str(exc))
            ui.notify(translator.text("figures.style_invalid"), type="warning")
            return False
        sync_controls()
        set_ui_diagnostic(None)
        return True

    def apply_preset(value: Any) -> None:
        if restoring_spec_controls:
            return
        try:
            spec = FigureSpec.from_preset(str(value))
        except ValueError as exc:
            set_ui_diagnostic(str(exc))
            ui.notify(translator.text("figures.style_invalid"), type="warning")
            return
        restore_spec_controls(spec)
        try:
            controller.update_spec(spec)
        except RuntimeError as exc:
            restore_spec_controls(controller.state.spec)
            set_ui_diagnostic(str(exc))
            ui.notify(translator.text("error.job_busy"), type="warning")
        else:
            set_ui_diagnostic(None)
        sync_controls()

    def changed_format(token: str) -> None:
        nonlocal restoring_format_controls
        if restoring_format_controls:
            return
        control = format_boxes[token]
        if controller.has_unfinished_job:
            restoring_format_controls = True
            try:
                control.value = confirmed_formats[token]
            finally:
                restoring_format_controls = False
            sync_controls()
            return
        confirmed_formats[token] = bool(control.value)

    def refresh_preview() -> None:
        if not changed_spec():
            return
        try:
            job = controller.refresh_preview()
        except JobBusyError as exc:
            set_ui_diagnostic(str(exc))
            ui.notify(translator.text("error.job_busy"), type="warning")
            return
        except ValueError as exc:
            set_ui_diagnostic(str(exc))
            ui.notify(translator.text("figures.preview_failed"), type="warning")
            return
        if job is None:
            sync_controls()
            return
        sync_controls()
        timer.activate()

    async def final_export() -> None:
        if not changed_spec():
            return
        formats = tuple(
            token for token in FIGURE_FORMAT_ORDER if format_boxes[token].value
        )
        try:
            job, request = controller.reserve_cpu_export(formats)
        except JobBusyError as exc:
            set_ui_diagnostic(str(exc))
            ui.notify(translator.text("error.job_busy"), type="warning")
            return
        except (RuntimeError, ValueError) as exc:
            set_ui_diagnostic(str(exc))
            ui.notify(translator.text("figures.export_failed"), type="warning")
            return
        sync_controls()
        from nicegui import run

        try:
            result = await run.cpu_bound(_render_figure_export, request)
        except BaseException as exc:
            # Cancellation can arrive as ``BaseException``. Release the
            # reserved coordinator slot and source lease before propagating it
            # so a deleted page cannot strand the application's one-job gate.
            controller.abandon_cpu_export(job)
            if isinstance(exc, asyncio.CancelledError):
                raise
            result = _FigureJobResult(
                "export",
                request.revision,
                phase="error",
                errors=(
                    {"code": "figure.export.failed", "message": str(exc)},
                ),
                message=str(exc),
            )
        if result is None:
            state = controller.abandon_cpu_export(job)
        else:
            state = controller.complete_cpu_export(job, result)
        if page_client.is_deleted:
            return
        sync_controls(state)
        render_messages()

    def prepare_current_selection() -> None:
        if not invalidate_visible_source():
            return
        try:
            controller.prepare_selection(current_session)
        except JobBusyError as exc:
            set_ui_diagnostic(str(exc))
            ui.notify(translator.text("error.job_busy"), type="warning")
            return
        except (RuntimeError, ValueError) as exc:
            invalidate_visible_source(str(exc))
            ui.notify(
                translator.text("figures.current_selection_failed"),
                type="warning",
            )
            return
        sync_controls()
        timer.activate()

    def tick() -> None:
        state = controller.drain()
        if controller.has_unfinished_job:
            sync_controls(state)
            return
        timer.deactivate()
        sync_controls(state)
        render_messages()

    load_button.on("click", lambda: load_path())
    source_input.on_value_change(
        lambda _event: (
            None if restoring_source_control else invalidate_visible_source()
        )
    )
    refresh_button.on("click", refresh_preview)
    preset.on_value_change(lambda event: apply_preset(event.value))
    for field in (
        dpi,
        colormap,
        azimuth,
        elevation,
        vertical,
        station_color,
        station_size,
        title,
        max_pixels,
    ):
        field.on_value_change(lambda _event: changed_spec())
    for token, control in format_boxes.items():
        control.on_value_change(lambda _event, value=token: changed_format(value))
    if current_button is not None:
        current_button.on("click", prepare_current_selection)
    timer = ui.timer(0.1, tick, active=False)

    def cleanup() -> None:
        timer.deactivate()
        controller.close()

    ui.context.client.on_delete(cleanup)
    if initial_source is not None:
        load_path(
            source_output_root=route_output_root,
            source_parent_run_id=route_parent_run_id,
            source_parent_run_path=route_parent_run_path,
        )
    else:
        sync_controls()
    return FigurePageView(controller=controller, timer=timer)


__all__ = [
    "FIGURE_FORMAT_ORDER",
    "FigureController",
    "FigurePageState",
    "FigurePageView",
    "load_figure_source",
    "preview_cache_path",
    "preview_spec",
    "render_figures_page",
    "render_figures_unavailable",
]
