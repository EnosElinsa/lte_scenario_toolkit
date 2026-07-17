"""Explicit preview and final terrain-figure workflows for the local GUI."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any

from ... import io
from ...figure_service import (
    FigureService,
    FigureSource,
    FigureSpec,
    validate_csv_identity,
)
from ...jobs import Job, JobBusyError, JobCoordinator
from ...run_service import RunService

PREVIEW_CACHE_VERSION = "figure-preview-v1"
PREVIEW_DPI_LIMIT = 120
PREVIEW_PIXEL_LIMIT = 600
FIGURE_FORMAT_ORDER = ("png", "eps", "html")


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
    return replace(
        spec,
        dpi=min(spec.dpi, PREVIEW_DPI_LIMIT),
        max_pixels=min(spec.max_pixels, PREVIEW_PIXEL_LIMIT),
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
            phase="ready" if self.source is not None else "empty",
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
    ) -> None:
        self.repo_root = Path(repo_root).resolve(strict=False)
        self.coordinator = coordinator
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
            FigurePageState() if source is None else FigurePageState.for_source(source)
        )
        self._state = replace(initial_state, spec=initial_spec)
        self._job: Job | None = None
        self._closed = False

    @property
    def state(self) -> FigurePageState:
        with self._lock:
            return self._state

    @property
    def job(self) -> Job | None:
        with self._lock:
            return self._job

    def _set_state(self, state: FigurePageState) -> FigurePageState:
        with self._lock:
            self._state = state
            return state

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
        with self._lock:
            if self._closed:
                raise RuntimeError("figure controller is closed")
            self.output_root = (
                None
                if output_root is None
                else Path(output_root).expanduser().resolve(strict=False)
            )
            self.parent_run_id = parent_run_id
            self.parent_run_path = (
                None
                if parent_run_path is None
                else Path(parent_run_path).resolve(strict=False)
            )
            self._state = self._state.with_source(source)
            return self._state

    def update_spec(self, spec: FigureSpec) -> FigurePageState:
        with self._lock:
            if self._closed:
                raise RuntimeError("figure controller is closed")
            self._state = self._state.with_spec(spec)
            return self._state

    def clear_source(self) -> FigurePageState:
        with self._lock:
            if self._closed:
                raise RuntimeError("figure controller is closed")
            self.output_root = None
            self.parent_run_id = None
            self.parent_run_path = None
            self._state = FigurePageState(
                spec=self._state.spec,
                revision=self._state.revision + 1,
            )
            return self._state

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
        job.future.add_done_callback(lambda _future: self.coordinator.finish(job.job_id))
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

    def export(self, formats: Iterable[str]) -> Job:
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
        revision = state.revision

        def worker(_cancel: Any, _emit: Callable[[Any], None]) -> _FigureJobResult:
            try:
                path = FigureService.render(
                    source_snapshot,
                    spec,
                    RunService(output_root),
                    requested,
                    parent_run_id=parent_run_id,
                    entrypoint=("lte-gui", "figures"),
                    repository=self.repo_root,
                )
                entry = RunService(output_root).entry_for_path(path)
                record = entry.record
                metadata = record.get("metadata")
                if record.get("scenario_id") != (source_snapshot.scenario_id or "figures"):
                    raise ValueError("figure run scenario does not match its source")
                if record.get("profile_id") != (source_snapshot.profile_id or "figures"):
                    raise ValueError("figure run profile does not match its source")
                if record.get("parent_run_id") != parent_run_id:
                    raise ValueError("figure run parent does not match its source")
                if not isinstance(metadata, Mapping) or metadata.get("run_kind") != "figure":
                    raise ValueError("published run is not a figure run")
                recorded_formats = metadata.get("requested_formats")
                if not isinstance(recorded_formats, (list, tuple)) or tuple(
                    recorded_formats
                ) != requested:
                    raise ValueError("figure run formats do not match the request")
                recorded_spec = metadata.get("figure_spec")
                if not isinstance(recorded_spec, Mapping) or dict(
                    recorded_spec
                ) != spec.as_dict():
                    raise ValueError("figure run style does not match the request")
                source_metadata = metadata.get("source")
                if not isinstance(source_metadata, Mapping):
                    raise ValueError("figure run is missing source metadata")
                if source_metadata.get("run_id") != source_snapshot.run_id:
                    raise ValueError("figure run source ID does not match")
                if source_metadata.get("kind") != source_snapshot.source_kind:
                    raise ValueError("figure run source kind does not match")
                if source_metadata.get("path") != (
                    str(source_snapshot.path)
                    if source_snapshot.path is not None
                    else None
                ):
                    raise ValueError("figure run source path does not match")
                if source_metadata.get("csv") != (
                    str(source_snapshot.csv_path)
                    if source_snapshot.csv_path is not None
                    else None
                ):
                    raise ValueError("figure run source CSV does not match")
                expected_selection = (
                    source_snapshot.selection_identity.as_dict()
                    if source_snapshot.selection_identity is not None
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
                if any(
                    relative not in artifacts for relative in artifact_paths.values()
                ):
                    raise ValueError("figure run artifact paths are inconsistent")
                if len(set(artifact_paths.values())) != len(artifact_paths):
                    raise ValueError("figure run artifact paths must be unique")
                successful = set(artifact_paths)
                if not successful or not successful.issubset(set(requested)):
                    raise ValueError("figure run has invalid successful formats")
                phase = str(entry.record["status"])
                if phase == "completed" and successful != set(requested):
                    raise ValueError("completed figure run is missing formats")
                failed_tokens = {
                    item.get("artifact")
                    for item in record.get("errors", ())
                    if isinstance(item, Mapping)
                }
                if phase == "partial" and failed_tokens != set(requested) - successful:
                    raise ValueError("partial figure run errors do not match missing formats")
                if phase == "completed" and failed_tokens:
                    raise ValueError("completed figure run must not contain errors")
                if phase == "partial" and successful == set(requested):
                    raise ValueError("partial figure run must have a failed format")
                errors = tuple(
                    dict(item)
                    for item in entry.record["errors"]
                    if isinstance(item, Mapping)
                )
                callback_warnings = list(source_snapshot.warnings)
                if self.on_published is not None:
                    try:
                        self.on_published(entry.run_dir)
                    except Exception as exc:
                        callback_warnings.append(
                            f"figure.settings_failed:{exc}"
                        )
                return _FigureJobResult(
                    "export",
                    revision,
                    path=entry.run_dir,
                    phase=phase,
                    warnings=tuple(callback_warnings),
                    errors=errors,
                )
            except Exception as exc:
                return _FigureJobResult(
                    "export",
                    revision,
                    phase="error",
                    errors=(
                        {"code": "figure.export.failed", "message": str(exc)},
                    ),
                    message=str(exc),
                )

        job = self._submit("figure-export", worker)
        self._set_state(replace(state, phase="exporting", errors=()))
        return job

    def drain(self, job: Job | None = None) -> FigurePageState:
        active = self.job if job is None else job
        if active is None or active.future is None or not active.future.done():
            return self.state
        result = active.future.result()
        state = self.state
        if result.kind == "source":
            if result.revision != state.revision:
                self.coordinator.finish(active.job_id)
                return state
            if result.phase == "error" or result.source is None:
                updated = replace(
                    state,
                    phase="error",
                    warnings=result.warnings,
                    errors=result.errors,
                )
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
                self._set_state(updated)
                self.coordinator.finish(active.job_id)
                return updated
            updated = replace(
                state,
                phase=result.phase,
                run_path=result.path,
                warnings=result.warnings,
                errors=result.errors,
            )
        else:
            updated = state
        self._set_state(updated)
        self.coordinator.finish(active.job_id)
        return updated

    def close(self) -> None:
        with self._lock:
            self._closed = True


@dataclass(frozen=True, slots=True)
class FigurePageView:
    controller: FigureController
    timer: Any


def render_figures_unavailable(ui: Any, translator: Any) -> None:
    """Render an invalid opaque handoff without exposing filesystem details."""

    with ui.column().classes("lte-page lte-figures-page"):
        ui.label(translator.text("figures.unavailable")).classes("lte-page-title")
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
    current_session: Any | None = None,
    output_root: str | os.PathLike[str] | None = None,
    on_published: Callable[[Path], None] | None = None,
    preview_url_builder: Callable[[Path], str] | None = None,
    initial_formats: Iterable[str] | None = None,
    initial_spec: FigureSpec | None = None,
    parent_run_id: str | None = None,
    parent_run_path: str | os.PathLike[str] | None = None,
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
    )
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

    with ui.column().classes("lte-page lte-figures-page"):
        ui.label(translator.text("figures.title")).classes("lte-page-title")
        ui.label(translator.text("figures.subtitle")).classes("lte-page-subtitle")
        with ui.card().classes("lte-figure-source full-width"):
            ui.label(translator.text("figures.source")).classes("lte-section-title")
            with ui.row().classes("items-end full-width"):
                source_input = ui.input(
                    translator.text("figures.source_path"),
                    value="" if initial_source is None else str(initial_source),
                ).classes("grow").mark("figure-source-path")
                load_button = ui.button(
                    translator.text("figures.load_source")
                ).mark("figure-load-source")
                if current_session is not None:
                    current_button = ui.button(
                        translator.text("figures.current_selection")
                    ).props("outline").mark("figure-current-selection")
                else:
                    current_button = None
            source_status = ui.label(translator.text("figures.no_source"))
            warning_box = ui.column().classes("lte-figure-warnings")

        with ui.card().classes("lte-figure-style full-width"):
            ui.label(translator.text("figures.style")).classes("lte-section-title")
            with ui.row().classes("full-width"):
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
                colormap = ui.input(
                    translator.text("figures.colormap"),
                    value=form_spec.colormap,
                )
                azimuth = ui.number(
                    translator.text("figures.azimuth"),
                    value=form_spec.azimuth,
                )
                elevation = ui.number(
                    translator.text("figures.elevation"),
                    value=form_spec.elevation_angle,
                )
            with ui.row().classes("full-width"):
                vertical = ui.number(
                    translator.text("figures.vertical_exaggeration"),
                    value=form_spec.vertical_exaggeration,
                )
                station_color = ui.input(
                    translator.text("figures.station_color"),
                    value=form_spec.station_color,
                )
                station_size = ui.number(
                    translator.text("figures.station_size"),
                    value=form_spec.station_size,
                )
                title = ui.input(
                    translator.text("figures.figure_title"),
                    value=form_spec.title or "",
                )
                max_pixels = ui.number(
                    translator.text("figures.max_pixels"),
                    value=form_spec.max_pixels,
                    precision=0,
                )
            refresh_button = ui.button(
                translator.text("figures.refresh_preview")
            ).mark("figure-refresh-preview")
            stale_label = ui.label(translator.text("figures.preview_stale"))
            preview = ui.image().classes("lte-figure-preview")
            preview.set_visibility(False)

        with ui.card().classes("lte-figure-export full-width"):
            ui.label(translator.text("figures.final_export")).classes(
                "lte-section-title"
            )
            for token in FIGURE_FORMAT_ORDER:
                format_boxes[token] = ui.checkbox(
                    token.upper(),
                    value=token in requested_initial_formats,
                )
            export_button = ui.button(
                translator.text("figures.export")
            ).mark("figure-export")
            destination_label = ui.label(
                translator.text(
                    "figures.destination",
                    path=(
                        str(controller.output_root)
                        if controller.output_root is not None
                        else translator.text("value.none")
                    ),
                )
            ).classes("lte-path")
            result_label = ui.label("")
            error_box = ui.column().classes("lte-figure-errors")

    def render_messages() -> None:
        warning_box.clear()
        error_box.clear()
        state = controller.state
        with warning_box:
            for warning in state.warnings:
                if warning == "figure.stale_published":
                    text = translator.text("figures.warning.stale_published")
                elif warning.startswith("figure.settings_failed:"):
                    detail = warning.partition(":")[2]
                    text = translator.text(
                        "figures.warning.settings_failed",
                        detail=detail,
                    )
                else:
                    text = warning
                ui.label(text).classes("lte-callout lte-callout--warning")
        with error_box:
            for error in state.errors:
                ui.label(
                    f"{error.get('code', 'figure.error')}: {error.get('message', '')}"
                ).classes("lte-validation-result lte-validation-result--error")

    def apply_source(
        source: FigureSource,
        *,
        source_output_root: Path | None = None,
        source_parent_run_id: str | None = None,
        source_parent_run_path: Path | None = None,
    ) -> None:
        controller.set_source(
            source,
            output_root=source_output_root,
            parent_run_id=source_parent_run_id,
            parent_run_path=source_parent_run_path,
        )
        source_status.set_text(
            translator.text(
                "figures.source_ready",
                rect_id=source.rectangle["rect_id"],
            )
        )
        try:
            destination, _parent = controller._target(source)
        except ValueError:
            pass
        else:
            destination_label.set_text(
                translator.text("figures.destination", path=str(destination))
            )
        stale_label.set_text(translator.text("figures.preview_stale"))
        render_messages()

    def load_path(
        *,
        source_output_root: Path | None = None,
        source_parent_run_id: str | None = None,
        source_parent_run_path: Path | None = None,
    ) -> None:
        try:
            source = load_figure_source(source_input.value)
        except Exception as exc:
            ui.notify(str(exc), type="negative")
            return
        apply_source(
            source,
            source_output_root=source_output_root,
            source_parent_run_id=source_parent_run_id,
            source_parent_run_path=source_parent_run_path,
        )

    def changed_spec() -> bool:
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
                title=str(title.value) or None,
                max_pixels=_positive_integer_control(
                    max_pixels.value,
                    field="Maximum terrain pixels",
                ),
            )
            controller.update_spec(spec)
        except (TypeError, ValueError) as exc:
            ui.notify(str(exc), type="warning")
            return False
        stale_label.set_text(translator.text("figures.preview_stale"))
        return True

    def apply_preset(value: Any) -> None:
        try:
            spec = FigureSpec.from_preset(str(value))
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        dpi.value = spec.dpi
        colormap.value = spec.colormap
        azimuth.value = spec.azimuth
        elevation.value = spec.elevation_angle
        vertical.value = spec.vertical_exaggeration
        station_color.value = spec.station_color
        station_size.value = spec.station_size
        title.value = spec.title or ""
        max_pixels.value = spec.max_pixels
        controller.update_spec(spec)
        stale_label.set_text(translator.text("figures.preview_stale"))

    def refresh_preview() -> None:
        if not changed_spec():
            return
        try:
            job = controller.refresh_preview()
        except (JobBusyError, ValueError) as exc:
            ui.notify(str(exc), type="warning")
            return
        if job is None:
            state = controller.state
            if state.preview_path is not None:
                if preview_url_builder is not None:
                    preview.set_source(preview_url_builder(state.preview_path))
                    preview.set_visibility(True)
                stale_label.set_text(translator.text("figures.preview_fresh"))
            return
        refresh_button.disable()
        export_button.disable()
        timer.activate()

    def final_export() -> None:
        if not changed_spec():
            return
        formats = tuple(
            token for token in FIGURE_FORMAT_ORDER if format_boxes[token].value
        )
        try:
            controller.export(formats)
        except (JobBusyError, ValueError) as exc:
            ui.notify(str(exc), type="warning")
            return
        refresh_button.disable()
        export_button.disable()
        timer.activate()

    def prepare_current_selection() -> None:
        try:
            controller.prepare_selection(current_session)
        except (JobBusyError, RuntimeError, ValueError) as exc:
            ui.notify(str(exc), type="warning")
            return
        load_button.disable()
        if current_button is not None:
            current_button.disable()
        refresh_button.disable()
        export_button.disable()
        timer.activate()

    def tick() -> None:
        state = controller.drain()
        if state.phase in {"loading", "previewing", "exporting"}:
            return
        timer.deactivate()
        load_button.enable()
        if current_button is not None:
            current_button.enable()
        refresh_button.enable()
        export_button.enable()
        if isinstance(state.source, FigureSource):
            source_status.set_text(
                translator.text(
                    "figures.source_ready",
                    rect_id=state.source.rectangle["rect_id"],
                )
            )
            try:
                destination, _parent = controller._target(state.source)
            except ValueError:
                pass
            else:
                destination_label.set_text(
                    translator.text("figures.destination", path=str(destination))
                )
        if state.preview_path is not None and not state.preview_stale:
            if preview_url_builder is not None:
                preview.set_source(preview_url_builder(state.preview_path))
                preview.set_visibility(True)
            stale_label.set_text(translator.text("figures.preview_fresh"))
        if state.run_path is not None:
            result_label.set_text(
                translator.text("figures.exported", path=str(state.run_path))
            )
        render_messages()

    load_button.on("click", lambda: load_path())
    refresh_button.on("click", refresh_preview)
    export_button.on("click", final_export)
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
