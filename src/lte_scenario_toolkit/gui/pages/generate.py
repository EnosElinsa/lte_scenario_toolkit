"""Explicit, coordinated publication of one locked candidate session."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any

from ...jobs import Job, JobBusyError, JobCoordinator
from ...run_service import RunService
from ..presentation import (
    ActionSpec,
    PresentationSpec,
    artifact_label_presentation,
    artifact_state_presentation,
    render_status_badge,
    render_sticky_action_dock,
    render_technical_details,
)

ARTIFACT_ORDER = (
    "csv",
    "preview_png",
    "terrain_png",
    "terrain_eps",
    "terrain_html",
)
_ARTIFACT_SET = frozenset(ARTIFACT_ORDER)
_ARTIFACT_FORMATS = {
    "csv": "CSV",
    "preview_png": "PNG",
    "terrain_png": "PNG",
    "terrain_eps": "EPS",
    "terrain_html": "HTML",
}
_GENERATION_PHASES = {
    "ready": PresentationSpec("generate.phase.ready"),
    "running": PresentationSpec("generate.phase.running", "active"),
    "cancelling": PresentationSpec("generate.phase.cancelling", "warning"),
    "completed": PresentationSpec("generate.phase.completed", "success"),
    "partial": PresentationSpec("generate.phase.partial", "warning"),
    "error": PresentationSpec("generate.phase.error", "danger"),
}


def generation_action_roles(
    phase: object,
    *,
    can_open_figures: bool,
) -> tuple[tuple[str, str], ...]:
    """Return the single, state-appropriate action sequence for a run."""

    if phase == "ready":
        return (("generate", "primary"),)
    if phase == "running":
        return (("cancel", "secondary"),)
    if phase == "cancelling":
        return (("cancel", "tertiary"),)
    if phase == "completed":
        if can_open_figures:
            return (("open_figures", "primary"), ("open_history", "secondary"))
        return (("open_history", "secondary"),)
    if phase == "partial":
        return (("open_history", "secondary"), ("inspect", "tertiary"))
    if phase == "error":
        return (("retry", "secondary"), ("inspect", "tertiary"))
    return ()


def _ordered_artifacts(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, os.PathLike)):
        raise ValueError("artifacts must be a collection")
    try:
        requested = tuple(values)
    except TypeError as exc:
        raise ValueError("artifacts must be a collection") from exc
    if any(type(value) is not str or value not in _ARTIFACT_SET for value in requested):
        raise ValueError(
            "artifact must be one of: " + ", ".join(ARTIFACT_ORDER)
        )
    if len(set(requested)) != len(requested):
        raise ValueError("artifact selection must not contain duplicates")
    selected = set(requested)
    return tuple(token for token in ARTIFACT_ORDER if token in selected)


@dataclass(frozen=True, slots=True)
class GenerationModel:
    """Read-only confirmation model derived from one frozen candidate session."""

    scenario_id: str
    profile_id: str
    candidate: Any
    output_root: Path
    selected_artifacts: tuple[str, ...] = ARTIFACT_ORDER

    @property
    def can_generate(self) -> bool:
        return bool(self.selected_artifacts)

    def with_artifact(self, token: str, enabled: bool) -> GenerationModel:
        if token not in _ARTIFACT_SET:
            raise ValueError(f"Unknown artifact: {token!r}")
        selected = set(self.selected_artifacts)
        if enabled:
            selected.add(token)
        else:
            selected.discard(token)
        return replace(self, selected_artifacts=_ordered_artifacts(selected))

    def require_artifacts(self) -> tuple[str, ...]:
        if not self.selected_artifacts:
            raise ValueError("At least one artifact must be selected")
        return self.selected_artifacts


def generation_model(session: Any) -> GenerationModel:
    """Validate a locked final candidate without touching the filesystem."""

    candidate = getattr(session, "locked_candidate", None)
    if candidate is None:
        raise ValueError("generation requires one locked candidate")
    result = getattr(session, "scan_result", None)
    if result is None or getattr(result, "completed", None) is not True:
        raise ValueError("generation requires a completed final scan")
    matches = tuple(item for item in getattr(result, "candidates", ()) if item == candidate)
    if len(matches) != 1:
        raise ValueError("locked candidate must occur in exactly one final scan position")
    preflight = getattr(session, "preflight", None)
    profile = getattr(session, "profile_snapshot", None)
    if preflight is None or getattr(preflight, "profile", None) is not profile:
        raise ValueError("generation requires the validated frozen profile snapshot")
    output_value = getattr(preflight, "output_root", None)
    if not isinstance(output_value, (str, os.PathLike)):
        raise ValueError("generation requires a validated output root")
    output_root = Path(output_value).expanduser().resolve(strict=False)
    scenario_id = getattr(profile, "scenario_id", None)
    profile_id = getattr(profile, "profile_id", None)
    if type(scenario_id) is not str or not scenario_id:
        raise ValueError("generation requires a scenario ID")
    if type(profile_id) is not str or not profile_id:
        raise ValueError("generation requires a profile ID")
    return GenerationModel(
        scenario_id=scenario_id,
        profile_id=profile_id,
        candidate=candidate,
        output_root=output_root,
    )


@dataclass(frozen=True, slots=True)
class GenerationState:
    """Immutable generation status suitable for UI polling."""

    phase: str
    requested_artifacts: tuple[str, ...]
    run_path: Path | None = None
    artifact_states: tuple[tuple[str, str], ...] = ()
    errors: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    message: str | None = None

    def artifact_status(self, token: str) -> str:
        for name, status in self.artifact_states:
            if name == token:
                return status
        return "not-requested" if token not in self.requested_artifacts else "pending"


def _published_state(
    value: Any,
    *,
    model: GenerationModel,
    requested: tuple[str, ...],
) -> GenerationState:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("selection export did not return a run directory")
    entry = RunService(model.output_root).entry_for_path(value)
    run_path = entry.run_dir
    record = entry.record
    if record.get("scenario_id") != model.scenario_id:
        raise ValueError("published run scenario does not match the locked session")
    if record.get("profile_id") != model.profile_id:
        raise ValueError("published run profile does not match the locked session")
    status = record.get("status")
    if status not in {"completed", "partial"}:
        raise ValueError("published run status must be completed or partial")
    artifacts = record.get("artifacts")
    errors = record.get("errors")
    metadata = record.get("metadata")
    if not isinstance(artifacts, (list, tuple)) or any(
        type(item) is not str for item in artifacts
    ):
        raise ValueError("published run artifacts must be a list of paths")
    if not isinstance(errors, (list, tuple)) or any(
        not isinstance(item, Mapping) for item in errors
    ):
        raise ValueError("published run errors must be a list of objects")
    if not isinstance(metadata, Mapping):
        raise ValueError("published run metadata must be an object")
    if metadata.get("run_kind") != "selection":
        raise ValueError("published run must be a selection run")
    candidate = metadata.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("published run is missing candidate metadata")
    if candidate.get("flat_grid_id") != model.candidate.flat_grid_id:
        raise ValueError("published run candidate does not match the locked candidate")
    if candidate.get("point_count") != model.candidate.point_count:
        raise ValueError("published run station count does not match the locked candidate")
    for name in ("center_x", "center_y"):
        if name not in candidate or candidate.get(name) != getattr(model.candidate, name):
            raise ValueError("published run candidate geometry does not match")
    recorded_request = metadata.get("requested_artifacts")
    if not isinstance(recorded_request, (list, tuple)) or tuple(
        recorded_request
    ) != requested:
        raise ValueError("published run requested artifacts do not match the request")
    artifact_paths = metadata.get("artifact_paths")
    if not isinstance(artifact_paths, Mapping):
        raise ValueError("published run artifact_paths must be an object")
    states: list[tuple[str, str]] = []
    for token in requested:
        relative = artifact_paths.get(token)
        if relative is None:
            states.append((token, "failed"))
            continue
        if relative not in artifacts:
            raise ValueError("published artifact path is absent from run artifacts")
        states.append((token, "published"))
    published_count = sum(state == "published" for _, state in states)
    if published_count == 0:
        raise ValueError("published run contains none of the requested artifacts")
    if status == "completed" and published_count != len(requested):
        raise ValueError("completed run is missing requested artifacts")
    if status == "completed" and errors:
        raise ValueError("completed run must not contain errors")
    failed_tokens = {
        item.get("artifact") for item in errors if isinstance(item, Mapping)
    }
    missing_tokens = {
        token for token, artifact_status in states if artifact_status == "failed"
    }
    if status == "partial" and failed_tokens != missing_tokens:
        raise ValueError("partial run errors do not match failed artifacts")
    return GenerationState(
        phase=status,
        requested_artifacts=requested,
        run_path=run_path,
        artifact_states=tuple(states),
        errors=tuple(dict(item) for item in errors),
    )


class GenerationController:
    """Publish one immutable candidate through the shared job coordinator."""

    def __init__(
        self,
        session: Any,
        coordinator: JobCoordinator,
        *,
        on_published: Callable[[Path], None] | None = None,
    ) -> None:
        self.session = session
        self.model = generation_model(session)
        self.coordinator = coordinator
        self.on_published = on_published
        self._lock = Lock()
        self._job: Job | None = None
        self._state = GenerationState("ready", self.model.selected_artifacts)
        self._published_notified = False
        self._closed = False

    @property
    def state(self) -> GenerationState:
        with self._lock:
            return self._state

    @property
    def job(self) -> Job | None:
        with self._lock:
            return self._job

    def _notify_published(self, path: Path) -> dict[str, Any] | None:
        callback: Callable[[Path], None] | None
        with self._lock:
            if self._published_notified:
                return None
            self._published_notified = True
            callback = self.on_published
        if callback is not None:
            try:
                callback(path)
            except Exception as exc:
                # The run is already authoritative; a workstation preference failure
                # must not downgrade or delete it.
                return {
                    "code": "generation.settings_failed",
                    "message": str(exc),
                }
        return None

    def start(self, artifacts: Iterable[str]) -> Job:
        requested = _ordered_artifacts(artifacts)
        if not requested:
            raise ValueError("At least one artifact must be selected")
        with self._lock:
            if self._closed:
                raise RuntimeError("generation controller is closed")
            if self._job is not None:
                if self._job.future is not None and not self._job.future.done():
                    raise JobBusyError(
                        "Generation is already running",
                        active_job_id=self._job.job_id,
                        active_kind=self._job.kind,
                        requested_kind="generate",
                    )
                raise RuntimeError("this locked selection has already been generated")

        session = self.session
        service = getattr(session, "selection_service", None)
        preflight = session.preflight
        scan_result = session.scan_result
        candidate = session.locked_candidate
        output_root = self.model.output_root

        def worker(_cancel: Any, _emit: Callable[[Any], None]) -> GenerationState:
            try:
                published = service.export(
                    preflight,
                    scan_result,
                    candidate,
                    output_root=output_root,
                    artifacts=requested,
                    entrypoint=("lte-gui", "generate"),
                )
                result = _published_state(
                    published,
                    model=self.model,
                    requested=requested,
                )
            except Exception as exc:
                result = GenerationState(
                    phase="error",
                    requested_artifacts=requested,
                    artifact_states=tuple((token, "failed") for token in requested),
                    errors=(
                        {
                            "code": getattr(exc, "code", "generation.failed"),
                            "message": str(exc),
                        },
                    ),
                    message=str(exc),
                )
            if result.run_path is not None:
                warning = self._notify_published(result.run_path)
                if warning is not None:
                    result = replace(result, warnings=(warning,))
            return result

        job = self.coordinator.submit("generate", worker)
        with self._lock:
            self._job = job
            self._state = GenerationState("running", requested)

        def release(_future: Any) -> None:
            self.coordinator.finish(job.job_id)

        assert job.future is not None
        job.future.add_done_callback(release)
        return job

    def drain(self, job: Job | None = None) -> GenerationState:
        active = self.job if job is None else job
        if active is None or active.future is None or not active.future.done():
            return self.state
        try:
            state = active.future.result()
        except Exception as exc:
            state = GenerationState(
                phase="error",
                requested_artifacts=self.state.requested_artifacts,
                errors=({"code": "generation.failed", "message": str(exc)},),
                message=str(exc),
            )
        with self._lock:
            if self._job is None or self._job.job_id == active.job_id:
                self._state = state
        self.coordinator.finish(active.job_id)
        return state

    def cancel(self) -> bool:
        """Request cooperative cancellation for this controller's active job."""

        with self._lock:
            job = self._job
            if (
                self._closed
                or job is None
                or job.future is None
                or job.future.done()
                or self._state.phase not in {"running", "cancelling"}
            ):
                return False
            if self._state.phase == "cancelling":
                return False
            job_id = job.job_id
        if not self.coordinator.cancel(job_id):
            return False
        with self._lock:
            if self._job is not None and self._job.job_id == job_id:
                self._state = replace(self._state, phase="cancelling")
                return True
        return False

    def close(self) -> None:
        with self._lock:
            self._closed = True


@dataclass(frozen=True, slots=True)
class GenerationPageView:
    controller: GenerationController
    timer: Any


def render_generation_unavailable(ui: Any, translator: Any) -> None:
    """Render a fail-closed route when no locked opaque session is present."""

    with ui.column().classes("lte-page lte-generate-page"):
        ui.label(translator.text("generate.unavailable")).classes(
            "lte-page-title"
        ).props("role=heading aria-level=1")
        ui.label(translator.text("generate.unavailable_body")).classes(
            "lte-callout lte-callout--warning"
        )
        ui.button(
            translator.text("candidates.back_to_configure"),
            on_click=lambda: ui.navigate.to("/configure"),
        ).props("outline")


def render_generate_page(
    ui: Any,
    translator: Any,
    session: Any,
    coordinator: JobCoordinator,
    *,
    on_published: Callable[[Path], None] | None = None,
    on_complete: Callable[[GenerationState], None] | None = None,
    on_open_figures: Callable[[], None] | None = None,
) -> GenerationPageView:
    """Render immutable confirmation controls and one explicit Generate action."""

    controller = GenerationController(
        session,
        coordinator,
        on_published=on_published,
    )
    model = controller.model
    checkboxes: dict[str, Any] = {}
    artifact_status_holders: dict[str, Any] = {}
    navigated = False
    selection_locked = False
    locked_selection = ARTIFACT_ORDER
    submit: Any | None = None
    action_phase: str | None = None
    technical_details: Any | None = None

    with ui.column().classes("lte-page lte-generate-page"):
        ui.label(translator.text("generate.title")).classes("lte-page-title").props(
            "role=heading aria-level=1"
        )
        ui.label(translator.text("generate.subtitle")).classes("lte-page-subtitle")
        with ui.element("ol").classes("lte-generate-stepper").props(
            'aria-label="' + translator.text("generate.stepper") + '"'
        ):
            for key in ("inputs", "generate", "artifacts"):
                with ui.element("li").classes("lte-generate-step"):
                    ui.label(translator.text(f"generate.step.{key}"))
        with ui.element("div").classes("lte-generate-workspace"):
            with ui.card().classes("lte-generate-summary lte-generate-run-summary"):
                ui.label(translator.text("generate.summary")).classes("lte-section-title")
                ui.label(f"{model.scenario_id} / {model.profile_id}")
                ui.label(
                    translator.text(
                        "generate.candidate",
                        grid_id=model.candidate.flat_grid_id,
                        count=model.candidate.point_count,
                    )
                )
                ui.label(
                    translator.text("generate.destination", path=str(model.output_root))
                ).classes("lte-path")
                phase_holder = ui.row().classes("lte-generation-phase")
            with ui.card().classes("lte-generate-artifacts"):
                ui.label(translator.text("generate.artifacts")).classes("lte-section-title")
                with ui.column().classes("lte-generation-artifact-list"):
                    for token in ARTIFACT_ORDER:
                        presentation = artifact_label_presentation(token)
                        with ui.row().classes("lte-generation-artifact-row").mark(
                            f"generation-artifact-{token}"
                        ):
                            checkboxes[token] = (
                                ui.checkbox(value=True)
                                .props(
                                    'aria-label="'
                                    + translator.text(presentation.label_key)
                                    + '"'
                                )
                                .mark(f"generation-artifact-select-{token}")
                            )
                            with ui.column().classes("lte-generation-artifact-copy"):
                                ui.label(translator.text(presentation.label_key)).classes(
                                    "lte-generation-artifact-name"
                                )
                                assert presentation.description_key is not None
                                ui.label(
                                    translator.text(presentation.description_key)
                                ).classes("lte-generation-artifact-description")
                            ui.badge(_ARTIFACT_FORMATS[token]).classes(
                                "lte-format-badge"
                            )
                            holder = ui.row().classes(
                                "lte-generation-artifact-status"
                            )
                            artifact_status_holders[token] = holder
                            with holder:
                                render_status_badge(
                                    ui,
                                    translator,
                                    artifact_state_presentation("pending"),
                                    marker=f"generation-artifact-status-{token}",
                                )
        selection_guidance = ui.label(
            translator.text("generate.selection_required")
        ).classes("lte-callout lte-callout--warning").mark(
            "generation-selection-guidance"
        )
        selection_guidance.set_visibility(False)
        warning_box = ui.column().classes("lte-generate-warnings")
        error_box = ui.column().classes("lte-generate-errors")
        action_dock_holder = ui.element("div").classes("full-width")

    def render_phase(phase: str) -> None:
        presentation = _GENERATION_PHASES.get(
            phase,
            PresentationSpec("status.unknown"),
        )
        phase_holder.clear()
        with phase_holder:
            if phase == "running":
                ui.spinner(size="sm")
            render_status_badge(
                ui,
                translator,
                presentation,
                marker="generation-phase",
            )

    def render_artifact_status(token: str, raw_status: str) -> None:
        holder = artifact_status_holders[token]
        holder.clear()
        with holder:
            render_status_badge(
                ui,
                translator,
                artifact_state_presentation(raw_status),
                marker=f"generation-artifact-status-{token}",
            )

    def selected_artifacts() -> tuple[str, ...]:
        return tuple(
            token for token in ARTIFACT_ORDER if bool(checkboxes[token].value)
        )

    def sync_ready_selection() -> tuple[str, ...]:
        selected = selected_artifacts()
        if selection_locked or controller.state.phase != "ready":
            return selected
        selected_set = set(selected)
        for token in ARTIFACT_ORDER:
            render_artifact_status(
                token,
                "pending" if token in selected_set else "not-requested",
            )
        if submit is not None:
            submit.set_enabled(bool(selected))
        selection_guidance.set_visibility(not selected)
        return selected

    def selection_changed(token: str, value: object) -> None:
        if selection_locked or controller.state.phase != "ready":
            checkbox = checkboxes[token]
            expected = token in locked_selection
            checkbox.disable()
            if bool(value) != expected:
                checkbox.set_value(expected)
            return
        sync_ready_selection()

    def render_outcome_details(state: GenerationState) -> None:
        nonlocal technical_details
        warning_box.clear()
        error_box.clear()
        technical_details = None
        with warning_box:
            if state.warnings:
                ui.label(translator.text("generate.warning.summary")).classes(
                    "lte-callout lte-callout--warning"
                )
        with error_box:
            if state.phase == "partial":
                ui.label(translator.text("generate.error.partial")).classes(
                    "lte-callout lte-callout--error"
                ).mark("generation-primary-error")
            elif state.phase == "error":
                ui.label(translator.text("generate.error.failed")).classes(
                    "lte-callout lte-callout--error"
                ).mark("generation-primary-error")
            if state.errors or state.warnings or state.message:
                details = {
                    "run_path": None if state.run_path is None else str(state.run_path),
                    "message": state.message,
                    "errors": state.errors,
                    "warnings": state.warnings,
                    "artifact_states": state.artifact_states,
                }

                def render_details() -> None:
                    ui.code(
                        json.dumps(details, ensure_ascii=False, indent=2),
                        language="json",
                    ).classes("lte-technical-copy").mark(
                        "generation-technical-copy"
                    )

                technical_details = render_technical_details(
                    ui,
                    translator.text("generate.technical_details"),
                    render_details,
                    marker="generation-technical-details",
                )

    for artifact_token, checkbox in checkboxes.items():
        checkbox.on_value_change(
            lambda event, token=artifact_token: selection_changed(token, event.value)
        )

    render_phase("ready")
    sync_ready_selection()

    def tick() -> None:
        state = controller.drain()
        render_phase(state.phase)
        for token in artifact_status_holders:
            render_artifact_status(token, state.artifact_status(token))
        refresh_actions(state)
        if state.phase in {"running", "cancelling"}:
            return
        timer.deactivate()
        if submit is not None:
            submit.disable()
        render_outcome_details(state)
        refresh_actions(state)

    def start() -> None:
        nonlocal locked_selection, selection_locked
        selected = selected_artifacts()
        if not selected:
            sync_ready_selection()
            return
        try:
            controller.start(selected)
        except JobBusyError:
            ui.notify(translator.text("error.job_busy"), type="warning")
            return
        except ValueError:
            selection_guidance.set_visibility(True)
            submit.disable()
            return
        except RuntimeError:
            ui.notify(translator.text("operation.failed"), type="warning")
            return
        locked_selection = selected
        selection_locked = True
        if submit is not None:
            submit.disable()
        for checkbox in checkboxes.values():
            checkbox.disable()
        render_phase("running")
        for token in artifact_status_holders:
            render_artifact_status(token, controller.state.artifact_status(token))
        refresh_actions(controller.state)
        timer.activate()

    def open_history() -> None:
        nonlocal navigated
        if not navigated and on_complete is not None:
            navigated = True
            on_complete(controller.state)

    def inspect_details() -> None:
        if technical_details is not None and hasattr(technical_details, "set_value"):
            technical_details.set_value(True)

    def request_cancel() -> None:
        if controller.cancel():
            state = controller.state
            render_phase(state.phase)
            refresh_actions(state)

    def retry_generation() -> None:
        ui.navigate.reload()

    def refresh_actions(state: GenerationState) -> None:
        nonlocal action_phase, submit
        if action_phase == state.phase:
            return
        action_phase = state.phase
        action_dock_holder.clear()
        callbacks: dict[str, Callable[[], None]] = {
            "generate": start,
            "cancel": request_cancel,
            "open_history": open_history,
            "inspect": inspect_details,
            "retry": retry_generation,
        }
        labels = {
            "generate": translator.text("generate.action"),
            "cancel": translator.text("generate.action.cancel"),
            "open_history": translator.text("action.open_history"),
            "inspect": translator.text("action.inspect"),
            "retry": translator.text("generate.action.retry"),
        }
        if on_open_figures is not None:
            callbacks["open_figures"] = on_open_figures
            labels["open_figures"] = translator.text("action.open_figures")
        with action_dock_holder:
            buttons = render_sticky_action_dock(
                ui,
                tuple(
                    ActionSpec(
                        name,
                        labels[name],
                        callbacks[name],
                        role=role,
                        enabled=not (name == "cancel" and state.phase == "cancelling"),
                        marker={
                            "generate": "generation-submit",
                            "cancel": "generation-cancel",
                            "open_figures": "generation-open-figures",
                            "open_history": "generation-open-history",
                            "inspect": "generation-inspect",
                            "retry": "generation-retry",
                        }[name],
                    )
                    for name, role in generation_action_roles(
                        state.phase,
                        can_open_figures=on_open_figures is not None,
                    )
                ),
                label=translator.text("generate.action_dock"),
                marker="generation-action-dock",
                extra_classes="lte-generate-action-dock",
            )
        submit = buttons.get("generate")
        if submit is None:
            # Retain the disabled submit control while the dock exposes the
            # current action, so its stable marker remains available to clients.
            with action_dock_holder:
                submit = (
                    ui.button(translator.text("generate.action"))
                    .classes("lte-action lte-action--tertiary")
                    .props("flat")
                    .mark("generation-submit")
                )
                submit.disable()

    timer = ui.timer(0.1, tick, active=False)
    refresh_actions(controller.state)
    render_phase("ready")
    sync_ready_selection()

    def cleanup() -> None:
        timer.deactivate()
        controller.close()

    ui.context.client.on_delete(cleanup)
    return GenerationPageView(controller=controller, timer=timer)


__all__ = [
    "ARTIFACT_ORDER",
    "GenerationController",
    "GenerationModel",
    "GenerationPageView",
    "GenerationState",
    "generation_model",
    "render_generate_page",
    "render_generation_unavailable",
]
