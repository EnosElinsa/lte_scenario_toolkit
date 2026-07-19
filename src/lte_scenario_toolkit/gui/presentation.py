"""Semantic, localizable presentation for GUI machine values."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Final, Literal

StatusTone = Literal[
    "neutral",
    "info",
    "warning",
    "success",
    "danger",
    "active",
]


@dataclass(frozen=True, slots=True)
class PresentationSpec:
    """Translation and visual tone for one machine value."""

    label_key: str
    tone: StatusTone = "neutral"
    description_key: str | None = None


ActionRole = Literal["primary", "secondary", "danger"]


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """One framework-light action rendered consistently across GUI pages."""

    name: str
    label: str
    on_click: Callable[..., Any]
    role: ActionRole = "secondary"
    enabled: bool = True
    marker: str | None = None


_UNKNOWN: Final = PresentationSpec("status.unknown")

_READINESS: Final = {
    "ready": PresentationSpec(
        "status.ready",
        "success",
        "readiness.ready.description",
    ),
    "boundary-ready": PresentationSpec(
        "status.boundary_ready",
        "warning",
        "readiness.boundary_ready.description",
    ),
    "dem-pending": PresentationSpec(
        "status.dem_pending",
        "warning",
        "readiness.dem_pending.description",
    ),
    "invalid": PresentationSpec(
        "status.invalid",
        "danger",
        "readiness.invalid.description",
    ),
}

_CACHE: Final = {
    "none": PresentationSpec("cache.none"),
    "hit": PresentationSpec("cache.hit", "success"),
    "miss": PresentationSpec("cache.miss", "info"),
    "forced": PresentationSpec("cache.forced", "info"),
}

_SCAN_MODES: Final = {
    "fast": PresentationSpec("scan.fast"),
    "complete": PresentationSpec("scan.complete"),
}

_ARTIFACT_LABELS: Final = {
    "csv": PresentationSpec(
        "generate.artifact.csv",
        description_key="generate.artifact.csv.description",
    ),
    "preview_png": PresentationSpec(
        "generate.artifact.preview_png",
        description_key="generate.artifact.preview_png.description",
    ),
    "terrain_png": PresentationSpec(
        "generate.artifact.terrain_png",
        description_key="generate.artifact.terrain_png.description",
    ),
    "terrain_eps": PresentationSpec(
        "generate.artifact.terrain_eps",
        description_key="generate.artifact.terrain_eps.description",
    ),
    "terrain_html": PresentationSpec(
        "generate.artifact.terrain_html",
        description_key="generate.artifact.terrain_html.description",
    ),
}

_ARTIFACT_STATES: Final = {
    "not-requested": PresentationSpec("status.not_requested"),
    "pending": PresentationSpec("status.pending", "active"),
    "published": PresentationSpec("status.published", "success"),
    "failed": PresentationSpec("status.failed", "danger"),
}

_RUN_STATES: Final = {
    "completed": PresentationSpec("status.completed", "success"),
    "partial": PresentationSpec("status.partial", "warning"),
}

_JOB_KINDS: Final = {
    "validation.full_checksum": PresentationSpec(
        "job.kind.full_checksum",
        "active",
    ),
    "generate": PresentationSpec("job.kind.generate", "active"),
    "selection.scan": PresentationSpec("job.kind.selection_scan", "active"),
    "candidate.dem_style": PresentationSpec("job.kind.dem_style", "active"),
    "candidate.statistics": PresentationSpec("job.kind.statistics", "active"),
    "figure-source": PresentationSpec("job.kind.figure_source", "active"),
    "figure-preview": PresentationSpec("job.kind.figure_preview", "active"),
    "figure-export": PresentationSpec("job.kind.figure_export", "active"),
    "history.trash_move": PresentationSpec(
        "job.kind.history_trash_move",
        "active",
    ),
    "history.trash_restore": PresentationSpec(
        "job.kind.history_trash_restore",
        "active",
    ),
    "history.trash_purge": PresentationSpec(
        "job.kind.history_trash_purge",
        "active",
    ),
}


def _present(
    value: object,
    mapping: dict[str, PresentationSpec],
) -> PresentationSpec:
    if type(value) is not str:
        return _UNKNOWN
    return mapping.get(value, _UNKNOWN)


def readiness_presentation(value: object) -> PresentationSpec:
    """Present a catalog readiness value without leaking its machine token."""

    return _present(value, _READINESS)


def cache_presentation(value: object) -> PresentationSpec:
    """Present candidate scan cache provenance."""

    return _present(value, _CACHE)


def scan_mode_presentation(value: object) -> PresentationSpec:
    """Present a candidate scan mode."""

    return _present(value, _SCAN_MODES)


def artifact_label_presentation(value: object) -> PresentationSpec:
    """Present a generated artifact kind."""

    return _present(value, _ARTIFACT_LABELS)


def artifact_state_presentation(value: object) -> PresentationSpec:
    """Present the publication state of one generated artifact."""

    return _present(value, _ARTIFACT_STATES)


def run_state_presentation(value: object) -> PresentationSpec:
    """Present a published run state."""

    return _present(value, _RUN_STATES)


def job_kind_presentation(value: object) -> PresentationSpec:
    """Present the operation owned by the process-local job coordinator."""

    return _present(value, _JOB_KINDS)


def _mark(element: Any, marker: str | None) -> Any:
    if marker is not None:
        element.mark(marker)
    return element


def render_status_badge(
    ui: Any,
    translator: Any,
    presentation: PresentationSpec,
    *,
    marker: str | None = None,
) -> Any:
    """Render one localized, semantic status badge."""

    badge = ui.badge(translator.text(presentation.label_key)).classes(
        f"lte-status-badge lte-status-badge--{presentation.tone}"
    )
    return _mark(badge, marker)


def render_action_bar(
    ui: Any,
    actions: Iterable[ActionSpec],
    *,
    sticky: bool = False,
    marker: str | None = None,
) -> dict[str, Any]:
    """Render actions by semantic role and return direct button access by name."""

    buttons: dict[str, Any] = {}
    classes = "lte-action-bar"
    if sticky:
        classes += " lte-action-bar--sticky"
    with _mark(ui.row().classes(classes), marker):
        for action in actions:
            button = ui.button(action.label, on_click=action.on_click).classes(
                f"lte-action lte-action--{action.role}"
            )
            if action.role == "secondary":
                button.props("outline")
            elif action.role == "danger":
                button.props("outline color=negative")
            else:
                button.props("unelevated")
            button.set_enabled(action.enabled)
            _mark(button, action.marker)
            buttons[action.name] = button
    return buttons


def render_page_header(
    ui: Any,
    title: str,
    description: str | None = None,
    actions: Iterable[ActionSpec] = (),
) -> Any:
    """Render a page heading with optional contextual actions."""

    with ui.row().classes("lte-page-header") as header:
        with ui.column().classes("lte-page-header__copy"):
            ui.label(title).classes("lte-page-title").props(
                "role=heading aria-level=1"
            )
            if description:
                ui.label(description).classes("lte-page-subtitle")
        action_items = tuple(actions)
        if action_items:
            render_action_bar(ui, action_items)
    return header


def render_technical_details(
    ui: Any,
    summary: str,
    render_content: Callable[[], Any],
    *,
    marker: str | None = None,
) -> Any:
    """Keep machine-oriented detail available without dominating the workflow."""

    expansion = _mark(
        ui.expansion(summary).classes("lte-technical-details full-width"),
        marker,
    )
    with expansion:
        render_content()
    return expansion


def render_empty_state(
    ui: Any,
    title: str,
    description: str | None = None,
    actions: Iterable[ActionSpec] = (),
    *,
    marker: str | None = None,
) -> Any:
    """Render an intentional empty state with an optional recovery action."""

    with _mark(ui.column().classes("lte-empty-state"), marker) as container:
        ui.icon("inbox").classes("lte-empty-state__icon")
        ui.label(title).classes("lte-section-title")
        if description:
            ui.label(description).classes("lte-page-subtitle")
        action_items = tuple(actions)
        if action_items:
            render_action_bar(ui, action_items)
    return container


def render_loading_state(
    ui: Any,
    label: str,
    *,
    marker: str | None = None,
) -> Any:
    """Render an accessible loading surface for blocking page work."""

    with _mark(ui.row().classes("lte-loading-state"), marker) as container:
        ui.spinner(size="sm")
        ui.label(label).props('role="status" aria-live="polite"')
    return container


__all__ = [
    "ActionRole",
    "ActionSpec",
    "PresentationSpec",
    "StatusTone",
    "artifact_label_presentation",
    "artifact_state_presentation",
    "cache_presentation",
    "job_kind_presentation",
    "readiness_presentation",
    "render_action_bar",
    "render_empty_state",
    "render_loading_state",
    "render_page_header",
    "render_status_badge",
    "render_technical_details",
    "run_state_presentation",
    "scan_mode_presentation",
]
