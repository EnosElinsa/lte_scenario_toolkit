"""Semantic, localizable presentation for GUI machine values."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Literal

from ..run_trash import TrashState

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


ActionRole = Literal["primary", "secondary", "tertiary", "danger"]


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """One framework-light action rendered consistently across GUI pages."""

    name: str
    label: str
    on_click: Callable[..., Any]
    role: ActionRole = "secondary"
    enabled: bool = True
    marker: str | None = None


@dataclass(frozen=True, slots=True)
class MenuActionSpec:
    """One semantic overflow-menu action, independent of page state."""

    label: str
    icon: str | None
    handler: Callable[..., Any]
    enabled: bool = True
    separator: bool = False
    marker: str | None = None
    role: ActionRole = "secondary"


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


class TrashAction(str, Enum):
    """User actions supported by one transactional Trash state."""

    RESTORE = "restore"
    PURGE = "purge"
    RECOVER = "recover"


TRASH_ACTIONS_BY_STATE: Final = {
    TrashState.TRASHED: (TrashAction.RESTORE, TrashAction.PURGE),
    TrashState.RECOVERY_REQUIRED: (TrashAction.RECOVER,),
    TrashState.PURGE_FAILED: (TrashAction.PURGE,),
}


_TRASH_STATES: Final = {
    TrashState.TRASHED.value: PresentationSpec("trash.state.trashed", "success"),
    TrashState.RECOVERY_REQUIRED.value: PresentationSpec(
        "trash.state.recovery_required", "warning"
    ),
    TrashState.PURGE_FAILED.value: PresentationSpec(
        "trash.state.purge_failed", "danger"
    ),
    TrashState.MOVING.value: PresentationSpec("trash.state.moving", "active"),
    TrashState.RESTORING.value: PresentationSpec("trash.state.restoring", "active"),
    TrashState.PURGING.value: PresentationSpec("trash.state.purging", "active"),
}


_TRASH_ACTIONS: Final = {
    TrashAction.RESTORE.value: PresentationSpec("trash.action.restore"),
    TrashAction.PURGE.value: PresentationSpec("trash.action.purge", "danger"),
    TrashAction.RECOVER.value: PresentationSpec("trash.action.recover", "warning"),
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


def trash_state_presentation(value: object) -> PresentationSpec:
    """Present a Trash state and fail closed for unknown values."""

    token = value.value if isinstance(value, Enum) else value
    return _present(token, _TRASH_STATES)


def trash_action_presentation(value: object) -> PresentationSpec:
    """Present a Trash action and fail closed for unknown values."""

    token = value.value if isinstance(value, Enum) else value
    return _present(token, _TRASH_ACTIONS)


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
            elif action.role == "tertiary":
                button.props("flat")
            else:
                button.props("unelevated")
            button.set_enabled(action.enabled)
            _mark(button, action.marker)
            buttons[action.name] = button
    return buttons


def render_overflow_menu(
    ui: Any,
    actions: Iterable[MenuActionSpec],
    *,
    icon: str = "more_vert",
    label: str = "More actions",
    marker: str | None = None,
) -> Any:
    """Render a trigger-owned overflow menu from semantic action specifications."""

    trigger = _mark(
        ui.button(icon=icon)
        .props(add=f'flat round aria-haspopup=menu aria-label="{label}"')
        .classes("lte-overflow-menu__trigger"),
        marker,
    )
    with trigger:
        with ui.menu().classes("lte-overflow-menu"):
            for action in actions:
                if action.separator:
                    ui.separator().classes("lte-overflow-menu__separator")
                item = ui.menu_item(
                    action.label,
                    on_click=action.handler if action.enabled else None,
                ).classes("lte-overflow-menu__item")
                if action.role == "danger":
                    item.classes("lte-overflow-menu__item--danger text-negative")
                if not action.enabled:
                    item.props(add="disable aria-disabled=true")
                _mark(item, action.marker)
                if action.icon is not None:
                    with item:
                        ui.icon(action.icon).classes("lte-overflow-menu__icon")
    return trigger


def render_sticky_action_dock(
    ui: Any,
    actions: Iterable[ActionSpec],
    *,
    label: str = "Page actions",
    marker: str | None = None,
) -> dict[str, Any]:
    """Render a semantic action region fixed to the current workflow surface."""

    with _mark(
        ui.element("footer")
        .classes("lte-action-dock")
        .props(add=f'role=region aria-label="{label}"'),
        marker,
    ):
        return render_action_bar(ui, actions, sticky=True)


def render_inspector_drawer(
    ui: Any,
    title: str,
    render_content: Callable[[], Any],
    *,
    value: bool = False,
    on_value_change: Callable[..., Any] | None = None,
    marker: str | None = None,
) -> Any:
    """Render a detail drawer while leaving its open state to the caller."""

    drawer = _mark(
        ui.right_drawer(value=value, fixed=True, bordered=True).classes(
            "lte-inspector-drawer"
        ),
        marker,
    )
    if on_value_change is not None:
        drawer.on_value_change(on_value_change)
    with drawer:
        ui.label(title).classes("lte-inspector-drawer__title").props(
            add="role=heading aria-level=2"
        )
        render_content()
    return drawer


def render_section_heading(
    ui: Any,
    title: str,
    description: str | None = None,
    actions: Iterable[ActionSpec] = (),
    *,
    marker: str | None = None,
) -> Any:
    """Render a reusable second-level section heading with optional actions."""

    with _mark(ui.row().classes("lte-section-heading"), marker) as header:
        with ui.column().classes("lte-section-heading__copy"):
            ui.label(title).classes("lte-section-title").props(
                add="role=heading aria-level=2"
            )
            if description:
                ui.label(description).classes("lte-section-heading__description")
        action_items = tuple(actions)
        if action_items:
            render_action_bar(ui, action_items)
    return header


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
    "MenuActionSpec",
    "PresentationSpec",
    "StatusTone",
    "TRASH_ACTIONS_BY_STATE",
    "TrashAction",
    "artifact_label_presentation",
    "artifact_state_presentation",
    "cache_presentation",
    "job_kind_presentation",
    "readiness_presentation",
    "render_action_bar",
    "render_empty_state",
    "render_inspector_drawer",
    "render_loading_state",
    "render_overflow_menu",
    "render_page_header",
    "render_section_heading",
    "render_sticky_action_dock",
    "render_status_badge",
    "render_technical_details",
    "run_state_presentation",
    "scan_mode_presentation",
    "trash_action_presentation",
    "trash_state_presentation",
]
