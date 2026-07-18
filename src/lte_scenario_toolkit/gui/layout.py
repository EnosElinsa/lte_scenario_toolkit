"""Shared production shell for the local NiceGUI workstation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..jobs import JobSnapshot
from .i18n import Translator
from .presentation import job_kind_presentation

_NAVIGATION = (
    ("nav.scenarios", "/scenarios", "dataset"),
    ("nav.configure", "/configure", "tune"),
    ("nav.figures", "/figures", "insert_chart_outlined"),
    ("nav.history", "/history", "history"),
)

_JOB_TONE_CLASSES = " ".join(
    f"lte-job-indicator--{tone}"
    for tone in ("neutral", "info", "warning", "success", "danger", "active")
)


def render_app_shell(
    ui: Any,
    translator: Translator,
    *,
    active_route: str,
    page_context: str,
    get_job_snapshot: Callable[[], JobSnapshot],
    on_language_change: Callable[[Any], None],
) -> Any:
    """Render the responsive shell and return its empty main content element.

    ``page_context`` is translated by the caller because workflow pages can share
    an active navigation route while retaining distinct user-facing contexts.
    """

    drawer = (
        ui.left_drawer(value=False, fixed=True, bordered=False)
        .props("width=224 breakpoint=980 show-if-above")
        .classes("lte-navigation-rail")
        .mark("shell-navigation")
    )
    with drawer:
        with ui.column().classes("lte-rail-layout no-wrap"):
            with ui.row().classes("lte-rail-brand items-center no-wrap"):
                ui.element("span").classes("lte-brand-mark").props("aria-hidden=true")
                with ui.column().classes("lte-brand-copy gap-0"):
                    ui.label("LTE").classes("lte-brand-kicker")
                    ui.label(translator.text("app.title")).classes("lte-brand-title")
            with ui.column().classes("lte-rail-nav gap-1"):
                for key, target, icon in _NAVIGATION:
                    link = ui.link(target=target).classes(
                        "lte-nav__link items-center no-wrap"
                    )
                    link.props(
                        f'aria-label="{translator.text(key)}"'
                    )
                    with link:
                        ui.icon(icon).classes("lte-nav__icon")
                        ui.label(translator.text(key)).classes("lte-nav__label")
                    if target == active_route:
                        link.classes(add="lte-nav__link--active").props(
                            "aria-current=page"
                        )

    with ui.header().classes("lte-command-bar"):
        with ui.row().classes("lte-command-bar__inner items-center no-wrap"):
            (
                ui.button(icon="menu", on_click=drawer.toggle)
                .props(
                    "flat round "
                    f'aria-label="{translator.text("action.open_navigation")}"'
                )
                .classes("lte-shell-menu")
                .mark("shell-menu")
            )
            with ui.column().classes("lte-page-context-wrap gap-0"):
                ui.label(translator.text("shell.eyebrow")).classes(
                    "lte-page-context__eyebrow"
                ).mark("shell-eyebrow")
                ui.label(page_context).classes("lte-page-context").mark(
                    "shell-page-context"
                )
            ui.space()
            job_indicator = (
                ui.label()
                .classes("lte-job-indicator")
                .props("role=status aria-live=polite")
                .mark("shell-job-indicator")
            )
            ui.select(
                {"en": "English", "zh-CN": "\u7b80\u4f53\u4e2d\u6587"},
                value=translator.language,
                on_change=on_language_change,
            ).props(
                "dense borderless options-dense "
                f'aria-label="{translator.text("label.language")}"'
            ).classes("lte-language-select")

    def refresh_job() -> None:
        snapshot = get_job_snapshot()
        if snapshot.active and not snapshot.done:
            presentation = job_kind_presentation(snapshot.kind)
            text = translator.text(presentation.label_key)
            tone = presentation.tone
        else:
            text = translator.text("status.idle")
            tone = "neutral"
        job_indicator.set_text(text)
        job_indicator.classes(
            remove=_JOB_TONE_CLASSES,
            add=f"lte-job-indicator--{tone}",
        )

    refresh_job()
    client = ui.context.client
    refresh_timer: Any | None = None

    def stop_job_refresh() -> None:
        nonlocal refresh_timer
        if refresh_timer is not None:
            refresh_timer.cancel(with_current_invocation=True)
            refresh_timer = None

    def start_job_refresh() -> None:
        nonlocal refresh_timer
        stop_job_refresh()
        if client.is_deleted:
            return
        refresh_timer = ui.timer(0.5, refresh_job, immediate=False)

    client.on_connect(start_job_refresh)
    client.on_disconnect(stop_job_refresh)
    client.on_delete(stop_job_refresh)
    return ui.element("main").classes("lte-main").props("id=app-content")
