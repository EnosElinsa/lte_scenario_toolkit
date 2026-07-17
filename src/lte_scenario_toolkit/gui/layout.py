"""Framework-light shell composition for the local NiceGUI application."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .i18n import Translator

_NAVIGATION = (
    ("nav.scenarios", "/"),
    ("nav.configure", "/configure"),
    ("nav.figures", "/figures"),
    ("nav.history", "/history"),
)


def render_app_shell(
    ui: Any,
    translator: Translator,
    *,
    active_route: str,
    active_job: str | None,
    on_language_change: Callable[[Any], None],
) -> Any:
    """Render the shared top bar and return the empty main content element."""

    with ui.header().classes("lte-topbar"):
        with ui.row().classes("lte-topbar__inner items-center no-wrap"):
            ui.label(translator.text("app.title")).classes("lte-brand")
            with ui.row().classes("lte-nav items-center no-wrap"):
                for key, target in _NAVIGATION:
                    link = ui.link(translator.text(key), target).classes("lte-nav__link")
                    if target == active_route:
                        link.classes(add="lte-nav__link--active").props("aria-current=page")
            with ui.row().classes("lte-topbar__utilities items-center no-wrap"):
                ui.label(
                    translator.text("job.running", name=active_job)
                    if active_job
                    else translator.text("status.idle")
                ).classes("lte-job-status")
                ui.label(translator.text("status.ready")).classes("lte-app-status")
                ui.select(
                    {"en": "English", "zh-CN": "\u7b80\u4f53\u4e2d\u6587"},
                    value=translator.language,
                    on_change=on_language_change,
                ).props(
                    "dense borderless options-dense "
                    f"aria-label={translator.text('label.language')}"
                ).classes("lte-language-select")
    return ui.element("main").classes("lte-main").props("id=app-content")
