"""Scenario page models, validation jobs, and framework-light rendering."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from ...data_validation import validate_scenario_data
from ...jobs import Job, JobBusyError, JobCoordinator
from ..presentation import (
    ActionSpec,
    MenuActionSpec,
    readiness_presentation,
    render_action_bar,
    render_inspector_drawer,
    render_overflow_menu,
    render_page_header,
    render_status_badge,
    render_technical_details,
)
from ..scenario_previews import ScenarioPreviewRequest, ScenarioPreviewResult


@dataclass(frozen=True, slots=True)
class ScenarioCard:
    """Immutable catalog summary rendered by the scenario page."""

    scenario_id: str
    display_name: str
    status: str
    boundary_dataset_id: str
    boundary_entrypoint: str
    dem_dataset_id: str | None
    dem_entrypoint: str | None
    default_profile_path: str | None
    can_run: bool


@dataclass(frozen=True, slots=True)
class ValidationDiagnostic:
    """Immutable display copy of one domain validation message."""

    level: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Immutable validation result suitable for background-job transfer."""

    scenario_id: str
    status: str
    ok: bool
    messages: tuple[ValidationDiagnostic, ...]
    full_checksum: bool


@dataclass(frozen=True, slots=True)
class ValidationJobUpdate:
    """One immutable poll result for a full-checksum job."""

    done: bool
    result: ValidationResult | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ScenarioCatalogView:
    """Live cover holders for one rendered scenario catalog."""

    ui: Any
    translator: Any
    cards: tuple[ScenarioCard, ...]
    preview_holders: tuple[Any, ...]

    def clear(self) -> None:
        """Clear only preview covers, preserving the shell and catalog controls."""

        for holder in self.preview_holders:
            holder.clear()


_JOB_COORDINATOR: JobCoordinator | None = None
_JOB_COORDINATOR_LOCK = Lock()


def get_job_coordinator() -> JobCoordinator:
    """Return the one application-wide CPU job coordinator."""

    global _JOB_COORDINATOR
    with _JOB_COORDINATOR_LOCK:
        if _JOB_COORDINATOR is None:
            _JOB_COORDINATOR = JobCoordinator()
        return _JOB_COORDINATOR


def shutdown_job_coordinator() -> None:
    """Shut down the shared executor and allow a later app lifecycle to recreate it."""

    global _JOB_COORDINATOR
    with _JOB_COORDINATOR_LOCK:
        coordinator = _JOB_COORDINATOR
        _JOB_COORDINATOR = None
    if coordinator is not None:
        coordinator.shutdown()


def scenario_cards(catalog: Any) -> tuple[ScenarioCard, ...]:
    """Return all catalog scenarios in their declared order."""

    cards: list[ScenarioCard] = []
    for scenario_id in getattr(catalog, "scenarios_by_id", {}):
        scenario = catalog.scenario(scenario_id)
        boundary_id = scenario["boundary_dataset_id"]
        boundary = catalog.dataset(boundary_id)
        dem_id = scenario.get("dem_dataset_id")
        dem = None if dem_id is None else catalog.dataset(dem_id)
        status = catalog.scenario_status(scenario_id)
        cards.append(
            ScenarioCard(
                scenario_id=scenario_id,
                display_name=str(scenario["display_name"]),
                status=status,
                boundary_dataset_id=str(boundary_id),
                boundary_entrypoint=str(boundary["entrypoint"]),
                dem_dataset_id=None if dem_id is None else str(dem_id),
                dem_entrypoint=None if dem is None else str(dem["entrypoint"]),
                default_profile_path=(
                    None
                    if scenario.get("config_path") is None
                    else str(scenario["config_path"])
                ),
                can_run=status == "ready",
            )
        )
    return tuple(cards)


def _catalog_preview_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or isinstance(value, bytes):
        raise ValueError(f"{label} must be a repository-local path")
    text = os.fspath(value)
    if not isinstance(text, str) or "://" in text or "\x00" in text:
        raise ValueError(f"{label} must be a repository-local path")
    raw = Path(text).expanduser()
    if ".." in raw.parts:
        raise ValueError(f"{label} must not contain traversal components")
    candidate = raw if raw.is_absolute() else root / raw
    candidate = candidate.resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise ValueError(f"{label} must stay inside the repository")
    return candidate


def scenario_preview_requests(catalog: Any) -> tuple[ScenarioPreviewRequest, ...]:
    """Map declared catalog scenarios to repository-bounded preview requests."""

    root = Path(catalog.root).expanduser().resolve(strict=True)
    requests: list[ScenarioPreviewRequest] = []
    for card in scenario_cards(catalog):
        requests.append(
            ScenarioPreviewRequest(
                card.scenario_id,
                card.display_name,
                _catalog_preview_path(root, card.boundary_entrypoint, "boundary"),
                allowed_root=root,
                dem_path=(
                    None
                    if card.dem_entrypoint is None
                    else _catalog_preview_path(root, card.dem_entrypoint, "DEM")
                ),
            )
        )
    return tuple(requests)


def run_validation(
    catalog: Any,
    scenario_id: str,
    *,
    full_checksum: bool = False,
    dataset_ids: Iterable[str] = (),
) -> ValidationResult:
    """Run domain validation and return an immutable, print-free display copy."""

    requested_dataset_ids = tuple(dataset_ids)
    report = validate_scenario_data(
        catalog,
        scenario_id,
        full_checksum=full_checksum,
        dataset_ids=requested_dataset_ids,
    )
    return ValidationResult(
        scenario_id=str(report.scenario_id),
        status=str(report.status),
        ok=bool(report.ok),
        messages=tuple(
            ValidationDiagnostic(
                level=str(message.level),
                code=str(message.code),
                message=str(message.message),
            )
            for message in report.messages
        ),
        full_checksum=full_checksum,
    )


def submit_full_checksum(
    catalog: Any,
    scenario_id: str,
    *,
    dataset_ids: Iterable[str] = (),
    coordinator: JobCoordinator | None = None,
) -> Job:
    """Submit full validation without capturing or calling NiceGUI in the worker."""

    requested_dataset_ids = tuple(dataset_ids)

    def worker(_cancel, emit):
        result = run_validation(
            catalog,
            scenario_id,
            full_checksum=True,
            dataset_ids=requested_dataset_ids,
        )
        emit(result)
        return result

    active_coordinator = (
        get_job_coordinator() if coordinator is None else coordinator
    )
    job = active_coordinator.submit(
        "validation.full_checksum",
        worker,
    )
    if job.future is not None:
        job.future.add_done_callback(
            lambda _future: active_coordinator.finish(job.job_id)
        )
    return job


def poll_validation_job(
    coordinator: JobCoordinator,
    job: Job,
) -> ValidationJobUpdate:
    """Collect a finished validation job and always release its coordinator slot."""

    if job.future is None or not job.future.done():
        return ValidationJobUpdate(done=False)
    try:
        result = job.future.result()
        if not isinstance(result, ValidationResult):
            raise TypeError("full-checksum worker returned an invalid result")
        return ValidationJobUpdate(done=True, result=result)
    except Exception as exc:  # the UI must surface worker failures without crashing
        return ValidationJobUpdate(done=True, error=str(exc))
    finally:
        coordinator.finish(job.job_id)


def _render_validation_result(ui: Any, translator: Any, result: ValidationResult) -> None:
    outcome_key = "validation.passed" if result.ok else "validation.failed"
    ui.label(translator.text(outcome_key)).classes(
        "lte-validation-result lte-validation-result--ok"
        if result.ok
        else "lte-validation-result lte-validation-result--error"
    )
    if result.messages:
        def render_messages() -> None:
            for message in result.messages:
                ui.label(
                    f"{message.level.upper()} {message.code}: {message.message}"
                ).classes("lte-validation-message")

        render_technical_details(
            ui,
            translator.text("validation.details"),
            render_messages,
        )


def render_scenarios_page(
    ui: Any,
    translator: Any,
    catalog: Any,
    *,
    coordinator: JobCoordinator | None = None,
) -> ScenarioCatalogView:
    """Render a map-first catalog and return its isolated preview holders."""

    active_coordinator = (
        get_job_coordinator() if coordinator is None else coordinator
    )
    cards = scenario_cards(catalog)
    ready_count = sum(card.can_run for card in cards)
    preparation_count = len(cards) - ready_count
    preview_holders: list[Any] = []
    inspector_content: Any | None = None
    drawer: Any | None = None

    def inspector_body() -> None:
        nonlocal inspector_content
        inspector_content = ui.column().classes(
            "lte-scenario-inspector__content full-width"
        )

    def show_inspector(
        card: ScenarioCard,
        section: str,
        *,
        result: ValidationResult | None = None,
        error: str | None = None,
    ) -> None:
        assert inspector_content is not None
        assert drawer is not None
        inspector_content.clear()
        status = readiness_presentation(card.status)
        with inspector_content:
            with ui.row().classes("items-center justify-between no-wrap full-width"):
                ui.label(card.display_name).classes("lte-card-title")
                ui.button(
                    icon="close",
                    on_click=drawer.hide,
                ).props(
                    f'flat round aria-label="{translator.text("action.close")}"'
                )
            render_status_badge(ui, translator, status)
            ui.label(
                translator.text(
                    "dataset.boundary",
                    dataset_id=card.boundary_dataset_id,
                    path=card.boundary_entrypoint,
                )
            ).classes("lte-technical-copy")
            ui.label(
                translator.text(
                    "dataset.dem",
                    dataset_id=card.dem_dataset_id or translator.text("value.none"),
                    path=card.dem_entrypoint or translator.text("value.none"),
                )
            ).classes("lte-technical-copy")
            ui.label(
                translator.text(
                    "dataset.profile",
                    path=card.default_profile_path or translator.text("value.none"),
                )
            ).classes("lte-technical-copy")
            if section == "commands":
                ui.label(translator.text("guidance.commands")).classes(
                    "lte-section-title"
                )
                ui.label(translator.text("guidance.replace_path")).classes(
                    "lte-setup-guidance"
                )
                for command in (
                    f"lte-data validate {card.scenario_id}",
                    f"lte-data validate {card.scenario_id} --full-checksum",
                    f"lte-data dem export {card.scenario_id} --dry-run",
                    f"lte-data dem ingest {card.scenario_id} --tiles-dir '<path>'",
                ):
                    ui.code(command, language="powershell").classes("lte-cli-guidance")
            elif section == "validation":
                if result is not None:
                    _render_validation_result(ui, translator, result)
                elif error is not None:
                    ui.label(translator.text("validation.failed")).classes(
                        "lte-validation-result lte-validation-result--error"
                    )
                    ui.label(error).classes("lte-validation-message")
                else:
                    ui.spinner(size="sm")
                    ui.label(translator.text("validation.running")).props(
                        'role="status" aria-live="polite"'
                    )
            ui.label(
                translator.text(
                    "scenarios.next_step.ready"
                    if card.can_run
                    else "scenarios.next_step.preparation"
                )
            ).classes("lte-scenario-next-step")
        drawer.show()

    with ui.column().classes("lte-page lte-scenarios-page"):
        ui.label(translator.text("scenarios.breadcrumb")).classes(
            "lte-page-breadcrumb"
        )
        render_page_header(
            ui,
            translator.text("scenarios.title"),
            translator.text("scenarios.subtitle"),
            actions=(
                ActionSpec(
                    "refresh",
                    translator.text("action.refresh"),
                    ui.navigate.reload,
                    marker="scenarios-refresh",
                ),
            ),
        )
        with ui.card().classes("lte-summary-strip").mark("scenarios-summary"):
            ui.label(translator.text("scenarios.summary")).classes(
                "lte-summary-strip__title"
            )
            with ui.row().classes("lte-summary-metrics"):
                for marker, value, label in (
                    (
                        "scenarios-count-total",
                        len(cards),
                        translator.text("scenarios.count.total"),
                    ),
                    (
                        "scenarios-count-ready",
                        ready_count,
                        translator.text("scenarios.count.ready"),
                    ),
                    (
                        "scenarios-count-preparation",
                        preparation_count,
                        translator.text("scenarios.count.preparation"),
                    ),
                ):
                    with ui.column().classes("lte-summary-metric"):
                        ui.label(str(value)).classes("lte-summary-metric__value").mark(
                            marker
                        )
                        ui.label(label).classes("lte-summary-metric__label")
        drawer = render_inspector_drawer(
            ui,
            translator.text("scenarios.inspector.title"),
            inspector_body,
            marker="scenario-inspector",
        )
        with ui.element("div").classes("lte-card-grid lte-scenario-grid").mark(
            "scenarios-grid"
        ):
            for card in cards:
                preview_holders.append(
                    _render_scenario_card(
                        ui,
                        translator,
                        catalog,
                        card,
                        active_coordinator,
                        show_inspector,
                    )
                )

        with ui.card().classes("lte-guidance-card"):
            ui.label(translator.text("guidance.title")).classes("lte-section-title")
            ui.label(translator.text("guidance.register"))
            ui.code("lte-data scenario add --help", language="powershell").classes(
                "lte-cli-guidance"
            )
            ui.label(translator.text("guidance.dem"))
            ui.code("lte-data dem export --help", language="powershell").classes(
                "lte-cli-guidance"
            )
            ui.code("lte-data dem ingest --help", language="powershell").classes(
                "lte-cli-guidance"
            )
    return ScenarioCatalogView(
        ui=ui,
        translator=translator,
        cards=cards,
        preview_holders=tuple(preview_holders),
    )


def _render_preview_cover(
    ui: Any,
    translator: Any,
    card: ScenarioCard,
    result: ScenarioPreviewResult | None = None,
    *,
    asset_url: Callable[[Path], str] | None = None,
    diagnostic: str | None = None,
) -> None:
    with ui.element("div").classes("lte-scenario-cover"):
        if result is None:
            ui.element("div").classes("lte-scenario-cover__skeleton").mark(
                f"scenario-preview-loading-{card.scenario_id}"
            ).props('role="status" aria-live="polite"')
            kind_key = "scenarios.preview.loading"
        else:
            kind_key = f"scenarios.preview.{result.kind}"
            try:
                if asset_url is None:
                    raise ValueError("preview asset URL is unavailable")
                url = asset_url(result.path)
                image = ui.image(url).classes("lte-scenario-cover__image").mark(
                    f"scenario-preview-image-{card.scenario_id}"
                )
                image._props["alt"] = translator.text(
                    "scenarios.preview.alt",
                    scenario=card.display_name,
                    kind=translator.text(kind_key),
                )
            except Exception as exc:
                ui.icon("map").classes("lte-scenario-cover__fallback")
                diagnostic = diagnostic or result.diagnostic or str(exc)
                kind_key = "scenarios.preview.fallback"
        ui.label(translator.text(kind_key)).classes(
            "lte-scenario-cover__kind"
        ).mark(f"scenario-preview-kind-{card.scenario_id}")
        if diagnostic or (result is not None and result.diagnostic):
            ui.label(diagnostic or result.diagnostic).classes(
                "lte-scenario-cover__diagnostic"
            ).props('role="status"')


def render_scenario_preview_results(
    view: ScenarioCatalogView,
    results: Sequence[object],
    asset_url: Callable[[Path], str],
) -> None:
    """Replace each cover independently so one bad result cannot block peers."""

    for index, (card, holder) in enumerate(
        zip(view.cards, view.preview_holders, strict=True)
    ):
        raw_result = results[index] if index < len(results) else None
        result: ScenarioPreviewResult | None = None
        try:
            kind = raw_result.kind
            path = raw_result.path
            cache_hit = raw_result.cache_hit
            diagnostic = raw_result.diagnostic
            if kind not in {"terrain", "boundary", "fallback"}:
                raise ValueError("invalid preview kind")
            if not isinstance(path, (str, os.PathLike)) or isinstance(path, bytes):
                raise ValueError("invalid preview path")
            if type(cache_hit) is not bool:
                raise ValueError("invalid preview cache flag")
            if diagnostic is not None and not isinstance(diagnostic, str):
                raise ValueError("invalid preview diagnostic")
            result = ScenarioPreviewResult(
                kind,
                Path(path),
                cache_hit,
                diagnostic,
            )
        except (AttributeError, TypeError, ValueError):
            pass
        holder.clear()
        with holder:
            if result is not None:
                _render_preview_cover(
                    view.ui,
                    view.translator,
                    card,
                    result,
                    asset_url=asset_url,
                )
            else:
                _render_preview_cover(
                    view.ui,
                    view.translator,
                    card,
                    ScenarioPreviewResult(
                        "fallback",
                        Path("missing-preview.png"),
                        False,
                        view.translator.text("scenarios.preview.failed"),
                    ),
                    asset_url=asset_url,
                )


def _render_scenario_card(
    ui: Any,
    translator: Any,
    catalog: Any,
    card: ScenarioCard,
    coordinator: JobCoordinator,
    show_inspector: Callable[..., None],
) -> Any:
    status = readiness_presentation(card.status)
    with ui.card().classes("lte-scenario-card").mark(
        f"scenario-card-{card.scenario_id}"
    ):
        preview_holder = ui.element("div").classes(
            "lte-scenario-cover-holder"
        ).mark(f"scenario-preview-{card.scenario_id}")
        with preview_holder:
            _render_preview_cover(ui, translator, card)
        with ui.row().classes("items-center justify-between no-wrap full-width"):
            ui.label(card.display_name).classes("lte-card-title")
            with ui.row().classes("items-center no-wrap"):
                render_status_badge(
                    ui,
                    translator,
                    status,
                    marker=f"scenario-status-{card.scenario_id}",
                )
        if status.description_key is not None:
            ui.label(translator.text(status.description_key)).classes(
                "lte-scenario-card__description"
            )

        ui.label(
            translator.text(
                "scenarios.dataset_summary",
                boundary=card.boundary_dataset_id,
                dem=card.dem_dataset_id or translator.text("value.none"),
                profile=(
                    translator.text("scenarios.available")
                    if card.default_profile_path
                    else translator.text("scenarios.not_available")
                ),
            )
        ).classes("lte-scenario-dataset-summary")
        validation_generation = {"value": 0}
        active_timer: dict[str, Any | None] = {"value": None}
        menu_items: dict[str, Any] = {}

        def begin_validation_generation() -> int:
            previous = active_timer["value"]
            if previous is not None:
                previous.deactivate()
                active_timer["value"] = None
            validation_generation["value"] += 1
            return validation_generation["value"]

        def validate_fast() -> None:
            begin_validation_generation()
            try:
                result = run_validation(catalog, card.scenario_id)
            except Exception as exc:
                show_inspector(card, "validation", error=str(exc))
                return
            show_inspector(card, "validation", result=result)

        def validate_full() -> None:
            generation = begin_validation_generation()
            fast_button = menu_items[f"scenario-validate-{card.scenario_id}"]
            full_button = menu_items[f"scenario-checksum-{card.scenario_id}"]
            fast_button.set_enabled(False)
            full_button.set_enabled(False)
            try:
                job = submit_full_checksum(
                    catalog,
                    card.scenario_id,
                    coordinator=coordinator,
                )
            except JobBusyError as exc:
                fast_button.set_enabled(True)
                full_button.set_enabled(True)
                show_inspector(card, "validation", error=str(exc))
                return
            except Exception as exc:
                fast_button.set_enabled(True)
                full_button.set_enabled(True)
                show_inspector(card, "validation", error=str(exc))
                return
            show_inspector(card, "validation")

            def collect() -> None:
                if validation_generation["value"] != generation:
                    timer.deactivate()
                    return
                update = poll_validation_job(coordinator, job)
                if not update.done:
                    return
                timer.deactivate()
                active_timer["value"] = None
                fast_button.set_enabled(True)
                full_button.set_enabled(True)
                if update.result is not None:
                    show_inspector(card, "validation", result=update.result)
                else:
                    show_inspector(card, "validation", error=update.error or "")

            timer = ui.timer(0.2, collect, active=False)
            active_timer["value"] = timer
            timer.activate()

        workflow_action = (
            ActionSpec(
                "workflow",
                translator.text("action.configure_and_scan"),
                lambda: ui.navigate.to(f"/configure/{card.scenario_id}"),
                role="primary",
                marker=f"scenario-configure-{card.scenario_id}",
            )
            if card.can_run
            else ActionSpec(
                "workflow",
                translator.text("action.view_setup"),
                lambda: show_inspector(card, "commands"),
                role="primary",
                marker=f"scenario-guidance-{card.scenario_id}",
            )
        )
        with ui.row().classes("lte-scenario-support-row"):
            render_overflow_menu(
                ui,
                (
                    MenuActionSpec(
                        translator.text("action.validate"),
                        "fact_check",
                        validate_fast,
                        marker=f"scenario-validate-{card.scenario_id}",
                    ),
                    MenuActionSpec(
                        translator.text("action.full_checksum"),
                        "verified",
                        validate_full,
                        marker=f"scenario-checksum-{card.scenario_id}",
                    ),
                    MenuActionSpec(
                        translator.text("dataset.details"),
                        "database",
                        lambda: show_inspector(card, "details"),
                        separator=True,
                        marker=f"scenario-technical-{card.scenario_id}",
                    ),
                    MenuActionSpec(
                        translator.text("guidance.commands"),
                        "terminal",
                        lambda: show_inspector(card, "commands"),
                        marker=f"scenario-setup-{card.scenario_id}",
                    ),
                ),
                label=translator.text("scenarios.more_actions"),
                marker=f"scenario-overflow-{card.scenario_id}",
                item_sink=menu_items,
            )
        render_action_bar(
            ui,
            (workflow_action,),
        )
    return preview_holder


__all__ = [
    "ScenarioCatalogView",
    "ScenarioCard",
    "ValidationDiagnostic",
    "ValidationJobUpdate",
    "ValidationResult",
    "get_job_coordinator",
    "poll_validation_job",
    "render_scenario_preview_results",
    "render_scenarios_page",
    "run_validation",
    "scenario_cards",
    "scenario_preview_requests",
    "shutdown_job_coordinator",
    "submit_full_checksum",
]
