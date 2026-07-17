"""Scenario page models, validation jobs, and framework-light rendering."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from threading import Lock
from typing import Any

from ...data_validation import validate_scenario_data
from ...jobs import Job, JobBusyError, JobCoordinator


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
    for message in result.messages:
        ui.label(f"{message.level.upper()} {message.code}: {message.message}").classes(
            "lte-validation-message"
        )


def render_scenarios_page(
    ui: Any,
    translator: Any,
    catalog: Any,
    *,
    coordinator: JobCoordinator | None = None,
) -> None:
    """Render scenario cards while keeping all worker code outside NiceGUI."""

    active_coordinator = (
        get_job_coordinator() if coordinator is None else coordinator
    )
    with ui.column().classes("lte-page lte-scenarios-page"):
        ui.label(translator.text("scenarios.title")).classes("lte-page-title")
        ui.label(translator.text("scenarios.subtitle")).classes("lte-page-subtitle")
        with ui.row().classes("lte-card-grid"):
            for card in scenario_cards(catalog):
                _render_scenario_card(ui, translator, catalog, card, active_coordinator)

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


def _render_scenario_card(
    ui: Any,
    translator: Any,
    catalog: Any,
    card: ScenarioCard,
    coordinator: JobCoordinator,
) -> None:
    with ui.card().classes("lte-scenario-card"):
        with ui.row().classes("items-center justify-between no-wrap full-width"):
            ui.label(card.display_name).classes("lte-card-title")
            ui.chip(card.status).classes(f"lte-status-chip lte-status-chip--{card.status}")
        ui.label(card.scenario_id).classes("lte-card-id")
        with ui.expansion(translator.text("dataset.details")).classes(
            "lte-dataset-details full-width"
        ):
            ui.label(
                translator.text(
                    "dataset.boundary",
                    dataset_id=card.boundary_dataset_id,
                    path=card.boundary_entrypoint,
                )
            )
            ui.label(
                translator.text(
                    "dataset.dem",
                    dataset_id=card.dem_dataset_id or translator.text("value.none"),
                    path=card.dem_entrypoint or translator.text("value.none"),
                )
            )
            ui.label(
                translator.text(
                    "dataset.profile",
                    path=card.default_profile_path or translator.text("value.none"),
                )
            )
        validation_area = ui.column().classes("lte-validation-area full-width")
        validation_generation = {"value": 0}
        active_timer: dict[str, Any | None] = {"value": None}

        def begin_validation_generation() -> int:
            previous = active_timer["value"]
            if previous is not None:
                previous.deactivate()
                active_timer["value"] = None
            validation_generation["value"] += 1
            validation_area.clear()
            return validation_generation["value"]

        def validate_fast() -> None:
            begin_validation_generation()
            try:
                result = run_validation(catalog, card.scenario_id)
            except Exception as exc:
                with validation_area:
                    ui.label(translator.text("validation.failed")).classes(
                        "lte-validation-result lte-validation-result--error"
                    )
                    with ui.expansion(translator.text("validation.details")):
                        ui.label(str(exc)).classes("lte-validation-message")
                return
            with validation_area:
                _render_validation_result(ui, translator, result)

        def validate_full() -> None:
            generation = begin_validation_generation()
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
                with validation_area:
                    ui.label(translator.text("error.job_busy")).classes(
                        "lte-validation-result lte-validation-result--error"
                    )
                    with ui.expansion(translator.text("validation.details")):
                        ui.label(str(exc)).classes("lte-validation-message")
                return
            except Exception as exc:
                fast_button.set_enabled(True)
                full_button.set_enabled(True)
                with validation_area:
                    ui.label(translator.text("validation.failed")).classes(
                        "lte-validation-result lte-validation-result--error"
                    )
                    with ui.expansion(translator.text("validation.details")):
                        ui.label(str(exc)).classes("lte-validation-message")
                return
            with validation_area:
                pending = ui.label(translator.text("validation.running"))

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
                pending.delete()
                with validation_area:
                    if update.result is not None:
                        _render_validation_result(ui, translator, update.result)
                    else:
                        ui.label(translator.text("validation.failed")).classes(
                            "lte-validation-result lte-validation-result--error"
                        )
                        if update.error:
                            with ui.expansion(translator.text("validation.details")):
                                ui.label(update.error).classes("lte-validation-message")

            timer = ui.timer(0.2, collect, active=False)
            active_timer["value"] = timer
            timer.activate()

        with ui.row().classes("lte-card-actions"):
            fast_button = ui.button(
                translator.text("action.validate"),
                on_click=validate_fast,
            ).props("outline").mark(f"scenario-validate-{card.scenario_id}")
            full_button = ui.button(
                translator.text("action.full_checksum"),
                on_click=validate_full,
            ).props("outline").mark(f"scenario-checksum-{card.scenario_id}")
            configure = ui.button(
                translator.text("action.configure"),
                on_click=lambda: ui.navigate.to(f"/configure/{card.scenario_id}"),
            ).mark(f"scenario-configure-{card.scenario_id}")
            configure.set_enabled(card.can_run)
            run = ui.button(
                translator.text("action.run"),
                on_click=lambda: ui.navigate.to(f"/configure/{card.scenario_id}"),
            ).mark(f"scenario-run-{card.scenario_id}")
            run.set_enabled(card.can_run)
        with ui.expansion(translator.text("guidance.commands")).classes(
            "lte-cli-commands full-width"
        ):
            ui.code(
                f"lte-data validate {card.scenario_id}", language="powershell"
            ).classes(
                "lte-cli-guidance"
            )
            ui.code(
                f"lte-data validate {card.scenario_id} --full-checksum",
                language="powershell",
            ).classes("lte-cli-guidance")
            ui.code(
                f"lte-data dem export {card.scenario_id} --dry-run",
                language="powershell",
            ).classes("lte-cli-guidance")
            ui.code(
                f"lte-data dem ingest {card.scenario_id} --tiles-dir '<path>'",
                language="powershell",
            ).classes("lte-cli-guidance")


__all__ = [
    "ScenarioCard",
    "ValidationDiagnostic",
    "ValidationJobUpdate",
    "ValidationResult",
    "get_job_coordinator",
    "poll_validation_job",
    "render_scenarios_page",
    "run_validation",
    "scenario_cards",
    "shutdown_job_coordinator",
    "submit_full_checksum",
]
