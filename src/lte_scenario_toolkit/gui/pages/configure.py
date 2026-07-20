"""Profile configuration models, safe mutations, and page rendering."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ...profiles import (
    DEFAULT_PROFILE_VALUES,
    ExperimentProfile,
    FigureSettings,
    OutputSettings,
    validate_profile,
)
from ...selection_service import SelectionError
from ..presentation import (
    ActionSpec,
    MenuActionSpec,
    readiness_presentation,
    render_action_bar,
    render_overflow_menu,
    render_page_header,
    render_section_heading,
    render_status_badge,
    render_sticky_action_dock,
)

PROFILE_SELECTION_MESSAGE = (
    "The requested or configured default profile cannot be resolved safely."
)


class ConfirmationRequiredError(ValueError):
    """Raised when a confirmed GUI dialog did not authorize a guarded mutation."""

    code = "profile.confirmation_required"


class ProfileRefreshError(RuntimeError):
    """Report that a repository mutation committed before runtime refresh failed."""

    code = "profile.refresh_failed_after_commit"

    def __init__(self, mutation_result: Any, refresh_error: Exception) -> None:
        super().__init__(
            "Profile mutation committed, but the GUI runtime could not refresh: "
            f"{refresh_error}"
        )
        self.mutation_result = mutation_result
        self.refresh_error = refresh_error
        self.committed = True


@dataclass(frozen=True, slots=True)
class ConfigureModel:
    """Immutable state used to render one configuration page."""

    scenario_id: str
    display_name: str
    status: str
    profiles: tuple[ExperimentProfile, ...]
    profile_choices: tuple[tuple[str, str], ...]
    profile: ExperimentProfile
    is_persisted: bool
    is_default: bool
    dirty: bool
    output_root_default: Path
    selection_error: str | None = None

    @property
    def can_start(self) -> bool:
        return (
            self.status == "ready"
            and self.selection_error is None
        )


@dataclass(frozen=True, slots=True)
class PreflightOutcome:
    """Immutable result of freezing and validating a Start Scan snapshot."""

    snapshot: ExperimentProfile
    preflight: Any | None
    error_code: str | None
    field_errors: tuple[tuple[str, str], ...]

    @property
    def ok(self) -> bool:
        return self.preflight is not None and self.error_code is None


@dataclass(slots=True)
class ProfileActions:
    """Route all profile writes through ProfileStore and refresh on success."""

    store: Any
    on_mutation: Callable[[], Any]

    def _refresh(self, mutation_result: Any) -> None:
        try:
            self.on_mutation()
        except Exception as exc:
            raise ProfileRefreshError(mutation_result, exc) from exc

    def save(
        self,
        profile: ExperimentProfile,
        *,
        overwrite: bool = False,
        set_default: bool = False,
        confirmed: bool = False,
    ) -> Path:
        result = save_profile(
            self.store,
            profile,
            overwrite=overwrite,
            set_default=set_default,
            confirmed=confirmed,
        )
        self._refresh(result)
        return result

    def copy(
        self,
        source: str | Path,
        profile_id: str,
        display_name: str,
        *,
        confirmed: bool,
    ) -> Path:
        result = copy_profile(
            self.store,
            source,
            profile_id,
            display_name,
            confirmed=confirmed,
        )
        self._refresh(result)
        return result

    def rename(
        self,
        source: str | Path,
        profile_id: str,
        display_name: str,
        *,
        confirmed: bool,
    ) -> Path:
        result = rename_profile(
            self.store,
            source,
            profile_id,
            display_name,
            confirmed=confirmed,
        )
        self._refresh(result)
        return result

    def set_default(
        self,
        scenario_id: str,
        profile_path: str | Path,
        *,
        confirmed: bool,
    ) -> Path:
        result = set_default_profile(
            self.store,
            scenario_id,
            profile_path,
            confirmed=confirmed,
        )
        self._refresh(result)
        return result

    def delete(
        self,
        profile_path: str | Path,
        *,
        replacement_default: str | Path | None = None,
        confirmed: bool,
    ) -> None:
        delete_profile(
            self.store,
            profile_path,
            replacement_default=replacement_default,
            confirmed=confirmed,
        )
        self._refresh(None)


def _profile_source(profile: ExperimentProfile) -> Path | None:
    if profile.source_path is None:
        return None
    return Path(profile.source_path).resolve()


def _catalog_profile_path(catalog: Any, scenario: Mapping[str, Any]) -> Path | None:
    raw = scenario.get("config_path")
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = Path(catalog.root) / path
    return path.resolve()


def _points_dataset_id(catalog: Any) -> str:
    choices = tuple(
        str(dataset_id)
        for dataset_id, dataset in catalog.datasets_by_id.items()
        if dataset.get("role") == "points"
    )
    if len(choices) != 1:
        raise ValueError(
            "A new profile requires exactly one registered points dataset; "
            f"found {len(choices)}"
        )
    return choices[0]


def _draft_profile(catalog: Any, scenario_id: str, display_name: str) -> ExperimentProfile:
    return ExperimentProfile(
        profile_id="default",
        display_name=f"{display_name} Default",
        scenario_id=scenario_id,
        points_dataset_id=_points_dataset_id(catalog),
        random_seed=DEFAULT_PROFILE_VALUES["random_seed"],
        target_crs=DEFAULT_PROFILE_VALUES["target_crs"],
        rect_size=DEFAULT_PROFILE_VALUES["rect_size"],
        target_count=DEFAULT_PROFILE_VALUES["target_count"],
        tolerance=DEFAULT_PROFILE_VALUES["tolerance"],
        scan_mode=DEFAULT_PROFILE_VALUES["scan_mode"],
        strategy=DEFAULT_PROFILE_VALUES["strategy"],
        scan_step=DEFAULT_PROFILE_VALUES["scan_step"],
        max_rects=DEFAULT_PROFILE_VALUES["max_rects"],
        min_spacing=DEFAULT_PROFILE_VALUES["min_spacing"],
        output_root=(Path(catalog.root) / "results").resolve(),
        outputs=OutputSettings(),
        figure=FigureSettings(),
        source_path=None,
    )


def configure_model(
    catalog: Any,
    store: Any,
    scenario_id: str,
    *,
    selected_profile_id: str | None = None,
) -> ConfigureModel:
    """Build a saved/default profile model or an explicit current draft."""

    scenario = catalog.scenario(scenario_id)
    display_name = str(scenario["display_name"])
    status = str(catalog.scenario_status(scenario_id))
    selection_error: str | None = None
    profiles = tuple(store.discover(scenario_id))

    default_path = _catalog_profile_path(catalog, scenario)
    if selected_profile_id == "__new__":
        selected = None
    elif selected_profile_id is not None:
        selected = (
            next(
                (
                    profile
                    for profile in profiles
                    if profile.profile_id == selected_profile_id
                ),
                None,
            )
        )
        if selected is None:
            selection_error = PROFILE_SELECTION_MESSAGE
    else:
        if default_path is None:
            selected = profiles[0] if profiles else None
        else:
            selected = next(
                (
                    profile
                    for profile in profiles
                    if _profile_source(profile) == default_path
                ),
                None,
            )
            if selected is None:
                selection_error = PROFILE_SELECTION_MESSAGE
    persisted = selected is not None
    profile = selected or _draft_profile(catalog, scenario_id, display_name)
    output_root_default = (Path(catalog.root) / "results").resolve()
    return ConfigureModel(
        scenario_id=scenario_id,
        display_name=display_name,
        status=status,
        profiles=profiles,
        profile_choices=tuple(
            (item.profile_id, item.display_name) for item in profiles
        ),
        profile=profile,
        is_persisted=persisted,
        is_default=(
            persisted
            and default_path is not None
            and _profile_source(profile) == default_path
        ),
        dirty=False,
        output_root_default=output_root_default,
        selection_error=selection_error,
    )


_INTEGER_FIELDS = frozenset(
    {
        "random_seed",
        "rect_size",
        "target_count",
        "tolerance",
        "scan_step",
        "max_rects",
        "min_spacing",
    }
)
_TEXT_FIELDS = frozenset(
    {
        "profile_id",
        "display_name",
        "points_dataset_id",
        "target_crs",
        "scan_mode",
        "strategy",
    }
)
_FORM_FIELDS = _INTEGER_FIELDS | _TEXT_FIELDS | {"output_root"}


def _whole_number(value: Any, field_name: str) -> int:
    if type(value) is bool:
        raise ValueError(f"{field_name} must be a whole number")
    if type(value) is int:
        return value
    if type(value) is float:
        if math.isfinite(value) and value.is_integer():
            return int(value)
        raise ValueError(f"{field_name} must be a whole number")
    if type(value) is str:
        stripped = value.strip()
        if stripped and stripped.lstrip("+-").isdigit():
            return int(stripped)
    raise ValueError(f"{field_name} must be a whole number")


def profile_with_form_values(
    profile: ExperimentProfile,
    values: Mapping[str, Any],
) -> ExperimentProfile:
    """Apply validated form values without mutating or truncating the source profile."""

    unknown = set(values) - _FORM_FIELDS
    if unknown:
        raise ValueError(f"Unknown profile form fields: {', '.join(sorted(unknown))}")
    updates: dict[str, Any] = {}
    for field_name, value in values.items():
        if field_name in _INTEGER_FIELDS:
            updates[field_name] = _whole_number(value, field_name)
        elif field_name in _TEXT_FIELDS:
            if type(value) is not str or not value.strip():
                raise ValueError(f"{field_name} must be non-empty text")
            updates[field_name] = value.strip()
        elif field_name == "output_root":
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError("output_root must be a non-empty path")
            updates[field_name] = Path(value).expanduser()
    return validate_profile(replace(profile, **updates))


def _require_confirmed(confirmed: bool, operation: str) -> None:
    if confirmed is not True:
        raise ConfirmationRequiredError(
            f"{operation} requires explicit confirmation from the GUI dialog"
        )


def save_profile(
    store: Any,
    profile: ExperimentProfile,
    *,
    overwrite: bool = False,
    set_default: bool = False,
    confirmed: bool = False,
) -> Path:
    """Validate first, then delegate a profile save to ProfileStore."""

    validate_profile(profile)
    if overwrite or set_default:
        _require_confirmed(confirmed, "Save")
    return store.save(profile, overwrite=overwrite, set_default=set_default)


def copy_profile(
    store: Any,
    source: str | Path,
    profile_id: str,
    display_name: str,
    *,
    confirmed: bool,
) -> Path:
    _require_confirmed(confirmed, "Copy")
    return store.copy(source, profile_id, display_name)


def rename_profile(
    store: Any,
    source: str | Path,
    profile_id: str,
    display_name: str,
    *,
    confirmed: bool,
) -> Path:
    _require_confirmed(confirmed, "Rename")
    return store.rename(source, profile_id, display_name)


def set_default_profile(
    store: Any,
    scenario_id: str,
    profile_path: str | Path,
    *,
    confirmed: bool,
) -> Path:
    _require_confirmed(confirmed, "Default replacement")
    return store.set_default(scenario_id, profile_path)


def delete_profile(
    store: Any,
    profile_path: str | Path,
    *,
    replacement_default: str | Path | None = None,
    confirmed: bool,
) -> None:
    _require_confirmed(confirmed, "Delete")
    store.delete(profile_path, replacement_default=replacement_default)


_PREFLIGHT_FIELDS = {
    "profile.target_crs": "target_crs",
    "inputs.points_dataset_id": "points_dataset_id",
    "outputs.root": "output_root",
}


def start_scan_preflight(
    selection_service: Any,
    profile: ExperimentProfile,
) -> PreflightOutcome:
    """Freeze one replace snapshot and call only SelectionService.preflight."""

    snapshot = replace(profile)
    try:
        preflight = selection_service.preflight(snapshot, snapshot.output_root)
    except SelectionError as exc:
        field = _PREFLIGHT_FIELDS.get(exc.code, "__all__")
        return PreflightOutcome(
            snapshot=snapshot,
            preflight=None,
            error_code=exc.code,
            field_errors=((field, exc.message),),
        )
    return PreflightOutcome(
        snapshot=snapshot,
        preflight=preflight,
        error_code=None,
        field_errors=(),
    )


def _choose_host_directory(initial_directory: Path) -> str:
    """Open the optional host picker outside the event loop and always release Tk."""

    from tkinter import Tk, filedialog

    owner = None
    try:
        owner = Tk()
        owner.withdraw()
        return str(
            filedialog.askdirectory(
                initialdir=str(initial_directory),
                parent=owner,
            )
            or ""
        )
    finally:
        if owner is not None:
            owner.destroy()


def _profile_route(scenario_id: str, profile_id: str) -> str:
    return f"/configure/{scenario_id}?profile={quote(profile_id, safe='')}"


def render_configure_picker(ui: Any, translator: Any, catalog: Any) -> None:
    """Render a valid `/configure` route instead of leaving a dead navigation link."""

    with ui.column().classes("lte-page lte-configure-picker"):
        render_page_header(
            ui,
            translator.text("configure.choose_title"),
            translator.text("configure.choose_subtitle"),
        )
        with ui.row().classes("lte-card-grid"):
            for scenario_id in catalog.scenarios_by_id:
                scenario = catalog.scenario(scenario_id)
                status = catalog.scenario_status(scenario_id)
                with ui.card().classes("lte-scenario-card"):
                    presentation = readiness_presentation(status)
                    with ui.row().classes(
                        "items-center justify-between no-wrap full-width"
                    ):
                        ui.label(str(scenario["display_name"])).classes(
                            "lte-card-title"
                        )
                        render_status_badge(
                            ui,
                            translator,
                            presentation,
                            marker=f"picker-status-{scenario_id}",
                        )
                    if presentation.description_key is not None:
                        ui.label(
                            translator.text(presentation.description_key)
                        ).classes("lte-scenario-card__description")
                    if status == "ready":
                        action = ActionSpec(
                            "workflow",
                            translator.text("action.configure_and_scan"),
                            lambda scenario_id=scenario_id: ui.navigate.to(
                                f"/configure/{scenario_id}"
                            ),
                            role="primary",
                            marker=f"picker-configure-{scenario_id}",
                        )
                    else:
                        action = ActionSpec(
                            "workflow",
                            translator.text("action.view_setup"),
                            lambda: ui.navigate.to("/scenarios"),
                            role="primary",
                            marker=f"picker-guidance-{scenario_id}",
                        )
                    render_action_bar(ui, (action,))


def _confirmation_dialog(
    ui: Any,
    translator: Any,
    *,
    title_key: str,
    consequence: str,
    marker_name: str,
    on_confirm: Callable[[], None],
) -> Any:
    with ui.dialog() as dialog, ui.card().classes("lte-confirmation-dialog"):
        ui.label(translator.text(title_key)).classes("lte-section-title")
        ui.label(consequence).classes("lte-confirmation-consequence").mark(
            f"confirmation-{marker_name}-consequence"
        )
        with ui.row().classes("justify-end full-width"):
            ui.button(translator.text("action.cancel"), on_click=dialog.close).props(
                "flat"
            ).mark(f"confirmation-{marker_name}-cancel")

            def confirm() -> None:
                dialog.close()
                on_confirm()

            ui.button(translator.text("action.confirm"), on_click=confirm).mark(
                f"confirm-{title_key}"
            )
    return dialog


def render_configure_page(
    ui: Any,
    translator: Any,
    catalog: Any,
    store: Any,
    selection_service: Any,
    *,
    scenario_id: str,
    selected_profile_id: str | None = None,
    on_profile_mutation: Callable[[], Any],
    on_preflight_success: Callable[[PreflightOutcome], Any] | None = None,
) -> None:
    """Render profile selection, guarded CRUD, parameters, and read-only preflight."""

    try:
        model = configure_model(
            catalog,
            store,
            scenario_id,
            selected_profile_id=selected_profile_id,
        )
    except Exception as exc:
        with ui.column().classes("lte-page"):
            ui.label(translator.text("configure.unavailable")).classes(
                "lte-page-title"
            ).props("role=heading aria-level=1")
            ui.label(translator.text("operation.failed")).classes(
                "lte-validation-result--error"
            )
            with ui.expansion(translator.text("validation.details")):
                ui.label(str(exc)).classes("lte-validation-message")
        return

    actions = ProfileActions(store, on_profile_mutation)
    profile = model.profile
    with ui.column().classes("lte-page lte-configure-page"):
        render_page_header(
            ui,
            translator.text("configure.title", name=model.display_name),
            model.scenario_id,
        )

        if model.status != "ready":
            readiness = readiness_presentation(model.status)
            with ui.card().classes("lte-callout lte-callout--warning"):
                render_status_badge(ui, translator, readiness)
                readiness_key = readiness.description_key or "status.unknown"
                ui.label(translator.text(readiness_key)).mark(
                    "configure-readiness"
                )
        if model.selection_error is not None:
            with ui.card().classes("lte-callout lte-callout--warning"):
                ui.label(translator.text("configure.selection_title")).classes(
                    "lte-section-title"
                )
                ui.label(translator.text("configure.selection_body"))

        form_values: dict[str, Any] = {}
        field_elements: dict[str, Any] = {}
        switch_target: dict[str, str | None] = {"value": None}
        safety_controls: list[Any] = []
        workflow_state = {"preflight_passed": False}

        options = dict(model.profile_choices)
        options["__new__"] = translator.text("configure.new_profile")
        selected_option = profile.profile_id if model.is_persisted else "__new__"
        profile_state = translator.text(
            "configure.saved" if model.is_persisted else "configure.unsaved"
        )
        if model.is_default:
            profile_state = (
                f"{profile_state} / "
                f"{translator.text('configure.profile_state.default')}"
            )
        with ui.card().classes("lte-profile-header").mark(
            "configure-profile-header"
        ):
            with ui.row().classes("lte-profile-header__main"):
                profile_select = ui.select(
                    options,
                    value=selected_option,
                    label=translator.text("label.profile"),
                ).classes("lte-profile-select").mark("profile-select")
                profile_select.set_enabled(True)
                dirty_label = ui.label(profile_state).classes(
                    "lte-dirty-indicator"
                ).mark("configure-profile-state")
            management_area = ui.row().classes("lte-profile-management").mark(
                "profile-management-actions"
            )

        with ui.row().classes("lte-configure-stepper").props(
            'role="list" aria-label="Configuration workflow"'
        ).mark("configure-workflow-stepper"):
            data_step = ui.label(translator.text("configure.workflow.data")).classes(
                "lte-configure-step lte-configure-step--complete"
            ).props('role="listitem"').mark("configure-step-data")
            profile_step = ui.label(
                translator.text("configure.workflow.profile")
            ).classes("lte-configure-step lte-configure-step--current").props(
                'role="listitem"'
            ).mark("configure-step-profile")
            review_step = ui.label(
                translator.text("configure.workflow.review")
            ).classes("lte-configure-step").props('role="listitem"').mark(
                "configure-step-review"
            )

        with ui.row().classes("lte-configure-workspace").mark(
            "configure-workspace"
        ):
            form_column = ui.column().classes("lte-configure-form-column")
            with ui.card().classes("lte-configure-run-summary").mark(
                "configure-run-summary"
            ):
                render_section_heading(ui, translator.text("configure.summary"))
                with ui.column().classes("lte-configure-summary-list"):
                    ui.label(translator.text("configure.summary.scenario")).classes(
                        "lte-configure-summary-label"
                    )
                    ui.label(model.display_name).mark("configure-summary-scenario")
                    ui.label(translator.text("configure.summary.profile")).classes(
                        "lte-configure-summary-label"
                    )
                    ui.label(profile.display_name).mark("configure-summary-profile")
                    ui.label(translator.text("configure.summary.data")).classes(
                        "lte-configure-summary-label"
                    )
                    ui.label(profile.points_dataset_id).mark("configure-summary-data")
                    ui.label(translator.text("configure.summary.readiness")).classes(
                        "lte-configure-summary-label"
                    )
                    render_status_badge(
                        ui,
                        translator,
                        readiness_presentation(model.status),
                        marker="configure-summary-readiness",
                    )
                    summary_profile_state = ui.label(profile_state).classes(
                        "lte-dirty-indicator"
                    ).mark(
                        "configure-summary-profile-state"
                    )
                    ui.label(translator.text("configure.summary.next_step")).classes(
                        "lte-configure-summary-label"
                    )
                    next_step_summary = ui.label().mark("configure-summary-next-step")

        def refresh_workflow_presentation() -> None:
            dirty = bool(form_values)
            blocked = model.selection_error is not None or model.status != "ready"
            if blocked:
                next_key = "configure.next_step.blocked"
            elif workflow_state["preflight_passed"]:
                next_key = "configure.next_step.reviewed"
            elif dirty:
                next_key = "configure.next_step.dirty"
            else:
                next_key = "configure.next_step.ready"
            summary_profile_state.set_text(
                translator.text("configure.dirty") if dirty else profile_state
            )
            next_step_summary.set_text(translator.text(next_key))
            data_step.classes(
                add="lte-configure-step--blocked" if blocked else "lte-configure-step--complete",
                remove="lte-configure-step--blocked lte-configure-step--complete",
            )
            profile_step.classes(
                add="lte-configure-step--current" if dirty else "lte-configure-step--complete",
                remove="lte-configure-step--current lte-configure-step--complete",
            )
            review_step.classes(
                add=(
                    "lte-configure-step--blocked"
                    if blocked
                    else "lte-configure-step--complete"
                    if workflow_state["preflight_passed"]
                    else "lte-configure-step--current"
                ),
                remove=(
                    "lte-configure-step--blocked lte-configure-step--complete "
                    "lte-configure-step--current"
                ),
            )

        refresh_workflow_presentation()

        def navigate_to_profile(selected_id: str) -> None:
            ui.navigate.to(_profile_route(scenario_id, selected_id))

        def confirm_profile_switch() -> None:
            switch_dialog.close()
            selected_id = switch_target["value"]
            if selected_id is not None:
                navigate_to_profile(selected_id)

        with ui.dialog() as switch_dialog, ui.card().classes(
            "lte-confirmation-dialog"
        ):
            ui.label(translator.text("confirmation.discard_switch")).classes(
                "lte-section-title"
            )
            switch_consequence = ui.label(
                translator.text(
                    "confirmation.consequence.switch",
                    current_name=profile.display_name,
                    target_name=profile.display_name,
                )
            ).classes("lte-confirmation-consequence").mark(
                "confirmation-switch-consequence"
            )
            with ui.row().classes("justify-end full-width"):
                ui.button(
                    translator.text("action.cancel"), on_click=switch_dialog.close
                ).props("flat").mark("profile-switch-cancel")
                ui.button(
                    translator.text("action.discard"),
                    on_click=confirm_profile_switch,
                ).mark("profile-switch-discard")

        def select_profile(event) -> None:
            selected_id = str(event.value)
            if selected_id == selected_option:
                return
            if form_values:
                switch_target["value"] = selected_id
                profile_select.value = selected_option
                switch_consequence.set_text(
                    translator.text(
                        "confirmation.consequence.switch",
                        current_name=profile.display_name,
                        target_name=str(options.get(selected_id, selected_id)),
                    )
                )
                switch_dialog.open()
                return
            navigate_to_profile(selected_id)

        profile_select.on_value_change(select_profile)

        def changed(event, field_name: str) -> None:
            form_values[field_name] = event.value
            workflow_state["preflight_passed"] = False
            dirty_label.set_text(translator.text("configure.dirty"))
            dirty_label.classes(add="lte-dirty-indicator--dirty")
            clear_form_errors(field_name)
            refresh_workflow_presentation()

        with form_column:
            form_card = ui.card().classes("lte-form-card")
        with form_card:
            render_section_heading(ui, translator.text("configure.basic"))
            with ui.grid(columns=2).classes("lte-form-grid"):
                profile_id = ui.input(
                    translator.text("field.profile_id"), value=profile.profile_id
                ).mark("profile-id")
                profile_id.set_enabled(not model.is_persisted)
                profile_id.on_value_change(
                    lambda event: changed(event, "profile_id")
                )
                display_name = ui.input(
                    translator.text("field.display_name"), value=profile.display_name
                ).mark("profile-display-name")
                display_name.on_value_change(
                    lambda event: changed(event, "display_name")
                )
                target_crs = ui.input(
                    translator.text("field.target_crs"), value=profile.target_crs
                ).mark("profile-target-crs")
                field_elements["target_crs"] = target_crs
                target_crs.on_value_change(
                    lambda event: changed(event, "target_crs")
                )
                rect_size = ui.number(
                    translator.text("field.rect_size"), value=profile.rect_size
                ).mark("profile-rect-size")
                rect_size.on_value_change(lambda event: changed(event, "rect_size"))
                target_count = ui.number(
                    translator.text("field.target_count"), value=profile.target_count
                ).mark("profile-target-count")
                target_count.on_value_change(
                    lambda event: changed(event, "target_count")
                )
                scan_step = ui.number(
                    translator.text("field.scan_step"), value=profile.scan_step
                ).mark("profile-scan-step")
                scan_step.on_value_change(lambda event: changed(event, "scan_step"))
                max_rects = ui.number(
                    translator.text("field.max_rects"), value=profile.max_rects
                ).mark("profile-max-rects")
                max_rects.on_value_change(lambda event: changed(event, "max_rects"))
            with ui.expansion(translator.text("configure.advanced")).classes(
                "lte-advanced full-width"
            ):
                with ui.grid(columns=2).classes("lte-form-grid"):
                    tolerance = ui.number(
                        translator.text("field.tolerance"), value=profile.tolerance
                    ).mark("profile-tolerance")
                    tolerance.on_value_change(
                        lambda event: changed(event, "tolerance")
                    )
                    strategy = ui.select(
                        {
                            "uniform": translator.text("value.uniform"),
                            "sequential": translator.text("value.sequential"),
                        },
                        value=profile.strategy,
                        label=translator.text("field.strategy"),
                    ).mark("profile-strategy")
                    strategy.on_value_change(lambda event: changed(event, "strategy"))
                    random_seed = ui.number(
                        translator.text("field.random_seed"), value=profile.random_seed
                    ).mark("profile-random-seed")
                    random_seed.on_value_change(
                        lambda event: changed(event, "random_seed")
                    )
                    min_spacing = ui.number(
                        translator.text("field.min_spacing"), value=profile.min_spacing
                    ).mark("profile-min-spacing")
                    min_spacing.on_value_change(
                        lambda event: changed(event, "min_spacing")
                    )
                    scan_mode = ui.select(
                        {
                            "fast": translator.text("value.fast"),
                            "complete": translator.text("value.complete"),
                        },
                        value=profile.scan_mode,
                        label=translator.text("field.scan_mode"),
                    ).mark("profile-scan-mode")
                    scan_mode.on_value_change(
                        lambda event: changed(event, "scan_mode")
                    )

        with form_column:
            output_card = ui.card().classes("lte-output-card").mark(
                "configure-output"
            )
        with output_card:
            render_section_heading(ui, translator.text("configure.output"))
            with ui.row().classes("lte-output-row"):
                output_root = ui.input(
                    translator.text("field.output_root"),
                    value=str(profile.output_root),
                ).classes("grow").mark("profile-output-root")
                field_elements["output_root"] = output_root
                output_root.on_value_change(
                    lambda event: changed(event, "output_root")
                )

                async def browse() -> None:
                    try:
                        from nicegui import run

                        selected = await run.io_bound(
                            _choose_host_directory,
                            Path(output_root.value or profile.output_root),
                        )
                    except Exception:
                        ui.notify(translator.text("browse.unavailable"), type="warning")
                        return
                    if selected:
                        output_root.value = selected
                        form_values["output_root"] = selected
                        workflow_state["preflight_passed"] = False
                        dirty_label.set_text(translator.text("configure.dirty"))
                        dirty_label.classes(add="lte-dirty-indicator--dirty")
                        refresh_workflow_presentation()

                ui.button(
                    translator.text("action.browse"), on_click=browse
                ).props("outline").mark("profile-browse")

        with form_column:
            error_area = ui.column().classes("lte-form-errors full-width")

        def clear_form_errors(field_name: str | None = None) -> None:
            error_area.clear()
            names = tuple(field_elements) if field_name is None else (field_name,)
            for name in names:
                element = field_elements.get(name)
                if element is None:
                    continue
                element.props.pop("error", None)
                element.props.pop("error-message", None)
                element.update()

        def current_profile() -> ExperimentProfile:
            return profile_with_form_values(profile, form_values)

        def report_error(exc: Exception) -> None:
            error_area.clear()
            with error_area:
                ui.label(translator.text("operation.failed")).classes(
                    "lte-validation-result lte-validation-result--error"
                )
                with ui.expansion(translator.text("validation.details")):
                    ui.label(str(exc)).classes("lte-validation-message")

        def report_preflight_error(outcome: PreflightOutcome) -> None:
            error_area.clear()
            translation_key = {
                "profile.target_crs": "preflight.error.target_crs",
                "inputs.points_dataset_id": "preflight.error.points",
                "outputs.root": "preflight.error.output_root",
            }.get(outcome.error_code, "preflight.error.global")
            localized = translator.text(translation_key)
            field_name, technical_message = outcome.field_errors[0]
            element = field_elements.get(field_name)
            if element is not None:
                element.props["error"] = True
                element.props["error-message"] = localized
                element.update()
            with error_area:
                ui.label(localized).classes(
                    "lte-validation-result lte-validation-result--error"
                )
                with ui.expansion(translator.text("validation.details")):
                    ui.label(technical_message).classes("lte-validation-message")

        def report_action_error(exc: Exception, recovery_route: str) -> None:
            if not isinstance(exc, ProfileRefreshError):
                report_error(exc)
                return
            try:
                actions.on_mutation()
            except Exception as retry_error:
                for control in safety_controls:
                    control.set_enabled(False)
                error_area.clear()
                with error_area:
                    ui.label(translator.text("profile.refresh_failed")).classes(
                        "lte-validation-result lte-validation-result--error"
                    )
                    with ui.expansion(translator.text("validation.details")):
                        ui.label(str(retry_error)).classes("lte-validation-message")
                return
            ui.navigate.to(recovery_route)

        def after_success(profile_id: str) -> None:
            ui.navigate.to(_profile_route(scenario_id, profile_id))

        def perform_save(*, confirmed: bool) -> None:
            try:
                saved_profile = current_profile()
            except Exception as exc:
                report_error(exc)
                return
            try:
                actions.save(
                    saved_profile,
                    overwrite=model.is_persisted,
                    confirmed=confirmed,
                )
            except Exception as exc:
                report_action_error(
                    exc,
                    _profile_route(scenario_id, saved_profile.profile_id),
                )
                return
            after_success(saved_profile.profile_id)

        save_dialog = _confirmation_dialog(
            ui,
            translator,
            title_key="confirmation.overwrite",
            consequence=translator.text(
                "confirmation.consequence.overwrite",
                profile_name=profile.display_name,
                profile_id=profile.profile_id,
            ),
            marker_name="overwrite",
            on_confirm=lambda: perform_save(confirmed=True),
        )

        def request_save() -> None:
            if model.is_persisted:
                save_dialog.open()
            else:
                perform_save(confirmed=False)

        source_path = profile.source_path
        persisted_actions_enabled = (
            model.is_persisted
            and source_path is not None
            and model.status == "ready"
        )

        def copy_confirmed() -> None:
            if source_path is None:
                return
            try:
                copied_id = str(copy_id.value or "")
                actions.copy(
                    source_path,
                    copied_id,
                    str(copy_name.value or ""),
                    confirmed=True,
                )
            except Exception as exc:
                report_action_error(
                    exc,
                    _profile_route(scenario_id, copied_id),
                )
                return
            after_success(copied_id)

        def rename_confirmed() -> None:
            if source_path is None:
                return
            try:
                renamed_id = str(rename_id.value or "")
                actions.rename(
                    source_path,
                    renamed_id,
                    str(rename_name.value or ""),
                    confirmed=True,
                )
            except Exception as exc:
                report_action_error(
                    exc,
                    _profile_route(scenario_id, renamed_id),
                )
                return
            after_success(renamed_id)

        def default_confirmed() -> None:
            if source_path is None:
                return
            try:
                actions.set_default(
                    scenario_id,
                    source_path,
                    confirmed=True,
                )
            except Exception as exc:
                report_action_error(
                    exc,
                    _profile_route(scenario_id, profile.profile_id),
                )
                return
            after_success(profile.profile_id)

        default_path = _catalog_profile_path(catalog, catalog.scenario(scenario_id))
        alternatives = tuple(
            item
            for item in model.profiles
            if item.source_path is not None and _profile_source(item) != _profile_source(profile)
        )
        deleting_default = source_path is not None and _profile_source(profile) == default_path

        def delete_confirmed() -> None:
            if source_path is None:
                return
            replacement = None
            if deleting_default:
                replacement = replacement_select.value
                if not replacement:
                    report_error(ValueError(translator.text("delete.replacement_required")))
                    return
            try:
                actions.delete(
                    source_path,
                    replacement_default=replacement,
                    confirmed=True,
                )
            except Exception as exc:
                report_action_error(exc, "/scenarios")
                return
            ui.navigate.to("/scenarios")

        with ui.dialog() as copy_dialog, ui.card().classes("lte-confirmation-dialog"):
            ui.label(translator.text("confirmation.copy")).classes("lte-section-title")
            copy_consequence = ui.label().classes(
                "lte-confirmation-consequence"
            ).mark("confirmation-copy-consequence")
            copy_id = ui.input(
                translator.text("field.profile_id"),
                value=f"{profile.profile_id}-copy",
            ).mark("profile-copy-id")
            copy_name = ui.input(
                translator.text("field.display_name"),
                value=f"{profile.display_name} Copy",
            ).mark("profile-copy-name")

            def update_copy_consequence() -> None:
                copy_consequence.set_text(
                    translator.text(
                        "confirmation.consequence.copy",
                        profile_name=profile.display_name,
                        profile_id=profile.profile_id,
                        target_name=str(copy_name.value or ""),
                        target_id=str(copy_id.value or ""),
                    )
                )

            copy_id.on_value_change(lambda _event: update_copy_consequence())
            copy_name.on_value_change(lambda _event: update_copy_consequence())
            update_copy_consequence()
            with ui.row().classes("justify-end full-width"):
                ui.button(
                    translator.text("action.cancel"), on_click=copy_dialog.close
                ).props("flat").mark("profile-copy-cancel")

                def confirm_copy() -> None:
                    copy_dialog.close()
                    copy_confirmed()

                ui.button(
                    translator.text("action.confirm"), on_click=confirm_copy
                ).mark("profile-copy-confirm")

        with ui.dialog() as rename_dialog, ui.card().classes("lte-confirmation-dialog"):
            ui.label(translator.text("confirmation.rename")).classes("lte-section-title")
            rename_consequence = ui.label().classes(
                "lte-confirmation-consequence"
            ).mark("confirmation-rename-consequence")
            rename_id = ui.input(
                translator.text("field.profile_id"),
                value=profile.profile_id,
            ).mark("profile-rename-id")
            rename_name = ui.input(
                translator.text("field.display_name"),
                value=profile.display_name,
            ).mark("profile-rename-name")

            def update_rename_consequence() -> None:
                consequence_key = (
                    "confirmation.consequence.rename_default"
                    if model.is_default
                    else "confirmation.consequence.rename"
                )
                rename_consequence.set_text(
                    translator.text(
                        consequence_key,
                        profile_name=profile.display_name,
                        profile_id=profile.profile_id,
                        target_name=str(rename_name.value or ""),
                        target_id=str(rename_id.value or ""),
                    )
                )

            rename_id.on_value_change(lambda _event: update_rename_consequence())
            rename_name.on_value_change(lambda _event: update_rename_consequence())
            update_rename_consequence()
            with ui.row().classes("justify-end full-width"):
                ui.button(
                    translator.text("action.cancel"), on_click=rename_dialog.close
                ).props("flat").mark("profile-rename-cancel")

                def confirm_rename() -> None:
                    rename_dialog.close()
                    rename_confirmed()

                ui.button(
                    translator.text("action.confirm"), on_click=confirm_rename
                ).mark("profile-rename-confirm")

        default_dialog = _confirmation_dialog(
            ui,
            translator,
            title_key="confirmation.default",
            consequence=translator.text(
                "confirmation.consequence.default",
                scenario_name=model.display_name,
                profile_name=profile.display_name,
                profile_id=profile.profile_id,
            ),
            marker_name="default",
            on_confirm=default_confirmed,
        )
        with ui.dialog() as delete_dialog, ui.card().classes("lte-confirmation-dialog"):
            ui.label(translator.text("confirmation.delete")).classes("lte-section-title")
            if deleting_default and alternatives:
                delete_text = translator.text(
                    "confirmation.consequence.delete_default_pending",
                    scenario_name=model.display_name,
                    profile_name=profile.display_name,
                    profile_id=profile.profile_id,
                )
            else:
                delete_text = translator.text(
                    "confirmation.consequence.delete",
                    profile_name=profile.display_name,
                    profile_id=profile.profile_id,
                )
            delete_consequence = ui.label(delete_text).classes(
                "lte-confirmation-consequence"
            ).mark("confirmation-delete-consequence")
            replacement_select = ui.select(
                {
                    str(item.source_path): item.display_name
                    for item in alternatives
                },
                value=None,
                label=translator.text("delete.replacement"),
            ).mark("profile-delete-replacement")
            replacement_select.set_visibility(bool(deleting_default))

            def update_delete_consequence(event) -> None:
                selected = next(
                    (
                        item
                        for item in alternatives
                        if str(item.source_path) == str(event.value)
                    ),
                    None,
                )
                if selected is None:
                    return
                delete_consequence.set_text(
                    translator.text(
                        "confirmation.consequence.delete_default",
                        replacement_name=selected.display_name,
                        scenario_name=model.display_name,
                        profile_name=profile.display_name,
                        profile_id=profile.profile_id,
                    )
                )

            replacement_select.on_value_change(update_delete_consequence)
            with ui.row().classes("justify-end full-width"):
                ui.button(
                    translator.text("action.cancel"), on_click=delete_dialog.close
                ).props("flat").mark("profile-delete-cancel")

                def confirm_delete() -> None:
                    delete_dialog.close()
                    delete_confirmed()

                ui.button(
                    translator.text("action.confirm"), on_click=confirm_delete
                ).mark("profile-delete-confirm")

        def run_preflight() -> None:
            clear_form_errors()
            try:
                candidate = current_profile()
            except ValueError as exc:
                report_error(exc)
                return
            outcome = start_scan_preflight(selection_service, candidate)
            if not outcome.ok:
                workflow_state["preflight_passed"] = False
                refresh_workflow_presentation()
                report_preflight_error(outcome)
                return
            workflow_state["preflight_passed"] = True
            refresh_workflow_presentation()
            if on_preflight_success is None:
                ui.notify(translator.text("preflight.passed"), type="positive")
                return
            try:
                on_preflight_success(outcome)
            except Exception as exc:
                report_error(exc)

        with management_area:
            management_items: dict[str, Any] = {}
            render_overflow_menu(
                ui,
                (
                    MenuActionSpec(
                        translator.text("action.copy"),
                        None,
                        copy_dialog.open,
                        enabled=persisted_actions_enabled,
                        marker="profile-copy",
                    ),
                    MenuActionSpec(
                        translator.text("action.rename"),
                        None,
                        rename_dialog.open,
                        enabled=persisted_actions_enabled,
                        marker="profile-rename",
                    ),
                    MenuActionSpec(
                        translator.text("action.set_default"),
                        None,
                        default_dialog.open,
                        enabled=persisted_actions_enabled and not model.is_default,
                        marker="profile-set-default",
                    ),
                    MenuActionSpec(
                        translator.text("action.delete"),
                        None,
                        delete_dialog.open,
                        role="danger",
                        enabled=(
                            persisted_actions_enabled
                            and (not deleting_default or bool(alternatives))
                        ),
                        marker="profile-delete",
                    ),
                    MenuActionSpec(
                        translator.text("configure.validate"),
                        None,
                        run_preflight,
                        separator=True,
                        enabled=model.can_start,
                        marker="profile-validate",
                    ),
                ),
                label=translator.text("configure.profile_actions"),
                marker="profile-overflow",
                item_sink=management_items,
            )
        copy_button = management_items["profile-copy"]
        rename_button = management_items["profile-rename"]
        default_button = management_items["profile-set-default"]
        delete_button = management_items["profile-delete"]

        writes_enabled = model.selection_error is None and model.status == "ready"
        with ui.element("div").classes("full-width").mark("configure-action-bar"):
            workflow_buttons = render_sticky_action_dock(
                ui,
                (
                    ActionSpec(
                        "discard",
                        translator.text("action.discard"),
                        ui.navigate.reload,
                        role="tertiary",
                        marker="profile-discard",
                    ),
                    ActionSpec(
                        "save",
                        translator.text("action.save"),
                        request_save,
                        enabled=writes_enabled,
                        marker="profile-save",
                    ),
                    ActionSpec(
                        "start",
                        translator.text("action.start_scan"),
                        run_preflight,
                        role="primary",
                        enabled=model.can_start,
                        marker="profile-start-scan",
                    ),
                ),
                label=translator.text("configure.profile_actions"),
                marker="configure-action-dock",
                extra_classes="lte-configure-action-dock",
            )
        _discard = workflow_buttons["discard"]
        save = workflow_buttons["save"]
        start = workflow_buttons["start"]
        safety_controls.extend(
            (
                copy_button,
                rename_button,
                default_button,
                delete_button,
                management_items["profile-validate"],
                save,
                start,
            )
        )


__all__ = [
    "ConfigureModel",
    "ConfirmationRequiredError",
    "PreflightOutcome",
    "ProfileActions",
    "ProfileRefreshError",
    "configure_model",
    "copy_profile",
    "delete_profile",
    "profile_with_form_values",
    "rename_profile",
    "render_configure_page",
    "render_configure_picker",
    "save_profile",
    "set_default_profile",
    "start_scan_preflight",
]
