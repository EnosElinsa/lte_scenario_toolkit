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

LEGACY_MIGRATION_MESSAGE = (
    "This is a read-only preview of the legacy profile's effective values. "
    "Explicit Save converts it to schema version 2."
)
LEGACY_MANAGEMENT_MESSAGE = (
    "Profile discovery is temporarily unavailable because the repository contains "
    "legacy YAML. The in-memory draft can still run, but profile writes are disabled."
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
    migration_error: str | None = None
    management_error: str | None = None
    selection_error: str | None = None

    @property
    def can_start(self) -> bool:
        return (
            self.status == "ready"
            and self.migration_error is None
            and self.selection_error is None
        )

    @property
    def is_legacy_preview(self) -> bool:
        return self.profile.is_legacy_preview


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
        schema_version=2,
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


def _is_legacy_profile_error(error: ValueError) -> bool:
    message = str(error).casefold().replace("_", " ")
    return "schema version" in message and ("missing" in message or "legacy" in message)


def configure_model(
    catalog: Any,
    store: Any,
    scenario_id: str,
    *,
    selected_profile_id: str | None = None,
) -> ConfigureModel:
    """Build a saved/default profile model or an explicit schema-v2 draft."""

    scenario = catalog.scenario(scenario_id)
    display_name = str(scenario["display_name"])
    status = str(catalog.scenario_status(scenario_id))
    migration_error: str | None = None
    management_error: str | None = None
    selection_error: str | None = None
    discovery_blocked = False
    try:
        profiles = tuple(store.discover(scenario_id))
    except ValueError as exc:
        if not _is_legacy_profile_error(exc):
            raise
        discovery_blocked = True
        profiles = ()
        management_error = LEGACY_MANAGEMENT_MESSAGE
        migration_error = (
            LEGACY_MIGRATION_MESSAGE
            if scenario.get("config_path") is not None
            else None
        )
        if (
            scenario.get("config_path") is None
            and selected_profile_id not in {None, "__new__"}
        ):
            selection_error = PROFILE_SELECTION_MESSAGE

    default_path = _catalog_profile_path(catalog, scenario)
    if selected_profile_id == "__new__":
        selected = None
    elif selected_profile_id is not None:
        selected = (
            None
            if discovery_blocked
            else next(
                (
                    profile
                    for profile in profiles
                    if profile.profile_id == selected_profile_id
                ),
                None,
            )
        )
        if selected is None and not discovery_blocked:
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
            if selected is None and not discovery_blocked:
                selection_error = PROFILE_SELECTION_MESSAGE
    persisted = selected is not None
    profile = selected or _draft_profile(catalog, scenario_id, display_name)
    if profile.is_legacy_preview:
        migration_error = LEGACY_MIGRATION_MESSAGE
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
        migration_error=migration_error,
        management_error=management_error,
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

    if profile.is_legacy_preview and values:
        raise ValueError(
            "Legacy profile preview is read-only; use explicit Save to convert it"
        )
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


def _powershell_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def render_configure_picker(ui: Any, translator: Any, catalog: Any) -> None:
    """Render a valid `/configure` route instead of leaving a dead navigation link."""

    with ui.column().classes("lte-page lte-configure-picker"):
        ui.label(translator.text("configure.choose_title")).classes("lte-page-title")
        ui.label(translator.text("configure.choose_subtitle")).classes("lte-page-subtitle")
        with ui.row().classes("lte-card-grid"):
            for scenario_id in catalog.scenarios_by_id:
                scenario = catalog.scenario(scenario_id)
                status = catalog.scenario_status(scenario_id)
                with ui.card().classes("lte-scenario-card"):
                    ui.label(str(scenario["display_name"])).classes("lte-card-title")
                    ui.chip(status).classes("lte-status-chip")
                    button = ui.button(
                        translator.text("action.configure"),
                        on_click=lambda scenario_id=scenario_id: ui.navigate.to(
                            f"/configure/{scenario_id}"
                        ),
                    ).mark(f"picker-configure-{scenario_id}")
                    button.set_enabled(status == "ready")


def _confirmation_dialog(
    ui: Any,
    translator: Any,
    *,
    title_key: str,
    on_confirm: Callable[[], None],
) -> Any:
    with ui.dialog() as dialog, ui.card().classes("lte-confirmation-dialog"):
        ui.label(translator.text(title_key)).classes("lte-section-title")
        ui.label(translator.text("confirmation.explanation"))
        with ui.row().classes("justify-end full-width"):
            ui.button(translator.text("action.cancel"), on_click=dialog.close).props(
                "flat"
            )

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
            ui.label(translator.text("configure.unavailable")).classes("lte-page-title")
            ui.label(translator.text("operation.failed")).classes(
                "lte-validation-result--error"
            )
            with ui.expansion(translator.text("validation.details")):
                ui.label(str(exc)).classes("lte-validation-message")
        return

    actions = ProfileActions(store, on_profile_mutation)
    profile = model.profile
    with ui.column().classes("lte-page lte-configure-page"):
        with ui.row().classes("items-end justify-between full-width"):
            with ui.column().classes("gap-1"):
                ui.label(
                    translator.text("configure.title", name=model.display_name)
                ).classes("lte-page-title")
                ui.label(model.scenario_id).classes("lte-page-subtitle")
            dirty_label = ui.label(
                translator.text(
                    "configure.saved" if model.is_persisted else "configure.unsaved"
                )
            ).classes("lte-dirty-indicator")

        if model.status != "ready":
            ui.label(
                translator.text("configure.not_ready", status=model.status)
            ).classes("lte-callout lte-callout--warning")
        if model.migration_error is not None:
            with ui.card().classes("lte-callout lte-callout--warning"):
                ui.label(translator.text("configure.migration_title")).classes(
                    "lte-section-title"
                )
                ui.label(translator.text("configure.migration_body"))
                if model.is_legacy_preview:
                    ui.label(
                        translator.text("configure.migration_revision", revision=(
                            profile.legacy_source.source_revision
                        ))
                    ).classes("lte-technical-detail")
                else:
                    ui.code(
                        "lte-select-sites --config "
                        + _powershell_quote(
                            str(catalog.scenario(scenario_id).get("config_path"))
                        ),
                        language="powershell",
                    ).classes("lte-cli-guidance")
        elif model.selection_error is not None:
            with ui.card().classes("lte-callout lte-callout--warning"):
                ui.label(translator.text("configure.selection_title")).classes(
                    "lte-section-title"
                )
                ui.label(translator.text("configure.selection_body"))
        elif model.management_error is not None:
            with ui.card().classes("lte-callout lte-callout--warning"):
                ui.label(translator.text("configure.management_title")).classes(
                    "lte-section-title"
                )
                ui.label(translator.text("configure.management_body"))

        form_values: dict[str, Any] = {}
        field_elements: dict[str, Any] = {}
        switch_target: dict[str, str | None] = {"value": None}
        safety_controls: list[Any] = []

        options = dict(model.profile_choices)
        options["__new__"] = translator.text("configure.new_profile")
        selected_option = profile.profile_id if model.is_persisted else "__new__"
        profile_select = ui.select(
            options,
            value=selected_option,
            label=translator.text("label.profile"),
        ).classes("lte-profile-select").mark("profile-select")
        profile_select.set_enabled(
            (model.migration_error is None or model.is_legacy_preview)
            and model.management_error is None
        )

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
            ui.label(translator.text("confirmation.explanation"))
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
                switch_dialog.open()
                return
            navigate_to_profile(selected_id)

        profile_select.on_value_change(select_profile)

        def changed(event, field_name: str) -> None:
            form_values[field_name] = event.value
            dirty_label.set_text(translator.text("configure.dirty"))
            dirty_label.classes(add="lte-dirty-indicator--dirty")
            clear_form_errors(field_name)

        with ui.card().classes("lte-form-card"):
            ui.label(translator.text("configure.basic")).classes("lte-section-title")
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

            with ui.row().classes("items-end full-width no-wrap"):
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
                        dirty_label.set_text(translator.text("configure.dirty"))
                        dirty_label.classes(add="lte-dirty-indicator--dirty")

                browse_button = ui.button(
                    translator.text("action.browse"), on_click=browse
                ).props("outline").mark("profile-browse")

        if model.is_legacy_preview:
            for field in (
                profile_id,
                display_name,
                target_crs,
                rect_size,
                target_count,
                scan_step,
                max_rects,
                tolerance,
                strategy,
                random_seed,
                min_spacing,
                scan_mode,
                output_root,
                browse_button,
            ):
                field.set_enabled(False)

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
            on_confirm=lambda: perform_save(confirmed=True),
        )
        migration_dialog = _confirmation_dialog(
            ui,
            translator,
            title_key="confirmation.migrate",
            on_confirm=lambda: perform_save(confirmed=True),
        )

        def request_save() -> None:
            if model.is_legacy_preview:
                migration_dialog.open()
            elif model.is_persisted:
                save_dialog.open()
            else:
                perform_save(confirmed=False)

        source_path = profile.source_path
        persisted_actions_enabled = (
            model.is_persisted
            and source_path is not None
            and model.migration_error is None
            and model.management_error is None
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
            ui.label(translator.text("confirmation.explanation"))
            copy_id = ui.input(
                translator.text("field.profile_id"),
                value=f"{profile.profile_id}-copy",
            ).mark("profile-copy-id")
            copy_name = ui.input(
                translator.text("field.display_name"),
                value=f"{profile.display_name} Copy",
            ).mark("profile-copy-name")
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
            ui.label(translator.text("confirmation.explanation"))
            rename_id = ui.input(
                translator.text("field.profile_id"),
                value=profile.profile_id,
            ).mark("profile-rename-id")
            rename_name = ui.input(
                translator.text("field.display_name"),
                value=profile.display_name,
            ).mark("profile-rename-name")
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
            on_confirm=default_confirmed,
        )
        with ui.dialog() as delete_dialog, ui.card().classes("lte-confirmation-dialog"):
            ui.label(translator.text("confirmation.delete")).classes("lte-section-title")
            ui.label(translator.text("confirmation.explanation"))
            replacement_select = ui.select(
                {
                    str(item.source_path): item.display_name
                    for item in alternatives
                },
                value=None,
                label=translator.text("delete.replacement"),
            )
            replacement_select.set_visibility(bool(deleting_default))
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
                report_preflight_error(outcome)
                return
            if on_preflight_success is None:
                ui.notify(translator.text("preflight.passed"), type="positive")
                return
            try:
                on_preflight_success(outcome)
            except Exception as exc:
                report_error(exc)

        with ui.row().classes("lte-profile-actions full-width"):
            copy_button = ui.button(
                translator.text("action.copy"), on_click=copy_dialog.open
            ).mark("profile-copy")
            rename_button = ui.button(
                translator.text("action.rename"), on_click=rename_dialog.open
            ).mark("profile-rename")
            default_button = ui.button(
                translator.text("action.set_default"), on_click=default_dialog.open
            ).mark("profile-set-default")
            delete_button = ui.button(
                translator.text("action.delete"), on_click=delete_dialog.open
            ).mark("profile-delete")
            for button in (copy_button, rename_button, default_button, delete_button):
                button.set_enabled(persisted_actions_enabled)
            if model.is_default:
                default_button.set_enabled(False)
            if deleting_default and not alternatives:
                delete_button.set_enabled(False)

            ui.space()
            discard = ui.button(
                translator.text("action.discard"), on_click=ui.navigate.reload
            ).props("outline").mark("profile-discard")
            save = ui.button(
                translator.text("action.save"), on_click=request_save
            ).mark("profile-save")
            start = ui.button(
                translator.text("action.start_scan"), on_click=run_preflight
            ).mark("profile-start-scan")
            safety_controls.extend(
                (
                    copy_button,
                    rename_button,
                    default_button,
                    delete_button,
                    save,
                    start,
                )
            )
            writes_enabled = (
                model.migration_error is None
                and model.management_error is None
                and model.selection_error is None
                and model.status == "ready"
            )
            save.set_enabled(writes_enabled or (
                model.is_legacy_preview and model.status == "ready"
            ))
            discard.set_enabled(
                model.migration_error is None or model.is_legacy_preview
            )
            start.set_enabled(model.can_start)


__all__ = [
    "ConfigureModel",
    "ConfirmationRequiredError",
    "LEGACY_MIGRATION_MESSAGE",
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
