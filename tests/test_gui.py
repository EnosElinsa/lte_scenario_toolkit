from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _preserve_toolkit_modules_across_nicegui_reset():
    """Undo NiceGUI test cleanup that removes modules owning page functions."""

    importlib.import_module("lte_scenario_toolkit.gui.app")
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "lte_scenario_toolkit" or name.startswith("lte_scenario_toolkit.")
    }
    yield
    for name in sorted(saved, key=lambda value: value.count(".")):
        sys.modules[name] = saved[name]
    for name, module in saved.items():
        parent_name, separator, child_name = name.rpartition(".")
        if separator and parent_name in sys.modules:
            setattr(sys.modules[parent_name], child_name, module)


def _gui_module(name: str):
    try:
        return importlib.import_module(f"lte_scenario_toolkit.gui.{name}")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("lte_scenario_toolkit.gui"):
            pytest.fail(f"GUI module is missing: {exc.name}")
        raise


def _scenario_preview_fixture_builder(requests, cache_root):
    from PIL import Image

    from lte_scenario_toolkit.gui.scenario_previews import ScenarioPreviewResult

    cache_root.mkdir(parents=True, exist_ok=True)
    results = []
    for index, request in enumerate(requests):
        if index == 1:
            results.append(RuntimeError("one preview failed"))
            continue
        path = cache_root / f"{request.scenario_id}.png"
        Image.new("RGB", (24, 12), (30 + index, 70, 90)).save(path)
        results.append(
            ScenarioPreviewResult(
                "terrain" if index == 0 else "boundary",
                path,
                False,
                None,
            )
        )
    return results


def test_gui_test_dependencies_and_async_mode_are_declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "pytest-asyncio>=0.24" in project["project"]["optional-dependencies"]["dev"]
    assert project["tool"]["pytest"]["ini_options"]["asyncio_mode"] == "auto"
    assert project["tool"]["pytest"]["ini_options"]["main_file"] == ""


def test_gui_css_and_leaflet_extension_are_declared_as_package_data():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["tool"]["setuptools"]["package-data"]["lte_scenario_toolkit"] == [
        "gui/assets/*.css",
        "gui/assets/*.js",
    ]


def test_shared_gui_asset_installer_loads_css_and_registers_station_resource(
    monkeypatch,
):
    assets = _gui_module("assets")
    calls: list[tuple[object, ...]] = []
    fake_app = object()

    class FakeUi:
        def add_css(self, content, *, shared=False):
            calls.append(("css", content, shared))

    monkeypatch.setattr(
        assets,
        "register_station_dots_resource",
        lambda app: calls.append(("station", app)) or "/station-dots.js",
    )

    url = assets.install_gui_assets(fake_app, FakeUi())

    assert url == "/station-dots.js"
    assert calls == [
        ("css", assets.packaged_gui_css(), True),
        ("station", fake_app),
    ]
    assert ":root" in assets.packaged_gui_css()


def test_gui_css_defines_field_atlas_shell_and_mobile_touch_contract():
    css = (
        ROOT
        / "src/lte_scenario_toolkit/gui/assets/app.css"
    ).read_text(encoding="utf-8")

    for declaration in (
        "--lte-canvas: #f2f1eb;",
        "--lte-surface: #fbfaf6;",
        "--lte-surface-strong: #fff;",
        "--lte-frame: #0b3032;",
        "--lte-ink: #102d2e;",
        "--lte-ink-muted: #62716f;",
        "--lte-accent: #149786;",
        "--lte-accent-soft: #d8eee8;",
        "--lte-signal: #c9772e;",
        "--lte-success: #16745f;",
        "--lte-danger: #b64141;",
        "--lte-border: #d6ddd8;",
        "--lte-rail-width: 224px;",
    ):
        assert declaration in css

    assert "@media (max-width: 980px)" in css
    mobile_start = css.index("@media (max-width: 760px)")
    next_media = css.find("@media", mobile_start + 1)
    mobile_block = css[mobile_start : None if next_media == -1 else next_media]

    def declarations(selector: str) -> str:
        match = re.search(
            rf"(?m)^\s*{re.escape(selector)}\s*\{{(?P<body>[^}}]+)\}}",
            mobile_block,
        )
        assert match is not None, f"missing mobile rule for {selector}"
        return match.group("body")

    shell_menu = declarations(".lte-shell-menu")
    assert "min-width: 44px;" in shell_menu
    assert "min-height: 44px;" in shell_menu
    assert ".lte-navigation-toggle { display: none; }" in css
    assert ".lte-navigation-footer" in css
    assert "margin-top: auto;" in css
    assert "min-height: 44px;" in declarations(".lte-nav__link")
    assert "min-height: 44px;" in declarations(".lte-language-select")
    assert "min-height: 44px;" in declarations(
        ".lte-language-select .q-field__control"
    )
    artifact_checkbox = declarations(".lte-generation-artifact-row .q-checkbox")
    assert "min-width: 44px;" in artifact_checkbox
    assert "min-height: 44px;" in artifact_checkbox
    code_copy = declarations(".nicegui-code-copy")
    assert "min-width: 44px;" in code_copy
    assert "min-height: 44px;" in code_copy
    toggle = declarations(".q-toggle")
    assert "min-width: 44px;" in toggle
    assert "min-height: 44px;" in toggle
    checkbox = declarations(".q-checkbox")
    assert "min-width: 44px;" in checkbox
    assert "min-height: 44px;" in checkbox
    dialog_button = declarations(".q-dialog .q-btn")
    assert "min-width: 44px;" in dialog_button
    assert "min-height: 44px;" in dialog_button

    action_rule = re.search(
        r"\.lte-command-bar \.q-btn,(?P<selectors>[^\{]+)"
        r"\{(?P<body>[^}]+)\}",
        mobile_block,
    )
    assert action_rule is not None
    for selector in (
        ".lte-card-actions .q-btn",
        ".lte-profile-actions .q-btn",
        ".lte-confirmation-dialog .q-btn",
        ".lte-candidate-toolbar .q-btn",
        ".lte-generate-page .q-btn",
        ".lte-figures-page .q-btn",
        ".lte-history-page .q-btn",
    ):
        assert selector in action_rule.group("selectors")
    assert "min-height: 44px;" in action_rule.group("body")

    mobile_rules = tuple(
        (match.group("selectors"), match.group("body"))
        for match in re.finditer(
            r"(?P<selectors>[^{}]+)\{(?P<body>[^{}]+)\}",
            mobile_block,
        )
    )
    for selector in (
        ".lte-candidate-page .q-btn",
        ".lte-web-selector-actions .q-btn",
        ".lte-segmented-control .q-btn",
    ):
        matching_bodies = [
            body
            for selectors, body in mobile_rules
            if selector in {item.strip() for item in selectors.split(",")}
        ]
        assert matching_bodies, f"missing 391-760px touch rule for {selector}"
        assert any(
            "min-height: 44px;" in body and "min-width: 44px;" in body
            for body in matching_bodies
        ), f"incomplete 44px touch target for {selector}"
    assert ":focus-visible" in css
    assert "@media (prefers-reduced-motion: reduce)" in css


def test_gui_css_keeps_the_workstation_shell_viewport_bounded_and_motion_safe():
    css = (ROOT / "src/lte_scenario_toolkit/gui/assets/app.css").read_text(
        encoding="utf-8"
    )

    assert "--lte-command-height: 58px;" in css
    assert "height: 100dvh;" in css
    assert "min-height: calc(100dvh - var(--lte-command-height));" in css
    assert "calc(100vh - 64px)" not in css
    assert "box-sizing: border-box;" in css
    assert ".lte-main" in css
    assert "overflow-x: hidden;" in css
    assert "overflow-y: auto;" in css
    root_shell = re.search(
        r"html,\s*body,\s*#app,\s*\.nicegui-content,\s*\.q-page\s*"
        r"\{(?P<body>[^}]+)\}",
        css,
    )
    assert root_shell is not None
    assert "overflow-x: hidden;" in root_shell.group("body")
    assert "overflow-y: hidden;" not in root_shell.group("body")
    assert ".lte-rail-nav" in css
    assert "@media (max-width: 980px)" in css
    assert ".lte-navigation-rail.q-drawer--mobile.q-drawer--mini" in css
    assert "width: 224px !important;" in css
    assert "env(safe-area-inset-top)" in css
    assert "env(safe-area-inset-bottom)" in css
    assert "@media (prefers-reduced-motion: reduce)" in css

    shell_menu = re.search(r"\.lte-shell-menu\s*\{(?P<body>[^}]+)\}", css)
    assert shell_menu is not None
    assert "min-width: 44px;" in shell_menu.group("body")
    assert "min-height: 44px;" in shell_menu.group("body")


def test_every_page_title_declares_level_one_heading_semantics():
    paths = (
        ROOT / "src/lte_scenario_toolkit/gui/presentation.py",
        ROOT / "src/lte_scenario_toolkit/gui/pages/candidates.py",
        ROOT / "src/lte_scenario_toolkit/gui/pages/configure.py",
        ROOT / "src/lte_scenario_toolkit/gui/pages/figures.py",
        ROOT / "src/lte_scenario_toolkit/gui/pages/generate.py",
    )
    missing = []

    for path in paths:
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r'\.classes\("lte-page-title"\)', source):
            suffix = source[match.end() : match.end() + 80]
            if re.search(
                r'\.props\(\s*"role=heading aria-level=1"\s*\)',
                suffix,
            ) is None:
                line = source.count("\n", 0, match.start()) + 1
                missing.append(f"{path.name}:{line}")

    assert missing == []


def test_station_dot_resource_is_local_and_registered_once():
    from lte_scenario_toolkit.gui import leaflet_assets

    calls = []

    class App:
        def remove_route(self, path):
            calls.append(("remove", path))

        def add_static_file(self, **kwargs):
            calls.append(("add", kwargs))
            return kwargs["url_path"]

    url = leaflet_assets.register_station_dots_resource(App())

    assert url == "/_lte_gui/assets/station-dots.js"
    assert calls[0] == ("remove", url)
    assert calls[1][1]["local_file"].name == "station_dots.js"
    assert calls[1][1]["strict"] is True
    source = calls[1][1]["local_file"].read_text(encoding="utf-8")
    assert "L.circleMarker(latlng, dotStyle)" in source
    assert "L.marker(" not in source


def test_translation_dictionaries_have_identical_keys_and_format_values():
    module = _gui_module("i18n")

    assert hasattr(module, "TRANSLATIONS")
    assert hasattr(module, "Translator")
    assert set(module.TRANSLATIONS) == {"en", "zh-CN"}
    assert set(module.TRANSLATIONS["en"]) == set(module.TRANSLATIONS["zh-CN"])
    assert module.Translator("zh-CN").text("nav.scenarios") == "\u573a\u666f"
    assert module.Translator("en").text("nav.collapse") == "Collapse navigation"
    assert module.Translator("en").text("nav.expand") == "Expand navigation"
    assert module.Translator("en").text("nav.more") == "More navigation options"
    assert module.Translator("zh-CN").text("nav.collapse") == "\u6536\u8d77\u5bfc\u822a"
    assert module.Translator("zh-CN").text("nav.expand") == "\u5c55\u5f00\u5bfc\u822a"
    assert module.Translator("zh-CN").text("nav.more") == "\u66f4\u591a\u5bfc\u822a\u9009\u9879"
    assert module.Translator("en").text("job.running", name="Scan 1") == "Running Scan 1"
    assert module.Translator("en").text("shell.eyebrow") == "LTE / Operations"
    assert module.Translator("zh-CN").text("shell.eyebrow") == "LTE / \u4f5c\u4e1a\u53f0"
    assert (
        module.Translator("en").text("action.open_navigation")
        == "Open navigation menu"
    )
    assert (
        module.Translator("zh-CN").text("action.open_navigation")
        == "\u6253\u5f00\u5bfc\u822a\u83dc\u5355"
    )
    assert module.Translator("en").text("candidates.dem_opacity") == (
        "Terrain opacity"
    )
    assert module.Translator("zh-CN").text("candidates.dem_opacity") == (
        "\u5730\u5f62\u56fe\u5c42\u4e0d\u900f\u660e\u5ea6"
    )
    assert (
        module.Translator("en").text("technical.machine_status", status="ready")
        == "Machine status: ready"
    )
    assert (
        module.Translator("zh-CN").text(
            "technical.machine_status", status="ready"
        )
        == "\u5185\u90e8\u72b6\u6001\uff1aready"
    )


def test_presentation_spec_is_an_immutable_compact_value():
    module = _gui_module("presentation")

    spec = module.PresentationSpec("status.example")

    assert spec.label_key == "status.example"
    assert spec.tone == "neutral"
    assert spec.description_key is None
    assert not hasattr(spec, "__dict__")
    with pytest.raises(FrozenInstanceError):
        spec.tone = "info"


def test_presentation_action_spec_is_immutable_and_defaults_to_secondary():
    module = _gui_module("presentation")

    action = module.ActionSpec("save", "Save", lambda: None)

    assert action.name == "save"
    assert action.label == "Save"
    assert callable(action.on_click)
    assert action.role == "secondary"
    assert action.enabled is True
    assert action.marker is None
    assert not hasattr(action, "__dict__")
    with pytest.raises(FrozenInstanceError):
        action.enabled = False


def test_presentation_tertiary_action_keeps_a_distinct_css_hook_and_flat_button():
    module = _gui_module("presentation")
    trace = []

    class Element:
        def __init__(self, kind, *args, **kwargs):
            self.kind = kind
            self.args = args
            self.kwargs = kwargs
            self.class_values = []
            self.prop_calls = []
            self.enabled = None

        def classes(self, value):
            self.class_values.append(value)
            return self

        def props(self, add=None, *, remove=None):
            self.prop_calls.append((add, remove))
            return self

        def set_enabled(self, enabled):
            self.enabled = enabled
            return self

        def mark(self, marker):
            trace.append(("mark", self.kind, marker))
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Ui:
        def row(self):
            element = Element("row")
            trace.append(("row", element))
            return element

        def button(self, *args, **kwargs):
            element = Element("button", *args, **kwargs)
            trace.append(("button", element))
            return element

    buttons = module.render_action_bar(
        Ui(),
        (
            module.ActionSpec("inspect", "Inspect", lambda: None, role="tertiary"),
        ),
    )

    button = buttons["inspect"]
    assert "lte-action lte-action--tertiary" in button.class_values
    assert ("flat", None) in button.prop_calls
    assert button.enabled is True


def test_menu_action_spec_is_an_immutable_typed_menu_contract():
    module = _gui_module("presentation")

    action = module.MenuActionSpec("Inspect", "search", lambda: None)

    assert action.label == "Inspect"
    assert action.icon == "search"
    assert callable(action.handler)
    assert action.enabled is True
    assert action.separator is False
    assert action.marker is None
    assert action.role == "secondary"
    assert not hasattr(action, "__dict__")
    with pytest.raises(FrozenInstanceError):
        action.enabled = False


def test_render_overflow_menu_composes_accessible_semantic_actions():
    module = _gui_module("presentation")
    elements = []

    class Element:
        def __init__(self, kind, *args, **kwargs):
            self.kind = kind
            self.args = args
            self.kwargs = kwargs
            self.class_values = []
            self.prop_calls = []
            self.markers = []

        def classes(self, value):
            self.class_values.append(value)
            return self

        def props(self, add=None, *, remove=None):
            self.prop_calls.append((add, remove))
            return self

        def mark(self, marker):
            self.markers.append(marker)
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Ui:
        def _element(self, kind, *args, **kwargs):
            element = Element(kind, *args, **kwargs)
            elements.append(element)
            return element

        def button(self, *args, **kwargs):
            return self._element("button", *args, **kwargs)

        def menu(self):
            return self._element("menu")

        def menu_item(self, *args, **kwargs):
            return self._element("menu_item", *args, **kwargs)

        def separator(self):
            return self._element("separator")

        def icon(self, *args, **kwargs):
            return self._element("icon", *args, **kwargs)

    def inspect():
        return None

    def purge():
        return None
    trigger = module.render_overflow_menu(
        Ui(),
        (
            module.MenuActionSpec("Inspect", "search", inspect, marker="inspect"),
            module.MenuActionSpec(
                "Purge",
                "delete",
                purge,
                enabled=False,
                separator=True,
                marker="purge",
                role="danger",
            ),
        ),
        marker="more-actions",
    )

    assert trigger.kind == "button"
    assert trigger.kwargs == {"icon": "more_vert"}
    assert trigger.markers == ["more-actions"]
    assert (
        'flat round aria-haspopup=menu aria-label="More actions"',
        None,
    ) in trigger.prop_calls

    menu_items = [element for element in elements if element.kind == "menu_item"]
    assert [(item.args, item.kwargs) for item in menu_items] == [
        (("Inspect",), {"on_click": inspect}),
        (("Purge",), {"on_click": None}),
    ]
    assert menu_items[0].markers == ["inspect"]
    assert menu_items[1].markers == ["purge"]
    assert "lte-overflow-menu__item--danger text-negative" in menu_items[1].class_values
    assert ("disable aria-disabled=true", None) in menu_items[1].prop_calls
    assert len([element for element in elements if element.kind == "separator"]) == 1
    assert [element.args for element in elements if element.kind == "icon"] == [
        ("search",),
        ("delete",),
    ]


def test_shared_presentation_helpers_only_compose_callbacks_and_content():
    module = _gui_module("presentation")
    elements = []
    rendered = []

    class Element:
        def __init__(self, kind, *args, **kwargs):
            self.kind = kind
            self.args = args
            self.kwargs = kwargs
            self.class_values = []
            self.prop_calls = []
            self.markers = []
            self.value_callbacks = []

        def classes(self, value):
            self.class_values.append(value)
            return self

        def props(self, add=None, *, remove=None):
            self.prop_calls.append((add, remove))
            return self

        def mark(self, marker):
            self.markers.append(marker)
            return self

        def on_value_change(self, callback):
            self.value_callbacks.append(callback)
            return self

        def set_enabled(self, _enabled):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Ui:
        def _element(self, kind, *args, **kwargs):
            element = Element(kind, *args, **kwargs)
            elements.append(element)
            return element

        def element(self, *args, **kwargs):
            return self._element("element", *args, **kwargs)

        def row(self, *args, **kwargs):
            return self._element("row", *args, **kwargs)

        def column(self, *args, **kwargs):
            return self._element("column", *args, **kwargs)

        def label(self, *args, **kwargs):
            return self._element("label", *args, **kwargs)

        def button(self, *args, **kwargs):
            return self._element("button", *args, **kwargs)

        def right_drawer(self, *args, **kwargs):
            return self._element("right_drawer", *args, **kwargs)

    ui = Ui()
    actions = (module.ActionSpec("apply", "Apply", lambda: None, role="primary"),)
    buttons = module.render_sticky_action_dock(
        ui,
        actions,
        label="Candidate actions",
        marker="dock",
    )
    def drawer_callback():
        return None
    drawer = module.render_inspector_drawer(
        ui,
        "Details",
        lambda: rendered.append("drawer-content"),
        value=True,
        on_value_change=drawer_callback,
        marker="inspector",
    )
    heading = module.render_section_heading(
        ui,
        "Artifacts",
        "Available outputs",
        actions,
        marker="artifacts-heading",
    )

    assert set(buttons) == {"apply"}
    dock = next(
        element
        for element in elements
        if element.kind == "element" and element.args == ("footer",)
    )
    assert dock.markers == ["dock"]
    assert ('role=region aria-label="Candidate actions"', None) in dock.prop_calls
    assert "lte-action-dock" in dock.class_values

    assert drawer.kwargs == {"value": True, "fixed": True, "bordered": True}
    assert drawer.markers == ["inspector"]
    assert drawer.value_callbacks == [drawer_callback]
    assert "lte-inspector-drawer" in drawer.class_values
    assert rendered == ["drawer-content"]

    assert heading.markers == ["artifacts-heading"]
    assert "lte-section-heading" in heading.class_values
    labels = [element for element in elements if element.kind == "label"]
    title = next(label for label in labels if label.args == ("Artifacts",))
    assert ("role=heading aria-level=2", None) in title.prop_calls
    assert any(label.args == ("Available outputs",) for label in labels)


async def test_inspector_drawer_can_be_created_from_a_nested_page_body(user):
    from nicegui import ui

    from lte_scenario_toolkit.gui.presentation import render_inspector_drawer

    @ui.page("/presentation-inspector-drawer")
    def inspector_page():
        with ui.column().mark("inspector-page-body"):
            render_inspector_drawer(
                ui,
                "Inspector",
                lambda: ui.label("Drawer body").mark("inspector-drawer-content"),
                marker="inspector-drawer",
            )

    await user.open("/presentation-inspector-drawer")

    await user.should_see(marker="inspector-page-body")
    await user.should_see(marker="inspector-drawer")
    await user.should_see(marker="inspector-drawer-content", content="Drawer body")


def test_presentation_mappings_cover_current_gui_machine_values():
    module = _gui_module("presentation")
    spec = module.PresentationSpec

    assert {
        value: module.readiness_presentation(value)
        for value in ("ready", "boundary-ready", "dem-pending", "invalid")
    } == {
        "ready": spec(
            "status.ready", "success", "readiness.ready.description"
        ),
        "boundary-ready": spec(
            "status.boundary_ready",
            "warning",
            "readiness.boundary_ready.description",
        ),
        "dem-pending": spec(
            "status.dem_pending",
            "warning",
            "readiness.dem_pending.description",
        ),
        "invalid": spec(
            "status.invalid", "danger", "readiness.invalid.description"
        ),
    }
    assert {
        value: module.cache_presentation(value)
        for value in ("none", "hit", "miss", "forced")
    } == {
        "none": spec("cache.none"),
        "hit": spec("cache.hit", "success"),
        "miss": spec("cache.miss", "info"),
        "forced": spec("cache.forced", "info"),
    }
    assert {
        value: module.scan_mode_presentation(value)
        for value in ("fast", "complete")
    } == {
        "fast": spec("scan.fast"),
        "complete": spec("scan.complete"),
    }
    assert {
        value: module.artifact_label_presentation(value)
        for value in (
            "csv",
            "preview_png",
            "terrain_png",
            "terrain_eps",
            "terrain_html",
        )
    } == {
        "csv": spec(
            "generate.artifact.csv",
            description_key="generate.artifact.csv.description",
        ),
        "preview_png": spec(
            "generate.artifact.preview_png",
            description_key="generate.artifact.preview_png.description",
        ),
        "terrain_png": spec(
            "generate.artifact.terrain_png",
            description_key="generate.artifact.terrain_png.description",
        ),
        "terrain_eps": spec(
            "generate.artifact.terrain_eps",
            description_key="generate.artifact.terrain_eps.description",
        ),
        "terrain_html": spec(
            "generate.artifact.terrain_html",
            description_key="generate.artifact.terrain_html.description",
        ),
    }
    assert {
        value: module.artifact_state_presentation(value)
        for value in ("not-requested", "pending", "published", "failed")
    } == {
        "not-requested": spec("status.not_requested"),
        "pending": spec("status.pending", "active"),
        "published": spec("status.published", "success"),
        "failed": spec("status.failed", "danger"),
    }
    assert {
        value: module.run_state_presentation(value)
        for value in ("completed", "partial")
    } == {
        "completed": spec("status.completed", "success"),
        "partial": spec("status.partial", "warning"),
    }
    assert {
        value: module.job_kind_presentation(value)
        for value in (
            "validation.full_checksum",
            "generate",
            "selection.scan",
            "candidate.dem_style",
            "candidate.statistics",
            "figure-source",
            "figure-preview",
            "figure-export",
        )
    } == {
        "validation.full_checksum": spec("job.kind.full_checksum", "active"),
        "generate": spec("job.kind.generate", "active"),
        "selection.scan": spec("job.kind.selection_scan", "active"),
        "candidate.dem_style": spec("job.kind.dem_style", "active"),
        "candidate.statistics": spec("job.kind.statistics", "active"),
        "figure-source": spec("job.kind.figure_source", "active"),
        "figure-preview": spec("job.kind.figure_preview", "active"),
        "figure-export": spec("job.kind.figure_export", "active"),
    }


def test_presentation_mappings_fail_safe_for_unknown_values():
    module = _gui_module("presentation")
    unknown = module.PresentationSpec("status.unknown")

    for mapping in (
        module.readiness_presentation,
        module.cache_presentation,
        module.scan_mode_presentation,
        module.artifact_label_presentation,
        module.artifact_state_presentation,
        module.run_state_presentation,
        module.job_kind_presentation,
    ):
        assert mapping("future-machine-value") == unknown
        assert mapping(None) == unknown


def test_trash_action_presentation_preserves_state_order_and_semantic_labels():
    from lte_scenario_toolkit.run_trash import TrashState

    module = _gui_module("presentation")

    assert module.TRASH_ACTIONS_BY_STATE == {
        TrashState.TRASHED: (module.TrashAction.RESTORE, module.TrashAction.PURGE),
        TrashState.RECOVERY_REQUIRED: (module.TrashAction.RECOVER,),
        TrashState.PURGE_FAILED: (module.TrashAction.PURGE,),
    }
    assert [
        module.trash_action_presentation(action)
        for action in module.TRASH_ACTIONS_BY_STATE[TrashState.TRASHED]
    ] == [
        module.PresentationSpec("trash.action.restore"),
        module.PresentationSpec("trash.action.purge", "danger"),
    ]
    assert module.trash_action_presentation(module.TrashAction.RECOVER) == (
        module.PresentationSpec("trash.action.recover", "warning")
    )


def test_trash_job_kinds_use_localizable_shared_job_gate():
    from lte_scenario_toolkit.jobs import JobBusyError, JobCoordinator

    module = _gui_module("presentation")
    spec = module.PresentationSpec
    assert {
        value: module.job_kind_presentation(value)
        for value in (
            "history.trash_move",
            "history.trash_restore",
            "history.trash_purge",
        )
    } == {
        "history.trash_move": spec("job.kind.history_trash_move", "active"),
        "history.trash_restore": spec(
            "job.kind.history_trash_restore",
            "active",
        ),
        "history.trash_purge": spec("job.kind.history_trash_purge", "active"),
    }

    coordinator = JobCoordinator()
    blocker = coordinator.start("figure-preview")
    try:
        with pytest.raises(JobBusyError):
            coordinator.start("history.trash_move")
    finally:
        assert coordinator.finish(blocker.job_id) is True

    for fails in (False, True):
        job = coordinator.start("history.trash_restore")
        try:
            if fails:
                raise RuntimeError("simulated callback failure")
        except RuntimeError:
            pass
        finally:
            assert coordinator.finish(job.job_id) is True
        assert coordinator.snapshot().active is False
    coordinator.shutdown()


def test_presentation_translation_keys_are_complete_and_localized():
    i18n = _gui_module("i18n")
    presentation = _gui_module("presentation")
    machine_values = {
        presentation.readiness_presentation: (
            "ready",
            "boundary-ready",
            "dem-pending",
            "invalid",
        ),
        presentation.cache_presentation: ("none", "hit", "miss", "forced"),
        presentation.scan_mode_presentation: ("fast", "complete"),
        presentation.artifact_label_presentation: (
            "csv",
            "preview_png",
            "terrain_png",
            "terrain_eps",
            "terrain_html",
        ),
        presentation.artifact_state_presentation: (
            "not-requested",
            "pending",
            "published",
            "failed",
        ),
        presentation.run_state_presentation: ("completed", "partial"),
        presentation.job_kind_presentation: (
            "validation.full_checksum",
            "generate",
            "selection.scan",
            "candidate.dem_style",
            "candidate.statistics",
            "figure-source",
            "figure-preview",
            "figure-export",
        ),
    }
    keys = {"status.unknown"}
    for mapping, values in machine_values.items():
        for value in values:
            item = mapping(value)
            keys.add(item.label_key)
            if item.description_key is not None:
                keys.add(item.description_key)

    for language in i18n.SUPPORTED_LANGUAGES:
        assert keys <= set(i18n.TRANSLATIONS[language])
    assert i18n.Translator("en").text("status.boundary_ready") == "Boundary available"
    assert i18n.Translator("zh-CN").text("status.boundary_ready") == "\u8fb9\u754c\u6570\u636e\u53ef\u7528"
    assert i18n.Translator("en").text("cache.hit") == "Cached scan"
    assert i18n.Translator("zh-CN").text("cache.hit") == "\u5df2\u590d\u7528\u7f13\u5b58\u626b\u63cf"
    assert i18n.Translator("en").text("cache.forced") == "Forced fresh scan"
    assert i18n.Translator("zh-CN").text("cache.forced") == "\u5f3a\u5236\u5168\u65b0\u626b\u63cf"
    assert i18n.Translator("en").text("scan.complete") == "Complete scan"
    assert i18n.Translator("zh-CN").text("scan.complete") == "\u5b8c\u6574\u626b\u63cf"
    assert i18n.Translator("en").text("job.kind.figure_export") == "Figure export"
    assert i18n.Translator("zh-CN").text("job.kind.figure_export") == "\u56fe\u8868\u5bfc\u51fa"


def test_development_option_prefixes_are_absent_from_all_translations():
    module = _gui_module("i18n")
    prefixed = [
        (language, key, text)
        for language, translations in module.TRANSLATIONS.items()
        for key, text in translations.items()
        if re.match(r"^[ABC][:\uff1a]", text)
    ]

    assert prefixed == []
    assert module.Translator("en").text("candidates.view_map") == "Map"
    assert (
        module.Translator("en").text("candidates.view_filmstrip")
        == "Map and candidates"
    )
    assert module.Translator("zh-CN").text("candidates.view_map") == "\u5730\u56fe"
    assert (
        module.Translator("zh-CN").text("candidates.view_filmstrip")
        == "\u5730\u56fe\u4e0e\u5019\u9009\u533a\u57df"
    )


def test_gui_settings_are_local_and_atomic(tmp_path):
    module = _gui_module("settings")

    assert hasattr(module, "GuiSettingsStore")
    store = module.GuiSettingsStore(tmp_path)
    store.save(
        language="zh-CN",
        output_roots=[tmp_path / "outputs", Path("outputs"), tmp_path / "other"],
    )

    loaded = store.load()

    assert loaded.language == "zh-CN"
    assert loaded.output_roots == (
        (tmp_path / "outputs").resolve(),
        (tmp_path / "other").resolve(),
    )
    assert loaded.navigation_collapsed is False
    assert (tmp_path / ".lte-data/gui-settings.json").is_file()
    assert not list((tmp_path / ".lte-data").glob("*.tmp"))


def test_gui_settings_navigation_preference_roundtrip_preserves_other_settings(tmp_path):
    module = _gui_module("settings")
    store = module.GuiSettingsStore(tmp_path)
    first_root = tmp_path / "outputs"
    second_root = tmp_path / "other"

    saved = store.save(
        language="zh-CN",
        output_roots=[first_root],
        navigation_collapsed=True,
    )
    updated = store.update(
        navigation_collapsed=False,
        add_output_roots=[second_root],
    )
    restarted = module.GuiSettingsStore(tmp_path).load()

    assert saved.navigation_collapsed is True
    assert updated.language == "zh-CN"
    assert updated.output_roots == (first_root.resolve(), second_root.resolve())
    assert updated.navigation_collapsed is False
    assert restarted == updated
    assert json.loads(store.path.read_text(encoding="utf-8")) == {
        "language": "zh-CN",
        "output_roots": [str(first_root.resolve()), str(second_root.resolve())],
        "navigation_collapsed": False,
    }


def test_gui_settings_legacy_schema_preserves_existing_preferences(
    tmp_path,
):
    module = _gui_module("settings")
    store = module.GuiSettingsStore(tmp_path)
    output_root = (tmp_path / "outputs").resolve()
    store.path.parent.mkdir()
    legacy = {
        "language": "zh-CN",
        "output_roots": [str(output_root)],
    }
    store.path.write_text(json.dumps(legacy), encoding="utf-8")

    loaded = store.load()

    updated = store.update(navigation_collapsed=True)

    assert loaded.language == "zh-CN"
    assert loaded.output_roots == (output_root,)
    assert loaded.navigation_collapsed is False
    assert updated.language == "zh-CN"
    assert updated.output_roots == (output_root,)
    assert updated.navigation_collapsed is True
    assert json.loads(store.path.read_text(encoding="utf-8")) == {
        "language": "zh-CN",
        "output_roots": [str(output_root)],
        "navigation_collapsed": True,
    }


def test_gui_settings_save_preserves_navigation_preference_when_omitted(tmp_path):
    module = _gui_module("settings")
    store = module.GuiSettingsStore(tmp_path)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    store.save(
        language="en",
        output_roots=[first_root],
        navigation_collapsed=True,
    )

    saved = store.save(language="zh-CN", output_roots=[second_root])

    assert saved.language == "zh-CN"
    assert saved.output_roots == (second_root.resolve(),)
    assert saved.navigation_collapsed is True


def test_gui_defaults_to_loopback_and_opens_browser():
    module = _gui_module("app")

    assert hasattr(module, "build_parser")
    args = module.build_parser().parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 8080
    assert args.open_browser is True


def test_gui_parser_resolves_all_shell_options_and_rejects_invalid_ports(tmp_path):
    module = _gui_module("app")

    args = module.build_parser().parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "--catalog",
            "catalogs/local.yaml",
            "--host",
            "::1",
            "--port",
            "9000",
            "--no-browser",
            "--check",
        ]
    )

    assert args.repo_root == tmp_path
    assert args.catalog == Path("catalogs/local.yaml")
    assert args.host == "::1"
    assert args.port == 9000
    assert args.open_browser is False
    assert args.check is True
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--port", "0"])
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--port", "65536"])


def test_gui_check_uses_resolved_repo_paths_without_starting_app(
    tmp_path, monkeypatch, capsys
):
    import socket
    import urllib.request

    module = _gui_module("app")
    assert hasattr(module, "main")
    calls: dict[str, object] = {}
    catalog = SimpleNamespace(root=tmp_path.resolve())

    def load_catalog(path, *, repo_root):
        calls["catalog_path"] = path
        calls["repo_root"] = repo_root
        return catalog

    class RecordingSettingsStore:
        def __init__(self, repo_root):
            calls["settings_root"] = repo_root

        def load(self):
            calls["settings_loaded"] = True
            return SimpleNamespace(language="en", output_roots=())

    monkeypatch.setattr(module, "load_data_catalog", load_catalog)
    monkeypatch.setattr(module, "GuiSettingsStore", RecordingSettingsStore)
    monkeypatch.setattr(module, "validate_translations", lambda: calls.setdefault("i18n", True))
    monkeypatch.setattr(
        module,
        "create_app",
        lambda *args, **kwargs: pytest.fail("--check must not create the app"),
    )
    monkeypatch.setitem(
        sys.modules,
        "nicegui",
        SimpleNamespace(
            ui=SimpleNamespace(
                run=lambda *args, **kwargs: pytest.fail("--check must not run the UI")
            )
        ),
    )

    def fail_network(*_args, **_kwargs):
        pytest.fail("--check must not perform a network request")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(socket, "getaddrinfo", fail_network)
    monkeypatch.setattr(urllib.request, "urlopen", fail_network)

    result = module.main(
        [
            "--repo-root",
            str(tmp_path),
            "--catalog",
            "catalogs/local.yaml",
            "--host",
            "0.0.0.0",
            "--check",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == "GUI preflight OK\n"
    assert "WARNING:" in captured.err
    assert "0.0.0.0" in captured.err
    assert "read and write local experiment paths" in captured.err
    assert calls == {
        "catalog_path": (tmp_path / "catalogs/local.yaml").resolve(),
        "repo_root": tmp_path.resolve(),
        "settings_root": tmp_path.resolve(),
        "settings_loaded": True,
        "i18n": True,
    }


def test_gui_missing_extra_prints_exact_install_instruction(
    tmp_path, monkeypatch, capsys
):
    module = _gui_module("app")
    assert hasattr(module, "main")
    real_import = __import__

    def missing_nicegui(name, *args, **kwargs):
        if name == "nicegui":
            raise ModuleNotFoundError("No module named 'nicegui'", name="nicegui")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", missing_nicegui)

    result = module.main(["--repo-root", str(tmp_path), "--check"])

    captured = capsys.readouterr()
    assert result != 0
    assert 'python -m pip install -e ".[gui]"' in captured.err
    assert "Traceback" not in captured.err


def test_gui_does_not_mask_nicegui_internal_import_failures(tmp_path, monkeypatch):
    module = _gui_module("app")
    assert hasattr(module, "main")
    real_import = __import__

    def missing_internal_dependency(name, *args, **kwargs):
        if name == "nicegui":
            raise ModuleNotFoundError(
                "No module named 'nicegui_internal'", name="nicegui_internal"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", missing_internal_dependency)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        module.main(["--repo-root", str(tmp_path), "--check"])

    assert exc_info.value.name == "nicegui_internal"


def test_create_app_uses_injected_catalog_and_shared_local_css(
    tmp_path, monkeypatch
):
    module = _gui_module("app")
    assert hasattr(module, "create_app")
    calls: dict[str, object] = {}

    class FakeApp:
        def remove_route(self, path):
            calls["removed_route"] = path

        def add_static_file(self, **kwargs):
            calls["static_file"] = kwargs
            return kwargs["url_path"]

    fake_app = FakeApp()

    class FakeUi:
        def add_css(self, content, *, shared=False):
            calls["css"] = content
            calls["shared"] = shared

        def page(self, path, **kwargs):
            calls["page"] = (path, kwargs)
            return lambda function: function

    monkeypatch.setitem(sys.modules, "nicegui", SimpleNamespace(app=fake_app, ui=FakeUi()))
    monkeypatch.setattr(
        module,
        "load_data_catalog",
        lambda *args, **kwargs: pytest.fail("injected catalog must be used offline"),
    )
    catalog = SimpleNamespace(root=tmp_path.resolve())

    created = module.create_app(catalog=catalog, testing=True)

    assert created is fake_app
    assert calls["shared"] is True
    assert ":root" in calls["css"]
    assert "--lte-canvas: #f2f1eb" in calls["css"]
    assert calls["page"][0] == "/"
    assert calls["removed_route"] == "/_lte_gui/assets/station-dots.js"
    assert calls["static_file"]["url_path"] == "/_lte_gui/assets/station-dots.js"


def test_gui_main_passes_server_flags_to_nicegui(tmp_path, monkeypatch, capsys):
    module = _gui_module("app")
    catalog = SimpleNamespace(root=tmp_path.resolve())
    calls: dict[str, object] = {}
    fake_ui = SimpleNamespace(
        run=lambda **kwargs: calls.setdefault("run", kwargs),
    )
    monkeypatch.setitem(sys.modules, "nicegui", SimpleNamespace(ui=fake_ui))
    monkeypatch.setattr(module, "_preflight", lambda *args: catalog)
    monkeypatch.setattr(
        module,
        "create_app",
        lambda *, catalog: calls.setdefault("catalog", catalog),
    )

    result = module.main(
        [
            "--repo-root",
            str(tmp_path),
            "--host",
            "localhost",
            "--port",
            "8091",
            "--no-browser",
        ]
    )

    assert result == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Press Ctrl+C to stop." in captured.out
    assert calls["catalog"] is catalog
    assert calls["run"] == {
        "host": "localhost",
        "port": 8091,
        "show": False,
        "title": "LTE Scenario Toolkit",
        "reload": False,
    }


def test_gui_app_import_does_not_require_nicegui_in_fresh_process():
    script = """
import builtins

real_import = builtins.__import__
def reject_nicegui(name, *args, **kwargs):
    if name == 'nicegui' or name.startswith('nicegui.'):
        raise ModuleNotFoundError("No module named 'nicegui'", name='nicegui')
    return real_import(name, *args, **kwargs)

builtins.__import__ = reject_nicegui
import lte_scenario_toolkit.gui
import lte_scenario_toolkit.gui.app
print('GUI import OK')
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "GUI import OK\n"


def test_translator_rejects_unknown_languages_keys_and_format_values():
    module = _gui_module("i18n")

    with pytest.raises(ValueError, match="Unsupported GUI language"):
        module.Translator("fr")
    with pytest.raises(KeyError):
        module.Translator("en").text("missing.key")
    with pytest.raises(KeyError):
        module.Translator("en").text("job.running")


def test_translation_validation_rejects_key_and_placeholder_drift(monkeypatch):
    module = _gui_module("i18n")

    monkeypatch.delitem(module.TRANSLATIONS["zh-CN"], "nav.history")
    with pytest.raises(ValueError, match="keys do not match"):
        module.validate_translations()
    monkeypatch.undo()

    monkeypatch.setitem(
        module.TRANSLATIONS["zh-CN"],
        "job.running",
        "\u6b63\u5728\u8fd0\u884c {job_id}",
    )
    with pytest.raises(ValueError, match="placeholders do not match"):
        module.validate_translations()


@pytest.mark.parametrize("placeholder", ["{obj.attr}", "{obj[index]}"])
def test_translation_validation_rejects_traversal_placeholders(
    placeholder, monkeypatch
):
    module = _gui_module("i18n")
    for language in module.SUPPORTED_LANGUAGES:
        monkeypatch.setitem(
            module.TRANSLATIONS[language],
            "job.running",
            f"Running {placeholder}",
        )

    with pytest.raises(ValueError, match="Unsafe translation placeholder"):
        module.validate_translations()


def test_translation_values_preserve_literal_braces():
    module = _gui_module("i18n")

    assert module.Translator("en").text("job.running", name="scan {A}") == (
        "Running scan {A}"
    )


def test_missing_gui_settings_use_english_defaults_without_writing(tmp_path):
    module = _gui_module("settings")

    settings = module.GuiSettingsStore(tmp_path).load()

    assert settings.language == "en"
    assert settings.output_roots == ()
    assert settings.navigation_collapsed is False
    assert not (tmp_path / ".lte-data").exists()


@pytest.mark.parametrize(
    "document",
    [
        {"language": "fr", "output_roots": []},
        {"language": True, "output_roots": []},
        {"language": "en", "output_roots": "outputs"},
        {"language": "en", "output_roots": ["relative"]},
        {
            "language": "en",
            "output_roots": [],
            "unknown": True,
        },
    ],
)
def test_gui_settings_fall_back_for_malformed_documents(tmp_path, document):
    module = _gui_module("settings")
    path = tmp_path / ".lte-data/gui-settings.json"
    path.parent.mkdir()
    path.write_text(json.dumps(document), encoding="utf-8")
    original = path.read_bytes()

    settings = module.GuiSettingsStore(tmp_path).load()

    assert settings == module.GuiSettings()
    assert path.read_bytes() == original


def test_gui_settings_fall_back_for_invalid_json_without_overwriting_it(tmp_path):
    module = _gui_module("settings")
    path = tmp_path / ".lte-data/gui-settings.json"
    path.parent.mkdir()
    path.write_text("{broken", encoding="utf-8")

    settings = module.GuiSettingsStore(tmp_path).load()

    assert settings == module.GuiSettings()
    assert path.read_text(encoding="utf-8") == "{broken"


def test_gui_settings_reject_invalid_languages_and_file_output_roots(tmp_path):
    module = _gui_module("settings")
    store = module.GuiSettingsStore(tmp_path)
    output_file = tmp_path / "output.txt"
    output_file.write_text("occupied", encoding="utf-8")

    with pytest.raises(module.GuiSettingsError, match="Unsupported GUI language"):
        store.save(language="fr", output_roots=[])
    with pytest.raises(module.GuiSettingsError, match="not a directory"):
        store.save(language="en", output_roots=[output_file])

    assert not (tmp_path / ".lte-data").exists()


def test_gui_settings_reject_traversal_output_roots(tmp_path):
    module = _gui_module("settings")
    store = module.GuiSettingsStore(tmp_path)

    with pytest.raises(module.GuiSettingsError, match="must not contain traversal"):
        store.save(
            language="en",
            output_roots=[tmp_path / "results" / ".." / "outside"],
        )


def test_gui_settings_reject_redirected_output_roots(tmp_path):
    module = _gui_module("settings")
    store = module.GuiSettingsStore(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    redirected = tmp_path / "redirected"
    try:
        redirected.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(module.GuiSettingsError, match="symlink or junction"):
        store.save(language="en", output_roots=[redirected])


def test_gui_settings_reject_junction_output_root(tmp_path, monkeypatch):
    module = _gui_module("settings")
    output_root = tmp_path / "output"
    output_root.mkdir()
    original = module._is_link_or_junction

    def mark_output_as_junction(path):
        return path == output_root or original(path)

    monkeypatch.setattr(module, "_is_link_or_junction", mark_output_as_junction)
    with pytest.raises(module.GuiSettingsError, match="symlink or junction"):
        module.GuiSettingsStore(tmp_path).save(
            language="en",
            output_roots=[output_root],
        )


def test_gui_settings_redirect_detector_supports_windows_reparse_points(monkeypatch):
    import stat

    module = _gui_module("settings")
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    monkeypatch.setattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", flag, raising=False)

    class ReparsePath:
        @staticmethod
        def is_symlink():
            return False

        @staticmethod
        def lstat():
            return SimpleNamespace(st_file_attributes=flag)

    assert module._is_link_or_junction(ReparsePath()) is True


def test_gui_settings_never_persist_environment_or_credentials(tmp_path, monkeypatch):
    module = _gui_module("settings")
    monkeypatch.setenv("EARTHENGINE_TOKEN", "do-not-persist")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "secret.json")

    module.GuiSettingsStore(tmp_path).save(
        language="en",
        output_roots=[tmp_path / "results"],
    )

    document = (tmp_path / ".lte-data" / "gui-settings.json").read_text(
        encoding="utf-8"
    )
    assert "do-not-persist" not in document
    assert "secret.json" not in document
    assert set(json.loads(document)) == {
        "language",
        "output_roots",
        "navigation_collapsed",
    }

def test_gui_settings_atomic_failure_preserves_previous_file(tmp_path, monkeypatch):
    module = _gui_module("settings")
    store = module.GuiSettingsStore(tmp_path)
    store.save(language="en", output_roots=[])
    original = store.path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(module.os, "replace", fail_replace)

    with pytest.raises(module.GuiSettingsError, match="Could not write"):
        store.save(language="zh-CN", output_roots=[])

    assert store.path.read_bytes() == original
    assert not list(store.path.parent.glob("*.tmp"))


def test_gui_settings_reject_non_directory_storage_path(tmp_path):
    module = _gui_module("settings")
    settings_dir = tmp_path / ".lte-data"
    settings_dir.write_text("occupied", encoding="utf-8")

    with pytest.raises(module.GuiSettingsError, match="not a directory"):
        module.GuiSettingsStore(tmp_path).load()


def test_gui_settings_atomic_write_uses_sibling_temp_fsync_and_replace(
    tmp_path, monkeypatch
):
    module = _gui_module("settings")
    store = module.GuiSettingsStore(tmp_path)
    calls: dict[str, object] = {}
    real_fsync = os.fsync
    real_replace = os.replace

    def record_fsync(descriptor):
        calls["fsync"] = True
        return real_fsync(descriptor)

    def record_replace(source, destination):
        calls["source"] = Path(source)
        calls["destination"] = Path(destination)
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "fsync", record_fsync)
    monkeypatch.setattr(module.os, "replace", record_replace)

    store.save(language="en", output_roots=[])

    assert calls["fsync"] is True
    assert calls["source"].parent == store.path.parent
    assert calls["destination"] == store.path
    assert not list(store.path.parent.glob("*.tmp"))


def test_gui_settings_reject_junction_storage_escape(tmp_path, monkeypatch):
    module = _gui_module("settings")
    if not hasattr(Path, "is_junction"):
        pytest.skip("Path.is_junction is unavailable")
    settings_dir = tmp_path / ".lte-data"
    settings_dir.mkdir()
    real_is_junction = Path.is_junction

    def mark_settings_directory_as_junction(path):
        return path == settings_dir or real_is_junction(path)

    monkeypatch.setattr(Path, "is_junction", mark_settings_directory_as_junction)

    with pytest.raises(module.GuiSettingsError, match="symlink or junction"):
        module.GuiSettingsStore(tmp_path).load()


def test_gui_settings_reject_symlink_storage_escape(tmp_path):
    module = _gui_module("settings")
    outside = tmp_path / "outside"
    outside.mkdir()
    settings_dir = tmp_path / ".lte-data"
    try:
        settings_dir.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(module.GuiSettingsError, match="symlink or junction"):
        module.GuiSettingsStore(tmp_path).load()


def test_gui_settings_canonical_containment_rejects_redirected_path(
    tmp_path, monkeypatch
):
    module = _gui_module("settings")
    store = module.GuiSettingsStore(tmp_path)
    outside = tmp_path.parent / "outside-settings" / "gui-settings.json"
    real_resolve = Path.resolve

    def redirect_settings_path(path, *args, **kwargs):
        if path == store.path:
            return outside
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", redirect_settings_path)
    monkeypatch.setattr(module, "_is_link_or_junction", lambda path: False)

    with pytest.raises(module.GuiSettingsError, match="escapes repository root"):
        store.load()


def test_gui_settings_concurrent_saves_publish_one_complete_document(tmp_path):
    module = _gui_module("settings")
    store = module.GuiSettingsStore(tmp_path)
    barrier = Barrier(8)
    choices = (
        ("en", [tmp_path / "output-a"]),
        ("zh-CN", [tmp_path / "output-b"]),
    )

    def save(index):
        barrier.wait()
        language, roots = choices[index % 2]
        store.save(language=language, output_roots=roots)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(save, range(8)))

    document = json.loads(store.path.read_text(encoding="utf-8"))
    complete_documents = {
        (language, (str(roots[0].resolve()),)) for language, roots in choices
    }
    assert (document["language"], tuple(document["output_roots"])) in complete_documents
    assert store.load().language == document["language"]
    assert not list(store.path.parent.glob("*.tmp"))


def test_gui_settings_updates_merge_against_latest_document(tmp_path):
    module = _gui_module("settings")
    store = module.GuiSettingsStore(tmp_path)
    first_root = tmp_path / "output-a"
    second_root = tmp_path / "output-b"
    store.save(language="en", output_roots=[first_root])

    barrier = Barrier(2)

    def change_language():
        barrier.wait()
        store.update(language="zh-CN")

    def remember_root():
        barrier.wait()
        store.update(add_output_roots=(second_root,))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(change_language), executor.submit(remember_root)]
        for future in futures:
            future.result()

    settings = store.load()
    assert settings.language == "zh-CN"
    assert settings.output_roots == (first_root.resolve(), second_root.resolve())


async def test_gui_shell_receives_and_persists_navigation_preference(
    tmp_path, monkeypatch, user
):
    module = _gui_module("app")
    output_root = tmp_path / "outputs"
    module.GuiSettingsStore(tmp_path).save(
        language="zh-CN",
        output_roots=[output_root],
        navigation_collapsed=True,
    )
    received: list[dict[str, object]] = []
    original_render_shell = module.render_app_shell

    def record_shell(*args, **kwargs):
        received.append(kwargs)
        return original_render_shell(*args, **kwargs)

    monkeypatch.setattr(module, "render_app_shell", record_shell)
    module.create_app(catalog=SimpleNamespace(root=tmp_path.resolve()), testing=True)

    await user.open("/")

    shell = received[-1]
    assert shell["navigation_collapsed"] is True
    shell["on_navigation_toggle"](False)
    settings = module.GuiSettingsStore(tmp_path).load()
    assert settings.language == "zh-CN"
    assert settings.output_roots == (output_root.resolve(),)
    assert settings.navigation_collapsed is False


async def test_gui_shell_applies_navigation_mini_state_and_persists_rail_toggle(
    tmp_path, user
):
    module = _gui_module("app")
    module.GuiSettingsStore(tmp_path).save(
        language="en",
        output_roots=[],
        navigation_collapsed=True,
    )
    module.create_app(catalog=SimpleNamespace(root=tmp_path.resolve()), testing=True)

    await user.open("/")

    navigation = next(iter(user.find(marker="shell-navigation").elements))
    menu = next(iter(user.find(marker="shell-menu").elements))
    rail_toggle = next(iter(user.find(marker="shell-navigation-toggle").elements))
    assert navigation._props["mini"] is True
    assert str(navigation._props["mini-width"]) == "68"
    assert rail_toggle._props["aria-label"] == "Expand navigation"
    assert menu._props["aria-label"] == "Open navigation menu"

    user.find(marker="shell-navigation-toggle").click()
    await asyncio.sleep(0.05)

    assert "mini" not in navigation._props
    assert rail_toggle._props["aria-label"] == "Collapse navigation"
    assert module.GuiSettingsStore(tmp_path).load().navigation_collapsed is False

    user.find(marker="shell-menu").click()
    await asyncio.sleep(0.05)

    assert navigation.value is True
    assert module.GuiSettingsStore(tmp_path).load().navigation_collapsed is False


async def test_gui_shell_keeps_the_mobile_overlay_control_separate_from_mini_mode(
    tmp_path, user
):
    module = _gui_module("app")
    module.GuiSettingsStore(tmp_path).save(
        language="en",
        output_roots=[],
        navigation_collapsed=True,
    )
    module.create_app(catalog=SimpleNamespace(root=tmp_path.resolve()), testing=True)

    await user.open("/")

    navigation = next(iter(user.find(marker="shell-navigation").elements))
    assert navigation._props["mini"] is True
    assert str(navigation._props["width"]) == "224"
    assert str(navigation._props["mini-width"]) == "68"
    assert str(navigation._props["breakpoint"]) == "980"
    assert navigation._props["show-if-above"] is True

    user.find(marker="shell-menu").click()
    await asyncio.sleep(0.05)

    assert navigation.value is True
    assert module.GuiSettingsStore(tmp_path).load().navigation_collapsed is True


async def test_gui_shell_renders_and_switches_language_offline(
    tmp_path, monkeypatch, user
):
    module = _gui_module("app")
    catalog = SimpleNamespace(root=tmp_path.resolve())
    monkeypatch.setattr(
        module,
        "load_data_catalog",
        lambda *args, **kwargs: pytest.fail("injected catalog must avoid file loading"),
    )
    module.create_app(catalog=catalog, testing=True)

    await user.open("/")

    await user.should_see("LTE Scenario Toolkit")
    await user.should_see("Scenarios")
    await user.should_see("No active job")
    await user.should_see(marker="shell-navigation")
    await user.should_see(marker="shell-menu")
    await user.should_see(marker="shell-page-context", content="Scenarios")
    await user.should_see(marker="shell-job-indicator", content="No active job")
    await user.should_see(marker="shell-eyebrow", content="LTE / Operations")
    navigation = next(iter(user.find(marker="shell-navigation").elements))
    menu = next(iter(user.find(marker="shell-menu").elements))
    assert navigation.tag == "q-drawer"
    assert str(navigation._props["width"]) == "224"
    assert str(navigation._props["breakpoint"]) == "980"
    assert navigation._props["show-if-above"] is True
    assert menu._props["aria-label"] == "Open navigation menu"
    user.find("English").click()
    user.find("\u7b80\u4f53\u4e2d\u6587").click()
    await user.should_see("\u573a\u666f")
    await user.should_see(marker="shell-eyebrow", content="LTE / \u4f5c\u4e1a\u53f0")
    chinese_menu = next(iter(user.find(marker="shell-menu").elements))
    assert chinese_menu._props["aria-label"] == "\u6253\u5f00\u5bfc\u822a\u83dc\u5355"
    assert json.loads(
        (tmp_path / ".lte-data/gui-settings.json").read_text(encoding="utf-8")
    )["language"] == "zh-CN"


async def test_gui_shell_does_not_query_drawer_state_from_browser_on_connect(
    tmp_path, monkeypatch, user
):
    from nicegui.client import Client

    module = _gui_module("app")
    catalog = SimpleNamespace(root=tmp_path.resolve())
    monkeypatch.setattr(
        module,
        "load_data_catalog",
        lambda *args, **kwargs: pytest.fail("injected catalog must avoid file loading"),
    )
    drawer_queries: list[str] = []
    original_run_javascript = Client.run_javascript

    def record_run_javascript(self, code: str, *, timeout: float = 1.0):
        if "__IS_DRAWER_OPEN__" in code:
            drawer_queries.append(code)
        return original_run_javascript(self, code, timeout=timeout)

    monkeypatch.setattr(Client, "run_javascript", record_run_javascript)

    module.create_app(catalog=catalog, testing=True)
    await user.open("/")
    await user.should_see(marker="shell-navigation")
    await asyncio.sleep(0.05)

    assert drawer_queries == []


async def test_gui_shell_defers_refresh_timer_until_client_connect(
    tmp_path, monkeypatch, user
):
    from nicegui import ui

    module = _gui_module("app")
    catalog = SimpleNamespace(root=tmp_path.resolve())
    monkeypatch.setattr(
        module,
        "load_data_catalog",
        lambda *args, **kwargs: pytest.fail("injected catalog must avoid file loading"),
    )
    timer_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        ui,
        "timer",
        lambda *args, **kwargs: timer_calls.append((args, kwargs)),
    )

    module.create_app(catalog=catalog, testing=True)
    response = await user.http_client.get("/")

    assert response.status_code == 200
    assert timer_calls == []


async def test_gui_shell_job_indicator_tracks_coordinator_without_navigation(
    tmp_path, user
):
    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.jobs import JobCoordinator

    coordinator = JobCoordinator()
    try:
        create_app(
            catalog=SimpleNamespace(root=tmp_path.resolve()),
            coordinator=coordinator,
            testing=True,
        )
        await user.open("/generate")

        await user.should_see(
            marker="shell-page-context",
            content="Generate Scenario",
        )
        await user.should_see(
            marker="shell-job-indicator",
            content="No active job",
        )
        await user.should_not_see("Ready")
        await user.should_not_see(marker="shell-app-status")

        job = coordinator.start("selection.scan")
        await user.should_see(
            marker="shell-job-indicator",
            content="Candidate scan",
            retries=12,
        )

        assert coordinator.finish(job.job_id) is True
        await user.should_see(
            marker="shell-job-indicator",
            content="No active job",
            retries=12,
        )
        await user.should_not_see("Ready")
        await user.should_not_see(marker="shell-app-status")
    finally:
        coordinator.shutdown()


def test_nicegui_cleanup_preserves_regular_toolkit_module_imports(monkeypatch):
    import lte_scenario_toolkit
    import lte_scenario_toolkit.io as io_module

    assert lte_scenario_toolkit.io is io_module
    monkeypatch.setattr(
        "lte_scenario_toolkit.io.metadata.version",
        lambda package: f"{package}-version",
    )
    assert io_module.metadata.version("numpy") == "numpy-version"


class _Task13Catalog:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.scenarios_by_id = {
            "ready-city": {
                "scenario_id": "ready-city",
                "display_name": "Ready City",
                "boundary_dataset_id": "boundary",
                "dem_dataset_id": "dem",
                "config_path": "configs/ready.yaml",
            },
            "pending-city": {
                "scenario_id": "pending-city",
                "display_name": "Pending City",
                "boundary_dataset_id": "boundary",
                "dem_dataset_id": "pending-dem",
                "config_path": None,
            },
            "invalid-city": {
                "scenario_id": "invalid-city",
                "display_name": "Invalid City",
                "boundary_dataset_id": "missing-boundary",
                "dem_dataset_id": "dem",
                "config_path": None,
            },
            "boundary-only": {
                "scenario_id": "boundary-only",
                "display_name": "Boundary Only",
                "boundary_dataset_id": "boundary",
                "dem_dataset_id": None,
                "config_path": None,
            },
            "without-default": {
                "scenario_id": "without-default",
                "display_name": "Without Default",
                "boundary_dataset_id": "boundary",
                "dem_dataset_id": "dem",
                "config_path": None,
            },
        }
        self.datasets_by_id = {
            "boundary": {
                "dataset_id": "boundary",
                "role": "boundary",
                "entrypoint": "data/boundary.geojson",
            },
            "missing-boundary": {
                "dataset_id": "missing-boundary",
                "role": "boundary",
                "entrypoint": "data/missing.geojson",
            },
            "dem": {
                "dataset_id": "dem",
                "role": "dem",
                "entrypoint": "data/dem.tif",
            },
            "pending-dem": {
                "dataset_id": "pending-dem",
                "role": "dem",
                "entrypoint": "data/pending.tif",
            },
            "points": {
                "dataset_id": "points",
                "role": "points",
                "entrypoint": "data/points.geojson",
            },
        }

    def scenario_status(self, scenario_id):
        return {
            "ready-city": "ready",
            "pending-city": "dem-pending",
            "invalid-city": "invalid",
            "boundary-only": "boundary-ready",
            "without-default": "ready",
        }[scenario_id]

    def scenario(self, scenario_id):
        return self.scenarios_by_id[scenario_id]

    def dataset(self, dataset_id):
        return self.datasets_by_id[dataset_id]


def _task13_profile(tmp_path: Path):
    from lte_scenario_toolkit.profiles import (
        ExperimentProfile,
        FigureSettings,
        OutputSettings,
    )

    return ExperimentProfile(
        profile_id="default",
        display_name="Default",
        scenario_id="ready-city",
        points_dataset_id="points",
        random_seed=42,
        target_crs="EPSG:3857",
        rect_size=2000,
        target_count=20,
        tolerance=0,
        scan_mode="fast",
        strategy="uniform",
        scan_step=10,
        max_rects=100,
        min_spacing=2000,
        output_root=tmp_path / "results",
        outputs=OutputSettings(),
        figure=FigureSettings(),
        source_path=tmp_path / "configs/ready.yaml",
    )




def test_scenario_cards_are_immutable_and_enable_only_ready_statuses(tmp_path):
    from lte_scenario_toolkit.gui.pages.scenarios import scenario_cards

    cards = scenario_cards(_Task13Catalog(tmp_path))

    assert tuple(card.scenario_id for card in cards) == (
        "ready-city",
        "pending-city",
        "invalid-city",
        "boundary-only",
        "without-default",
    )
    assert {card.status: card.can_run for card in cards[:4]} == {
        "ready": True,
        "dem-pending": False,
        "invalid": False,
        "boundary-ready": False,
    }
    ready = cards[0]
    assert ready.boundary_entrypoint == "data/boundary.geojson"
    assert ready.dem_entrypoint == "data/dem.tif"
    assert ready.default_profile_path == "configs/ready.yaml"
    with pytest.raises(FrozenInstanceError):
        ready.can_run = False
    boundary_only = next(card for card in cards if card.scenario_id == "boundary-only")
    assert boundary_only.dem_dataset_id is None
    assert boundary_only.dem_entrypoint is None


def test_scenario_preview_requests_map_catalog_paths_inside_repository(tmp_path):
    from lte_scenario_toolkit.gui.pages.scenarios import scenario_preview_requests

    (tmp_path / "data").mkdir()
    (tmp_path / "data/boundary.geojson").write_text("{}", encoding="utf-8")
    (tmp_path / "data/dem.tif").write_bytes(b"dem")

    requests = scenario_preview_requests(_Task13Catalog(tmp_path))

    ready = requests[0]
    assert ready.scenario_id == "ready-city"
    assert ready.scenario_name == "Ready City"
    assert ready.boundary_path == (tmp_path / "data/boundary.geojson").resolve()
    assert ready.dem_path == (tmp_path / "data/dem.tif").resolve()
    assert ready.allowed_root == tmp_path.resolve()
    boundary_only = next(
        request for request in requests if request.scenario_id == "boundary-only"
    )
    assert boundary_only.dem_path is None


@pytest.mark.parametrize(
    "entrypoint",
    ("../outside.geojson", "https://example.invalid/boundary.geojson"),
)
def test_scenario_preview_requests_reject_arbitrary_catalog_paths(
    tmp_path, entrypoint
):
    from lte_scenario_toolkit.gui.pages.scenarios import scenario_preview_requests

    catalog = _Task13Catalog(tmp_path)
    catalog.datasets_by_id["boundary"]["entrypoint"] = entrypoint

    with pytest.raises(ValueError, match="repository-local|traversal|inside"):
        scenario_preview_requests(catalog)


def test_scenario_preview_requests_reject_symlink_escape(tmp_path):
    from lte_scenario_toolkit.gui.pages.scenarios import scenario_preview_requests

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "boundary.geojson").write_text("{}", encoding="utf-8")
    try:
        (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    catalog = _Task13Catalog(tmp_path)
    catalog.datasets_by_id["boundary"]["entrypoint"] = "linked/boundary.geojson"

    with pytest.raises(ValueError, match="inside"):
        scenario_preview_requests(catalog)


def test_run_validation_returns_immutable_diagnostics_without_printing(
    tmp_path, monkeypatch, capsys
):
    from lte_scenario_toolkit.data_validation import ValidationMessage, ValidationReport
    from lte_scenario_toolkit.gui.pages import scenarios

    report = ValidationReport(
        scenario_id="ready-city",
        status="ready",
        messages=[ValidationMessage("warning", "manifest.stale", "Manifest is stale")],
    )
    calls = []

    def validate(catalog, scenario_id, **kwargs):
        calls.append((catalog, scenario_id, kwargs))
        return report

    monkeypatch.setattr(scenarios, "validate_scenario_data", validate)
    catalog = _Task13Catalog(tmp_path)

    result = scenarios.run_validation(
        catalog,
        "ready-city",
        full_checksum=True,
        dataset_ids=("points",),
    )

    assert capsys.readouterr() == ("", "")
    assert result.ok is True
    assert result.full_checksum is True
    assert result.messages[0].code == "manifest.stale"
    assert calls == [
        (
            catalog,
            "ready-city",
            {"full_checksum": True, "dataset_ids": ("points",)},
        )
    ]
    with pytest.raises(FrozenInstanceError):
        result.status = "invalid"


def test_full_checksum_uses_shared_coordinator_and_framework_free_worker(
    tmp_path, monkeypatch
):
    from lte_scenario_toolkit.gui.pages import scenarios

    assert scenarios.get_job_coordinator() is scenarios.get_job_coordinator()
    expected = object()
    calls = []

    class RecordingCoordinator:
        def submit(self, kind, worker):
            calls.append((kind, worker))
            return SimpleNamespace(job_id="job-1", future=None)

    monkeypatch.setattr(scenarios, "run_validation", lambda *args, **kwargs: expected)
    coordinator = RecordingCoordinator()

    handle = scenarios.submit_full_checksum(
        _Task13Catalog(tmp_path),
        "ready-city",
        coordinator=coordinator,
    )

    assert handle.job_id == "job-1"
    kind, worker = calls[0]
    emitted = []
    assert kind == "validation.full_checksum"
    assert worker(SimpleNamespace(), emitted.append) is expected
    assert emitted == [expected]


@pytest.mark.parametrize("fails", [False, True])
def test_full_checksum_releases_slot_without_page_polling(tmp_path, monkeypatch, fails):
    from lte_scenario_toolkit.gui.pages import scenarios
    from lte_scenario_toolkit.jobs import JobCoordinator

    coordinator = JobCoordinator()
    callback_finished = Event()

    def validate(*args, **kwargs):
        if fails:
            raise RuntimeError("checksum failed")
        return object()

    monkeypatch.setattr(scenarios, "run_validation", validate)
    try:
        job = scenarios.submit_full_checksum(
            _Task13Catalog(tmp_path),
            "ready-city",
            coordinator=coordinator,
        )
        assert job.future is not None
        job.future.add_done_callback(lambda _future: callback_finished.set())
        assert callback_finished.wait(2)
        assert coordinator.snapshot().active is False
    finally:
        coordinator.shutdown()


def test_shared_coordinator_can_be_recreated_after_app_shutdown():
    from lte_scenario_toolkit.gui.pages import scenarios

    first = scenarios.get_job_coordinator()
    scenarios.shutdown_job_coordinator()
    second = scenarios.get_job_coordinator()

    assert second is not first
    scenarios.shutdown_job_coordinator()


def test_configure_model_prefers_catalog_default_and_builds_explicit_draft(tmp_path):
    from lte_scenario_toolkit.gui.pages.configure import configure_model

    profile = _task13_profile(tmp_path)

    class FakeStore:
        def discover(self, scenario_id):
            return [profile] if scenario_id == "ready-city" else []

    catalog = _Task13Catalog(tmp_path)
    saved = configure_model(catalog, FakeStore(), "ready-city")
    draft = configure_model(catalog, FakeStore(), "without-default")

    assert saved.profile is profile
    assert saved.is_persisted is True
    assert saved.dirty is False
    assert draft.is_persisted is False
    assert draft.profile.rect_size == 3000
    assert draft.profile.target_count == 30
    assert draft.profile.min_spacing == 3000
    assert draft.profile.points_dataset_id == "points"
    assert draft.profile.output_root == (tmp_path / "results").resolve()
    with pytest.raises(FrozenInstanceError):
        draft.dirty = True


def test_configure_model_selects_catalog_default_by_resolved_source_path(tmp_path):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages.configure import configure_model

    first = replace(
        _task13_profile(tmp_path),
        profile_id="first",
        display_name="First",
        source_path=tmp_path / "configs/first.yaml",
    )
    expected = replace(
        _task13_profile(tmp_path),
        profile_id="ready",
        display_name="Ready",
        source_path=tmp_path / "configs/ready.yaml",
    )

    class FakeStore:
        def discover(self, scenario_id):
            return [first, expected]

    model = configure_model(_Task13Catalog(tmp_path), FakeStore(), "ready-city")

    assert model.profiles == (first, expected)
    assert model.profile is expected
    assert model.is_default is True
    assert model.profile_choices == (("first", "First"), ("ready", "Ready"))
    assert model.output_root_default == (tmp_path / "results").resolve()

    selected = configure_model(
        _Task13Catalog(tmp_path),
        FakeStore(),
        "ready-city",
        selected_profile_id="first",
    )
    assert selected.profile is first
    assert selected.is_default is False


def test_configure_model_does_not_silently_replace_missing_catalog_default(tmp_path):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages.configure import configure_model

    nondefault = replace(
        _task13_profile(tmp_path),
        profile_id="other",
        source_path=tmp_path / "configs/other.yaml",
    )

    class FakeStore:
        def discover(self, scenario_id):
            return [nondefault]

    model = configure_model(_Task13Catalog(tmp_path), FakeStore(), "ready-city")

    assert model.profile is not nondefault
    assert model.selection_error is not None
    assert model.can_start is False


def test_configure_draft_rejects_missing_or_ambiguous_points_datasets(tmp_path):
    from lte_scenario_toolkit.gui.pages.configure import configure_model

    class EmptyStore:
        def discover(self, scenario_id):
            return []

    missing = _Task13Catalog(tmp_path)
    missing.datasets_by_id.pop("points")
    with pytest.raises(ValueError, match="exactly one registered points dataset"):
        configure_model(missing, EmptyStore(), "without-default")

    ambiguous = _Task13Catalog(tmp_path)
    ambiguous.datasets_by_id["points-2"] = {
        "dataset_id": "points-2",
        "role": "points",
        "entrypoint": "data/points-2.geojson",
    }
    with pytest.raises(ValueError, match="found 2"):
        configure_model(ambiguous, EmptyStore(), "without-default")












def test_form_integer_fields_reject_fractional_values_without_truncating(tmp_path):
    from lte_scenario_toolkit.gui.pages.configure import profile_with_form_values

    profile = _task13_profile(tmp_path)

    with pytest.raises(ValueError, match="whole number"):
        profile_with_form_values(profile, {"rect_size": 2000.5})
    with pytest.raises(ValueError, match="whole number"):
        profile_with_form_values(profile, {"target_count": "20.5"})

    updated = profile_with_form_values(
        profile,
        {"rect_size": 2500.0, "target_count": 25, "output_root": tmp_path / "out"},
    )
    assert updated.rect_size == 2500
    assert type(updated.rect_size) is int
    assert profile.rect_size == 2000


def test_profile_mutations_require_explicit_confirmation_and_delegate_to_store(tmp_path):
    from lte_scenario_toolkit.gui.pages.configure import (
        ConfirmationRequiredError,
        copy_profile,
        delete_profile,
        rename_profile,
        save_profile,
        set_default_profile,
    )

    profile = _task13_profile(tmp_path)
    calls = []

    class RecordingStore:
        def save(self, *args, **kwargs):
            calls.append(("save", args, kwargs))
            return tmp_path / "saved.yaml"

        def copy(self, *args, **kwargs):
            calls.append(("copy", args, kwargs))
            return tmp_path / "copy.yaml"

        def rename(self, *args, **kwargs):
            calls.append(("rename", args, kwargs))
            return tmp_path / "renamed.yaml"

        def set_default(self, *args, **kwargs):
            calls.append(("set_default", args, kwargs))
            return tmp_path / "default.yaml"

        def delete(self, *args, **kwargs):
            calls.append(("delete", args, kwargs))

    store = RecordingStore()
    assert save_profile(store, profile) == tmp_path / "saved.yaml"
    with pytest.raises(ConfirmationRequiredError):
        save_profile(store, profile, overwrite=True, confirmed=False)
    with pytest.raises(ConfirmationRequiredError):
        save_profile(store, profile, set_default=True, confirmed=False)
    with pytest.raises(ConfirmationRequiredError):
        copy_profile(store, profile.source_path, "copy", "Copy", confirmed=1)
    with pytest.raises(ConfirmationRequiredError):
        rename_profile(store, profile.source_path, "renamed", "Renamed", confirmed=False)
    with pytest.raises(ConfirmationRequiredError):
        set_default_profile(store, "ready-city", profile.source_path, confirmed=False)
    with pytest.raises(ConfirmationRequiredError):
        delete_profile(store, profile.source_path, confirmed=False)

    copy_profile(store, profile.source_path, "copy", "Copy", confirmed=True)
    rename_profile(store, profile.source_path, "renamed", "Renamed", confirmed=True)
    set_default_profile(store, "ready-city", profile.source_path, confirmed=True)
    delete_profile(
        store,
        profile.source_path,
        replacement_default=tmp_path / "copy.yaml",
        confirmed=True,
    )

    assert [call[0] for call in calls] == [
        "save",
        "copy",
        "rename",
        "set_default",
        "delete",
    ]
    assert calls[0][2] == {"overwrite": False, "set_default": False}
    assert calls[1][1] == (profile.source_path, "copy", "Copy")
    assert calls[2][1] == (profile.source_path, "renamed", "Renamed")
    assert calls[3][1] == ("ready-city", profile.source_path)
    assert calls[4][1] == (profile.source_path,)
    assert calls[-1][2] == {"replacement_default": tmp_path / "copy.yaml"}


def test_save_validates_before_store_mutation(tmp_path):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages.configure import save_profile

    calls = []

    class RecordingStore:
        def save(self, *args, **kwargs):
            calls.append((args, kwargs))

    invalid = replace(_task13_profile(tmp_path), rect_size=0)

    with pytest.raises(ValueError, match="greater than zero"):
        save_profile(RecordingStore(), invalid)

    assert calls == []


def test_gui_runtime_refreshes_catalog_and_selection_service_after_profile_change(
    tmp_path
):
    from lte_scenario_toolkit.gui.app import GuiRuntime

    old_catalog = _Task13Catalog(tmp_path)
    new_catalog = _Task13Catalog(tmp_path)
    new_catalog.scenarios_by_id["ready-city"] = {
        **new_catalog.scenarios_by_id["ready-city"],
        "config_path": "configs/replaced.yaml",
    }
    loaded = []
    services = []

    def catalog_loader(catalog):
        loaded.append(catalog)
        return new_catalog

    def selection_factory(catalog):
        service = SimpleNamespace(catalog=catalog)
        services.append(service)
        return service

    runtime = GuiRuntime(
        old_catalog,
        profile_store=object(),
        catalog_loader=catalog_loader,
        selection_service_factory=selection_factory,
    )

    runtime.refresh_after_profile_mutation()

    assert loaded == [old_catalog]
    assert runtime.catalog is new_catalog
    assert runtime.selection_service.catalog is new_catalog
    assert len(services) == 2


def test_gui_runtime_refresh_failure_preserves_matching_old_services(tmp_path):
    from lte_scenario_toolkit.gui.app import GuiRuntime

    old_catalog = _Task13Catalog(tmp_path)
    new_catalog = _Task13Catalog(tmp_path)
    old_service = SimpleNamespace(catalog=old_catalog)
    calls = []

    def selection_factory(catalog):
        calls.append(catalog)
        if catalog is new_catalog:
            raise RuntimeError("service rebuild failed")
        return old_service

    runtime = GuiRuntime(
        old_catalog,
        profile_store=object(),
        catalog_loader=lambda _catalog: new_catalog,
        selection_service_factory=selection_factory,
    )

    with pytest.raises(RuntimeError, match="service rebuild failed"):
        runtime.refresh_after_profile_mutation()

    assert runtime.catalog is old_catalog
    assert runtime.selection_service is old_service
    assert calls == [old_catalog, new_catalog]


@pytest.mark.parametrize("operation", ["copy", "delete"])
def test_profile_actions_report_committed_mutation_when_runtime_refresh_fails(
    tmp_path, operation
):
    from lte_scenario_toolkit.gui.pages.configure import (
        ProfileActions,
        ProfileRefreshError,
    )

    calls = []
    copied_path = tmp_path / "configs/copy.yaml"

    class RecordingStore:
        def copy(self, *args):
            calls.append(("copy", args))
            return copied_path

        def delete(self, *args, **kwargs):
            calls.append(("delete", args, kwargs))

    def fail_refresh():
        raise RuntimeError("reload failed")

    actions = ProfileActions(RecordingStore(), fail_refresh)
    with pytest.raises(ProfileRefreshError) as exc_info:
        if operation == "copy":
            actions.copy("source.yaml", "copy", "Copy", confirmed=True)
        else:
            actions.delete("source.yaml", confirmed=True)

    assert exc_info.value.committed is True
    assert exc_info.value.mutation_result == (
        copied_path if operation == "copy" else None
    )
    assert len(calls) == 1


def test_start_scan_preflight_freezes_replace_snapshot_and_maps_domain_errors(tmp_path):
    from lte_scenario_toolkit.gui.pages.configure import start_scan_preflight
    from lte_scenario_toolkit.selection_service import SelectionPreflightError

    profile = _task13_profile(tmp_path)
    calls = []

    class SuccessfulService:
        def preflight(self, snapshot, output_root):
            calls.append((snapshot, output_root))
            return "prepared"

        def scan(self, *args, **kwargs):
            pytest.fail("Start Scan setup must not start the scan")

    result = start_scan_preflight(SuccessfulService(), profile)

    assert result.ok is True
    assert result.preflight == "prepared"
    assert result.snapshot == profile
    assert result.snapshot is not profile
    assert calls == [(result.snapshot, profile.output_root)]

    class FailingService:
        def preflight(self, snapshot, output_root):
            raise SelectionPreflightError("outputs.root", "Output root is invalid")

    failed = start_scan_preflight(FailingService(), profile)
    assert failed.ok is False
    assert failed.error_code == "outputs.root"
    assert failed.field_errors == (("output_root", "Output root is invalid"),)

    class CrsFailure:
        def preflight(self, snapshot, output_root):
            raise SelectionPreflightError("profile.target_crs", "CRS is invalid")

    failed_crs = start_scan_preflight(CrsFailure(), profile)
    assert failed_crs.field_errors == (("target_crs", "CRS is invalid"),)

    class GlobalFailure:
        def preflight(self, snapshot, output_root):
            raise SelectionPreflightError("scenario.validation_failed", "Data drifted")

    failed_global = start_scan_preflight(GlobalFailure(), profile)
    assert failed_global.field_errors == (("__all__", "Data drifted"),)


async def test_scenario_routes_share_content_and_disable_nonready_actions(
    user, tmp_path
):
    from lte_scenario_toolkit.gui.app import create_app

    class EmptyStore:
        def discover(self, scenario_id):
            return []

    catalog = _Task13Catalog(tmp_path)
    create_app(catalog=catalog, profile_store=EmptyStore(), testing=True)

    scenario_ids = tuple(catalog.scenarios_by_id)
    expected_ready = sum(
        catalog.scenario_status(scenario_id) == "ready"
        for scenario_id in scenario_ids
    )
    expected_preparation = len(scenario_ids) - expected_ready

    for route in ("/", "/scenarios"):
        await user.open(route)
        await user.should_see(marker="scenarios-summary")
        assert next(
            iter(user.find(marker="scenarios-count-total").elements)
        ).text == str(len(scenario_ids))
        assert next(
            iter(user.find(marker="scenarios-count-ready").elements)
        ).text == str(expected_ready)
        assert next(
            iter(user.find(marker="scenarios-count-preparation").elements)
        ).text == str(expected_preparation)
        for scenario_id in scenario_ids:
            await user.should_see(marker=f"scenario-card-{scenario_id}")
            await user.should_see(marker=f"scenario-status-{scenario_id}")
            await user.should_see(marker=f"scenario-technical-{scenario_id}")
            await user.should_see(marker=f"scenario-setup-{scenario_id}")
            assert len(user.find(marker=f"scenario-overflow-{scenario_id}").elements) == 1
            workflow_marker = (
                f"scenario-configure-{scenario_id}"
                if catalog.scenario_status(scenario_id) == "ready"
                else f"scenario-guidance-{scenario_id}"
            )
            assert len(user.find(marker=workflow_marker).elements) == 1
            await user.should_not_see(marker=f"scenario-run-{scenario_id}")

        assert len(user.find(marker="scenario-configure-ready-city").elements) == 1
        assert next(
            iter(user.find(marker="scenario-configure-ready-city").elements)
        ).enabled
        await user.should_not_see(marker="scenario-guidance-ready-city")
        await user.should_not_see(marker="scenario-configure-pending-city")
        assert len(user.find(marker="scenario-guidance-pending-city").elements) == 1
        await user.should_see("Terrain data required")
        assert (
            next(
                iter(user.find(marker="scenario-status-pending-city").elements)
            ).text
            == "Terrain data required"
        )

    user.find(marker="scenario-guidance-pending-city").click()
    await user.should_see(
        "Replace <path> with the directory containing the downloaded DEM tiles"
    )
    await user.should_see("Register a scenario with the CLI")
    assert not (tmp_path / ".lte-data").exists()

    await user.open("/configure")
    await user.should_see("Choose a scenario")
    assert (
        next(iter(user.find(marker="picker-status-pending-city").elements)).text
        == "Terrain data required"
    )
    await user.should_not_see(marker="picker-configure-pending-city")
    assert len(user.find(marker="picker-guidance-pending-city").elements) == 1


async def test_scenario_catalog_maps_injected_preview_batch_with_local_fallback(
    user, tmp_path, monkeypatch
):
    from nicegui import run

    from lte_scenario_toolkit.gui.app import create_app

    async def cpu_bound(builder, *args):
        return await asyncio.to_thread(builder, *args)

    monkeypatch.setattr(run, "cpu_bound", cpu_bound)

    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=object(),
        scenario_preview_builder=_scenario_preview_fixture_builder,
        testing=True,
    )

    await user.open("/scenarios")

    await user.should_see(
        marker="scenario-preview-kind-ready-city", content="Terrain preview", retries=30
    )
    await user.should_see(
        marker="scenario-preview-kind-pending-city", content="Preview unavailable"
    )
    await user.should_see(
        marker="scenario-preview-kind-invalid-city", content="Boundary preview"
    )
    ready_image = next(
        iter(user.find(marker="scenario-preview-image-ready-city").elements)
    )
    assert ready_image._props["alt"] == "Ready City Terrain preview"
    await user.should_not_see(marker="scenario-preview-image-pending-city")


async def test_collapsed_shell_keeps_scenario_catalog_route_renderable(
    user, tmp_path, monkeypatch
):
    from nicegui import run

    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.settings import GuiSettingsStore

    async def cpu_bound(builder, *args):
        return await asyncio.to_thread(builder, *args)

    monkeypatch.setattr(run, "cpu_bound", cpu_bound)
    GuiSettingsStore(tmp_path).save(
        language="en",
        output_roots=(),
        navigation_collapsed=True,
    )
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=object(),
        scenario_preview_builder=_scenario_preview_fixture_builder,
        testing=True,
    )

    await user.open("/")

    navigation = next(iter(user.find(marker="shell-navigation").elements))
    assert navigation._props["mini"] is True
    await user.should_see(marker="scenarios-grid")
    await user.should_see(marker="scenario-card-ready-city")


def test_scenario_catalog_css_declares_three_two_one_column_layout():
    css = (
        ROOT / "src/lte_scenario_toolkit/gui/assets/app.css"
    ).read_text(encoding="utf-8")

    assert ".lte-scenario-grid" in css
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in css
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in css
    assert ".lte-card-grid { grid-template-columns: minmax(0, 1fr); }" in css
    assert "aspect-ratio: 19 / 9;" in css
    reduced_motion = css.split("@media (prefers-reduced-motion: reduce)", 1)[1]
    assert ".lte-scenario-cover__skeleton" in reduced_motion
    assert "animation: none !important;" in reduced_motion
    assert "transition: none !important;" in reduced_motion






async def test_configure_route_renders_complete_form_and_blocks_nonready_direct_url(
    user, tmp_path
):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.app import create_app

    pending_profile = replace(
        _task13_profile(tmp_path),
        scenario_id="pending-city",
        source_path=tmp_path / "configs/pending.yaml",
    )

    class EmptyStore:
        def discover(self, scenario_id):
            return [pending_profile] if scenario_id == "pending-city" else []

    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=EmptyStore(),
        testing=True,
    )

    await user.open("/configure/without-default")
    for marker in (
        "configure-profile-header",
        "configure-profile-state",
        "profile-management-actions",
        "configure-output",
        "configure-action-bar",
    ):
        await user.should_see(marker=marker)
    for text in (
        "Basic parameters",
        "Advanced",
        "Output root",
        "Browse",
        "Copy",
        "Rename",
        "Set Default",
        "Delete",
        "Save",
        "Discard",
        "Start Scan",
    ):
        await user.should_see(text)
    assert any(
        element.enabled
        for element in user.find(marker="profile-start-scan").elements
    )
    for marker in (
        "profile-copy",
        "profile-rename",
        "profile-set-default",
        "profile-delete",
    ):
        assert all(not element.enabled for element in user.find(marker=marker).elements)

    await user.open("/configure/pending-city")
    await user.should_see("Terrain data required")
    assert "dem-pending" not in next(
        iter(user.find(marker="configure-readiness").elements)
    ).text
    assert all(
        not element.enabled
        for element in user.find(marker="profile-start-scan").elements
    )
    for marker in (
        "profile-copy",
        "profile-rename",
        "profile-set-default",
        "profile-delete",
        "profile-save",
    ):
        assert all(not element.enabled for element in user.find(marker=marker).elements)


async def test_configure_workbench_groups_profile_actions_and_run_review(user, tmp_path):
    from lte_scenario_toolkit.gui.app import create_app

    class EmptyStore:
        def discover(self, scenario_id):
            return []

    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=EmptyStore(),
        testing=True,
    )
    await user.open("/configure/without-default")

    for marker in (
        "configure-workflow-stepper",
        "configure-step-data",
        "configure-step-profile",
        "configure-step-review",
        "configure-workspace",
        "configure-run-summary",
        "profile-overflow",
        "profile-management-actions",
        "configure-action-dock",
    ):
        await user.should_see(marker=marker)

    overflow = next(iter(user.find(marker="profile-overflow").elements))
    assert overflow.props["aria-label"] == "Profile actions"
    user.find(marker="profile-overflow").click()
    for marker in (
        "profile-copy",
        "profile-rename",
        "profile-set-default",
        "profile-delete",
        "profile-validate",
    ):
        await user.should_see(marker=marker)

    dock = next(iter(user.find(marker="configure-action-dock").elements))
    assert dock.tag == "footer"
    assert "lte-configure-action-dock" in dock._classes
    for marker, role in (
        ("profile-discard", "tertiary"),
        ("profile-save", "secondary"),
        ("profile-start-scan", "primary"),
    ):
        action = next(iter(user.find(marker=marker).elements))
        expected_prop = {"tertiary": "flat", "secondary": "outline", "primary": "unelevated"}[role]
        assert action.props[expected_prop] is True


@pytest.mark.parametrize(
    ("language", "workflow_label", "data_label", "review_label", "pending_label"),
    (
        (
            "en",
            "Configuration workflow",
            "Data: complete",
            "Review: current step",
            "Review: pending",
        ),
        (
            "zh-CN",
            "\u914d\u7f6e\u5de5\u4f5c\u6d41\u7a0b",
            "\u6570\u636e\uff1a\u5df2\u5b8c\u6210",
            "\u590d\u6838\uff1a\u5f53\u524d\u6b65\u9aa4",
            "\u590d\u6838\uff1a\u7b49\u5f85\u4e2d",
        ),
    ),
)
async def test_configure_stepper_exposes_localized_current_state(
    user,
    tmp_path,
    language,
    workflow_label,
    data_label,
    review_label,
    pending_label,
):
    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.settings import GuiSettingsStore

    class EmptyStore:
        def discover(self, scenario_id):
            return []

    GuiSettingsStore(tmp_path).save(language=language, output_roots=())
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=EmptyStore(),
        testing=True,
    )
    await user.open("/configure/without-default")

    stepper = next(iter(user.find(marker="configure-workflow-stepper").elements))
    data = next(iter(user.find(marker="configure-step-data").elements))
    review = next(iter(user.find(marker="configure-step-review").elements))
    assert stepper.props["role"] == "list"
    assert stepper.props["aria-label"] == workflow_label
    assert data.props["role"] == "listitem"
    assert data.props["aria-label"] == data_label
    assert review.props["aria-current"] == "step"
    assert review.props["aria-label"] == review_label

    user.find(marker="profile-display-name").clear().type("Changed")
    profile = next(iter(user.find(marker="configure-step-profile").elements))
    assert profile.props["aria-current"] == "step"
    assert "aria-current" not in review.props
    assert review.props["aria-label"] == pending_label


async def test_configure_stepper_has_one_current_step_after_validation_and_blocking(
    user, tmp_path
):
    from dataclasses import replace
    from types import SimpleNamespace

    from lte_scenario_toolkit.gui.app import create_app

    class EmptyStore:
        def discover(self, scenario_id):
            return []

    class Service:
        def preflight(self, profile, output_root):
            return SimpleNamespace(profile=profile, output_root=output_root)

    def current_step_markers():
        return tuple(
            marker
            for marker in (
                "configure-step-data",
                "configure-step-profile",
                "configure-step-review",
            )
            if next(iter(user.find(marker=marker).elements)).props.get("aria-current")
            == "step"
        )

    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=EmptyStore(),
        selection_service_factory=lambda _catalog: Service(),
        testing=True,
    )
    await user.open("/configure/without-default")
    assert current_step_markers() == ("configure-step-review",)

    user.find(marker="profile-display-name").clear().type("Changed")
    assert current_step_markers() == ("configure-step-profile",)

    user.find(marker="profile-overflow").click()
    user.find(marker="profile-validate").click()
    await user.should_see("Preflight passed")
    assert current_step_markers() == ("configure-step-profile",)
    review = next(iter(user.find(marker="configure-step-review").elements))
    assert review.props["aria-label"] == "Review: pending"

    blocked_catalog = _Task13Catalog(tmp_path)
    blocked_profile = replace(
        _task13_profile(tmp_path),
        scenario_id="pending-city",
        source_path=tmp_path / "configs/pending.yaml",
    )

    class PendingStore:
        def discover(self, scenario_id):
            return [blocked_profile] if scenario_id == "pending-city" else []

    create_app(
        catalog=blocked_catalog,
        profile_store=PendingStore(),
        testing=True,
    )
    await user.open("/configure/pending-city")
    assert current_step_markers() == ("configure-step-data",)
    data = next(iter(user.find(marker="configure-step-data").elements))
    assert data.props["aria-label"] == "Data: needs attention"


async def test_configure_overflow_validation_does_not_open_candidate_session(user, tmp_path):
    from types import SimpleNamespace

    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.pages.candidates import CandidateSessionRegistry

    class EmptyStore:
        def discover(self, scenario_id):
            return []

    class Service:
        def preflight(self, profile, output_root):
            return SimpleNamespace(profile=profile, output_root=output_root)

    class RecordingRegistry(CandidateSessionRegistry):
        def __init__(self):
            super().__init__()
            self.create_calls = []

        def create(self, *args, **kwargs):
            self.create_calls.append((args, kwargs))
            return super().create(*args, **kwargs)

    registry = RecordingRegistry()
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=EmptyStore(),
        selection_service_factory=lambda _catalog: Service(),
        candidate_registry=registry,
        testing=True,
    )
    await user.open("/configure/without-default")

    user.find(marker="profile-overflow").click()
    user.find(marker="profile-validate").click()

    await user.should_see("Preflight passed")
    assert registry.create_calls == []
    assert not any("/candidates/" in route for route in user.back_history)


async def test_configure_workbench_smoke_renders_with_collapsed_shell(user, tmp_path):
    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.settings import GuiSettingsStore

    class EmptyStore:
        def discover(self, scenario_id):
            return []

    GuiSettingsStore(tmp_path).save(
        language="en",
        output_roots=(),
        navigation_collapsed=True,
    )
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=EmptyStore(),
        testing=True,
    )

    await user.open("/configure/without-default")

    navigation = next(iter(user.find(marker="shell-navigation").elements))
    assert navigation._props["mini"] is True
    await user.should_see(marker="configure-workflow-stepper")
    await user.should_see(marker="configure-action-dock")


def test_configure_workbench_has_explicit_390px_overflow_safeguards():
    css = (
        ROOT / "src/lte_scenario_toolkit/gui/assets/app.css"
    ).read_text(encoding="utf-8")

    narrow = css[css.index("@media (max-width: 390px)") :]
    for selector in (
        ".lte-configure-workspace",
        ".lte-configure-run-summary",
        ".lte-configure-stepper",
        ".lte-configure-action-dock",
    ):
        assert selector in narrow
    assert "grid-template-columns: minmax(0, 1fr);" in narrow
    assert "flex-direction: column;" in narrow
    assert "min-width: 44px;" in narrow
    assert "overflow-x: clip;" in narrow


def test_configure_workbench_stacks_and_wraps_before_phone_width():
    css = (
        ROOT / "src/lte_scenario_toolkit/gui/assets/app.css"
    ).read_text(encoding="utf-8")

    tablet = css[
        css.index("@media (max-width: 760px)") : css.index("@media (max-width: 390px)")
    ]
    for selector in (
        ".lte-configure-workspace",
        ".lte-configure-run-summary",
        ".lte-configure-action-dock .lte-action-bar",
        ".lte-configure-action-dock .q-btn",
    ):
        assert selector in tablet
    assert "grid-template-columns: minmax(0, 1fr);" in tablet
    assert "position: static;" in tablet
    assert "flex-wrap: wrap;" in tablet
    assert "min-width: 0;" in tablet
    assert "min-height: 44px;" in tablet


async def test_scenario_page_new_copy_is_localized_after_language_switch(user, tmp_path):
    from lte_scenario_toolkit.gui.app import create_app

    class EmptyStore:
        def discover(self, scenario_id):
            return []

    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=EmptyStore(),
        testing=True,
    )
    await user.open("/scenarios")

    user.find("English").click()
    user.find("\u7b80\u4f53\u4e2d\u6587").click()

    await user.should_see("\u573a\u666f\u76ee\u5f55")
    await user.should_see("\u4f7f\u7528 CLI \u6ce8\u518c\u573a\u666f")


async def test_full_checksum_button_blocks_duplicate_timer_generation(
    user, tmp_path, monkeypatch
):
    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.pages import scenarios
    from lte_scenario_toolkit.jobs import JobCoordinator

    release = Event()
    calls = []
    coordinator = JobCoordinator()

    def validate(catalog, scenario_id, **kwargs):
        calls.append((scenario_id, kwargs))
        assert release.wait(2)
        return scenarios.ValidationResult(
            scenario_id=scenario_id,
            status="ready",
            ok=True,
            messages=(),
            full_checksum=True,
        )

    monkeypatch.setattr(scenarios, "run_validation", validate)
    try:
        create_app(
            catalog=_Task13Catalog(tmp_path),
            profile_store=object(),
            coordinator=coordinator,
            testing=True,
        )
        await user.open("/scenarios")

        checksum = user.find(marker="scenario-checksum-ready-city")
        checksum.click()
        assert all(not element.enabled for element in checksum.elements)
        checksum.click()
        await asyncio.sleep(0.05)
        assert len(calls) == 1

        release.set()
        await user.should_see("Validation passed", retries=10)
        assert all(element.enabled for element in checksum.elements)
    finally:
        release.set()
        coordinator.shutdown()


async def test_full_checksum_completion_does_not_replace_new_inspector_selection(
    user, tmp_path, monkeypatch
):
    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.pages import scenarios
    from lte_scenario_toolkit.jobs import JobCoordinator

    started = Event()
    release = Event()
    coordinator = JobCoordinator()

    def validate(catalog, scenario_id, **kwargs):
        started.set()
        assert release.wait(2)
        return scenarios.ValidationResult(
            scenario_id=scenario_id,
            status="ready",
            ok=True,
            messages=(),
            full_checksum=True,
        )

    monkeypatch.setattr(scenarios, "run_validation", validate)
    try:
        create_app(
            catalog=_Task13Catalog(tmp_path),
            profile_store=object(),
            coordinator=coordinator,
            testing=True,
        )
        await user.open("/scenarios")

        checksum = user.find(marker="scenario-checksum-ready-city")
        checksum.click()
        assert await asyncio.to_thread(started.wait, 1)
        user.find(marker="scenario-technical-pending-city").click()
        await user.should_see(
            marker="scenario-inspector-selected", content="Pending City"
        )

        release.set()
        for _ in range(20):
            if all(element.enabled for element in checksum.elements):
                break
            await asyncio.sleep(0.05)

        assert all(element.enabled for element in checksum.elements)
        await user.should_see(
            marker="scenario-inspector-selected", content="Pending City"
        )
    finally:
        release.set()
        coordinator.shutdown()


async def test_full_checksum_completion_does_not_reopen_closed_inspector(
    user, tmp_path, monkeypatch
):
    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.pages import scenarios
    from lte_scenario_toolkit.jobs import JobCoordinator

    started = Event()
    release = Event()
    coordinator = JobCoordinator()

    def validate(catalog, scenario_id, **kwargs):
        started.set()
        assert release.wait(2)
        return scenarios.ValidationResult(
            scenario_id=scenario_id,
            status="ready",
            ok=True,
            messages=(),
            full_checksum=True,
        )

    monkeypatch.setattr(scenarios, "run_validation", validate)
    try:
        create_app(
            catalog=_Task13Catalog(tmp_path),
            profile_store=object(),
            coordinator=coordinator,
            testing=True,
        )
        await user.open("/scenarios")

        checksum = user.find(marker="scenario-checksum-ready-city")
        checksum.click()
        assert await asyncio.to_thread(started.wait, 1)
        user.find(marker="scenario-inspector-close").click()
        drawer = next(iter(user.find(marker="scenario-inspector").elements))
        assert drawer.value is False

        release.set()
        for _ in range(20):
            if all(element.enabled for element in checksum.elements):
                break
            await asyncio.sleep(0.05)

        assert all(element.enabled for element in checksum.elements)
        assert drawer.value is False
    finally:
        release.set()
        coordinator.shutdown()


async def test_profile_copy_ui_requires_confirm_before_store_call(user, tmp_path):
    from lte_scenario_toolkit.gui.app import create_app

    profile = _task13_profile(tmp_path)
    calls = []

    class RecordingStore:
        def discover(self, scenario_id):
            return [profile]

        def copy(self, *args):
            calls.append(args)
            return tmp_path / "configs/copy.yaml"

    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=RecordingStore(),
        testing=True,
    )
    await user.open("/configure/ready-city")

    user.find(marker="profile-copy").click()
    assert calls == []
    await user.should_see(marker="confirmation-copy-consequence")
    await user.should_see(
        "Copying Default (default) creates Default Copy (default-copy); the original stays unchanged."
    )
    user.find(marker="profile-copy-cancel").click()
    assert calls == []

    user.find(marker="profile-copy").click()
    user.find(marker="profile-copy-confirm").click()
    assert calls == [
        (profile.source_path, "default-copy", "Default Copy")
    ]


async def test_profile_confirmation_dialogs_name_targets_and_do_not_mutate_early(
    user, tmp_path
):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.app import create_app

    default = _task13_profile(tmp_path)
    alternate = replace(
        default,
        profile_id="alternate",
        display_name="Alternate",
        source_path=tmp_path / "configs/alternate.yaml",
    )
    calls: list[tuple[object, ...]] = []

    class RecordingStore:
        def discover(self, scenario_id):
            return [default, alternate]

        def save(self, *args, **kwargs):
            calls.append(("save", args, kwargs))

        def rename(self, *args):
            calls.append(("rename", *args))

        def set_default(self, *args):
            calls.append(("default", *args))

        def delete(self, *args, **kwargs):
            calls.append(("delete", args, kwargs))

    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=RecordingStore(),
        testing=True,
    )
    await user.open("/configure/ready-city")

    user.find(marker="profile-save").click()
    await user.should_see(marker="confirmation-overwrite-consequence")
    await user.should_see(
        "Saving replaces the stored values for Default (default)."
    )
    assert calls == []
    user.find(marker="confirmation-overwrite-cancel").click()

    user.find(marker="profile-rename").click()
    user.find(marker="profile-rename-id").clear().type("renamed")
    user.find(marker="profile-rename-name").clear().type("Renamed")
    await user.should_see(marker="confirmation-rename-consequence")
    await user.should_see(
        "Default (default) moves to Renamed (renamed); its default reference follows."
    )
    assert calls == []
    user.find(marker="profile-rename-cancel").click()

    user.find(marker="profile-delete").click()
    await user.should_see(marker="confirmation-delete-consequence")
    await user.should_see(
        "Choose a replacement for Ready City; it becomes the default before Default (default) is permanently deleted."
    )
    user.find(marker="profile-delete-replacement").click()
    user.find("Alternate").click()
    await user.should_see(
        "Alternate becomes the default for Ready City before Default (default) is permanently deleted."
    )
    assert calls == []
    user.find(marker="profile-delete-cancel").click()

    user.find(marker="profile-display-name").clear().type("Changed")
    user.find(marker="profile-select").click()
    user.find("Alternate").click()
    await user.should_see(marker="confirmation-switch-consequence")
    await user.should_see(
        "Switching from Default to Alternate discards the unsaved changes in Default."
    )
    assert calls == []
    user.find(marker="profile-switch-cancel").click()

    await user.open("/configure/ready-city?profile=alternate")
    user.find(marker="profile-set-default").click()
    await user.should_see(marker="confirmation-default-consequence")
    await user.should_see(
        "Future configuration for Ready City opens Alternate (alternate); no profiles are deleted."
    )
    assert calls == []
    user.find(marker="confirmation-default-cancel").click()

    user.find(marker="profile-delete").click()
    await user.should_see(marker="confirmation-delete-consequence")
    await user.should_see(
        "Alternate (alternate) is permanently deleted, then you return to Scenarios."
    )
    assert calls == []
    user.find(marker="profile-delete-cancel").click()


async def test_committed_refresh_failure_locks_stale_profile_page(user, tmp_path):
    from lte_scenario_toolkit.gui.app import create_app

    profile = _task13_profile(tmp_path)
    copy_calls = []
    preflight_calls = []

    class RecordingStore:
        def discover(self, scenario_id):
            return [profile]

        def copy(self, *args):
            copy_calls.append(args)
            return tmp_path / "configs/copy.yaml"

    class RecordingSelectionService:
        def preflight(self, *args):
            preflight_calls.append(args)
            return object()

    def fail_refresh(_catalog):
        raise RuntimeError("catalog reload failed")

    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=RecordingStore(),
        catalog_loader=fail_refresh,
        selection_service_factory=lambda _catalog: RecordingSelectionService(),
        testing=True,
    )
    await user.open("/configure/ready-city")

    user.find(marker="profile-copy").click()
    user.find(marker="profile-copy-confirm").click()
    await user.should_see(
        "The profile change was saved, but the runtime could not refresh."
    )
    for marker in (
        "profile-copy",
        "profile-rename",
        "profile-set-default",
        "profile-delete",
        "profile-validate",
        "profile-save",
        "profile-start-scan",
    ):
        assert all(not element.enabled for element in user.find(marker=marker).elements)

    user.find(marker="profile-copy").click()
    user.find(marker="profile-validate").click()
    user.find(marker="profile-start-scan").click()
    assert len(copy_calls) == 1
    assert preflight_calls == []


async def test_new_profile_save_navigates_to_committed_profile_id(user, tmp_path):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.app import create_app

    profiles = []

    class RecordingStore:
        def discover(self, scenario_id):
            return list(profiles)

        def save(self, profile, **kwargs):
            saved = replace(
                profile,
                source_path=tmp_path / f"configs/{profile.scenario_id}/{profile.profile_id}.yaml",
            )
            profiles.append(saved)
            return saved.source_path

    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=RecordingStore(),
        testing=True,
    )
    await user.open("/configure/without-default")

    user.find(marker="profile-save").click()
    expected_route = "/configure/without-default?profile=default"
    for _ in range(20):
        if user.back_history and user.back_history[-1] == expected_route:
            break
        await asyncio.sleep(0.05)

    assert user.back_history[-1] == expected_route
    assert profiles[0].profile_id == "default"


async def test_profile_rename_navigates_to_new_profile_id(user, tmp_path):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.app import create_app

    profiles = [_task13_profile(tmp_path)]

    class RecordingStore:
        def discover(self, scenario_id):
            return list(profiles)

        def rename(self, source, profile_id, display_name):
            profiles[0] = replace(
                profiles[0],
                profile_id=profile_id,
                display_name=display_name,
                source_path=tmp_path / f"configs/ready-city/{profile_id}.yaml",
            )
            return profiles[0].source_path

    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=RecordingStore(),
        testing=True,
    )
    await user.open("/configure/ready-city")

    user.find(marker="profile-rename").click()
    user.find(marker="profile-rename-id").clear().type("renamed")
    user.find(marker="profile-rename-name").clear().type("Renamed")
    user.find(marker="profile-rename-confirm").click()
    expected_route = "/configure/ready-city?profile=renamed"
    for _ in range(20):
        if user.back_history and user.back_history[-1] == expected_route:
            break
        await asyncio.sleep(0.05)

    assert user.back_history[-1] == expected_route
    assert profiles[0].profile_id == "renamed"




async def test_fractional_gui_integer_does_not_call_profile_store(user, tmp_path):
    from lte_scenario_toolkit.gui.app import create_app

    calls = []

    class RecordingStore:
        def discover(self, scenario_id):
            return []

        def save(self, *args, **kwargs):
            calls.append((args, kwargs))

    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=RecordingStore(),
        testing=True,
    )
    await user.open("/configure/without-default")

    user.find(marker="profile-rect-size").clear().type("2000.5")
    user.find(marker="profile-save").click()

    await user.should_see("The operation could not be completed")
    assert calls == []


async def test_corrected_preflight_clears_previous_field_error(user, tmp_path):
    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.selection_service import SelectionPreflightError

    calls = []

    class EmptyStore:
        def discover(self, scenario_id):
            return []

    class RecordingSelectionService:
        def preflight(self, profile, output_root):
            calls.append(profile.target_crs)
            if len(calls) == 1:
                raise SelectionPreflightError(
                    "profile.target_crs",
                    "technical CRS failure",
                )
            return object()

    service = RecordingSelectionService()
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=EmptyStore(),
        selection_service_factory=lambda catalog: service,
        testing=True,
    )
    await user.open("/configure/without-default")

    user.find(marker="profile-target-crs").clear().type("INVALID")
    user.find(marker="profile-start-scan").click()
    await user.should_see("Enter a valid target CRS.")

    user.find(marker="profile-target-crs").clear().type("EPSG:3857")
    user.find(marker="profile-start-scan").click()
    await user.should_see("Preflight passed")
    await user.should_not_see("Enter a valid target CRS.")


def test_custom_coordinator_does_not_create_or_own_shared_coordinator(
    tmp_path, monkeypatch
):
    module = _gui_module("app")
    calls = []
    fake_app = SimpleNamespace(on_shutdown=lambda callback: calls.append(callback))

    class FakeUi:
        def add_css(self, content, *, shared=False):
            pass

        def page(self, path, **kwargs):
            return lambda function: function

    monkeypatch.setitem(sys.modules, "nicegui", SimpleNamespace(app=fake_app, ui=FakeUi()))
    monkeypatch.setattr(
        module,
        "install_gui_assets",
        lambda _app, _ui: "/_lte_gui/assets/station-dots.js",
    )
    monkeypatch.setattr(
        module,
        "get_job_coordinator",
        lambda: pytest.fail("custom coordinator must not create the shared coordinator"),
    )
    custom = object()

    module.create_app(
        catalog=SimpleNamespace(root=tmp_path.resolve()),
        profile_store=object(),
        selection_service_factory=lambda catalog: SimpleNamespace(catalog=catalog),
        coordinator=custom,
        testing=True,
    )

    assert len(calls) == 2
    assert all(callable(callback) for callback in calls)
    assert module.shutdown_job_coordinator not in calls
    for callback in calls:
        callback()


def _task14_scan_result(*, completed=True):
    from lte_scenario_toolkit.candidate_scanner import Candidate, ScanResult

    return ScanResult(
        candidates=(
            Candidate(0, 1, 0.0, 0.0, 0.5, 0.5),
            Candidate(4, 2, 0.25, 0.25, 0.75, 0.75),
        ),
        checked_positions=10,
        total_positions=10,
        completed=completed,
        algorithm_version="row-sweep-v1",
    )


def test_candidate_state_layout_switch_preserves_identity_map_and_layers():
    from lte_scenario_toolkit.gui.pages.candidates import CandidatePageState

    state = CandidatePageState.from_scan("job-1", _task14_scan_result())
    state = (
        state.with_view("map")
        .with_selected(1)
        .with_map_bounds((-1.0, -2.0, 3.0, 4.0))
        .with_layer("stations", False)
        .with_dem(opacity=0.4)
    )

    switched = state.with_view("filmstrip")

    assert switched.selected_flat_grid_id == 4
    assert switched.selected_index == 1
    assert switched.map_bounds == (-1.0, -2.0, 3.0, 4.0)
    assert "stations" not in switched.enabled_layers
    assert switched.dem_opacity == pytest.approx(0.4)
    assert switched.phase == state.phase
    with pytest.raises(FrozenInstanceError):
        switched.view = "map"


def test_candidate_map_status_keys_cover_loading_empty_error_online_and_offline():
    from lte_scenario_toolkit.gui.pages.candidates import (
        CandidatePageState,
        _candidate_status_keys,
    )

    assert _candidate_status_keys(CandidatePageState()) == (
        "candidates.map_ready",
        "candidates.source_offline",
    )
    assert _candidate_status_keys(CandidatePageState(phase="scanning")) == (
        "candidates.map_loading",
        "candidates.source_offline",
    )
    assert _candidate_status_keys(CandidatePageState(phase="failed")) == (
        "candidates.map_error",
        "candidates.source_offline",
    )
    empty = CandidatePageState(phase="completed", scan_completed=True)
    assert _candidate_status_keys(empty) == (
        "candidates.map_empty",
        "candidates.source_offline",
    )
    assert _candidate_status_keys(empty.with_layer("online", True)) == (
        "candidates.map_empty",
        "candidates.source_online",
    )


def test_candidate_confirm_requires_authoritative_completed_result_and_selection():
    from lte_scenario_toolkit.gui.pages.candidates import CandidatePageState

    running = CandidatePageState.from_scan(
        "job-1", _task14_scan_result(completed=False)
    )
    complete = CandidatePageState.from_scan("job-1", _task14_scan_result())

    assert running.with_selected(0).can_confirm is False
    assert complete.can_confirm is False
    assert complete.with_selected(0).can_confirm is True
    assert complete.with_selected_flat_grid_id(999).can_confirm is False


def test_final_scan_reorder_preserves_selection_only_by_flat_grid_id():
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages.candidates import CandidatePageState

    result = _task14_scan_result()
    selected = CandidatePageState.from_scan("job-1", result).with_selected(1)
    reordered = replace(result, candidates=tuple(reversed(result.candidates)))

    preserved = selected.with_scan_result(reordered)
    removed = selected.with_scan_result(
        replace(result, candidates=(result.candidates[0],))
    )

    assert preserved.selected_flat_grid_id == 4
    assert preserved.selected_index == 0
    assert removed.selected_flat_grid_id is None
    assert removed.statistics is None


def test_final_scan_keeps_elapsed_and_failed_scan_clears_provisional_details():
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages.candidates import CandidatePageState

    provisional = CandidatePageState.starting("job-1")
    provisional = replace(
        provisional,
        elapsed_seconds=2.75,
        candidates=_task14_scan_result().candidates,
        found_count=2,
        selected_flat_grid_id=4,
    )

    completed = provisional.with_scan_result(_task14_scan_result())
    failed = provisional.failed(
        "bad scan",
        code="scan.failed",
        details={"dataset": "points"},
    )

    assert completed.elapsed_seconds == pytest.approx(2.75)
    assert failed.candidates == ()
    assert failed.selected_flat_grid_id is None
    assert failed.error_code == "scan.failed"
    assert failed.error_details == (("dataset", "points"),)


def test_progress_reducer_replays_deltas_and_ignores_stale_jobs():
    from dataclasses import replace

    from lte_scenario_toolkit.candidate_scanner import Candidate
    from lte_scenario_toolkit.gui.pages.candidates import (
        CandidatePageState,
        CandidateProgressEvent,
        reduce_progress,
    )
    from lte_scenario_toolkit.selection_service import SelectionProgress

    first, selected = _task14_scan_result().candidates
    state = CandidatePageState.starting("job-2")
    event = SelectionProgress(
        phase="scanning",
        checked_positions=5,
        total_positions=10,
        candidate_count=2,
        elapsed_seconds=0.25,
        added_candidates=(first, selected),
        removed_flat_grid_ids=(),
        cache_status="miss",
        cache_key="key-1",
    )
    state = reduce_progress(
        state,
        CandidateProgressEvent("job-2", event),
    ).with_selected_flat_grid_id(4)
    replacement = Candidate(8, 3, 1.0, 1.0, 1.5, 1.5)
    replacing = replace(
        event,
        checked_positions=10,
        candidate_count=2,
        added_candidates=(replacement,),
        removed_flat_grid_ids=(4,),
        phase="completed",
    )

    updated = reduce_progress(state, CandidateProgressEvent("job-2", replacing))
    stale = reduce_progress(updated, CandidateProgressEvent("old-job", event))

    assert tuple(candidate.flat_grid_id for candidate in updated.candidates) == (0, 8)
    assert updated.selected_flat_grid_id is None
    assert updated.checked_positions == 10
    assert updated.found_count == 2
    assert updated.cache_status == "miss"
    assert updated.scan_completed is False
    assert stale is updated


def test_progress_duplicate_upsert_does_not_duplicate_candidate():
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages.candidates import (
        CandidatePageState,
        reduce_progress,
    )
    from lte_scenario_toolkit.selection_service import SelectionProgress

    candidate = _task14_scan_result().candidates[0]
    event = SelectionProgress(
        phase="scanning",
        checked_positions=1,
        total_positions=2,
        candidate_count=1,
        elapsed_seconds=0.1,
        added_candidates=(candidate,),
        removed_flat_grid_ids=(),
        cache_status="hit",
        cache_key="key",
    )
    state = reduce_progress(CandidatePageState.starting("job"), event)
    state = reduce_progress(
        state,
        replace(event, added_candidates=(replace(candidate, point_count=9),)),
    )

    assert len(state.candidates) == 1
    assert state.candidates[0].point_count == 9


def test_progress_cannot_revive_a_cancelling_scan():
    from lte_scenario_toolkit.gui.pages.candidates import (
        CandidatePageState,
        reduce_progress,
    )
    from lte_scenario_toolkit.selection_service import SelectionProgress

    state = CandidatePageState.starting("job").with_phase("cancelling")
    progress = SelectionProgress(
        phase="scanning",
        checked_positions=1,
        total_positions=10,
        candidate_count=0,
        elapsed_seconds=0.1,
        added_candidates=(),
        removed_flat_grid_ids=(),
        cache_status="miss",
        cache_key="key",
    )

    assert reduce_progress(state, progress).phase == "cancelling"


def test_statistics_reducer_rejects_stale_scan_job_candidate_and_stats_job():
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages.candidates import (
        CandidatePageState,
        CandidateStatisticsEvent,
        reduce_statistics,
    )
    from lte_scenario_toolkit.selection_service import DemStatistics

    state = CandidatePageState.from_scan("scan-2", _task14_scan_result())
    state = state.with_selected_flat_grid_id(4).with_statistics_job("stats-2", 4)
    statistics = DemStatistics(1.0, 5.0, 3.0, 4.0, 10)
    good = CandidateStatisticsEvent("scan-2", "stats-2", 4, statistics, None)

    assert reduce_statistics(
        state,
        replace(good, scan_job_id="scan-1"),
    ) is state
    assert reduce_statistics(
        state,
        replace(good, statistics_job_id="stats-1"),
    ) is state
    assert reduce_statistics(state.with_selected_flat_grid_id(0), good).statistics is None
    applied = reduce_statistics(state, good)
    assert applied.statistics == statistics
    assert applied.statistics_flat_grid_id == 4


def test_candidate_bounds_hit_testing_handles_sparse_overlaps_and_non_3857_crs():
    from lte_scenario_toolkit.gui.pages.candidates import (
        candidate_display_bounds,
        hit_test_candidate_indices,
    )

    candidates = _task14_scan_result().candidates
    bounds = candidate_display_bounds(candidates, rectangle_size=1.0, crs="EPSG:4326")

    assert hit_test_candidate_indices(bounds, latitude=0.5, longitude=0.5) == (0, 1)
    assert hit_test_candidate_indices(bounds, latitude=2.0, longitude=2.0) == ()
    assert bounds[1].flat_grid_id == 4


def test_online_tile_probe_is_optional_and_failure_is_nonfatal():
    from lte_scenario_toolkit.gui.pages.candidates import online_tiles_available

    assert online_tiles_available(None) is False
    assert online_tiles_available(lambda: True) is True
    assert online_tiles_available(lambda: (_ for _ in ()).throw(OSError("offline"))) is False


def _task14_session(tmp_path, service, *, session_id="session-1", map_bundle=None):
    from lte_scenario_toolkit.gui.pages.candidates import CandidateSession

    profile = _task13_profile(tmp_path)
    preflight = SimpleNamespace(profile=profile)
    return CandidateSession(
        session_id=session_id,
        profile_snapshot=profile,
        preflight=preflight,
        selection_service=service,
        repo_root=tmp_path.resolve(),
        map_bundle=map_bundle,
    )


def test_candidate_scan_provenance_is_frozen_validated_and_fail_safe():
    from lte_scenario_toolkit.gui.pages.candidates import (
        CandidatePageState,
        CandidateScanProvenance,
    )

    provenance = CandidateScanProvenance(8.2, "miss", "cache-key")

    assert provenance.elapsed_seconds == pytest.approx(8.2)
    assert provenance.cache_status == "miss"
    assert provenance.cache_key == "cache-key"
    assert not hasattr(provenance, "__dict__")
    with pytest.raises(FrozenInstanceError):
        provenance.cache_status = "hit"

    fallback = CandidatePageState.from_scan("scan", _task14_scan_result())
    assert fallback.elapsed_seconds == pytest.approx(0.0)
    assert fallback.cache_status == "none"
    assert fallback.cache_key == ""

    invalid_values = (
        (True, "miss", "cache-key"),
        (-0.1, "miss", "cache-key"),
        (float("inf"), "miss", "cache-key"),
        (float("nan"), "miss", "cache-key"),
        ("8.2", "miss", "cache-key"),
        (8.2, "", "cache-key"),
        (8.2, 1, "cache-key"),
        (8.2, "miss", None),
    )
    for values in invalid_values:
        with pytest.raises(ValueError):
            CandidateScanProvenance(*values)


def test_candidate_session_preserves_prior_positional_constructor_order(tmp_path):
    from lte_scenario_toolkit.gui.pages.candidates import CandidateSession

    profile = _task13_profile(tmp_path)
    preflight = SimpleNamespace(profile=profile)
    service = object()
    bundle = _task14_map_bundle(tmp_path)
    result = _task14_scan_result()
    locked = result.candidates[1]

    session = CandidateSession(
        "legacy-positional",
        profile,
        preflight,
        service,
        tmp_path,
        bundle,
        result,
        4,
        locked,
        123.5,
    )

    assert session.map_bundle is bundle
    assert session.scan_result is result
    assert session.confirmed_flat_grid_id == 4
    assert session.locked_candidate is locked
    assert session.created_at == pytest.approx(123.5)
    assert session.scan_provenance is None


def test_completed_candidate_session_restores_scan_provenance_from_registry(tmp_path):
    from lte_scenario_toolkit.gui.pages.candidates import (
        CandidateExplorerController,
        CandidateScanProvenance,
        CandidateSessionRegistry,
    )
    from lte_scenario_toolkit.jobs import JobCoordinator

    result = _task14_scan_result()
    provenance = CandidateScanProvenance(8.2, "miss", "cache-key")
    registry = CandidateSessionRegistry()
    session = registry.add(_task14_session(tmp_path, object()))
    completed = registry.set_scan_result(session.session_id, result, provenance)
    coordinator = JobCoordinator()
    controller = CandidateExplorerController(completed, coordinator, registry=registry)
    try:
        state = controller.state
        assert state.elapsed_seconds == pytest.approx(8.2)
        assert state.cache_status == "miss"
        assert state.cache_key == "cache-key"
        assert state.candidates is result.candidates
        assert state.found_count == len(result.candidates)
        assert state.checked_positions == result.checked_positions
        assert state.total_positions == result.total_positions
        assert state.algorithm_version == result.algorithm_version
        assert registry.get(session.session_id).scan_provenance is provenance
    finally:
        controller.close()
        coordinator.shutdown()


def test_scan_controller_atomically_persists_completed_result_and_provenance(tmp_path):
    from lte_scenario_toolkit.gui.pages.candidates import (
        CandidateExplorerController,
        CandidateSessionRegistry,
    )
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.selection_service import SelectionProgress

    result = _task14_scan_result()

    class Service:
        def scan(self, _preflight, *, force=False, progress=None, cancel=None):
            assert force is False
            assert cancel is not None
            progress(
                SelectionProgress(
                    phase="scanning",
                    checked_positions=result.checked_positions,
                    total_positions=result.total_positions,
                    candidate_count=len(result.candidates),
                    elapsed_seconds=8.2,
                    added_candidates=result.candidates,
                    removed_flat_grid_ids=(),
                    cache_status="miss",
                    cache_key="cache-key",
                )
            )
            return result

    registry = CandidateSessionRegistry()
    session = registry.add(_task14_session(tmp_path, Service()))
    coordinator = JobCoordinator()
    controller = CandidateExplorerController(session, coordinator, registry=registry)
    restored = None
    try:
        job = controller.start_scan()
        assert job.future is not None
        job.future.result(timeout=2)
        state = controller.drain_scan(job)

        stored = registry.get(session.session_id)
        assert stored is not None
        assert stored.scan_result is result
        assert stored.scan_provenance is not None
        assert stored.scan_provenance.elapsed_seconds == pytest.approx(8.2)
        assert stored.scan_provenance.cache_status == "miss"
        assert stored.scan_provenance.cache_key == "cache-key"
        assert controller.session is stored
        assert state.elapsed_seconds == pytest.approx(8.2)

        controller.close()
        restored = CandidateExplorerController(stored, coordinator, registry=registry)
        assert restored.state.elapsed_seconds == pytest.approx(8.2)
        assert restored.state.cache_status == "miss"
        assert restored.state.cache_key == "cache-key"
    finally:
        if restored is not None:
            restored.close()
        else:
            controller.close()
        coordinator.shutdown()


def test_cancelled_rescan_restores_prior_completed_candidate_state(tmp_path):
    from lte_scenario_toolkit.candidate_scanner import ScanCancelled
    from lte_scenario_toolkit.gui.pages.candidates import (
        CandidateExplorerController,
        CandidateScanProvenance,
        CandidateSessionRegistry,
    )
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.selection_service import SelectionProgress

    entered = Event()
    original = _task14_scan_result()
    provenance = CandidateScanProvenance(8.2, "miss", "original-key")

    class Service:
        def scan(self, _preflight, *, force=False, progress=None, cancel=None):
            assert force is True
            progress(
                SelectionProgress(
                    phase="scanning",
                    checked_positions=1,
                    total_positions=100,
                    candidate_count=1,
                    elapsed_seconds=0.1,
                    added_candidates=(original.candidates[0],),
                    removed_flat_grid_ids=(),
                    cache_status="forced",
                    cache_key="rescan-key",
                )
            )
            entered.set()
            assert cancel.wait(2)
            raise ScanCancelled()

    registry = CandidateSessionRegistry()
    registered = registry.add(_task14_session(tmp_path, Service()))
    completed = registry.set_scan_result(
        registered.session_id,
        original,
        provenance,
    )
    coordinator = JobCoordinator()
    controller = CandidateExplorerController(completed, coordinator, registry=registry)
    try:
        controller.select_flat_grid_id(4)
        controller.set_view("filmstrip")
        controller.set_map_bounds((-1.0, -2.0, 3.0, 4.0))
        controller.set_layer("stations", False)
        controller.set_dem_opacity(0.4)

        job = controller.start_scan(force=True)
        assert entered.wait(2)
        controller.drain_scan(job)
        assert controller.cancel_scan() is True
        assert job.future is not None
        with pytest.raises(ScanCancelled):
            job.future.result(timeout=2)
        state = controller.drain_scan(job)

        assert state.phase == "cancelled"
        assert state.scan_completed is True
        assert state.candidates is original.candidates
        assert state.found_count == len(original.candidates)
        assert state.checked_positions == original.checked_positions
        assert state.total_positions == original.total_positions
        assert state.elapsed_seconds == pytest.approx(8.2)
        assert state.cache_status == "miss"
        assert state.cache_key == "original-key"
        assert state.selected_flat_grid_id == 4
        assert state.view == "filmstrip"
        assert state.map_bounds == (-1.0, -2.0, 3.0, 4.0)
        assert "stations" not in state.enabled_layers
        assert state.dem_opacity == pytest.approx(0.4)
        assert state.can_confirm is True
        stored = registry.get(registered.session_id)
        assert stored.scan_result is original
        assert stored.scan_provenance is provenance
        assert controller.session.scan_result is original
        assert controller.session.scan_provenance is provenance
    finally:
        controller.close()
        coordinator.shutdown()


def test_failed_rescan_restores_prior_completed_candidate_state(tmp_path):
    from lte_scenario_toolkit.gui.pages.candidates import (
        CandidateExplorerController,
        CandidateScanProvenance,
        CandidateSessionRegistry,
    )
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.selection_service import SelectionScanError

    original = _task14_scan_result()
    provenance = CandidateScanProvenance(8.2, "hit", "original-key")

    class Service:
        def scan(self, *_args, **_kwargs):
            raise SelectionScanError(
                "scan.failed",
                "rescan failed",
                details={"dataset": "points"},
            )

    registry = CandidateSessionRegistry()
    registered = registry.add(_task14_session(tmp_path, Service()))
    completed = registry.set_scan_result(
        registered.session_id,
        original,
        provenance,
    )
    coordinator = JobCoordinator()
    controller = CandidateExplorerController(completed, coordinator, registry=registry)
    try:
        controller.select_flat_grid_id(4)
        controller.set_layer("boundary", False)
        job = controller.start_scan(force=True)
        assert job.future is not None
        with pytest.raises(SelectionScanError):
            job.future.result(timeout=2)
        state = controller.drain_scan(job)

        assert state.phase == "failed"
        assert state.error == "rescan failed"
        assert state.error_code == "scan.failed"
        assert state.error_details == (("dataset", "points"),)
        assert state.scan_completed is True
        assert state.candidates is original.candidates
        assert state.selected_flat_grid_id == 4
        assert "boundary" not in state.enabled_layers
        assert state.elapsed_seconds == pytest.approx(8.2)
        assert state.cache_status == "hit"
        assert state.cache_key == "original-key"
        assert state.can_confirm is True
        stored = registry.get(registered.session_id)
        assert stored.scan_result is original
        assert stored.scan_provenance is provenance
    finally:
        controller.close()
        coordinator.shutdown()


def test_invalid_progress_rescan_restores_prior_completed_candidate_state(tmp_path):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages.candidates import (
        CandidateExplorerController,
        CandidateScanProvenance,
        CandidateSessionRegistry,
    )
    from lte_scenario_toolkit.jobs import JobCoordinator

    original = _task14_scan_result()
    replacement = replace(
        original,
        candidates=(original.candidates[0],),
        algorithm_version="replacement-version",
    )
    provenance = CandidateScanProvenance(8.2, "miss", "original-key")

    class Service:
        def scan(self, _preflight, *, progress=None, **_kwargs):
            progress(object())
            return replacement

    registry = CandidateSessionRegistry()
    registered = registry.add(_task14_session(tmp_path, Service()))
    completed = registry.set_scan_result(
        registered.session_id,
        original,
        provenance,
    )
    coordinator = JobCoordinator()
    controller = CandidateExplorerController(completed, coordinator, registry=registry)
    try:
        controller.select_flat_grid_id(4)
        job = controller.start_scan(force=True)
        assert job.future is not None
        job.future.result(timeout=2)
        state = controller.drain_scan(job)

        assert state.phase == "failed"
        assert state.error == "Invalid scan progress event"
        assert state.scan_completed is True
        assert state.candidates is original.candidates
        assert state.algorithm_version == original.algorithm_version
        assert state.selected_flat_grid_id == 4
        assert state.can_confirm is True
        stored = registry.get(registered.session_id)
        assert stored.scan_result is original
        assert stored.scan_provenance is provenance
    finally:
        controller.close()
        coordinator.shutdown()


def test_incomplete_rescan_restores_prior_completed_candidate_state(tmp_path):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages.candidates import (
        CandidateExplorerController,
        CandidateScanProvenance,
        CandidateSessionRegistry,
    )
    from lte_scenario_toolkit.jobs import JobCoordinator

    original = _task14_scan_result()
    incomplete = replace(
        original,
        candidates=(original.candidates[0],),
        checked_positions=5,
        completed=False,
    )
    provenance = CandidateScanProvenance(8.2, "miss", "original-key")

    class Service:
        def scan(self, *_args, **_kwargs):
            return incomplete

    registry = CandidateSessionRegistry()
    registered = registry.add(_task14_session(tmp_path, Service()))
    completed = registry.set_scan_result(
        registered.session_id,
        original,
        provenance,
    )
    coordinator = JobCoordinator()
    controller = CandidateExplorerController(completed, coordinator, registry=registry)
    try:
        controller.select_flat_grid_id(4)
        job = controller.start_scan(force=True)
        assert job.future is not None
        job.future.result(timeout=2)
        state = controller.drain_scan(job)

        assert state.phase == "failed"
        assert state.error is not None and "incomplete" in state.error
        assert state.scan_completed is True
        assert state.candidates is original.candidates
        assert state.selected_flat_grid_id == 4
        assert state.can_confirm is True
        stored = registry.get(registered.session_id)
        assert stored.scan_result is original
        assert stored.scan_provenance is provenance
    finally:
        controller.close()
        coordinator.shutdown()


def test_candidate_session_registry_is_bounded_opaque_and_confirms_final_identity(tmp_path):
    from lte_scenario_toolkit.gui.pages.candidates import (
        CandidateScanProvenance,
        CandidateSessionRegistry,
    )

    service = object()
    registry = CandidateSessionRegistry(max_sessions=2)
    first = registry.add(_task14_session(tmp_path, service, session_id="opaque-1"))
    second = registry.add(_task14_session(tmp_path, service, session_id="opaque-2"))
    third = registry.add(_task14_session(tmp_path, service, session_id="opaque-3"))

    assert first.session_id == "opaque-1"
    assert registry.get("opaque-1") is None
    assert registry.get(second.session_id) is second
    result = _task14_scan_result()
    updated = registry.set_scan_result(
        third.session_id,
        result,
        CandidateScanProvenance(0.0, "none", ""),
    )
    confirmed = registry.confirm(third.session_id, 4)
    assert updated.scan_result is result
    assert confirmed.confirmed_flat_grid_id == 4
    assert confirmed.locked_candidate is result.candidates[1]
    confirmed_sessions = getattr(registry, "confirmed_sessions", None)
    assert callable(confirmed_sessions)
    assert confirmed_sessions() == (confirmed,)
    with pytest.raises(ValueError, match="final scan"):
        registry.confirm(second.session_id, 4)


def test_candidate_registry_never_evicts_a_pinned_page_session(tmp_path):
    from lte_scenario_toolkit.gui.pages.candidates import CandidateSessionRegistry

    registry = CandidateSessionRegistry(max_sessions=1)
    active = registry.add(_task14_session(tmp_path, object(), session_id="active"))
    registry.pin(active.session_id)
    try:
        with pytest.raises(RuntimeError, match="active pages"):
            registry.add(_task14_session(tmp_path, object(), session_id="new"))
        assert registry.get("active") is active
        assert registry.get("new") is None
    finally:
        registry.unpin(active.session_id)


def test_scan_controller_runs_framework_free_worker_and_uses_final_result_order(tmp_path):
    from threading import Event

    from lte_scenario_toolkit.gui.pages.candidates import CandidateExplorerController
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.selection_service import SelectionProgress

    result = _task14_scan_result()
    worker_entered = Event()

    class FakeService:
        def scan(self, preflight, force=False, progress=None, cancel=None):
            worker_entered.set()
            progress(
                SelectionProgress(
                    phase="scanning",
                    checked_positions=5,
                    total_positions=10,
                    candidate_count=1,
                    elapsed_seconds=0.1,
                    added_candidates=(result.candidates[1],),
                    removed_flat_grid_ids=(),
                    cache_status="miss",
                    cache_key="key",
                )
            )
            return result

    coordinator = JobCoordinator()
    controller = CandidateExplorerController(
        _task14_session(tmp_path, FakeService()), coordinator
    )
    try:
        job = controller.start_scan()
        assert worker_entered.wait(2)
        assert job.future is not None
        job.future.result(timeout=2)
        state = controller.drain_scan(job)

        assert state.scan_completed is True
        assert state.phase == "completed"
        assert state.candidates == result.candidates
        assert coordinator.snapshot().active is False
    finally:
        controller.close()
        coordinator.shutdown()


def test_scan_controller_cancellation_clears_provisional_only_after_worker_stops(tmp_path):
    from threading import Event

    from lte_scenario_toolkit.candidate_scanner import ScanCancelled
    from lte_scenario_toolkit.gui.pages.candidates import CandidateExplorerController
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.selection_service import SelectionProgress

    entered = Event()
    candidate = _task14_scan_result().candidates[0]

    class CancellingService:
        def scan(self, preflight, force=False, progress=None, cancel=None):
            progress(
                SelectionProgress(
                    phase="scanning",
                    checked_positions=1,
                    total_positions=100,
                    candidate_count=1,
                    elapsed_seconds=0.1,
                    added_candidates=(candidate,),
                    removed_flat_grid_ids=(),
                    cache_status="miss",
                    cache_key="key",
                )
            )
            entered.set()
            assert cancel.wait(2)
            raise ScanCancelled()

    coordinator = JobCoordinator()
    controller = CandidateExplorerController(
        _task14_session(tmp_path, CancellingService()), coordinator
    )
    try:
        job = controller.start_scan()
        assert entered.wait(2)
        provisional = controller.drain_scan(job)
        assert provisional.candidates == (candidate,)

        assert controller.cancel_scan() is True
        assert controller.state.phase == "cancelling"
        assert controller.state.candidates == (candidate,)
        assert job.future is not None
        with pytest.raises(ScanCancelled):
            job.future.result(timeout=2)
        cancelled = controller.drain_scan(job)
        assert cancelled.phase == "cancelled"
        assert cancelled.candidates == ()
        assert cancelled.can_confirm is False
    finally:
        controller.close()
        coordinator.shutdown()


def test_force_rescan_busy_preserves_completed_candidates_and_selection(tmp_path):
    from lte_scenario_toolkit.gui.pages.candidates import (
        CandidateExplorerController,
        CandidatePageState,
    )
    from lte_scenario_toolkit.jobs import JobBusyError, JobCoordinator

    class Service:
        def scan(self, *args, **kwargs):
            return _task14_scan_result()

    coordinator = JobCoordinator()
    initial_state = CandidatePageState.from_scan(
        "completed-job", _task14_scan_result()
    ).with_selected_flat_grid_id(4)
    controller = CandidateExplorerController(
        _task14_session(tmp_path, Service()),
        coordinator,
        initial_state=initial_state,
    )
    blocker = coordinator.start("other")
    before = controller.state
    try:
        with pytest.raises(JobBusyError):
            controller.start_scan(force=True)
        assert controller.state is before
    finally:
        coordinator.finish(blocker.job_id)
        controller.close()
        coordinator.shutdown()


def test_statistics_job_ignores_stale_selection_and_preview_failure_is_local(tmp_path):
    from dataclasses import replace
    from threading import Event

    from lte_scenario_toolkit.gui.pages.candidates import CandidateExplorerController
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.map_assets import MapStyle
    from lte_scenario_toolkit.selection_service import DemStatistics

    entered = Event()
    release = Event()
    calls = []

    class Service:
        def candidate_statistics(self, preflight, candidate):
            calls.append(candidate.flat_grid_id)
            if len(calls) == 1:
                entered.set()
                assert release.wait(2)
            return DemStatistics(1.0, 5.0, 3.0, 4.0, 10)

    result = _task14_scan_result()
    session = replace(
        _task14_session(tmp_path, Service()),
        scan_result=result,
    )
    coordinator = JobCoordinator()
    controller = CandidateExplorerController(
        session,
        coordinator,
        candidate_overlay_builder=lambda *_args: (_ for _ in ()).throw(
            OSError("preview failed")
        ),
    )
    try:
        controller.select_flat_grid_id(0)
        first = controller.request_statistics()
        assert first is not None
        assert entered.wait(2)
        controller.select_flat_grid_id(4)
        release.set()
        assert first.future is not None
        first.future.result(timeout=2)
        stale = controller.drain_statistics(first)
        assert stale.selected_flat_grid_id == 4
        assert stale.statistics is None
        assert coordinator.snapshot().active is False

        second = controller.request_statistics()
        assert second is not None and second.future is not None
        second.future.result(timeout=2)
        applied = controller.drain_statistics(second)
        assert applied.statistics == DemStatistics(1.0, 5.0, 3.0, 4.0, 10)
        assert applied.statistics_flat_grid_id == 4
        assert applied.candidate_preview_asset is None
        assert applied.candidate_preview_error == "preview failed"
        assert applied.dem_style is MapStyle.COMBINED
        assert coordinator.snapshot().active is False
    finally:
        controller.close()
        coordinator.shutdown()


def _task14_map_bundle(tmp_path):
    from lte_scenario_toolkit.gui.pages.candidates import CandidateMapBundle
    from lte_scenario_toolkit.map_assets import MapAsset, MapStyle

    cache_root = tmp_path / ".lte-data" / "cache" / "maps"
    cache_root.mkdir(parents=True, exist_ok=True)
    overlay = cache_root / "overview.png"
    overlay.write_bytes(b"png")
    return CandidateMapBundle(
        dem_asset=MapAsset(
            path=overlay,
            bounds=(-1.0, -1.0, 2.0, 2.0),
            bounds_crs="EPSG:4326",
            style=MapStyle.COMBINED,
        ),
        boundary_geojson={
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-1, -1], [2, -1], [2, 2], [-1, 2], [-1, -1]]],
            },
        },
        stations_geojson={"type": "FeatureCollection", "features": []},
        map_bounds=(-1.0, -1.0, 2.0, 2.0),
    )


async def test_candidate_route_connects_before_slow_map_preparation_finishes(
    user, tmp_path
):
    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.pages.candidates import CandidateSessionRegistry

    started = Event()
    release = Event()

    class Service:
        def scan(self, *args, **kwargs):
            return _task14_scan_result()

    def slow_bundle_builder(_session, _assets):
        started.set()
        assert release.wait(2)
        return _task14_map_bundle(tmp_path)

    registry = CandidateSessionRegistry()
    registry.add(_task14_session(tmp_path, Service()))
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=object(),
        candidate_registry=registry,
        candidate_bundle_builder=slow_bundle_builder,
        testing=True,
    )

    opening = asyncio.create_task(user.open("/candidates/session-1"))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        await asyncio.wait_for(asyncio.shield(opening), 1)
        await user.should_see("Preparing the offline candidate map...")
    finally:
        release.set()
        await asyncio.wait_for(opening, 2)

    await user.should_see("Candidate Explorer")


def test_gui_file_route_resolver_accepts_only_allowlisted_local_files(tmp_path):
    module = _gui_module("app")
    cache_root = tmp_path / ".lte-data" / "cache"
    run_root = tmp_path / "runs"
    cache_root.mkdir(parents=True)
    run_root.mkdir()
    cache_asset = cache_root / "map.png"
    run_asset = run_root / "artifact.csv"
    outside = tmp_path / "outside.png"
    cache_asset.write_bytes(b"png")
    run_asset.write_text("x,y\n1,2\n", encoding="utf-8")
    outside.write_bytes(b"png")

    assert module._resolve_allowlisted_file(
        cache_asset,
        roots=(cache_root, run_root),
        suffixes=(".png",),
        label="map asset",
    ) == cache_asset.resolve()
    assert module._resolve_allowlisted_file(
        run_asset,
        roots=(cache_root, run_root),
        suffixes=(".csv",),
        label="run artifact",
    ) == run_asset.resolve()

    with pytest.raises(ValueError, match="outside the allowlisted roots"):
        module._resolve_allowlisted_file(
            outside,
            roots=(cache_root, run_root),
            suffixes=(".png",),
            label="map asset",
        )
    with pytest.raises(ValueError, match="must not contain traversal"):
        module._resolve_allowlisted_file(
            cache_root / "nested" / ".." / "map.png",
            roots=(cache_root,),
            suffixes=(".png",),
            label="map asset",
        )
    with pytest.raises(ValueError, match="local filesystem path"):
        module._resolve_allowlisted_file(
            "https://example.invalid/map.png",
            roots=(cache_root,),
            suffixes=(".png",),
            label="map asset",
        )


def test_gui_file_route_resolver_rejects_symlink_escape(tmp_path):
    module = _gui_module("app")
    cache_root = tmp_path / "cache"
    outside_root = tmp_path / "outside"
    cache_root.mkdir()
    outside_root.mkdir()
    (outside_root / "secret.png").write_bytes(b"png")
    redirected = cache_root / "redirected"
    try:
        redirected.symlink_to(outside_root, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(ValueError, match="must not be redirected"):
        module._resolve_allowlisted_file(
            redirected / "secret.png",
            roots=(cache_root,),
            suffixes=(".png",),
            label="map asset",
        )


def test_scenario_preview_file_route_is_limited_to_png_cache_leaf(tmp_path):
    module = _gui_module("app")
    preview_root = tmp_path / ".lte-data/cache/scenario-previews"
    preview_root.mkdir(parents=True)
    preview = preview_root / "chicago.png"
    preview.write_bytes(b"png")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"png")
    wrong_suffix = preview_root / "details.json"
    wrong_suffix.write_text("{}", encoding="utf-8")

    assert module._resolve_scenario_preview_file(preview, tmp_path) == preview.resolve()
    with pytest.raises(ValueError, match="traversal"):
        module._resolve_scenario_preview_file(
            preview_root / "nested/../chicago.png", tmp_path
        )
    with pytest.raises(ValueError, match="outside"):
        module._resolve_scenario_preview_file(outside, tmp_path)
    with pytest.raises(ValueError, match="regular .png"):
        module._resolve_scenario_preview_file(wrong_suffix, tmp_path)


def test_scenario_preview_file_route_rejects_symlinked_cache_file(tmp_path):
    module = _gui_module("app")
    preview_root = tmp_path / ".lte-data/cache/scenario-previews"
    preview_root.mkdir(parents=True)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"png")
    linked = preview_root / "linked.png"
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks are unavailable")

    with pytest.raises(ValueError, match="redirected"):
        module._resolve_scenario_preview_file(linked, tmp_path)


@pytest.mark.asyncio
async def test_scenario_preview_delivery_waits_for_connection_and_skips_deleted_client(
    tmp_path,
):
    module = _gui_module("app")
    events = []

    class Client:
        is_deleted = False

        async def connected(self):
            events.append("connected")

    class Holder:
        def clear(self):
            events.append("cleared")

    async def cpu_bound(builder, requests, cache_root):
        events.append("cpu")
        result = builder(requests, cache_root)
        Client.is_deleted = True
        return result

    result = await module.load_scenario_previews(
        client=Client(),
        holder=Holder(),
        requests=("request",),
        cache_root=tmp_path,
        builder=lambda requests, cache_root: [requests[0], cache_root],
        render=lambda previews: events.append(("rendered", previews)),
        cpu_bound=cpu_bound,
    )

    assert result is None
    assert events == ["connected", "cpu"]


def test_gui_file_route_resolver_rejects_redirected_allowlist_ancestor(
    tmp_path,
    monkeypatch,
):
    module = _gui_module("app")
    redirected_parent = tmp_path / ".lte-data"
    cache_root = redirected_parent / "cache" / "maps"
    cache_root.mkdir(parents=True)
    asset = cache_root / "map.png"
    asset.write_bytes(b"png")
    original = module._is_redirected_path
    monkeypatch.setattr(
        module,
        "_is_redirected_path",
        lambda path: Path(path) == redirected_parent or original(path),
    )

    with pytest.raises(ValueError, match="allowlisted root must not be redirected"):
        module._resolve_allowlisted_file(
            asset,
            roots=(cache_root,),
            suffixes=(".png",),
            label="map asset",
        )


def test_gui_file_route_redirect_detector_supports_windows_reparse_points(monkeypatch):
    import stat

    module = _gui_module("app")
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    monkeypatch.setattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", flag, raising=False)

    class ReparsePath:
        @staticmethod
        def is_symlink():
            return False

        @staticmethod
        def lstat():
            return SimpleNamespace(st_file_attributes=flag)

    assert module._is_redirected_path(ReparsePath()) is True


def test_map_bundle_and_selected_overlay_use_frozen_registered_dem_inputs(tmp_path):
    from lte_scenario_toolkit.gui.pages.candidates import (
        CandidateSession,
        build_candidate_map_bundle,
        build_candidate_overlay,
    )
    from lte_scenario_toolkit.map_assets import MapAsset, MapStyle

    profile = _task13_profile(tmp_path)
    preflight = SimpleNamespace(
        profile=profile,
        dem_path=tmp_path / "registered-dem.tif",
        dem_fingerprint="dem-fingerprint",
    )
    prepared = SimpleNamespace(
        boundary=SimpleNamespace(bounds=(0.0, 0.0, 2000.0, 3000.0)),
        points=object(),
    )
    calls = []

    class Service:
        def prepared_selection(self, exact_preflight):
            assert exact_preflight is preflight
            return prepared

    class Assets:
        def dem_overlay(self, path, **kwargs):
            calls.append(("dem", path, kwargs))
            return MapAsset(
                tmp_path / "cached.png",
                kwargs["bounds"],
                kwargs["bounds_crs"],
                kwargs["style"],
            )

        def boundary_geojson(self, boundary, **kwargs):
            calls.append(("boundary", boundary, kwargs))
            return {"type": "FeatureCollection", "features": []}

        def station_geojson(self, points, boundary, **kwargs):
            calls.append(("stations", points, boundary, kwargs))
            return {"type": "FeatureCollection", "features": []}

    session = CandidateSession(
        "opaque",
        profile,
        preflight,
        Service(),
        tmp_path,
    )
    assets = Assets()
    bundle = build_candidate_map_bundle(session, assets)
    selected = build_candidate_overlay(
        session,
        assets,
        _task14_scan_result().candidates[0],
        style=MapStyle.HILLSHADE,
    )

    assert bundle.map_bounds[0] == pytest.approx(0.0)
    assert bundle.map_bounds[2] > bundle.map_bounds[0]
    assert calls[0][1] is preflight.dem_path
    assert calls[0][2]["fingerprint"] == "dem-fingerprint"
    assert calls[0][2]["bounds_crs"] == "EPSG:4326"
    assert selected.style is MapStyle.HILLSHADE
    assert calls[-1][2]["bounds"] == (0.0, 0.0, 2000.0, 2000.0)
    assert calls[-1][2]["max_dimension"] == 640


async def _new_nicegui_background_tasks(previous):
    from nicegui import background_tasks

    for _ in range(20):
        tasks = tuple(background_tasks.running_tasks - previous)
        if tasks:
            return tasks
        await asyncio.sleep(0.01)
    raise AssertionError("NiceGUI did not create an event handler background task")


async def test_candidate_online_layer_obeys_latest_toggle_intent(
    user,
    tmp_path,
    monkeypatch,
):
    from dataclasses import replace

    from nicegui import background_tasks
    from nicegui.elements.leaflet.leaflet_layers import TileLayer

    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.gui.pages.candidates import CandidateSessionRegistry
    from lte_scenario_toolkit.jobs import JobCoordinator

    entered = Event()
    release = Event()
    views = []
    actual_render = app_module.render_candidate_page

    def delayed_probe():
        entered.set()
        assert release.wait(2)
        return True

    def capture_render(*args, **kwargs):
        view = actual_render(*args, **kwargs)
        views.append(view)
        return view

    monkeypatch.setattr(app_module, "render_candidate_page", capture_render)
    registry = CandidateSessionRegistry()
    registry.add(
        replace(
            _task14_session(
                tmp_path,
                object(),
                map_bundle=_task14_map_bundle(tmp_path),
            ),
            scan_result=_task14_scan_result(),
        )
    )
    coordinator = JobCoordinator()
    try:
        app_module.create_app(
            catalog=_Task13Catalog(tmp_path),
            profile_store=object(),
            candidate_registry=registry,
            coordinator=coordinator,
            online_tile_probe=delayed_probe,
            testing=True,
        )
        await user.open("/candidates/session-1")
        assert len(views) == 1
        view = views[0]

        before_enable = set(background_tasks.running_tasks)
        user.find(marker="candidate-layer-online").click()
        enable_tasks = await _new_nicegui_background_tasks(before_enable)
        assert await asyncio.to_thread(entered.wait, 1)
        before_disable = set(background_tasks.running_tasks)
        user.find(marker="candidate-layer-online").click()
        disable_tasks = await _new_nicegui_background_tasks(before_disable)
        await asyncio.wait_for(asyncio.gather(*disable_tasks), timeout=2)
        assert "online" not in view.controller.state.enabled_layers

        release.set()
        await asyncio.wait_for(asyncio.gather(*enable_tasks), timeout=2)

        assert "online" not in view.controller.state.enabled_layers
        assert not any(isinstance(layer, TileLayer) for layer in view.map_element.layers)
    finally:
        release.set()
        if user.client is not None and not user.client.is_deleted:
            user.client.delete()
        coordinator.shutdown()


async def test_candidate_online_probe_completion_after_page_cleanup_is_noop(
    user,
    tmp_path,
    monkeypatch,
):
    from dataclasses import replace

    from nicegui import background_tasks, core
    from nicegui.elements.leaflet.leaflet_layers import TileLayer

    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.gui.pages.candidates import CandidateSessionRegistry
    from lte_scenario_toolkit.jobs import JobCoordinator

    entered = Event()
    release = Event()
    unhandled = []
    views = []
    actual_render = app_module.render_candidate_page

    def delayed_probe():
        entered.set()
        assert release.wait(2)
        return True

    def capture_render(*args, **kwargs):
        view = actual_render(*args, **kwargs)
        views.append(view)
        return view

    monkeypatch.setattr(app_module, "render_candidate_page", capture_render)
    monkeypatch.setattr(core.app, "handle_exception", unhandled.append)
    registry = CandidateSessionRegistry()
    registry.add(
        replace(
            _task14_session(
                tmp_path,
                object(),
                map_bundle=_task14_map_bundle(tmp_path),
            ),
            scan_result=_task14_scan_result(),
        )
    )
    coordinator = JobCoordinator()
    try:
        app_module.create_app(
            catalog=_Task13Catalog(tmp_path),
            profile_store=object(),
            candidate_registry=registry,
            coordinator=coordinator,
            online_tile_probe=delayed_probe,
            testing=True,
        )
        client = await user.open("/candidates/session-1")
        assert len(views) == 1
        view = views[0]

        before_enable = set(background_tasks.running_tasks)
        user.find(marker="candidate-layer-online").click()
        enable_tasks = await _new_nicegui_background_tasks(before_enable)
        assert await asyncio.to_thread(entered.wait, 1)
        client.delete()
        release.set()
        await asyncio.wait_for(asyncio.gather(*enable_tasks), timeout=2)

        assert "online" not in view.controller.state.enabled_layers
        assert not any(isinstance(layer, TileLayer) for layer in view.map_element.layers)
        assert unhandled == []
    finally:
        release.set()
        if user.client is not None and not user.client.is_deleted:
            user.client.delete()
        coordinator.shutdown()


async def test_candidate_route_uses_one_offline_leaflet_and_preserves_it_on_view_switch(
    user, tmp_path
):
    from nicegui import ui
    from nicegui.elements.leaflet.leaflet_layers import TileLayer

    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.pages.candidates import CandidateSessionRegistry

    class Service:
        def scan(self, *args, **kwargs):
            return _task14_scan_result()

    registry = CandidateSessionRegistry()
    registry.add(
        _task14_session(
            tmp_path,
            Service(),
            map_bundle=_task14_map_bundle(tmp_path),
        )
    )
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=object(),
        candidate_registry=registry,
        online_tile_probe=lambda: (_ for _ in ()).throw(OSError("offline")),
        testing=True,
    )

    await user.open("/candidates/session-1")
    maps = user.find(kind=ui.leaflet).elements
    assert len(maps) == 1
    map_element = next(iter(maps))
    assert map_element._props["options"]["preferCanvas"] is True
    assert not any(isinstance(layer, TileLayer) for layer in map_element.layers)
    assert not any("tile.osm" in str(layer) for layer in map_element.layers)
    station_layers = [
        layer
        for layer in map_element.layers
        if getattr(layer, "name", None) == "stationDots"
    ]
    assert len(station_layers) == 1
    assert station_layers[0].args[1]["dotStyle"] == {
        "radius": 2.5,
        "stroke": True,
        "color": "#0b5f8a",
        "weight": 1,
        "opacity": 0.75,
        "fillColor": "#4f93c8",
        "fillOpacity": 0.55,
        "interactive": True,
        "bubblingMouseEvents": False,
    }
    assert map_element._props["additional-resources"] == [
        "/_lte_gui/assets/station-dots.js"
    ]

    for _ in range(20):
        rectangles = [
            layer
            for layer in map_element.layers
            if getattr(layer, "name", None) == "rectangle"
        ]
        if len(rectangles) == 2:
            break
        await asyncio.sleep(0.05)
    assert len(rectangles) == 2
    station_index = map_element.layers.index(station_layers[0])
    assert all(station_index < map_element.layers.index(layer) for layer in rectangles)
    assert {layer.args[1]["color"] for layer in rectangles} == {"#dc3f4f"}
    assert all(layer.args[1]["pane"] == "overlayPane" for layer in rectangles)

    user.find(marker="candidate-map").trigger(
        "mapClick", {"latlng": {"lat": 0.005, "lng": 0.005}}
    )
    await user.should_see("Grid ID 0")
    selected = next(layer for layer in rectangles if layer.args[1]["color"] == "#16a36a")
    assert selected.args[0][0][0] <= 0.005 <= selected.args[0][1][0]

    user.find(marker="candidate-map").trigger(
        "mapClick", {"latlng": {"lat": 0.005, "lng": 0.005}}
    )
    await user.should_see("Grid ID 4")
    user.find(marker="candidate-previous").click()
    await user.should_see("Grid ID 0")
    user.find(marker="candidate-next").click()
    await user.should_see("Grid ID 4")

    user.find(marker="candidate-layer-candidates").click()
    user.find(marker="candidate-layer-candidates").click()
    user.find(marker="candidate-layer-online").click()
    await asyncio.sleep(0.1)
    assert not any(isinstance(layer, TileLayer) for layer in map_element.layers)

    user.find(marker="candidate-view-filmstrip").click()
    await user.should_see("Candidate filmstrip")
    assert next(iter(user.find(kind=ui.leaflet).elements)) is map_element
    selected_card = next(iter(user.find(marker="candidate-card-4").elements))
    assert selected_card.props["role"] == "button"
    assert selected_card.props["aria-pressed"] == "true"
    user.find(marker="candidate-view-map").click()
    assert next(iter(user.find(kind=ui.leaflet).elements)) is map_element

    user.find(marker="candidate-confirm").click()
    assert registry.get("session-1").locked_candidate.flat_grid_id == 4


async def test_candidate_workbench_presents_progress_and_technical_details(
    user, tmp_path
):
    from dataclasses import replace

    from nicegui import ui

    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.pages.candidates import (
        CandidateScanProvenance,
        CandidateSessionRegistry,
    )
    from lte_scenario_toolkit.selection_service import DemStatistics

    class Service:
        @staticmethod
        def candidate_statistics(*_args, **_kwargs):
            return DemStatistics(1.0, 5.0, 3.0, 4.0, 10)

    registry = CandidateSessionRegistry()
    registry.add(
        replace(
            _task14_session(
                tmp_path,
                Service(),
                map_bundle=_task14_map_bundle(tmp_path),
            ),
            scan_result=_task14_scan_result(),
            scan_provenance=CandidateScanProvenance(8.2, "miss", "cache-key"),
        )
    )
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=object(),
        candidate_registry=registry,
        testing=True,
    )

    await user.open("/candidates/session-1")

    await user.should_see("100%")
    await user.should_see("10 / 10")
    await user.should_see("Fresh scan")
    await user.should_see("8.2 s elapsed")
    await user.should_not_see("Cache: miss")
    await user.should_see(marker="candidate-view-control")
    progress = next(iter(user.find(kind=ui.linear_progress).elements))
    assert tuple(progress.default_slot.children) == ()
    opacity = next(iter(user.find(marker="candidate-dem-opacity").elements))
    assert opacity.props["aria-label"] == "Terrain opacity"

    map_button = next(iter(user.find(marker="candidate-view-map").elements))
    filmstrip_button = next(
        iter(user.find(marker="candidate-view-filmstrip").elements)
    )
    assert map_button.props["aria-pressed"] == "true"
    assert filmstrip_button.props["aria-pressed"] == "false"
    assert "outline" not in map_button.props
    assert "unelevated" not in filmstrip_button.props

    user.find(marker="candidate-next").click()
    await user.should_see("Candidate 1")
    await user.should_see("1 station")
    await user.should_see("Min 1.0 m", retries=15)
    primary = next(iter(user.find(marker="candidate-primary-summary").elements))
    assert primary.text == "1 station"
    assert "Grid ID" not in primary.text
    await user.should_see(marker="candidate-technical-details")
    technical = next(iter(user.find(marker="candidate-technical-copy").elements))
    assert "Grid ID 0" in technical.text
    assert "Fast scan" in technical.text
    assert "10 pixels" in technical.text
    assert "cache-key" in technical.text
    assert "row-sweep-v1" in technical.text

    map_element = next(iter(user.find(kind=ui.leaflet).elements))
    map_identity = id(map_element)
    layer_identities = tuple(id(layer) for layer in map_element.layers)
    user.find(marker="candidate-view-filmstrip").click()
    assert map_button.props["aria-pressed"] == "false"
    assert filmstrip_button.props["aria-pressed"] == "true"
    assert "unelevated" not in map_button.props
    assert "outline" not in filmstrip_button.props
    user.find(marker="candidate-view-map").click()
    assert id(next(iter(user.find(kind=ui.leaflet).elements))) == map_identity
    assert tuple(id(layer) for layer in map_element.layers) == layer_identities


async def test_candidate_map_review_workspace_composes_map_inspector_dock_and_overflow(
    user, tmp_path
):
    from dataclasses import replace

    from nicegui import ElementFilter

    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.pages.candidates import CandidateSessionRegistry

    registry = CandidateSessionRegistry()
    registry.add(
        replace(
            _task14_session(
                tmp_path,
                object(),
                map_bundle=_task14_map_bundle(tmp_path),
            ),
            scan_result=_task14_scan_result(),
        )
    )
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=object(),
        candidate_registry=registry,
        testing=True,
    )

    await user.open("/candidates/session-1")

    expected_classes = {
        "candidate-review-workspace": "lte-candidate-review-workspace",
        "candidate-map-panel": "lte-candidate-map-panel",
        "candidate-inspector": "lte-candidate-review-inspector",
        "candidate-action-dock": "lte-candidate-action-dock",
        "candidate-overflow": "lte-overflow-menu__trigger",
    }
    for marker, class_name in expected_classes.items():
        element = next(iter(user.find(marker=marker).elements))
        assert class_name in element._classes

    for marker in (
        "candidate-counts",
        "candidate-layer-controls",
        "candidate-navigation",
        "candidate-map-status",
        "candidate-source-status",
    ):
        await user.should_see(marker=marker)

    with user.client:
        force_rescan = next(
            iter(ElementFilter(marker="candidate-force", only_visible=False))
        )
        dock = next(
            iter(ElementFilter(marker="candidate-action-dock", only_visible=False))
        )
    assert "lte-overflow-menu__item" in force_rescan._classes
    assert any(
        getattr(child, "_text", "") == "Force Rescan"
        for child in force_rescan.default_slot.children
    )
    assert any(ancestor.__class__.__name__ == "Menu" for ancestor in force_rescan.ancestors())

    primary_actions = [
        element
        for element in dock.descendants()
        if "lte-action--primary" in getattr(element, "_classes", [])
    ]
    assert len(primary_actions) == 1
    assert primary_actions[0].text == "Confirm Candidate"
    assert "outline" in next(iter(user.find(marker="candidate-start").elements)).props
    assert "flat" in next(iter(user.find(marker="candidate-cancel").elements)).props

    await user.should_see("Offline map assets active")
    await user.should_see("Map ready for candidate review")


@pytest.mark.parametrize(
    ("language", "map_name", "selected_name", "filmstrip_name", "figure_name"),
    (
        (
            "en",
            "Candidate selection map",
            "Terrain preview for Candidate 1",
            "Map preview for Candidate 1",
            "Terrain figure preview",
        ),
        (
            "zh-CN",
            "\u5019\u9009\u533a\u57df\u9009\u62e9\u5730\u56fe",
            "\u5019\u9009\u533a\u57df 1 \u7684\u5730\u5f62\u9884\u89c8",
            "\u5019\u9009\u533a\u57df 1 \u7684\u5730\u56fe\u9884\u89c8",
            "\u5730\u5f62\u56fe\u8868\u9884\u89c8",
        ),
    ),
)
async def test_candidate_and_figure_visual_surfaces_have_localized_accessible_names(
    user,
    tmp_path,
    language,
    map_name,
    selected_name,
    filmstrip_name,
    figure_name,
):
    from dataclasses import replace

    from nicegui import ElementFilter, ui

    from lte_scenario_toolkit.gui.i18n import Translator
    from lte_scenario_toolkit.gui.pages.candidates import render_candidate_page
    from lte_scenario_toolkit.gui.pages.figures import render_figures_page
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.selection_service import DemStatistics

    class Service:
        @staticmethod
        def candidate_statistics(*_args, **_kwargs):
            return DemStatistics(1.0, 5.0, 3.0, 4.0, 10)

    coordinator = JobCoordinator()
    session = replace(
        _task14_session(
            tmp_path,
            Service(),
            map_bundle=_task14_map_bundle(tmp_path),
        ),
        scan_result=_task14_scan_result(),
    )

    @ui.page(f"/accessible-visuals-{language}")
    def accessible_visuals():
        translator = Translator(language)
        render_candidate_page(
            ui,
            translator,
            session,
            coordinator,
            station_layer_resource="/_lte_gui/assets/station-dots.js",
            allow_rescan=False,
            auto_start=False,
        )
        render_figures_page(ui, translator, tmp_path, coordinator)

    try:
        await user.open(f"/accessible-visuals-{language}")
        candidate_map = next(iter(user.find(kind=ui.leaflet).elements))
        assert candidate_map.props["role"] == "region"
        assert candidate_map.props["aria-label"] == map_name

        user.find(marker="candidate-next").click()
        with user.client:
            selected_preview = next(
                iter(
                    ElementFilter(
                        marker="candidate-selected-preview",
                        only_visible=False,
                    )
                )
            )
        assert selected_preview.props["alt"] == selected_name

        user.find(marker="candidate-view-filmstrip").click()
        filmstrip_preview = next(
            iter(user.find(marker="candidate-thumbnail-0").elements)
        )
        assert filmstrip_preview.props["role"] == "img"
        assert filmstrip_preview.props["aria-label"] == filmstrip_name

        with user.client:
            figure_preview = next(
                iter(
                    ElementFilter(
                        marker="figure-preview-surface",
                        only_visible=False,
                    )
                )
            )
        assert figure_preview.props["alt"] == figure_name
    finally:
        coordinator.shutdown()


def test_candidate_workbench_css_is_bounded_and_mobile_safe():
    css = (
        ROOT / "src/lte_scenario_toolkit/gui/assets/app.css"
    ).read_text(encoding="utf-8")

    assert (
        "grid-template-columns: minmax(0, 1.55fr) minmax(280px, 0.85fr);"
        in css
    )
    assert "grid-auto-flow: column;" in css
    assert "grid-auto-columns: clamp(220px, 22vw, 280px);" in css
    assert "overflow-x: auto;" in css
    desktop = css.index(".lte-candidate-workspace")
    tablet = css.index("@media (max-width: 980px)")
    mobile = css.index("@media (max-width: 760px)")
    assert desktop < tablet < mobile
    assert "lte-candidate-review-workspace" not in css[tablet:mobile]
    mobile_block = css[mobile:]
    assert "grid-template-columns: minmax(0, 1fr)" in mobile_block
    assert "min-height: 420px" in mobile_block
    assert ".lte-candidate-action-dock" in mobile_block
    assert "flex-wrap: wrap" in mobile_block
    assert ".lte-filmstrip-grid { grid-template-columns" not in mobile_block
    card_rule = re.search(
        r"(?m)^\.lte-candidate-map-card\s*\{(?P<body>[^}]+)\}", css
    )
    assert card_rule is not None
    assert "min-height: calc(560px + 2 * var(--lte-space-4) + 2px);" in card_rule.group(
        "body"
    )
    assert "padding: 10px 12px calc(10px + env(safe-area-inset-bottom));" in css
    reduced_motion = css.index("@media (prefers-reduced-motion: reduce)")
    assert ".lte-candidate-page *" in css[reduced_motion:]


def test_candidate_workbench_has_explicit_390px_overflow_safeguards():
    css = (
        ROOT / "src/lte_scenario_toolkit/gui/assets/app.css"
    ).read_text(encoding="utf-8")

    narrow_start = css.index("@media (max-width: 390px)")
    next_media = css.find("@media", narrow_start + 1)
    narrow = css[narrow_start : None if next_media == -1 else next_media]

    for selector in (
        ".lte-web-selector-frame",
        ".lte-candidate-page",
        ".lte-candidate-review-workspace",
        ".lte-candidate-map-panel",
        ".lte-candidate-map-wrap",
        ".lte-candidate-review-inspector",
        ".lte-candidate-action-dock",
    ):
        assert selector in narrow
    assert "min-width: 0;" in narrow
    assert "max-width: 100%;" in narrow
    assert "overflow-x: clip;" in narrow
    assert ".lte-candidate-action-dock .lte-action-bar" in narrow
    assert "flex-wrap: wrap;" in narrow
    assert ".lte-candidate-id-input" in narrow
    assert "width: 100%;" in narrow
    assert ".lte-segmented-control .q-btn" in narrow
    assert "min-height: 44px;" in narrow
    assert ".lte-filmstrip-grid" in narrow
    assert "grid-auto-flow: column;" in narrow
    assert "overflow-x: auto;" in narrow
    assert ".lte-filmstrip-grid { grid-template-columns" not in narrow


def test_candidate_map_card_height_contract_covers_768_and_800_viewports():
    css = (ROOT / "src/lte_scenario_toolkit/gui/assets/app.css").read_text(
        encoding="utf-8"
    )
    required_content_height = 560 + (2 * 16) + 2
    assert (
        "min-height: calc(560px + 2 * var(--lte-space-4) + 2px);" in css
    )
    for viewport_height in (768, 800):
        bounded_height = min(viewport_height * 0.72, 780)
        assert bounded_height < required_content_height


async def test_candidate_page_can_disable_all_rescan_entrypoints(user, tmp_path):
    from dataclasses import replace

    from nicegui import ui

    from lte_scenario_toolkit.gui.i18n import Translator
    from lte_scenario_toolkit.gui.pages.candidates import render_candidate_page
    from lte_scenario_toolkit.jobs import JobCoordinator

    scan_calls = []

    class Service:
        def scan(self, *args, **kwargs):
            scan_calls.append((args, kwargs))
            return _task14_scan_result()

    coordinator = JobCoordinator()
    session = replace(
        _task14_session(
            tmp_path,
            Service(),
            map_bundle=_task14_map_bundle(tmp_path),
        ),
        scan_result=_task14_scan_result(),
    )

    @ui.page("/candidate-without-rescan")
    def candidate_without_rescan():
        render_candidate_page(
            ui,
            Translator("en"),
            session,
            coordinator,
            station_layer_resource="/_lte_gui/assets/station-dots.js",
            allow_rescan=False,
            auto_start=False,
        )

    try:
        await user.open("/candidate-without-rescan")
        for marker in ("candidate-start", "candidate-cancel", "candidate-force"):
            await user.should_not_see(marker=marker)
        await asyncio.sleep(0.05)
        assert scan_calls == []
    finally:
        coordinator.shutdown()


async def test_direct_candidate_route_without_session_fails_closed_offline(user, tmp_path):
    from lte_scenario_toolkit.gui.app import create_app

    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=object(),
        testing=True,
    )

    await user.open("/candidates/missing")

    await user.should_see("Candidate session unavailable")
    assert not (tmp_path / ".lte-data").exists()


async def test_candidate_style_and_selected_preview_reuse_short_static_asset_urls(
    user, tmp_path
):
    from nicegui import ui
    from nicegui.elements.leaflet.leaflet_layers import ImageOverlay

    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.pages.candidates import CandidateSessionRegistry
    from lte_scenario_toolkit.map_assets import MapAsset
    from lte_scenario_toolkit.selection_service import DemStatistics

    cache_root = tmp_path / ".lte-data" / "cache" / "maps"
    cache_root.mkdir(parents=True, exist_ok=True)
    selected_png = cache_root / "selected.png"
    selected_png.write_bytes(b"png")
    hillshade_png = cache_root / "hillshade.png"
    hillshade_png.write_bytes(b"png")

    class Service:
        def scan(self, *args, **kwargs):
            return _task14_scan_result()

        def candidate_statistics(self, *args, **kwargs):
            return DemStatistics(1.0, 5.0, 3.0, 4.0, 10)

    registry = CandidateSessionRegistry()
    registry.add(
        _task14_session(
            tmp_path,
            Service(),
            map_bundle=_task14_map_bundle(tmp_path),
        )
    )
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=object(),
        candidate_registry=registry,
        candidate_overlay_asset_builder=lambda _session, _candidate, style: MapAsset(
            selected_png,
            (0.0, 0.0, 1.0, 1.0),
            "EPSG:4326",
            style,
        ),
        candidate_style_asset_builder=lambda _session, style: MapAsset(
            hillshade_png,
            (-1.0, -1.0, 2.0, 2.0),
            "EPSG:4326",
            style,
        ),
        testing=True,
    )
    await user.open("/candidates/session-1")
    map_element = next(iter(user.find(kind=ui.leaflet).elements))
    for _ in range(20):
        if any(getattr(layer, "name", None) == "rectangle" for layer in map_element.layers):
            break
        await asyncio.sleep(0.05)
    user.find(marker="candidate-next").click()
    await user.should_see(marker="candidate-selected-preview", retries=15)
    preview = next(iter(user.find(marker="candidate-selected-preview").elements))
    assert preview.props["src"].startswith("/_candidate_assets/")
    assert not preview.props["src"].startswith("data:")

    overlay = next(layer for layer in map_element.layers if isinstance(layer, ImageOverlay))
    initial_overlay_url = overlay.url
    user.find(marker="candidate-dem-style").click()
    user.find("Hillshade").click()
    for _ in range(20):
        if overlay.url != initial_overlay_url:
            break
        await asyncio.sleep(0.05)
    assert overlay.url.startswith("/_candidate_assets/")
    assert overlay.url != initial_overlay_url
    assert not overlay.url.startswith("data:")

    user.find(marker="candidate-view-filmstrip").click()
    unselected = next(iter(user.find(marker="candidate-thumbnail-4").elements))
    assert "data:image" not in str(unselected._style)
    assert "/_candidate_assets/" in str(unselected._style)


async def test_candidate_page_retries_statistics_after_external_shared_job(user, tmp_path):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.pages.candidates import CandidateSessionRegistry
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.selection_service import DemStatistics

    class Service:
        def candidate_statistics(self, *args, **kwargs):
            return DemStatistics(1.0, 5.0, 3.0, 4.0, 10)

    coordinator = JobCoordinator()
    external = coordinator.start("external.validation")
    registry = CandidateSessionRegistry()
    registry.add(
        replace(
            _task14_session(
                tmp_path,
                Service(),
                map_bundle=_task14_map_bundle(tmp_path),
            ),
            scan_result=_task14_scan_result(),
        )
    )
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=object(),
        candidate_registry=registry,
        coordinator=coordinator,
        testing=True,
    )
    try:
        await user.open("/candidates/session-1")
        start = next(iter(user.find(marker="candidate-start").elements))
        force = next(iter(user.find(marker="candidate-force").elements))
        assert start.enabled is False
        assert force.enabled is False

        user.find(marker="candidate-next").click()
        await user.should_see("Statistics load after final scan selection.")
        assert coordinator.finish(external.job_id) is True

        await user.should_see("Min 1.0 m", retries=15)
        assert start.enabled is True
        assert force.enabled is True
    finally:
        coordinator.shutdown()


async def test_configure_start_passes_exact_frozen_preflight_to_opaque_session(
    user, tmp_path
):
    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.pages.candidates import CandidateSessionRegistry

    calls = []

    class EmptyStore:
        def discover(self, scenario_id):
            return []

    class Service:
        def preflight(self, snapshot, output_root):
            preflight = SimpleNamespace(profile=snapshot, output_root=output_root)
            calls.append((snapshot, preflight))
            return preflight

        def scan(self, *args, **kwargs):
            return _task14_scan_result()

    service = Service()
    registry = CandidateSessionRegistry()
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=EmptyStore(),
        selection_service_factory=lambda _catalog: service,
        candidate_registry=registry,
        candidate_bundle_builder=lambda _session, _assets: _task14_map_bundle(tmp_path),
        testing=True,
    )
    await user.open("/configure/without-default")

    user.find(marker="profile-start-scan").click()
    for _ in range(20):
        if user.back_history and "/candidates/" in user.back_history[-1]:
            break
        await asyncio.sleep(0.05)

    route = user.back_history[-1]
    assert route.startswith("/candidates/")
    session_id = route.rsplit("/", 1)[-1]
    assert len(session_id) >= 32
    session = registry.get(session_id)
    assert session is not None
    assert session.profile_snapshot is calls[0][0]
    assert session.preflight is calls[0][1]
    assert str(tmp_path) not in route


def _task15_locked_session(tmp_path, service, *, session_id="generate-session"):
    from dataclasses import replace

    result = _task14_scan_result()
    session = _task14_session(tmp_path, service, session_id=session_id)
    return replace(
        session,
        preflight=SimpleNamespace(
            profile=session.profile_snapshot,
            output_root=tmp_path / "results",
        ),
        scan_result=result,
        confirmed_flat_grid_id=4,
        locked_candidate=result.candidates[1],
    )


def test_generate_model_requires_one_locked_candidate():
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages.generate import generation_model

    with pytest.raises(ValueError, match="locked candidate"):
        generation_model(SimpleNamespace(locked_candidate=None))

    session = _task15_locked_session(Path.cwd(), object())
    with pytest.raises(ValueError, match="completed"):
        generation_model(
            replace(
                session,
                scan_result=replace(session.scan_result, completed=False),
            )
        )
    with pytest.raises(ValueError, match="exactly one"):
        generation_model(
            replace(
                session,
                scan_result=replace(
                    session.scan_result,
                    candidates=(session.locked_candidate, session.locked_candidate),
                ),
            )
        )


def test_generation_model_is_read_only_and_requires_one_artifact(tmp_path):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages.generate import generation_model

    output_root = tmp_path / "results"
    session = _task15_locked_session(tmp_path, object())
    session = replace(
        session,
        preflight=SimpleNamespace(
            profile=session.profile_snapshot,
            output_root=output_root,
        ),
    )

    model = generation_model(session)

    assert model.scenario_id == session.profile_snapshot.scenario_id
    assert model.profile_id == session.profile_snapshot.profile_id
    assert model.candidate is session.locked_candidate
    assert model.output_root == output_root.resolve()
    assert model.can_generate is True
    assert not output_root.exists()
    empty = model.with_artifact("csv", False)
    for token in tuple(empty.selected_artifacts):
        empty = empty.with_artifact(token, False)
    assert empty.can_generate is False
    with pytest.raises(ValueError, match="artifact"):
        empty.require_artifacts()
    with pytest.raises(ValueError, match="artifact"):
        model.with_artifact("pdf", True)


def test_generation_controller_publishes_partial_record_and_remembers_root(tmp_path):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages.generate import GenerationController
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.run_service import RunService

    output_root = tmp_path / "results"
    remembered = []

    class Service:
        def export(
            self,
            preflight,
            scan_result,
            candidate,
            *,
            output_root,
            artifacts,
            entrypoint,
        ):
            assert candidate is session.locked_candidate
            assert scan_result is session.scan_result
            assert preflight is session.preflight
            assert tuple(artifacts) == ("csv", "terrain_png")
            assert tuple(entrypoint) == ("lte-gui", "generate")
            run_service = RunService(output_root)
            run = run_service.begin("ready-city", "default")
            (run.path / "scenario.csv").write_text("X,Y\n1,2\n", encoding="utf-8")
            return run_service.publish(
                run,
                status="partial",
                artifacts=["scenario.csv"],
                metadata={
                    "run_kind": "selection",
                    "requested_artifacts": list(artifacts),
                    "artifact_paths": {"csv": "scenario.csv"},
                    "candidate": {
                        "flat_grid_id": candidate.flat_grid_id,
                        "point_count": candidate.point_count,
                        "center_x": candidate.center_x,
                        "center_y": candidate.center_y,
                    },
                },
                errors=[{"artifact": "terrain_png", "code": "png.failed"}],
            )

    profile = _task13_profile(tmp_path)
    preflight = SimpleNamespace(profile=profile, output_root=output_root)
    session = _task15_locked_session(tmp_path, Service())
    session = replace(
        session,
        profile_snapshot=profile,
        preflight=preflight,
    )
    coordinator = JobCoordinator()
    controller = GenerationController(
        session,
        coordinator,
        on_published=lambda path: remembered.append(path),
    )
    try:
        assert not output_root.exists()
        job = controller.start({"csv", "terrain_png"})
        assert job.future is not None
        job.future.result(timeout=2)
        state = controller.drain(job)

        assert state.phase == "partial"
        assert state.run_path is not None and state.run_path.is_dir()
        assert state.errors[0]["code"] == "png.failed"
        assert state.artifact_status("csv") == "published"
        assert state.artifact_status("terrain_png") == "failed"
        assert remembered == [state.run_path]
        assert coordinator.snapshot().active is False
    finally:
        controller.close()
        coordinator.shutdown()


async def test_generation_page_uses_semantic_artifact_rows_and_live_partial_state(
    user,
    tmp_path,
):
    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.pages.candidates import CandidateSessionRegistry
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.run_service import RunService

    entered = Event()
    release = Event()

    class Service:
        def export(
            self,
            preflight,
            scan_result,
            candidate,
            *,
            output_root,
            artifacts,
            entrypoint,
        ):
            del scan_result, entrypoint
            entered.set()
            assert release.wait(3)
            service = RunService(output_root)
            run = service.begin(
                preflight.profile.scenario_id,
                preflight.profile.profile_id,
            )
            artifact_paths = {
                "csv": "scenario.csv",
                "preview_png": "preview.png",
                "terrain_eps": "terrain.eps",
                "terrain_html": "terrain.html",
            }
            for token, name in artifact_paths.items():
                (run.path / name).write_bytes(f"{token}\n".encode("ascii"))
            return service.publish(
                run,
                status="partial",
                artifacts=list(artifact_paths.values()),
                metadata={
                    "run_kind": "selection",
                    "requested_artifacts": list(artifacts),
                    "artifact_paths": artifact_paths,
                    "candidate": {
                        "flat_grid_id": candidate.flat_grid_id,
                        "point_count": candidate.point_count,
                        "center_x": candidate.center_x,
                        "center_y": candidate.center_y,
                    },
                },
                errors=[
                    {
                        "artifact": "terrain_png",
                        "code": "terrain.renderer_failed",
                        "message": "renderer traceback terrain_png",
                    }
                ],
            )

    coordinator = JobCoordinator()
    registry = CandidateSessionRegistry()
    registry.add(_task15_locked_session(tmp_path, Service(), session_id="semantic-run"))
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=object(),
        candidate_registry=registry,
        coordinator=coordinator,
        testing=True,
    )

    try:
        await user.open("/generate/semantic-run")
        stepper = next(iter(user.find(marker="generation-workflow-stepper").elements))
        inputs_step = next(iter(user.find(marker="generation-step-inputs").elements))
        generate_step = next(
            iter(user.find(marker="generation-step-generate").elements)
        )
        artifacts_step = next(
            iter(user.find(marker="generation-step-artifacts").elements)
        )
        assert stepper.props["role"] == "list"
        assert stepper.props["aria-label"] == "Generation progress"
        assert inputs_step.props["role"] == "listitem"
        assert inputs_step.props["aria-current"] == "step"
        assert inputs_step.props["aria-label"] == "Inputs: ready"
        assert "aria-current" not in generate_step.props
        expected = {
            "csv": ("Scenario CSV", "Station records", "CSV"),
            "preview_png": ("2D preview", "plan-view image", "PNG"),
            "terrain_png": ("3D terrain", "raster terrain", "PNG"),
            "terrain_eps": ("3D terrain", "vector terrain", "EPS"),
            "terrain_html": ("Interactive terrain", "interactive terrain", "HTML"),
        }
        await user.should_see(marker="generation-phase", content="Ready to generate")
        for token, phrases in expected.items():
            await user.should_see(marker=f"generation-artifact-{token}")
            await user.should_see(marker=f"generation-artifact-status-{token}")
            for phrase in phrases:
                await user.should_see(phrase)
            await user.should_not_see(token)

        user.find(marker="generation-submit").click()
        assert await asyncio.to_thread(entered.wait, 2)
        await user.should_see(marker="generation-phase", content="Generating artifacts")
        await user.should_not_see(marker="generation-cancel")
        assert generate_step.props["aria-current"] == "step"
        assert generate_step.props["aria-label"] == "Generate: in progress"
        assert "aria-current" not in inputs_step.props
        await user.should_see(
            marker="shell-job-indicator",
            content="Scenario generation",
            retries=12,
        )
        assert all(
            not next(
                iter(user.find(marker=f"generation-artifact-select-{token}").elements)
            ).enabled
            for token in expected
        )
        assert not next(iter(user.find(marker="generation-submit").elements)).enabled
        csv_checkbox = next(
            iter(user.find(marker="generation-artifact-select-csv").elements)
        )
        csv_checkbox.set_value(False)
        await asyncio.sleep(0.1)
        assert csv_checkbox.value is True
        assert csv_checkbox.enabled is False
        await user.should_see(
            marker="generation-artifact-status-csv",
            content="Pending",
        )

        release.set()
        await user.should_see(marker="generation-phase", content="Some artifacts failed", retries=30)
        await user.should_see(
            marker="generation-artifact-status-csv",
            content="Published",
        )
        await user.should_see(
            marker="generation-artifact-status-terrain_png",
            content="Failed",
        )
        assert artifacts_step.props["aria-current"] == "step"
        assert artifacts_step.props["aria-label"] == "Artifacts: partial output"
        assert "aria-current" not in generate_step.props
        assert user.back_history[-1] != "/history"
        await user.should_see(
            marker="generation-primary-error",
            content="Some artifacts could not be generated",
        )
        assert registry.get("semantic-run") is None
        technical = next(iter(user.find(marker="generation-technical-copy").elements))
        assert "terrain.renderer_failed" in technical.content
        assert "terrain_png" in technical.content
        assert "traceback" in technical.content
        primary = next(iter(user.find(marker="generation-primary-error").elements))
        assert "terrain.renderer_failed" not in primary.text
        assert "terrain_png" not in primary.text
    finally:
        release.set()
        coordinator.shutdown()


async def test_generation_selection_state_gates_submit_and_updates_status(
    user,
    tmp_path,
):
    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.pages.candidates import CandidateSessionRegistry
    from lte_scenario_toolkit.gui.pages.generate import ARTIFACT_ORDER
    from lte_scenario_toolkit.jobs import JobCoordinator

    coordinator = JobCoordinator()
    registry = CandidateSessionRegistry()
    registry.add(_task15_locked_session(tmp_path, object(), session_id="selection-state"))
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=object(),
        candidate_registry=registry,
        coordinator=coordinator,
        testing=True,
    )

    try:
        await user.open("/generate/selection-state")
        for token in ARTIFACT_ORDER:
            user.find(marker=f"generation-artifact-select-{token}").click()

        await user.should_see(
            marker="generation-selection-guidance",
            content="Select at least one artifact",
            retries=12,
        )
        assert not next(iter(user.find(marker="generation-submit").elements)).enabled
        for token in ARTIFACT_ORDER:
            await user.should_see(
                marker=f"generation-artifact-status-{token}",
                content="Not requested",
            )

        user.find(marker="generation-artifact-select-csv").click()
        await user.should_not_see(marker="generation-selection-guidance", retries=12)
        assert next(iter(user.find(marker="generation-submit").elements)).enabled
        await user.should_see(
            marker="generation-artifact-status-csv",
            content="Pending",
        )
        await user.should_see(
            marker="generation-artifact-status-terrain_png",
            content="Not requested",
        )
    finally:
        coordinator.shutdown()


@pytest.mark.parametrize("result_kind", ["malformed", "escaped"])
def test_generation_controller_rejects_untrusted_publish_result(tmp_path, result_kind):
    from lte_scenario_toolkit.gui.pages.generate import GenerationController
    from lte_scenario_toolkit.jobs import JobCoordinator

    output_root = tmp_path / "results"

    class Service:
        def export(self, *args, **kwargs):
            if result_kind == "escaped":
                published = tmp_path / "escaped"
                published.mkdir()
            else:
                published = output_root / "city" / "default" / "run"
                published.mkdir(parents=True)
            (published / "run.json").write_text(
                "not-json" if result_kind == "malformed" else json.dumps({}),
                encoding="utf-8",
            )
            return published

    session = _task15_locked_session(tmp_path, Service())
    controller = GenerationController(session, JobCoordinator())
    try:
        job = controller.start({"csv"})
        assert job.future is not None
        job.future.result(timeout=2)
        state = controller.drain(job)

        assert state.phase == "error"
        assert state.run_path is None
        assert controller.coordinator.snapshot().active is False
    finally:
        controller.close()
        controller.coordinator.shutdown()


def test_generation_controller_preserves_the_shared_busy_slot(tmp_path):
    from lte_scenario_toolkit.gui.pages.generate import GenerationController
    from lte_scenario_toolkit.jobs import JobBusyError, JobCoordinator

    coordinator = JobCoordinator()
    active = coordinator.start("scan")
    controller = GenerationController(
        _task15_locked_session(tmp_path, object()),
        coordinator,
    )
    try:
        with pytest.raises(JobBusyError):
            controller.start({"csv"})
        assert coordinator.snapshot().job_id == active.job_id
    finally:
        assert coordinator.finish(active.job_id) is True
        controller.close()
        coordinator.shutdown()


def test_generation_done_callback_releases_slot_after_page_cleanup(tmp_path):
    from threading import Event

    from lte_scenario_toolkit.gui.pages.generate import GenerationController
    from lte_scenario_toolkit.jobs import JobCoordinator

    entered = Event()
    release = Event()
    published = tmp_path / "published"

    class Service:
        def export(self, *args, **kwargs):
            entered.set()
            assert release.wait(2)
            published.mkdir()
            (published / "run.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "artifacts": ["scenario.csv"],
                        "metadata": {
                            "requested_artifacts": ["csv"],
                            "artifact_paths": {"csv": "scenario.csv"},
                        },
                        "errors": [],
                    }
                ),
                encoding="utf-8",
            )
            return published

    controller = GenerationController(
        _task15_locked_session(tmp_path, Service()),
        JobCoordinator(),
    )
    try:
        job = controller.start({"csv"})
        assert entered.wait(2)
        controller.close()
        release.set()
        assert job.future is not None
        job.future.result(timeout=2)
        for _ in range(20):
            if not controller.coordinator.snapshot().active:
                break
            Event().wait(0.01)
        assert controller.coordinator.snapshot().active is False
    finally:
        controller.coordinator.shutdown()


def test_generation_settings_failure_is_a_warning_and_generation_is_one_shot(
    tmp_path,
):
    from lte_scenario_toolkit.gui.pages.generate import GenerationController
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.run_service import RunService

    class Service:
        def export(
            self,
            preflight,
            scan_result,
            candidate,
            *,
            output_root,
            artifacts,
            entrypoint,
        ):
            run_service = RunService(output_root)
            run = run_service.begin("ready-city", "default")
            (run.path / "scenario.csv").write_text("ok\n", encoding="utf-8")
            return run_service.publish(
                run,
                status="completed",
                artifacts=["scenario.csv"],
                metadata={
                    "run_kind": "selection",
                    "requested_artifacts": list(artifacts),
                    "artifact_paths": {"csv": "scenario.csv"},
                    "candidate": {
                        "flat_grid_id": candidate.flat_grid_id,
                        "point_count": candidate.point_count,
                        "center_x": candidate.center_x,
                        "center_y": candidate.center_y,
                    },
                    "entrypoint": list(entrypoint),
                },
            )

    coordinator = JobCoordinator()
    controller = GenerationController(
        _task15_locked_session(tmp_path, Service()),
        coordinator,
        on_published=lambda _path: (_ for _ in ()).throw(
            OSError("settings unavailable")
        ),
    )
    try:
        job = controller.start(("csv",))
        assert job.future is not None
        job.future.result(timeout=2)
        state = controller.drain(job)

        assert state.phase == "completed"
        assert state.warnings == (
            {
                "code": "generation.settings_failed",
                "message": "settings unavailable",
            },
        )
        with pytest.raises(RuntimeError, match="already"):
            controller.start(("csv",))
        controller.close()
        with pytest.raises(RuntimeError, match="closed"):
            controller.start(("csv",))
    finally:
        coordinator.shutdown()


@pytest.mark.parametrize(
    ("phase", "can_open_figures", "expected"),
    [
        ("ready", True, (("generate", "primary"),)),
        ("running", True, ()),
        (
            "completed",
            True,
            (("open_figures", "primary"), ("open_history", "secondary")),
        ),
        ("partial", True, (("open_history", "secondary"), ("inspect", "tertiary"))),
        ("error", True, (("retry", "secondary"), ("inspect", "tertiary"))),
    ],
)
def test_generation_workspace_action_roles_follow_the_run_state(
    phase, can_open_figures, expected
):
    from lte_scenario_toolkit.gui.pages.generate import generation_action_roles

    assert generation_action_roles(phase, can_open_figures=can_open_figures) == expected


@pytest.mark.parametrize(
    ("phase", "current"),
    [
        ("ready", "inputs"),
        ("running", "generate"),
        ("completed", "artifacts"),
        ("partial", "artifacts"),
        ("error", "artifacts"),
    ],
)
def test_generation_stepper_assigns_one_current_step_for_each_run_state(
    phase, current
):
    from lte_scenario_toolkit.gui.pages.generate import generation_current_step

    assert generation_current_step(phase) == current


@pytest.mark.parametrize(
    ("language", "workflow_label", "state_label"),
    [
        ("en", "Generation progress", "Generate: in progress"),
        ("zh-CN", "\u751f\u6210\u8fdb\u5ea6", "\u751f\u6210\uff1a\u6b63\u5728\u8fdb\u884c"),
    ],
)
def test_generation_stepper_uses_localized_progress_and_state_labels(
    language, workflow_label, state_label
):
    from lte_scenario_toolkit.gui.i18n import Translator

    translator = Translator(language)

    assert translator.text("generate.workflow.aria_label") == workflow_label
    assert (
        translator.text(
            "generate.workflow.state.running",
            step=translator.text("generate.step.generate"),
        )
        == state_label
    )


def test_generation_workspace_uses_progressive_structure_and_compact_css():
    page = (
        ROOT / "src/lte_scenario_toolkit/gui/pages/generate.py"
    ).read_text(encoding="utf-8")
    stylesheet = (ROOT / "src/lte_scenario_toolkit/gui/assets/app.css").read_text(
        encoding="utf-8"
    )
    translations = (ROOT / "src/lte_scenario_toolkit/gui/i18n.py").read_text(
        encoding="utf-8"
    )

    for class_name in (
        "lte-generate-stepper",
        "lte-generate-summary",
        "lte-generation-artifact-row",
        "lte-generate-action-dock",
    ):
        assert class_name in page
        assert class_name in stylesheet
    for key in (
        "generate.step.inputs",
        "generate.step.generate",
        "generate.step.artifacts",
        "action.inspect",
        "generate.action.retry",
    ):
        assert key in translations
    assert "@media (max-width: 760px)" in stylesheet
    assert "@media (max-width: 390px)" in stylesheet
    assert "generation-cancel" not in page
    assert "generate.action.cancel" not in translations
    assert "generate.phase.cancelling" not in translations


def test_generation_controller_exposes_no_uninterruptible_cancel_promise(tmp_path):
    from lte_scenario_toolkit.gui.pages.generate import GenerationController
    from lte_scenario_toolkit.jobs import JobCoordinator

    coordinator = JobCoordinator()
    controller = GenerationController(_task15_locked_session(tmp_path, object()), coordinator)
    try:
        assert not hasattr(controller, "cancel")
    finally:
        controller.close()
        coordinator.shutdown()


def test_history_model_includes_partial_and_parent_runs(tmp_path):
    from lte_scenario_toolkit.gui.pages.history import history_rows
    from lte_scenario_toolkit.run_service import RunService

    service = RunService(tmp_path)
    parent = service.begin("city", "default", created_at="2026-07-16T10:00:00Z")
    (parent.path / "scenario.csv").write_text("X,Y\n1,2\n", encoding="utf-8")
    service.publish(parent, status="completed", artifacts=["scenario.csv"])
    child = service.begin(
        "city",
        "default",
        created_at="2026-07-16T10:00:01Z",
        parent_run_id=parent.run_id,
    )
    (child.path / "terrain.png").write_bytes(b"png")
    service.publish(
        child,
        status="partial",
        artifacts=["terrain.png"],
        errors=[{"code": "html.failed"}],
    )

    rows = history_rows(service)

    assert {row.status for row in rows} == {"completed", "partial"}
    derived = next(row for row in rows if row.parent_run_id)
    assert derived.can_open_figures is True
    assert derived.figure_source_path == next(
        row.path for row in rows if row.run_id == parent.run_id
    )


def _task6_history_family(tmp_path):
    """Build one published parent/child family for the History trash tests."""

    from lte_scenario_toolkit.gui.pages.history import history_rows
    from lte_scenario_toolkit.run_service import RunService

    root = tmp_path / "results"
    service = RunService(root)
    parent = service.begin("city", "default", created_at="2026-07-16T10:00:00Z")
    (parent.path / "scenario.csv").write_text("X,Y\n1,2\n", encoding="utf-8")
    service.publish(parent, status="completed", artifacts=["scenario.csv"])
    child = service.begin(
        "city",
        "default",
        created_at="2026-07-16T10:00:01Z",
        parent_run_id=parent.run_id,
    )
    (child.path / "terrain.png").write_bytes(b"figure")
    service.publish(child, status="completed", artifacts=["terrain.png"])
    rows = history_rows(service)
    return root, service, parent, child, rows


def test_published_parent_offers_one_whole_family_move_action(tmp_path):
    from lte_scenario_toolkit.gui.pages.history import (
        HistoryAction,
        build_history_trash_plan,
    )
    from lte_scenario_toolkit.run_trash import RunUsageLeaseRegistry, TrashManager

    root, _service, parent, child, rows = _task6_history_family(tmp_path)
    row = next(item for item in rows if item.run_id == parent.run_id)
    manager = TrashManager(lambda: (root,), RunUsageLeaseRegistry())

    view = build_history_trash_plan(row, manager)

    assert HistoryAction.MOVE_TO_TRASH in row.available_actions
    assert view.run_count == 2
    assert view.run_ids == (parent.run_id, child.run_id)
    assert view.has_descendants is True
    assert view.direct_descendant_count == 1
    assert view.indirect_descendant_count == 0
    assert "orphan" not in {action.value for action in view.actions}


def test_pending_selection_does_not_render_move_to_trash(tmp_path):
    from types import SimpleNamespace

    from lte_scenario_toolkit.gui.pages.history import render_history_content

    pending_session = SimpleNamespace(
        session_id="pending-1",
        profile_snapshot=SimpleNamespace(
            scenario_id="city",
            profile_id="default",
        ),
        locked_candidate=SimpleNamespace(flat_grid_id=7, point_count=3),
    )

    class Element:
        def __init__(self, text=""):
            self.text = text

        def classes(self, *_args):
            return self

        def mark(self, marker):
            markers.add(marker)
            return self

        def props(self, *_args):
            return self

        def clear(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Ui:
        def __init__(self):
            self.navigate = type("Navigate", (), {"reload": lambda: None})()

        def __getattr__(self, name):
            if name == "notify":
                return lambda *_args, **_kwargs: None
            return lambda *args, **kwargs: Element(str(args[0]) if args else "")

    markers = set()
    ui = Ui()
    translator = _gui_module("i18n").Translator("en")
    history = _gui_module("pages.history")
    snapshot = history.HistorySnapshot(
        (),
        (),
        (),
        (tmp_path / ".lte-data/cache/history-index.json").absolute(),
    )
    render_history_content(ui, translator, Element(), snapshot, pending_selections=(pending_session,))

    assert "history-pending-section" in markers
    assert not any(marker.startswith("history-trash-move-pending-") for marker in markers)


def test_move_dialog_lists_every_affected_run_and_requires_fresh_confirmation(tmp_path):
    from lte_scenario_toolkit.gui.i18n import Translator
    from lte_scenario_toolkit.gui.pages.history import (
        HistoryTrashPlan,
        TrashImpactRow,
        render_move_to_trash_dialog,
    )

    class Element:
        def __init__(self, text=""):
            self.text = text

        def classes(self, *_args):
            return self

        def mark(self, marker):
            markers.add(marker)
            return self

        def props(self, *_args):
            return self

        def set_enabled(self, enabled):
            enabled_values.append(enabled)
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Ui:
        def __getattr__(self, name):
            return lambda *args, **kwargs: Element(" ".join(str(arg) for arg in args))

    root = tmp_path / "results"
    reference = object()
    rows = tuple(
        TrashImpactRow(
            run_id=f"{index:032x}",
            scenario_id="city",
            profile_id="default",
            local_created_at="2026-07-16T10:00:00+00:00",
            run_kind="selection",
            status="completed",
            artifact_count=1,
            size_bytes=1024,
            root=root,
            root_digest="a" * 64,
        )
        for index in range(1, 4)
    )
    plan = HistoryTrashPlan(
        reference=reference,
        selected_identity=object(),
        run_ids=tuple(row.run_id for row in rows),
        run_count=3,
        total_size_bytes=4096,
        roots=(root,),
        has_descendants=True,
        direct_descendant_count=2,
        indirect_descendant_count=0,
        graph_fingerprint="b" * 64,
        plan_fingerprint="c" * 64,
        impact_rows=rows,
        actions=(),
        trash_plan=object(),
    )
    markers = set()
    enabled_values = []
    render_move_to_trash_dialog(Ui(), Translator("en"), plan, on_confirm=lambda: None)

    assert "history-trash-impact" in markers
    assert "history-trash-confirm" in markers
    assert "history-trash-orphan-option" not in markers


@pytest.mark.parametrize(
    ("state", "expected_actions"),
    [
        ("trashed", ("restore", "purge")),
        ("recovery_required", ("recover",)),
        ("purge_failed", ("purge",)),
    ],
)
def test_trash_card_actions_follow_transaction_state(state, expected_actions):
    from lte_scenario_toolkit.gui.pages.history import (
        TrashCard,
        trash_card_actions,
    )

    card = TrashCard(
        transaction_id="a" * 32,
        state=state,
        deleted_at="2026-07-16T10:00:00Z",
        run_count=1,
        size_bytes=10,
        artifact_count=1,
        scenario_profiles=("city / default",),
        roots=(),
        blockers=(),
        enabled_actions=trash_card_actions(state),
        transaction=object(),
    )

    assert tuple(action.value for action in card.available_actions) == expected_actions


def test_history_published_card_renders_move_to_trash_overflow(tmp_path):
    from lte_scenario_toolkit.gui.i18n import Translator
    from lte_scenario_toolkit.gui.pages.history import (
        HistorySnapshot,
        render_history_content,
    )
    from lte_scenario_toolkit.run_trash import RunUsageLeaseRegistry, TrashManager

    root, _service, parent, _child, rows = _task6_history_family(tmp_path)
    row = next(item for item in rows if item.run_id == parent.run_id)
    markers = set()

    class Element:
        def classes(self, *_args):
            return self

        def props(self, *_args):
            return self

        def mark(self, marker):
            markers.add(marker)
            return self

        def set_enabled(self, *_args):
            return self

        def clear(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Ui:
        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: Element()

    manager = TrashManager(lambda: (root,), RunUsageLeaseRegistry())
    render_history_content(
        Ui(),
        Translator("en"),
        Element(),
        HistorySnapshot((root.resolve(),), (row,), (), tmp_path / "index"),
        trash_manager=manager,
        on_move_to_trash=lambda _plan: None,
    )

    assert f"history-trash-overflow-{parent.run_id}" in markers
    assert f"history-trash-move-{parent.run_id}" in markers


def test_trash_card_restore_blocker_keeps_purge_enabled():
    from lte_scenario_toolkit.gui.pages.history import (
        TrashAction,
        TrashCard,
    )

    card = TrashCard(
        transaction_id="a" * 32,
        state="trashed",
        deleted_at="2026-07-16T10:00:00Z",
        run_count=2,
        size_bytes=4096,
        artifact_count=3,
        scenario_profiles=("city / default",),
        roots=(),
        blockers=("trash.restore.destination_occupied",),
        enabled_actions=(TrashAction.PURGE,),
    )

    assert card.available_actions == (TrashAction.RESTORE, TrashAction.PURGE)
    assert card.enabled_actions == (TrashAction.PURGE,)


def test_permanent_delete_requires_exact_transaction_prefix():
    from lte_scenario_toolkit.gui.pages.history import (
        TrashCard,
        permanent_delete_matches,
    )

    card = TrashCard(
        transaction_id="abcdef0123456789abcdef0123456789",
        state="trashed",
        deleted_at="2026-07-16T10:00:00Z",
        run_count=2,
        size_bytes=4096,
        artifact_count=3,
        scenario_profiles=("city / default",),
        roots=(),
        blockers=(),
        enabled_actions=(),
    )

    assert permanent_delete_matches(card, "abcdef01") is True
    assert permanent_delete_matches(card, "ABCDEF01") is False
    assert permanent_delete_matches(card, "abcdef0") is False


def test_opaque_callback_body_type_error_is_not_retried():
    from lte_scenario_toolkit.gui.pages.history import _invoke_opaque

    calls = []

    def callback(value):
        calls.append(value)
        raise TypeError("callback body failure")

    with pytest.raises(TypeError, match="callback body failure"):
        _invoke_opaque(callback, "opaque-plan")

    assert calls == ["opaque-plan"]


def test_move_to_trash_history_action_fails_closed_before_path_resolution(tmp_path):
    from lte_scenario_toolkit.gui.pages.history import (
        HistoryAction,
        HistoryActionError,
        resolve_history_action,
    )

    _root, _service, _parent, _child, rows = _task6_history_family(tmp_path)
    row = rows[0]

    with pytest.raises(HistoryActionError, match="TrashPlan|trash plan|build"):
        resolve_history_action(row, HistoryAction.MOVE_TO_TRASH)


def test_trash_cards_disable_conflicting_mutations_for_unsafe_blockers():
    from types import SimpleNamespace

    from lte_scenario_toolkit.gui.pages.history import (
        TrashAction,
        TrashSnapshot,
        build_trash_cards,
    )
    from lte_scenario_toolkit.run_trash import TrashState

    member = SimpleNamespace(
        scenario_id="city",
        profile_id="default",
        artifact_count=1,
        identity=SimpleNamespace(root=Path("C:/results"), run_id="a" * 32),
        original_relative_path=Path("city/default/run"),
    )

    def transaction(state):
        return SimpleNamespace(
            transaction_id="b" * 32,
            state=state,
            deleted_at="2026-07-16T10:00:00Z",
            members=(member,),
            roots=(Path("C:/results"),),
            total_size_bytes=10,
            completed_move_ids=(member.identity.run_id,),
            errors=(),
        )

    trashed = TrashSnapshot((transaction(TrashState.TRASHED),), ())
    blocked = build_trash_cards(
        trashed,
        restore_blockers={
            "b" * 32: ("trash.restore.root_unavailable",),
        },
    )[0]
    occupied = build_trash_cards(
        trashed,
        restore_blockers={
            "b" * 32: ("trash.restore.destination_occupied",),
        },
    )[0]
    failed = TrashSnapshot((transaction(TrashState.PURGE_FAILED),), ())
    retry = build_trash_cards(
        failed,
        restore_blockers={
            "b" * 32: ("trash.restore.journal_invalid",),
        },
    )[0]

    assert blocked.enabled_actions == ()
    assert occupied.enabled_actions == (TrashAction.PURGE,)
    assert retry.enabled_actions == (TrashAction.PURGE,)


def test_trash_card_renders_state_actions_and_restore_blocker(tmp_path):
    from lte_scenario_toolkit.gui.i18n import Translator
    from lte_scenario_toolkit.gui.pages.history import (
        TrashAction,
        TrashCard,
        render_trash_card,
    )

    markers = set()
    classes_seen = []

    class Element:
        def classes(self, *_args):
            classes_seen.extend(_args)
            return self

        def props(self, *_args):
            return self

        def mark(self, marker):
            markers.add(marker)
            return self

        def set_enabled(self, *_args):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Ui:
        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: Element()

    card = TrashCard(
        transaction_id="abcdef0123456789abcdef0123456789",
        state="trashed",
        deleted_at="2026-07-16T10:00:00Z",
        run_count=2,
        size_bytes=4096,
        artifact_count=3,
        scenario_profiles=("city / default",),
        roots=(tmp_path / "results",),
        blockers=("trash.restore.destination_occupied",),
        enabled_actions=(TrashAction.PURGE,),
    )

    render_trash_card(
        Ui(),
        Translator("en"),
        card,
        on_restore=lambda _id: None,
        on_purge=lambda _id: None,
        on_recover=lambda _id: None,
    )

    assert "trash-card-abcdef01" in markers
    assert "trash-action-restore-abcdef01" in markers
    assert "trash-action-purge-abcdef01" in markers
    assert "trash-restore-blockers-abcdef01" in markers
    assert "lte-history-primary" in classes_seen


def test_history_row_matches_search_and_status_filters(tmp_path):
    from lte_scenario_toolkit.gui.pages.history import history_row_matches

    _root, _service, parent, _child, rows = _task6_history_family(tmp_path)
    row = next(item for item in rows if item.run_id == parent.run_id)

    assert history_row_matches(row, "city/default") is False
    assert history_row_matches(row, "city") is True
    assert history_row_matches(row, row.run_id[:8]) is True
    assert history_row_matches(row, status="completed") is True
    assert history_row_matches(row, status="partial") is False


def test_trash_count_labels_are_localized():
    from lte_scenario_toolkit.gui.i18n import Translator

    assert Translator("en").text("history.trash_run_count", count=2) == "2 runs"
    assert Translator("zh-CN").text("history.trash_run_count", count=2) == "2 \u4e2a\u8fd0\u884c"
    assert Translator("zh-CN").text("history.trash_artifact_count", count=3) == "3 \u4e2a\u6210\u679c"


def test_history_partial_csv_never_becomes_its_own_figure_source(tmp_path):
    from lte_scenario_toolkit.figure_service import FigureSpec
    from lte_scenario_toolkit.gui.pages.history import (
        HistoryAction,
        HistoryActionError,
        _figure_source_path,
        history_rows,
        resolve_history_action,
    )
    from lte_scenario_toolkit.run_service import RunService

    service = RunService(tmp_path / "results")
    parent = service.begin("city", "default", created_at="2026-07-16T10:00:00Z")
    (parent.path / "source.csv").write_text("X,Y\n1,2\n", encoding="utf-8")
    parent_path = service.publish(
        parent,
        status="completed",
        artifacts=["source.csv"],
        metadata={"run_kind": "selection"},
    )
    child = service.begin(
        "city",
        "default",
        created_at="2026-07-16T10:00:01Z",
        parent_run_id=parent.run_id,
    )
    (child.path / "source.csv").write_text("X,Y\n1,2\n", encoding="utf-8")
    (child.path / "terrain.png").write_bytes(b"png")
    service.publish(
        child,
        status="partial",
        artifacts=["source.csv", "terrain.png"],
        metadata={
            "run_kind": "figure",
            "source": {"path": str(parent_path), "run_id": parent.run_id},
            "requested_formats": ["png", "html"],
            "artifact_paths": {"png": "terrain.png"},
            "figure_spec": FigureSpec.from_preset("publication").as_dict(),
        },
        errors=[{"artifact": "html", "code": "figure.html.failed"}],
    )
    orphan = service.begin("city", "default", created_at="2026-07-16T10:00:02Z")
    (orphan.path / "source.csv").write_text("X,Y\n1,2\n", encoding="utf-8")
    service.publish(
        orphan,
        status="partial",
        artifacts=["source.csv"],
        metadata={
            "run_kind": "selection",
            "requested_artifacts": ["csv", "terrain_png"],
            "artifact_paths": {"csv": "source.csv"},
        },
        errors=[{"artifact": "terrain_png", "code": "terrain.failed"}],
    )

    rows = history_rows(service)
    child_row = next(row for row in rows if row.run_id == child.run_id)
    orphan_row = next(row for row in rows if row.run_id == orphan.run_id)
    parent_row = next(row for row in rows if row.run_id == parent.run_id)

    assert child_row.figure_source_path == parent_path.resolve()
    assert child_row.figure_source_reference is not None
    assert child_row.figure_source_reference.run_id == parent.run_id
    assert resolve_history_action(child_row, HistoryAction.RETRY_MISSING).path == parent_path
    assert resolve_history_action(child_row, HistoryAction.OPEN_FIGURES).path == parent_path
    assert orphan_row.can_open_figures is False
    with pytest.raises(HistoryActionError, match="compatible figure source"):
        resolve_history_action(orphan_row, HistoryAction.OPEN_FIGURES)
    with pytest.raises(ValueError, match="completed"):
        _figure_source_path(orphan_row.path, orphan_row.record)

    manifest_path = parent_path / "run.json"
    downgraded = json.loads(manifest_path.read_text(encoding="utf-8"))
    downgraded["status"] = "partial"
    manifest_path.write_text(json.dumps(downgraded), encoding="utf-8")
    with pytest.raises(HistoryActionError, match="compatible figure source"):
        resolve_history_action(parent_row, HistoryAction.OPEN_FIGURES)










def test_history_roots_reject_lexical_traversal(tmp_path):
    from lte_scenario_toolkit.gui.pages.history import history_roots

    traversal_parent = tmp_path / "configured"
    traversal_parent.mkdir()
    traversal = traversal_parent / ".." / "escaped"

    with pytest.raises(ValueError, match="traversal|history|root"):
        history_roots(tmp_path, (traversal,))


def test_history_roots_reject_symlink_root(tmp_path):
    from lte_scenario_toolkit.gui.pages.history import history_roots

    target = tmp_path / "real-results"
    target.mkdir()
    redirected = tmp_path / "redirected-results"
    try:
        redirected.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="redirected|history|root"):
        history_roots(tmp_path, (redirected,))


def test_history_roots_reject_junction_or_reparse_root(tmp_path, monkeypatch):
    import lte_scenario_toolkit.run_service as run_module
    from lte_scenario_toolkit.gui.pages.history import history_roots

    redirected = tmp_path / "redirected-results"
    redirected.mkdir()
    original = run_module._is_redirected_path
    monkeypatch.setattr(
        run_module,
        "_is_redirected_path",
        lambda path: Path(path) == redirected or original(Path(path)),
    )

    with pytest.raises(ValueError, match="redirected|history|root"):
        history_roots(tmp_path, (redirected,))




def test_history_rows_sort_fractional_seconds_newest_first(tmp_path):
    from lte_scenario_toolkit.gui.pages.history import history_rows
    from lte_scenario_toolkit.run_service import RunService

    service = RunService(tmp_path)
    earlier = service.begin("city", "default", created_at="2026-07-16T10:00:00Z")
    (earlier.path / "scenario.csv").write_text("ok\n", encoding="utf-8")
    service.publish(earlier, status="completed", artifacts=["scenario.csv"])
    later = service.begin(
        "city",
        "default",
        created_at="2026-07-16T10:00:00.500000Z",
    )
    (later.path / "scenario.csv").write_text("ok\n", encoding="utf-8")
    service.publish(later, status="completed", artifacts=["scenario.csv"])

    rows = history_rows(service)

    assert [row.run_id for row in rows] == [later.run_id, earlier.run_id]


def test_history_rebuild_deduplicates_roots_and_index_is_not_authoritative(tmp_path):
    from lte_scenario_toolkit.gui.pages.history import rebuild_history
    from lte_scenario_toolkit.run_service import RunService

    root = tmp_path / "results"
    service = RunService(root)
    first = service.begin("city", "default", created_at="2026-07-16T10:00:00Z")
    (first.path / "scenario.csv").write_text("ok\n", encoding="utf-8")
    service.publish(first, status="completed", artifacts=["scenario.csv"])
    index = tmp_path / ".lte-data/cache/history-index.json"

    initial = rebuild_history(tmp_path, (root, root / "."), index_path=index)
    assert [row.run_id for row in initial.rows] == [first.run_id]
    assert index.is_file()

    second = service.begin("city", "default", created_at="2026-07-16T10:00:01Z")
    (second.path / "scenario.csv").write_text("ok\n", encoding="utf-8")
    service.publish(second, status="completed", artifacts=["scenario.csv"])
    refreshed = rebuild_history(tmp_path, (root,), index_path=index)

    assert {row.run_id for row in refreshed.rows} == {first.run_id, second.run_id}
    cached = json.loads(index.read_text(encoding="utf-8"))
    assert {item["run_id"] for item in cached["rows"]} == {
        first.run_id,
        second.run_id,
    }


def test_history_rebuild_keeps_live_rows_when_derived_index_is_unsafe(tmp_path):
    from lte_scenario_toolkit.gui.pages.history import rebuild_history
    from lte_scenario_toolkit.run_service import RunService

    root = tmp_path / "results"
    service = RunService(root)
    run = service.begin("city", "default")
    (run.path / "scenario.csv").write_text("ok\n", encoding="utf-8")
    service.publish(run, status="completed", artifacts=["scenario.csv"])
    cache_parent = tmp_path / ".lte-data"
    cache_parent.mkdir()
    (cache_parent / "cache").write_text("occupied", encoding="utf-8")

    snapshot = rebuild_history(tmp_path, (root,))

    assert [row.run_id for row in snapshot.rows] == [run.run_id]
    assert any("History index was not updated" in item.error for item in snapshot.diagnostics)
    assert snapshot.index_path == (cache_parent / "cache/history-index.json").absolute()


def test_figure_form_does_not_render_until_refresh():
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages.figures import FigurePageState, preview_spec

    state = FigurePageState.for_source(object()).with_dpi(200)

    assert state.preview_path is None
    assert state.preview_stale is True
    preview = preview_spec(state.spec)
    publication = preview_spec(
        replace(state.spec, preset="publication", dpi=600, max_pixels=5000)
    )
    assert publication.dpi > preview.dpi
    assert publication.max_pixels > preview.max_pixels
    assert publication.dpi < 600
    assert publication.max_pixels < 5000


def test_figure_and_history_pages_accept_bounded_selection_sources():
    import inspect

    from lte_scenario_toolkit.gui.pages.figures import render_figures_page
    from lte_scenario_toolkit.gui.pages.history import render_history_content

    figure_parameters = inspect.signature(render_figures_page).parameters
    history_parameters = inspect.signature(render_history_content).parameters

    assert "source_options" in figure_parameters
    assert "pending_selections" in history_parameters
    assert "on_open_pending_figures" in history_parameters


def test_history_figure_source_options_list_each_completed_selection_once(tmp_path):
    from lte_scenario_toolkit.gui.pages import history
    from lte_scenario_toolkit.run_service import RunService

    service = RunService(tmp_path / "results")
    source = service.begin("new-york-city", "new-york-default")
    (source.path / "selection.csv").write_text("X,Y\n1,2\n", encoding="utf-8")
    source_path = service.publish(
        source,
        status="completed",
        artifacts=["selection.csv"],
        metadata={"run_kind": "selection"},
    )
    derived = service.begin(
        "new-york-city",
        "new-york-default",
        parent_run_id=source.run_id,
    )
    (derived.path / "terrain.png").write_bytes(b"png")
    service.publish(
        derived,
        status="completed",
        artifacts=["terrain.png"],
        metadata={
            "run_kind": "figure",
            "source": {"path": str(source_path), "run_id": source.run_id},
        },
    )
    snapshot = history.rebuild_history(tmp_path, (tmp_path / "results",))
    build_options = getattr(history, "figure_source_options", None)

    assert callable(build_options)
    options = build_options(snapshot)
    assert list(options) == [str(source_path.resolve())]
    assert "new-york-city / new-york-default" in options[str(source_path.resolve())]


async def test_figure_style_and_source_fields_are_selection_controls(user, tmp_path):
    from nicegui import ui

    from lte_scenario_toolkit.gui.i18n import Translator
    from lte_scenario_toolkit.gui.pages.figures import render_figures_page
    from lte_scenario_toolkit.jobs import JobCoordinator

    coordinator = JobCoordinator()
    source = (tmp_path / "results/city/default/run-a").resolve()

    @ui.page("/figure-selection-controls")
    def figure_selection_controls():
        render_figures_page(
            ui,
            Translator("en"),
            tmp_path,
            coordinator,
            source_options={str(source): "City / Default"},
        )

    try:
        await user.open("/figure-selection-controls")
        source_control = next(iter(user.find(marker="figure-source-path").elements))
        colormap_control = next(iter(user.find(marker="figure-colormap").elements))
        station_control = next(iter(user.find(marker="figure-station-color").elements))

        assert type(source_control).__name__ == "Select"
        assert type(colormap_control).__name__ == "Select"
        assert type(station_control).__name__ == "Select"
        assert str(source) in source_control.options
        assert "terrain" in colormap_control.options
        assert "viridis" in colormap_control.options
        assert "red" in station_control.options
        assert "royalblue" in station_control.options
    finally:
        coordinator.shutdown()


async def test_figures_route_discovers_completed_run_sources_without_path_typing(
    user,
    tmp_path,
):
    from types import SimpleNamespace

    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.run_service import RunService

    service = RunService(tmp_path / "results")
    source = service.begin("new-york-city", "new-york-default")
    (source.path / "selection.csv").write_text("X,Y\n1,2\n", encoding="utf-8")
    source_path = service.publish(
        source,
        status="completed",
        artifacts=["selection.csv"],
        metadata={"run_kind": "selection"},
    )
    coordinator = JobCoordinator()
    try:
        create_app(
            catalog=SimpleNamespace(root=tmp_path.resolve()),
            coordinator=coordinator,
            testing=True,
        )
        await user.open("/figures")
        source_control = next(iter(user.find(marker="figure-source-path").elements))

        assert str(source_path.resolve()) in source_control.options
        assert (
            "new-york-city / new-york-default"
            in source_control.options[str(source_path.resolve())]
        )
    finally:
        coordinator.shutdown()


async def test_history_exposes_confirmed_unpublished_selection_with_safe_actions(
    user,
    tmp_path,
):
    from nicegui import ui

    from lte_scenario_toolkit.gui.i18n import Translator
    from lte_scenario_toolkit.gui.pages.history import (
        HistorySnapshot,
        render_history_content,
        render_history_frame,
    )

    session = _task15_locked_session(
        tmp_path,
        object(),
        session_id="new-york-pending",
    )
    opened = []
    continued = []

    @ui.page("/history-pending-selection")
    def history_pending_selection():
        translator = Translator("en")
        holder = render_history_frame(ui, translator)
        render_history_content(
            ui,
            translator,
            holder,
            HistorySnapshot(
                roots=((tmp_path / "results").resolve(),),
                rows=(),
                diagnostics=(),
                index_path=(tmp_path / ".lte-data/cache/history-index.json").absolute(),
            ),
            pending_selections=(session,),
            on_open_pending_figures=opened.append,
            on_continue_pending=continued.append,
        )

    await user.open("/history-pending-selection")

    await user.should_see(marker="history-pending-new-york-pending")
    await user.should_see("ready-city / default")
    await user.should_see(
        marker="history-pending-status-new-york-pending",
        content="Not published",
    )
    user.find(marker="history-pending-open-new-york-pending").click()
    user.find(marker="history-pending-continue-new-york-pending").click()
    assert opened == ["new-york-pending"]
    assert continued == ["new-york-pending"]


@pytest.mark.asyncio
async def test_run_trash_mutation_finishes_and_skips_deleted_client_render():
    """A completed Trash worker must not send a result to a gone client."""

    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator

    class DeletedClient:
        is_deleted = True

    async def fake_io_bound(worker, *args, **kwargs):
        return worker(*args, **kwargs)

    coordinator = JobCoordinator()
    rendered: list[object] = []
    try:
        await app_module.run_trash_mutation(
            client=DeletedClient(),
            coordinator=coordinator,
            kind="history.trash_restore",
            worker=lambda: object(),
            on_success=lambda value: rendered.append(value),
            io_bound=fake_io_bound,
        )
        assert rendered == []
        assert coordinator.snapshot().active is False
    finally:
        coordinator.shutdown()


def test_client_is_deleted_accepts_bool_and_callable_flags():
    from lte_scenario_toolkit.gui.app import client_is_deleted

    assert client_is_deleted(SimpleNamespace(is_deleted=True)) is True
    assert client_is_deleted(SimpleNamespace(is_deleted=False)) is False
    assert client_is_deleted(SimpleNamespace(is_deleted=lambda: True)) is True
    assert client_is_deleted(SimpleNamespace(is_deleted=lambda: False)) is False


def test_trash_error_mapping_and_safe_structured_logging(monkeypatch, tmp_path):
    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.gui.pages.history import HistoryActionError
    from lte_scenario_toolkit.jobs import JobBusyError
    from lte_scenario_toolkit.run_trash import (
        RunLeaseConflictError,
        TrashPlanStaleError,
    )

    class CapturedLogger:
        def __init__(self):
            self.messages: list[str] = []

        def info(self, message, fields):
            self.messages.append(f"{message} {fields}")

        def warning(self, message, fields):
            self.messages.append(f"{message} {fields}")

    logger = CapturedLogger()
    monkeypatch.setattr(app_module, "_LOGGER", logger)
    assert (
        app_module.trash_error_translation_key(
            JobBusyError(
                "busy",
                active_job_id="a",
                active_kind="x",
                requested_kind="y",
            ),
            default="history.trash_error",
        )
        == "history.trash_busy"
    )
    assert app_module.trash_error_translation_key(
        TrashPlanStaleError("changed"), default="history.trash_error"
    ) == "history.trash_stale"
    assert app_module.trash_error_translation_key(
        RunLeaseConflictError("in use"), default="history.trash_error"
    ) == "history.trash_lease_conflict"
    assert app_module.trash_error_translation_key(
        ValueError("original destination is occupied"),
        default="history.trash_error",
    ) == "history.trash_destination_occupied"
    assert app_module.trash_error_translation_key(
        ValueError("an expected root is unavailable"),
        default="history.trash_error",
    ) == "history.trash_root_unavailable"
    assert app_module.trash_error_translation_key(
        ValueError("trash journal is invalid"),
        default="history.trash_error",
    ) == "history.trash_journal_invalid"
    assert app_module.trash_error_translation_key(
        ValueError("transaction is not purgeable"),
        default="history.trash_error",
    ) == "history.trash_permanent_error"

    try:
        raise HistoryActionError("changed or is in use") from TrashPlanStaleError(
            "changed"
        )
    except HistoryActionError as wrapped_stale:
        assert app_module.trash_error_translation_key(
            wrapped_stale,
            default="history.trash_error",
        ) == "history.trash_stale"

    try:
        raise HistoryActionError("changed or is in use") from RunLeaseConflictError(
            "in use"
        )
    except HistoryActionError as wrapped_lease:
        assert app_module.trash_error_translation_key(
            wrapped_lease,
            default="history.trash_error",
        ) == "history.trash_lease_conflict"

    app_module.log_trash_mutation(
        "history.trash_move",
        transaction_id="a" * 32,
        state="trashed",
        member_count=2,
        roots=(tmp_path,),
    )
    assert logger.messages
    rendered = logger.messages[-1]
    assert str(tmp_path) not in rendered
    assert "root_digests" in rendered


@pytest.mark.asyncio
async def test_run_trash_mutation_none_result_defers_finish_until_worker_done():
    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator

    started = Event()
    release = Event()
    finished = Event()

    def worker():
        started.set()
        assert release.wait(2)
        finished.set()
        return object()

    async def canceled_io_bound(callback):
        import threading

        threading.Thread(target=callback, daemon=True).start()
        assert await asyncio.to_thread(started.wait, 2)
        return None

    coordinator = JobCoordinator()
    rendered: list[str] = []
    try:
        await app_module.run_trash_mutation(
            client=SimpleNamespace(is_deleted=False),
            coordinator=coordinator,
            kind="history.trash_restore",
            worker=worker,
            on_success=lambda _value: rendered.append("success"),
            on_refresh=lambda: rendered.append("refresh"),
            io_bound=canceled_io_bound,
        )
        assert coordinator.snapshot().active is True
        assert rendered == []
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
        for _ in range(20):
            if not coordinator.snapshot().active:
                break
            await asyncio.sleep(0.01)
        assert coordinator.snapshot().active is False
    finally:
        release.set()
        coordinator.shutdown()


@pytest.mark.asyncio
async def test_run_trash_mutation_none_result_handles_delayed_worker_start():
    import threading

    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator

    release = Event()
    started = Event()

    def worker():
        started.set()
        release.wait(2)

    async def canceled_io_bound(callback):
        threading.Thread(target=callback, daemon=True).start()
        return None

    coordinator = JobCoordinator()
    try:
        await app_module.run_trash_mutation(
            client=SimpleNamespace(is_deleted=False),
            coordinator=coordinator,
            kind="history.trash_purge",
            worker=worker,
            io_bound=canceled_io_bound,
        )
        assert coordinator.snapshot().active is True
        assert await asyncio.to_thread(started.wait, 2)
        release.set()
        for _ in range(20):
            if not coordinator.snapshot().active:
                break
            await asyncio.sleep(0.01)
        assert coordinator.snapshot().active is False
    finally:
        release.set()
        coordinator.shutdown()


@pytest.mark.asyncio
async def test_run_trash_mutation_pre_submit_cancel_releases_slot():
    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator

    async def canceled_io_bound(_callback):
        raise asyncio.CancelledError

    coordinator = JobCoordinator()
    try:
        await app_module.run_trash_mutation(
            client=SimpleNamespace(is_deleted=False),
            coordinator=coordinator,
            kind="history.trash_restore",
            worker=lambda: object(),
            io_bound=canceled_io_bound,
        )
        assert coordinator.snapshot().active is False
    finally:
        coordinator.shutdown()


@pytest.mark.asyncio
async def test_run_trash_mutation_pre_submit_runtime_error_releases_slot():
    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator

    async def failed_io_bound(_callback):
        raise RuntimeError("bridge is unavailable")

    coordinator = JobCoordinator()
    try:
        await app_module.run_trash_mutation(
            client=SimpleNamespace(is_deleted=False),
            coordinator=coordinator,
            kind="history.trash_purge",
            worker=lambda: object(),
            io_bound=failed_io_bound,
        )
        assert coordinator.snapshot().active is False
    finally:
        coordinator.shutdown()


@pytest.mark.asyncio
async def test_run_trash_mutation_executor_creation_failure_releases_slot(monkeypatch):
    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator

    class BrokenExecutor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("cannot start worker thread")

    monkeypatch.setattr(app_module, "ThreadPoolExecutor", BrokenExecutor)
    coordinator = JobCoordinator()
    errors: list[type[BaseException]] = []
    try:
        await app_module.run_trash_mutation(
            client=SimpleNamespace(is_deleted=False),
            coordinator=coordinator,
            kind="history.trash_move",
            worker=lambda: object(),
            on_error=lambda error: errors.append(type(error)),
            io_bound=lambda callback: callback(),
        )
        assert errors == [RuntimeError]
        assert coordinator.snapshot().active is False
    finally:
        coordinator.shutdown()


@pytest.mark.asyncio
async def test_run_trash_mutation_bridge_failure_never_renders_mid_worker():
    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator

    started = Event()
    release = Event()
    events: list[str] = []

    def worker():
        started.set()
        assert release.wait(2)
        return object()

    async def failed_io_bound(_callback):
        raise RuntimeError("bridge is unavailable")

    coordinator = JobCoordinator()
    try:
        await app_module.run_trash_mutation(
            client=SimpleNamespace(is_deleted=False),
            coordinator=coordinator,
            kind="history.trash_purge",
            worker=worker,
            on_refresh=lambda: events.append("refresh"),
            on_error=lambda _error: events.append("error"),
            io_bound=failed_io_bound,
        )
        assert started.wait(2)
        assert events == []
        assert coordinator.snapshot().active is True
        release.set()
        for _ in range(20):
            if not coordinator.snapshot().active:
                break
            await asyncio.sleep(0.01)
        assert coordinator.snapshot().active is False
    finally:
        release.set()
        coordinator.shutdown()


@pytest.mark.asyncio
async def test_run_trash_mutation_refresh_failure_does_not_relabel_committed_work():
    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator

    async def fake_io_bound(callback):
        return callback()

    def failed_refresh():
        events.append("refresh")
        raise RuntimeError("render failed")

    events: list[str] = []
    coordinator = JobCoordinator()
    result = object()
    try:
        returned = await app_module.run_trash_mutation(
            client=SimpleNamespace(is_deleted=False),
            coordinator=coordinator,
            kind="history.trash_restore",
            worker=lambda: result,
            on_refresh=failed_refresh,
            on_success=lambda _value: events.append("success"),
            on_error=lambda _error: events.append("error"),
            on_refresh_error=lambda _error: events.append("refresh_error"),
            io_bound=fake_io_bound,
        )
        assert returned is result
        assert events == ["refresh", "refresh_error"]
        assert coordinator.snapshot().active is False
    finally:
        coordinator.shutdown()


@pytest.mark.asyncio
async def test_run_trash_mutation_logging_failure_cannot_strand_job(monkeypatch):
    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator

    async def fake_io_bound(callback):
        return callback()

    monkeypatch.setattr(
        app_module._LOGGER,
        "info",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("log failed")),
    )
    coordinator = JobCoordinator()
    try:
        await app_module.run_trash_mutation(
            client=SimpleNamespace(is_deleted=False),
            coordinator=coordinator,
            kind="history.trash_restore",
            worker=lambda: object(),
            io_bound=fake_io_bound,
        )
        assert coordinator.snapshot().active is False
    finally:
        coordinator.shutdown()


@pytest.mark.asyncio
async def test_run_trash_mutation_failure_and_refresh_failure_reports_both():
    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator

    async def fake_io_bound(callback):
        return callback()

    def failed_refresh():
        raise RuntimeError("refresh failed")

    coordinator = JobCoordinator()
    events: list[str] = []
    try:
        await app_module.run_trash_mutation(
            client=SimpleNamespace(is_deleted=False),
            coordinator=coordinator,
            kind="history.trash_purge",
            worker=lambda: (_ for _ in ()).throw(RuntimeError("purge failed")),
            on_refresh=failed_refresh,
            on_refresh_error=lambda _error: events.append("refresh_error"),
            on_error=lambda _error: events.append("mutation_error"),
            io_bound=fake_io_bound,
        )
        assert events == ["refresh_error", "mutation_error"]
        assert coordinator.snapshot().active is False
    finally:
        coordinator.shutdown()


@pytest.mark.asyncio
async def test_run_trash_mutation_outer_cancel_waits_for_submitted_worker():
    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator

    started = Event()
    release = Event()

    def worker():
        started.set()
        assert release.wait(2)
        return object()

    async def bridge(callback):
        return await asyncio.to_thread(callback)

    coordinator = JobCoordinator()
    client = SimpleNamespace(is_deleted=False)
    task = asyncio.create_task(
        app_module.run_trash_mutation(
            client=client,
            coordinator=coordinator,
            kind="history.trash_restore",
            worker=worker,
            io_bound=bridge,
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)
        client.is_deleted = True
        task.cancel()
        await asyncio.sleep(0.05)
        assert coordinator.snapshot().active is True
        release.set()
        await task
        assert coordinator.snapshot().active is False
    finally:
        release.set()
        if not task.done():
            await task
        coordinator.shutdown()


@pytest.mark.asyncio
async def test_run_trash_mutation_cancel_suppresses_late_worker_error_callback():
    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator

    started = Event()
    release = Event()
    errors: list[type[BaseException]] = []

    def worker():
        started.set()
        assert release.wait(2)
        raise RuntimeError("late worker failure")

    async def bridge(callback):
        return await asyncio.to_thread(callback)

    coordinator = JobCoordinator()
    task = asyncio.create_task(
        app_module.run_trash_mutation(
            client=SimpleNamespace(is_deleted=False),
            coordinator=coordinator,
            kind="history.trash_restore",
            worker=worker,
            on_error=lambda error: errors.append(type(error)),
            io_bound=bridge,
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0.05)
        assert coordinator.snapshot().active is True
        release.set()
        await task
        assert errors == []
        assert coordinator.snapshot().active is False
    finally:
        release.set()
        if not task.done():
            await task
        coordinator.shutdown()


@pytest.mark.asyncio
async def test_run_trash_mutation_refreshes_before_success():
    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator

    async def fake_io_bound(worker):
        return worker()

    coordinator = JobCoordinator()
    events: list[str] = []
    try:
        await app_module.run_trash_mutation(
            client=SimpleNamespace(is_deleted=False),
            coordinator=coordinator,
            kind="history.trash_move",
            worker=lambda: object(),
            on_refresh=lambda: events.append("refresh"),
            on_success=lambda _value: events.append("success"),
            io_bound=fake_io_bound,
        )
        assert events == ["refresh", "success"]
        assert coordinator.snapshot().active is False
    finally:
        coordinator.shutdown()


@pytest.mark.parametrize(
    "kind",
    (
        "history.trash_move",
        "history.trash_restore",
        "history.trash_purge",
        "history.trash_recover",
    ),
)
@pytest.mark.asyncio
async def test_run_trash_mutation_finishes_each_operation_kind(kind):
    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator

    async def fake_io_bound(worker):
        return worker()

    coordinator = JobCoordinator()
    try:
        await app_module.run_trash_mutation(
            client=SimpleNamespace(is_deleted=False),
            coordinator=coordinator,
            kind=kind,
            worker=lambda: object(),
            io_bound=fake_io_bound,
        )
        assert coordinator.snapshot().active is False
    finally:
        coordinator.shutdown()


@pytest.mark.asyncio
async def test_run_trash_mutation_finishes_on_error_and_blocks_second_job():
    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobBusyError, JobCoordinator

    class LiveClient:
        is_deleted = False

    gate = Event()
    entered = Event()

    async def fake_io_bound(worker, *args, **kwargs):
        entered.set()
        await asyncio.to_thread(gate.wait, 2)
        return worker(*args, **kwargs)

    coordinator = JobCoordinator()
    errors: list[type[BaseException]] = []
    try:
        pending = asyncio.create_task(
            app_module.run_trash_mutation(
                client=LiveClient(),
                coordinator=coordinator,
                kind="history.trash_purge",
                worker=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                on_error=lambda error: errors.append(type(error)),
                io_bound=fake_io_bound,
            )
        )
        assert await asyncio.to_thread(entered.wait, 2)
        with pytest.raises(JobBusyError):
            coordinator.start("history.trash_restore")
        gate.set()
        await pending
        assert errors == [RuntimeError]
        assert coordinator.snapshot().active is False
    finally:
        gate.set()
        coordinator.shutdown()


@pytest.mark.asyncio
async def test_trash_route_renders_loading_before_io(monkeypatch, tmp_path, user):
    from nicegui import run

    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.run_trash import RunUsageLeaseRegistry, TrashManager

    events: list[str] = []
    leases = RunUsageLeaseRegistry()
    manager = TrashManager(lambda: (tmp_path / "results",), leases)
    coordinator = JobCoordinator()

    monkeypatch.setattr(
        app_module,
        "render_trash_loading",
        lambda *args, **kwargs: events.append("loading") or object(),
    )

    async def fake_io_bound(worker, *args, **kwargs):
        events.append("io")
        return worker(*args, **kwargs)

    monkeypatch.setattr(run, "io_bound", fake_io_bound)
    monkeypatch.setattr(
        app_module,
        "render_trash_page",
        lambda *args, **kwargs: events.append("render"),
    )
    try:
        app_module.create_app(
            catalog=SimpleNamespace(root=tmp_path.resolve()),
            coordinator=coordinator,
            trash_manager=manager,
            usage_leases=leases,
            testing=True,
        )
        await user.open("/history/trash")
        for _ in range(20):
            if len(events) >= 2:
                break
            await asyncio.sleep(0.05)
        assert events[:2] == ["loading", "io"]
    finally:
        coordinator.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("/history", "/history/trash"))
async def test_history_routes_treat_none_io_result_as_cancellation(
    monkeypatch,
    tmp_path,
    user,
    route,
):
    from nicegui import run

    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.run_trash import RunUsageLeaseRegistry, TrashManager

    leases = RunUsageLeaseRegistry()
    manager = TrashManager(lambda: (tmp_path / "results",), leases)
    rendered: list[str] = []

    async def canceled_io_bound(*_args, **_kwargs):
        return None

    monkeypatch.setattr(run, "io_bound", canceled_io_bound)
    monkeypatch.setattr(
        app_module,
        "render_history_page",
        lambda *args, **kwargs: rendered.append("history"),
    )
    monkeypatch.setattr(
        app_module,
        "render_trash_page",
        lambda *args, **kwargs: rendered.append("trash"),
    )
    coordinator = JobCoordinator()
    try:
        app_module.create_app(
            catalog=SimpleNamespace(root=tmp_path.resolve()),
            coordinator=coordinator,
            trash_manager=manager,
            usage_leases=leases,
            testing=True,
        )
        await user.open(route)
        await asyncio.sleep(0.05)
        assert rendered == []
    finally:
        coordinator.shutdown()


@pytest.mark.asyncio
async def test_trash_route_callbacks_use_transaction_id_and_confirmation(
    monkeypatch, tmp_path, user
):
    from nicegui import run

    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.run_trash import (
        RunUsageLeaseRegistry,
        TrashManager,
        TrashMutationReceipt,
    )

    leases = RunUsageLeaseRegistry()
    manager = TrashManager(lambda: (tmp_path / "results",), leases)
    calls: list[tuple[str, tuple[object, ...]]] = []
    manager.transaction = lambda value: SimpleNamespace(  # type: ignore[method-assign]
        transaction_id=value,
        state="trashed",
        members=(),
        roots=(),
    )
    manager.restore = lambda value: calls.append(("restore", (value,))) or TrashMutationReceipt(value, restored=True)  # type: ignore[method-assign]
    manager.purge = lambda value, *, confirmation: calls.append(  # type: ignore[method-assign]
        ("purge", (value, confirmation))
    ) or TrashMutationReceipt(value, purged=True)
    manager.recover = lambda value: calls.append(("recover", (value,))) or object()  # type: ignore[method-assign]
    captured: dict[str, object] = {}

    def capture_page(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(app_module, "render_trash_page", capture_page)

    async def fake_io_bound(worker, *args, **kwargs):
        return worker(*args, **kwargs)

    monkeypatch.setattr(run, "io_bound", fake_io_bound)
    coordinator = JobCoordinator()
    try:
        app_module.create_app(
            catalog=SimpleNamespace(root=tmp_path.resolve()),
            coordinator=coordinator,
            trash_manager=manager,
            usage_leases=leases,
            testing=True,
        )
        await user.open("/history/trash")
        for _ in range(20):
            if "on_restore" in captured:
                break
            await asyncio.sleep(0.05)
        transaction_id = "a" * 32
        await captured["on_restore"](transaction_id)  # type: ignore[index,operator]
        await captured["on_purge"](transaction_id, "typed")  # type: ignore[index,operator]
        await captured["on_recover"](transaction_id)  # type: ignore[index,operator]
        assert calls == [
            ("restore", (transaction_id,)),
            ("purge", (transaction_id, "typed")),
            ("recover", (transaction_id,)),
        ]
    finally:
        coordinator.shutdown()


@pytest.mark.asyncio
async def test_create_app_passes_shared_trash_dependencies_to_figures(
    monkeypatch, tmp_path, user
):
    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.run_trash import RunUsageLeaseRegistry, TrashManager

    captured: list[dict[str, object]] = []
    leases = RunUsageLeaseRegistry()
    manager = TrashManager(lambda: (tmp_path / "configured",), leases)
    coordinator = JobCoordinator()
    from lte_scenario_toolkit.gui.settings import GuiSettingsStore

    GuiSettingsStore(tmp_path).update(
        add_output_roots=(tmp_path / "configured",)
    )

    def fake_render_figures(*args, **kwargs):
        captured.append(kwargs)
        return object()

    monkeypatch.setattr(app_module, "render_figures_page", fake_render_figures)
    try:
        app_module.create_app(
            catalog=SimpleNamespace(root=tmp_path.resolve()),
            coordinator=coordinator,
            trash_manager=manager,
            usage_leases=leases,
            testing=True,
        )
        await user.open("/figures")
        assert captured
        assert captured[0]["usage_leases"] is leases
        roots_provider = captured[0]["run_roots"]
        assert callable(roots_provider)
        assert (tmp_path / "results").resolve() in tuple(roots_provider())
        assert (tmp_path / "configured").resolve() in tuple(roots_provider())
    finally:
        coordinator.shutdown()


def test_figure_reserved_cpu_close_defers_job_release_until_abandon(tmp_path):
    from lte_scenario_toolkit.gui.pages.figures import FigureController
    from lte_scenario_toolkit.jobs import JobCoordinator

    coordinator = JobCoordinator()
    controller = FigureController(tmp_path, coordinator)
    try:
        job = controller._reserve("figure-export")
        controller.close()
        assert coordinator.snapshot().active is True

        controller.abandon_cpu_export(job)
        assert coordinator.snapshot().active is False
    finally:
        controller.close()
        coordinator.shutdown()


async def test_app_history_prioritizes_current_confirmed_selection(user, tmp_path):
    from types import SimpleNamespace

    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.pages.candidates import CandidateSessionRegistry
    from lte_scenario_toolkit.jobs import JobCoordinator

    registry = CandidateSessionRegistry()
    registry.add(
        _task15_locked_session(
            tmp_path,
            object(),
            session_id="new-york-current",
        )
    )
    coordinator = JobCoordinator()
    try:
        create_app(
            catalog=SimpleNamespace(root=tmp_path.resolve()),
            candidate_registry=registry,
            coordinator=coordinator,
            testing=True,
        )
        await user.open("/history")

        await user.should_see(
            marker="history-pending-new-york-current",
            retries=15,
        )
        await user.should_see("ready-city / default")
        user.find(marker="history-pending-open-new-york-current").click()
        for _ in range(20):
            if user.back_history and user.back_history[-1].startswith("/figures/"):
                break
            await asyncio.sleep(0.05)
        assert user.back_history[-1].startswith("/figures/")
    finally:
        coordinator.shutdown()




def test_preview_cache_is_explicit_and_style_changes_only_mark_stale(tmp_path):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages.figures import (
        FigurePageState,
        preview_cache_path,
    )

    source = _task15_figure_source(tmp_path)
    expected = preview_cache_path(tmp_path, source, FigurePageState.for_source(source).spec)
    assert expected.parent == tmp_path / ".lte-data/cache/previews"
    assert not expected.exists()
    assert not expected.parent.exists()

    rendered = replace(
        FigurePageState.for_source(source),
        preview_path=expected,
        preview_stale=False,
    )
    changed = rendered.with_dpi(200)
    assert changed.preview_path == expected
    assert changed.preview_stale is True


def test_station_visibility_renderer_invalidates_v2_preview_cache(
    tmp_path,
    monkeypatch,
):
    from lte_scenario_toolkit.gui.pages import figures

    source = _task15_figure_source(tmp_path)
    spec = figures.FigurePageState.for_source(source).spec
    current = figures.preview_cache_path(tmp_path, source, spec)

    monkeypatch.setattr(figures, "PREVIEW_CACHE_VERSION", "figure-preview-v2")
    legacy = figures.preview_cache_path(tmp_path, source, spec)

    assert current != legacy


def test_figure_controller_invalidate_source_clears_all_exportable_state(tmp_path):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages.figures import FigureController
    from lte_scenario_toolkit.jobs import JobCoordinator

    coordinator = JobCoordinator()
    source = _task15_figure_source(tmp_path)
    controller = FigureController(
        tmp_path,
        coordinator,
        source=source,
        output_root=tmp_path / "outputs",
        parent_run_id="parent-run",
        parent_run_path=tmp_path / "parent-run",
    )
    preview_path = tmp_path / "preview.png"
    run_path = tmp_path / "figure-run"
    controller._set_state(
        replace(
            controller.state,
            preview_path=preview_path,
            preview_stale=False,
            run_path=run_path,
        )
    )
    revision = controller.state.revision
    try:
        state = controller.invalidate_source()

        assert state.source is None
        assert state.source_dirty is True
        assert state.source_error is None
        assert state.preview_path is None
        assert state.preview_stale is True
        assert state.run_path is None
        assert state.revision == revision + 1
        assert controller.output_root is None
        assert controller.parent_run_id is None
        assert controller.parent_run_path is None
        with pytest.raises(ValueError, match="source"):
            controller.refresh_preview()
        with pytest.raises(ValueError, match="source"):
            controller.export(("png",))
    finally:
        controller.close()
        coordinator.shutdown()


def test_failed_current_selection_preparation_remains_fail_closed(tmp_path):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages.figures import FigureController
    from lte_scenario_toolkit.jobs import JobCoordinator

    class FailingSelectionService:
        def prepare_figure_source(self, *_args):
            raise ValueError("selection source is unavailable")

    coordinator = JobCoordinator()
    controller = FigureController(
        tmp_path,
        coordinator,
        source=_task15_figure_source(tmp_path),
        output_root=tmp_path / "old-output",
    )
    session = SimpleNamespace(
        selection_service=FailingSelectionService(),
        preflight=SimpleNamespace(output_root=tmp_path / "new-output"),
        scan_result=object(),
        locked_candidate=object(),
    )
    try:
        controller.invalidate_source()
        job = controller.prepare_selection(session)
        assert job.future is not None
        job.future.result(timeout=2)

        state = controller.drain(job)

        assert state.source is None
        assert state.source_dirty is True
        assert state.source_error == "selection source is unavailable"
        assert state.preview_path is None
        assert state.run_path is None
        assert controller.parent_run_id is None
        assert controller.parent_run_path is None
        restyled = controller.update_spec(replace(state.spec, dpi=200))
        assert restyled.source_error == "selection source is unavailable"
        assert restyled.source_dirty is True
    finally:
        controller.close()
        coordinator.shutdown()


async def test_figure_page_invalidates_old_source_on_edit_and_failed_load(
    tmp_path,
    monkeypatch,
    user,
):
    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.pages import figures
    from lte_scenario_toolkit.jobs import JobCoordinator

    source = _task15_figure_source(tmp_path)
    completed_a = tmp_path / "completed-a"
    missing_b = tmp_path / "missing-b"

    def load_source(path):
        if Path(path) == completed_a:
            return source
        raise FileNotFoundError(f"Figure source does not exist: {path}")

    monkeypatch.setattr(figures, "load_figure_source", load_source)
    coordinator = JobCoordinator()
    try:
        create_app(
            catalog=SimpleNamespace(root=tmp_path.resolve()),
            coordinator=coordinator,
            figure_source_options_provider=lambda: {
                str(completed_a): "Completed A",
                str(missing_b): "Missing B",
            },
            testing=True,
        )
        await user.open("/figures")
        source_field = next(iter(user.find(marker="figure-source-path").elements))
        with user.client:
            source_field.set_value(str(completed_a.resolve()))
        user.find(marker="figure-load-source").click()

        await user.should_see(marker="figure-source-ready", content="Rectangle 1")
        assert next(iter(user.find(marker="figure-refresh-preview").elements)).enabled
        assert next(iter(user.find(marker="figure-export").elements)).enabled

        with user.client:
            source_field.set_value(str(missing_b.resolve()))

        await user.should_see(marker="figure-source-dirty", content="Load to continue")
        await user.should_not_see(marker="figure-source-ready")
        await user.should_not_see(marker="figure-preview-surface")
        assert not next(
            iter(user.find(marker="figure-refresh-preview").elements)
        ).enabled
        assert not next(iter(user.find(marker="figure-export").elements)).enabled

        user.find(marker="figure-load-source").click()

        await user.should_see(
            marker="figure-source-error",
            content="The figure source could not be loaded",
        )
        technical = next(
            iter(user.find(marker="figure-technical-copy").elements)
        )
        assert f"Figure source does not exist: {missing_b}" in technical.text
        await user.should_not_see(marker="figure-source-ready")
        assert not next(iter(user.find(marker="figure-export").elements)).enabled
        assert coordinator.snapshot().active is False
    finally:
        coordinator.shutdown()


async def test_figure_primary_source_error_is_localized_and_raw_detail_is_collapsed(
    tmp_path,
    monkeypatch,
    user,
):
    from nicegui import ui

    from lte_scenario_toolkit.gui.i18n import Translator
    from lte_scenario_toolkit.gui.pages import figures
    from lte_scenario_toolkit.jobs import JobCoordinator

    raw_detail = "figure.html.failed: traceback exception"
    monkeypatch.setattr(
        figures,
        "load_figure_source",
        lambda _path: (_ for _ in ()).throw(RuntimeError(raw_detail)),
    )
    coordinator = JobCoordinator()

    @ui.page("/figures-human-errors")
    def figures_human_errors():
        figures.render_figures_page(
            ui,
            Translator("zh-CN"),
            tmp_path,
            coordinator,
            source_options={str(tmp_path / "missing-run"): "Missing run"},
        )

    try:
        await user.open("/figures-human-errors")
        source = next(iter(user.find(marker="figure-source-path").elements))
        with user.client:
            source.set_value(str((tmp_path / "missing-run").resolve()))
        user.find(marker="figure-load-source").click()

        await user.should_see(
            marker="figure-source-error",
            content=(
                "\u65e0\u6cd5\u52a0\u8f7d\u6240\u9009\u56fe\u8868\u6765\u6e90\u3002"
                "\u8bf7\u5237\u65b0\u8fd0\u884c\u5217\u8868\u540e\u91cd\u8bd5\u3002"
            ),
        )
        primary = next(iter(user.find(marker="figure-source-error").elements))
        assert raw_detail not in primary.text
        technical = next(
            iter(user.find(marker="figure-technical-copy").elements)
        )
        assert raw_detail in technical.text
        expansion = next(
            iter(user.find(marker="figure-technical-details").elements)
        )
        assert expansion.value is False
    finally:
        coordinator.shutdown()


async def test_figure_warning_error_and_invalid_style_keep_raw_detail_technical(
    tmp_path,
    user,
):
    from dataclasses import replace

    from nicegui import ui

    from lte_scenario_toolkit.gui.i18n import Translator
    from lte_scenario_toolkit.gui.pages import figures
    from lte_scenario_toolkit.jobs import JobCoordinator

    coordinator = JobCoordinator()
    views = []

    @ui.page("/figures-technical-demotion")
    def figures_technical_demotion():
        views.append(
            figures.render_figures_page(
                ui,
                Translator("zh-CN"),
                tmp_path,
                coordinator,
            )
        )

    raw_warning = "figure.html.failed: traceback warning"
    raw_error = "traceback export exception"
    try:
        await user.open("/figures-technical-demotion")
        view = views[0]
        view.controller._set_state(
            replace(
                view.controller.state,
                warnings=(raw_warning,),
                errors=(
                    {
                        "code": "figure.export.failed",
                        "message": raw_error,
                    },
                ),
            )
        )
        view.timer.activate()

        await user.should_see(
            marker="figure-warning-summary-0",
            content=(
                "\u56fe\u8868\u64cd\u4f5c\u5df2\u5b8c\u6210\uff0c"
                "\u4f46\u51fa\u73b0\u4e86\u8b66\u544a"
            ),
        )
        await user.should_see(
            marker="figure-error-summary",
            content="\u56fe\u8868\u64cd\u4f5c\u672a\u80fd\u5b8c\u6210",
        )
        warning = next(
            iter(user.find(marker="figure-warning-summary-0").elements)
        )
        error = next(iter(user.find(marker="figure-error-summary").elements))
        assert raw_warning not in warning.text
        assert raw_error not in error.text
        technical = next(
            iter(user.find(marker="figure-technical-copy").elements)
        )
        assert raw_warning in technical.text
        assert raw_error in technical.text

        user.find(marker="figure-dpi").clear().type("0")
        await user.should_see(
            "\u8bf7\u68c0\u67e5\u56fe\u8868\u6837\u5f0f\u8bbe\u7f6e"
        )
        assert "DPI must be a positive integer" in technical.text
        assert not user.notify.contains("DPI must be a positive integer")
    finally:
        coordinator.shutdown()


def test_figure_controller_rejects_style_change_while_preview_future_is_unfinished(
    tmp_path,
    monkeypatch,
):
    from dataclasses import replace
    from threading import Event

    from lte_scenario_toolkit.figure_service import FigureService
    from lte_scenario_toolkit.gui.pages import figures
    from lte_scenario_toolkit.gui.pages.figures import FigureController
    from lte_scenario_toolkit.jobs import JobCoordinator

    entered = Event()
    release = Event()

    def delayed_preview(_source, _spec, output):
        entered.set()
        assert release.wait(30)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"preview")
        return Path(output)

    monkeypatch.setattr(FigureService, "preview", staticmethod(delayed_preview))
    monkeypatch.setattr(figures, "_valid_preview", lambda path: path.is_file())
    coordinator = JobCoordinator()
    controller = FigureController(
        tmp_path,
        coordinator,
        source=_task15_figure_source(tmp_path),
    )
    job = None
    try:
        job = controller.refresh_preview()
        assert job is not None and job.future is not None
        assert entered.wait(2)
        revision = controller.state.revision
        spec = controller.state.spec

        with pytest.raises(RuntimeError, match="running|unfinished"):
            controller.update_spec(replace(spec, dpi=200))

        assert controller.state.phase == "previewing"
        assert controller.state.revision == revision
        assert controller.state.spec == spec
        assert not job.future.done()
    finally:
        release.set()
        if job is not None and job.future is not None:
            job.future.result(timeout=2)
            controller.drain(job)
        controller.close()
        coordinator.shutdown()


async def test_figure_page_locks_all_controls_and_restores_rejected_edits(
    tmp_path,
    monkeypatch,
    user,
):
    from dataclasses import replace
    from threading import Event

    from lte_scenario_toolkit.figure_service import FigureService
    from lte_scenario_toolkit.gui import app as app_module
    from lte_scenario_toolkit.gui.pages import figures
    from lte_scenario_toolkit.gui.settings import GuiSettingsStore
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.run_service import RunService

    entered = Event()
    release = Event()
    source_root = (tmp_path / "source-runs").resolve()
    source_service = RunService(source_root)
    source_run = source_service.begin("city", "default")
    (source_run.path / "scenario.csv").write_text("X,Y\n1,2\n", encoding="utf-8")
    completed_a = source_service.publish(
        source_run,
        status="completed",
        artifacts=["scenario.csv"],
    ).resolve()
    rejected_b = (tmp_path / "rejected-b").resolve()
    source = replace(
        _task15_figure_source(tmp_path),
        path=completed_a,
        source_kind="run",
        run_id=source_run.run_id,
    )
    views = []
    actual_render = app_module.render_figures_page

    def capture_render(*args, **kwargs):
        view = actual_render(*args, **kwargs)
        views.append(view)
        return view

    def delayed_preview(_source, _spec, output):
        entered.set()
        assert release.wait(30)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"preview")
        return Path(output)

    monkeypatch.setattr(app_module, "render_figures_page", capture_render)
    monkeypatch.setattr(figures, "load_figure_source", lambda _path: source)
    monkeypatch.setattr(FigureService, "preview", staticmethod(delayed_preview))
    monkeypatch.setattr(figures, "_valid_preview", lambda path: path.is_file())
    coordinator = JobCoordinator()
    GuiSettingsStore(tmp_path).update(add_output_roots=(source_root,))
    try:
        app_module.create_app(
            catalog=SimpleNamespace(root=tmp_path.resolve()),
            coordinator=coordinator,
            figure_source_options_provider=lambda: {
                str(completed_a): "Completed A",
                str(rejected_b): "Rejected B",
            },
            testing=True,
        )
        await user.open("/figures")
        control_markers = (
            "figure-source-path",
            "figure-load-source",
            "figure-preset",
            "figure-dpi",
            "figure-colormap",
            "figure-azimuth",
            "figure-elevation",
            "figure-vertical-exaggeration",
            "figure-station-color",
            "figure-station-size",
            "figure-title",
            "figure-max-pixels",
            "figure-format-png",
            "figure-format-eps",
            "figure-format-html",
            "figure-refresh-preview",
            "figure-export",
        )
        control_elements = {
            marker: next(iter(user.find(marker=marker).elements))
            for marker in control_markers
        }
        source_element = control_elements["figure-source-path"]
        preset_element = control_elements["figure-preset"]
        dpi_element = control_elements["figure-dpi"]
        with user.client:
            source_element.set_value(str(completed_a))
        user.find(marker="figure-load-source").click()
        source_menu_controls = {
            marker: next(iter(user.find(marker=marker).elements))
            for marker in (
                "figure-refresh-local-menu",
                "figure-open-source-menu",
            )
        }
        assert all(control.enabled for control in source_menu_controls.values())
        user.find(marker="figure-refresh-preview").click()
        assert await asyncio.to_thread(entered.wait, 2)
        assert len(views) == 1
        view = views[0]
        job = view.controller.job
        assert job is not None and job.future is not None and not job.future.done()
        revision = view.controller.state.revision
        spec = view.controller.state.spec

        enabled_while_blocked = {
            marker: element.enabled for marker, element in control_elements.items()
        }
        enabled_while_blocked.update(
            {
                marker: element.enabled
                for marker, element in source_menu_controls.items()
            }
        )

        with user.client:
            source_element.value = str(rejected_b)
            preset_element.value = "publication"
            dpi_element.value = 200

        await asyncio.sleep(0.2)
        source_value_while_blocked = source_element.value
        preset_value_while_blocked = preset_element.value
        dpi_value_while_blocked = dpi_element.value
        state_while_blocked = view.controller.state
        timer_active_while_blocked = view.timer.active
        future_done_while_blocked = job.future.done()

        release.set()
        await asyncio.to_thread(job.future.result, 2)

        assert not any(enabled_while_blocked.values())
        assert source_value_while_blocked == str(completed_a)
        assert preset_value_while_blocked == spec.preset
        assert dpi_value_while_blocked == spec.dpi
        assert state_while_blocked.source is source
        assert state_while_blocked.revision == revision
        assert state_while_blocked.phase == "previewing"
        assert timer_active_while_blocked is True
        assert future_done_while_blocked is False

        await user.should_see("Preview is current", retries=20)

        assert view.timer.active is False
        assert view.controller.state.phase == "ready"
        assert all(element.enabled for element in control_elements.values())
        assert all(control.enabled for control in source_menu_controls.values())
    finally:
        release.set()
        if views:
            views[0].timer.deactivate()
        coordinator.shutdown()


async def test_figure_page_runs_final_export_in_a_cpu_bound_process(
    tmp_path,
    monkeypatch,
    user,
):
    import pickle
    from dataclasses import replace

    from nicegui import run, ui

    from lte_scenario_toolkit.gui.i18n import Translator
    from lte_scenario_toolkit.gui.pages import figures
    from lte_scenario_toolkit.jobs import JobCoordinator

    source_path = (tmp_path / "completed-source").resolve()
    source = replace(
        _task15_figure_source(tmp_path),
        path=source_path,
        source_kind="run",
        run_id="a" * 32,
    )
    output_root = (tmp_path / "results").resolve()
    published = output_root / "city" / "default" / "published"
    published.mkdir(parents=True)
    cpu_calls = []
    remembered = []
    views = []

    async def fake_cpu_bound(callback, *args, **kwargs):
        pickle.dumps((callback, args, kwargs))
        cpu_calls.append((callback, args, kwargs))
        request = args[0]
        return figures._FigureJobResult(
            "export",
            request.revision,
            path=published,
            phase="completed",
        )

    monkeypatch.setattr(run, "cpu_bound", fake_cpu_bound)
    monkeypatch.setattr(figures, "load_figure_source", lambda _path: source)
    coordinator = JobCoordinator()

    @ui.page("/figure-cpu-export")
    def figure_cpu_export():
        views.append(
            figures.render_figures_page(
                ui,
                Translator("en"),
                tmp_path,
                coordinator,
                initial_source=source_path,
                source_options={str(source_path): "Completed source"},
                output_root=output_root,
                on_published=remembered.append,
            )
        )

    try:
        await user.open("/figure-cpu-export")
        await user.should_see(marker="figure-source-ready", content="Rectangle 1")

        user.find(marker="figure-export").click()
        for _ in range(30):
            if cpu_calls:
                break
            await asyncio.sleep(0.05)

        assert len(cpu_calls) == 1
        callback, args, kwargs = cpu_calls[0]
        assert callback is figures._render_figure_export
        assert len(args) == 1
        assert kwargs == {}
        await user.should_see(str(published))
        assert remembered == [published]
        assert views[0].controller.state.run_path == published
        assert coordinator.snapshot().active is False
    finally:
        if views:
            views[0].timer.cancel(with_current_invocation=True)
        coordinator.shutdown()


async def test_figure_page_shows_loaded_no_dem_reason_and_no_legacy_controls(
    tmp_path,
    monkeypatch,
    user,
):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.pages import figures
    from lte_scenario_toolkit.jobs import JobCoordinator

    source = replace(_task15_figure_source(tmp_path), dem_path=None)
    completed = tmp_path / "completed-without-dem"
    monkeypatch.setattr(figures, "load_figure_source", lambda _path: source)
    coordinator = JobCoordinator()
    try:
        create_app(
            catalog=SimpleNamespace(root=tmp_path.resolve()),
            coordinator=coordinator,
            figure_source_options_provider=lambda: {
                str(completed): "Completed without DEM"
            },
            testing=True,
        )
        await user.open("/figures")
        source_control = next(
            iter(user.find(marker="figure-source-path").elements)
        )
        with user.client:
            source_control.set_value(str(completed.resolve()))
        user.find(marker="figure-load-source").click()

        await user.should_see(marker="figure-source-ready", content="Rectangle 1")
        await user.should_see(
            marker="figure-terrain-unavailable",
            content="Terrain data is unavailable",
        )
        assert not next(
            iter(user.find(marker="figure-refresh-preview").elements)
        ).enabled
        assert not next(iter(user.find(marker="figure-export").elements)).enabled
        await user.should_not_see(marker="figure-csv-source")
        await user.should_not_see(marker="figure-attach-dem")
        await user.should_not_see("Attach DEM")
        await user.should_not_see("CSV file")
    finally:
        coordinator.shutdown()


def test_figure_workspace_css_has_bounded_sidebar_and_responsive_stack():
    css = (ROOT / "src/lte_scenario_toolkit/gui/assets/app.css").read_text(
        encoding="utf-8"
    )

    workspace = re.search(r"\.lte-figure-workspace\s*\{(?P<body>[^}]+)\}", css)
    assert workspace is not None
    assert "grid-template-columns: minmax(0, 1fr) minmax(300px, 360px);" in (
        workspace.group("body")
    )
    assert "overflow-wrap: anywhere;" in css
    responsive = css[css.index("@media (max-width: 980px)") :]
    assert re.search(
        r"\.lte-figure-workspace\s*\{[^}]*grid-template-columns: 1fr",
        responsive,
    )


def test_figure_workstation_css_has_source_bar_and_safe_export_dock():
    css = (ROOT / "src/lte_scenario_toolkit/gui/assets/app.css").read_text(
        encoding="utf-8"
    )

    for selector in (
        ".lte-figure-source-bar",
        ".lte-figure-source-current",
        ".lte-figure-export-dock",
    ):
        assert selector in css
    assert "env(safe-area-inset-bottom)" in css
    assert "@media (max-width: 760px)" in css
    assert "@media (max-width: 390px)" in css
    assert ".lte-figures-page *" in css[css.index("@media (prefers-reduced-motion: reduce)") :]


async def test_figure_workstation_exposes_source_menu_and_export_dock(user, tmp_path):
    from nicegui import ui

    from lte_scenario_toolkit.gui.i18n import Translator
    from lte_scenario_toolkit.gui.pages.figures import render_figures_page
    from lte_scenario_toolkit.jobs import JobCoordinator

    coordinator = JobCoordinator()

    @ui.page("/figure-workstation-composition")
    def figure_workstation_composition():
        render_figures_page(
            ui,
            Translator("en"),
            tmp_path,
            coordinator,
            source_options={str(tmp_path / "run"): "City / Default"},
        )

    try:
        await user.open("/figure-workstation-composition")
        await user.should_see(marker="figure-source-overflow")
        await user.should_see(marker="figure-source-current")
        await user.should_see(marker="figure-export-dock")
        dock = next(iter(user.find(marker="figure-export-dock").elements))
        assert dock.tag == "footer"
        primary = [
            element
            for element in dock.descendants()
            if "lte-action--primary" in getattr(element, "_classes", [])
        ]
        assert [element.text for element in primary] == ["Export Figures"]
        assert user.find(marker="figure-refresh-local").elements
        assert user.find(marker="figure-open-source").elements
    finally:
        coordinator.shutdown()


async def test_figure_export_dock_aria_label_uses_active_language(user, tmp_path):
    from nicegui import ui

    from lte_scenario_toolkit.gui.i18n import Translator
    from lte_scenario_toolkit.gui.pages.figures import render_figures_page
    from lte_scenario_toolkit.jobs import JobCoordinator

    coordinator = JobCoordinator()

    @ui.page("/figure-workstation-aria-zh")
    def figure_workstation_aria_zh():
        render_figures_page(ui, Translator("zh-CN"), tmp_path, coordinator)

    try:
        await user.open("/figure-workstation-aria-zh")
        dock = next(iter(user.find(marker="figure-export-dock").elements))
        assert dock.props["aria-label"] == Translator("zh-CN").text(
            "figures.export_actions"
        )
        assert dock.props["aria-label"] != Translator("en").text(
            "figures.export_actions"
        )
    finally:
        coordinator.shutdown()


def _task15_figure_source(tmp_path):
    import geopandas as gpd
    import pandas as pd

    from lte_scenario_toolkit.figure_service import (
        FigureSource,
        SelectionFigureIdentity,
    )
    row = {
        "rect_id": 1,
        "pt_count": 1,
        "left_x": 0.0,
        "bottom_y": 0.0,
        "center_x": 5.0,
        "center_y": 5.0,
        "X": 1.0,
        "Y": 1.0,
        "elevation": 10.0,
    }
    frame = pd.DataFrame([row])
    points = gpd.GeoDataFrame(
        frame.copy(),
        geometry=gpd.points_from_xy(frame["X"], frame["Y"]),
        crs="EPSG:3857",
    )
    dem_path = tmp_path / "controller-dem.tif"
    dem_path.write_bytes(b"dem identity")
    return FigureSource(
        path=None,
        csv_path=None,
        csv_identity=None,
        frame=frame,
        rectangle={
            key: row[key]
            for key in (
                "rect_id",
                "pt_count",
                "left_x",
                "bottom_y",
                "center_x",
                "center_y",
            )
        },
        points=points,
        target_crs="EPSG:3857",
        rectangle_size_m=10.0,
        source_kind="selection",
        dem_path=dem_path,
        dem_fingerprint="controller-dem",
        scenario_id="city",
        profile_id="default",
        selection_identity=SelectionFigureIdentity(
            scenario_id="city",
            profile_id="default",
            profile_fingerprint="profile",
            points_fingerprint="points",
            boundary_fingerprint="boundary",
            dem_fingerprint="controller-dem",
            scan_algorithm_version="row-sweep-v1",
            scan_checked_positions=1,
            scan_total_positions=1,
            candidate_index=1,
            candidate_flat_grid_id=0,
            candidate_point_count=1,
            candidate_left_x=0.0,
            candidate_bottom_y=0.0,
            candidate_center_x=5.0,
            candidate_center_y=5.0,
        ),
    )


def test_figure_controller_busy_submission_does_not_leave_running_phase(tmp_path):
    from lte_scenario_toolkit.gui.pages.figures import FigureController
    from lte_scenario_toolkit.jobs import JobBusyError, JobCoordinator

    coordinator = JobCoordinator()
    active = coordinator.start("scan")
    controller = FigureController(
        tmp_path,
        coordinator,
        source=_task15_figure_source(tmp_path),
        output_root=tmp_path / "runs",
    )
    try:
        with pytest.raises(JobBusyError):
            controller.refresh_preview()
        assert controller.state.phase == "ready"
        with pytest.raises(JobBusyError):
            controller.export(("png",))
        assert controller.state.phase == "ready"
    finally:
        assert coordinator.finish(active.job_id) is True
        controller.close()
        coordinator.shutdown()


def test_figure_target_does_not_infer_cross_root_parent_and_revalidates_explicit_parent(
    tmp_path,
):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages.figures import FigureController
    from lte_scenario_toolkit.jobs import JobCoordinator
    from lte_scenario_toolkit.run_service import RunService

    source_root = tmp_path / "source-runs"
    destination_root = tmp_path / "destination-runs"
    source_service = RunService(source_root)
    source_run = source_service.begin("city", "default")
    (source_run.path / "source.csv").write_text(
        "rect_id,pt_count,left_x,bottom_y,center_x,center_y,X,Y\n"
        "1,1,0,0,5,5,1,1\n",
        encoding="utf-8",
    )
    source_path = source_service.publish(
        source_run,
        status="completed",
        artifacts=["source.csv"],
    )
    source = replace(
        _task15_figure_source(tmp_path),
        source_kind="run",
        path=source_path,
        run_id=source_run.run_id,
        scenario_id="city",
        profile_id="default",
    )
    destination_service = RunService(destination_root)
    parent = destination_service.begin("city", "default")
    (parent.path / "source.csv").write_text("ok\n", encoding="utf-8")
    parent_path = destination_service.publish(
        parent,
        status="completed",
        artifacts=["source.csv"],
    )
    coordinator = JobCoordinator()
    controller = FigureController(
        tmp_path,
        coordinator,
        source=source,
        output_root=destination_root,
    )
    try:
        assert controller._target(source) == (destination_root.resolve(), None)
        controller.set_source(
            source,
            output_root=destination_root,
            parent_run_id=parent.run_id,
            parent_run_path=parent_path,
        )
        assert controller._target(source) == (
            destination_root.resolve(),
            parent.run_id,
        )
        (parent_path / "source.csv").unlink()
        with pytest.raises(ValueError, match="no longer|changed"):
            controller._target(source)
    finally:
        controller.close()
        coordinator.shutdown()




def test_stale_preview_job_cannot_replace_changed_style(tmp_path, monkeypatch):
    from dataclasses import replace
    from threading import Event

    from lte_scenario_toolkit.figure_service import FigureService
    from lte_scenario_toolkit.gui.pages import figures
    from lte_scenario_toolkit.gui.pages.figures import FigureController
    from lte_scenario_toolkit.jobs import JobCoordinator

    entered = Event()
    release = Event()

    def preview(source, spec, output):
        entered.set()
        assert release.wait(2)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"png")
        return Path(output)

    monkeypatch.setattr(FigureService, "preview", staticmethod(preview))
    monkeypatch.setattr(figures, "_valid_preview", lambda path: path.is_file())
    controller = FigureController(
        tmp_path,
        JobCoordinator(),
        source=_task15_figure_source(tmp_path),
    )
    try:
        job = controller.refresh_preview()
        assert job is not None and entered.wait(2)
        release.set()
        assert job.future is not None
        job.future.result(timeout=2)
        controller.update_spec(replace(controller.state.spec, dpi=200))
        state = controller.drain(job)

        assert state.spec.dpi == 200
        assert state.preview_path is None
        assert state.preview_stale is True
    finally:
        controller.close()
        controller.coordinator.shutdown()


@pytest.mark.parametrize("phase", ["partial", "error"])
def test_stale_figure_export_preserves_current_inputs_and_true_result(
    tmp_path,
    phase,
):
    from dataclasses import replace

    from lte_scenario_toolkit.gui.pages import figures
    from lte_scenario_toolkit.gui.pages.figures import FigureController
    from lte_scenario_toolkit.jobs import JobCoordinator

    coordinator = JobCoordinator()
    controller = FigureController(
        tmp_path,
        coordinator,
        source=_task15_figure_source(tmp_path),
    )
    original_revision = controller.state.revision
    run_path = tmp_path / "published-figure"
    errors = ({"artifact": "html", "code": "figure.html.failed"},)
    result = figures._FigureJobResult(
        "export",
        original_revision,
        path=run_path if phase == "partial" else None,
        phase=phase,
        warnings=("figure.settings_failed:read only",),
        errors=errors,
    )
    job = coordinator.submit("figure-export", lambda _cancel, _emit: result)
    controller._job = job
    try:
        assert job.future is not None
        job.future.result(timeout=2)
        controller.update_spec(replace(controller.state.spec, dpi=200))
        state = controller.drain(job)

        assert state.spec.dpi == 200
        assert state.phase == phase
        assert state.errors == errors
        if phase == "partial":
            assert state.run_path == run_path
            assert "figure.stale_published" in state.warnings
            assert "figure.settings_failed:read only" in state.warnings
        else:
            assert state.run_path is None
            assert "figure.stale_published" not in state.warnings
    finally:
        controller.close()
        coordinator.shutdown()


def test_history_page_model_has_no_delete_action(tmp_path):
    from lte_scenario_toolkit.gui.pages.history import HistoryAction

    assert "delete" not in {action.value for action in HistoryAction}


def test_history_cross_root_source_uses_path_and_run_id_without_collision(tmp_path):
    import shutil
    from dataclasses import replace

    from lte_scenario_toolkit.figure_service import FigureSpec
    from lte_scenario_toolkit.gui.pages.history import (
        HistoryAction,
        rebuild_history,
        resolve_history_action,
    )
    from lte_scenario_toolkit.run_service import RunService

    source_root = tmp_path / "source-runs"
    child_root = tmp_path / "child-runs"
    duplicate_root = tmp_path / "duplicate-runs"
    source_service = RunService(source_root)
    parent = source_service.begin("city", "default")
    (parent.path / "scenario.csv").write_text("ok\n", encoding="utf-8")
    parent_path = source_service.publish(
        parent,
        status="completed",
        artifacts=["scenario.csv"],
    )
    duplicate_path = duplicate_root / parent_path.relative_to(source_root)
    duplicate_path.parent.mkdir(parents=True)
    shutil.copytree(parent_path, duplicate_path)

    child_service = RunService(child_root)
    retry_spec = replace(
        FigureSpec.from_preset("publication"),
        dpi=240,
        azimuth=-35.0,
    )
    child = child_service.begin(
        "city",
        "default",
        parent_run_id=parent.run_id,
    )
    (child.path / "terrain.png").write_bytes(b"png")
    child_service.publish(
        child,
        status="partial",
        artifacts=["terrain.png"],
        metadata={
            "run_kind": "figure",
            "source": {"path": str(parent_path), "run_id": parent.run_id},
            "requested_formats": ["png", "html"],
            "artifact_paths": {"png": "terrain.png"},
            "figure_spec": retry_spec.as_dict(),
        },
        errors=[{"artifact": "html", "code": "figure.html.failed"}],
    )

    snapshot = rebuild_history(
        tmp_path,
        (source_root, child_root, duplicate_root),
    )

    child_row = next(row for row in snapshot.rows if row.root == child_root.resolve())
    assert child_row.figure_source_path == parent_path.resolve()
    assert child_row.retry_formats == ("html",)
    action = resolve_history_action(child_row, HistoryAction.RETRY_MISSING)
    assert action.retry_formats == ("html",)
    assert action.figure_spec == retry_spec
    assert action.path == parent_path.resolve()
    assert action.destination_root == child_root.resolve()
    assert action.derived_parent_run_id == child.run_id


def test_history_action_relinks_current_intermediate_source_chain(tmp_path):
    from lte_scenario_toolkit.gui.pages.history import (
        HistoryAction,
        history_rows,
        resolve_history_action,
    )
    from lte_scenario_toolkit.run_service import RunService

    service = RunService(tmp_path / "runs")

    def completed_csv(name, created_at):
        run = service.begin("city", "default", created_at=created_at)
        (run.path / f"{name}.csv").write_text("X,Y\n1,2\n", encoding="utf-8")
        path = service.publish(
            run,
            status="completed",
            artifacts=[f"{name}.csv"],
            metadata={"run_kind": "selection"},
        )
        return run, path

    source_a, path_a = completed_csv("source-a", "2026-07-16T10:00:00Z")
    source_d, path_d = completed_csv("source-d", "2026-07-16T10:00:01Z")
    intermediate = service.begin(
        "city",
        "default",
        created_at="2026-07-16T10:00:02Z",
    )
    (intermediate.path / "intermediate.png").write_bytes(b"png")
    intermediate_path = service.publish(
        intermediate,
        status="completed",
        artifacts=["intermediate.png"],
        metadata={
            "run_kind": "figure",
            "source": {"path": str(path_a), "run_id": source_a.run_id},
        },
    )
    child = service.begin(
        "city",
        "default",
        created_at="2026-07-16T10:00:03Z",
        parent_run_id=intermediate.run_id,
    )
    (child.path / "child.png").write_bytes(b"png")
    service.publish(
        child,
        status="completed",
        artifacts=["child.png"],
        metadata={
            "run_kind": "figure",
            "source": {
                "path": str(intermediate_path),
                "run_id": intermediate.run_id,
            },
        },
    )
    child_row = next(
        row for row in history_rows(service) if row.run_id == child.run_id
    )
    assert child_row.figure_source_path == path_a.resolve()

    manifest_path = intermediate_path / "run.json"
    rewired = json.loads(manifest_path.read_text(encoding="utf-8"))
    rewired["metadata"]["source"] = {
        "path": str(path_d),
        "run_id": source_d.run_id,
    }
    manifest_path.write_text(json.dumps(rewired), encoding="utf-8")

    action = resolve_history_action(child_row, HistoryAction.OPEN_FIGURES)
    assert action.path == path_d.resolve()
    assert action.path != path_a.resolve()
    assert action.run_id == source_d.run_id


def test_history_action_rejects_completed_source_without_current_csv(tmp_path):
    from lte_scenario_toolkit.gui.pages.history import (
        HistoryAction,
        HistoryActionError,
        _figure_source_path,
        history_rows,
        resolve_history_action,
    )
    from lte_scenario_toolkit.run_service import RunService

    service = RunService(tmp_path / "runs")
    source = service.begin("city", "default", created_at="2026-07-16T10:00:00Z")
    (source.path / "source.csv").write_text("X,Y\n1,2\n", encoding="utf-8")
    source_path = service.publish(
        source,
        status="completed",
        artifacts=["source.csv"],
        metadata={"run_kind": "selection"},
    )
    child = service.begin(
        "city",
        "default",
        created_at="2026-07-16T10:00:01Z",
        parent_run_id=source.run_id,
    )
    (child.path / "terrain.png").write_bytes(b"png")
    service.publish(
        child,
        status="completed",
        artifacts=["terrain.png"],
        metadata={
            "run_kind": "figure",
            "source": {"path": str(source_path), "run_id": source.run_id},
        },
    )
    child_row = next(
        row for row in history_rows(service) if row.run_id == child.run_id
    )
    assert child_row.figure_source_path == source_path.resolve()

    manifest_path = source_path / "run.json"
    without_csv = json.loads(manifest_path.read_text(encoding="utf-8"))
    without_csv["artifacts"] = []
    (source_path / "source.csv").unlink()
    manifest_path.write_text(json.dumps(without_csv), encoding="utf-8")

    with pytest.raises(ValueError, match="CSV"):
        _figure_source_path(source_path, without_csv)
    with pytest.raises(HistoryActionError, match="compatible figure source"):
        resolve_history_action(child_row, HistoryAction.OPEN_FIGURES)


def test_history_source_cycle_is_non_actionable_instead_of_recursive(tmp_path):
    from lte_scenario_toolkit.gui.pages.history import history_rows
    from lte_scenario_toolkit.run_service import RunService

    service = RunService(tmp_path / "runs")
    first = service.begin("city", "default")
    second = service.begin("city", "default")
    for run in (first, second):
        (run.path / "terrain.png").write_bytes(b"png")
    service.publish(
        first,
        status="completed",
        artifacts=["terrain.png"],
        metadata={
            "run_kind": "figure",
            "source": {"path": str(second.final_path), "run_id": second.run_id},
        },
    )
    service.publish(
        second,
        status="completed",
        artifacts=["terrain.png"],
        metadata={
            "run_kind": "figure",
            "source": {"path": str(first.final_path), "run_id": first.run_id},
        },
    )

    rows = history_rows(service)

    assert len(rows) == 2
    assert all(row.can_open_figures is False for row in rows)


def test_history_action_revalidates_clicked_child_before_opening_parent(tmp_path):
    from lte_scenario_toolkit.gui.pages.history import (
        HistoryAction,
        HistoryActionError,
        history_rows,
        resolve_history_action,
    )
    from lte_scenario_toolkit.run_service import RunService

    service = RunService(tmp_path / "runs")
    parent = service.begin("city", "default")
    (parent.path / "scenario.csv").write_text("ok\n", encoding="utf-8")
    service.publish(parent, status="completed", artifacts=["scenario.csv"])
    child = service.begin("city", "default", parent_run_id=parent.run_id)
    (child.path / "terrain.png").write_bytes(b"png")
    child_path = service.publish(
        child,
        status="completed",
        artifacts=["terrain.png"],
    )
    child_row = next(row for row in history_rows(service) if row.run_id == child.run_id)
    (child_path / "terrain.png").unlink()

    with pytest.raises(HistoryActionError, match="no longer"):
        resolve_history_action(child_row, HistoryAction.OPEN_FIGURES)


async def test_task15_output_routes_render_offline(user, tmp_path, monkeypatch):
    import socket
    import urllib.request

    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.settings import GuiSettingsStore
    from lte_scenario_toolkit.run_service import RunService

    def forbid_network(*args, **kwargs):
        pytest.fail("Task 15 pages must remain offline")

    monkeypatch.setattr(socket, "create_connection", forbid_network)
    monkeypatch.setattr(urllib.request, "urlopen", forbid_network)
    run_root = tmp_path / "external-runs"
    service = RunService(run_root)
    run = service.begin("city", "default", created_at="2026-07-16T10:00:00Z")
    (run.path / "scenario.csv").write_text("ok\n", encoding="utf-8")
    service.publish(run, status="completed", artifacts=["scenario.csv"])
    GuiSettingsStore(tmp_path).save(language="en", output_roots=[run_root])
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=object(),
        testing=True,
    )

    await user.open("/generate")
    await user.should_see("Generation session unavailable")
    await user.open("/figures")
    await user.should_see("Figures")
    await user.should_see("Figure source")
    await user.open("/history")
    await user.should_see("Run History")
    await user.should_see("city", retries=15)
    await user.should_not_see("Delete")


async def test_history_route_renders_loading_shell_before_discovery_finishes(
    user,
    tmp_path,
    monkeypatch,
):
    import lte_scenario_toolkit.gui.app as app_module
    from lte_scenario_toolkit.gui.app import create_app

    entered = Event()
    release = Event()
    original = app_module.rebuild_history

    def delayed_rebuild(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(app_module, "rebuild_history", delayed_rebuild)
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=object(),
        testing=True,
    )

    open_task = asyncio.create_task(user.open("/history"))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        await user.should_see(marker="shell-navigation")
        await user.should_see("Run History")
        await user.should_see(marker="history-loading")
        await user.should_see("Loading run history")
    finally:
        release.set()
        await asyncio.wait_for(open_task, timeout=4)

    await user.should_not_see(marker="history-loading")
    await user.should_see("No published runs were found")


async def test_history_route_replaces_failed_loader_with_durable_recovery(
    user,
    tmp_path,
    monkeypatch,
):
    import lte_scenario_toolkit.gui.app as app_module
    from lte_scenario_toolkit.gui.app import create_app

    entered = Event()
    release = Event()

    def failed_rebuild(*_args, **_kwargs):
        entered.set()
        assert release.wait(3)
        raise RuntimeError("history index traceback SECRET-PATH")

    monkeypatch.setattr(app_module, "rebuild_history", failed_rebuild)
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=object(),
        testing=True,
    )

    open_task = asyncio.create_task(user.open("/history"))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        await user.should_see(marker="history-loading")
    finally:
        release.set()
        await asyncio.wait_for(open_task, timeout=4)

    await user.should_not_see(marker="history-loading")
    await user.should_see(
        marker="history-load-error",
        content="Run history could not be loaded",
    )
    primary = next(iter(user.find(marker="history-load-error").elements))
    assert "SECRET-PATH" not in primary.text
    technical = next(
        iter(user.find(marker="history-load-error-technical-copy").elements)
    )
    assert "RuntimeError" in technical.text
    assert "SECRET-PATH" in technical.text
    await user.should_see(marker="history-refresh")
    await user.should_see(marker="shell-menu")


async def test_history_cards_prioritize_human_summary_and_gate_retry_action(
    user,
    tmp_path,
):
    from lte_scenario_toolkit.figure_service import FigureSpec
    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.settings import GuiSettingsStore
    from lte_scenario_toolkit.run_service import RunService

    run_root = tmp_path / "external-runs"
    service = RunService(run_root)
    parent = service.begin("city", "default", created_at="2026-07-16T10:00:00Z")
    (parent.path / "scenario.csv").write_text("ok\n", encoding="utf-8")
    parent_path = service.publish(
        parent,
        status="completed",
        artifacts=["scenario.csv"],
        metadata={
            "run_kind": "selection",
            "parameters": {"rectangle_size_m": 2000},
            "candidate": {"flat_grid_id": 42},
        },
    )
    child = service.begin(
        "city",
        "default",
        created_at="2026-07-16T10:01:00Z",
        parent_run_id=parent.run_id,
    )
    (child.path / "terrain.png").write_bytes(b"png")
    service.publish(
        child,
        status="partial",
        artifacts=["terrain.png"],
        metadata={
            "run_kind": "figure",
            "source": {"path": str(parent_path), "run_id": parent.run_id},
            "requested_formats": ["png", "html"],
            "artifact_paths": {"png": "terrain.png"},
            "figure_spec": FigureSpec.from_preset("publication").as_dict(),
        },
        errors=[
            {
                "artifact": "html",
                "code": "figure.html.failed",
                "message": "plotly traceback",
            }
        ],
    )
    GuiSettingsStore(tmp_path).save(language="en", output_roots=[run_root])
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=object(),
        testing=True,
    )

    await user.open("/history")
    await user.should_see(marker=f"history-primary-{child.run_id}", retries=20)
    child_primary = next(
        iter(user.find(marker=f"history-primary-{child.run_id}").elements)
    )
    assert "city / default" in child_primary.text
    status = next(iter(user.find(marker=f"history-status-{child.run_id}").elements))
    artifact_count = next(
        iter(user.find(marker=f"history-artifact-count-{child.run_id}").elements)
    )
    lineage = next(iter(user.find(marker=f"history-lineage-{child.run_id}").elements))
    assert status.text == "Partial"
    assert artifact_count.text == "1 artifact"
    assert lineage.text == "Derived from an earlier run"
    primary_copy = " ".join(
        (child_primary.text, status.text, artifact_count.text, lineage.text)
    )
    for machine_detail in (
        child.run_id,
        parent.run_id,
        "terrain.png",
        "figure.html.failed",
        "plotly traceback",
    ):
        assert machine_detail not in primary_copy

    technical = next(
        iter(user.find(marker=f"history-technical-copy-{child.run_id}").elements)
    )
    assert child.run_id in technical.content
    assert parent.run_id in technical.content
    assert "terrain.png" in technical.content
    assert "figure.html.failed" in technical.content
    await user.should_see(marker=f"history-inspect-{child.run_id}")
    await user.should_see(marker=f"history-open-{child.run_id}")
    await user.should_see(marker=f"history-reveal-{child.run_id}")
    await user.should_see(marker=f"history-retry-{child.run_id}")
    await user.should_not_see(marker=f"history-retry-{parent.run_id}")
    inspect = next(iter(user.find(marker=f"history-inspect-{child.run_id}").elements))
    open_figures = next(iter(user.find(marker=f"history-open-{child.run_id}").elements))
    retry = next(iter(user.find(marker=f"history-retry-{child.run_id}").elements))
    reveal = next(iter(user.find(marker=f"history-reveal-{child.run_id}").elements))
    assert "outline" not in inspect.props
    assert "outline" not in open_figures.props
    assert "outline" in retry.props
    assert "unelevated" not in retry.props
    assert "outline" in reveal.props


async def test_offline_candidate_to_history_flow(user, tmp_path, monkeypatch):
    import socket
    import urllib.request

    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.gui.pages.candidates import CandidateSessionRegistry
    from lte_scenario_toolkit.gui.settings import GuiSettingsStore
    from lte_scenario_toolkit.run_service import RunService

    export_calls = []

    def forbid_network(*args, **kwargs):
        pytest.fail("candidate-to-history flow attempted external network access")

    monkeypatch.setattr(socket, "create_connection", forbid_network)
    monkeypatch.setattr(urllib.request, "urlopen", forbid_network)

    class EmptyStore:
        def discover(self, scenario_id):
            return []

    class Service:
        def preflight(self, snapshot, output_root):
            return SimpleNamespace(profile=snapshot, output_root=Path(output_root))

        def scan(self, *args, **kwargs):
            return _task14_scan_result()

        def export(
            self,
            preflight,
            scan_result,
            candidate,
            *,
            output_root,
            artifacts,
            entrypoint,
        ):
            export_calls.append(
                (
                    preflight,
                    scan_result,
                    candidate,
                    Path(output_root),
                    tuple(artifacts),
                    tuple(entrypoint),
                )
            )
            names = {
                "csv": "scenario.csv",
                "preview_png": "preview.png",
                "terrain_png": "terrain.png",
                "terrain_eps": "terrain.eps",
                "terrain_html": "terrain.html",
            }
            service = RunService(output_root)
            run = service.begin(
                preflight.profile.scenario_id,
                preflight.profile.profile_id,
            )
            published = []
            artifact_paths = {}
            for token in artifacts:
                name = names[token]
                (run.path / name).write_bytes(f"{token}\n".encode("ascii"))
                published.append(name)
                artifact_paths[token] = name
            return service.publish(
                run,
                status="completed",
                artifacts=published,
                metadata={
                    "run_kind": "selection",
                    "requested_artifacts": list(artifacts),
                    "artifact_paths": artifact_paths,
                    "candidate": {
                        "flat_grid_id": candidate.flat_grid_id,
                        "point_count": candidate.point_count,
                        "center_x": candidate.center_x,
                        "center_y": candidate.center_y,
                    },
                    "parameters": {
                        "rectangle_size_m": preflight.profile.rect_size,
                    },
                },
            )

    service = Service()
    registry = CandidateSessionRegistry(max_sessions=1)
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=EmptyStore(),
        selection_service_factory=lambda _catalog: service,
        candidate_bundle_builder=lambda _session, _assets: _task14_map_bundle(tmp_path),
        candidate_registry=registry,
        testing=True,
    )
    output_root = (tmp_path / "results").resolve()
    assert not output_root.exists()
    await user.open("/configure/without-default")
    user.find(marker="profile-start-scan").click()
    for _ in range(30):
        if user.back_history and "/candidates/" in user.back_history[-1]:
            break
        await asyncio.sleep(0.05)
    assert "/candidates/" in user.back_history[-1]
    for _ in range(30):
        next_button = next(iter(user.find(marker="candidate-next").elements))
        if next_button.enabled:
            break
        await asyncio.sleep(0.05)
    user.find(marker="candidate-next").click()
    await user.should_see("Grid ID 0")
    user.find(marker="candidate-confirm").click()
    for _ in range(30):
        if user.back_history and "/generate/" in user.back_history[-1]:
            break
        await asyncio.sleep(0.05)
    assert "/generate/" in user.back_history[-1]
    assert not output_root.exists()
    await user.should_see("Generate Scenario")

    user.find(marker="generation-submit").click()
    await user.should_see(marker="generation-phase", content="All artifacts generated")
    await user.should_see(marker="generation-open-figures")
    assert registry.confirmed_sessions() == ()
    user.find(marker="generation-open-figures").click()
    await user.should_see("Figure source")
    await user.should_see("Use Current Selection")
    assert len(export_calls) == 1
    assert export_calls[0][3] == output_root
    assert export_calls[0][4] == (
        "csv",
        "preview_png",
        "terrain_png",
        "terrain_eps",
        "terrain_html",
    )
    assert export_calls[0][5] == ("lte-gui", "generate")
    assert output_root in GuiSettingsStore(tmp_path).load().output_roots
