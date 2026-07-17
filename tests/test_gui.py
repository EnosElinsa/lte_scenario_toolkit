from __future__ import annotations

import asyncio
import importlib
import json
import os
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


def test_gui_test_dependencies_and_async_mode_are_declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "pytest-asyncio>=0.24" in project["project"]["optional-dependencies"]["dev"]
    assert project["tool"]["pytest"]["ini_options"]["asyncio_mode"] == "auto"
    assert project["tool"]["pytest"]["ini_options"]["main_file"] == ""


def test_gui_css_is_declared_as_package_data():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["tool"]["setuptools"]["package-data"]["lte_scenario_toolkit"] == [
        "gui/assets/*.css"
    ]


def test_translation_dictionaries_have_identical_keys_and_format_values():
    module = _gui_module("i18n")

    assert hasattr(module, "TRANSLATIONS")
    assert hasattr(module, "Translator")
    assert set(module.TRANSLATIONS) == {"en", "zh-CN"}
    assert set(module.TRANSLATIONS["en"]) == set(module.TRANSLATIONS["zh-CN"])
    assert module.Translator("zh-CN").text("nav.scenarios") == "\u573a\u666f"
    assert module.Translator("en").text("job.running", name="Scan 1") == "Running Scan 1"


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
    assert (tmp_path / ".lte-data/gui-settings.json").is_file()
    assert not list((tmp_path / ".lte-data").glob("*.tmp"))


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
    fake_app = object()

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
    assert "--lte-canvas: #ffffff" in calls["css"]
    assert calls["page"][0] == "/"


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
    assert not (tmp_path / ".lte-data").exists()


@pytest.mark.parametrize(
    "document",
    [
        {"schema_version": 1, "language": "fr", "output_roots": []},
        {"schema_version": True, "language": "en", "output_roots": []},
        {"schema_version": 1, "language": "en", "output_roots": "outputs"},
        {"schema_version": 1, "language": "en", "output_roots": ["relative"]},
        {
            "schema_version": 1,
            "language": "en",
            "output_roots": [],
            "unknown": True,
        },
    ],
)
def test_gui_settings_reject_malformed_documents(tmp_path, document):
    module = _gui_module("settings")
    path = tmp_path / ".lte-data/gui-settings.json"
    path.parent.mkdir()
    path.write_text(json.dumps(document), encoding="utf-8")
    original = path.read_bytes()

    with pytest.raises(module.GuiSettingsError):
        module.GuiSettingsStore(tmp_path).load()

    assert path.read_bytes() == original


def test_gui_settings_reject_invalid_json_without_overwriting_it(tmp_path):
    module = _gui_module("settings")
    path = tmp_path / ".lte-data/gui-settings.json"
    path.parent.mkdir()
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(module.GuiSettingsError, match="Could not read"):
        module.GuiSettingsStore(tmp_path).load()

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
    user.find("English").click()
    user.find("\u7b80\u4f53\u4e2d\u6587").click()
    await user.should_see("\u573a\u666f")
    assert json.loads(
        (tmp_path / ".lte-data/gui-settings.json").read_text(encoding="utf-8")
    )["language"] == "zh-CN"


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
        schema_version=2,
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
    assert draft.migration_error is None
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


def test_configure_model_reports_legacy_yaml_without_parsing_or_writing(tmp_path):
    from lte_scenario_toolkit.gui.pages.configure import configure_model

    class LegacyStore:
        def discover(self, scenario_id):
            raise ValueError("Missing required configuration value: schema_version")

    model = configure_model(_Task13Catalog(tmp_path), LegacyStore(), "ready-city")

    assert model.is_persisted is False
    assert model.migration_error is not None
    assert "schema version 2" in model.migration_error
    assert not (tmp_path / ".lte-data").exists()


def test_unrelated_legacy_discovery_error_does_not_block_profileless_draft(tmp_path):
    from lte_scenario_toolkit.gui.pages.configure import configure_model

    class LegacyStore:
        def discover(self, scenario_id):
            raise ValueError("Missing required configuration value: schema_version")

    model = configure_model(
        _Task13Catalog(tmp_path),
        LegacyStore(),
        "without-default",
    )

    assert model.is_persisted is False
    assert model.migration_error is None
    assert model.management_error is not None
    assert model.can_start is True
    assert model.profile.schema_version == 2


def test_real_profile_store_degrades_cleanly_with_unrelated_legacy_yaml(tmp_path):
    from lte_scenario_toolkit.gui.pages.configure import configure_model
    from lte_scenario_toolkit.profiles import ProfileStore

    legacy = tmp_path / "configs/legacy.yaml"
    legacy.parent.mkdir()
    legacy.write_text("experiment:\n  name: legacy\n", encoding="utf-8")
    store = ProfileStore(tmp_path, tmp_path / "data/datasets.yaml")
    catalog = _Task13Catalog(tmp_path)

    draft = configure_model(catalog, store, "without-default")
    blocked_selection = configure_model(
        catalog,
        store,
        "without-default",
        selected_profile_id="default",
    )

    assert draft.management_error is not None
    assert draft.migration_error is None
    assert draft.can_start is True
    assert blocked_selection.migration_error is None
    assert blocked_selection.selection_error is not None
    assert blocked_selection.can_start is False


async def test_legacy_blocked_profile_management_keeps_only_draft_run_available(
    user, tmp_path
):
    from lte_scenario_toolkit.gui.app import create_app
    from lte_scenario_toolkit.profiles import ProfileStore

    legacy = tmp_path / "configs/legacy.yaml"
    legacy.parent.mkdir()
    legacy.write_text("experiment:\n  name: legacy\n", encoding="utf-8")
    store = ProfileStore(tmp_path, tmp_path / "data/datasets.yaml")
    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=store,
        testing=True,
    )

    await user.open("/configure/without-default")
    await user.should_see("Profile management temporarily unavailable")
    assert all(not element.enabled for element in user.find(marker="profile-save").elements)
    assert all(element.enabled for element in user.find(marker="profile-start-scan").elements)

    await user.open("/configure/without-default?profile=default")
    await user.should_see("Saved profile temporarily unavailable")
    assert all(
        not element.enabled
        for element in user.find(marker="profile-start-scan").elements
    )
    await user.should_not_see("lte-select-sites --config 'None'")


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

    await user.open("/")
    await user.should_see("Ready City")
    await user.should_see("Pending City")
    await user.should_see("dem-pending")
    assert all(
        element.enabled
        for element in user.find(marker="scenario-configure-ready-city").elements
    )
    assert all(
        not element.enabled
        for element in user.find(marker="scenario-configure-pending-city").elements
    )
    assert all(
        element.enabled
        for element in user.find(marker="scenario-run-ready-city").elements
    )
    assert all(
        not element.enabled
        for element in user.find(marker="scenario-run-pending-city").elements
    )

    await user.open("/scenarios")
    await user.should_see("Ready City")
    await user.should_see("Pending City")
    await user.should_see("Register a scenario with the CLI")
    assert not (tmp_path / ".lte-data").exists()

    await user.open("/configure")
    await user.should_see("Choose a scenario")


async def test_configure_route_shows_legacy_migration_placeholder(user, tmp_path):
    from lte_scenario_toolkit.gui.app import create_app

    class LegacyStore:
        def discover(self, scenario_id):
            raise ValueError("Missing required configuration value: schema_version")

    create_app(
        catalog=_Task13Catalog(tmp_path),
        profile_store=LegacyStore(),
        testing=True,
    )

    await user.open("/configure/ready-city")
    await user.should_see("Profile migration required")
    await user.should_see("schema version 2")
    await user.should_see("Start Scan")
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
    await user.should_see("Scenario is not ready")
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
    user.find(marker="profile-copy-cancel").click()
    assert calls == []

    user.find(marker="profile-copy").click()
    user.find(marker="profile-copy-confirm").click()
    assert calls == [
        (profile.source_path, "default-copy", "Default Copy")
    ]


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
        "profile-save",
        "profile-start-scan",
    ):
        assert all(not element.enabled for element in user.find(marker=marker).elements)

    user.find(marker="profile-copy").click()
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


async def test_legacy_cli_guidance_quotes_unsafe_powershell_path(user, tmp_path):
    from lte_scenario_toolkit.gui.app import create_app

    class LegacyStore:
        def discover(self, scenario_id):
            raise ValueError("Missing required configuration value: schema_version")

    catalog = _Task13Catalog(tmp_path)
    catalog.scenarios_by_id["ready-city"] = {
        **catalog.scenarios_by_id["ready-city"],
        "config_path": "configs/legacy; profile.yaml",
    }
    create_app(catalog=catalog, profile_store=LegacyStore(), testing=True)

    await user.open("/configure/ready-city")

    await user.should_see(
        "lte-select-sites --config 'configs/legacy; profile.yaml'"
    )


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

    assert calls == []


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


def test_candidate_session_registry_is_bounded_opaque_and_confirms_final_identity(tmp_path):
    from lte_scenario_toolkit.gui.pages.candidates import CandidateSessionRegistry

    service = object()
    registry = CandidateSessionRegistry(max_sessions=2)
    first = registry.add(_task14_session(tmp_path, service, session_id="opaque-1"))
    second = registry.add(_task14_session(tmp_path, service, session_id="opaque-2"))
    third = registry.add(_task14_session(tmp_path, service, session_id="opaque-3"))

    assert first.session_id == "opaque-1"
    assert registry.get("opaque-1") is None
    assert registry.get(second.session_id) is second
    result = _task14_scan_result()
    updated = registry.set_scan_result(third.session_id, result)
    confirmed = registry.confirm(third.session_id, 4)
    assert updated.scan_result is result
    assert confirmed.confirmed_flat_grid_id == 4
    assert confirmed.locked_candidate is result.candidates[1]
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

    overlay = tmp_path / "overview.png"
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
    assert not any(isinstance(layer, TileLayer) for layer in map_element.layers)
    assert not any("tile.osm" in str(layer) for layer in map_element.layers)

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
    assert {layer.args[1]["color"] for layer in rectangles} == {"#dc3f4f"}

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

    selected_png = tmp_path / "selected.png"
    selected_png.write_bytes(b"png")
    hillshade_png = tmp_path / "hillshade.png"
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
