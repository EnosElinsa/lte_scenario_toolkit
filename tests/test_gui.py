from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
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
