"""Command-line entrypoint and application factory for the local GUI."""

from __future__ import annotations

import argparse
import ipaddress
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from ..data_catalog import CatalogError, DataCatalog, load_data_catalog
from ..jobs import JobCoordinator
from ..profiles import ProfileStore
from ..selection_service import SelectionService
from .i18n import Translator, validate_translations
from .layout import render_app_shell
from .pages.configure import render_configure_page, render_configure_picker
from .pages.scenarios import (
    get_job_coordinator,
    render_scenarios_page,
    shutdown_job_coordinator,
)
from .settings import GuiSettingsError, GuiSettingsStore

GUI_INSTALL_INSTRUCTION = 'python -m pip install -e ".[gui]"'


class _EmptyProfileStore:
    def discover(self, scenario_id: str | None = None) -> list[Any]:
        return []


def _default_profile_store(catalog: Any) -> Any:
    path = getattr(catalog, "path", None)
    if path is None:
        return _EmptyProfileStore()
    return ProfileStore(catalog.root, path)


def _default_catalog_loader(catalog: Any) -> Any:
    path = getattr(catalog, "path", None)
    if path is None:
        return catalog
    return load_data_catalog(path, repo_root=catalog.root)


@dataclass(slots=True)
class GuiRuntime:
    """Mutable application services refreshed after repository profile writes."""

    catalog: Any
    profile_store: Any
    catalog_loader: Callable[[Any], Any] = _default_catalog_loader
    selection_service_factory: Callable[[Any], Any] = SelectionService
    coordinator: JobCoordinator = field(default_factory=get_job_coordinator)
    selection_service: Any = field(init=False)

    def __post_init__(self) -> None:
        self.selection_service = self.selection_service_factory(self.catalog)

    def refresh_after_profile_mutation(self) -> Any:
        """Reload catalog-owned defaults and rebuild the selection service."""

        refreshed_catalog = self.catalog_loader(self.catalog)
        refreshed_selection_service = self.selection_service_factory(
            refreshed_catalog
        )
        self.catalog = refreshed_catalog
        self.selection_service = refreshed_selection_service
        return self.catalog


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _is_loopback(host: str) -> bool:
    normalized = host.strip().lower().removeprefix("[").removesuffix("]")
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _resolved_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    repo_root = args.repo_root.expanduser().resolve()
    catalog_path = args.catalog.expanduser()
    if not catalog_path.is_absolute():
        catalog_path = repo_root / catalog_path
    return repo_root, catalog_path.resolve()


def _preflight(repo_root: Path, catalog_path: Path) -> DataCatalog:
    validate_translations()
    settings_store = GuiSettingsStore(repo_root)
    settings_store.load()
    return load_data_catalog(catalog_path, repo_root=repo_root)


def _css_text() -> str:
    return (
        resources.files("lte_scenario_toolkit.gui")
        .joinpath("assets", "app.css")
        .read_text(encoding="utf-8")
    )


def create_app(
    catalog: DataCatalog | None = None,
    testing: bool = False,
    *,
    profile_store: Any | None = None,
    catalog_loader: Callable[[Any], Any] | None = None,
    selection_service_factory: Callable[[Any], Any] = SelectionService,
    coordinator: JobCoordinator | None = None,
):
    """Register and return the NiceGUI application for one validated catalog."""

    from nicegui import app, ui

    if catalog is None:
        repo_root = Path.cwd().resolve()
        catalog = load_data_catalog(
            repo_root / "data" / "datasets.yaml",
            repo_root=repo_root,
        )
    repo_root = Path(catalog.root).resolve()
    store = GuiSettingsStore(repo_root)
    initial_settings = store.load()
    validate_translations()
    uses_shared_coordinator = coordinator is None
    runtime = GuiRuntime(
        catalog,
        profile_store=(
            profile_store if profile_store is not None else _default_profile_store(catalog)
        ),
        catalog_loader=catalog_loader or _default_catalog_loader,
        selection_service_factory=selection_service_factory,
        coordinator=get_job_coordinator() if coordinator is None else coordinator,
    )

    ui.add_css(_css_text(), shared=True)

    if hasattr(app, "on_shutdown") and uses_shared_coordinator:
        app.on_shutdown(shutdown_job_coordinator)

    page_options = {
        "title": Translator(initial_settings.language).text("app.title"),
        "reconnect_timeout": 0 if testing else 3.0,
    }

    def render_shell(active_route: str, body: Callable[[Translator], None]) -> None:
        settings = store.load()
        translator = Translator(settings.language)

        def change_language(event) -> None:
            try:
                store.save(
                    language=event.value,
                    output_roots=settings.output_roots,
                )
            except GuiSettingsError as exc:
                ui.notify(str(exc), type="negative")
                return
            ui.navigate.reload()

        content = render_app_shell(
            ui,
            translator,
            active_route=active_route,
            active_job=None,
            on_language_change=change_language,
        )
        with content:
            body(translator)

    @ui.page("/configure/{scenario_id}", **page_options)
    def configure_scenario(scenario_id: str, profile: str | None = None) -> None:
        render_shell(
            "/configure",
            lambda translator: render_configure_page(
                ui,
                translator,
                runtime.catalog,
                runtime.profile_store,
                runtime.selection_service,
                scenario_id=scenario_id,
                selected_profile_id=profile,
                on_profile_mutation=runtime.refresh_after_profile_mutation,
            ),
        )

    @ui.page("/configure", **page_options)
    def configure_picker() -> None:
        render_shell(
            "/configure",
            lambda translator: render_configure_picker(
                ui,
                translator,
                runtime.catalog,
            ),
        )

    @ui.page("/scenarios", **page_options)
    def scenarios() -> None:
        render_shell(
            "/scenarios",
            lambda translator: render_scenarios_page(
                ui,
                translator,
                runtime.catalog,
                coordinator=runtime.coordinator,
            ),
        )

    @ui.page("/", **page_options)
    def index() -> None:
        render_shell(
            "/scenarios",
            lambda translator: render_scenarios_page(
                ui,
                translator,
                runtime.catalog,
                coordinator=runtime.coordinator,
            ),
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    """Build the local GUI command-line parser without importing NiceGUI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="repository root (default: current directory)",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/datasets.yaml"),
        help="schema-v2 dataset catalog, relative to --repo-root by default",
    )
    parser.add_argument("--host", default="127.0.0.1", help="server bind host")
    parser.add_argument("--port", type=_port, default=8080, help="server port")
    parser.add_argument(
        "--no-browser",
        dest="open_browser",
        action="store_false",
        help="do not open the default browser",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate local GUI inputs without starting the server",
    )
    parser.set_defaults(open_browser=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate local inputs and start the loopback-first NiceGUI server."""

    args = build_parser().parse_args(argv)
    repo_root, catalog_path = _resolved_inputs(args)
    if not _is_loopback(args.host):
        print(
            f"WARNING: lte-gui is binding to non-loopback host {args.host!r}; "
            "the application can read and write local experiment paths, so use "
            "a trusted network only.",
            file=sys.stderr,
        )

    try:
        import nicegui
    except ModuleNotFoundError as exc:
        if exc.name != "nicegui":
            raise
        print(GUI_INSTALL_INSTRUCTION, file=sys.stderr)
        return 2

    try:
        catalog = _preflight(repo_root, catalog_path)
    except (CatalogError, GuiSettingsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.check:
        print("GUI preflight OK")
        return 0

    create_app(catalog=catalog)
    print("GUI server starting. Press Ctrl+C to stop.", flush=True)
    nicegui.ui.run(
        host=args.host,
        port=args.port,
        show=args.open_browser,
        title="LTE Scenario Toolkit",
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
