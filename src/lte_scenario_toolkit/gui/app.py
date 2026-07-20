"""Command-line entrypoint and application factory for the local GUI."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import ipaddress
import logging
import os
import stat
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from ..data_catalog import CatalogError, DataCatalog, load_data_catalog
from ..figure_service import FigureSpec
from ..jobs import JobBusyError, JobCoordinator
from ..map_assets import MapAssetService
from ..profiles import ProfileStore
from ..run_trash import (
    RunLeaseConflictError,
    RunUsageLeaseRegistry,
    TrashManager,
    TrashPlanStaleError,
)
from ..selection_service import SelectionService
from .assets import install_gui_assets
from .i18n import Translator, validate_translations
from .layout import render_app_shell
from .pages.candidates import (
    CandidateMapBundle,
    CandidateSession,
    CandidateSessionRegistry,
    build_candidate_map_bundle,
    build_candidate_overlay,
    build_candidate_style_overlay,
    default_online_tile_probe,
    render_candidate_page,
    render_candidate_unavailable,
)
from .pages.configure import render_configure_page, render_configure_picker
from .pages.figures import render_figures_page, render_figures_unavailable
from .pages.generate import (
    generation_model,
    render_generate_page,
    render_generation_unavailable,
)
from .pages.history import (
    HistorySnapshot,
    HistoryTrashPlan,
    TrashSnapshot,
    build_trash_snapshot,
    confirm_history_trash_plan,
    figure_source_options,
    history_roots,
    rebuild_history,
    render_history_error,
    render_history_loading,
    render_history_page,
    render_trash_loading,
    render_trash_page,
)
from .pages.scenarios import (
    get_job_coordinator,
    render_scenarios_page,
    shutdown_job_coordinator,
)
from .settings import GuiSettingsError, GuiSettingsStore

GUI_INSTALL_INSTRUCTION = 'python -m pip install -e ".[gui]"'
_LOGGER = logging.getLogger(__name__)


class TrashViewRefreshError(RuntimeError):
    """Raised when a committed Trash mutation cannot rebuild its view."""

    code = "trash.refresh_failed"


def _root_digest_for_log(root: object) -> str:
    """Return a stable root digest without logging the filesystem path."""

    try:
        value = os.path.normcase(os.path.normpath(os.fspath(root)))
    except (OSError, TypeError, ValueError):
        value = "invalid-root"
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def log_trash_mutation(
    kind: str,
    *,
    transaction_id: str | None = None,
    state: object | None = None,
    member_count: int | None = None,
    roots: Iterable[object] = (),
    error: BaseException | None = None,
    cancelled: bool = False,
) -> None:
    """Emit safe structured lifecycle fields for one Trash operation."""

    fields = {
        "kind": kind,
        "transaction_id": transaction_id,
        "state": None if state is None else str(getattr(state, "value", state)),
        "member_count": member_count,
        "root_digests": tuple(_root_digest_for_log(root) for root in roots),
    }
    if cancelled:
        _LOGGER.info("trash mutation cancelled %s", fields)
    elif error is not None:
        fields["error_type"] = type(error).__name__
        fields["error_code"] = getattr(error, "code", "trash.transaction_failed")
        _LOGGER.warning("trash mutation failed %s", fields)
    else:
        _LOGGER.info("trash mutation completed %s", fields)


def log_trash_result(kind: str, result: Any) -> None:
    """Log a completed mutation before any client callback is attempted.

    Restore/purge workers return ``(receipt, before)`` so that the UI can
    report the affected family.  Keeping this extraction here means a
    disconnected NiceGUI client still leaves one complete, path-safe
    lifecycle record and route callbacks do not need to duplicate it.
    """

    receipt = result
    before = None
    if isinstance(result, tuple) and len(result) == 2:
        receipt, before = result
    source = before if before is not None else receipt
    members = getattr(source, "members", ())
    member_count = getattr(receipt, "run_count", None)
    if type(member_count) is not int:
        member_count = len(members)
    state = getattr(source, "state", None)
    if before is not None:
        if kind == "history.trash_restore":
            state = "restored"
        elif kind == "history.trash_purge":
            state = "purged"
    log_trash_mutation(
        kind,
        transaction_id=getattr(receipt, "transaction_id", None),
        state=state,
        member_count=member_count,
        roots=getattr(source, "roots", ()),
    )


def log_trash_delivery_cancelled(kind: str) -> None:
    """Record a disconnected UI waiter without changing mutation outcome."""

    _LOGGER.info("trash mutation delivery cancelled %s", {"kind": kind})


def trash_error_translation_key(
    error: BaseException,
    *,
    default: str,
) -> str:
    """Map domain failures to stable localized UI keys without raw details."""

    chain: list[BaseException] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__

    for candidate in chain:
        code = getattr(candidate, "code", None)
        if isinstance(candidate, JobBusyError) or code == "job.busy":
            return "history.trash_busy"
        if isinstance(candidate, TrashPlanStaleError) or code == "trash.plan_stale":
            return "history.trash_stale"
        if isinstance(candidate, RunLeaseConflictError) or code in {
            "run.in_use",
            "run.lease_conflict",
        }:
            return "history.trash_lease_conflict"

    code = getattr(error, "code", None)
    text = " ".join(str(candidate) for candidate in chain).casefold()
    if "in use" in text or "lease" in text:
        return "history.trash_lease_conflict"
    if "stale" in text or "refresh" in text or "changed" in text:
        return "history.trash_stale"
    if code == "trash.root_unavailable" or "root" in text and "unavailable" in text:
        return "history.trash_root_unavailable"
    if "destination" in text or "occupied" in text:
        return "history.trash_destination_occupied"
    if "journal" in text or "payload" in text:
        return "history.trash_journal_invalid"
    if "purgeable" in text or "partially purged" in text or "purge state" in text:
        return "history.trash_permanent_error"
    if "not restorable" in text or "recovery required" in text:
        return "history.trash_restore_error"
    return default


def client_is_deleted(client: Any) -> bool:
    """Normalize NiceGUI's deleted-client flag across supported releases."""

    try:
        value = getattr(client, "is_deleted", False)
        return bool(value() if callable(value) else value)
    except Exception:
        # A client that cannot report its state must be treated as gone. This
        # keeps late worker completions from enqueueing messages to a dead
        # websocket.
        return True


async def run_trash_mutation(
    *,
    client: Any,
    coordinator: JobCoordinator,
    kind: str,
    log_kind: str | None = None,
    worker: Callable[[], Any],
    on_success: Callable[[Any], Any] | None = None,
    on_error: Callable[[BaseException], Any] | None = None,
    on_error_log: Callable[[BaseException], None] | None = None,
    on_refresh: Callable[[], Any] | None = None,
    on_refresh_error: Callable[[BaseException], Any] | None = None,
    io_bound: Callable[..., Any] | None = None,
) -> Any:
    """Run one serialized Trash mutation and always release its job slot.

    ``io_bound`` is injectable for tests; production callers use NiceGUI's
    worker bridge.  Result/error callbacks are skipped when the page client
    has already been deleted, preventing late websocket writes.
    """

    if io_bound is None:
        from nicegui import run

        io_bound = run.io_bound
    lifecycle_kind = kind if log_kind is None else log_kind
    try:
        job = coordinator.start(kind)
    except BaseException as exc:
        logged = False
        if on_error_log is not None:
            try:
                on_error_log(exc)
                logged = True
            except BaseException:
                pass
        if not logged:
            log_trash_mutation(lifecycle_kind, error=exc)
        if not client_is_deleted(client) and on_error is not None:
            try:
                result = on_error(exc)
                if inspect.isawaitable(result):
                    await result
            except BaseException:
                pass
        return None

    finish_lock = Lock()
    worker_done = False
    finish_requested = False
    finish_done = False

    def bridge_cannot_submit() -> bool:
        try:
            from nicegui import core

            return bool(core.app.is_stopping)
        except Exception:
            return True

    def finish_once() -> None:
        nonlocal finish_requested, finish_done
        with finish_lock:
            if finish_done:
                return
            finish_requested = True
            if not worker_done:
                return
            finish_done = True
        coordinator.finish(job.job_id)

    def log_worker_error(error: BaseException) -> None:
        if on_error_log is not None:
            try:
                on_error_log(error)
                return
            except BaseException:
                pass
        log_trash_mutation(lifecycle_kind, error=error)

    async def invoke_ui_callback(
        callback: Callable[..., Any] | None,
        *args: Any,
    ) -> None:
        if callback is None or client_is_deleted(client):
            return
        try:
            callback_result = callback(*args)
            if inspect.isawaitable(callback_result):
                await callback_result
        except BaseException:
            pass

    if client_is_deleted(client) or bridge_cannot_submit():
        with finish_lock:
            finish_requested = True
            finish_done = True
        coordinator.finish(job.job_id)
        log_trash_mutation(lifecycle_kind, cancelled=True)
        return None

    executor: ThreadPoolExecutor | None = None
    try:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lte-trash")
        worker_future = executor.submit(worker)
    except BaseException as exc:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        with finish_lock:
            finish_requested = True
            finish_done = True
        coordinator.finish(job.job_id)
        log_worker_error(exc)
        await invoke_ui_callback(on_error, exc)
        return None

    def worker_finished(completed_future: Any) -> None:
        nonlocal worker_done, finish_done
        should_finish = False
        try:
            try:
                completed_result = completed_future.result()
            except BaseException as exc:
                log_worker_error(exc)
            else:
                log_trash_result(lifecycle_kind, completed_result)
        except BaseException:
            # Diagnostics must never strand the mutation gate.
            pass
        finally:
            try:
                with finish_lock:
                    worker_done = True
                    should_finish = finish_requested and not finish_done
                    if should_finish:
                        finish_done = True
                if should_finish:
                    coordinator.finish(job.job_id)
            except BaseException:
                pass
            finally:
                try:
                    executor.shutdown(wait=False, cancel_futures=False)
                except BaseException:
                    pass

    worker_future.add_done_callback(worker_finished)

    try:
        async def bridge_call() -> Any:
            work_result = io_bound(worker_future.result)
            return (
                await work_result
                if inspect.isawaitable(work_result)
                else work_result
            )

        bridge_task = asyncio.create_task(bridge_call())

        def consume_bridge_result(task: asyncio.Task[Any]) -> None:
            try:
                task.exception()
            except BaseException:
                pass

        bridge_task.add_done_callback(consume_bridge_result)
        worker_result: Any = None
        worker_error: BaseException | None = None
        try:
            bridge_result = await asyncio.shield(bridge_task)
        except asyncio.CancelledError:
            finish_once()
            try:
                await asyncio.shield(bridge_task)
            except BaseException:
                pass
            log_trash_delivery_cancelled(lifecycle_kind)
            return None
        except BaseException as bridge_error:
            if worker_future.done():
                try:
                    worker_result = worker_future.result()
                except BaseException as exc:
                    if exc is bridge_error:
                        worker_error = exc
                    else:
                        _LOGGER.warning(
                            "trash worker bridge failed %s",
                            {
                                "kind": lifecycle_kind,
                                "error_type": type(bridge_error).__name__,
                            },
                        )
                        return None
                else:
                    _LOGGER.warning(
                        "trash worker bridge failed %s",
                        {
                            "kind": lifecycle_kind,
                            "error_type": type(bridge_error).__name__,
                        },
                    )
                    return None
            else:
                _LOGGER.warning(
                    "trash worker bridge failed %s",
                    {
                        "kind": lifecycle_kind,
                        "error_type": type(bridge_error).__name__,
                    },
                )
                return None
        else:
            if bridge_result is None:
                log_trash_delivery_cancelled(lifecycle_kind)
                return None
            if not worker_future.done():
                _LOGGER.warning(
                    "trash worker bridge returned before completion %s",
                    {"kind": lifecycle_kind},
                )
                return None
            try:
                worker_result = worker_future.result()
            except BaseException as exc:
                worker_error = exc

        if worker_error is not None:
            if not client_is_deleted(client) and on_refresh is not None:
                try:
                    refresh_result = on_refresh()
                    if inspect.isawaitable(refresh_result):
                        await refresh_result
                except BaseException as refresh_error:
                    _LOGGER.warning(
                        "trash view refresh failed %s",
                        {
                            "kind": lifecycle_kind,
                            "error_type": type(refresh_error).__name__,
                            "error_code": getattr(
                                refresh_error,
                                "code",
                                "trash.refresh_failed",
                            ),
                        },
                    )
                    await invoke_ui_callback(on_refresh_error, refresh_error)
            await invoke_ui_callback(on_error, worker_error)
            return None

        if not client_is_deleted(client) and on_refresh is not None:
            try:
                refresh_result = on_refresh()
                if inspect.isawaitable(refresh_result):
                    await refresh_result
            except BaseException as refresh_error:
                _LOGGER.warning(
                    "trash view refresh failed %s",
                    {
                        "kind": lifecycle_kind,
                        "error_type": type(refresh_error).__name__,
                        "error_code": getattr(
                            refresh_error,
                            "code",
                            "trash.refresh_failed",
                        ),
                    },
                )
                await invoke_ui_callback(on_refresh_error, refresh_error)
                return worker_result
        if not client_is_deleted(client):
            await invoke_ui_callback(on_success, worker_result)
            return worker_result
        return None
    finally:
        finish_once()


@dataclass(frozen=True, slots=True)
class _FigureRouteRequest:
    source: Path | None = None
    session_id: str | None = None
    output_root: Path | None = None
    formats: tuple[str, ...] = ()
    figure_spec: FigureSpec | None = None
    parent_run_id: str | None = None
    parent_run_path: Path | None = None

    def __post_init__(self) -> None:
        if (self.source is None) == (self.session_id is None):
            raise ValueError("figure route request requires one source or session")


class _FigureRouteRegistry:
    def __init__(self, max_requests: int = 64) -> None:
        self.max_requests = max_requests
        self._lock = Lock()
        self._requests: OrderedDict[str, _FigureRouteRequest] = OrderedDict()

    def add(self, request: _FigureRouteRequest) -> str:
        token = uuid4().hex
        with self._lock:
            self._requests[token] = request
            while len(self._requests) > self.max_requests:
                self._requests.popitem(last=False)
        return token

    def get(self, token: str) -> _FigureRouteRequest | None:
        with self._lock:
            request = self._requests.get(token)
            if request is not None:
                self._requests.move_to_end(token)
            return request

    def clear(self) -> None:
        with self._lock:
            self._requests.clear()


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


def _is_redirected_path(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, FileNotFoundError):
        return False
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_point)


def _redirected_component(root: Path, candidate: Path) -> Path | None:
    if _is_redirected_path(root):
        return root
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return None
    current = root
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current) and _is_redirected_path(current):
            return current
    return None


def _resolve_allowlisted_file(
    path: str | os.PathLike[str],
    *,
    roots: Iterable[str | os.PathLike[str]],
    suffixes: Iterable[str],
    label: str,
) -> Path:
    """Resolve one local regular file without crossing an allowlisted root."""

    if not isinstance(path, (str, os.PathLike)) or isinstance(path, bytes):
        raise ValueError(f"{label} must be a local filesystem path")
    raw_text = os.fspath(path)
    if not isinstance(raw_text, str) or "://" in raw_text or "\x00" in raw_text:
        raise ValueError(f"{label} must be a local filesystem path")
    raw = Path(raw_text).expanduser()
    if not raw.is_absolute():
        raise ValueError(f"{label} must be an absolute local filesystem path")
    if ".." in raw.parts:
        raise ValueError(f"{label} must not contain traversal components")

    allowed_suffixes = {str(value).casefold() for value in suffixes}
    if not allowed_suffixes or any(not value.startswith(".") for value in allowed_suffixes):
        raise ValueError("allowlisted file suffixes must be non-empty extensions")

    lexical = raw.absolute()
    for value in roots:
        root = Path(value).expanduser()
        if not root.is_absolute():
            raise ValueError("allowlisted file roots must be absolute")
        root = root.absolute()
        redirected_root = _redirected_component(Path(root.anchor), root)
        if redirected_root is not None:
            raise ValueError(
                f"allowlisted root must not be redirected: {redirected_root}"
            )
        try:
            lexical.relative_to(root)
        except ValueError:
            continue
        redirected = _redirected_component(root, lexical)
        if redirected is not None:
            raise ValueError(f"{label} must not be redirected: {redirected}")
        try:
            resolved_root = root.resolve(strict=True)
            resolved = lexical.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"{label} does not exist or cannot be resolved") from exc
        if not resolved.is_relative_to(resolved_root):
            raise ValueError(f"{label} is outside the allowlisted roots")
        if not resolved.is_file() or resolved.suffix.casefold() not in allowed_suffixes:
            extensions = ", ".join(sorted(allowed_suffixes))
            raise ValueError(f"{label} must be a regular {extensions} file")
        return resolved
    raise ValueError(f"{label} is outside the allowlisted roots")


def create_app(
    catalog: DataCatalog | None = None,
    testing: bool = False,
    *,
    profile_store: Any | None = None,
    catalog_loader: Callable[[Any], Any] | None = None,
    selection_service_factory: Callable[[Any], Any] = SelectionService,
    coordinator: JobCoordinator | None = None,
    candidate_registry: CandidateSessionRegistry | None = None,
    candidate_bundle_builder: Callable[
        [CandidateSession, MapAssetService], CandidateMapBundle
    ]
    | None = None,
    candidate_overlay_asset_builder: Callable[[CandidateSession, Any, Any], Any]
    | None = None,
    candidate_style_asset_builder: Callable[[CandidateSession, Any], Any]
    | None = None,
    online_tile_probe: Callable[[], bool] | None = None,
    figure_source_options_provider: Callable[[], Mapping[str, str]] | None = None,
    trash_manager: TrashManager | None = None,
    usage_leases: RunUsageLeaseRegistry | None = None,
):
    """Register and return the NiceGUI application for one validated catalog."""

    from nicegui import app, ui

    station_layer_resource = install_gui_assets(app, ui)

    if usage_leases is not None and not isinstance(
        usage_leases,
        RunUsageLeaseRegistry,
    ):
        raise ValueError("usage_leases must be a RunUsageLeaseRegistry")
    if trash_manager is not None and not isinstance(trash_manager, TrashManager):
        raise ValueError("trash_manager must be a TrashManager")
    manager_leases = (
        None if trash_manager is None else getattr(trash_manager, "_leases", None)
    )
    if (
        manager_leases is not None
        and usage_leases is not None
        and manager_leases is not usage_leases
    ):
        raise ValueError("trash_manager and usage_leases must share one registry")

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
    sessions = candidate_registry or CandidateSessionRegistry()
    figure_requests = _FigureRouteRegistry()
    ephemeral_roots: set[Path] = set()
    ephemeral_roots_lock = Lock()
    map_assets = MapAssetService(repo_root)
    bundle_builder = candidate_bundle_builder or build_candidate_map_bundle
    static_asset_urls: dict[Path, str] = {}

    if hasattr(app, "on_shutdown"):
        if uses_shared_coordinator:
            app.on_shutdown(shutdown_job_coordinator)
        app.on_shutdown(sessions.clear)
        app.on_shutdown(figure_requests.clear)

    page_options = {
        "title": Translator(initial_settings.language).text("app.title"),
        "reconnect_timeout": 0 if testing else 3.0,
    }

    def render_shell(
        active_route: str,
        page_context_key: str,
        body: Callable[[Translator], Any],
    ) -> Any:
        settings = store.load()
        translator = Translator(settings.language)

        def change_language(event) -> None:
            try:
                store.update(language=event.value)
            except GuiSettingsError as exc:
                ui.notify(str(exc), type="negative")
                return
            ui.navigate.reload()

        def change_navigation_collapsed(collapsed: bool) -> None:
            try:
                store.update(navigation_collapsed=collapsed)
            except GuiSettingsError as exc:
                ui.notify(str(exc), type="negative")

        content = render_app_shell(
            ui,
            translator,
            active_route=active_route,
            page_context=translator.text(page_context_key),
            get_job_snapshot=runtime.coordinator.snapshot,
            on_language_change=change_language,
            navigation_collapsed=settings.navigation_collapsed,
            on_navigation_toggle=change_navigation_collapsed,
        )
        with content:
            return body(translator)

    def candidate_asset_url(path: Path) -> str:
        resolved = _resolve_allowlisted_file(
            path,
            roots=(map_assets.cache_root,),
            suffixes=(".png",),
            label="candidate map asset",
        )
        existing = static_asset_urls.get(resolved)
        if existing is not None:
            return existing
        url = app.add_static_file(
            local_file=resolved,
            url_path=f"/_candidate_assets/{len(static_asset_urls):08x}.png",
            strict=True,
            max_cache_age=3600,
        )
        static_asset_urls[resolved] = url
        return url

    def preview_asset_url(path: Path) -> str:
        preview_root = (
            repo_root / ".lte-data" / "cache" / "previews"
        )
        resolved = _resolve_allowlisted_file(
            path,
            roots=(preview_root,),
            suffixes=(".png",),
            label="figure preview asset",
        )
        existing = static_asset_urls.get(resolved)
        if existing is not None:
            return existing
        url = app.add_static_file(
            local_file=resolved,
            url_path=f"/_preview_assets/{len(static_asset_urls):08x}.png",
            strict=True,
            max_cache_age=3600,
        )
        static_asset_urls[resolved] = url
        return url

    def remember_output_root(root: str | os.PathLike[str]) -> None:
        resolved = Path(root).expanduser().resolve(strict=False)
        with ephemeral_roots_lock:
            ephemeral_roots.add(resolved)
        store.update(add_output_roots=(resolved,))

    def history_output_roots() -> tuple[Path, ...]:
        configured = store.load().output_roots
        with ephemeral_roots_lock:
            current = tuple(ephemeral_roots)
        return (*configured, *current)

    def authoritative_history_roots() -> tuple[Path, ...]:
        """Return exactly the roots used by History and Trash planning."""

        return history_roots(repo_root, history_output_roots())

    active_leases = (
        usage_leases
        or manager_leases
        or RunUsageLeaseRegistry()
    )
    active_trash = trash_manager or TrashManager(
        authoritative_history_roots,
        active_leases,
    )

    def available_figure_source_options() -> dict[str, str]:
        if figure_source_options_provider is not None:
            provided = figure_source_options_provider()
            if not isinstance(provided, Mapping):
                raise ValueError("figure source options provider must return a mapping")
            return {str(path): str(label) for path, label in provided.items()}
        try:
            snapshot = rebuild_history(
                repo_root,
                history_output_roots(),
            )
        except (OSError, ValueError):
            return {}
        return figure_source_options(snapshot)

    def remember_published_run(path: Path) -> None:
        try:
            root = path.parents[2]
        except IndexError as exc:
            raise GuiSettingsError("Published run has no output root") from exc
        remember_output_root(root)

    def open_candidate_session(outcome: Any) -> None:
        if getattr(outcome.preflight, "profile", None) is not outcome.snapshot:
            ui.notify(
                Translator(store.load().language).text("preflight.passed"),
                type="positive",
            )
            return
        session = sessions.create(
            outcome,
            runtime.selection_service,
            repo_root,
        )
        ui.navigate.to(f"/candidates/{session.session_id}")

    def open_generation_session(confirmed: CandidateSession) -> None:
        ui.navigate.to(f"/generate/{confirmed.session_id}")

    def render_candidate_explorer_body(
        translator: Translator,
        session: CandidateSession,
        bundle: CandidateMapBundle,
    ) -> None:
        asset_url = candidate_asset_url(bundle.dem_asset.path)
        render_candidate_page(
            ui,
            translator,
            session,
            runtime.coordinator,
            station_layer_resource=station_layer_resource,
            registry=sessions,
            dem_asset_url=asset_url,
            dem_asset_url_builder=candidate_asset_url,
            online_tile_probe=online_tile_probe or default_online_tile_probe,
            candidate_overlay_builder=(
                candidate_overlay_asset_builder
                or (
                    lambda active_session, candidate, style: (
                        build_candidate_overlay(
                            active_session,
                            map_assets,
                            candidate,
                            style=style,
                        )
                    )
                )
            ),
            dem_style_builder=(
                candidate_style_asset_builder
                or (
                    lambda active_session, style: (
                        build_candidate_style_overlay(
                            active_session,
                            map_assets,
                            bundle.map_bounds,
                            style,
                        )
                    )
                )
            ),
            on_confirm=open_generation_session,
        )

    @ui.page("/configure/{scenario_id}", **page_options)
    def configure_scenario(scenario_id: str, profile: str | None = None) -> None:
        render_shell(
            "/configure",
            "nav.configure",
            lambda translator: render_configure_page(
                ui,
                translator,
                runtime.catalog,
                runtime.profile_store,
                runtime.selection_service,
                scenario_id=scenario_id,
                selected_profile_id=profile,
                on_profile_mutation=runtime.refresh_after_profile_mutation,
                on_preflight_success=open_candidate_session,
            ),
        )

    @ui.page("/candidates/{session_id}", **page_options)
    async def candidate_explorer(session_id: str) -> None:
        session = sessions.get(session_id)
        if session is None:
            render_shell(
                "/configure",
                "candidates.title",
                lambda translator: render_candidate_unavailable(ui, translator),
            )
            return

        prepared_bundle = session.map_bundle
        if prepared_bundle is not None:
            render_shell(
                "/configure",
                "candidates.title",
                lambda translator: render_candidate_explorer_body(
                    translator,
                    session,
                    prepared_bundle,
                ),
            )
            return

        loading_container: Any | None = None
        loading_translator: Translator | None = None

        def render_loading(translator: Translator) -> None:
            nonlocal loading_container, loading_translator
            loading_translator = translator
            loading_container = ui.column().classes("full-width")
            with loading_container:
                with ui.column().classes("items-center justify-center full-width"):
                    ui.spinner(size="lg")
                    ui.label(translator.text("candidates.preparing_map"))

        render_shell(
            "/configure",
            "candidates.title",
            render_loading,
        )
        assert loading_container is not None
        assert loading_translator is not None

        client = ui.context.client
        sessions.pin(session_id)
        try:
            await client.connected()
            if client.is_deleted:
                return
            if session.map_bundle is None:
                from nicegui import run

                try:
                    bundle = await run.io_bound(
                        bundle_builder,
                        session,
                        map_assets,
                    )
                    session = sessions.set_map_bundle(session_id, bundle)
                except Exception as error:
                    if not client.is_deleted:
                        loading_container.clear()
                        with loading_container:
                            render_candidate_unavailable(
                                ui,
                                loading_translator,
                                str(error),
                            )
                    return
            bundle = session.map_bundle
            if client.is_deleted:
                return
            loading_container.clear()
            with loading_container:
                if bundle is None:
                    render_candidate_unavailable(ui, loading_translator)
                else:
                    render_candidate_explorer_body(
                        loading_translator,
                        session,
                        bundle,
                    )
        finally:
            sessions.unpin(session_id)

    @ui.page("/candidates", **page_options)
    def candidate_without_session() -> None:
        render_shell(
            "/configure",
            "candidates.title",
            lambda translator: render_candidate_unavailable(ui, translator),
        )

    def generation_body(translator: Translator, session_id: str) -> None:
        session = sessions.get(session_id)
        try:
            if session is None:
                raise ValueError("candidate session is unavailable")
            generation_model(session)
        except ValueError:
            render_generation_unavailable(ui, translator)
            return
        def remember_generation_run(path: Path) -> None:
            remember_output_root(session.preflight.output_root)
            sessions.discard(session.session_id)

        render_generate_page(
            ui,
            translator,
            session,
            runtime.coordinator,
            on_published=remember_generation_run,
            on_complete=lambda _state: ui.navigate.to("/history"),
            on_open_figures=lambda: ui.navigate.to(
                "/figures/"
                + figure_requests.add(
                    _FigureRouteRequest(session_id=session.session_id)
                )
            ),
        )

    @ui.page("/generate/{session_id}", **page_options)
    def generate_session(session_id: str) -> None:
        render_shell(
            "/configure",
            "generate.title",
            lambda translator: generation_body(translator, session_id),
        )

    @ui.page("/generate", **page_options)
    def generate_without_session() -> None:
        render_shell(
            "/configure",
            "generate.title",
            lambda translator: render_generation_unavailable(ui, translator),
        )

    def request_figure_session(
        request: _FigureRouteRequest | None,
    ) -> CandidateSession | None:
        session = (
            None
            if request is None or request.session_id is None
            else sessions.get(request.session_id)
        )
        try:
            if session is not None:
                generation_model(session)
        except ValueError:
            return None
        return session

    def figures_body(
        translator: Translator,
        request: _FigureRouteRequest | None = None,
    ) -> None:
        def remember_figure_run(path: Path) -> None:
            remember_published_run(path)
            if request is not None and request.session_id is not None:
                sessions.discard(request.session_id)

        render_figures_page(
            ui,
            translator,
            repo_root,
            runtime.coordinator,
            initial_source=None if request is None else request.source,
            source_options=available_figure_source_options(),
            current_session=request_figure_session(request),
            output_root=None if request is None else request.output_root,
            initial_formats=(
                None if request is None or not request.formats else request.formats
            ),
            initial_spec=None if request is None else request.figure_spec,
            parent_run_id=None if request is None else request.parent_run_id,
            parent_run_path=None if request is None else request.parent_run_path,
            on_published=remember_figure_run,
            preview_url_builder=preview_asset_url,
            usage_leases=active_leases,
            run_roots=authoritative_history_roots,
        )

    @ui.page("/figures/{request_id}", **page_options)
    def figures_request(request_id: str) -> None:
        request = figure_requests.get(request_id)
        valid_request = request is not None and (
            request.source is not None
            or request_figure_session(request) is not None
        )
        render_shell(
            "/figures",
            "figures.title",
            lambda translator: (
                render_figures_unavailable(ui, translator)
                if not valid_request
                else figures_body(translator, request)
            ),
        )

    @ui.page("/figures", **page_options)
    def figures() -> None:
        render_shell(
            "/figures",
            "figures.title",
            lambda translator: figures_body(translator),
        )

    def open_history_source(
        path: Path,
        output_root: Path | None = None,
        parent_run_id: str | None = None,
        parent_run_path: Path | None = None,
        formats: tuple[str, ...] = (),
        figure_spec: FigureSpec | None = None,
    ) -> None:
        if output_root is None:
            try:
                output_root = path.parents[2]
            except IndexError:
                output_root = None
        token = figure_requests.add(
            _FigureRouteRequest(
                source=path,
                output_root=output_root,
                formats=formats,
                figure_spec=figure_spec,
                parent_run_id=parent_run_id,
                parent_run_path=parent_run_path,
            )
        )
        ui.navigate.to(f"/figures/{token}")

    def reveal_directory(path: Path) -> None:
        startfile = getattr(os, "startfile", None)
        if startfile is None:
            raise OSError("Directory reveal is unavailable on this platform")
        startfile(str(path))

    def open_pending_figures(session_id: str) -> None:
        token = figure_requests.add(_FigureRouteRequest(session_id=session_id))
        ui.navigate.to(f"/figures/{token}")

    def continue_pending_generation(session_id: str) -> None:
        ui.navigate.to(f"/generate/{session_id}")

    @ui.page("/history", **page_options)
    async def history() -> None:
        from nicegui import run

        roots = history_output_roots()
        history_translator: Translator | None = None
        history_root: Any | None = None
        loading_holder: Any | None = None

        def loading_body(translator: Translator) -> Any:
            nonlocal history_translator, history_root, loading_holder
            history_translator = translator
            history_root = ui.column().classes("full-width")
            with history_root:
                loading_holder = render_history_loading(ui, translator)
            return history_root

        render_shell(
            "/history",
            "history.title",
            loading_body,
        )
        assert history_translator is not None
        assert history_root is not None
        assert loading_holder is not None
        client = ui.context.client
        await client.connected()
        if client_is_deleted(client):
            return
        try:
            snapshot = await run.io_bound(rebuild_history, repo_root, roots)
            trash_snapshot = await run.io_bound(build_trash_snapshot, active_trash)
        except Exception as error:
            if not client_is_deleted(client):
                render_history_error(ui, history_translator, loading_holder, error)
            return
        if (
            not isinstance(snapshot, HistorySnapshot)
            or not isinstance(trash_snapshot, TrashSnapshot)
            or client_is_deleted(client)
        ):
            return

        async def move_to_trash(displayed: HistoryTrashPlan) -> None:
            def worker() -> Any:
                current = rebuild_history(repo_root, history_output_roots())
                matches = [
                    row
                    for row in current.rows
                    if row.reference == displayed.reference
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "The selected run is no longer available; refresh History"
                    )
                return confirm_history_trash_plan(
                    displayed,
                    matches[0],
                    active_trash,
                )

            async def on_success(receipt: Any) -> None:
                if client_is_deleted(client):
                    return
                count = getattr(receipt, "run_count", None)
                if type(count) is not int:
                    count = len(getattr(receipt, "members", ()))
                ui.notify(
                    history_translator.text("history.trash_success", count=count),
                    type="positive",
                )

            def on_refresh_error(_error: BaseException) -> None:
                if client_is_deleted(client):
                    return
                ui.notify(
                    history_translator.text("history.trash_refresh_error"),
                    type="warning",
                )

            async def refresh() -> None:
                if client_is_deleted(client):
                    return
                fresh_history = await run.io_bound(
                    rebuild_history,
                    repo_root,
                    history_output_roots(),
                )
                fresh_trash = await run.io_bound(
                    build_trash_snapshot,
                    active_trash,
                )
                if client_is_deleted(client):
                    return
                if not isinstance(fresh_history, HistorySnapshot) or not isinstance(
                    fresh_trash,
                    TrashSnapshot,
                ):
                    raise TrashViewRefreshError(
                        "History rebuild did not return current snapshots"
                    )
                history_root.clear()
                with history_root:
                    render_history_page(
                        ui,
                        history_translator,
                        repo_root,
                        history_output_roots(),
                        snapshot=fresh_history,
                        on_reveal=reveal_directory,
                        on_open_figures=lambda path, root, parent, parent_path, figure_spec: open_history_source(
                            path,
                            root,
                            parent,
                            parent_path,
                            figure_spec=figure_spec,
                        ),
                        on_retry_missing=lambda path, root, parent, parent_path, formats, figure_spec: open_history_source(
                            path,
                            root,
                            parent,
                            parent_path,
                            formats,
                            figure_spec,
                        ),
                        pending_selections=sessions.confirmed_sessions(),
                        on_open_pending_figures=open_pending_figures,
                        on_continue_pending=continue_pending_generation,
                        trash_manager=active_trash,
                        on_move_to_trash=move_to_trash,
                        trash_count=fresh_trash.count,
                        on_open_trash=lambda: ui.navigate.to("/history/trash"),
                    )

            def log_error(_error: BaseException) -> None:
                plan_value = displayed.trash_plan
                log_trash_mutation(
                    "history.trash_move",
                    transaction_id=getattr(plan_value, "transaction_id", None),
                    member_count=displayed.run_count,
                    roots=displayed.roots,
                    error=_error,
                )

            def on_error(_error: BaseException) -> None:
                if client_is_deleted(client):
                    return
                key = trash_error_translation_key(
                    _error,
                    default="history.trash_error",
                )
                ui.notify(history_translator.text(key), type="negative")

            await run_trash_mutation(
                client=client,
                coordinator=runtime.coordinator,
                kind="history.trash_move",
                worker=worker,
                on_success=on_success,
                on_error=on_error,
                on_error_log=log_error,
                on_refresh=refresh,
                on_refresh_error=on_refresh_error,
            )

        history_root.clear()
        with history_root:
            render_history_page(
                ui,
                history_translator,
                repo_root,
                history_output_roots(),
                snapshot=snapshot,
                on_reveal=reveal_directory,
                on_open_figures=lambda path, root, parent, parent_path, figure_spec: open_history_source(
                    path,
                    root,
                    parent,
                    parent_path,
                    figure_spec=figure_spec,
                ),
                on_retry_missing=lambda path, root, parent, parent_path, formats, figure_spec: open_history_source(
                    path,
                    root,
                    parent,
                    parent_path,
                    formats,
                    figure_spec,
                ),
                pending_selections=sessions.confirmed_sessions(),
                on_open_pending_figures=open_pending_figures,
                on_continue_pending=continue_pending_generation,
                trash_manager=active_trash,
                on_move_to_trash=move_to_trash,
                trash_count=trash_snapshot.count,
                on_open_trash=lambda: ui.navigate.to("/history/trash"),
            )

    @ui.page("/history/trash", **page_options)
    async def history_trash() -> None:
        from nicegui import run

        translator_holder: Translator | None = None
        page_root: Any | None = None
        loading_holder: Any | None = None

        def loading_body(translator: Translator) -> Any:
            nonlocal translator_holder, page_root, loading_holder
            translator_holder = translator
            page_root = ui.column().classes("full-width")
            with page_root:
                loading_holder = render_trash_loading(
                    ui,
                    translator,
                    on_back=lambda: ui.navigate.to("/history"),
                )
            return page_root

        render_shell("/history", "history.trash_title", loading_body)
        assert translator_holder is not None
        assert page_root is not None
        assert loading_holder is not None
        client = ui.context.client
        await client.connected()
        if client_is_deleted(client):
            return
        try:
            snapshot = await run.io_bound(build_trash_snapshot, active_trash)
        except Exception as error:
            if not client_is_deleted(client):
                # The loading holder is deliberately the only mutable area;
                # never attempt to write a dead client after this point.
                render_history_error(ui, translator_holder, loading_holder, error)
            return
        if not isinstance(snapshot, TrashSnapshot) or client_is_deleted(client):
            return

        async def refresh_trash() -> None:
            if client_is_deleted(client):
                return
            fresh = await run.io_bound(build_trash_snapshot, active_trash)
            if client_is_deleted(client):
                return
            if not isinstance(fresh, TrashSnapshot):
                raise TrashViewRefreshError(
                    "Trash rebuild did not return a current snapshot"
                )
            page_root.clear()
            with page_root:
                render_trash_page(
                    ui,
                    translator_holder,
                    fresh,
                    on_back=lambda: ui.navigate.to("/history"),
                    on_restore=restore_transaction,
                    on_purge=purge_transaction,
                    on_recover=recover_transaction,
                )

        def _notify_trash_success(kind: str, result: Any) -> None:
            receipt = result
            before = None
            if (
                isinstance(result, tuple)
                and len(result) == 2
            ):
                receipt, before = result
            member_count = getattr(receipt, "run_count", None)
            if before is not None:
                member_count = len(getattr(before, "members", ()))
            if client_is_deleted(client):
                return
            if kind == "history.trash_restore":
                key = "history.trash_restore_success"
            elif kind == "history.trash_purge":
                key = "history.trash_permanent_success"
            else:
                key = "history.trash_recover_success"
            values = {
                "count": member_count or getattr(receipt, "run_count", 0),
            }
            ui.notify(translator_holder.text(key, **values), type="positive")

        def _notify_trash_refresh_error(_error: BaseException) -> None:
            if client_is_deleted(client):
                return
            ui.notify(
                translator_holder.text("history.trash_refresh_error"),
                type="warning",
            )

        def _log_trash_error(
            *,
            kind: str,
            transaction_id: str | None,
            context: Mapping[str, Any] | None = None,
        ) -> Callable[[BaseException], None]:
            def log_error(_error: BaseException) -> None:
                before = None if context is None else context.get("before")
                log_trash_mutation(
                    kind,
                    transaction_id=transaction_id,
                    state=None if before is None else getattr(before, "state", None),
                    member_count=(
                        None
                        if before is None
                        else len(getattr(before, "members", ()))
                    ),
                    roots=() if before is None else getattr(before, "roots", ()),
                    error=_error,
                )

            return log_error

        def _notify_trash_error(
            key: str,
        ) -> Callable[[BaseException], None]:
            def notify(_error: BaseException) -> None:
                if client_is_deleted(client):
                    return
                error_key = trash_error_translation_key(
                    _error,
                    default=key,
                )
                ui.notify(translator_holder.text(error_key), type="negative")

            return notify

        async def restore_transaction(transaction_id: str) -> None:
            context: dict[str, Any] = {}

            def worker() -> Any:
                try:
                    before = active_trash.transaction(transaction_id)
                except Exception:
                    before = None
                context["before"] = before
                return active_trash.restore(transaction_id), before

            await run_trash_mutation(
                client=client,
                coordinator=runtime.coordinator,
                kind="history.trash_restore",
                worker=worker,
                on_success=lambda receipt: _notify_trash_success(
                    "history.trash_restore", receipt
                ),
                on_error=_notify_trash_error(
                    "history.trash_restore_error",
                ),
                on_error_log=_log_trash_error(
                    kind="history.trash_restore",
                    transaction_id=transaction_id,
                    context=context,
                ),
                on_refresh=refresh_trash,
                on_refresh_error=_notify_trash_refresh_error,
            )

        async def purge_transaction(
            transaction_id: str,
            confirmation: str | None = None,
        ) -> None:
            supplied = "" if confirmation is None else confirmation
            context: dict[str, Any] = {}

            def worker() -> Any:
                try:
                    before = active_trash.transaction(transaction_id)
                except Exception:
                    before = None
                context["before"] = before
                return active_trash.purge(
                    transaction_id,
                    confirmation=supplied,
                ), before

            await run_trash_mutation(
                client=client,
                coordinator=runtime.coordinator,
                kind="history.trash_purge",
                worker=worker,
                on_success=lambda receipt: _notify_trash_success(
                    "history.trash_purge", receipt
                ),
                on_error=_notify_trash_error(
                    "history.trash_permanent_error",
                ),
                on_error_log=_log_trash_error(
                    kind="history.trash_purge",
                    transaction_id=transaction_id,
                    context=context,
                ),
                on_refresh=refresh_trash,
                on_refresh_error=_notify_trash_refresh_error,
            )

        async def recover_transaction(transaction_id: str) -> None:
            context: dict[str, Any] = {}

            def worker() -> Any:
                try:
                    before = active_trash.transaction(transaction_id)
                except Exception:
                    before = None
                context["before"] = before
                return active_trash.recover(transaction_id), before

            await run_trash_mutation(
                client=client,
                coordinator=runtime.coordinator,
                kind="history.trash_restore",
                log_kind="history.trash_recover",
                worker=worker,
                on_success=lambda receipt: _notify_trash_success(
                    "history.trash_recover", receipt
                ),
                on_error=_notify_trash_error(
                    "history.trash_recover_error",
                ),
                on_error_log=_log_trash_error(
                    kind="history.trash_recover",
                    transaction_id=transaction_id,
                    context=context,
                ),
                on_refresh=refresh_trash,
                on_refresh_error=_notify_trash_refresh_error,
            )

        page_root.clear()
        with page_root:
            render_trash_page(
                ui,
                translator_holder,
                snapshot,
                on_back=lambda: ui.navigate.to("/history"),
                on_restore=restore_transaction,
                on_purge=purge_transaction,
                on_recover=recover_transaction,
            )

    @ui.page("/configure", **page_options)
    def configure_picker() -> None:
        render_shell(
            "/configure",
            "nav.configure",
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
            "nav.scenarios",
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
            "nav.scenarios",
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
        help="dataset catalog, relative to --repo-root by default",
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
