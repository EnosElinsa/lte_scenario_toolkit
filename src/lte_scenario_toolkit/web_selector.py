"""Blocking local-browser adapter for selecting one scanned candidate."""

from __future__ import annotations

import socket
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic, sleep
from typing import Any
from uuid import uuid4

from .candidate_scanner import Candidate, ScanResult
from .gui.i18n import Translator
from .gui.leaflet_assets import register_station_dots_resource
from .gui.pages.candidates import (
    CandidateSession,
    build_candidate_map_bundle,
    build_candidate_overlay,
    build_candidate_style_overlay,
    render_candidate_page,
)
from .jobs import JobCoordinator
from .map_assets import MapAssetService


class WebSelectorError(RuntimeError):
    """Raised when the local browser selector cannot start or stop safely."""


@dataclass(frozen=True, slots=True)
class WebSelectorPayload:
    """Frozen services and scan data needed to render the shared candidate page."""

    preflight: Any
    selection_service: Any
    scan_result: ScanResult
    repo_root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.scan_result, ScanResult) or not self.scan_result.completed:
            raise ValueError("web selector requires a completed ScanResult")
        if getattr(self.preflight, "profile", None) is None:
            raise ValueError("web selector preflight requires a frozen profile")
        if self.selection_service is None:
            raise ValueError("web selector requires a selection service")
        object.__setattr__(
            self,
            "repo_root",
            Path(self.repo_root).expanduser().resolve(strict=False),
        )


_UNSET = object()


class _SelectorSession:
    """Thread-safe one-result bridge between NiceGUI and a blocking caller."""

    def __init__(self, candidates: tuple[Candidate, ...], map_payload: Any) -> None:
        self.candidates = candidates
        self.map_payload = map_payload
        self._event = Event()
        self._lock = Lock()
        self._result: Candidate | None | object = _UNSET

    def confirm(self, index: int) -> bool:
        if type(index) is not int:
            raise ValueError("candidate index must be a zero-based integer")
        if not 0 <= index < len(self.candidates):
            raise IndexError("candidate index is outside the available candidates")
        with self._lock:
            if self._result is not _UNSET:
                return False
            self._result = self.candidates[index]
            self._event.set()
            return True

    def cancel(self) -> bool:
        with self._lock:
            if self._result is not _UNSET:
                return False
            self._result = None
            self._event.set()
            return True

    def close(self) -> bool:
        return self.cancel()

    def wait(self, timeout: float | None = None) -> Candidate | None:
        if not self._event.wait(timeout):
            raise TimeoutError("web selector session has not settled")
        with self._lock:
            result = self._result
        if result is _UNSET:
            raise RuntimeError("web selector session signalled without a result")
        return result


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.1):
            return True
    except OSError:
        return False


def _wait_for_server(host: str, port: int, stop: Event) -> bool:
    deadline = monotonic() + 10.0
    while monotonic() < deadline:
        if stop.is_set():
            return False
        if _port_is_open(host, port):
            return True
        sleep(0.05)
    return False


def _wait_for_port_release(host: str, port: int) -> bool:
    deadline = monotonic() + 5.0
    while monotonic() < deadline:
        if not _port_is_open(host, port):
            return True
        sleep(0.05)
    return False


def _open_default_browser(url: str) -> None:
    guidance = (
        "Could not open the local web selector in a browser. Use --select-index "
        "for headless selection."
    )
    try:
        opened = webbrowser.open(url, new=2)
    except Exception as exc:
        raise WebSelectorError(f"{guidance} ({exc})") from exc
    if not opened:
        raise WebSelectorError(guidance)


def _build_candidate_session(
    payload: WebSelectorPayload,
) -> tuple[CandidateSession, MapAssetService]:
    assets = MapAssetService(payload.repo_root)
    candidate_session = CandidateSession(
        session_id=uuid4().hex,
        profile_snapshot=payload.preflight.profile,
        preflight=payload.preflight,
        selection_service=payload.selection_service,
        repo_root=payload.repo_root,
        scan_result=payload.scan_result,
    )
    bundle = build_candidate_map_bundle(candidate_session, assets)
    candidate_session = CandidateSession(
        session_id=candidate_session.session_id,
        profile_snapshot=candidate_session.profile_snapshot,
        preflight=candidate_session.preflight,
        selection_service=candidate_session.selection_service,
        repo_root=candidate_session.repo_root,
        map_bundle=bundle,
        scan_result=candidate_session.scan_result,
    )
    return candidate_session, assets


def _safe_shutdown(app: Any) -> None:
    try:
        app.shutdown()
    except Exception:
        pass


def _render_selector_page(
    selector: _SelectorSession,
    candidate_session: CandidateSession,
    assets: MapAssetService,
    coordinator: JobCoordinator,
    ui: Any,
    app: Any,
    *,
    station_layer_resource: str,
) -> None:
    translator = Translator("en")

    def stop_if_settled(settled: bool) -> None:
        if settled:
            _safe_shutdown(app)

    def confirm(confirmed: CandidateSession) -> None:
        selected = confirmed.locked_candidate
        matches = tuple(
            index
            for index, candidate in enumerate(selector.candidates)
            if candidate == selected
        )
        if len(matches) != 1:
            ui.notify(
                "The confirmed candidate no longer matches this selector session.",
                type="negative",
            )
            return
        stop_if_settled(selector.confirm(matches[0]))

    def cancel() -> None:
        stop_if_settled(selector.cancel())

    ui.button(translator.text("action.cancel"), on_click=cancel).props(
        "outline"
    ).mark("web-selector-cancel")
    render_candidate_page(
        ui,
        translator,
        candidate_session,
        coordinator,
        station_layer_resource=station_layer_resource,
        candidate_overlay_builder=lambda active, candidate, style: (
            build_candidate_overlay(active, assets, candidate, style=style)
        ),
        dem_style_builder=lambda active, style: build_candidate_style_overlay(
            active,
            assets,
            active.map_bundle.map_bounds,
            style,
        ),
        on_confirm=confirm,
        auto_start=False,
        allow_rescan=False,
    )
    ui.context.client.on_delete(cancel)


def _run_server(session: _SelectorSession) -> None:
    payload = session.map_payload
    if not isinstance(payload, WebSelectorPayload):
        raise WebSelectorError(
            "The real web selector requires a WebSelectorPayload from the completed scan"
        )
    try:
        from nicegui import app, ui
    except ImportError as exc:
        raise WebSelectorError(
            "The web selector requires GUI dependencies. Install them with "
            "python -m pip install -e \".[gui]\", or use --select-index."
        ) from exc

    candidate_session, assets = _build_candidate_session(payload)
    coordinator = JobCoordinator()
    station_layer_resource = register_station_dots_resource(app)
    host = "127.0.0.1"
    port = _available_port()
    stop_browser = Event()
    browser_errors: list[WebSelectorError] = []

    def root() -> None:
        _render_selector_page(
            session,
            candidate_session,
            assets,
            coordinator,
            ui,
            app,
            station_layer_resource=station_layer_resource,
        )

    def open_browser_when_ready() -> None:
        try:
            if not _wait_for_server(host, port, stop_browser):
                if stop_browser.is_set():
                    return
                raise WebSelectorError(
                    "The local web selector server did not become ready. "
                    "Use --select-index in headless environments."
                )
            _open_default_browser(f"http://{host}:{port}/")
        except WebSelectorError as exc:
            browser_errors.append(exc)
            session.close()
            _safe_shutdown(app)
        except Exception as exc:
            browser_errors.append(
                WebSelectorError(f"Could not open the local web selector: {exc}")
            )
            session.close()
            _safe_shutdown(app)

    browser_thread = Thread(
        target=open_browser_when_ready,
        name="lte-web-selector-browser",
        daemon=True,
    )
    server_error: WebSelectorError | None = None
    port_released = False
    browser_thread.start()
    try:
        ui.run(
            root,
            host=host,
            port=port,
            show=False,
            reload=False,
            reconnect_timeout=0,
            title="LTE Scenario Candidate Selector",
            uvicorn_logging_level="error",
            show_welcome_message=False,
        )
    except SystemExit as exc:
        server_error = WebSelectorError(
            f"Could not start the loopback web selector on {host}:{port}: {exc}"
        )
    except Exception as exc:
        server_error = WebSelectorError(
            f"Could not start the loopback web selector on {host}:{port}: {exc}"
        )
    finally:
        session.close()
        stop_browser.set()
        _safe_shutdown(app)
        browser_thread.join(timeout=2.0)
        coordinator.shutdown()
        port_released = _wait_for_port_release(host, port)

    if browser_thread.is_alive():
        raise WebSelectorError("The web selector browser launcher did not stop")
    if not port_released:
        raise WebSelectorError(f"The web selector did not release {host}:{port}")
    if browser_errors:
        raise browser_errors[0]
    if server_error is not None:
        raise server_error


def select_candidate(
    candidates: Any,
    *,
    map_payload: Any,
) -> Candidate | None:
    """Open a blocking selector and return one candidate or ``None`` on cancel."""

    try:
        frozen = tuple(candidates)
    except TypeError as exc:
        raise ValueError("candidates must be an iterable of Candidate values") from exc
    if not frozen:
        raise ValueError("web selection requires at least one candidate")
    if any(not isinstance(candidate, Candidate) for candidate in frozen):
        raise ValueError("candidates must contain only Candidate values")
    if (
        isinstance(map_payload, WebSelectorPayload)
        and map_payload.scan_result.candidates != frozen
    ):
        raise ValueError("candidates must exactly match the payload ScanResult")
    session = _SelectorSession(frozen, map_payload)
    _run_server(session)
    return session.wait()


__all__ = ["WebSelectorError", "WebSelectorPayload", "select_candidate"]
