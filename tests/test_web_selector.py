from __future__ import annotations

import sys
from threading import Barrier, Thread
from time import sleep
from types import SimpleNamespace

import pytest

from lte_scenario_toolkit.candidate_scanner import Candidate


def _candidates() -> list[Candidate]:
    return [
        Candidate(0, 1, 0.0, 0.0, 1.0, 1.0),
        Candidate(4, 2, 4.0, 0.0, 5.0, 1.0),
    ]


def test_web_selector_freezes_candidates_and_confirms_zero_based_index(monkeypatch):
    from lte_scenario_toolkit import web_selector

    candidates = _candidates()

    def choose_second(session):
        assert session.candidates == tuple(candidates)
        assert isinstance(session.candidates, tuple)
        assert session.confirm(1) is True

    monkeypatch.setattr(web_selector, "_run_server", choose_second)

    chosen = web_selector.select_candidate(candidates, map_payload={})

    assert chosen is candidates[1]


def test_web_selector_blocks_until_another_thread_settles(monkeypatch):
    from lte_scenario_toolkit import web_selector

    candidates = _candidates()
    worker = None

    def confirm_later(session):
        nonlocal worker

        def settle():
            sleep(0.05)
            session.confirm(0)

        worker = Thread(target=settle)
        worker.start()

    monkeypatch.setattr(web_selector, "_run_server", confirm_later)

    chosen = web_selector.select_candidate(candidates, map_payload={})
    worker.join(timeout=1)

    assert chosen is candidates[0]


def test_web_selector_close_cancels_once_and_cannot_be_overwritten(monkeypatch):
    from lte_scenario_toolkit import web_selector

    def close_then_confirm(session):
        assert session.close() is True
        assert session.cancel() is False
        assert session.confirm(0) is False

    monkeypatch.setattr(web_selector, "_run_server", close_then_confirm)

    assert web_selector.select_candidate(_candidates(), map_payload={}) is None


def test_web_selector_concurrent_confirm_and_cancel_settle_exactly_once(monkeypatch):
    from lte_scenario_toolkit import web_selector

    outcomes = []

    def race(session):
        barrier = Barrier(3)

        def confirm():
            barrier.wait()
            outcomes.append(session.confirm(0))

        def cancel():
            barrier.wait()
            outcomes.append(session.cancel())

        threads = (Thread(target=confirm), Thread(target=cancel))
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=1)
            assert thread.is_alive() is False

    monkeypatch.setattr(web_selector, "_run_server", race)

    chosen = web_selector.select_candidate(_candidates(), map_payload={})

    assert sorted(outcomes) == [False, True]
    assert chosen is None or chosen == _candidates()[0]


def test_web_selector_payload_requires_a_completed_frozen_scan(tmp_path):
    from lte_scenario_toolkit import web_selector
    from lte_scenario_toolkit.candidate_scanner import ScanResult

    profile = object()
    preflight = SimpleNamespace(profile=profile)
    result = ScanResult(
        candidates=tuple(_candidates()),
        checked_positions=10,
        total_positions=10,
        completed=True,
        algorithm_version="row-sweep-v1",
    )

    payload = web_selector.WebSelectorPayload(
        preflight=preflight,
        selection_service=object(),
        scan_result=result,
        repo_root=tmp_path,
    )

    assert payload.scan_result is result
    assert payload.repo_root == tmp_path.resolve()
    with pytest.raises(ValueError, match="completed"):
        web_selector.WebSelectorPayload(
            preflight=preflight,
            selection_service=object(),
            scan_result=ScanResult(
                candidates=result.candidates,
                checked_positions=5,
                total_positions=10,
                completed=False,
                algorithm_version="row-sweep-v1",
            ),
            repo_root=tmp_path,
        )


def test_web_selector_rejects_candidates_that_differ_from_payload_scan(
    tmp_path,
    monkeypatch,
):
    from lte_scenario_toolkit import web_selector
    from lte_scenario_toolkit.candidate_scanner import ScanResult

    candidates = tuple(_candidates())
    payload = web_selector.WebSelectorPayload(
        preflight=SimpleNamespace(profile=object()),
        selection_service=object(),
        scan_result=ScanResult(
            candidates=candidates,
            checked_positions=10,
            total_positions=10,
            completed=True,
            algorithm_version="row-sweep-v1",
        ),
        repo_root=tmp_path,
    )
    monkeypatch.setattr(
        web_selector,
        "_run_server",
        lambda _session: pytest.fail("mismatched candidates must fail before server start"),
    )

    with pytest.raises(ValueError, match="ScanResult"):
        web_selector.select_candidate(reversed(candidates), map_payload=payload)


def test_run_server_opens_loopback_browser_and_releases_owned_runtime(
    tmp_path,
    monkeypatch,
):
    from threading import Event

    from lte_scenario_toolkit import web_selector
    from lte_scenario_toolkit.candidate_scanner import ScanResult

    candidates = tuple(_candidates())
    payload = web_selector.WebSelectorPayload(
        preflight=SimpleNamespace(profile=object()),
        selection_service=object(),
        scan_result=ScanResult(
            candidates=candidates,
            checked_positions=10,
            total_positions=10,
            completed=True,
            algorithm_version="row-sweep-v1",
        ),
        repo_root=tmp_path,
    )
    session = web_selector._SelectorSession(candidates, payload)
    browser_opened = Event()
    stopped = Event()
    calls = {"shutdown": 0, "coordinator_shutdown": 0}

    class FakeApp:
        def shutdown(self):
            calls["shutdown"] += 1
            stopped.set()

    class FakeUi:
        def run(self, root, **kwargs):
            calls["run"] = kwargs
            assert browser_opened.wait(1)
            root()
            assert stopped.wait(1)

    class FakeCoordinator:
        def shutdown(self):
            calls["coordinator_shutdown"] += 1

    fake_app = FakeApp()
    fake_ui = FakeUi()
    monkeypatch.setitem(
        sys.modules,
        "nicegui",
        SimpleNamespace(app=fake_app, ui=fake_ui),
    )
    monkeypatch.setattr(web_selector, "JobCoordinator", FakeCoordinator)
    monkeypatch.setattr(
        web_selector,
        "_build_candidate_session",
        lambda _payload: (object(), object()),
        raising=False,
    )
    monkeypatch.setattr(web_selector, "_available_port", lambda: 43123, raising=False)
    monkeypatch.setattr(
        web_selector,
        "_wait_for_server",
        lambda host, port, stop: host == "127.0.0.1" and port == 43123,
        raising=False,
    )

    def open_browser(url):
        calls["url"] = url
        browser_opened.set()

    monkeypatch.setattr(
        web_selector,
        "_open_default_browser",
        open_browser,
        raising=False,
    )
    monkeypatch.setattr(
        web_selector,
        "_wait_for_port_release",
        lambda host, port: host == "127.0.0.1" and port == 43123,
        raising=False,
    )

    def render_page(selector, *_args, **_kwargs):
        selector.confirm(1)
        fake_app.shutdown()

    monkeypatch.setattr(
        web_selector,
        "_render_selector_page",
        render_page,
        raising=False,
    )

    web_selector._run_server(session)

    assert session.wait(0) is candidates[1]
    assert calls["run"]["host"] == "127.0.0.1"
    assert calls["run"]["port"] == 43123
    assert calls["run"]["show"] is False
    assert calls["run"]["reload"] is False
    assert calls["url"] == "http://127.0.0.1:43123/"
    assert calls["shutdown"] >= 1
    assert calls["coordinator_shutdown"] == 1


def test_run_server_browser_failure_is_actionable_and_still_cleans_up(
    tmp_path,
    monkeypatch,
):
    from threading import Event

    from lte_scenario_toolkit import web_selector
    from lte_scenario_toolkit.candidate_scanner import ScanResult

    candidates = tuple(_candidates())
    payload = web_selector.WebSelectorPayload(
        preflight=SimpleNamespace(profile=object()),
        selection_service=object(),
        scan_result=ScanResult(
            candidates=candidates,
            checked_positions=10,
            total_positions=10,
            completed=True,
            algorithm_version="row-sweep-v1",
        ),
        repo_root=tmp_path,
    )
    session = web_selector._SelectorSession(candidates, payload)
    stopped = Event()
    calls = {"shutdown": 0, "coordinator_shutdown": 0}

    class FakeApp:
        def shutdown(self):
            calls["shutdown"] += 1
            stopped.set()

    class FakeUi:
        def run(self, _root, **_kwargs):
            assert stopped.wait(1)

    class FakeCoordinator:
        def shutdown(self):
            calls["coordinator_shutdown"] += 1

    fake_app = FakeApp()
    monkeypatch.setitem(
        sys.modules,
        "nicegui",
        SimpleNamespace(app=fake_app, ui=FakeUi()),
    )
    monkeypatch.setattr(web_selector, "JobCoordinator", FakeCoordinator)
    monkeypatch.setattr(
        web_selector,
        "_build_candidate_session",
        lambda _payload: (object(), object()),
    )
    monkeypatch.setattr(web_selector, "_available_port", lambda: 43124)
    monkeypatch.setattr(
        web_selector,
        "_wait_for_server",
        lambda _host, _port, _stop: True,
    )
    monkeypatch.setattr(
        web_selector,
        "_open_default_browser",
        lambda _url: (_ for _ in ()).throw(
            web_selector.WebSelectorError(
                "Use --select-index because no default browser is available"
            )
        ),
    )
    monkeypatch.setattr(
        web_selector,
        "_wait_for_port_release",
        lambda _host, _port: True,
    )

    with pytest.raises(web_selector.WebSelectorError, match="--select-index"):
        web_selector._run_server(session)

    assert session.wait(0) is None
    assert calls["shutdown"] >= 1
    assert calls["coordinator_shutdown"] == 1


def test_run_server_wraps_unexpected_nicegui_failure_after_cleanup(
    tmp_path,
    monkeypatch,
):
    from threading import Event

    from lte_scenario_toolkit import web_selector
    from lte_scenario_toolkit.candidate_scanner import ScanResult

    candidates = tuple(_candidates())
    payload = web_selector.WebSelectorPayload(
        preflight=SimpleNamespace(profile=object()),
        selection_service=object(),
        scan_result=ScanResult(
            candidates=candidates,
            checked_positions=10,
            total_positions=10,
            completed=True,
            algorithm_version="row-sweep-v1",
        ),
        repo_root=tmp_path,
    )
    session = web_selector._SelectorSession(candidates, payload)
    stop_observed = Event()
    calls = {"shutdown": 0, "coordinator_shutdown": 0}

    class FakeApp:
        def shutdown(self):
            calls["shutdown"] += 1

    class FakeUi:
        @staticmethod
        def run(_root, **_kwargs):
            raise ValueError("broken NiceGUI startup")

    class FakeCoordinator:
        def shutdown(self):
            calls["coordinator_shutdown"] += 1

    monkeypatch.setitem(
        sys.modules,
        "nicegui",
        SimpleNamespace(app=FakeApp(), ui=FakeUi()),
    )
    monkeypatch.setattr(web_selector, "JobCoordinator", FakeCoordinator)
    monkeypatch.setattr(
        web_selector,
        "_build_candidate_session",
        lambda _payload: (object(), object()),
    )
    monkeypatch.setattr(web_selector, "_available_port", lambda: 43125)

    def wait_until_stopped(_host, _port, stop):
        stop_observed.set()
        return stop.wait(1) and False

    monkeypatch.setattr(web_selector, "_wait_for_server", wait_until_stopped)
    monkeypatch.setattr(
        web_selector,
        "_wait_for_port_release",
        lambda _host, _port: True,
    )

    with pytest.raises(web_selector.WebSelectorError, match="broken NiceGUI startup"):
        web_selector._run_server(session)

    assert stop_observed.wait(1)
    assert calls["shutdown"] >= 1
    assert calls["coordinator_shutdown"] == 1


def test_run_server_missing_gui_extra_is_actionable(tmp_path, monkeypatch):
    from lte_scenario_toolkit import web_selector
    from lte_scenario_toolkit.candidate_scanner import ScanResult

    candidates = tuple(_candidates())
    payload = web_selector.WebSelectorPayload(
        preflight=SimpleNamespace(profile=object()),
        selection_service=object(),
        scan_result=ScanResult(
            candidates=candidates,
            checked_positions=10,
            total_positions=10,
            completed=True,
            algorithm_version="row-sweep-v1",
        ),
        repo_root=tmp_path,
    )
    session = web_selector._SelectorSession(candidates, payload)
    monkeypatch.setitem(sys.modules, "nicegui", None)

    with pytest.raises(web_selector.WebSelectorError, match=r"\[gui\].*--select-index"):
        web_selector._run_server(session)


def test_selector_renderer_reuses_precomputed_candidate_page_without_rescan(
    monkeypatch,
):
    from lte_scenario_toolkit import web_selector

    candidates = tuple(_candidates())
    selector = web_selector._SelectorSession(candidates, map_payload={})
    captured = {}
    delete_handlers = []

    class Element:
        def props(self, _value):
            return self

        def mark(self, _value):
            return self

    class FakeUi:
        context = SimpleNamespace(
            client=SimpleNamespace(on_delete=delete_handlers.append)
        )

        @staticmethod
        def button(*_args, **_kwargs):
            return Element()

        @staticmethod
        def notify(*_args, **_kwargs):
            pytest.fail("the authoritative candidate must match exactly")

    class FakeApp:
        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    def render_candidate(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        kwargs["on_confirm"](
            SimpleNamespace(locked_candidate=candidates[1])
        )

    monkeypatch.setattr(web_selector, "render_candidate_page", render_candidate)
    app = FakeApp()
    candidate_session = object()
    coordinator = object()

    web_selector._render_selector_page(
        selector,
        candidate_session,
        object(),
        coordinator,
        FakeUi(),
        app,
    )

    assert captured["args"][2] is candidate_session
    assert captured["args"][3] is coordinator
    assert captured["kwargs"]["auto_start"] is False
    assert captured["kwargs"]["allow_rescan"] is False
    assert selector.wait(0) is candidates[1]
    assert app.shutdown_calls == 1
    assert len(delete_handlers) == 1
    delete_handlers[0]()
    assert selector.wait(0) is candidates[1]
