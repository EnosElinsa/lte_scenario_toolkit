import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lte_scenario_toolkit import select_sites
from lte_scenario_toolkit.candidate_scanner import Candidate, ScanResult
from lte_scenario_toolkit.config import load_experiment_config
from lte_scenario_toolkit.data_catalog import load_data_catalog
from lte_scenario_toolkit.selection_service import SelectionProgress

ROOT = Path(__file__).resolve().parents[1]


def _candidate(flat_grid_id=0, point_count=1):
    return Candidate(flat_grid_id, point_count, 0, 0, 1, 1)


def _scan_result(*candidates):
    return ScanResult(tuple(candidates), 4, 4, True, "row-sweep-v1")


def _progress(candidate, *, cache_status="hit"):
    return SelectionProgress(
        phase="completed",
        checked_positions=4,
        total_positions=4,
        candidate_count=1,
        elapsed_seconds=0.1,
        added_candidates=(candidate,),
        removed_flat_grid_ids=(),
        cache_status=cache_status,
        cache_key="a" * 64,
    )


def test_selection_profile_reuses_current_loader_snapshot_and_cli_overrides(tmp_path):
    config = load_experiment_config(
        ROOT / "configs" / "example.yaml",
        output_dir=tmp_path / "runs",
    )
    catalog = load_data_catalog(ROOT / "data" / "datasets.yaml", repo_root=ROOT)
    config["rect_size"] = 3000
    config["target_count"] = 42

    profile = select_sites._selection_profile(
        config,
        catalog,
        config["scenario_id"],
    )

    assert profile.rect_size == 3000
    assert profile.target_count == 42
    assert profile.output_root == tmp_path / "runs"
    assert profile.profile_id == config["profile_id"]


def test_selection_profile_rejects_scenario_mismatch(tmp_path):
    config = load_experiment_config(
        ROOT / "configs" / "example.yaml",
        output_dir=tmp_path / "runs",
    )
    catalog = load_data_catalog(ROOT / "data" / "datasets.yaml", repo_root=ROOT)

    with pytest.raises(ValueError, match="does not match"):
        select_sites._selection_profile(config, catalog, "different")


def test_selection_io_paths_are_derived_from_preflight(tmp_path):
    preflight = SimpleNamespace(
        scenario_id="chicago",
        output_root=tmp_path / "runs",
        points_path=tmp_path / "stations.geojson",
        boundary_path=tmp_path / "boundary.geojson",
        dem_path=tmp_path / "dem.tif",
    )

    class Catalog:
        @staticmethod
        def scenario(scenario_id):
            assert scenario_id == "chicago"
            return {"scenario_id": scenario_id}

    paths = select_sites._selection_io_paths(
        {"rect_size": 2400, "target_count": 24, "tolerance": 2},
        Catalog(),
        preflight,
    )

    assert paths["points_shp"] == preflight.points_path
    assert paths["boundary_shp"] == preflight.boundary_path
    assert paths["dem_path"] == preflight.dem_path
    assert paths["output_dir"] == preflight.output_root
    assert paths["output_csv"].name == "chicago_2400m_target24_tol2.csv"


@pytest.mark.parametrize(
    ("cache_status", "prefix"),
    [
        ("hit", "Loaded 1 cached candidates:"),
        ("miss", "Saved 1 candidates:"),
        ("forced", "Saved 1 candidates:"),
    ],
)
def test_shared_cache_message_reports_current_cache(cache_status, prefix):
    candidate = _candidate()

    message = select_sites._shared_cache_message(
        _scan_result(candidate),
        _progress(candidate, cache_status=cache_status),
    )

    assert message.startswith(prefix)
    assert message.endswith(f"{'a' * 64}.json")


def test_shared_cache_message_requires_a_completed_cache_event():
    candidate = _candidate()
    progress = replace(_progress(candidate), cache_status="pending")

    with pytest.raises(ValueError, match="cache status"):
        select_sites._shared_cache_message(_scan_result(candidate), progress)


def test_scanned_candidate_requires_one_authoritative_match():
    candidate = _candidate()
    scan_result = _scan_result(candidate)

    assert select_sites._scanned_candidate(candidate, scan_result) is candidate
    with pytest.raises(ValueError, match="scanned Candidate"):
        select_sites._scanned_candidate({"flat_grid_id": 0}, scan_result)
    with pytest.raises(ValueError, match="exactly one"):
        select_sites._scanned_candidate(replace(candidate, point_count=2), scan_result)


def test_select_index_bypasses_web_selector(monkeypatch):
    first = _candidate()
    second = _candidate(1, 2)
    monkeypatch.setattr(
        select_sites,
        "_web_selected_candidate",
        lambda *args, **kwargs: pytest.fail("web selector must be bypassed"),
    )

    selected = select_sites._select_candidate(
        _scan_result(first, second),
        select_index=2,
        config={"repo_root": ROOT},
        preflight=object(),
        selection_service=object(),
    )

    assert selected is second


@pytest.mark.parametrize("index", [0, 3, True, 1.0])
def test_select_index_rejects_invalid_values(index):
    with pytest.raises(ValueError, match="between 1 and 1"):
        select_sites._select_candidate(
            _scan_result(_candidate()),
            select_index=index,
            config={"repo_root": ROOT},
            preflight=object(),
            selection_service=object(),
        )


def test_web_selection_accepts_cancel_and_rejects_untrusted_candidate(monkeypatch):
    candidate = _candidate()
    scan_result = _scan_result(candidate)
    values = iter((None, replace(candidate, point_count=2)))
    monkeypatch.setattr(
        select_sites,
        "_web_selected_candidate",
        lambda *args, **kwargs: next(values),
    )

    assert (
        select_sites._select_candidate(
            scan_result,
            select_index=None,
            config={"repo_root": ROOT},
            preflight=object(),
            selection_service=object(),
        )
        is None
    )
    with pytest.raises(ValueError, match="exactly one"):
        select_sites._select_candidate(
            scan_result,
            select_index=None,
            config={"repo_root": ROOT},
            preflight=object(),
            selection_service=object(),
        )


def test_web_selector_import_failure_only_recommends_select_index(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "lte_scenario_toolkit.web_selector", None)

    with pytest.raises(select_sites.SelectorError, match="--select-index") as error:
        select_sites._web_selected_candidate(
            _scan_result(_candidate()),
            preflight=object(),
            selection_service=object(),
            repo_root=tmp_path,
            scan_progress=_progress(_candidate(), cache_status="miss"),
        )

    assert "legacy" not in str(error.value).lower()


def test_web_selector_receives_real_scan_provenance(tmp_path, monkeypatch):
    candidate = _candidate()
    progress = replace(
        _progress(candidate, cache_status="forced"),
        elapsed_seconds=8.2,
        cache_key="real-cache-key",
    )
    captured = {}

    class Payload:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = SimpleNamespace(
        WebSelectorError=RuntimeError,
        WebSelectorPayload=Payload,
        select_candidate=lambda candidates, *, map_payload: candidates[0],
    )
    monkeypatch.setitem(
        sys.modules,
        "lte_scenario_toolkit.web_selector",
        fake_module,
    )

    selected = select_sites._web_selected_candidate(
        _scan_result(candidate),
        preflight=SimpleNamespace(profile=object()),
        selection_service=object(),
        repo_root=tmp_path,
        scan_progress=progress,
    )

    assert selected is candidate
    provenance = captured["scan_provenance"]
    assert provenance.elapsed_seconds == pytest.approx(8.2)
    assert provenance.cache_status == "forced"
    assert provenance.cache_key == "real-cache-key"


def test_export_artifacts_uses_current_profile_flags():
    assert select_sites._export_artifacts(
        {
            "save_csv": True,
            "save_preview_png": False,
            "save_terrain_png": True,
            "save_terrain_eps": False,
            "save_terrain_html": True,
        }
    ) == ("csv", "terrain_png", "terrain_html")


def test_publish_candidate_delegates_to_unique_run_export(tmp_path):
    candidate = _candidate()
    scan_result = _scan_result(candidate)
    preflight = SimpleNamespace(output_root=tmp_path / "runs")
    expected = tmp_path / "runs" / "city" / "default" / "run-1"

    class Service:
        @staticmethod
        def export(*args, **kwargs):
            assert args == (preflight, scan_result, candidate)
            assert kwargs == {
                "output_root": preflight.output_root,
                "artifacts": ("csv",),
                "entrypoint": ("lte-select-sites",),
            }
            return expected

    assert select_sites._publish_candidate(
        Service(),
        preflight,
        scan_result,
        candidate,
        artifacts=("csv",),
        entrypoint=("lte-select-sites",),
    ) == expected


def test_report_published_run_prints_artifacts_and_partial_errors(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "artifacts": ["scenario.csv", "preview.png"],
                "errors": [
                    {"artifact": "terrain_png", "message": "render failed"}
                ],
            }
        ),
        encoding="utf-8",
    )

    select_sites._report_published_run(run_dir)

    captured = capsys.readouterr()
    assert f"Scenario CSV: {run_dir / 'scenario.csv'}" in captured.out
    assert f"Preview: {run_dir / 'preview.png'}" in captured.out
    assert f"Run record: {run_dir / 'run.json'}" in captured.out
    assert "WARNING: terrain_png: render failed" in captured.err


def test_parser_exposes_only_unique_output_and_web_or_index_selection(tmp_path):
    parsed = select_sites._parse_args(
        [
            "--config",
            str(tmp_path / "profile.yaml"),
            "--output-root",
            str(tmp_path / "runs"),
            "--select-index",
            "2",
        ]
    )

    assert parsed.output_root == tmp_path / "runs"
    assert parsed.select_index == 2
    with pytest.raises(SystemExit):
        select_sites._parse_args(
            ["--config", str(tmp_path / "profile.yaml"), "--selector", "legacy"]
        )
    with pytest.raises(SystemExit):
        select_sites._parse_args(
            ["--config", str(tmp_path / "profile.yaml"), "--output-dir", "out"]
        )


def test_main_publishes_one_selected_candidate(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "profile.yaml"
    output_root = tmp_path / "runs"
    config = {
        "repo_root": tmp_path,
        "config_path": config_path,
        "scenario_id": "chicago",
        "output_root": output_root,
        "rect_size": 2400,
        "target_count": 24,
        "tolerance": 2,
        "save_csv": True,
        "save_preview_png": False,
        "save_terrain_png": False,
        "save_terrain_eps": False,
        "save_terrain_html": False,
    }
    preflight = SimpleNamespace(
        scenario_id="chicago",
        output_root=output_root,
        points_path=tmp_path / "stations.geojson",
        boundary_path=tmp_path / "boundary.geojson",
        dem_path=tmp_path / "dem.tif",
    )
    candidate = _candidate()
    scan_result = _scan_result(candidate)
    run_dir = output_root / "chicago" / "default" / "run-1"
    export_calls = []

    class Catalog:
        @staticmethod
        def scenario(scenario_id):
            return {"scenario_id": scenario_id}

    class Service:
        @staticmethod
        def preflight(profile, output_root):
            assert profile is profile_snapshot
            assert output_root == config["output_root"]
            return preflight

        @staticmethod
        def scan(received, *, progress):
            assert received is preflight
            progress(_progress(candidate, cache_status="miss"))
            return scan_result

        @staticmethod
        def export(
            received,
            completed,
            selected,
            *,
            output_root,
            artifacts,
            entrypoint,
        ):
            assert (received, completed, selected) == (
                preflight,
                scan_result,
                candidate,
            )
            assert output_root == config["output_root"]
            assert artifacts == ("csv",)
            assert entrypoint == [
                "lte-select-sites",
                "--config",
                str(config_path),
                "--select-index",
                "1",
            ]
            export_calls.append(selected)
            run_dir.mkdir(parents=True)
            (run_dir / "scenario.csv").write_text("cell\n1\n", encoding="utf-8")
            (run_dir / "run.json").write_text(
                json.dumps({"artifacts": ["scenario.csv"], "errors": []}),
                encoding="utf-8",
            )
            return run_dir

    profile_snapshot = object()
    monkeypatch.setattr(
        select_sites,
        "load_experiment_config",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(select_sites, "load_data_catalog", lambda *args, **kwargs: Catalog())
    monkeypatch.setattr(
        select_sites,
        "_selection_profile",
        lambda *args: profile_snapshot,
    )
    monkeypatch.setattr(select_sites, "SelectionService", lambda catalog: Service())

    exit_code = select_sites.main(
        ["--config", str(config_path), "--select-index", "1"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert export_calls == [candidate]
    assert "Saved 1 candidates:" in captured.out
    assert "Scenario CSV:" in captured.out
    assert "Run record:" in captured.out
    assert captured.err == ""


def test_main_maps_preflight_failure_to_exit_code_two_without_creating_output(
    tmp_path,
    monkeypatch,
    capsys,
):
    output_root = tmp_path / "runs"
    config = {
        "repo_root": tmp_path,
        "config_path": tmp_path / "profile.yaml",
        "scenario_id": "chicago",
        "output_root": output_root,
        "rect_size": 2400,
        "target_count": 24,
        "tolerance": 2,
    }

    class RejectingService:
        def __init__(self, catalog):
            pass

        @staticmethod
        def preflight(profile, output_root):
            raise ValueError("preflight rejected")

    monkeypatch.setattr(
        select_sites,
        "load_experiment_config",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(select_sites, "load_data_catalog", lambda *args, **kwargs: object())
    monkeypatch.setattr(select_sites, "_selection_profile", lambda *args: object())
    monkeypatch.setattr(select_sites, "SelectionService", RejectingService)

    exit_code = select_sites.main(["--config", str(config["config_path"])])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "preflight rejected" in captured.err
    assert not output_root.exists()
