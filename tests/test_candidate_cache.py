import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

import lte_scenario_toolkit.candidate_cache as cache_module
from lte_scenario_toolkit.candidate_cache import (
    CACHE_SCHEMA_VERSION,
    CandidateCache,
    cache_key,
)
from lte_scenario_toolkit.candidate_scanner import Candidate, ScanRequest, ScanResult


def make_request(**overrides):
    values = {
        "rectangle_size": 2,
        "target_count": 2,
        "tolerance": 1,
        "step": 1,
        "max_candidates": 2,
        "minimum_spacing": 2,
        "strategy": "sequential",
        "mode": "fast",
        "random_seed": 7,
        "algorithm_version": "row-sweep-v1",
    }
    values.update(overrides)
    return ScanRequest(**values)


def make_result(**overrides):
    values = {
        "candidates": (
            Candidate(1, 2, 1.0, 1.0, 2.0, 2.0),
            Candidate(4, 3, 4.0, 1.0, 5.0, 2.0),
        ),
        "checked_positions": 8,
        "total_positions": 10,
        "completed": True,
        "algorithm_version": "row-sweep-v1",
    }
    values.update(overrides)
    return ScanResult(**values)


def key_for(request=None):
    return cache_key(
        request or make_request(),
        "chicago",
        "boundary-sha256",
        "points-sha256",
        "EPSG:3857",
    )


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")


def _valid_legacy_setup():
    request = make_request(
        rectangle_size=2,
        target_count=2,
        tolerance=0,
        step=2,
        max_candidates=2,
        minimum_spacing=3,
    )
    boundary = box(-1, -1, 8, 8)
    coordinates = np.asarray(
        [[1.5, 1.5], [2.5, 2.5], [5.5, 1.5], [6.5, 2.5]],
        dtype=float,
    )
    legacy = [
        {
            "left_x": 1.0,
            "bottom_y": 1.0,
            "center_x": 2.0,
            "center_y": 2.0,
            "pt_count": 2,
        },
        {
            "left_x": 5.0,
            "bottom_y": 1.0,
            "center_x": 6.0,
            "center_y": 2.0,
            "pt_count": 2,
        },
    ]
    return request, boundary, coordinates, legacy


@pytest.mark.parametrize(
    ("field", "alternate"),
    [
        ("rectangle_size", 3),
        ("target_count", 3),
        ("tolerance", 0),
        ("step", 2),
        ("max_candidates", 3),
        ("minimum_spacing", 3),
        ("strategy", "uniform"),
        ("mode", "complete"),
        ("random_seed", 8),
        ("algorithm_version", "row-sweep-v2"),
    ],
)
def test_cache_key_is_stable_hex_and_includes_every_request_field(field, alternate):
    request = make_request()
    baseline = key_for(request)

    assert re.fullmatch(r"[0-9a-f]{64}", baseline)
    assert key_for(request) == baseline
    assert key_for(replace(request, **{field: alternate})) != baseline


@pytest.mark.parametrize(
    ("field", "alternate"),
    [
        ("scenario_id", "new-york"),
        ("boundary_fingerprint", "other-boundary"),
        ("points_fingerprint", "other-points"),
        ("target_crs", "EPSG:4326"),
    ],
)
def test_cache_key_includes_all_external_identity_fields(field, alternate):
    arguments = {
        "request": make_request(),
        "scenario_id": "chicago",
        "boundary_fingerprint": "boundary-sha256",
        "points_fingerprint": "points-sha256",
        "target_crs": "EPSG:3857",
    }
    baseline = cache_key(**arguments)
    arguments[field] = alternate

    assert cache_key(**arguments) != baseline


def test_cache_key_includes_schema_version(monkeypatch):
    baseline = key_for()

    monkeypatch.setattr(cache_module, "CACHE_SCHEMA_VERSION", CACHE_SCHEMA_VERSION + 1)

    assert key_for() != baseline


@pytest.mark.parametrize(
    "field",
    ["scenario_id", "boundary_fingerprint", "points_fingerprint", "target_crs"],
)
@pytest.mark.parametrize("invalid", ["", "   ", None, 1])
def test_cache_key_requires_non_empty_string_identity_fields(field, invalid):
    arguments = {
        "request": make_request(),
        "scenario_id": "chicago",
        "boundary_fingerprint": "boundary-sha256",
        "points_fingerprint": "points-sha256",
        "target_crs": "EPSG:3857",
    }
    arguments[field] = invalid

    with pytest.raises(ValueError, match=field):
        cache_key(**arguments)


def test_store_and_load_round_trip_losslessly_without_geometry(tmp_path):
    cache = CandidateCache(tmp_path)
    request = make_request()
    result = make_result()
    key = key_for(request)

    stored = cache.store(key, request, result)
    loaded = cache.load(key, request)

    assert stored == (
        tmp_path / ".lte-data" / "cache" / "candidates" / f"{key}.json"
    ).resolve()
    assert cache.path_for(key) == stored
    assert loaded == result
    raw_text = stored.read_text(encoding="utf-8")
    assert "geometry" not in raw_text
    payload = json.loads(raw_text)
    assert payload["schema_version"] == CACHE_SCHEMA_VERSION
    assert payload["key"] == key
    assert len(payload["result"]["candidates"]) == 2


@pytest.mark.parametrize(
    ("flat_grid_id", "total_positions"),
    [(10, 10), (11, 10), (0, 0)],
)
def test_store_rejects_candidate_ids_outside_total_grid(
    tmp_path,
    flat_grid_id,
    total_positions,
):
    cache = CandidateCache(tmp_path)
    request = make_request()
    key = key_for(request)
    result = make_result(
        candidates=(Candidate(flat_grid_id, 2, 1.0, 1.0, 2.0, 2.0),),
        checked_positions=min(8, total_positions),
        total_positions=total_positions,
    )

    with pytest.raises(ValueError, match="flat_grid_id|total_positions"):
        cache.store(key, request, result)

    assert not cache.path_for(key).exists()


@pytest.mark.parametrize(
    ("flat_grid_id", "total_positions"),
    [(10, 10), (11, 10), (0, 0)],
)
def test_load_quarantines_candidate_ids_outside_total_grid(
    tmp_path,
    flat_grid_id,
    total_positions,
):
    cache = CandidateCache(tmp_path)
    request = make_request()
    key = key_for(request)
    path = cache.store(key, request, make_result())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["result"]["candidates"][0]["flat_grid_id"] = flat_grid_id
    payload["result"]["checked_positions"] = min(8, total_positions)
    payload["result"]["total_positions"] = total_positions
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert cache.load(key, request) is None
    assert not path.exists()
    assert list(path.parent.glob(f"{key}.json.corrupt-*"))


def test_complete_cache_requires_full_coverage_on_store_and_load(tmp_path):
    cache = CandidateCache(tmp_path)
    request = make_request(mode="complete")
    key = key_for(request)
    partial = make_result(checked_positions=8, total_positions=10)

    with pytest.raises(ValueError, match="checked_positions|complete|coverage"):
        cache.store(key, request, partial)

    complete = make_result(checked_positions=10, total_positions=10)
    path = cache.store(key, request, complete)
    assert cache.load(key, request) == complete
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["result"]["checked_positions"] = 8
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert cache.load(key, request) is None
    assert not path.exists()
    assert list(path.parent.glob(f"{key}.json.corrupt-*"))


def test_fast_completed_partial_coverage_remains_cacheable(tmp_path):
    cache = CandidateCache(tmp_path)
    request = make_request(mode="fast")
    result = make_result(checked_positions=8, total_positions=10)
    key = key_for(request)

    path = cache.store(key, request, result)

    assert path.is_file()
    assert cache.load(key, request) == result


def test_corrupt_cache_is_quarantined_collision_safely_without_harming_other_entries(
    tmp_path,
):
    cache = CandidateCache(tmp_path)
    request = make_request()
    corrupt_key = key_for(request)
    other_request = replace(request, random_seed=8)
    other_key = key_for(other_request)
    cache.store(other_key, other_request, make_result())
    corrupt_path = cache.path_for(corrupt_key)
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(2):
        corrupt_path.write_text("{not json", encoding="utf-8")
        assert cache.load(corrupt_key, request) is None

    quarantined = sorted(corrupt_path.parent.glob(f"{corrupt_key}.json.corrupt-*"))
    assert len(quarantined) == 2
    assert len({path.name for path in quarantined}) == 2
    assert not corrupt_path.exists()
    assert cache.load(other_key, other_request) == make_result()


def test_quarantine_does_not_move_concurrent_valid_atomic_replacement(
    tmp_path,
    monkeypatch,
):
    cache = CandidateCache(tmp_path)
    request = make_request()
    result = make_result()
    key = key_for(request)
    path = cache.path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken json", encoding="utf-8")
    real_quarantine = cache._quarantine
    replaced = False

    def replace_then_quarantine(candidate, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            cache.store(key, request, result)
            replaced = True
        return real_quarantine(candidate, *args, **kwargs)

    monkeypatch.setattr(cache, "_quarantine", replace_then_quarantine)

    assert cache.load(key, request) is None
    assert replaced is True
    assert path.is_file()
    assert cache.load(key, request) == result
    assert not list(path.parent.glob(f"{key}.json.corrupt-*"))


def test_symlink_quarantine_does_not_move_concurrent_valid_replacement(
    tmp_path,
    monkeypatch,
):
    cache = CandidateCache(tmp_path)
    request = make_request()
    result = make_result()
    key = key_for(request)
    path = cache.path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside-broken.json"
    outside.write_text("{broken json", encoding="utf-8")
    _symlink_or_skip(path, outside)
    real_quarantine = cache._quarantine
    replaced = False

    def replace_then_quarantine(candidate, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            candidate.unlink()
            cache.store(key, request, result)
            replaced = True
        return real_quarantine(candidate, *args, **kwargs)

    monkeypatch.setattr(cache, "_quarantine", replace_then_quarantine)

    assert cache.load(key, request) is None
    assert replaced is True
    assert path.is_file() and not path.is_symlink()
    assert cache.load(key, request) == result
    assert outside.read_text(encoding="utf-8") == "{broken json"


def test_missing_cache_returns_none_without_creating_cache_directories(tmp_path):
    cache = CandidateCache(tmp_path)

    assert cache.load(key_for(), make_request()) is None
    assert not (tmp_path / ".lte-data").exists()


def test_store_rejects_incomplete_results_without_writing(tmp_path):
    cache = CandidateCache(tmp_path)
    request = make_request()
    key = key_for(request)

    with pytest.raises(ValueError, match="completed"):
        cache.store(key, request, make_result(completed=False))

    assert not cache.path_for(key).exists()


def test_store_rejects_more_candidates_than_checked_positions(tmp_path):
    cache = CandidateCache(tmp_path)
    request = make_request()
    key = key_for(request)

    with pytest.raises(ValueError, match="candidates|checked_positions"):
        cache.store(key, request, make_result(checked_positions=1))

    assert not cache.path_for(key).exists()


def test_load_quarantines_more_candidates_than_checked_positions(tmp_path):
    cache = CandidateCache(tmp_path)
    request = make_request()
    key = key_for(request)
    path = cache.store(key, request, make_result())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["result"]["checked_positions"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert cache.load(key, request) is None
    assert not path.exists()
    assert list(path.parent.glob(f"{key}.json.corrupt-*"))


def test_fast_partial_coverage_requires_capacity_candidate_count_on_store(
    tmp_path,
):
    cache = CandidateCache(tmp_path)
    request = make_request(mode="fast", max_candidates=2)
    key = key_for(request)
    partial = make_result(
        candidates=(Candidate(1, 2, 1.0, 1.0, 2.0, 2.0),),
        checked_positions=8,
        total_positions=10,
    )

    with pytest.raises(ValueError, match="fast|max_candidates|candidates"):
        cache.store(key, request, partial)

    assert not cache.path_for(key).exists()


def test_load_quarantines_fast_partial_coverage_below_capacity(tmp_path):
    cache = CandidateCache(tmp_path)
    request = make_request(mode="fast", max_candidates=2)
    key = key_for(request)
    path = cache.store(key, request, make_result())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["result"]["candidates"].pop()
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert cache.load(key, request) is None
    assert not path.exists()
    assert list(path.parent.glob(f"{key}.json.corrupt-*"))


def test_fast_full_coverage_may_store_fewer_than_capacity(tmp_path):
    cache = CandidateCache(tmp_path)
    request = make_request(mode="fast", max_candidates=2)
    key = key_for(request)
    result = make_result(
        candidates=(Candidate(1, 2, 1.0, 1.0, 2.0, 2.0),),
        checked_positions=10,
        total_positions=10,
    )

    path = cache.store(key, request, result)

    assert path.is_file()
    assert cache.load(key, request) == result


@pytest.mark.parametrize(
    "case",
    [
        "schema",
        "payload_key",
        "request_missing",
        "result_missing",
        "algorithm",
        "too_many_candidates",
        "duplicate_id",
        "boolean_integer",
        "non_finite",
        "center_mismatch",
        "count_outside_tolerance",
        "negative_flat_id",
        "spacing_violation",
        "checked_boolean",
        "checked_above_total",
        "negative_total",
        "not_completed",
    ],
)
def test_load_quarantines_schema_and_result_invariant_damage(tmp_path, case):
    cache = CandidateCache(tmp_path)
    request = make_request()
    key = key_for(request)
    path = cache.store(key, request, make_result())
    payload = json.loads(path.read_text(encoding="utf-8"))

    if case == "schema":
        payload["schema_version"] = CACHE_SCHEMA_VERSION + 1
    elif case == "payload_key":
        payload["key"] = "0" * 64
    elif case == "request_missing":
        payload["request"].pop("mode")
    elif case == "result_missing":
        payload["result"].pop("completed")
    elif case == "algorithm":
        payload["result"]["algorithm_version"] = "wrong-version"
    elif case == "too_many_candidates":
        extra = dict(payload["result"]["candidates"][1])
        extra.update({"flat_grid_id": 8, "left_x": 8.0, "center_x": 9.0})
        payload["result"]["candidates"].append(extra)
    elif case == "duplicate_id":
        payload["result"]["candidates"][1]["flat_grid_id"] = 1
    elif case == "boolean_integer":
        payload["result"]["candidates"][0]["point_count"] = True
    elif case == "non_finite":
        payload["result"]["candidates"][0]["left_x"] = float("nan")
    elif case == "center_mismatch":
        payload["result"]["candidates"][0]["center_x"] = 99.0
    elif case == "count_outside_tolerance":
        payload["result"]["candidates"][0]["point_count"] = 99
    elif case == "negative_flat_id":
        payload["result"]["candidates"][0]["flat_grid_id"] = -1
    elif case == "spacing_violation":
        payload["result"]["candidates"][1].update(
            {"left_x": 1.5, "bottom_y": 1.0, "center_x": 2.5, "center_y": 2.0}
        )
    elif case == "checked_boolean":
        payload["result"]["checked_positions"] = True
    elif case == "checked_above_total":
        payload["result"]["checked_positions"] = 11
    elif case == "negative_total":
        payload["result"]["total_positions"] = -1
    elif case == "not_completed":
        payload["result"]["completed"] = False

    path.write_text(json.dumps(payload), encoding="utf-8")

    assert cache.load(key, request) is None
    assert not path.exists()
    assert list(path.parent.glob(f"{key}.json.corrupt-*"))


def test_wrong_expected_request_is_quarantined(tmp_path):
    cache = CandidateCache(tmp_path)
    request = make_request()
    key = key_for(request)
    path = cache.store(key, request, make_result())

    assert cache.load(key, replace(request, random_seed=8)) is None
    assert not path.exists()
    assert list(path.parent.glob(f"{key}.json.corrupt-*"))


@pytest.mark.parametrize("key", ["", "abc", "A" * 64, "../" + "0" * 64])
def test_cache_paths_reject_invalid_keys(tmp_path, key):
    cache = CandidateCache(tmp_path)

    with pytest.raises(ValueError, match="key"):
        cache.path_for(key)
    with pytest.raises(ValueError, match="key"):
        cache.load(key, make_request())
    with pytest.raises(ValueError, match="key"):
        cache.store(key, make_request(), make_result())


def test_atomic_store_failure_leaves_no_new_final_or_partial_files(tmp_path, monkeypatch):
    cache = CandidateCache(tmp_path)
    request = make_request()
    key = key_for(request)
    path = cache.path_for(key)

    def fail_atomic_write(destination, payload):
        del destination, payload
        raise OSError("write failed")

    monkeypatch.setattr(cache_module, "atomic_write_json", fail_atomic_write)

    with pytest.raises(OSError, match="write failed"):
        cache.store(key, request, make_result())

    assert not path.exists()
    assert not list(path.parent.glob(f".{path.name}.*.tmp")) if path.parent.exists() else True


def test_atomic_store_failure_preserves_existing_final(tmp_path, monkeypatch):
    cache = CandidateCache(tmp_path)
    request = make_request()
    key = key_for(request)
    path = cache.store(key, request, make_result())
    original = path.read_bytes()

    monkeypatch.setattr(
        cache_module,
        "atomic_write_json",
        lambda destination, payload: (_ for _ in ()).throw(OSError("write failed")),
    )

    with pytest.raises(OSError, match="write failed"):
        cache.store(key, request, make_result())

    assert path.read_bytes() == original


def test_load_never_follows_symlink_cache_file_or_modifies_external_target(tmp_path):
    cache = CandidateCache(tmp_path / "repo")
    request = make_request()
    key = key_for(request)
    cache_path = cache.path_for(key)
    cache_path.parent.mkdir(parents=True)
    external = tmp_path / "external.json"
    external.write_text("external owner\n", encoding="utf-8")
    _symlink_or_skip(cache_path, external)

    assert cache.load(key, request) is None
    assert external.read_text(encoding="utf-8") == "external owner\n"
    assert not cache_path.exists()
    assert list(cache_path.parent.glob(f"{key}.json.corrupt-*"))


def test_cache_rejects_symlinked_parent_escape_without_touching_external_directory(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink_or_skip(repo / ".lte-data", outside, directory=True)
    cache = CandidateCache(repo)

    with pytest.raises(ValueError, match="symlink|outside|cache"):
        cache.path_for(key_for())

    assert list(outside.iterdir()) == []


def test_import_legacy_validates_and_atomically_overwrites_same_key(tmp_path):
    cache = CandidateCache(tmp_path)
    request, boundary, coordinates, legacy = _valid_legacy_setup()
    key = key_for(request)
    existing = ScanResult(
        candidates=(Candidate(5, 2, 1.0, 1.0, 2.0, 2.0),),
        checked_positions=16,
        total_positions=16,
        completed=True,
        algorithm_version=request.algorithm_version,
    )
    cache.store(key, request, existing)
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy, indent=2) + "\n", encoding="utf-8")
    original_legacy = legacy_path.read_bytes()

    imported = cache.import_legacy(
        legacy_path,
        key,
        request,
        boundary,
        coordinates,
    )

    assert imported.candidates == (
        Candidate(5, 2, 1.0, 1.0, 2.0, 2.0),
        Candidate(7, 2, 5.0, 1.0, 6.0, 2.0),
    )
    assert imported.checked_positions == imported.total_positions == 16
    assert cache.load(key, request) == imported
    assert legacy_path.read_bytes() == original_legacy


@pytest.mark.parametrize(
    "case",
    ["wrong_count", "touching", "spacing", "off_grid", "missing_field", "not_list"],
)
def test_import_legacy_rejects_invalid_content_without_publishing(tmp_path, case):
    cache = CandidateCache(tmp_path)
    request, boundary, coordinates, legacy = _valid_legacy_setup()
    document = json.loads(json.dumps(legacy))
    if case == "wrong_count":
        document[0]["pt_count"] = 3
    elif case == "touching":
        request = replace(request, target_count=0)
        coordinates = np.empty((0, 2), dtype=float)
        document = [
            {
                "left_x": -1.0,
                "bottom_y": -1.0,
                "center_x": 0.0,
                "center_y": 0.0,
                "pt_count": 0,
            }
        ]
    elif case == "spacing":
        request = replace(request, target_count=0)
        coordinates = np.empty((0, 2), dtype=float)
        document = [
            {
                "left_x": 1.0,
                "bottom_y": 1.0,
                "center_x": 2.0,
                "center_y": 2.0,
                "pt_count": 0,
            },
            {
                "left_x": 3.0,
                "bottom_y": 1.0,
                "center_x": 4.0,
                "center_y": 2.0,
                "pt_count": 0,
            },
        ]
    elif case == "off_grid":
        request = replace(request, target_count=0)
        coordinates = np.empty((0, 2), dtype=float)
        document = [
            {
                "left_x": 1.25,
                "bottom_y": 1.0,
                "center_x": 2.25,
                "center_y": 2.0,
                "pt_count": 0,
            }
        ]
    elif case == "missing_field":
        document[0].pop("center_x")
    elif case == "not_list":
        document = {"results": document}
    key = key_for(request)
    legacy_path = tmp_path / f"legacy-{case}.json"
    legacy_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        cache.import_legacy(
            legacy_path,
            key,
            request,
            boundary,
            coordinates,
        )

    assert not cache.path_for(key).exists()
