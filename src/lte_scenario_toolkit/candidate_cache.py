"""Strict, atomic, and safely quarantined candidate-scan caches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import shapely
from shapely.geometry import box

from .candidate_scanner import Candidate, ScanRequest, ScanResult, grid_axes
from .io import atomic_write_json
from .scenario import validate_results

CACHE_SCHEMA_VERSION = 1
_CACHE_KEY_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_FIELDS = frozenset({"schema_version", "key", "request", "result"})
_REQUEST_FIELDS = frozenset(field.name for field in fields(ScanRequest))
_RESULT_FIELDS = frozenset(
    {
        "candidates",
        "checked_positions",
        "total_positions",
        "completed",
        "algorithm_version",
    }
)
_CANDIDATE_FIELDS = frozenset(field.name for field in fields(Candidate))
_LEGACY_COORDINATE_DECIMALS = 2
_THREAD_LOCKS_GUARD = Lock()
_THREAD_LOCKS: dict[Path, Lock] = {}


@dataclass(frozen=True)
class _CacheLeafSnapshot:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    digest: str | None = None


def _leaf_snapshot(path: Path) -> _CacheLeafSnapshot:
    status = path.lstat()
    return _CacheLeafSnapshot(
        device=status.st_dev,
        inode=status.st_ino,
        mode=status.st_mode,
        size=status.st_size,
        mtime_ns=status.st_mtime_ns,
    )


def _same_leaf(
    left: _CacheLeafSnapshot,
    right: _CacheLeafSnapshot,
) -> bool:
    return replace(left, digest=None) == replace(right, digest=None)


def _is_redirected_path(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    try:
        attributes = path.lstat().st_file_attributes
    except FileNotFoundError:
        return False
    except AttributeError:
        return False
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_point)


def _assert_unredirected_regular_file(path: Path, *, description: str) -> Path:
    """Reject traversal and every existing redirected component in one file path."""

    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if ".." in candidate.parts:
        raise ValueError(f"{description} must not contain path traversal: {path}")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        if os.path.lexists(current) and _is_redirected_path(current):
            raise ValueError(f"{description} must not use redirected paths: {path}")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{description} is not a readable regular file: {path}") from exc
    if resolved != candidate or not resolved.is_file():
        raise ValueError(f"{description} must be an unredirected regular file: {path}")
    return resolved


def _thread_lock_for(path: Path) -> Lock:
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(path, Lock())


def _open_lock_file(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ValueError(f"Candidate cache lock cannot be opened safely: {path}") from exc

    try:
        opened = os.fstat(descriptor)
        leaf = path.lstat()
        if (
            _is_redirected_path(path)
            or not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(leaf.st_mode)
            or opened.st_dev != leaf.st_dev
            or opened.st_ino != leaf.st_ino
        ):
            raise ValueError(f"Candidate cache lock must be a regular file: {path}")
        if opened.st_size == 0:
            os.write(descriptor, b"\0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException as primary_error:
        try:
            os.close(descriptor)
        except BaseException as cleanup_error:
            if hasattr(primary_error, "add_note"):
                primary_error.add_note(
                    f"Candidate cache lock close failed: {cleanup_error}"
                )
        raise


def _acquire_os_lock(descriptor: int) -> None:
    if os.name == "nt":
        import errno
        import msvcrt

        busy_errors = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
        while True:
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                if exc.errno not in busy_errors:
                    raise
                time.sleep(0.05)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX)


def _release_os_lock(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)


def _required_text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validated_key(key: Any) -> str:
    if type(key) is not str or _CACHE_KEY_PATTERN.fullmatch(key) is None:
        raise ValueError("cache key must be 64 lowercase hexadecimal characters")
    return key


def cache_key(
    request: ScanRequest,
    scenario_id: str,
    boundary_fingerprint: str,
    points_fingerprint: str,
    target_crs: str,
) -> str:
    """Hash every input that determines a candidate scan result."""

    if not isinstance(request, ScanRequest):
        raise ValueError("request must be a ScanRequest")
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "request": asdict(request),
        "scenario_id": _required_text(scenario_id, field="scenario_id"),
        "boundary_fingerprint": _required_text(
            boundary_fingerprint,
            field="boundary_fingerprint",
        ),
        "points_fingerprint": _required_text(
            points_fingerprint,
            field="points_fingerprint",
        ),
        "target_crs": _required_text(target_crs, field="target_crs"),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _legacy_filename_number(value: Real, *, field: str) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return str(int(number)) if number.is_integer() else format(number, ".15g")


def legacy_cache_filename(scenario_tag: str, request: ScanRequest) -> str:
    """Return the exact cache basename emitted by the legacy CLI workflow."""

    if (
        type(scenario_tag) is not str
        or not scenario_tag.strip()
        or scenario_tag != scenario_tag.strip()
        or Path(scenario_tag).name != scenario_tag
        or "/" in scenario_tag
        or "\\" in scenario_tag
    ):
        raise ValueError("scenario_tag must be one non-empty path-safe filename token")
    if not isinstance(request, ScanRequest):
        raise ValueError("request must be a ScanRequest")
    rectangle_size = _legacy_filename_number(
        request.rectangle_size,
        field="request.rectangle_size",
    )
    step = _legacy_filename_number(request.step, field="request.step")
    spacing = _legacy_filename_number(
        request.minimum_spacing,
        field="request.minimum_spacing",
    )
    return (
        f"{scenario_tag}_{rectangle_size}m_"
        f"target{request.target_count}_tol{request.tolerance}_"
        f"step{step}_sp{spacing}_{request.strategy}_"
        f"seed{request.random_seed}_cache.json"
    )


def _strict_mapping(
    value: Any,
    expected_fields: frozenset[str],
    *,
    field: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    if set(value) != expected_fields:
        raise ValueError(f"{field} has invalid fields")
    return value


def _strict_integer(
    value: Any,
    *,
    field: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _request_from_payload(value: Any) -> ScanRequest:
    mapping = _strict_mapping(value, _REQUEST_FIELDS, field="request")
    try:
        return ScanRequest(**mapping)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"request is invalid: {exc}") from exc


def _candidate_from_payload(
    value: Any,
    request: ScanRequest,
    *,
    index: int,
) -> Candidate:
    field_prefix = f"result.candidates[{index}]"
    mapping = _strict_mapping(value, _CANDIDATE_FIELDS, field=field_prefix)
    flat_grid_id = _strict_integer(
        mapping["flat_grid_id"],
        field=f"{field_prefix}.flat_grid_id",
        minimum=0,
    )
    point_count = _strict_integer(
        mapping["point_count"],
        field=f"{field_prefix}.point_count",
        minimum=0,
    )
    left_x = _finite_number(mapping["left_x"], field=f"{field_prefix}.left_x")
    bottom_y = _finite_number(
        mapping["bottom_y"],
        field=f"{field_prefix}.bottom_y",
    )
    center_x = _finite_number(
        mapping["center_x"],
        field=f"{field_prefix}.center_x",
    )
    center_y = _finite_number(
        mapping["center_y"],
        field=f"{field_prefix}.center_y",
    )
    expected_center_x = left_x + request.rectangle_size / 2.0
    expected_center_y = bottom_y + request.rectangle_size / 2.0
    if not math.isclose(center_x, expected_center_x, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{field_prefix}.center_x is inconsistent")
    if not math.isclose(center_y, expected_center_y, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{field_prefix}.center_y is inconsistent")
    target_minimum = request.target_count - request.tolerance
    target_maximum = request.target_count + request.tolerance
    if not target_minimum <= point_count <= target_maximum:
        raise ValueError(f"{field_prefix}.point_count is outside request tolerance")
    return Candidate(
        flat_grid_id=flat_grid_id,
        point_count=point_count,
        left_x=left_x,
        bottom_y=bottom_y,
        center_x=center_x,
        center_y=center_y,
    )


def _result_from_payload(value: Any, request: ScanRequest) -> ScanResult:
    mapping = _strict_mapping(value, _RESULT_FIELDS, field="result")
    candidate_values = mapping["candidates"]
    if not isinstance(candidate_values, list):
        raise ValueError("result.candidates must be an array")
    if len(candidate_values) > request.max_candidates:
        raise ValueError("result.candidates exceeds max_candidates")
    candidates = tuple(
        _candidate_from_payload(item, request, index=index)
        for index, item in enumerate(candidate_values)
    )
    flat_grid_ids = [candidate.flat_grid_id for candidate in candidates]
    if len(flat_grid_ids) != len(set(flat_grid_ids)):
        raise ValueError("result.candidates contains duplicate flat_grid_id values")
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            if (
                math.hypot(
                    left.center_x - right.center_x,
                    left.center_y - right.center_y,
                )
                < request.minimum_spacing
            ):
                raise ValueError("result.candidates violates minimum_spacing")

    checked_positions = _strict_integer(
        mapping["checked_positions"],
        field="result.checked_positions",
        minimum=0,
    )
    total_positions = _strict_integer(
        mapping["total_positions"],
        field="result.total_positions",
        minimum=0,
    )
    if checked_positions > total_positions:
        raise ValueError("result.checked_positions exceeds total_positions")
    if any(
        candidate.flat_grid_id >= total_positions
        for candidate in candidates
    ):
        raise ValueError(
            "result.candidates flat_grid_id must be less than total_positions"
        )
    if len(candidates) > checked_positions:
        raise ValueError(
            "result.candidates exceeds result.checked_positions"
        )
    if request.mode == "complete" and checked_positions != total_positions:
        raise ValueError(
            "result.checked_positions must equal total_positions for complete scans"
        )
    if (
        request.mode == "fast"
        and checked_positions < total_positions
        and len(candidates) != request.max_candidates
    ):
        raise ValueError(
            "fast partial-coverage result.candidates must equal max_candidates"
        )
    if mapping["completed"] is not True:
        raise ValueError("result.completed must be true")
    algorithm_version = _required_text(
        mapping["algorithm_version"],
        field="result.algorithm_version",
    )
    if algorithm_version != request.algorithm_version:
        raise ValueError("result.algorithm_version does not match request")
    return ScanResult(
        candidates=candidates,
        checked_positions=checked_positions,
        total_positions=total_positions,
        completed=True,
        algorithm_version=algorithm_version,
    )


def _result_payload(result: ScanResult) -> dict[str, Any]:
    return {
        "candidates": [asdict(candidate) for candidate in result.candidates],
        "checked_positions": result.checked_positions,
        "total_positions": result.total_positions,
        "completed": result.completed,
        "algorithm_version": result.algorithm_version,
    }


class CandidateCache:
    """Repository-local cache with strict validation and per-file quarantine."""

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.cache_root = self.repo_root / ".lte-data" / "cache" / "candidates"

    def _ensure_cache_root(self) -> None:
        if not self.repo_root.is_dir():
            raise ValueError(
                f"Candidate cache repository root is not a directory: {self.repo_root}"
            )
        current = self.repo_root
        for part in (".lte-data", "cache", "candidates"):
            current /= part
            self._assert_safe_path(current, allow_leaf_symlink=False)
            try:
                current.mkdir()
            except FileExistsError:
                pass
            self._assert_safe_path(current, allow_leaf_symlink=False)
            try:
                status = current.lstat()
            except OSError as exc:
                raise ValueError(
                    f"Candidate cache directory cannot be inspected: {current}"
                ) from exc
            if _is_redirected_path(current) or not stat.S_ISDIR(status.st_mode):
                raise ValueError(
                    f"Candidate cache path must contain only real directories: {current}"
                )

    def _assert_safe_path(
        self,
        candidate: Path,
        *,
        allow_leaf_symlink: bool,
    ) -> Path:
        try:
            relative = candidate.relative_to(self.repo_root)
        except ValueError as exc:
            raise ValueError(f"Candidate cache path is outside repository: {candidate}") from exc
        current = self.repo_root
        for index, part in enumerate(relative.parts):
            current /= part
            is_leaf = index == len(relative.parts) - 1
            if _is_redirected_path(current) and not (
                allow_leaf_symlink and is_leaf
            ):
                raise ValueError(f"Candidate cache path must not use symlinks: {candidate}")
        path_to_resolve = candidate.parent if allow_leaf_symlink else candidate
        try:
            path_to_resolve.resolve(strict=False).relative_to(self.repo_root)
        except ValueError as exc:
            raise ValueError(f"Candidate cache path escapes repository: {candidate}") from exc
        return candidate

    def _path_for(self, key: Any, *, allow_leaf_symlink: bool) -> Path:
        validated = _validated_key(key)
        candidate = self.cache_root / f"{validated}.json"
        return self._assert_safe_path(
            candidate,
            allow_leaf_symlink=allow_leaf_symlink,
        )

    def path_for(self, key: str) -> Path:
        return self._path_for(key, allow_leaf_symlink=False)

    def _lock_path_for(self, key: str) -> Path:
        self._ensure_cache_root()
        candidate = self.cache_root / f"{_validated_key(key)}.lock"
        self._assert_safe_path(candidate, allow_leaf_symlink=False)
        if os.path.lexists(candidate):
            try:
                status = candidate.lstat()
            except OSError as exc:
                raise ValueError(
                    f"Candidate cache lock cannot be inspected: {candidate}"
                ) from exc
            if _is_redirected_path(candidate) or not stat.S_ISREG(status.st_mode):
                raise ValueError(
                    f"Candidate cache lock must be a regular file: {candidate}"
                )
        return candidate

    @contextmanager
    def _key_lock(self, key: str) -> Iterator[None]:
        lock_path = self._lock_path_for(_validated_key(key))
        resolved_lock_path = lock_path.parent.resolve(strict=True) / lock_path.name
        thread_lock = _thread_lock_for(resolved_lock_path)
        with thread_lock:
            descriptor = _open_lock_file(lock_path)
            acquired = False
            primary_error: BaseException | None = None
            try:
                _acquire_os_lock(descriptor)
                acquired = True
                yield
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                cleanup_errors: list[BaseException] = []
                if acquired:
                    try:
                        _release_os_lock(descriptor)
                    except BaseException as exc:
                        cleanup_errors.append(exc)
                try:
                    os.close(descriptor)
                except BaseException as exc:
                    cleanup_errors.append(exc)
                if cleanup_errors and primary_error is None:
                    raise cleanup_errors[0]
                if cleanup_errors and hasattr(primary_error, "add_note"):
                    primary_error.add_note(
                        f"Candidate cache lock cleanup failed: {cleanup_errors[0]}"
                    )

    @staticmethod
    def _quarantine(path: Path, expected: _CacheLeafSnapshot) -> None:
        try:
            current = _leaf_snapshot(path)
        except FileNotFoundError:
            return
        if not _same_leaf(current, expected):
            return
        if expected.digest is not None:
            try:
                content = path.read_bytes()
                after_read = _leaf_snapshot(path)
            except OSError:
                return
            if (
                not _same_leaf(after_read, expected)
                or len(content) != expected.size
                or hashlib.sha256(content).hexdigest() != expected.digest
            ):
                return
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = path.with_name(f"{path.name}.corrupt-{timestamp}")
        if os.path.lexists(target):
            target = path.with_name(
                f"{path.name}.corrupt-{timestamp}-{uuid.uuid4().hex[:8]}"
            )
        try:
            path.replace(target)
        except OSError:
            return

    def load(
        self,
        key: str,
        expected_request: ScanRequest,
    ) -> ScanResult | None:
        validated_key = _validated_key(key)
        if not isinstance(expected_request, ScanRequest):
            raise ValueError("expected_request must be a ScanRequest")
        path = self._path_for(validated_key, allow_leaf_symlink=True)
        if not os.path.lexists(path):
            return None
        with self._key_lock(validated_key):
            path = self._path_for(validated_key, allow_leaf_symlink=True)
            if not os.path.lexists(path):
                return None
            try:
                before_read = _leaf_snapshot(path)
            except FileNotFoundError:
                return None
            if stat.S_ISLNK(before_read.mode):
                self._quarantine(path, before_read)
                return None
            if not stat.S_ISREG(before_read.mode):
                self._quarantine(path, before_read)
                return None
            read_snapshot = before_read
            try:
                content = path.read_bytes()
                after_read = _leaf_snapshot(path)
                if (
                    not _same_leaf(before_read, after_read)
                    or len(content) != after_read.size
                ):
                    return None
                read_snapshot = replace(
                    after_read,
                    digest=hashlib.sha256(content).hexdigest(),
                )
                payload = json.loads(content.decode("utf-8"))
                mapping = _strict_mapping(payload, _PAYLOAD_FIELDS, field="cache")
                schema_version = mapping["schema_version"]
                if (
                    type(schema_version) is not int
                    or schema_version != CACHE_SCHEMA_VERSION
                ):
                    raise ValueError("cache.schema_version is invalid")
                if mapping["key"] != validated_key:
                    raise ValueError("cache.key does not match its path")
                cached_request = _request_from_payload(mapping["request"])
                if cached_request != expected_request:
                    raise ValueError("cache.request does not match expected_request")
                return _result_from_payload(mapping["result"], cached_request)
            except FileNotFoundError:
                return None
            except Exception:
                self._quarantine(path, read_snapshot)
                return None

    def store(
        self,
        key: str,
        request: ScanRequest,
        result: ScanResult,
    ) -> Path:
        validated_key = _validated_key(key)
        if not isinstance(request, ScanRequest):
            raise ValueError("request must be a ScanRequest")
        if not isinstance(result, ScanResult):
            raise ValueError("result must be a ScanResult")
        request_payload = asdict(request)
        validated_request = _request_from_payload(request_payload)
        result_payload = _result_payload(result)
        _result_from_payload(result_payload, validated_request)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "key": validated_key,
            "request": request_payload,
            "result": result_payload,
        }
        with self._key_lock(validated_key):
            path = self.path_for(validated_key)
            return atomic_write_json(path, payload)

    @staticmethod
    def _legacy_number(value: Any, *, field: str) -> float:
        return _finite_number(value, field=field)

    @staticmethod
    def _axis_index(axis: np.ndarray, value: float, *, field: str) -> int:
        serialized_axis = np.fromiter(
            (round(float(item), _LEGACY_COORDINATE_DECIMALS) for item in axis),
            dtype=float,
            count=len(axis),
        )
        matches = np.flatnonzero(
            np.isclose(serialized_axis, value, rtol=0.0, atol=1e-12),
        )
        if len(matches) != 1:
            raise ValueError(f"{field} does not map unambiguously to the scan grid")
        return int(matches[0])

    def import_legacy(
        self,
        legacy_path: str | Path,
        key: str,
        request: ScanRequest,
        boundary: Any,
        coordinates: Any,
        *,
        scenario_tag: str,
    ) -> ScanResult:
        validated_key = _validated_key(key)
        if not isinstance(request, ScanRequest):
            raise ValueError("request must be a ScanRequest")
        if request.mode != "fast":
            raise ValueError(
                "Legacy candidate caches lack exhaustive coverage metadata and "
                "cannot be imported for complete mode"
            )
        source = _assert_unredirected_regular_file(
            Path(legacy_path),
            description="Legacy candidate cache",
        )
        expected_filename = legacy_cache_filename(scenario_tag, request)
        if source.name != expected_filename:
            raise ValueError(
                "Legacy candidate cache filename does not match its effective "
                f"configuration: expected {expected_filename!r}, got {source.name!r}"
            )
        try:
            document = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Legacy candidate cache is unreadable: {source}") from exc
        if not isinstance(document, list):
            raise ValueError("Legacy candidate cache must be a JSON array")
        if len(document) > request.max_candidates:
            raise ValueError("Legacy candidate cache exceeds max_candidates")

        points = np.asarray(coordinates)
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("coordinates must be an N-by-at-least-2 array")
        if not np.issubdtype(points.dtype, np.number) or np.issubdtype(
            points.dtype,
            np.complexfloating,
        ):
            raise ValueError("coordinates must contain real numeric values")
        if not bool(np.isfinite(points[:, :2]).all()):
            raise ValueError("coordinates must contain finite real values")

        x_origins, y_origins = grid_axes(
            boundary,
            request.rectangle_size,
            request.step,
        )
        candidates: list[Candidate] = []
        legacy_results: list[dict[str, Any]] = []
        for index, item in enumerate(document):
            prefix = f"legacy[{index}]"
            mapping = _strict_mapping(
                item,
                frozenset(
                    {"left_x", "bottom_y", "center_x", "center_y", "pt_count"}
                ),
                field=prefix,
            )
            left_x = self._legacy_number(mapping["left_x"], field=f"{prefix}.left_x")
            bottom_y = self._legacy_number(
                mapping["bottom_y"],
                field=f"{prefix}.bottom_y",
            )
            center_x = self._legacy_number(
                mapping["center_x"],
                field=f"{prefix}.center_x",
            )
            center_y = self._legacy_number(
                mapping["center_y"],
                field=f"{prefix}.center_y",
            )
            point_count = _strict_integer(
                mapping["pt_count"],
                field=f"{prefix}.pt_count",
                minimum=0,
            )
            original_x_index = self._axis_index(
                x_origins,
                left_x,
                field=f"{prefix}.left_x",
            )
            original_y_index = self._axis_index(
                y_origins,
                bottom_y,
                field=f"{prefix}.bottom_y",
            )
            original_left_x = float(x_origins[original_x_index])
            original_bottom_y = float(y_origins[original_y_index])
            original_center_x = original_left_x + request.rectangle_size / 2.0
            original_center_y = original_bottom_y + request.rectangle_size / 2.0
            if not math.isclose(
                center_x,
                round(original_center_x, _LEGACY_COORDINATE_DECIMALS),
                rel_tol=0.0,
                abs_tol=1e-12,
            ) or not math.isclose(
                center_y,
                round(original_center_y, _LEGACY_COORDINATE_DECIMALS),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{prefix} center is inconsistent")
            geometry = box(
                original_left_x,
                original_bottom_y,
                original_left_x + request.rectangle_size,
                original_bottom_y + request.rectangle_size,
            )
            if not bool(shapely.contains(boundary, geometry)) or bool(
                shapely.intersects(boundary.boundary, geometry)
            ):
                raise ValueError(f"{prefix} rectangle is not strictly inside boundary")
            flat_grid_id = original_y_index * len(x_origins) + original_x_index
            candidates.append(
                Candidate(
                    flat_grid_id=flat_grid_id,
                    point_count=point_count,
                    left_x=original_left_x,
                    bottom_y=original_bottom_y,
                    center_x=original_center_x,
                    center_y=original_center_y,
                )
            )
            legacy_results.append(
                {
                    "left_x": original_left_x,
                    "bottom_y": original_bottom_y,
                    "pt_count": point_count,
                }
            )

        mismatches = validate_results(
            legacy_results,
            points,
            request.rectangle_size,
        )
        if mismatches:
            raise ValueError(f"Legacy candidate point counts are invalid: {mismatches}")
        total_positions = int(len(x_origins) * len(y_origins))
        result = ScanResult(
            candidates=tuple(candidates),
            checked_positions=total_positions,
            total_positions=total_positions,
            completed=True,
            algorithm_version=request.algorithm_version,
        )
        _result_from_payload(_result_payload(result), request)
        self.store(validated_key, request, result)
        return result
