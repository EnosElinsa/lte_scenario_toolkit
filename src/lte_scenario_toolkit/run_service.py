"""Atomic staging, publication, and discovery for experiment runs."""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import uuid
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import _json_safe, atomic_write_json

_SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
_RUN_RECORD_FIELDS = frozenset(
    {
        "run_id",
        "scenario_id",
        "profile_id",
        "created_at",
        "parent_run_id",
        "status",
        "artifacts",
        "metadata",
        "errors",
    }
)


@dataclass(frozen=True)
class StagingRun:
    run_id: str
    scenario_id: str
    profile_id: str
    created_at: str
    path: Path
    final_path: Path
    parent_run_id: str | None = None


@dataclass(frozen=True)
class RunDiscovery:
    records: tuple[dict[str, Any], ...]
    diagnostics: tuple[dict[str, Any], ...]


def _safe_slug(value: Any, *, field: str) -> str:
    if (
        type(value) is not str
        or _SLUG_PATTERN.fullmatch(value) is None
        or value in _WINDOWS_RESERVED
    ):
        raise ValueError(
            f"{field} must be a safe lowercase slug and not a Windows device name"
        )
    return value


def _safe_run_id(value: Any, *, field: str = "run_id") -> str:
    if type(value) is not str or _RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a 32-character lowercase hexadecimal run ID")
    return value


def _normalise_created_at(value: str | None) -> tuple[str, datetime]:
    if value is None:
        parsed = datetime.now(timezone.utc)
    else:
        if type(value) is not str or not value.strip() or "T" not in value:
            raise ValueError("created_at must be a timezone-aware ISO timestamp")
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise ValueError(
                "created_at must be a timezone-aware ISO timestamp"
            ) from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("created_at must include an explicit timezone")
    utc_value = parsed.astimezone(timezone.utc)
    canonical = utc_value.isoformat().replace("+00:00", "Z")
    return canonical, utc_value


def _timestamp_directory_prefix(created_at: datetime) -> str:
    return created_at.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")


class RunService:
    VALID_STATUSES = {"completed", "partial"}

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root).resolve()
        self._owned_runs: dict[str, StagingRun] = {}

    def _ensure_directory_component(self, path: Path, *, description: str) -> Path:
        self._assert_contained(path, description=description)
        if not os.path.lexists(path):
            try:
                path.mkdir()
            except FileExistsError:
                pass
        self._assert_contained(path, description=description)
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"{description} path must be a real directory: {path}")
        return path

    def _prepare_run_parent(self, scenario_id: str, profile_id: str) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._assert_contained(self.output_root, description="run output root")
        if self.output_root.is_symlink() or not self.output_root.is_dir():
            raise ValueError(
                f"run output root must be a real directory: {self.output_root}"
            )
        scenario_path = self._ensure_directory_component(
            self.output_root / scenario_id,
            description="run scenario parent",
        )
        return self._ensure_directory_component(
            scenario_path / profile_id,
            description="run profile parent",
        )

    def _expected_paths(
        self,
        *,
        scenario_id: str,
        profile_id: str,
        created_at: str,
        run_id: str,
    ) -> tuple[Path, Path]:
        _, parsed = _normalise_created_at(created_at)
        parent = self.output_root / scenario_id / profile_id
        final_path = parent / (
            f"{_timestamp_directory_prefix(parsed)}-{run_id[:8]}"
        )
        staging_path = parent / f".staging-{run_id}"
        return staging_path, final_path

    def _assert_contained(self, path: Path, *, description: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            raise ValueError(f"{description} path must be absolute: {candidate}")
        try:
            candidate.relative_to(self.output_root)
        except ValueError as exc:
            raise ValueError(
                f"{description} path is outside this service output root: {candidate}"
            ) from exc

        current = self.output_root
        relative = candidate.relative_to(self.output_root)
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"{description} path must not use symlinks: {candidate}")
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.output_root)
        except ValueError as exc:
            raise ValueError(
                f"{description} path escapes this service output root: {candidate}"
            ) from exc
        if resolved != candidate:
            raise ValueError(f"{description} path must not use redirected paths: {candidate}")
        return candidate

    def _validate_run(
        self,
        run: StagingRun,
        *,
        require_staging: bool,
    ) -> tuple[Path, Path]:
        if not isinstance(run, StagingRun):
            raise ValueError("run must be a StagingRun owned by this service")
        run_id = _safe_run_id(run.run_id)
        if self._owned_runs.get(run_id) is not run:
            raise ValueError("run staging is not owned by this service instance")
        scenario_id = _safe_slug(run.scenario_id, field="scenario_id")
        profile_id = _safe_slug(run.profile_id, field="profile_id")
        canonical_created_at, _ = _normalise_created_at(run.created_at)
        if canonical_created_at != run.created_at:
            raise ValueError("created_at must use canonical UTC Z form")
        if run.parent_run_id is not None:
            _safe_run_id(run.parent_run_id, field="parent_run_id")
        expected_staging, expected_final = self._expected_paths(
            scenario_id=scenario_id,
            profile_id=profile_id,
            created_at=canonical_created_at,
            run_id=run_id,
        )
        if Path(run.path) != expected_staging or Path(run.final_path) != expected_final:
            raise ValueError("run staging and final paths do not belong to this service")
        staging = self._assert_contained(
            expected_staging,
            description="run staging",
        )
        final_path = self._assert_contained(
            expected_final,
            description="run final",
        )
        if os.path.lexists(staging):
            if staging.is_symlink() or not staging.is_dir():
                raise ValueError(f"run staging path must be a real directory: {staging}")
        elif require_staging:
            raise FileNotFoundError(staging)
        return staging, final_path

    @staticmethod
    def _active_prefix_exists(parent: Path, prefix: str) -> bool:
        for candidate in parent.glob(".staging-*"):
            active_id = candidate.name.removeprefix(".staging-")
            if active_id[:8] == prefix:
                return True
        return False

    def begin(
        self,
        scenario_id: str,
        profile_id: str,
        created_at: str | None = None,
        parent_run_id: str | None = None,
    ) -> StagingRun:
        scenario = _safe_slug(scenario_id, field="scenario_id")
        profile = _safe_slug(profile_id, field="profile_id")
        canonical_created_at, parsed_created_at = _normalise_created_at(created_at)
        if parent_run_id is not None:
            _safe_run_id(parent_run_id, field="parent_run_id")

        parent = self._prepare_run_parent(scenario, profile)
        timestamp_prefix = _timestamp_directory_prefix(parsed_created_at)
        for _ in range(128):
            run_id = uuid.uuid4().hex
            _safe_run_id(run_id)
            staging = parent / f".staging-{run_id}"
            final_path = parent / f"{timestamp_prefix}-{run_id[:8]}"
            if (
                os.path.lexists(staging)
                or os.path.lexists(final_path)
                or self._active_prefix_exists(parent, run_id[:8])
            ):
                continue
            try:
                staging.mkdir()
            except FileExistsError:
                continue
            if os.path.lexists(final_path):
                try:
                    staging.rmdir()
                except OSError:
                    pass
                continue
            run = StagingRun(
                run_id=run_id,
                scenario_id=scenario,
                profile_id=profile,
                created_at=canonical_created_at,
                path=staging,
                final_path=final_path,
                parent_run_id=parent_run_id,
            )
            self._owned_runs[run_id] = run
            return run
        raise FileExistsError(
            f"Could not allocate a unique run directory below {parent}"
        )

    def _artifact_paths(
        self,
        staging: Path,
        artifacts: Iterable[str | Path],
    ) -> list[str]:
        if isinstance(artifacts, (str, bytes, os.PathLike)):
            raise ValueError("artifacts must be a collection of relative file paths")
        try:
            requested = list(artifacts)
        except TypeError as exc:
            raise ValueError("artifacts must be an iterable of relative file paths") from exc
        if not requested:
            raise ValueError("at least one artifact is required to publish a run")

        relative_artifacts: list[str] = []
        seen: set[Path] = set()
        for item in requested:
            if not isinstance(item, (str, os.PathLike)):
                raise ValueError(f"artifact path must be text or path-like: {item!r}")
            relative = Path(item)
            if (
                relative.is_absolute()
                or not relative.parts
                or relative == Path(".")
                or ".." in relative.parts
            ):
                raise ValueError(f"artifact must be a contained relative path: {item}")
            candidate = staging / relative
            self._assert_contained(candidate, description="artifact")
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"artifact must be an existing regular file: {item}")
            resolved = candidate.resolve(strict=True)
            try:
                canonical_relative = resolved.relative_to(staging).as_posix()
            except ValueError as exc:
                raise ValueError(f"artifact escapes run staging: {item}") from exc
            if canonical_relative.casefold() == "run.json":
                raise ValueError("run.json is reserved and cannot be an artifact")
            if resolved in seen:
                raise ValueError(f"duplicate artifact path: {item}")
            seen.add(resolved)
            relative_artifacts.append(canonical_relative)
        return relative_artifacts

    def publish(
        self,
        run: StagingRun,
        status: str,
        artifacts: Iterable[str | Path],
        metadata: Any = None,
        errors: Any = None,
    ) -> Path:
        if status not in self.VALID_STATUSES:
            choices = ", ".join(sorted(self.VALID_STATUSES))
            raise ValueError(f"status must be one of: {choices}")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        if errors is not None and not isinstance(errors, list):
            raise ValueError("errors must be a list")
        staging, final_path = self._validate_run(run, require_staging=True)
        if os.path.lexists(final_path):
            raise FileExistsError(final_path)
        relative_artifacts = self._artifact_paths(staging, artifacts)

        safe_metadata = _json_safe(
            deepcopy({} if metadata is None else dict(metadata))
        )
        safe_errors = _json_safe(deepcopy([] if errors is None else list(errors)))
        published_at, _ = _normalise_created_at(None)
        payload = {
            "run_id": run.run_id,
            "scenario_id": run.scenario_id,
            "profile_id": run.profile_id,
            "created_at": run.created_at,
            "parent_run_id": run.parent_run_id,
            "status": status,
            "artifacts": relative_artifacts,
            "metadata": safe_metadata,
            "errors": safe_errors,
            "published_at": published_at,
        }
        atomic_write_json(staging / "run.json", payload)

        self._validate_run(run, require_staging=True)
        self._artifact_paths(staging, relative_artifacts)
        with self._publication_claim(final_path):
            if os.path.lexists(final_path):
                raise FileExistsError(f"run final path already exists: {final_path}")
            try:
                staging.replace(final_path)
            except OSError as exc:
                if isinstance(exc, FileExistsError) or exc.errno in {
                    errno.EEXIST,
                    errno.ENOTEMPTY,
                }:
                    raise FileExistsError(
                        f"run final path already exists: {final_path}"
                    ) from exc
                raise
        return final_path

    @contextmanager
    def _publication_claim(self, final_path: Path):
        claim_path = final_path.parent / f".publish-{final_path.name}.lock"
        self._assert_contained(claim_path, description="publication claim")
        descriptor: int | None = None
        created = False
        operation_error: BaseException | None = None
        try:
            try:
                descriptor = os.open(
                    claim_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                created = True
            except OSError as exc:
                if isinstance(exc, FileExistsError) or exc.errno == errno.EEXIST:
                    raise FileExistsError(
                        f"publication claim already exists: {claim_path}"
                    ) from exc
                raise
            os.close(descriptor)
            descriptor = None
            yield claim_path
        except BaseException as exc:
            operation_error = exc
            raise
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as close_error:
                    if operation_error is None:
                        raise
                    operation_error.add_note(
                        f"Publication claim close also failed: {close_error}"
                    )
            if created:
                try:
                    claim_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as unlink_error:
                    if operation_error is None:
                        raise
                    operation_error.add_note(
                        f"Publication claim cleanup also failed: {unlink_error}"
                    )

    def abandon(self, run: StagingRun) -> None:
        staging, _ = self._validate_run(run, require_staging=False)
        if not os.path.lexists(staging):
            return
        self._assert_contained(staging, description="run staging")
        if staging.is_symlink() or not staging.is_dir():
            raise ValueError(f"run staging path must be a real directory: {staging}")
        shutil.rmtree(staging)

    def _validate_discovered_record(
        self,
        record_path: Path,
        document: Any,
    ) -> tuple[dict[str, Any], datetime]:
        if not isinstance(document, dict):
            raise ValueError("run record must be a JSON object")
        missing = sorted(_RUN_RECORD_FIELDS - document.keys())
        if missing:
            raise ValueError(f"run record is missing required fields: {', '.join(missing)}")

        run_id = _safe_run_id(document["run_id"])
        scenario_id = _safe_slug(document["scenario_id"], field="scenario_id")
        profile_id = _safe_slug(document["profile_id"], field="profile_id")
        created_at, parsed_created_at = _normalise_created_at(document["created_at"])
        parent_run_id = document["parent_run_id"]
        if parent_run_id is not None:
            _safe_run_id(parent_run_id, field="parent_run_id")
        status = document["status"]
        if status not in self.VALID_STATUSES:
            raise ValueError(f"invalid run status: {status!r}")
        if not isinstance(document["metadata"], Mapping):
            raise ValueError("run metadata must be a JSON object")
        if not isinstance(document["errors"], list):
            raise ValueError("run errors must be a JSON array")

        final_path = record_path.parent
        self._assert_contained(record_path, description="run record")
        if record_path.is_symlink() or not record_path.is_file():
            raise ValueError("run record must be a regular file")
        expected_staging, expected_final = self._expected_paths(
            scenario_id=scenario_id,
            profile_id=profile_id,
            created_at=created_at,
            run_id=run_id,
        )
        del expected_staging
        if final_path != expected_final:
            raise ValueError(
                f"run record path does not match its identifiers: {record_path}"
            )
        artifacts = self._artifact_paths(final_path, document["artifacts"])
        record = dict(document)
        record["created_at"] = created_at
        record["artifacts"] = artifacts
        return record, parsed_created_at

    def discover(self) -> RunDiscovery:
        if not self.output_root.is_dir():
            return RunDiscovery(records=(), diagnostics=())

        records: list[tuple[datetime, dict[str, Any]]] = []
        diagnostics: list[dict[str, Any]] = []
        candidates = sorted(
            self.output_root.glob("*/*/*/run.json"),
            key=lambda path: path.as_posix(),
        )
        for record_path in candidates:
            if record_path.parent.name.startswith(".staging-"):
                continue
            try:
                document = json.loads(record_path.read_text(encoding="utf-8"))
                record, parsed_created_at = self._validate_discovered_record(
                    record_path,
                    document,
                )
            except Exception as exc:
                diagnostics.append(
                    {
                        "path": str(record_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            records.append((parsed_created_at, record))

        records.sort(key=lambda item: (item[0], item[1]["run_id"]))
        diagnostics.sort(key=lambda item: item["path"])
        return RunDiscovery(
            records=tuple(record for _, record in records),
            diagnostics=tuple(diagnostics),
        )
