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
from types import MappingProxyType
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


def _freeze_record(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_record(item) for key, item in deepcopy(dict(value)).items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_record(item) for item in deepcopy(list(value)))
    return deepcopy(value)


def _thaw_record(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_record(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_record(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True)
class RunEntry:
    """One live, path-validated run and its immutable manifest snapshot."""

    root: Path
    run_dir: Path
    record: Mapping[str, Any]

    def __post_init__(self) -> None:
        raw_root = Path(self.root)
        raw_run_dir = Path(self.run_dir)
        if not raw_root.is_absolute() or not raw_run_dir.is_absolute():
            raise ValueError("run entry paths must be absolute")
        if raw_root.is_symlink() or raw_run_dir.is_symlink():
            raise ValueError("run entry paths must not use symlinks")
        root = raw_root.resolve(strict=True)
        run_dir = raw_run_dir.resolve(strict=True)
        if not root.is_dir() or not run_dir.is_dir():
            raise ValueError("run entry paths must be real directories")
        try:
            lexical_relative = raw_run_dir.relative_to(raw_root)
            run_dir.relative_to(root)
        except ValueError as exc:
            raise ValueError("run entry directory must remain inside its output root") from exc
        current = raw_root
        for part in lexical_relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError("run entry directory must not use symlinks")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "run_dir", run_dir)
        object.__setattr__(self, "record", _freeze_record(self.record))

    @property
    def run_id(self) -> str:
        return str(self.record["run_id"])


@dataclass(frozen=True)
class RunEntryDiscovery:
    entries: tuple[RunEntry, ...]
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

    def discover_entries(self) -> RunEntryDiscovery:
        """Discover live runs together with their validated final directories."""

        if not self.output_root.is_dir():
            return RunEntryDiscovery(entries=(), diagnostics=())

        entries: list[tuple[datetime, RunEntry]] = []
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
            entries.append(
                (
                    parsed_created_at,
                    RunEntry(
                        root=self.output_root,
                        run_dir=record_path.parent,
                        record=record,
                    ),
                )
            )

        entries.sort(key=lambda item: (item[0], item[1].run_id))
        diagnostics.sort(key=lambda item: item["path"])
        return RunEntryDiscovery(
            entries=tuple(entry for _, entry in entries),
            diagnostics=tuple(diagnostics),
        )

    def discover(self) -> RunDiscovery:
        """Return backward-compatible manifest dictionaries for valid live runs."""

        discovered = self.discover_entries()
        return RunDiscovery(
            records=tuple(_thaw_record(entry.record) for entry in discovered.entries),
            diagnostics=discovered.diagnostics,
        )

    def entry_for_path(self, path: str | os.PathLike[str]) -> RunEntry:
        """Re-discover and resolve exactly one live run directory by path."""

        if not isinstance(path, (str, os.PathLike)):
            raise ValueError("run path must be path-like")
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            raise ValueError("run path must be absolute")
        if candidate.is_symlink():
            raise ValueError("run path must not be a symlink")
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.output_root)
        except ValueError as exc:
            raise ValueError("run path is outside this service output root") from exc
        matches = [
            entry
            for entry in self.discover_entries().entries
            if entry.run_dir == resolved
        ]
        if len(matches) != 1:
            raise ValueError("run path is not one currently available valid run")
        return matches[0]

    @staticmethod
    def _compatibility_payload(
        entry: RunEntry,
        exact_directory: Path,
        compatibility_record: str,
    ) -> dict[str, Any]:
        """Build the former operation-record shape from a modern run manifest."""

        record = _thaw_record(entry.record)
        metadata_value = record.get("metadata", {})
        metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
        entrypoint = metadata.get("entrypoint", [])
        command = (
            list(entrypoint)
            if isinstance(entrypoint, (list, tuple))
            and all(type(item) is str for item in entrypoint)
            else []
        )
        raw_inputs = metadata.get("inputs", {})
        inputs = []
        if isinstance(raw_inputs, Mapping):
            for name, value in sorted(raw_inputs.items(), key=lambda item: str(item[0])):
                item = {"name": str(name)}
                if isinstance(value, Mapping):
                    item.update(_thaw_record(value))
                inputs.append(item)
        parameters = metadata.get("parameters", {})
        config: dict[str, Any] = {
            "schema_version": 2,
            "scenario_id": record["scenario_id"],
            "profile_id": record["profile_id"],
            "parameters": (
                _thaw_record(parameters) if isinstance(parameters, Mapping) else {}
            ),
        }
        if (entry.run_dir / "run-config.yaml").is_file():
            config["run_config"] = str(exact_directory / "run-config.yaml")
        for field in (
            "target_crs",
            "rectangle_size_m",
            "figure_spec",
            "source",
            "candidate",
        ):
            if field in metadata:
                config[field] = _thaw_record(metadata[field])
        software = metadata.get("software_versions", {})
        artifacts = list(record.get("artifacts", []))
        if compatibility_record.casefold() == "run-generate-figures.json":
            artifacts = [
                relative
                for relative in artifacts
                if Path(relative).name.casefold() != "source.csv"
            ]
        return {
            "timestamp": record["created_at"],
            "command": command,
            "git_commit": metadata.get("git_commit"),
            "config": config,
            "inputs": inputs,
            "software": (
                _thaw_record(software) if isinstance(software, Mapping) else {}
            ),
            "outputs": [str(exact_directory / relative) for relative in artifacts],
            "run_id": record["run_id"],
            "status": record["status"],
            "source_record": "run.json",
        }

    @staticmethod
    def _exact_source_files(run_directory: Path) -> tuple[Path, ...]:
        files: list[Path] = []
        for source in sorted(run_directory.iterdir(), key=lambda path: path.name):
            if source.is_symlink() or not source.is_file():
                raise ValueError(
                    "exact-directory publication supports regular run files only: "
                    f"{source}"
                )
            files.append(source)
        if not any(path.name.casefold() == "run.json" for path in files):
            raise ValueError("exact-directory publication requires run.json")
        return tuple(files)

    @staticmethod
    def _copy_file_without_replacement(source: Path, destination: Path) -> None:
        """Fallback publication for writable filesystems without hard links."""

        descriptor: int | None = None
        created = False
        destination_identity: tuple[int, int] | None = None
        try:
            mode = source.stat().st_mode & 0o777
            descriptor = os.open(
                destination,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                mode,
            )
            created = True
            status = os.fstat(descriptor)
            destination_identity = (status.st_dev, status.st_ino)
            with source.open("rb") as input_stream, os.fdopen(
                descriptor,
                "wb",
                closefd=True,
            ) as output_stream:
                descriptor = None
                shutil.copyfileobj(input_stream, output_stream)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            current = destination.stat()
            if (current.st_dev, current.st_ino) != destination_identity:
                raise OSError(
                    f"exact output changed during fallback publication: {destination}"
                )
            source.unlink()
            created = False
        except FileExistsError as exc:
            raise FileExistsError(
                f"exact output conflict: {destination.name}"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if created:
                try:
                    current = destination.stat()
                except FileNotFoundError:
                    pass
                else:
                    if (current.st_dev, current.st_ino) == destination_identity:
                        destination.unlink()

    @staticmethod
    def _move_file_without_replacement(source: Path, destination: Path) -> None:
        """Move a same-filesystem regular file without an overwrite race."""

        try:
            os.link(source, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"exact output conflict: {destination.name}"
            ) from exc
        except OSError as exc:
            fallback_errors = {
                errno.EACCES,
                errno.EINVAL,
                errno.ENOSYS,
                errno.EPERM,
                errno.EXDEV,
            }
            for name in ("ENOTSUP", "EOPNOTSUPP"):
                value = getattr(errno, name, None)
                if value is not None:
                    fallback_errors.add(value)
            if exc.errno not in fallback_errors:
                raise
            RunService._copy_file_without_replacement(source, destination)
            return
        try:
            source.unlink()
        except BaseException as exc:
            try:
                destination.unlink()
            except OSError as cleanup_error:
                exc.add_note(
                    "Exact-directory link cleanup also failed: "
                    f"{destination}: {cleanup_error}"
                )
            raise

    def relocate_to_exact_directory(
        self,
        run_path: str | os.PathLike[str],
        exact_directory: str | os.PathLike[str],
        *,
        compatibility_record: str | None = None,
    ) -> Path:
        """Merge one just-published run into an exact legacy output directory.

        All destination conflicts are rejected before the first move. Files are
        rolled back to the validated unique run if publication fails, and the
        authoritative ``run.json`` is always moved last.
        """

        raw_target = Path(exact_directory).expanduser()
        if not raw_target.is_absolute():
            raise ValueError("exact output directory must be absolute")
        if raw_target.is_symlink():
            raise ValueError("exact output directory must not be a symlink")
        target = raw_target.resolve(strict=False)
        if target != self.output_root:
            raise ValueError(
                "exact output directory must equal this RunService output root"
            )
        if not target.is_dir():
            raise ValueError("exact output directory must be an existing directory")
        entry = self.entry_for_path(run_path)
        source_directory = entry.run_dir
        if source_directory == target:
            raise ValueError("published run is already the exact output directory")

        compatibility_name: str | None = None
        if compatibility_record is not None:
            if type(compatibility_record) is not str:
                raise ValueError("compatibility record name must be text")
            relative = Path(compatibility_record)
            if (
                relative.is_absolute()
                or len(relative.parts) != 1
                or relative.name != compatibility_record
                or relative.suffix.casefold() != ".json"
                or relative.name.casefold() == "run.json"
            ):
                raise ValueError("compatibility record must be a safe JSON filename")
            compatibility_name = relative.name

        original_sources = self._exact_source_files(source_directory)
        original_names = tuple(path.name for path in original_sources)
        if compatibility_name is not None and compatibility_name in original_names:
            raise FileExistsError(
                f"compatibility record already exists in published run: "
                f"{compatibility_name}"
            )
        planned_names = (
            *original_names,
            *((compatibility_name,) if compatibility_name else ()),
        )

        def conflicts() -> tuple[Path, ...]:
            values = []
            for name in planned_names:
                destination = target / name
                self._assert_contained(
                    destination,
                    description="exact publication destination",
                )
                if os.path.lexists(destination):
                    values.append(destination)
            return tuple(values)

        initial_conflicts = conflicts()
        if initial_conflicts:
            names = ", ".join(path.name for path in initial_conflicts)
            raise FileExistsError(f"exact output conflict: {names}")

        generated_compatibility: Path | None = None
        moved: list[tuple[Path, Path]] = []
        try:
            with self._publication_claim(target / ".exact-directory"):
                current_entry = self.entry_for_path(source_directory)
                if current_entry.record != entry.record:
                    raise ValueError("published run changed before exact relocation")
                current_sources = self._exact_source_files(source_directory)
                if tuple(path.name for path in current_sources) != original_names:
                    raise ValueError(
                        "published run files changed before exact relocation"
                    )
                claimed_conflicts = conflicts()
                if claimed_conflicts:
                    names = ", ".join(path.name for path in claimed_conflicts)
                    raise FileExistsError(f"exact output conflict: {names}")
                if compatibility_name is not None:
                    generated_compatibility = source_directory / compatibility_name
                    atomic_write_json(
                        generated_compatibility,
                        self._compatibility_payload(
                            entry,
                            target,
                            compatibility_name,
                        ),
                    )
                sources = list(self._exact_source_files(source_directory))
                sources.sort(
                    key=lambda path: (path.name.casefold() == "run.json", path.name)
                )
                for source in sources:
                    destination = target / source.name
                    self._move_file_without_replacement(source, destination)
                    moved.append((source, destination))
        except BaseException as exc:
            rollback_errors: list[str] = []
            for source, destination in reversed(moved):
                try:
                    self._move_file_without_replacement(destination, source)
                except OSError as rollback_error:
                    rollback_errors.append(f"{destination}: {rollback_error}")
            if (
                generated_compatibility is not None
                and generated_compatibility.exists()
            ):
                try:
                    generated_compatibility.unlink()
                except OSError as cleanup_error:
                    rollback_errors.append(
                        f"{generated_compatibility}: {cleanup_error}"
                    )
            if rollback_errors:
                exc.add_note(
                    "Exact-directory rollback also failed: "
                    + "; ".join(rollback_errors)
                )
            raise

        current = source_directory
        while current != target:
            parent = current.parent
            try:
                current.rmdir()
            except OSError:
                break
            current = parent
        return target


__all__ = [
    "RunDiscovery",
    "RunEntry",
    "RunEntryDiscovery",
    "RunService",
    "StagingRun",
]
