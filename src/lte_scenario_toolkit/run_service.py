"""Atomic staging, publication, and discovery for experiment runs."""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import stat
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
_TRASH_TRANSACTION_PATTERN = re.compile(r"[0-9a-f]{32}")
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
        "published_at",
    }
)


def _is_redirected_path(path: Any) -> bool:
    """Return whether a path leaf is a symlink, junction, or reparse point."""

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


def _absolute_lexical_path(value: str | os.PathLike[str] | Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _assert_unredirected_chain(path: Path, *, description: str) -> Path:
    """Validate lexical traversal and every component without following redirects."""

    candidate = _absolute_lexical_path(path)
    if ".." in candidate.parts:
        raise ValueError(f"{description} path must not contain traversal: {candidate}")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        if _is_redirected_path(current):
            raise ValueError(
                f"{description} path must not use redirected paths: {candidate}"
            )
    return candidate


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
        input_root = Path(self.root)
        input_run_dir = Path(self.run_dir)
        if not input_root.is_absolute() or not input_run_dir.is_absolute():
            raise ValueError("run entry paths must be absolute")
        raw_root = _assert_unredirected_chain(
            input_root,
            description="run entry root",
        )
        raw_run_dir = _assert_unredirected_chain(
            input_run_dir,
            description="run entry directory",
        )
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
            if _is_redirected_path(current):
                raise ValueError("run entry directory must not use redirected paths")
        if not isinstance(self.record, Mapping):
            raise ValueError("run entry record must be a mapping")
        artifacts = self.record.get("artifacts", ())
        if not isinstance(artifacts, (list, tuple)):
            raise ValueError("run entry artifacts must be a path collection")
        seen: set[Path] = set()
        for item in artifacts:
            if not isinstance(item, (str, os.PathLike)):
                raise ValueError("run entry artifact paths must be path-like")
            relative = Path(item)
            if (
                relative.is_absolute()
                or not relative.parts
                or relative == Path(".")
                or ".." in relative.parts
            ):
                raise ValueError("run entry artifacts must be contained relative paths")
            artifact = _assert_unredirected_chain(
                raw_run_dir / relative,
                description="run entry artifact",
            )
            try:
                resolved_artifact = artifact.resolve(strict=True)
                resolved_artifact.relative_to(run_dir)
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError("run entry artifact must remain inside its run") from exc
            if not resolved_artifact.is_file() or resolved_artifact in seen:
                raise ValueError("run entry artifacts must be unique regular files")
            seen.add(resolved_artifact)
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


def _safe_trash_transaction_id(value: Any) -> str:
    if (
        type(value) is not str
        or _TRASH_TRANSACTION_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(
            "trash transaction ID must be 32 lowercase hexadecimal characters"
        )
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
        lexical_root = _assert_unredirected_chain(
            _absolute_lexical_path(output_root),
            description="run output root",
        )
        resolved_root = lexical_root.resolve(strict=False)
        if resolved_root != lexical_root:
            raise ValueError("run output root must not use redirected paths or traversal")
        self.output_root = resolved_root
        self._owned_runs: dict[str, StagingRun] = {}

    def _ensure_directory_component(self, path: Path, *, description: str) -> Path:
        self._assert_contained(path, description=description)
        if not os.path.lexists(path):
            try:
                path.mkdir()
            except FileExistsError:
                pass
        self._assert_contained(path, description=description)
        if _is_redirected_path(path) or not path.is_dir():
            raise ValueError(f"{description} path must be a real directory: {path}")
        return path

    def _prepare_run_parent(self, scenario_id: str, profile_id: str) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._assert_contained(self.output_root, description="run output root")
        if _is_redirected_path(self.output_root) or not self.output_root.is_dir():
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

    def prepare_trash_transaction(self, transaction_id: str) -> Path:
        transaction_id = _safe_trash_transaction_id(transaction_id)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._assert_contained(self.output_root, description="run output root")
        if _is_redirected_path(self.output_root) or not self.output_root.is_dir():
            raise ValueError("run output root must be a real directory")
        trash_root = self._ensure_directory_component(
            self.output_root / ".trash",
            description="run trash root",
        )
        transaction = trash_root / transaction_id
        self._assert_contained(transaction, description="trash transaction")
        if not os.path.lexists(transaction):
            transaction.mkdir()
        if _is_redirected_path(transaction) or not transaction.is_dir():
            raise ValueError("trash transaction must be a real directory")
        return transaction

    def _existing_trash_transaction(self, transaction_id: Any) -> Path:
        validated_id = _safe_trash_transaction_id(transaction_id)
        trash_root = self.output_root / ".trash"
        self._assert_contained(trash_root, description="run trash root")
        if not os.path.lexists(trash_root):
            raise FileNotFoundError(f"run trash root does not exist: {trash_root}")
        if _is_redirected_path(trash_root) or not trash_root.is_dir():
            raise ValueError("run trash root must be a real directory")

        transaction = trash_root / validated_id
        self._assert_contained(transaction, description="trash transaction")
        if not os.path.lexists(transaction):
            raise FileNotFoundError(
                f"trash transaction does not exist: {transaction}"
            )
        if _is_redirected_path(transaction) or not transaction.is_dir():
            raise ValueError("trash transaction must be a real directory")
        if transaction.parent != trash_root or transaction.name != validated_id:
            raise ValueError("trash transaction path does not have the exact shape")
        if transaction.resolve(strict=True) != transaction:
            raise ValueError("trash transaction path must not be redirected")
        return transaction

    def _entry_snapshot_identity(
        self,
        entry: RunEntry,
    ) -> tuple[str, str, str, Path]:
        if type(entry) is not RunEntry:
            raise ValueError("run entry must be an immutable RunEntry snapshot")
        if Path(entry.root) != self.output_root:
            raise ValueError("run entry root does not belong to this service")
        if not isinstance(entry.record, Mapping):
            raise ValueError("run entry snapshot manifest must be a mapping")
        try:
            run_id = _safe_run_id(entry.record["run_id"])
            scenario_id = _safe_slug(
                entry.record["scenario_id"],
                field="scenario_id",
            )
            profile_id = _safe_slug(
                entry.record["profile_id"],
                field="profile_id",
            )
            created_at, _ = _normalise_created_at(entry.record["created_at"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("run entry snapshot manifest is invalid") from exc
        _, expected_path = self._expected_paths(
            scenario_id=scenario_id,
            profile_id=profile_id,
            created_at=created_at,
            run_id=run_id,
        )
        candidate = Path(entry.run_dir)
        self._assert_contained(candidate, description="run entry destination")
        if candidate != expected_path:
            raise ValueError(
                "run entry snapshot destination does not match its manifest"
            )
        return run_id, scenario_id, profile_id, expected_path

    def _live_entry_from_snapshot(self, entry: RunEntry) -> RunEntry:
        run_id, _, _, expected_path = self._entry_snapshot_identity(entry)
        current = self.entry_for_path(expected_path, run_id=run_id)
        if _thaw_record(current.record) != _thaw_record(entry.record):
            raise ValueError("live run manifest changed from the immutable snapshot")
        return current

    def _trash_payload_path(
        self,
        path: str | os.PathLike[str],
        entry: RunEntry,
        *,
        description: str,
        require_exists: bool,
    ) -> tuple[Path, Path]:
        if not isinstance(path, (str, os.PathLike)):
            raise ValueError(f"{description} path must be path-like")
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            raise ValueError(f"{description} path must be absolute")
        self._assert_contained(candidate, description=description)
        _, scenario_id, profile_id, expected_live_path = (
            self._entry_snapshot_identity(entry)
        )
        try:
            relative = candidate.relative_to(self.output_root)
        except ValueError as exc:
            raise ValueError(f"{description} path is outside this service root") from exc
        if len(relative.parts) != 6 or relative.parts[0] != ".trash":
            raise ValueError(f"{description} path has an invalid trash payload shape")
        transaction_id = _safe_trash_transaction_id(relative.parts[1])
        expected_relative = Path(
            ".trash",
            transaction_id,
            "runs",
            scenario_id,
            profile_id,
            expected_live_path.name,
        )
        if relative != expected_relative:
            raise ValueError(f"{description} path does not match the run identity")
        transaction = self._existing_trash_transaction(transaction_id)
        if candidate.parent.parent.parent.parent != transaction:
            raise ValueError(f"{description} path is outside the exact transaction")
        exists = os.path.lexists(candidate)
        if require_exists and not exists:
            raise FileNotFoundError(f"{description} path does not exist: {candidate}")
        if not require_exists and exists:
            raise FileExistsError(f"{description} already exists: {candidate}")
        if require_exists and (
            _is_redirected_path(candidate) or not candidate.is_dir()
        ):
            raise ValueError(f"{description} must be a real directory")
        return transaction, candidate

    def _assert_real_tree(self, root: Path, *, description: str) -> None:
        pending = [root]
        while pending:
            current = pending.pop()
            try:
                self._assert_contained(current, description=description)
                if _is_redirected_path(current):
                    raise ValueError(f"{description} contains a redirected path")
                details = current.lstat()
                if stat.S_ISDIR(details.st_mode):
                    pending.extend(current.iterdir())
                elif not stat.S_ISREG(details.st_mode):
                    raise ValueError(
                        f"{description} contains a non-regular filesystem entry"
                    )
            except OSError as exc:
                raise ValueError(f"{description} could not be safely validated") from exc

    def _prepare_trash_payload_parent(
        self,
        transaction: Path,
        *,
        scenario_id: str,
        profile_id: str,
    ) -> Path:
        runs = self._ensure_directory_component(
            transaction / "runs",
            description="trash runs parent",
        )
        scenario = self._ensure_directory_component(
            runs / scenario_id,
            description="trash scenario parent",
        )
        return self._ensure_directory_component(
            scenario / profile_id,
            description="trash profile parent",
        )

    def move_entry_to(
        self,
        entry: RunEntry,
        destination: str | os.PathLike[str],
    ) -> Path:
        current = self._live_entry_from_snapshot(entry)
        transaction, target = self._trash_payload_path(
            destination,
            current,
            description="trash destination",
            require_exists=False,
        )
        _, scenario_id, profile_id, _ = self._entry_snapshot_identity(current)
        self._assert_real_tree(current.run_dir, description="live run contents")
        parent = self._prepare_trash_payload_parent(
            transaction,
            scenario_id=scenario_id,
            profile_id=profile_id,
        )
        if target.parent != parent:
            raise ValueError("trash destination parent does not match the transaction")

        current = self._live_entry_from_snapshot(entry)
        self._assert_real_tree(current.run_dir, description="live run contents")
        self._existing_trash_transaction(transaction.name)
        self._assert_contained(target, description="trash destination")
        if os.path.lexists(target):
            raise FileExistsError(f"trash destination already exists: {target}")
        current.run_dir.replace(target)
        return target

    def _validated_trash_source_record(
        self,
        source: Path,
        entry: RunEntry,
        expected_live_path: Path,
    ) -> dict[str, Any]:
        self._assert_real_tree(source, description="trash source contents")
        record_path = source / "run.json"
        self._assert_contained(record_path, description="trash source manifest")
        if _is_redirected_path(record_path) or not record_path.is_file():
            raise ValueError("trash source manifest must be a regular file")
        try:
            document = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("trash source manifest is invalid") from exc
        record, _ = self._validate_discovered_record(
            record_path,
            document,
            expected_live_path=expected_live_path,
        )
        if record != _thaw_record(entry.record):
            raise ValueError("trash source manifest changed from the immutable snapshot")
        return record

    def restore_entry_from(
        self,
        source: str | os.PathLike[str],
        entry: RunEntry,
    ) -> Path:
        _, scenario_id, profile_id, destination = self._entry_snapshot_identity(entry)
        _, trash_source = self._trash_payload_path(
            source,
            entry,
            description="trash source",
            require_exists=True,
        )
        self._validated_trash_source_record(trash_source, entry, destination)
        if os.path.lexists(destination):
            raise FileExistsError(f"restore destination already exists: {destination}")

        self.output_root.mkdir(parents=True, exist_ok=True)
        self._assert_contained(self.output_root, description="run output root")
        scenario = self._ensure_directory_component(
            self.output_root / scenario_id,
            description="restore scenario parent",
        )
        profile = self._ensure_directory_component(
            scenario / profile_id,
            description="restore profile parent",
        )
        if destination.parent != profile:
            raise ValueError("restore destination parent does not match the run identity")

        self._validated_trash_source_record(trash_source, entry, destination)
        self._assert_contained(destination, description="restore destination")
        if os.path.lexists(destination):
            raise FileExistsError(f"restore destination already exists: {destination}")
        trash_source.replace(destination)
        return destination

    def remove_trash_transaction(self, transaction_id: str) -> None:
        validated_id = _safe_trash_transaction_id(transaction_id)
        transaction = self._existing_trash_transaction(validated_id)
        trash_root = self.output_root / ".trash"
        expected = trash_root / validated_id
        self._assert_contained(expected, description="trash transaction purge target")
        if (
            transaction != expected
            or transaction.parent != trash_root
            or transaction.name != validated_id
            or transaction in {self.output_root, trash_root}
        ):
            raise ValueError("trash transaction purge target is not exact")
        self._assert_real_tree(transaction, description="trash transaction purge target")
        transaction = self._existing_trash_transaction(validated_id)
        if transaction != expected:
            raise ValueError("trash transaction purge target changed")
        shutil.rmtree(transaction)

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
        _assert_unredirected_chain(self.output_root, description="run output root")
        _assert_unredirected_chain(candidate, description=description)
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
            if _is_redirected_path(current):
                raise ValueError(
                    f"{description} path must not use redirected paths: {candidate}"
                )
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
            if _is_redirected_path(staging) or not staging.is_dir():
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
            if _is_redirected_path(candidate) or not candidate.is_file():
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
        if _is_redirected_path(staging) or not staging.is_dir():
            raise ValueError(f"run staging path must be a real directory: {staging}")
        shutil.rmtree(staging)

    def _record_candidates(
        self,
    ) -> tuple[tuple[Path, ...], tuple[dict[str, Any], ...]]:
        """Walk only unredirected directories and collect supported record files."""

        records: list[Path] = []
        diagnostics: list[dict[str, Any]] = []
        pending = [self.output_root]
        while pending:
            directory = pending.pop()
            try:
                self._assert_contained(directory, description="run discovery directory")
                children = sorted(directory.iterdir(), key=lambda path: path.name)
            except Exception as exc:
                diagnostics.append(
                    {
                        "path": str(directory),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            for child in children:
                name = child.name.casefold()
                try:
                    redirected = _is_redirected_path(child)
                except OSError as exc:
                    diagnostics.append(
                        {
                            "path": str(child),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                if redirected:
                    diagnostics.append(
                        {
                            "path": str(child),
                            "error": "ValueError: run discovery path is redirected",
                        }
                    )
                    continue
                if child.is_dir() and child.name == ".trash":
                    continue
                if child.is_dir() and not child.name.startswith(".staging-"):
                    pending.append(child)
                    continue
                if name == "run.json":
                    try:
                        relative = child.relative_to(self.output_root)
                    except ValueError:
                        continue
                    if len(relative.parts) == 4:
                        records.append(child)
        records.sort(key=lambda path: path.as_posix())
        diagnostics.sort(key=lambda item: item["path"])
        return tuple(records), tuple(diagnostics)

    def _validate_discovered_record(
        self,
        record_path: Path,
        document: Any,
        *,
        expected_live_path: Path | None = None,
    ) -> tuple[dict[str, Any], datetime]:
        if not isinstance(document, dict):
            raise ValueError("run record must be a JSON object")
        missing = sorted(_RUN_RECORD_FIELDS - document.keys())
        if missing:
            raise ValueError(f"run record is missing required fields: {', '.join(missing)}")
        unexpected = sorted(set(document) - _RUN_RECORD_FIELDS)
        if unexpected:
            raise ValueError(
                "run record has unexpected fields: " + ", ".join(unexpected)
            )

        run_id = _safe_run_id(document["run_id"])
        scenario_id = _safe_slug(document["scenario_id"], field="scenario_id")
        profile_id = _safe_slug(document["profile_id"], field="profile_id")
        created_at, parsed_created_at = _normalise_created_at(document["created_at"])
        published_at, _ = _normalise_created_at(document["published_at"])
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

        actual_run_path = record_path.parent
        self._assert_contained(record_path, description="run record")
        if _is_redirected_path(record_path) or not record_path.is_file():
            raise ValueError("run record must be a regular file")
        expected_staging, expected_final = self._expected_paths(
            scenario_id=scenario_id,
            profile_id=profile_id,
            created_at=created_at,
            run_id=run_id,
        )
        del expected_staging
        required_live_path = (
            actual_run_path if expected_live_path is None else Path(expected_live_path)
        )
        self._assert_contained(required_live_path, description="expected live run")
        if required_live_path != expected_final:
            raise ValueError(
                f"run record path does not match its identifiers: {record_path}"
            )
        artifacts = self._artifact_paths(actual_run_path, document["artifacts"])
        record = dict(document)
        record["created_at"] = created_at
        record["published_at"] = published_at
        record["artifacts"] = artifacts
        return record, parsed_created_at

    def discover_entries(self) -> RunEntryDiscovery:
        """Discover live runs together with their validated final directories."""

        if not self.output_root.is_dir():
            return RunEntryDiscovery(entries=(), diagnostics=())

        entries: list[tuple[datetime, RunEntry]] = []
        candidates, traversal_diagnostics = self._record_candidates()
        diagnostics: list[dict[str, Any]] = list(traversal_diagnostics)
        for record_path in candidates:
            if record_path.parent.name.startswith(".staging-"):
                continue
            try:
                self._assert_contained(record_path, description="run record")
                if _is_redirected_path(record_path) or not record_path.is_file():
                    raise ValueError("run record must be a regular file")
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
        """Return manifest dictionaries for valid live runs."""

        discovered = self.discover_entries()
        return RunDiscovery(
            records=tuple(_thaw_record(entry.record) for entry in discovered.entries),
            diagnostics=discovered.diagnostics,
        )

    def entry_for_path(
        self,
        path: str | os.PathLike[str],
        *,
        run_id: str | None = None,
    ) -> RunEntry:
        """Re-discover one live run by path, optionally disambiguated by ID."""

        if not isinstance(path, (str, os.PathLike)):
            raise ValueError("run path must be path-like")
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            raise ValueError("run path must be absolute")
        self._assert_contained(candidate, description="run path")
        if _is_redirected_path(candidate):
            raise ValueError("run path must not be redirected")
        expected_run_id = None if run_id is None else _safe_run_id(run_id)
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.output_root)
        except ValueError as exc:
            raise ValueError("run path is outside this service output root") from exc
        matches = [
            entry
            for entry in self.discover_entries().entries
            if entry.run_dir == resolved
            and (expected_run_id is None or entry.run_id == expected_run_id)
        ]
        if not matches:
            raise ValueError("run path is not one currently available valid run")
        if len(matches) != 1:
            raise ValueError("run path is ambiguous; provide one unique run_id")
        return matches[0]

__all__ = [
    "RunDiscovery",
    "RunEntry",
    "RunEntryDiscovery",
    "RunService",
    "StagingRun",
]
