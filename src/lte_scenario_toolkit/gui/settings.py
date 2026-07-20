"""Validated local workstation settings for the browser interface."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from .i18n import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES

SETTINGS_RELATIVE_PATH = Path(".lte-data") / "gui-settings.json"
_SETTINGS_WRITE_LOCK = Lock()


class GuiSettingsError(ValueError):
    """Raised when local GUI settings are unsafe or malformed."""


def _is_link_or_junction(path: Path) -> bool:
    """Return whether a storage component redirects to another location."""

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
    """Return the first redirected component between an absolute root and path."""

    if _is_link_or_junction(root):
        return root
    try:
        parts = candidate.relative_to(root).parts
    except ValueError:
        return None
    current = root
    for part in parts:
        current = current / part
        if os.path.lexists(current) and _is_link_or_junction(current):
            return current
    return None


@dataclass(frozen=True, slots=True)
class GuiSettings:
    """Normalized workstation preferences."""

    language: str = DEFAULT_LANGUAGE
    output_roots: tuple[Path, ...] = ()
    navigation_collapsed: bool = False


class GuiSettingsStore:
    """Read and atomically replace ignored repository-local GUI settings."""

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.path = self.repo_root / SETTINGS_RELATIVE_PATH

    def load(self) -> GuiSettings:
        """Load current settings or return defaults for stale local content."""

        self._validate_storage_path()
        if not self.path.exists():
            return GuiSettings()
        if not self.path.is_file():
            raise GuiSettingsError(f"GUI settings path is not a file: {self.path}")
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            return self._settings_from_document(document)
        except (OSError, UnicodeError, json.JSONDecodeError, GuiSettingsError):
            return GuiSettings()

    def save(
        self,
        *,
        language: str,
        output_roots: Iterable[str | os.PathLike[str]],
        navigation_collapsed: bool | None = None,
    ) -> GuiSettings:
        """Validate, normalize, and atomically persist workstation settings."""

        with _SETTINGS_WRITE_LOCK:
            current = self.load()
            settings = self._validated_settings(
                language,
                output_roots,
                (
                    current.navigation_collapsed
                    if navigation_collapsed is None
                    else navigation_collapsed
                ),
            )
            self._persist_locked(settings)
        return settings

    def update(
        self,
        *,
        language: str | None = None,
        add_output_roots: Iterable[str | os.PathLike[str]] = (),
        navigation_collapsed: bool | None = None,
    ) -> GuiSettings:
        """Atomically merge one preference change with the latest file contents."""

        if isinstance(add_output_roots, (str, bytes, os.PathLike)):
            raise GuiSettingsError("GUI output_roots must be a path collection")
        try:
            additions = tuple(add_output_roots)
        except TypeError as exc:
            raise GuiSettingsError("GUI output_roots must be a path collection") from exc
        with _SETTINGS_WRITE_LOCK:
            current = self.load()
            settings = self._validated_settings(
                current.language if language is None else language,
                (*current.output_roots, *additions),
                (
                    current.navigation_collapsed
                    if navigation_collapsed is None
                    else navigation_collapsed
                ),
            )
            self._persist_locked(settings)
        return settings

    def _persist_locked(self, settings: GuiSettings) -> None:
        self._validate_storage_path()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise GuiSettingsError(
                f"Could not create GUI settings directory: {self.path.parent}"
            ) from exc
        self._validate_storage_path()
        payload = {
            "language": settings.language,
            "output_roots": [str(path) for path in settings.output_roots],
            "navigation_collapsed": settings.navigation_collapsed,
        }
        self._atomic_write(payload)

    def _settings_from_document(self, document: object) -> GuiSettings:
        if not isinstance(document, dict):
            raise GuiSettingsError("GUI settings must be a JSON object")
        legacy_expected = {"language", "output_roots"}
        expected = {*legacy_expected, "navigation_collapsed"}
        keys = set(document)
        if keys == legacy_expected:
            navigation_collapsed = False
        elif keys == expected:
            navigation_collapsed = document["navigation_collapsed"]
        else:
            raise GuiSettingsError(
                "GUI settings must contain language, output_roots, and navigation_collapsed"
            )
        roots = document["output_roots"]
        if not isinstance(roots, list):
            raise GuiSettingsError("GUI output_roots must be a JSON array")
        if any(not isinstance(root, str) or not Path(root).is_absolute() for root in roots):
            raise GuiSettingsError("GUI output roots must be absolute paths")
        settings = self._validated_settings(
            document["language"],
            roots,
            navigation_collapsed,
        )
        return settings

    def _validated_settings(
        self,
        language: object,
        output_roots: Iterable[object],
        navigation_collapsed: object,
    ) -> GuiSettings:
        if not isinstance(language, str) or language not in SUPPORTED_LANGUAGES:
            raise GuiSettingsError(f"Unsupported GUI language: {language!r}")
        if type(navigation_collapsed) is not bool:
            raise GuiSettingsError(
                "GUI navigation_collapsed must be a boolean"
            )
        if isinstance(output_roots, (str, bytes)):
            raise GuiSettingsError("GUI output_roots must be a path collection")
        try:
            values = list(output_roots)
        except TypeError as exc:
            raise GuiSettingsError("GUI output_roots must be a path collection") from exc

        normalized: list[Path] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, (str, os.PathLike)):
                raise GuiSettingsError(f"Invalid GUI output root: {value!r}")
            try:
                requested = Path(value).expanduser()
                if ".." in requested.parts:
                    raise GuiSettingsError(
                        f"GUI output root must not contain traversal: {value!r}"
                    )
                if not requested.is_absolute():
                    requested = self.repo_root / requested
                lexical = requested.absolute()
                anchor = Path(lexical.anchor)
                redirected = _redirected_component(anchor, lexical)
                if redirected is not None:
                    raise GuiSettingsError(
                        "GUI output root must not use a symlink or junction: "
                        f"{redirected}"
                    )
                path = lexical.resolve(strict=False)
            except GuiSettingsError:
                raise
            except (OSError, RuntimeError, ValueError) as exc:
                raise GuiSettingsError(f"Invalid GUI output root: {value!r}") from exc
            if path.exists() and not path.is_dir():
                raise GuiSettingsError(f"GUI output root is not a directory: {path}")
            identity = os.path.normcase(str(path))
            if identity not in seen:
                seen.add(identity)
                normalized.append(path)
        return GuiSettings(
            language=language,
            output_roots=tuple(normalized),
            navigation_collapsed=navigation_collapsed,
        )

    def _validate_storage_path(self) -> None:
        settings_dir = self.path.parent
        if os.path.lexists(settings_dir):
            if _is_link_or_junction(settings_dir):
                raise GuiSettingsError(
                    "GUI settings directory must not be a symlink or junction: "
                    f"{settings_dir}"
                )
            if not settings_dir.is_dir():
                raise GuiSettingsError(
                    f"GUI settings directory is not a directory: {settings_dir}"
                )
        if os.path.lexists(self.path) and _is_link_or_junction(self.path):
            raise GuiSettingsError(
                f"GUI settings file must not be a symlink or junction: {self.path}"
            )
        try:
            self.path.resolve(strict=False).relative_to(self.repo_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise GuiSettingsError(
                f"GUI settings path escapes repository root: {self.path}"
            ) from exc

    def _atomic_write(self, payload: dict[str, object]) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
        except OSError as exc:
            raise GuiSettingsError(f"Could not write GUI settings: {self.path}") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
