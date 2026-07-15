"""Boundary-source staging and normalization."""

from __future__ import annotations

import hashlib
import re
import shutil
import stat
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import unquote, urlsplit, urlunsplit

import geopandas as gpd

_VECTOR_SUFFIXES = frozenset({".shp", ".geojson", ".json", ".gpkg"})
_SHAPEFILE_REQUIRED_SUFFIXES = (".shp", ".shx", ".dbf", ".prj")
_SHAPEFILE_OPTIONAL_SUFFIXES = (".cpg",)
_NORMALIZED_CRS = "EPSG:3857"
_POLYGON_TYPES = frozenset({"Polygon", "MultiPolygon"})
REMOTE_TIMEOUT_SECONDS = 30.0
MAX_REMOTE_BYTES = 512 * 1024 * 1024
MAX_ZIP_MEMBERS = 4096
MAX_ZIP_MEMBER_BYTES = 256 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 1024 * 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 64 * 1024
_SCENARIO_ID_PATTERN = re.compile(r"[a-z][a-z0-9-]*\Z")
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)


class BoundaryImportError(ValueError):
    """Raised when a boundary source cannot be safely imported."""


class _BoundaryDownloadLimitError(BoundaryImportError):
    """Internal marker preventing retries after a remote size limit is hit."""


@dataclass(frozen=True)
class BoundaryArtifact:
    """A normalized boundary and the provenance of its staged source."""

    directory: Path
    entrypoint: Path
    source_url: str | None
    source_sha256: str
    crs: str
    geometry_type: str
    feature_count: int


@dataclass(frozen=True)
class _StagedSource:
    primary: Path
    files: tuple[Path, ...]


@dataclass
class _DownloadBudget:
    consumed: int = 0


@dataclass(frozen=True)
class _VectorLayer:
    path: Path
    layer: str | None
    root: Path

    @property
    def display_name(self) -> str:
        relative = self.path.relative_to(self.root).as_posix()
        return f"{relative}:{self.layer}" if self.layer is not None else relative

    def aliases(self) -> set[str]:
        relative = self.path.relative_to(self.root).as_posix()
        aliases = {
            self.path.name,
            self.path.stem,
            relative,
            str(PurePosixPath(relative).with_suffix("")),
        }
        if self.layer is not None:
            aliases.add(self.layer)
            aliases.add(f"{self.path.stem}:{self.layer}")
            aliases.add(f"{relative}:{self.layer}")
        return aliases


def _safe_zip_member_path(
    name: str,
    destination: Path,
    *,
    seen_names: set[str] | None = None,
) -> Path:
    normalized = name.replace("\\", "/")
    windows_name = PureWindowsPath(normalized)
    if (
        not normalized
        or "\x00" in normalized
        or normalized.startswith("/")
        or normalized.startswith("\\")
        or windows_name.is_absolute()
        or bool(windows_name.drive)
    ):
        raise BoundaryImportError(f"Unsafe ZIP member path: {name!r}")

    raw_parts = normalized.split("/")
    if raw_parts and raw_parts[-1] == "":
        raw_parts.pop()
    if any(not part or part in {".", ".."} for part in raw_parts):
        raise BoundaryImportError(f"Unsafe ZIP member path traversal: {name!r}")
    if any(
        ":" in part
        or part.endswith((".", " "))
        or part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES
        for part in raw_parts
    ):
        raise BoundaryImportError(f"Windows-unsafe ZIP member path: {name!r}")

    canonical_name = "/".join(part.casefold() for part in raw_parts)
    if seen_names is not None:
        if canonical_name in seen_names:
            raise BoundaryImportError(f"duplicate ZIP member path: {name!r}")
        seen_names.add(canonical_name)

    parts = PurePosixPath(normalized).parts
    target = destination.joinpath(*parts)
    try:
        target.resolve().relative_to(destination.resolve())
    except ValueError as exc:
        raise BoundaryImportError(f"Unsafe ZIP member path: {name!r}") from exc
    return target


def safe_extract_zip(archive: str | Path, destination: str | Path) -> Path:
    """Extract a ZIP archive after rejecting traversal and symbolic links."""

    archive_path = Path(archive)
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as handle:
            infos = handle.infolist()
            if len(infos) > MAX_ZIP_MEMBERS:
                raise BoundaryImportError(
                    f"ZIP member count exceeds limit ({MAX_ZIP_MEMBERS})"
                )
            members: list[tuple[zipfile.ZipInfo, Path]] = []
            seen_names: set[str] = set()
            declared_total = 0
            for info in infos:
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise BoundaryImportError(
                        f"ZIP symlink entries are not allowed: {info.filename!r}"
                    )
                if info.file_size > MAX_ZIP_MEMBER_BYTES:
                    raise BoundaryImportError(
                        f"ZIP member expanded size exceeds limit ({MAX_ZIP_MEMBER_BYTES}): "
                        f"{info.filename!r}"
                    )
                declared_total += info.file_size
                if declared_total > MAX_ZIP_TOTAL_BYTES:
                    raise BoundaryImportError(
                        f"ZIP aggregate expanded size exceeds limit ({MAX_ZIP_TOTAL_BYTES})"
                    )
                members.append(
                    (
                        info,
                        _safe_zip_member_path(
                            info.filename,
                            destination_path,
                            seen_names=seen_names,
                        ),
                    )
                )

            written_total = 0
            for info, target in members:
                if info.is_dir() or info.filename.endswith(("/", "\\")):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                written_member = 0
                with handle.open(info) as source, target.open("wb") as output:
                    while True:
                        chunk = source.read(_DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        written_member += len(chunk)
                        written_total += len(chunk)
                        if written_member > MAX_ZIP_MEMBER_BYTES:
                            raise BoundaryImportError(
                                f"ZIP member expanded size exceeds limit "
                                f"({MAX_ZIP_MEMBER_BYTES}): {info.filename!r}"
                            )
                        if written_total > MAX_ZIP_TOTAL_BYTES:
                            raise BoundaryImportError(
                                f"ZIP aggregate expanded size exceeds limit "
                                f"({MAX_ZIP_TOTAL_BYTES})"
                            )
                        output.write(chunk)
    except zipfile.BadZipFile as exc:
        raise BoundaryImportError(f"Invalid ZIP archive: {archive_path}") from exc
    return destination_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_source(files: tuple[Path, ...]) -> str:
    if len(files) == 1:
        return _sha256_file(files[0])

    digest = hashlib.sha256()
    digest.update(b"lte-scenario-boundary-bundle-v1\0")
    for path in sorted(files, key=lambda item: (item.suffix.casefold(), item.name.casefold())):
        label = path.name.encode("utf-8")
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _find_shapefile_component(source: Path, suffix: str) -> Path | None:
    matches = [
        path
        for path in source.parent.iterdir()
        if path.is_file()
        and path.stem.casefold() == source.stem.casefold()
        and path.suffix.casefold() == suffix
    ]
    if len(matches) > 1:
        raise BoundaryImportError(
            f"Ambiguous Shapefile component {suffix} for source: {source}"
        )
    return matches[0] if matches else None


def _stage_local_source(source: Path, source_dir: Path) -> _StagedSource:
    if not source.is_file():
        raise BoundaryImportError(f"Boundary source does not exist or is not a file: {source}")

    suffix = source.suffix.casefold()
    if suffix == ".shp":
        components: list[Path] = []
        missing: list[str] = []
        for component_suffix in _SHAPEFILE_REQUIRED_SUFFIXES:
            component = _find_shapefile_component(source, component_suffix)
            if component is None:
                missing.append(component_suffix)
                continue
            target = source_dir / f"{source.stem}{component_suffix}"
            shutil.copy2(component, target)
            components.append(target)
        if missing:
            detail = ", ".join(missing)
            if ".prj" in missing:
                detail += " (the .prj file declares the CRS)"
            raise BoundaryImportError(f"Missing required Shapefile component(s): {detail}")
        for component_suffix in _SHAPEFILE_OPTIONAL_SUFFIXES:
            component = _find_shapefile_component(source, component_suffix)
            if component is not None:
                target = source_dir / f"{source.stem}{component_suffix}"
                shutil.copy2(component, target)
                components.append(target)
        return _StagedSource(source_dir / f"{source.stem}.shp", tuple(components))

    if suffix not in _VECTOR_SUFFIXES and suffix != ".zip":
        raise BoundaryImportError(f"Unsupported boundary source type: {source.suffix or '<none>'}")
    target = source_dir / source.name
    shutil.copy2(source, target)
    return _StagedSource(target, (target,))


def _url_with_suffix(url: str, suffix: str) -> str:
    parts = urlsplit(url)
    path = parts.path
    dot = path.rfind(".")
    if dot < path.rfind("/"):
        raise BoundaryImportError(f"URL does not have a replaceable file suffix: {url}")
    return urlunsplit((parts.scheme, parts.netloc, f"{path[:dot]}{suffix}", parts.query, parts.fragment))


def _url_suffix_variants(url: str, suffix: str) -> list[str]:
    parts = urlsplit(url)
    path = parts.path
    dot = path.rfind(".")
    if dot < path.rfind("/"):
        raise BoundaryImportError(f"URL does not have a replaceable file suffix: {url}")
    source_suffix = path[dot:]
    if source_suffix.isupper():
        convention = suffix.upper()
    elif source_suffix.islower():
        convention = suffix.lower()
    else:
        convention = suffix.capitalize()
    variants: list[str] = []
    for candidate_suffix in (convention, suffix.lower(), suffix.upper()):
        candidate = _url_with_suffix(url, candidate_suffix)
        if candidate not in variants:
            variants.append(candidate)
    return variants


def _response_content_length(response) -> int | None:
    headers = getattr(response, "headers", None)
    value = headers.get("Content-Length") if headers is not None else None
    if value is None and hasattr(response, "getheader"):
        value = response.getheader("Content-Length")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _download_url(url: str, target: Path, *, budget: _DownloadBudget) -> Path:
    remaining = MAX_REMOTE_BYTES - budget.consumed
    if remaining <= 0:
        raise BoundaryImportError(
            f"Remote boundary download exceeds maximum aggregate size ({MAX_REMOTE_BYTES} bytes)"
        )
    try:
        with urllib.request.urlopen(url, timeout=REMOTE_TIMEOUT_SECONDS) as response:
            content_length = _response_content_length(response)
            if content_length is not None and content_length > remaining:
                raise _BoundaryDownloadLimitError(
                    f"Remote boundary download exceeds maximum size ({MAX_REMOTE_BYTES} bytes)"
                )
            downloaded = 0
            with target.open("wb") as output:
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    budget.consumed += len(chunk)
                    if downloaded > remaining:
                        raise _BoundaryDownloadLimitError(
                            f"Remote boundary download exceeds maximum size "
                            f"({MAX_REMOTE_BYTES} bytes)"
                        )
                    output.write(chunk)
    except BoundaryImportError:
        target.unlink(missing_ok=True)
        raise
    except (OSError, urllib.error.URLError) as exc:
        target.unlink(missing_ok=True)
        raise BoundaryImportError(f"Could not download boundary source: {url}") from exc
    return target


def _download_remote_component(
    source_url: str,
    suffix: str,
    target: Path,
    *,
    budget: _DownloadBudget,
) -> tuple[Path, str]:
    errors: list[BoundaryImportError] = []
    for component_url in _url_suffix_variants(source_url, suffix):
        try:
            return _download_url(component_url, target, budget=budget), component_url
        except _BoundaryDownloadLimitError:
            raise
        except BoundaryImportError as exc:
            errors.append(exc)
    detail = "; ".join(str(error) for error in errors)
    raise BoundaryImportError(f"Could not download any matching sidecar URL for {suffix}: {detail}")


def _stage_remote_source(url: str, source_dir: Path) -> _StagedSource:
    parsed = urlsplit(url)
    source_name = Path(unquote(parsed.path)).name or "boundary-download"
    suffix = Path(source_name).suffix.casefold()
    budget = _DownloadBudget()
    primary_target = (
        source_dir / f"{Path(source_name).stem}.shp" if suffix == ".shp" else source_dir / source_name
    )
    primary = _download_url(url, primary_target, budget=budget)

    if zipfile.is_zipfile(primary):
        return _StagedSource(primary, (primary,))
    if suffix == ".shp":
        components = [primary]
        for component_suffix in _SHAPEFILE_REQUIRED_SUFFIXES[1:]:
            try:
                component, _ = _download_remote_component(
                    url,
                    component_suffix,
                    source_dir / f"{Path(source_name).stem}{component_suffix}",
                    budget=budget,
                )
                components.append(component)
            except _BoundaryDownloadLimitError:
                raise
            except BoundaryImportError as exc:
                detail = f"Missing required remote Shapefile component {component_suffix}"
                if component_suffix == ".prj":
                    detail += " declaring the CRS"
                raise BoundaryImportError(f"{detail}: {url}") from exc
        for component_suffix in _SHAPEFILE_OPTIONAL_SUFFIXES:
            try:
                component, _ = _download_remote_component(
                    url,
                    component_suffix,
                    source_dir / f"{Path(source_name).stem}{component_suffix}",
                    budget=budget,
                )
                components.append(component)
            except _BoundaryDownloadLimitError:
                raise
            except BoundaryImportError:
                pass
        return _StagedSource(primary, tuple(components))

    if suffix not in _VECTOR_SUFFIXES and suffix != ".zip":
        raise BoundaryImportError(
            f"Unsupported downloaded boundary source type: {suffix or '<none>'}"
        )
    if suffix == ".zip":
        raise BoundaryImportError(f"Downloaded file is not a valid ZIP archive: {url}")
    return _StagedSource(primary, (primary,))


def _discover_vector_layers(root: Path) -> list[_VectorLayer]:
    candidates = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() in _VECTOR_SUFFIXES
        ),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    layers: list[_VectorLayer] = []
    for path in candidates:
        if path.suffix.casefold() == ".json" and not _is_usable_json_vector(path):
            continue
        if path.suffix.casefold() != ".gpkg":
            layers.append(_VectorLayer(path=path, layer=None, root=root))
            continue
        try:
            available = gpd.list_layers(path)
        except Exception as exc:
            raise BoundaryImportError(f"Could not inspect GeoPackage layers: {path}") from exc
        geometry_layers = available[available["geometry_type"].notna()]
        for layer_name in geometry_layers["name"].astype(str):
            layers.append(_VectorLayer(path=path, layer=layer_name, root=root))
    return layers


def _is_usable_json_vector(path: Path) -> bool:
    try:
        available = gpd.list_layers(path)
    except Exception:
        try:
            frame = gpd.read_file(path)
        except Exception:
            return False
        return _has_nonempty_geometry(frame)
    if available.empty or "geometry_type" not in available.columns:
        return False
    geometry_types = available["geometry_type"].dropna().astype(str)
    if geometry_types.empty or any(value.casefold() == "unknown" for value in geometry_types):
        return False
    try:
        frame = gpd.read_file(path)
    except Exception:
        return False
    return _has_nonempty_geometry(frame)


def _has_nonempty_geometry(frame: gpd.GeoDataFrame) -> bool:
    if not hasattr(frame, "geometry") or frame.empty:
        return False
    geometry = frame.geometry
    return bool((geometry.notna() & ~geometry.is_empty).any())


def _select_vector_layer(root: Path, requested: str | None) -> _VectorLayer:
    layers = _discover_vector_layers(root)
    if not layers:
        raise BoundaryImportError(f"No supported vector boundary found below: {root}")
    if requested is None:
        if len(layers) != 1:
            choices = ", ".join(layer.display_name for layer in layers)
            raise BoundaryImportError(
                f"Multiple vector layers found; specify a layer. Available layers: {choices}"
            )
        return layers[0]

    requested_folded = str(requested).casefold()
    matches = [
        layer
        for layer in layers
        if requested_folded in {alias.casefold() for alias in layer.aliases()}
    ]
    if len(matches) != 1:
        choices = ", ".join(layer.display_name for layer in layers)
        if not matches:
            raise BoundaryImportError(
                f"Unknown vector layer {requested!r}. Available layers: {choices}"
            )
        raise BoundaryImportError(
            f"Ambiguous vector layer {requested!r}. Matching layers: "
            + ", ".join(layer.display_name for layer in matches)
        )
    return matches[0]


def _read_and_validate_boundary(vector: _VectorLayer) -> gpd.GeoDataFrame:
    read_options = {"layer": vector.layer} if vector.layer is not None else {}
    try:
        frame = gpd.read_file(vector.path, **read_options)
    except Exception as exc:
        raise BoundaryImportError(f"Could not read vector boundary: {vector.display_name}") from exc

    if frame.crs is None:
        raise BoundaryImportError("Boundary source must declare a CRS")
    if frame.empty:
        raise BoundaryImportError("Boundary source is empty")
    geometry = frame.geometry
    if geometry.isna().any() or geometry.is_empty.any():
        raise BoundaryImportError("Boundary geometries must be nonempty")
    geometry_types = set(geometry.geom_type)
    if not geometry_types <= _POLYGON_TYPES:
        found = ", ".join(sorted(geometry_types))
        raise BoundaryImportError(
            f"Boundary geometries must be Polygon or MultiPolygon; found: {found}"
        )
    if not geometry.is_valid.all():
        raise BoundaryImportError("All boundary geometries must be valid")
    return frame


def _normalize_boundary(
    frame: gpd.GeoDataFrame,
    *,
    scenario_id: str,
    display_name: str,
) -> gpd.GeoDataFrame:
    try:
        projected = frame.to_crs(_NORMALIZED_CRS)
        dissolved = projected.geometry.union_all()
    except Exception as exc:
        raise BoundaryImportError(f"Could not reproject boundary to {_NORMALIZED_CRS}") from exc
    if dissolved is None or dissolved.is_empty:
        raise BoundaryImportError("Dissolved boundary geometry is empty")
    if dissolved.geom_type not in _POLYGON_TYPES:
        raise BoundaryImportError(
            f"Dissolved boundary must be Polygon or MultiPolygon; found: {dissolved.geom_type}"
        )
    if not dissolved.is_valid:
        raise BoundaryImportError("Dissolved boundary geometry is invalid")
    return gpd.GeoDataFrame(
        {"scenario": [scenario_id], "name": [display_name]},
        geometry=[dissolved],
        crs=_NORMALIZED_CRS,
    )


def _validate_scenario_id(scenario_id: str) -> str:
    if not isinstance(scenario_id, str) or _SCENARIO_ID_PATTERN.fullmatch(scenario_id) is None:
        raise BoundaryImportError("scenario_id must match [a-z][a-z0-9-]*")
    if scenario_id in _WINDOWS_RESERVED_NAMES:
        raise BoundaryImportError("scenario_id must not be a Windows device name")
    return scenario_id


def import_boundary_source(
    source: str | Path,
    *,
    scenario_id: str,
    display_name: str,
    staging_dir: str | Path,
    layer: str | None = None,
) -> BoundaryArtifact:
    """Stage, validate, dissolve, and normalize one polygon boundary source."""

    scenario = _validate_scenario_id(scenario_id)
    staging = Path(staging_dir)
    try:
        staging.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise BoundaryImportError(f"Staging directory already exists: {staging}") from exc
    except OSError as exc:
        raise BoundaryImportError(f"Could not create staging directory: {staging}") from exc

    source_dir = staging / "source"
    source_dir.mkdir()
    source_text = str(source)
    scheme = urlsplit(source_text).scheme.casefold()
    source_url = source_text if scheme in {"http", "https"} else None
    if source_url is not None:
        staged = _stage_remote_source(source_url, source_dir)
    else:
        staged = _stage_local_source(Path(source), source_dir)

    vector_root = source_dir
    if staged.primary.suffix.casefold() == ".zip" or zipfile.is_zipfile(staged.primary):
        vector_root = safe_extract_zip(staged.primary, staging / "extracted")
    selected = _select_vector_layer(vector_root, layer)
    source_frame = _read_and_validate_boundary(selected)
    normalized = _normalize_boundary(
        source_frame,
        scenario_id=scenario,
        display_name=str(display_name),
    )

    normalized_dir = staging / "normalized"
    normalized_dir.mkdir()
    entrypoint = normalized_dir / f"{scenario}.shp"
    try:
        normalized.to_file(
            entrypoint,
            driver="ESRI Shapefile",
            encoding="UTF-8",
            index=False,
        )
    except Exception as exc:
        raise BoundaryImportError(f"Could not write normalized Shapefile: {entrypoint}") from exc
    cpg = entrypoint.with_suffix(".cpg")
    if not cpg.exists():
        cpg.write_text("UTF-8", encoding="ascii")
    missing_sidecars = [
        suffix
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg")
        if not entrypoint.with_suffix(suffix).is_file()
    ]
    if missing_sidecars:
        raise BoundaryImportError(
            "Normalized Shapefile is missing sidecars: " + ", ".join(missing_sidecars)
        )

    output_geometry = normalized.geometry.iloc[0]
    return BoundaryArtifact(
        directory=normalized_dir,
        entrypoint=entrypoint,
        source_url=source_url,
        source_sha256=_sha256_source(staged.files),
        crs=_NORMALIZED_CRS,
        geometry_type=output_geometry.geom_type,
        feature_count=len(normalized),
    )
