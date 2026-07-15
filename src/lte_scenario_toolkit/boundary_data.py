"""Boundary-source staging and normalization."""

from __future__ import annotations

import hashlib
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


class BoundaryImportError(ValueError):
    """Raised when a boundary source cannot be safely imported."""


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


def _safe_zip_member_path(name: str, destination: Path) -> Path:
    normalized = name.replace("\\", "/")
    windows_name = PureWindowsPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or normalized.startswith("\\")
        or windows_name.is_absolute()
        or bool(windows_name.drive)
    ):
        raise BoundaryImportError(f"Unsafe ZIP member path: {name!r}")

    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        raise BoundaryImportError(f"Unsafe ZIP member path traversal: {name!r}")

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
            members: list[tuple[zipfile.ZipInfo, Path]] = []
            for info in handle.infolist():
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise BoundaryImportError(
                        f"ZIP symlink entries are not allowed: {info.filename!r}"
                    )
                members.append((info, _safe_zip_member_path(info.filename, destination_path)))

            for info, target in members:
                if info.is_dir() or info.filename.endswith(("/", "\\")):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
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


def _download_url(url: str, target: Path) -> Path:
    try:
        with urllib.request.urlopen(url) as response, target.open("wb") as output:
            shutil.copyfileobj(response, output)
    except (OSError, urllib.error.URLError) as exc:
        target.unlink(missing_ok=True)
        raise BoundaryImportError(f"Could not download boundary source: {url}") from exc
    return target


def _stage_remote_source(url: str, source_dir: Path) -> _StagedSource:
    parsed = urlsplit(url)
    source_name = Path(unquote(parsed.path)).name or "boundary-download"
    primary = _download_url(url, source_dir / source_name)
    suffix = Path(source_name).suffix.casefold()

    if zipfile.is_zipfile(primary):
        return _StagedSource(primary, (primary,))
    if suffix == ".shp":
        components = [primary]
        for component_suffix in _SHAPEFILE_REQUIRED_SUFFIXES[1:]:
            component_url = _url_with_suffix(url, component_suffix)
            target = source_dir / f"{Path(source_name).stem}{component_suffix}"
            try:
                components.append(_download_url(component_url, target))
            except BoundaryImportError as exc:
                detail = f"Missing required remote Shapefile component {component_suffix}"
                if component_suffix == ".prj":
                    detail += " declaring the CRS"
                raise BoundaryImportError(f"{detail}: {component_url}") from exc
        for component_suffix in _SHAPEFILE_OPTIONAL_SUFFIXES:
            component_url = _url_with_suffix(url, component_suffix)
            target = source_dir / f"{Path(source_name).stem}{component_suffix}"
            try:
                components.append(_download_url(component_url, target))
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
    value = str(scenario_id)
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise BoundaryImportError("scenario_id must be a nonempty filename-safe value")
    return value


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
