"""Safe, deterministic cached previews for registered local scenarios."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import geopandas as gpd
from PIL import Image, ImageDraw, ImageEnhance

from ..map_assets import MapAssetService, MapStyle

PREVIEW_STYLE_VERSION = "scenario-preview-v1"
PREVIEW_SIZE = (760, 360)
PreviewKind = Literal["terrain", "boundary", "fallback"]


@dataclass(frozen=True, slots=True)
class ScenarioPreviewRequest:
    scenario_id: str
    scenario_name: str
    boundary_path: Path
    dem_path: Path | None
    allowed_root: Path


@dataclass(frozen=True, slots=True)
class ScenarioPreviewResult:
    kind: PreviewKind
    path: Path
    cache_hit: bool
    diagnostic: str | None


def _redirected(path: Path) -> bool:
    if path.is_symlink():
        return True
    checker = getattr(path, "is_junction", None)
    if callable(checker) and checker():
        return True
    try:
        attrs = path.lstat().st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    return bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _check_components(path: Path, *, existing_only: bool = False) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if (not existing_only or os.path.lexists(current)) and _redirected(current):
            raise ValueError(f"path must not use a symlink or junction: {current}")


def _root(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("allowed_root must be a local directory")
    raw = Path(value).expanduser()
    if ".." in raw.parts:
        raise ValueError("allowed_root must not contain traversal")
    if raw.is_symlink() or _redirected(raw):
        raise ValueError("allowed_root must not be redirected")
    lexical = Path(os.path.abspath(raw))
    _check_components(lexical, existing_only=True)
    if not lexical.exists() or not lexical.is_dir():
        raise ValueError("allowed_root must be an existing directory")
    return lexical.resolve(strict=True)


def _safe_input(value: str | os.PathLike[str], root: Path, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or "://" in str(value):
        raise ValueError(f"{label} must be a local path")
    requested = Path(value).expanduser()
    if ".." in requested.parts:
        raise ValueError(f"{label} path must not contain traversal")
    lexical = Path(os.path.abspath(requested if requested.is_absolute() else root / requested))
    try:
        lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes allowed_root") from exc
    _check_components(lexical, existing_only=True)
    if not os.path.lexists(lexical):
        raise FileNotFoundError(f"{label} does not exist: {lexical}")
    if _redirected(lexical):
        raise ValueError(f"{label} must not be redirected")
    try:
        mode = lexical.stat()
    except OSError as exc:
        raise ValueError(f"{label} cannot be inspected") from exc
    if not stat.S_ISREG(mode.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if lexical.suffix.lower() == ".shp":
        for sidecar in lexical.parent.glob(f"{lexical.stem}.*"):
            if _redirected(sidecar):
                raise ValueError(f"{label} sidecar must not be redirected: {sidecar}")
            try:
                sidecar.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"{label} sidecar escapes allowed_root") from exc
    return lexical.resolve(strict=True)


def _safe_cache_root(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("cache_root must be a local directory")
    raw = Path(value).expanduser()
    if not raw.is_absolute() or ".." in raw.parts:
        raise ValueError("cache_root must be an absolute path without traversal")
    lexical = Path(os.path.abspath(raw))
    _check_components(lexical, existing_only=True)
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if not os.path.lexists(current):
            current.mkdir()
        if _redirected(current) or not current.is_dir():
            raise ValueError(f"cache path must be a regular directory: {current}")
    return lexical.resolve(strict=True)


def _stat_fingerprint(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        info = path.stat()
    except OSError:
        return {"path": str(path), "missing": True}
    return {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def _valid_png(path: Path) -> bool:
    if not os.path.lexists(path) or _redirected(path):
        return False
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            return False
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.format == "PNG" and image.size == PREVIEW_SIZE
    except (OSError, ValueError):
        return False


def _slug(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value)).strip("-_")
    return cleaned[:48] or "scenario"


def _cache_key(request: ScenarioPreviewRequest, boundary: Path | None, dem: Path | None) -> str:
    boundary_identity = {
        "requested": str(request.boundary_path),
        "resolved": _stat_fingerprint(boundary),
    }
    dem_identity = None
    if request.dem_path is not None:
        dem_identity = {
            "requested": str(request.dem_path),
            "resolved": _stat_fingerprint(dem),
        }
    payload = {
        "style": PREVIEW_STYLE_VERSION,
        "size": PREVIEW_SIZE,
        "id": str(request.scenario_id),
        "name": str(request.scenario_name),
        "allowed_root": str(request.allowed_root),
        "boundary": boundary_identity,
        "dem": dem_identity,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"{_slug(request.scenario_id)}-{hashlib.sha256(encoded).hexdigest()}.png"


def _bounded_diagnostic(value: str | None) -> str | None:
    if value is None:
        return None
    return value if len(value) <= 240 else value[:237] + "..."


def _atomic_save(image: Image.Image, destination: Path) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        image.save(temporary, format="PNG")
        if not _valid_png(temporary):
            raise ValueError("generated preview PNG is invalid")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _grid(draw: ImageDraw.ImageDraw) -> None:
    width, height = PREVIEW_SIZE
    for x in range(0, width + 1, 76):
        draw.line((x, 0, x, height), fill=(210, 214, 218), width=1)
    for y in range(0, height + 1, 60):
        draw.line((0, y, width, y), fill=(210, 214, 218), width=1)


def _padded_bounds(geometry: object) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = geometry.bounds
    spanx = max(maxx - minx, 1.0)
    spany = max(maxy - miny, 1.0)
    target_aspect = PREVIEW_SIZE[0] / PREVIEW_SIZE[1]
    spanx *= 1.12
    spany *= 1.12
    if spanx / spany < target_aspect:
        spanx = spany * target_aspect
    else:
        spany = spanx / target_aspect
    centerx = (minx + maxx) / 2.0
    centery = (miny + maxy) / 2.0
    return (
        centerx - spanx / 2.0,
        centery - spany / 2.0,
        centerx + spanx / 2.0,
        centery + spany / 2.0,
    )


def _rings(geometry: object):
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    for polygon in polygons:
        yield polygon.exterior, True
        for ring in polygon.interiors:
            yield ring, False


def _point_mapper(bounds: tuple[float, float, float, float]):
    minx, miny, maxx, maxy = bounds

    def point(x: float, y: float) -> tuple[int, int]:
        return (
            round((x - minx) / (maxx - minx) * (PREVIEW_SIZE[0] - 1)),
            round((maxy - y) / (maxy - miny) * (PREVIEW_SIZE[1] - 1)),
        )

    return point


class ScenarioPreviewService:
    """Validate one request and publish one bounded local preview."""

    def __init__(self, cache_root: str | os.PathLike[str]) -> None:
        self.cache_root = _safe_cache_root(cache_root)

    def build(self, request: ScenarioPreviewRequest) -> ScenarioPreviewResult:
        if not isinstance(request, ScenarioPreviewRequest):
            raise ValueError("request must be a ScenarioPreviewRequest")
        diagnostic: str | None = None
        boundary: Path | None = None
        dem: Path | None = None
        try:
            if not isinstance(request.scenario_id, str) or not request.scenario_id.strip():
                raise ValueError("scenario_id must be non-empty text")
            if not isinstance(request.scenario_name, str) or not request.scenario_name.strip():
                raise ValueError("scenario_name must be non-empty text")
            root = _root(request.allowed_root)
            boundary = _safe_input(request.boundary_path, root, "boundary")
        except Exception as exc:
            diagnostic = f"Boundary preview unavailable: {exc}"
        if boundary is not None and request.dem_path is not None:
            try:
                dem = _safe_input(request.dem_path, root, "DEM")
            except Exception as exc:
                diagnostic = f"DEM preview unavailable; showing boundary only: {exc}"

        filename = _cache_key(request, boundary, dem)
        destination = self.cache_root / filename
        if os.path.lexists(destination) and _redirected(destination):
            fallback_name = hashlib.sha256(f"redirected:{filename}".encode()).hexdigest()
            destination = self.cache_root / f"{_slug(request.scenario_id)}-{fallback_name}.png"
            diagnostic = "Preview cache entry was redirected; using a safe fallback path."
            kind: PreviewKind = "fallback"
        else:
            kind = "fallback"
        diagnostic = _bounded_diagnostic(diagnostic)
        if _valid_png(destination):
            cached_kind: PreviewKind = "fallback" if boundary is None else "boundary"
            return ScenarioPreviewResult(cached_kind, destination, True, diagnostic)

        if boundary is not None:
            try:
                frame = gpd.read_file(boundary)
                if frame.empty or frame.crs is None:
                    raise ValueError("boundary has no usable geometry or CRS")
                geometry = frame.to_crs("EPSG:3857").geometry.union_all()
                if geometry.is_empty or not geometry.is_valid:
                    raise ValueError("boundary geometry is empty or invalid")
                if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
                    raise ValueError("boundary must contain polygon geometry")
                bounds = _padded_bounds(geometry)
                point = _point_mapper(bounds)
                image = Image.new("RGB", PREVIEW_SIZE, (244, 245, 246))
                draw = ImageDraw.Draw(image)
                _grid(draw)
                kind = "boundary"
                if dem is None:
                    if diagnostic is None:
                        diagnostic = "DEM unavailable; showing boundary only."
                else:
                    try:
                        fingerprint = json.dumps(_stat_fingerprint(dem), sort_keys=True)
                        overlay = MapAssetService(root).dem_overlay(
                            dem,
                            fingerprint=fingerprint,
                            bounds=bounds,
                            bounds_crs="EPSG:3857",
                            style=MapStyle.COMBINED,
                            max_dimension=PREVIEW_SIZE[0],
                        )
                        with Image.open(overlay.path) as source:
                            terrain = source.convert("RGB").resize(
                                PREVIEW_SIZE, Image.Resampling.BILINEAR
                            )
                        terrain = ImageEnhance.Color(terrain).enhance(0.55)
                        terrain = Image.blend(
                            terrain,
                            Image.new("RGB", PREVIEW_SIZE, (242, 243, 239)),
                            0.18,
                        )
                        mask = Image.new("L", PREVIEW_SIZE, 0)
                        mask_draw = ImageDraw.Draw(mask)
                        for ring, exterior in _rings(geometry):
                            mask_draw.polygon(
                                [point(x, y) for x, y in ring.coords],
                                fill=255 if exterior else 0,
                            )
                        image.paste(terrain, mask=mask)
                        _grid(ImageDraw.Draw(image))
                        kind = "terrain"
                    except Exception as exc:
                        diagnostic = f"DEM preview unavailable; showing boundary only: {exc}"
                draw = ImageDraw.Draw(image)
                for ring, exterior in _rings(geometry):
                    draw.line(
                        [point(x, y) for x, y in ring.coords],
                        fill=(65, 74, 82),
                        width=3 if exterior else 2,
                        joint="curve",
                    )
            except Exception as exc:
                diagnostic = f"Boundary preview unavailable: {exc}"
                image = Image.new("RGB", PREVIEW_SIZE, (238, 240, 242))
                draw = ImageDraw.Draw(image)
                _grid(draw)
        else:
            image = Image.new("RGB", PREVIEW_SIZE, (238, 240, 242))
            draw = ImageDraw.Draw(image)
            _grid(draw)

        try:
            _atomic_save(image, destination)
        except Exception as exc:
            if diagnostic is None:
                diagnostic = f"Preview cache write failed: {exc}"
            # Return a stable path even when a write is interrupted; callers can
            # still display a neutral image when an earlier cache exists.
            if not _valid_png(destination):
                raise
        return ScenarioPreviewResult(kind, destination, False, _bounded_diagnostic(diagnostic))


def build_scenario_previews(
    requests: tuple[ScenarioPreviewRequest, ...] | list[ScenarioPreviewRequest],
    cache_root: str | os.PathLike[str],
) -> list[ScenarioPreviewResult]:
    """Build every requested preview, isolating failures per scenario."""

    service = ScenarioPreviewService(cache_root)
    results: list[ScenarioPreviewResult] = []
    for request in requests:
        try:
            results.append(service.build(request))
        except Exception as exc:
            fallback = Image.new("RGB", PREVIEW_SIZE, (238, 240, 242))
            digest = hashlib.sha256(repr(request).encode()).hexdigest()
            destination = service.cache_root / f"fallback-{digest}.png"
            try:
                _atomic_save(fallback, destination)
            except Exception:
                destination = service.cache_root / "fallback-unavailable.png"
            results.append(
                ScenarioPreviewResult("fallback", destination, False, _bounded_diagnostic(str(exc)))
            )
    return results


__all__ = [
    "PREVIEW_STYLE_VERSION",
    "PREVIEW_SIZE",
    "ScenarioPreviewRequest",
    "ScenarioPreviewResult",
    "ScenarioPreviewService",
    "build_scenario_previews",
]
