"""Progressive candidate explorer state, jobs, map data, and rendering."""

from __future__ import annotations

import math
from base64 import b64encode
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from queue import Empty
from threading import Lock
from time import monotonic
from typing import Any
from uuid import uuid4

from ...candidate_scanner import Candidate, ScanCancelled, ScanResult
from ...jobs import Job, JobBusyError, JobCoordinator
from ...map_assets import MapAsset, MapAssetService, MapStyle
from ...selection_service import (
    DemStatistics,
    SelectionError,
    SelectionProgress,
)

DEFAULT_LAYERS = frozenset({"dem", "boundary", "stations", "candidates"})
ALLOWED_LAYERS = DEFAULT_LAYERS | {"online"}
ALLOWED_VIEWS = frozenset({"map", "filmstrip"})


@dataclass(frozen=True, slots=True)
class CandidatePageState:
    """Immutable candidate explorer state keyed by flat grid identity."""

    job_id: str | None = None
    view: str = "map"
    phase: str = "idle"
    checked_positions: int = 0
    total_positions: int = 0
    elapsed_seconds: float = 0.0
    candidates: tuple[Candidate, ...] = ()
    found_count: int = 0
    selected_flat_grid_id: int | None = None
    map_bounds: tuple[float, float, float, float] | None = None
    enabled_layers: frozenset[str] = DEFAULT_LAYERS
    dem_style: MapStyle = MapStyle.COMBINED
    dem_opacity: float = 0.65
    dem_style_job_id: str | None = None
    dem_style_requested: MapStyle | None = None
    dem_style_asset: MapAsset | None = None
    dem_style_error: str | None = None
    cache_status: str = "none"
    cache_key: str = ""
    error: str | None = None
    error_code: str | None = None
    error_details: tuple[tuple[str, str], ...] = ()
    scan_completed: bool = False
    algorithm_version: str | None = None
    statistics_job_id: str | None = None
    statistics_requested_flat_grid_id: int | None = None
    statistics: DemStatistics | None = None
    statistics_flat_grid_id: int | None = None
    statistics_error: str | None = None
    candidate_preview_asset: MapAsset | None = None
    candidate_preview_error: str | None = None

    def __post_init__(self) -> None:
        if self.view not in ALLOWED_VIEWS:
            raise ValueError("view must be one of: filmstrip, map")
        if type(self.phase) is not str or not self.phase:
            raise ValueError("phase must be non-empty text")
        for name in ("checked_positions", "total_positions", "found_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.checked_positions > self.total_positions and self.total_positions > 0:
            raise ValueError("checked_positions cannot exceed total_positions")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        if not isinstance(self.candidates, tuple) or not all(
            isinstance(candidate, Candidate) for candidate in self.candidates
        ):
            raise ValueError("candidates must be a tuple of Candidate values")
        ids = tuple(candidate.flat_grid_id for candidate in self.candidates)
        if len(ids) != len(set(ids)):
            raise ValueError("candidate flat_grid_id values must be unique")
        if not isinstance(self.enabled_layers, frozenset):
            raise ValueError("enabled_layers must be a frozenset")
        if not self.enabled_layers <= ALLOWED_LAYERS:
            raise ValueError("enabled_layers contains an unsupported layer")
        if not isinstance(self.dem_style, MapStyle):
            raise ValueError("dem_style must be a MapStyle")
        if not math.isfinite(self.dem_opacity) or not 0.0 <= self.dem_opacity <= 1.0:
            raise ValueError("dem_opacity must be between zero and one")

    @classmethod
    def starting(
        cls,
        job_id: str,
        *,
        previous: CandidatePageState | None = None,
    ) -> CandidatePageState:
        if type(job_id) is not str or not job_id:
            raise ValueError("job_id must be non-empty text")
        source = previous or cls()
        return cls(
            job_id=job_id,
            view=source.view,
            phase="starting",
            map_bounds=source.map_bounds,
            enabled_layers=source.enabled_layers,
            dem_style=source.dem_style,
            dem_opacity=source.dem_opacity,
            dem_style_asset=source.dem_style_asset,
        )

    @classmethod
    def from_scan(cls, job_id: str, result: ScanResult) -> CandidatePageState:
        if not isinstance(result, ScanResult):
            raise ValueError("result must be a ScanResult")
        return cls(
            job_id=job_id,
            phase="completed" if result.completed else "scanning",
            checked_positions=result.checked_positions,
            total_positions=result.total_positions,
            candidates=result.candidates,
            found_count=len(result.candidates),
            scan_completed=result.completed,
            algorithm_version=result.algorithm_version,
        )

    @property
    def selected_index(self) -> int | None:
        if self.selected_flat_grid_id is None:
            return None
        return next(
            (
                index
                for index, candidate in enumerate(self.candidates)
                if candidate.flat_grid_id == self.selected_flat_grid_id
            ),
            None,
        )

    @property
    def selected_candidate(self) -> Candidate | None:
        index = self.selected_index
        return None if index is None else self.candidates[index]

    @property
    def can_confirm(self) -> bool:
        return (
            self.phase == "completed"
            and self.scan_completed
            and self.selected_candidate is not None
        )

    @property
    def progress_fraction(self) -> float:
        if self.total_positions <= 0:
            return 0.0
        return min(1.0, self.checked_positions / self.total_positions)

    def with_view(self, view: str) -> CandidatePageState:
        return replace(self, view=view)

    def with_selected(self, index: int | None) -> CandidatePageState:
        if index is None or type(index) is not int or not 0 <= index < len(self.candidates):
            return self.with_selected_flat_grid_id(None)
        return self.with_selected_flat_grid_id(self.candidates[index].flat_grid_id)

    def with_selected_flat_grid_id(
        self,
        flat_grid_id: int | None,
    ) -> CandidatePageState:
        valid = (
            flat_grid_id
            if type(flat_grid_id) is int
            and any(
                candidate.flat_grid_id == flat_grid_id
                for candidate in self.candidates
            )
            else None
        )
        if valid == self.selected_flat_grid_id:
            return self
        return replace(
            self,
            selected_flat_grid_id=valid,
            statistics=None,
            statistics_flat_grid_id=None,
            statistics_error=None,
            candidate_preview_asset=None,
            candidate_preview_error=None,
        )

    def with_map_bounds(
        self,
        bounds: tuple[float, float, float, float] | None,
    ) -> CandidatePageState:
        if bounds is not None:
            if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
                raise ValueError("map bounds must contain four finite values")
            if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
                raise ValueError("map bounds must have positive area")
        return replace(self, map_bounds=bounds)

    def with_layer(self, layer: str, enabled: bool) -> CandidatePageState:
        if layer not in ALLOWED_LAYERS:
            raise ValueError(f"Unsupported candidate map layer: {layer}")
        layers = set(self.enabled_layers)
        if enabled:
            layers.add(layer)
        else:
            layers.discard(layer)
        return replace(self, enabled_layers=frozenset(layers))

    def with_dem(
        self,
        *,
        style: MapStyle | None = None,
        opacity: float | None = None,
    ) -> CandidatePageState:
        return replace(
            self,
            dem_style=self.dem_style if style is None else style,
            dem_opacity=self.dem_opacity if opacity is None else opacity,
        )

    def with_dem_style_job(self, job_id: str, style: MapStyle) -> CandidatePageState:
        return replace(
            self,
            dem_style_job_id=job_id,
            dem_style_requested=style,
            dem_style_error=None,
        )

    def with_dem_style_result(
        self,
        job_id: str,
        style: MapStyle,
        asset: MapAsset | None,
        error: str | None,
    ) -> CandidatePageState:
        if self.dem_style_job_id != job_id or self.dem_style_requested is not style:
            return self
        return replace(
            self,
            dem_style=style if asset is not None else self.dem_style,
            dem_style_job_id=None,
            dem_style_requested=None,
            dem_style_asset=asset if asset is not None else self.dem_style_asset,
            dem_style_error=error,
            statistics=None if asset is not None else self.statistics,
            statistics_flat_grid_id=(
                None if asset is not None else self.statistics_flat_grid_id
            ),
            statistics_error=None if asset is not None else self.statistics_error,
            candidate_preview_asset=(
                None if asset is not None else self.candidate_preview_asset
            ),
            candidate_preview_error=(
                None if asset is not None else self.candidate_preview_error
            ),
        )

    def with_scan_result(self, result: ScanResult) -> CandidatePageState:
        authoritative = CandidatePageState.from_scan(self.job_id or "scan", result)
        selected = self.selected_flat_grid_id
        selected = (
            selected
            if selected is not None
            and any(candidate.flat_grid_id == selected for candidate in result.candidates)
            else None
        )
        preserve_statistics = selected is not None and (
            self.statistics_flat_grid_id == selected
        )
        return replace(
            authoritative,
            job_id=self.job_id,
            view=self.view,
            map_bounds=self.map_bounds,
            enabled_layers=self.enabled_layers,
            dem_style=self.dem_style,
            dem_opacity=self.dem_opacity,
            dem_style_asset=self.dem_style_asset,
            dem_style_error=self.dem_style_error,
            cache_status=self.cache_status,
            cache_key=self.cache_key,
            elapsed_seconds=self.elapsed_seconds,
            selected_flat_grid_id=selected,
            statistics=self.statistics if preserve_statistics else None,
            statistics_flat_grid_id=(
                self.statistics_flat_grid_id if preserve_statistics else None
            ),
            statistics_error=self.statistics_error if preserve_statistics else None,
            candidate_preview_asset=(
                self.candidate_preview_asset if preserve_statistics else None
            ),
            candidate_preview_error=(
                self.candidate_preview_error if preserve_statistics else None
            ),
        )

    def with_phase(self, phase: str, *, error: str | None = None) -> CandidatePageState:
        return replace(
            self,
            phase=phase,
            error=error,
            error_code=None,
            error_details=(),
        )

    def cancelled(self) -> CandidatePageState:
        return replace(
            self,
            phase="cancelled",
            candidates=(),
            found_count=0,
            selected_flat_grid_id=None,
            scan_completed=False,
            error=None,
            error_code=None,
            error_details=(),
            statistics_job_id=None,
            statistics_requested_flat_grid_id=None,
            statistics=None,
            statistics_flat_grid_id=None,
            statistics_error=None,
            candidate_preview_asset=None,
            candidate_preview_error=None,
        )

    def failed(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> CandidatePageState:
        detail_items = tuple(
            sorted((str(key), str(value)) for key, value in (details or {}).items())
        )
        return replace(
            self,
            phase="failed",
            candidates=(),
            found_count=0,
            selected_flat_grid_id=None,
            scan_completed=False,
            error=message,
            error_code=code,
            error_details=detail_items,
            statistics_job_id=None,
            statistics_requested_flat_grid_id=None,
            statistics=None,
            statistics_flat_grid_id=None,
            statistics_error=None,
            candidate_preview_asset=None,
            candidate_preview_error=None,
        )

    def with_statistics_job(
        self,
        job_id: str,
        flat_grid_id: int,
    ) -> CandidatePageState:
        return replace(
            self,
            statistics_job_id=job_id,
            statistics_requested_flat_grid_id=flat_grid_id,
            statistics=None,
            statistics_flat_grid_id=None,
            statistics_error=None,
            candidate_preview_asset=None,
            candidate_preview_error=None,
        )

    def clear_statistics_job(self, job_id: str) -> CandidatePageState:
        if self.statistics_job_id != job_id:
            return self
        return replace(
            self,
            statistics_job_id=None,
            statistics_requested_flat_grid_id=None,
        )


@dataclass(frozen=True, slots=True)
class CandidateProgressEvent:
    job_id: str
    progress: SelectionProgress


def reduce_progress(
    state: CandidatePageState,
    event: SelectionProgress | CandidateProgressEvent,
) -> CandidatePageState:
    """Apply one immutable progress delta without trusting it as final output."""

    if isinstance(event, CandidateProgressEvent):
        if event.job_id != state.job_id:
            return state
        progress = event.progress
    elif isinstance(event, SelectionProgress):
        progress = event
    else:
        raise TypeError("event must be SelectionProgress or CandidateProgressEvent")

    removed = set(progress.removed_flat_grid_ids)
    by_id: OrderedDict[int, Candidate] = OrderedDict(
        (candidate.flat_grid_id, candidate)
        for candidate in state.candidates
        if candidate.flat_grid_id not in removed
    )
    for candidate in progress.added_candidates:
        by_id[candidate.flat_grid_id] = candidate
    candidates = tuple(by_id.values())
    selected = state.selected_flat_grid_id
    if selected is not None and selected not in by_id:
        selected = None
    preserve_statistics = selected is not None and state.statistics_flat_grid_id == selected
    phase = (
        "cancelling"
        if state.phase == "cancelling" and progress.phase not in {"cancelled", "failed"}
        else progress.phase
    )
    return replace(
        state,
        phase=phase,
        checked_positions=progress.checked_positions,
        total_positions=progress.total_positions,
        elapsed_seconds=progress.elapsed_seconds,
        candidates=candidates,
        found_count=progress.candidate_count,
        selected_flat_grid_id=selected,
        cache_status=progress.cache_status,
        cache_key=progress.cache_key,
        error=None,
        error_code=None,
        error_details=(),
        statistics=state.statistics if preserve_statistics else None,
        statistics_flat_grid_id=(state.statistics_flat_grid_id if preserve_statistics else None),
        statistics_error=state.statistics_error if preserve_statistics else None,
        candidate_preview_asset=(
            state.candidate_preview_asset if preserve_statistics else None
        ),
    )


@dataclass(frozen=True, slots=True)
class CandidateStatisticsEvent:
    scan_job_id: str
    statistics_job_id: str
    flat_grid_id: int
    statistics: DemStatistics | None
    error: str | None
    preview_asset: MapAsset | None = None
    preview_error: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateInspection:
    statistics: DemStatistics
    preview_asset: MapAsset | None = None
    preview_error: str | None = None


def reduce_statistics(
    state: CandidatePageState,
    event: CandidateStatisticsEvent,
) -> CandidatePageState:
    """Apply statistics only to the exact scan, job, and selected candidate."""

    if (
        event.scan_job_id != state.job_id
        or event.statistics_job_id != state.statistics_job_id
        or event.flat_grid_id != state.selected_flat_grid_id
    ):
        return state
    return replace(
        state,
        statistics_job_id=None,
        statistics_requested_flat_grid_id=None,
        statistics=event.statistics,
        statistics_flat_grid_id=(
            event.flat_grid_id if event.statistics is not None else None
        ),
        statistics_error=event.error,
        candidate_preview_asset=event.preview_asset,
        candidate_preview_error=event.preview_error,
    )


@dataclass(frozen=True, slots=True)
class CandidateDisplayBounds:
    flat_grid_id: int
    bounds: tuple[float, float, float, float]


def candidate_display_bounds(
    candidates: Iterable[Candidate],
    *,
    rectangle_size: float,
    crs: str,
) -> tuple[CandidateDisplayBounds, ...]:
    """Transform candidate rectangles to WGS84 for Leaflet hit testing."""

    from pyproj import Transformer

    if not math.isfinite(rectangle_size) or rectangle_size <= 0:
        raise ValueError("rectangle_size must be positive and finite")
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    output: list[CandidateDisplayBounds] = []
    for candidate in candidates:
        left, bottom, right, top = transformer.transform_bounds(
            candidate.left_x,
            candidate.bottom_y,
            candidate.left_x + rectangle_size,
            candidate.bottom_y + rectangle_size,
        )
        values = (float(left), float(bottom), float(right), float(top))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("candidate display bounds must be finite")
        output.append(CandidateDisplayBounds(candidate.flat_grid_id, values))
    return tuple(output)


def hit_test_candidate_indices(
    bounds: Iterable[CandidateDisplayBounds],
    *,
    latitude: float,
    longitude: float,
) -> tuple[int, ...]:
    """Return every overlapping tuple index in deterministic order."""

    if not math.isfinite(latitude) or not math.isfinite(longitude):
        return ()
    return tuple(
        index
        for index, display in enumerate(bounds)
        if display.bounds[0] <= longitude <= display.bounds[2]
        and display.bounds[1] <= latitude <= display.bounds[3]
    )


def online_tiles_available(probe: Callable[[], bool] | None) -> bool:
    """Run only an explicitly injected probe; offline mode performs no request."""

    if probe is None:
        return False
    try:
        return probe() is True
    except Exception:
        return False


def default_online_tile_probe() -> bool:
    """Probe one tile only after an explicit user opt-in action."""

    from urllib.request import Request, urlopen

    request = Request(
        "https://tile.openstreetmap.org/0/0/0.png",
        headers={"User-Agent": "lte-scenario-toolkit/1"},
        method="HEAD",
    )
    try:
        with urlopen(request, timeout=1.5) as response:
            return int(getattr(response, "status", 200)) < 500
    except Exception:
        return False


def _leaflet_bounds(
    bounds: tuple[float, float, float, float],
) -> list[list[float]]:
    left, bottom, right, top = bounds
    return [[bottom, left], [top, right]]


def _png_data_url(path: Path) -> str:
    return "data:image/png;base64," + b64encode(path.read_bytes()).decode("ascii")


def _thumbnail_background_style(
    image_url: str,
    map_bounds: tuple[float, float, float, float],
    candidate_bounds: tuple[float, float, float, float],
) -> str:
    map_left, map_bottom, map_right, map_top = map_bounds
    left, bottom, right, top = candidate_bounds
    map_width = map_right - map_left
    map_height = map_top - map_bottom
    x = min(1.0, max(0.0, (left - map_left) / map_width))
    y = min(1.0, max(0.0, (map_top - top) / map_height))
    width = min(1.0, max(1e-6, (right - left) / map_width))
    height = min(1.0, max(1e-6, (top - bottom) / map_height))
    position_x = 0.0 if width >= 1.0 else 100.0 * x / (1.0 - width)
    position_y = 0.0 if height >= 1.0 else 100.0 * y / (1.0 - height)
    return (
        f'background-image:url("{image_url}");'
        f"background-size:{100.0 / width:.4f}% {100.0 / height:.4f}%;"
        f"background-position:{position_x:.4f}% {position_y:.4f}%;"
        "background-repeat:no-repeat;"
    )


@lru_cache(maxsize=512)
def candidate_thumbnail_data_url(
    image_path: Path,
    map_bounds: tuple[float, float, float, float],
    candidate_bounds: tuple[float, float, float, float],
    *,
    width: int = 240,
    height: int = 144,
) -> str:
    """Crop one real candidate extent from the cached local DEM overview."""

    from PIL import Image

    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("thumbnail dimensions must be positive integers")
    map_left, map_bottom, map_right, map_top = map_bounds
    left, bottom, right, top = candidate_bounds
    if map_left >= map_right or map_bottom >= map_top:
        raise ValueError("map bounds must have positive area")
    with Image.open(image_path) as source:
        source.load()
        rgba = source.convert("RGBA")
        source_width, source_height = rgba.size
        x0 = math.floor(
            (max(map_left, left) - map_left)
            / (map_right - map_left)
            * source_width
        )
        x1 = math.ceil(
            (min(map_right, right) - map_left)
            / (map_right - map_left)
            * source_width
        )
        y0 = math.floor(
            (map_top - min(map_top, top))
            / (map_top - map_bottom)
            * source_height
        )
        y1 = math.ceil(
            (map_top - max(map_bottom, bottom))
            / (map_top - map_bottom)
            * source_height
        )
        x0 = min(source_width - 1, max(0, x0))
        y0 = min(source_height - 1, max(0, y0))
        x1 = min(source_width, max(x0 + 1, x1))
        y1 = min(source_height, max(y0 + 1, y1))
        crop = rgba.crop((x0, y0, x1, y1))
        crop.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (width, height), (14, 31, 34, 255))
        offset = ((width - crop.width) // 2, (height - crop.height) // 2)
        canvas.alpha_composite(crop, offset)
        buffer = BytesIO()
        canvas.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + b64encode(buffer.getvalue()).decode("ascii")


@dataclass(frozen=True, slots=True)
class CandidateMapBundle:
    dem_asset: MapAsset
    boundary_geojson: dict[str, Any]
    stations_geojson: dict[str, Any]
    map_bounds: tuple[float, float, float, float]


def build_candidate_map_bundle(
    session: CandidateSession,
    assets: MapAssetService,
    *,
    style: MapStyle = MapStyle.COMBINED,
) -> CandidateMapBundle:
    """Build bounded offline map assets from the frozen selection snapshot."""

    from pyproj import Transformer

    prepared = session.selection_service.prepared_selection(session.preflight)
    profile = session.profile_snapshot
    transformer = Transformer.from_crs(
        profile.target_crs,
        "EPSG:4326",
        always_xy=True,
    )
    left, bottom, right, top = transformer.transform_bounds(*prepared.boundary.bounds)
    map_bounds = (float(left), float(bottom), float(right), float(top))
    overlay = assets.dem_overlay(
        session.preflight.dem_path,
        fingerprint=session.preflight.dem_fingerprint,
        bounds=map_bounds,
        bounds_crs="EPSG:4326",
        style=style,
    )
    boundary = assets.boundary_geojson(
        prepared.boundary,
        crs=profile.target_crs,
    )
    stations = assets.station_geojson(
        prepared.points,
        prepared.boundary,
        boundary_crs=profile.target_crs,
    )
    return CandidateMapBundle(overlay, boundary, stations, map_bounds)


def build_candidate_overlay(
    session: CandidateSession,
    assets: MapAssetService,
    candidate: Candidate,
    *,
    style: MapStyle = MapStyle.COMBINED,
    max_dimension: int = 640,
) -> MapAsset:
    """Build one cached high-resolution DEM window only after selection."""

    size = float(session.profile_snapshot.rect_size)
    return assets.dem_overlay(
        session.preflight.dem_path,
        fingerprint=session.preflight.dem_fingerprint,
        bounds=(
            candidate.left_x,
            candidate.bottom_y,
            candidate.left_x + size,
            candidate.bottom_y + size,
        ),
        bounds_crs=session.profile_snapshot.target_crs,
        style=style,
        max_dimension=max_dimension,
    )


def build_candidate_style_overlay(
    session: CandidateSession,
    assets: MapAssetService,
    bounds: tuple[float, float, float, float],
    style: MapStyle,
) -> MapAsset:
    """Build one cached overview style without exposing the source DEM."""

    return assets.dem_overlay(
        session.preflight.dem_path,
        fingerprint=session.preflight.dem_fingerprint,
        bounds=bounds,
        bounds_crs="EPSG:4326",
        style=style,
    )


@dataclass(frozen=True, slots=True)
class CandidateSession:
    session_id: str
    profile_snapshot: Any
    preflight: Any
    selection_service: Any
    repo_root: Path
    map_bundle: CandidateMapBundle | None = None
    scan_result: ScanResult | None = None
    confirmed_flat_grid_id: int | None = None
    locked_candidate: Candidate | None = None
    created_at: float = field(default_factory=monotonic)

    def __post_init__(self) -> None:
        if type(self.session_id) is not str or not self.session_id:
            raise ValueError("session_id must be non-empty text")
        if getattr(self.preflight, "profile", None) is not self.profile_snapshot:
            raise ValueError("preflight must retain the exact frozen profile snapshot")
        object.__setattr__(self, "repo_root", Path(self.repo_root).resolve())


class CandidateSessionRegistry:
    """Bounded, process-local registry for opaque frozen selection sessions."""

    def __init__(self, max_sessions: int = 32) -> None:
        if type(max_sessions) is not int or max_sessions <= 0:
            raise ValueError("max_sessions must be a positive integer")
        self.max_sessions = max_sessions
        self._lock = Lock()
        self._sessions: OrderedDict[str, CandidateSession] = OrderedDict()
        self._pins: dict[str, int] = {}

    def _trim_locked(self, *, protected_id: str | None = None) -> None:
        while len(self._sessions) > self.max_sessions:
            evictable = next(
                (
                    session_id
                    for session_id in self._sessions
                    if session_id != protected_id
                    and self._pins.get(session_id, 0) == 0
                ),
                None,
            )
            if evictable is None:
                raise RuntimeError("candidate session capacity is occupied by active pages")
            self._sessions.pop(evictable)

    def add(self, session: CandidateSession) -> CandidateSession:
        if not isinstance(session, CandidateSession):
            raise ValueError("session must be a CandidateSession")
        with self._lock:
            previous = self._sessions.get(session.session_id)
            self._sessions.pop(session.session_id, None)
            self._sessions[session.session_id] = session
            try:
                self._trim_locked(protected_id=session.session_id)
            except RuntimeError:
                self._sessions.pop(session.session_id, None)
                if previous is not None:
                    self._sessions[session.session_id] = previous
                raise
        return session

    def create(
        self,
        outcome: Any,
        selection_service: Any,
        repo_root: str | Path,
    ) -> CandidateSession:
        if not getattr(outcome, "ok", False) or outcome.preflight is None:
            raise ValueError("candidate session requires a successful preflight")
        return self.add(
            CandidateSession(
                session_id=uuid4().hex,
                profile_snapshot=outcome.snapshot,
                preflight=outcome.preflight,
                selection_service=selection_service,
                repo_root=Path(repo_root),
            )
        )

    def get(self, session_id: str) -> CandidateSession | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions.move_to_end(session_id)
            return session

    def pin(self, session_id: str) -> CandidateSession:
        """Prevent one page-owned session from being evicted while it is active."""

        with self._lock:
            try:
                session = self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"Unknown candidate session: {session_id}") from exc
            self._pins[session_id] = self._pins.get(session_id, 0) + 1
            self._sessions.move_to_end(session_id)
            return session

    def unpin(self, session_id: str) -> None:
        with self._lock:
            count = self._pins.get(session_id, 0)
            if count <= 1:
                self._pins.pop(session_id, None)
            else:
                self._pins[session_id] = count - 1
            self._trim_locked()

    def _replace(self, session_id: str, **changes: Any) -> CandidateSession:
        with self._lock:
            try:
                current = self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"Unknown candidate session: {session_id}") from exc
            updated = replace(current, **changes)
            self._sessions[session_id] = updated
            self._sessions.move_to_end(session_id)
            return updated

    def set_map_bundle(
        self,
        session_id: str,
        bundle: CandidateMapBundle,
    ) -> CandidateSession:
        if not isinstance(bundle, CandidateMapBundle):
            raise ValueError("bundle must be a CandidateMapBundle")
        return self._replace(session_id, map_bundle=bundle)

    def set_scan_result(
        self,
        session_id: str,
        result: ScanResult,
    ) -> CandidateSession:
        if not isinstance(result, ScanResult) or not result.completed:
            raise ValueError("session scan result must be a completed ScanResult")
        return self._replace(
            session_id,
            scan_result=result,
            confirmed_flat_grid_id=None,
            locked_candidate=None,
        )

    def confirm(self, session_id: str, flat_grid_id: int) -> CandidateSession:
        with self._lock:
            try:
                current = self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"Unknown candidate session: {session_id}") from exc
            result = current.scan_result
            if result is None or not result.completed:
                raise ValueError("candidate confirmation requires a final scan result")
            candidate = next(
                (
                    item
                    for item in result.candidates
                    if item.flat_grid_id == flat_grid_id
                ),
                None,
            )
            if candidate is None:
                raise ValueError("confirmed candidate is absent from the final scan")
            updated = replace(
                current,
                confirmed_flat_grid_id=flat_grid_id,
                locked_candidate=candidate,
            )
            self._sessions[session_id] = updated
            self._sessions.move_to_end(session_id)
            return updated

    def discard(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._pins.pop(session_id, None)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._pins.clear()


class CandidateExplorerController:
    """Framework-free controller for scan, cancel, statistics, and confirmation."""

    def __init__(
        self,
        session: CandidateSession,
        coordinator: JobCoordinator,
        *,
        registry: CandidateSessionRegistry | None = None,
        initial_state: CandidatePageState | None = None,
        candidate_overlay_builder: Callable[
            [CandidateSession, Candidate, MapStyle], MapAsset
        ]
        | None = None,
        dem_style_builder: Callable[[CandidateSession, MapStyle], MapAsset]
        | None = None,
    ) -> None:
        self.session = session
        self.coordinator = coordinator
        self.registry = registry
        self.candidate_overlay_builder = candidate_overlay_builder
        self.dem_style_builder = dem_style_builder
        if registry is not None:
            self.session = registry.pin(session.session_id)
        if initial_state is not None:
            self._state = initial_state
        elif session.scan_result is not None:
            self._state = CandidatePageState.from_scan(
                f"session-{session.session_id}",
                session.scan_result,
            )
        else:
            self._state = CandidatePageState()
        self._scan_job: Job | None = None
        self._statistics_job: Job | None = None
        self._dem_style_job: Job | None = None
        self._dem_style_requested: MapStyle | None = None
        self._statistics_scan_job_id: str | None = None
        self._statistics_flat_grid_id: int | None = None
        self._statistics_style: MapStyle | None = None
        self._statistics_cache: dict[
            tuple[str, int, MapStyle], CandidateInspection
        ] = {}
        self._scan_protocol_failed_jobs: set[str] = set()
        self._closed = False

    @property
    def state(self) -> CandidatePageState:
        return self._state

    @property
    def scan_job(self) -> Job | None:
        return self._scan_job

    @property
    def statistics_job(self) -> Job | None:
        return self._statistics_job

    @property
    def dem_style_job(self) -> Job | None:
        return self._dem_style_job

    @property
    def active(self) -> bool:
        return any(
            job is not None and job.future is not None and not job.future.done()
            for job in (self._scan_job, self._statistics_job, self._dem_style_job)
        )

    @staticmethod
    def _release_when_done(coordinator: JobCoordinator, job: Job) -> None:
        if job.future is not None:
            job.future.add_done_callback(
                lambda _future, job_id=job.job_id: coordinator.finish(job_id)
            )

    def start_scan(self, *, force: bool = False) -> Job:
        if self._closed:
            raise RuntimeError("candidate explorer is closed")
        if self._scan_job is not None and self._scan_job.future is not None:
            if self._scan_job.future.done():
                self.drain_scan(self._scan_job)

        service = self.session.selection_service
        preflight = self.session.preflight

        def worker(cancel, emit):
            return service.scan(
                preflight,
                force=force,
                progress=emit,
                cancel=cancel,
            )

        job = self.coordinator.submit("selection.scan", worker)
        self._release_when_done(self.coordinator, job)
        self._scan_job = job
        self._state = CandidatePageState.starting(job.job_id, previous=self._state)
        return job

    def drain_scan(self, job: Job | None = None) -> CandidatePageState:
        target = job or self._scan_job
        if self._closed or target is None or target.job_id != self._state.job_id:
            return self._state
        if target.job_id in self._scan_protocol_failed_jobs:
            future = target.future
            if future is None or not future.done():
                return self._state
            try:
                future.result()
            except Exception:
                pass
            self._scan_protocol_failed_jobs.discard(target.job_id)
            if self._scan_job is not None and self._scan_job.job_id == target.job_id:
                self._scan_job = None
            return self._state
        while True:
            try:
                event = target.progress.get_nowait()
            except Empty:
                break
            if not isinstance(event, SelectionProgress):
                self._state = self._state.failed("Invalid scan progress event")
                self._scan_protocol_failed_jobs.add(target.job_id)
                break
            self._state = reduce_progress(
                self._state,
                CandidateProgressEvent(target.job_id, event),
            )
        future = target.future
        if future is None or not future.done():
            return self._state
        if target.job_id in self._scan_protocol_failed_jobs:
            try:
                future.result()
            except Exception:
                pass
            self._scan_protocol_failed_jobs.discard(target.job_id)
            if self._scan_job is not None and self._scan_job.job_id == target.job_id:
                self._scan_job = None
            return self._state
        try:
            result = future.result()
            if not isinstance(result, ScanResult):
                raise TypeError("selection scan returned an invalid result")
            self._state = self._state.with_scan_result(result)
            if result.completed and self.registry is not None:
                try:
                    self.session = self.registry.set_scan_result(
                        self.session.session_id,
                        result,
                    )
                except KeyError:
                    self.session = replace(self.session, scan_result=result)
            elif result.completed:
                self.session = replace(self.session, scan_result=result)
        except ScanCancelled:
            self._state = self._state.cancelled()
        except SelectionError as exc:
            self._state = self._state.failed(
                exc.message,
                code=exc.code,
                details=exc.details,
            )
        except Exception as exc:
            self._state = self._state.failed(str(exc))
        finally:
            if self._scan_job is not None and self._scan_job.job_id == target.job_id:
                self._scan_job = None
        return self._state

    def cancel_scan(self) -> bool:
        job = self._scan_job
        if (
            job is None
            or job.future is None
            or job.future.done()
            or job.job_id != self._state.job_id
        ):
            return False
        if not self.coordinator.cancel(job.job_id):
            return False
        self._state = self._state.with_phase("cancelling")
        return True

    def select_index(self, index: int | None) -> CandidatePageState:
        self._state = self._state.with_selected(index)
        return self._state

    def select_flat_grid_id(self, flat_grid_id: int | None) -> CandidatePageState:
        self._state = self._state.with_selected_flat_grid_id(flat_grid_id)
        return self._state

    def set_map_bounds(
        self,
        bounds: tuple[float, float, float, float],
    ) -> CandidatePageState:
        self._state = self._state.with_map_bounds(bounds)
        return self._state

    def set_view(self, view: str) -> CandidatePageState:
        self._state = self._state.with_view(view)
        return self._state

    def set_layer(self, layer: str, enabled: bool) -> CandidatePageState:
        self._state = self._state.with_layer(layer, enabled)
        return self._state

    def set_dem_opacity(self, opacity: float) -> CandidatePageState:
        self._state = self._state.with_dem(opacity=opacity)
        return self._state

    def request_dem_style(self, style: MapStyle) -> Job | None:
        if self._closed or not isinstance(style, MapStyle):
            return None
        if style is self._state.dem_style and self._state.dem_style_asset is not None:
            return None
        if self.dem_style_builder is None:
            return None

        def worker(_cancel, _emit):
            return self.dem_style_builder(self.session, style)

        job = self.coordinator.submit("candidate.dem_style", worker)
        self._release_when_done(self.coordinator, job)
        self._dem_style_job = job
        self._dem_style_requested = style
        self._state = self._state.with_dem_style_job(job.job_id, style)
        return job

    def drain_dem_style(self) -> CandidatePageState:
        job = self._dem_style_job
        style = self._dem_style_requested
        if self._closed or job is None or style is None:
            return self._state
        future = job.future
        if future is None or not future.done():
            return self._state
        try:
            asset = future.result()
            if not isinstance(asset, MapAsset):
                raise TypeError("DEM style worker returned an invalid asset")
            self._state = self._state.with_dem_style_result(
                job.job_id, style, asset, None
            )
        except Exception as exc:
            self._state = self._state.with_dem_style_result(
                job.job_id, style, None, str(exc)
            )
        finally:
            self._dem_style_job = None
            self._dem_style_requested = None
        return self._state

    def request_statistics(self) -> Job | None:
        candidate = self._state.selected_candidate
        scan_job_id = self._state.job_id
        if (
            not self._state.scan_completed
            or candidate is None
            or scan_job_id is None
            or self._closed
        ):
            return None
        style = self._state.dem_style
        cache_key = (scan_job_id, candidate.flat_grid_id, style)
        cached = self._statistics_cache.get(cache_key)
        if cached is not None:
            self._state = replace(
                self._state,
                statistics=cached.statistics,
                statistics_flat_grid_id=candidate.flat_grid_id,
                statistics_error=None,
                candidate_preview_asset=cached.preview_asset,
                candidate_preview_error=cached.preview_error,
            )
            return None
        if self._statistics_job is not None and self._statistics_job.future is not None:
            if not self._statistics_job.future.done():
                return None
            self.drain_statistics(self._statistics_job)

        service = self.session.selection_service
        preflight = self.session.preflight

        def worker(_cancel, _emit):
            statistics = service.candidate_statistics(preflight, candidate)
            preview_asset = None
            preview_error = None
            if self.candidate_overlay_builder is not None:
                try:
                    preview_asset = self.candidate_overlay_builder(
                        self.session,
                        candidate,
                        style,
                    )
                except Exception as exc:
                    preview_error = str(exc)
            return CandidateInspection(statistics, preview_asset, preview_error)

        job = self.coordinator.submit("candidate.statistics", worker)
        self._release_when_done(self.coordinator, job)
        self._statistics_job = job
        self._statistics_scan_job_id = scan_job_id
        self._statistics_flat_grid_id = candidate.flat_grid_id
        self._statistics_style = style
        self._state = self._state.with_statistics_job(
            job.job_id,
            candidate.flat_grid_id,
        )
        return job

    def drain_statistics(self, job: Job | None = None) -> CandidatePageState:
        target = job or self._statistics_job
        if self._closed or target is None:
            return self._state
        if self._statistics_job is None or target.job_id != self._statistics_job.job_id:
            return self._state
        future = target.future
        if future is None or not future.done():
            return self._state
        scan_job_id = self._statistics_scan_job_id or ""
        flat_grid_id = self._statistics_flat_grid_id
        style = self._statistics_style or self._state.dem_style
        try:
            inspection = future.result()
            if isinstance(inspection, DemStatistics):
                inspection = CandidateInspection(inspection)
            if not isinstance(inspection, CandidateInspection):
                raise TypeError("candidate statistics returned an invalid result")
            event = CandidateStatisticsEvent(
                scan_job_id,
                target.job_id,
                -1 if flat_grid_id is None else flat_grid_id,
                inspection.statistics,
                None,
                inspection.preview_asset,
                inspection.preview_error,
            )
            reduced = reduce_statistics(self._state, event)
            if reduced is not self._state and flat_grid_id is not None:
                self._statistics_cache[(scan_job_id, flat_grid_id, style)] = inspection
            self._state = reduced
        except SelectionError as exc:
            event = CandidateStatisticsEvent(
                scan_job_id,
                target.job_id,
                -1 if flat_grid_id is None else flat_grid_id,
                None,
                f"{exc.code}: {exc.message}",
            )
            self._state = reduce_statistics(self._state, event)
        except Exception as exc:
            event = CandidateStatisticsEvent(
                scan_job_id,
                target.job_id,
                -1 if flat_grid_id is None else flat_grid_id,
                None,
                str(exc),
            )
            self._state = reduce_statistics(self._state, event)
        finally:
            self._state = self._state.clear_statistics_job(target.job_id)
            self._statistics_job = None
            self._statistics_scan_job_id = None
            self._statistics_flat_grid_id = None
            self._statistics_style = None
        return self._state

    def confirm(self) -> CandidateSession:
        candidate = self._state.selected_candidate
        if not self._state.can_confirm or candidate is None:
            raise ValueError("confirmation requires one candidate from a completed scan")
        result = self.session.scan_result
        if result is None or not any(
            item.flat_grid_id == candidate.flat_grid_id for item in result.candidates
        ):
            raise ValueError("selected candidate is absent from the final scan result")
        if self.registry is not None:
            self.session = self.registry.confirm(
                self.session.session_id,
                candidate.flat_grid_id,
            )
        else:
            self.session = replace(
                self.session,
                confirmed_flat_grid_id=candidate.flat_grid_id,
                locked_candidate=candidate,
            )
        return self.session

    def close(self) -> None:
        if self._closed:
            return
        self.cancel_scan()
        statistics_job = self._statistics_job
        if statistics_job is not None:
            self.coordinator.cancel(statistics_job.job_id)
        dem_style_job = self._dem_style_job
        if dem_style_job is not None:
            self.coordinator.cancel(dem_style_job.job_id)
        self._closed = True
        if self.registry is not None:
            self.registry.unpin(self.session.session_id)


@dataclass(slots=True)
class CandidatePageView:
    """Rendered page handles retained for cleanup and focused GUI tests."""

    controller: CandidateExplorerController
    map_element: Any
    timer: Any
    coordinator_timer: Any
    map_container: Any
    filmstrip_container: Any


def _event_latlng(event: Any) -> tuple[float, float] | None:
    payload = getattr(event, "args", None)
    if not isinstance(payload, dict):
        return None
    latlng = payload.get("latlng", payload)
    if not isinstance(latlng, dict):
        return None
    try:
        latitude = float(latlng["lat"])
        longitude = float(latlng["lng"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        return None
    return latitude, longitude


def render_candidate_unavailable(ui: Any, translator: Any, message: str | None = None) -> None:
    """Render a safe local failure page without creating map assets or jobs."""

    with ui.column().classes("lte-page lte-candidate-page"):
        ui.label(translator.text("candidates.unavailable")).classes("lte-page-title")
        ui.label(message or translator.text("candidates.unavailable_body")).classes(
            "lte-callout lte-callout--warning"
        )
        ui.button(
            translator.text("candidates.back_to_configure"),
            on_click=lambda: ui.navigate.to("/configure"),
        ).props("outline")


def render_candidate_page(
    ui: Any,
    translator: Any,
    session: CandidateSession,
    coordinator: JobCoordinator,
    *,
    registry: CandidateSessionRegistry | None = None,
    dem_asset_url: str | None = None,
    dem_asset_url_builder: Callable[[Path], str] | None = None,
    online_tile_probe: Callable[[], bool] | None = None,
    candidate_overlay_builder: Callable[
        [CandidateSession, Candidate, MapStyle], MapAsset
    ]
    | None = None,
    dem_style_builder: Callable[[CandidateSession, MapStyle], MapAsset] | None = None,
    on_confirm: Callable[[CandidateSession], None] | None = None,
    auto_start: bool = True,
) -> CandidatePageView:
    """Render one progressive, offline-first candidate selection page."""

    bundle = session.map_bundle
    if bundle is None:
        raise ValueError("candidate page requires a prepared map bundle")
    controller = CandidateExplorerController(
        session,
        coordinator,
        registry=registry,
        candidate_overlay_builder=candidate_overlay_builder,
        dem_style_builder=dem_style_builder,
    )
    controller.set_map_bounds(bundle.map_bounds)
    image_url = dem_asset_url or _png_data_url(bundle.dem_asset.path)

    def asset_url(path: Path) -> str:
        return (
            dem_asset_url_builder(path)
            if dem_asset_url_builder is not None
            else _png_data_url(path)
        )

    profile = session.profile_snapshot
    rectangle_size = float(profile.rect_size)
    display_bounds: tuple[CandidateDisplayBounds, ...] = ()
    candidate_layers: dict[int, Any] = {}
    online_layer: Any | None = None
    last_filmstrip_signature: tuple[
        tuple[int, ...], int | None, str | None, str
    ] | None = None
    last_candidate_layer_signature: tuple[Any, ...] | None = None
    last_rendered_state: CandidatePageState | None = None
    last_coordinator_active: bool | None = None
    last_dem_asset_path = str(bundle.dem_asset.path)
    current_overview_url = image_url

    with ui.column().classes("lte-page lte-candidate-page"):
        with ui.row().classes("lte-candidate-heading items-end justify-between full-width"):
            with ui.column().classes("gap-1"):
                ui.label(translator.text("candidates.title")).classes("lte-page-title")
                ui.label(
                    translator.text(
                        "candidates.subtitle",
                        name=profile.display_name,
                    )
                ).classes("lte-page-subtitle")
            with ui.button_group().props("outline"):
                map_view_button = ui.button(
                    translator.text("candidates.view_map")
                ).mark("candidate-view-map")
                filmstrip_view_button = ui.button(
                    translator.text("candidates.view_filmstrip")
                ).mark("candidate-view-filmstrip")

        with ui.card().classes("lte-candidate-progress full-width"):
            with ui.row().classes("items-center justify-between full-width"):
                progress_phase = ui.label(
                    translator.text("candidates.phase_idle")
                ).classes("lte-section-title")
                progress_numbers = ui.label("0 / 0").classes("lte-candidate-metric")
            progress_bar = ui.linear_progress(value=0.0).props("rounded color=teal")
            with ui.row().classes("items-center justify-between full-width"):
                found_label = ui.label(
                    translator.text("candidates.found", count=0)
                ).classes("lte-page-subtitle")
                elapsed_label = ui.label(
                    translator.text("candidates.elapsed", seconds="0.0")
                ).classes("lte-page-subtitle")
                cache_label = ui.label(
                    translator.text("candidates.cache", status="none")
                ).classes("lte-page-subtitle")
            progress_error = ui.label("").classes(
                "lte-validation-result lte-validation-result--error"
            )
            progress_error.set_visibility(False)

        with ui.row().classes("lte-candidate-toolbar items-center full-width"):
            start_button = ui.button(
                translator.text("action.start_scan")
            ).mark("candidate-start")
            cancel_button = ui.button(
                translator.text("action.cancel")
            ).props("outline").mark("candidate-cancel")
            force_button = ui.button(
                translator.text("candidates.force_rescan")
            ).props("outline").mark("candidate-force")
            ui.separator().props("vertical")
            previous_button = ui.button(
                translator.text("candidates.previous")
            ).props("flat").mark("candidate-previous")
            next_button = ui.button(
                translator.text("candidates.next")
            ).props("flat").mark("candidate-next")
            candidate_id_input = ui.number(
                translator.text("candidates.grid_id"),
                value=None,
                precision=0,
            ).classes("lte-candidate-id-input").mark("candidate-id")
            choose_id_button = ui.button(
                translator.text("candidates.select_id")
            ).props("outline").mark("candidate-select-id")
            ui.space()
            confirm_button = ui.button(
                translator.text("candidates.confirm_selection")
            ).mark("candidate-confirm")

        with ui.grid(columns=2).classes("lte-candidate-workspace full-width"):
            with ui.card().classes("lte-candidate-map-card"):
                with ui.row().classes("items-center full-width lte-layer-controls"):
                    dem_switch = ui.switch(
                        translator.text("candidates.layer_dem"), value=True
                    ).mark("candidate-layer-dem")
                    boundary_switch = ui.switch(
                        translator.text("candidates.layer_boundary"), value=True
                    ).mark("candidate-layer-boundary")
                    stations_switch = ui.switch(
                        translator.text("candidates.layer_stations"), value=True
                    ).mark("candidate-layer-stations")
                    candidates_switch = ui.switch(
                        translator.text("candidates.layer_candidates"), value=True
                    ).mark("candidate-layer-candidates")
                    online_switch = ui.switch(
                        translator.text("candidates.layer_online"), value=False
                    ).mark("candidate-layer-online")
                    dem_opacity = ui.slider(
                        min=0.1,
                        max=1.0,
                        step=0.05,
                        value=controller.state.dem_opacity,
                    ).classes("lte-dem-opacity").mark("candidate-dem-opacity")
                    dem_style_select = ui.select(
                        {
                            MapStyle.ELEVATION: translator.text(
                                "candidates.style_elevation"
                            ),
                            MapStyle.HILLSHADE: translator.text(
                                "candidates.style_hillshade"
                            ),
                            MapStyle.COMBINED: translator.text(
                                "candidates.style_combined"
                            ),
                        },
                        value=controller.state.dem_style,
                        label=translator.text("candidates.dem_style"),
                    ).classes("lte-dem-style").mark("candidate-dem-style")
                map_container = ui.column().classes("lte-candidate-map-wrap full-width")
                with map_container:
                    left, bottom, right, top = bundle.map_bounds
                    map_element = ui.leaflet(
                        center=((bottom + top) / 2.0, (left + right) / 2.0),
                        zoom=10,
                        options={"zoomControl": True, "preferCanvas": True},
                    ).classes("lte-candidate-map full-width").mark("candidate-map")
                    map_element.clear_layers()
                    dem_layer = map_element.image_overlay(
                        url=image_url,
                        bounds=_leaflet_bounds(bundle.map_bounds),
                        options={"opacity": controller.state.dem_opacity},
                    )
                    boundary_layer = map_element.generic_layer(
                        name="geoJSON",
                        args=[
                            bundle.boundary_geojson,
                            {
                                "style": {
                                    "color": "#0f766e",
                                    "weight": 2,
                                    "fillOpacity": 0.03,
                                }
                            },
                        ],
                    )
                    stations_layer = map_element.generic_layer(
                        name="geoJSON",
                        args=[bundle.stations_geojson, {}],
                    )
                    leaflet_extent = _leaflet_bounds(bundle.map_bounds)
                    map_element.on(
                        "init",
                        lambda: map_element.run_map_method(
                            "fitBounds",
                            leaflet_extent,
                            {"padding": [16, 16]},
                        ),
                    )

            with ui.card().classes("lte-candidate-inspector"):
                ui.label(translator.text("candidates.inspector")).classes(
                    "lte-section-title"
                )
                selected_title = ui.label(
                    translator.text("candidates.none_selected")
                ).classes("lte-candidate-selected-title")
                selected_details = ui.label(
                    translator.text("candidates.select_hint")
                ).classes("lte-page-subtitle")
                selected_preview = ui.image().classes("lte-selected-dem-preview").mark(
                    "candidate-selected-preview"
                )
                selected_preview.set_visibility(False)
                stats_title = ui.label(
                    translator.text("candidates.terrain_statistics")
                ).classes("lte-section-title")
                stats_details = ui.label(
                    translator.text("candidates.statistics_waiting")
                ).classes("lte-candidate-statistics")
                stats_title.set_visibility(False)
                stats_details.set_visibility(False)

        filmstrip_container = ui.column().classes(
            "lte-candidate-filmstrip full-width"
        )
        filmstrip_container.set_visibility(False)

    def notify_busy() -> None:
        ui.notify(translator.text("error.job_busy"), type="warning")

    def set_scan_buttons(state: CandidatePageState) -> None:
        running = state.phase in {"starting", "cache", "scanning", "cancelling"}
        any_job = coordinator.snapshot().active
        start_button.set_enabled(not running and not any_job)
        cancel_button.set_enabled(state.phase in {"starting", "cache", "scanning"})
        force_button.set_enabled(not running and not any_job)
        previous_button.set_enabled(bool(state.candidates))
        next_button.set_enabled(bool(state.candidates))
        choose_id_button.set_enabled(bool(state.candidates))
        confirm_button.set_enabled(state.can_confirm)

    def phase_text(state: CandidatePageState) -> str:
        key = {
            "idle": "candidates.phase_idle",
            "starting": "candidates.phase_starting",
            "scanning": "candidates.phase_scanning",
            "cancelling": "candidates.phase_cancelling",
            "cancelled": "candidates.phase_cancelled",
            "completed": "candidates.phase_completed",
            "failed": "candidates.phase_failed",
            "cache": "candidates.phase_cache",
        }.get(state.phase, "candidates.phase_scanning")
        return translator.text(key)

    def candidate_style(selected: bool) -> dict[str, Any]:
        color = "#16a36a" if selected else "#dc3f4f"
        return {
            "color": color,
            "fillColor": color,
            "weight": 4 if selected else 2,
            "fillOpacity": 0.24 if selected else 0.09,
        }

    def sync_candidate_layers(state: CandidatePageState) -> None:
        nonlocal display_bounds, last_candidate_layer_signature
        signature = (
            tuple(
                (
                    candidate.flat_grid_id,
                    candidate.point_count,
                    candidate.left_x,
                    candidate.bottom_y,
                )
                for candidate in state.candidates
            ),
            state.selected_flat_grid_id,
            "candidates" in state.enabled_layers,
        )
        if signature == last_candidate_layer_signature:
            return
        last_candidate_layer_signature = signature
        display_bounds = candidate_display_bounds(
            state.candidates,
            rectangle_size=rectangle_size,
            crs=profile.target_crs,
        )
        current_ids = {item.flat_grid_id for item in display_bounds}
        for flat_grid_id in tuple(candidate_layers):
            if flat_grid_id not in current_ids:
                layer = candidate_layers.pop(flat_grid_id)
                if layer in map_element.layers:
                    map_element.remove_layer(layer)
        for item in display_bounds:
            selected = item.flat_grid_id == state.selected_flat_grid_id
            style = candidate_style(selected)
            layer = candidate_layers.get(item.flat_grid_id)
            if layer is None:
                layer = map_element.generic_layer(
                    name="rectangle",
                    args=[_leaflet_bounds(item.bounds), style],
                )
                candidate_layers[item.flat_grid_id] = layer
                if map_element.is_initialized:
                    candidate = next(
                        value
                        for value in state.candidates
                        if value.flat_grid_id == item.flat_grid_id
                    )
                    layer.run_method(
                        "bindTooltip",
                        translator.text(
                            "candidates.tooltip",
                            grid_id=candidate.flat_grid_id,
                            count=candidate.point_count,
                        ),
                        {"sticky": True},
                    )
            else:
                layer.args[1] = style
                layer.run_method("setStyle", style)
            opacity = 1.0 if "candidates" in state.enabled_layers else 0.0
            visible_style = dict(style)
            visible_style["opacity"] = opacity
            visible_style["fillOpacity"] = style["fillOpacity"] * opacity
            layer.args[1] = visible_style
            layer.run_method("setStyle", visible_style)

    def select_flat_grid_id(flat_grid_id: int | None) -> None:
        controller.select_flat_grid_id(flat_grid_id)
        refresh(controller.state)
        try_request_statistics()

    def rebuild_filmstrip(state: CandidatePageState) -> None:
        nonlocal last_filmstrip_signature
        if state.view != "filmstrip":
            return
        preview_path = (
            None
            if state.candidate_preview_asset is None
            else str(state.candidate_preview_asset.path)
        )
        signature = (
            tuple(candidate.flat_grid_id for candidate in state.candidates),
            state.selected_flat_grid_id,
            preview_path,
            current_overview_url,
        )
        if signature == last_filmstrip_signature:
            return
        last_filmstrip_signature = signature
        bounds_by_id = {item.flat_grid_id: item for item in display_bounds}
        filmstrip_container.clear()
        with filmstrip_container:
            ui.label(translator.text("candidates.filmstrip_title")).classes(
                "lte-section-title"
            )
            if not state.candidates:
                ui.label(translator.text("candidates.no_candidates")).classes(
                    "lte-page-subtitle"
                )
                return
            with ui.grid().classes("lte-filmstrip-grid full-width"):
                for index, candidate in enumerate(state.candidates):
                    display = bounds_by_id[candidate.flat_grid_id]
                    card = ui.card().classes(
                        "lte-filmstrip-card"
                        + (
                            " lte-filmstrip-card--selected"
                            if candidate.flat_grid_id == state.selected_flat_grid_id
                            else ""
                        )
                    )
                    is_selected = (
                        candidate.flat_grid_id == state.selected_flat_grid_id
                    )
                    card.props(
                        "tabindex=0 role=button "
                        f"aria-pressed={'true' if is_selected else 'false'} "
                        f"aria-current={'true' if is_selected else 'false'}"
                    ).mark(
                        f"candidate-card-{candidate.flat_grid_id}"
                    )
                    card.on(
                        "click",
                        lambda _event, candidate_id=candidate.flat_grid_id: (
                            select_flat_grid_id(candidate_id)
                        ),
                    )
                    card.on(
                        "keydown.enter",
                        lambda _event, candidate_id=candidate.flat_grid_id: (
                            select_flat_grid_id(candidate_id)
                        ),
                    )
                    card.on(
                        "keydown.space",
                        lambda _event, candidate_id=candidate.flat_grid_id: (
                            select_flat_grid_id(candidate_id)
                        ),
                    )
                    with card:
                        if (
                            candidate.flat_grid_id == state.selected_flat_grid_id
                            and state.candidate_preview_asset is not None
                        ):
                            ui.image(
                                asset_url(state.candidate_preview_asset.path)
                            ).classes("lte-filmstrip-thumbnail").mark(
                                f"candidate-thumbnail-{candidate.flat_grid_id}"
                            )
                        else:
                            ui.element("div").classes(
                                "lte-filmstrip-thumbnail"
                            ).mark(
                                f"candidate-thumbnail-{candidate.flat_grid_id}"
                            ).style(
                                _thumbnail_background_style(
                                    current_overview_url,
                                    bundle.map_bounds,
                                    display.bounds,
                                )
                            )
                        ui.label(
                            translator.text(
                                "candidates.card_title",
                                index=index + 1,
                            )
                        ).classes("lte-filmstrip-card-title")
                        ui.label(
                            translator.text(
                                "candidates.card_details",
                                grid_id=candidate.flat_grid_id,
                                count=candidate.point_count,
                            )
                        ).classes("lte-page-subtitle")

    def refresh(state: CandidatePageState) -> None:
        nonlocal last_rendered_state, last_coordinator_active
        nonlocal last_dem_asset_path, current_overview_url
        coordinator_active = coordinator.snapshot().active
        if state == last_rendered_state and coordinator_active == last_coordinator_active:
            return
        last_rendered_state = state
        last_coordinator_active = coordinator_active
        progress_phase.set_text(phase_text(state))
        progress_numbers.set_text(
            f"{state.checked_positions:,} / {state.total_positions:,}"
        )
        progress_bar.set_value(state.progress_fraction)
        found_label.set_text(
            translator.text("candidates.found", count=state.found_count)
        )
        elapsed_label.set_text(
            translator.text(
                "candidates.elapsed",
                seconds=f"{state.elapsed_seconds:.1f}",
            )
        )
        cache_label.set_text(
            translator.text("candidates.cache", status=state.cache_status)
        )
        if state.dem_style_asset is not None:
            asset_path = str(state.dem_style_asset.path)
            if asset_path != last_dem_asset_path:
                dem_layer.url = asset_url(state.dem_style_asset.path)
                dem_layer.run_method("setUrl", dem_layer.url)
                current_overview_url = dem_layer.url
                last_dem_asset_path = asset_path
        if dem_style_select.value != state.dem_style:
            dem_style_select.value = state.dem_style
        visible_error = state.error or state.dem_style_error
        if visible_error:
            error = (
                f"{state.error_code}: {state.error}"
                if state.error_code and state.error
                else visible_error
            )
            progress_error.set_text(error)
            progress_error.set_visibility(True)
        else:
            progress_error.set_visibility(False)
        set_scan_buttons(state)
        sync_candidate_layers(state)
        selected = state.selected_candidate
        if selected is None:
            selected_title.set_text(translator.text("candidates.none_selected"))
            selected_details.set_text(translator.text("candidates.select_hint"))
            selected_preview.set_visibility(False)
            stats_title.set_visibility(False)
            stats_details.set_visibility(False)
        else:
            selected_title.set_text(
                translator.text(
                    "candidates.selected_title",
                    index=(state.selected_index or 0) + 1,
                )
            )
            selected_details.set_text(
                translator.text(
                    "candidates.selected_details",
                    grid_id=selected.flat_grid_id,
                    count=selected.point_count,
                    x=f"{selected.center_x:.1f}",
                    y=f"{selected.center_y:.1f}",
                    left=f"{selected.left_x:.1f}",
                    bottom=f"{selected.bottom_y:.1f}",
                    right=f"{selected.left_x + rectangle_size:.1f}",
                    top=f"{selected.bottom_y + rectangle_size:.1f}",
                    rect_size=profile.rect_size,
                    target=profile.target_count,
                    tolerance=profile.tolerance,
                    mode=profile.scan_mode,
                )
            )
            if state.candidate_preview_asset is not None:
                selected_preview.set_source(
                    asset_url(state.candidate_preview_asset.path)
                )
                selected_preview.set_visibility(True)
            else:
                selected_preview.set_visibility(False)
            stats_title.set_visibility(True)
            stats_details.set_visibility(True)
            if state.statistics is not None:
                stats = state.statistics
                stats_details.set_text(
                    translator.text(
                        "candidates.statistics_values",
                        minimum=f"{stats.minimum:.1f}",
                        maximum=f"{stats.maximum:.1f}",
                        mean=f"{stats.mean:.1f}",
                        elevation_range=f"{stats.elevation_range:.1f}",
                        pixels=stats.valid_pixel_count,
                    )
                    + (
                        " "
                        + translator.text(
                            "candidates.preview_unavailable",
                            error=state.candidate_preview_error,
                        )
                        if state.candidate_preview_error
                        else ""
                    )
                )
            elif state.statistics_error:
                stats_details.set_text(state.statistics_error)
            elif state.statistics_job_id:
                stats_details.set_text(translator.text("candidates.statistics_loading"))
            else:
                stats_details.set_text(translator.text("candidates.statistics_waiting"))
        rebuild_filmstrip(state)

    def try_request_statistics() -> None:
        state = controller.state
        if (
            not state.scan_completed
            or state.selected_candidate is None
            or state.statistics is not None
            or state.statistics_error is not None
            or state.statistics_job_id is not None
        ):
            return
        try:
            job = controller.request_statistics()
        except JobBusyError:
            return
        if job is not None:
            timer.activate()

    def choose_relative(offset: int) -> None:
        state = controller.state
        if not state.candidates:
            return
        current = state.selected_index
        index = 0 if current is None else (current + offset) % len(state.candidates)
        controller.select_index(index)
        refresh(controller.state)
        try_request_statistics()

    def choose_direct() -> None:
        value = candidate_id_input.value
        try:
            candidate_id = int(value)
        except (TypeError, ValueError, OverflowError):
            candidate_id = None
        before = controller.state.selected_flat_grid_id
        controller.select_flat_grid_id(candidate_id)
        if controller.state.selected_flat_grid_id is None and before != candidate_id:
            ui.notify(translator.text("candidates.unknown_id"), type="warning")
        refresh(controller.state)
        try_request_statistics()

    def handle_map_click(event: Any) -> None:
        if "candidates" not in controller.state.enabled_layers:
            return
        coordinates = _event_latlng(event)
        if coordinates is None:
            return
        indices = hit_test_candidate_indices(
            display_bounds,
            latitude=coordinates[0],
            longitude=coordinates[1],
        )
        if not indices:
            return
        current = controller.state.selected_index
        if current in indices:
            index = indices[(indices.index(current) + 1) % len(indices)]
        else:
            index = indices[0]
        controller.select_index(index)
        refresh(controller.state)
        try_request_statistics()

    def switch_view(view: str) -> None:
        state = controller.set_view(view)
        map_container.set_visibility(True)
        filmstrip_container.set_visibility(view == "filmstrip")
        map_view_button.props("unelevated" if view == "map" else "outline")
        filmstrip_view_button.props(
            "unelevated" if view == "filmstrip" else "outline"
        )
        if view == "map":
            map_element.run_map_method("invalidateSize")
        refresh(state)

    def start_scan(*, force: bool = False) -> None:
        try:
            controller.start_scan(force=force)
        except JobBusyError:
            notify_busy()
            return
        except Exception as exc:
            ui.notify(str(exc), type="negative")
            return
        timer.activate()
        refresh(controller.state)

    def cancel_scan() -> None:
        if controller.cancel_scan():
            refresh(controller.state)

    def confirm_selection() -> None:
        try:
            confirmed = controller.confirm()
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        ui.notify(
            translator.text(
                "candidates.confirmed",
                grid_id=confirmed.confirmed_flat_grid_id,
            ),
            type="positive",
        )
        refresh(controller.state)
        if on_confirm is not None:
            on_confirm(confirmed)

    def toggle_simple_layer(layer_name: str, layer: Any, enabled: bool) -> None:
        controller.set_layer(layer_name, enabled)
        if layer_name == "candidates":
            refresh(controller.state)
            return
        if layer_name == "dem":
            layer.run_method(
                "setOpacity",
                controller.state.dem_opacity if enabled else 0.0,
            )
        elif layer_name == "boundary":
            layer.run_method(
                "setStyle",
                {"opacity": 1.0 if enabled else 0.0, "fillOpacity": 0.03 if enabled else 0.0},
            )
        elif layer_name == "stations":
            opacity = 1.0 if enabled else 0.0
            layer.run_method(
                ":eachLayer",
                f"(child) => {{ if (child.setOpacity) child.setOpacity({opacity}); "
                f"if (child.setStyle) child.setStyle({{opacity:{opacity},"
                f"fillOpacity:{opacity}}}); }}",
            )
        refresh(controller.state)

    async def toggle_online(enabled: bool) -> None:
        nonlocal online_layer
        if not enabled:
            if online_layer is not None and online_layer in map_element.layers:
                map_element.remove_layer(online_layer)
            online_layer = None
            controller.set_layer("online", False)
            return
        from nicegui import run

        available = await run.io_bound(online_tiles_available, online_tile_probe)
        if not available:
            online_switch.value = False
            controller.set_layer("online", False)
            ui.notify(translator.text("candidates.online_unavailable"), type="warning")
            return
        try:
            online_layer = map_element.tile_layer(
                url_template="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                options={"maxZoom": 19},
            )
        except Exception:
            online_layer = None
            online_switch.value = False
            controller.set_layer("online", False)
            ui.notify(translator.text("candidates.online_unavailable"), type="warning")
            return
        online_layer.run_method(
            ":on",
            "'tileerror'",
            "(event) => event.target.remove()",
        )
        controller.set_layer("online", True)

    def change_dem_style(value: Any) -> None:
        try:
            style = value if isinstance(value, MapStyle) else MapStyle(str(value))
        except ValueError:
            return
        if style is controller.state.dem_style:
            return
        try:
            job = controller.request_dem_style(style)
        except JobBusyError:
            dem_style_select.value = controller.state.dem_style
            notify_busy()
            return
        if job is None:
            dem_style_select.value = controller.state.dem_style
            return
        timer.activate()
        refresh(controller.state)

    def change_opacity(value: Any) -> None:
        try:
            opacity = float(value)
        except (TypeError, ValueError):
            return
        controller.set_dem_opacity(opacity)
        dem_layer.run_method(
            "setOpacity",
            opacity if "dem" in controller.state.enabled_layers else 0.0,
        )

    def tick() -> None:
        if controller.scan_job is not None:
            controller.drain_scan()
        if controller.statistics_job is not None:
            controller.drain_statistics()
        controller.drain_dem_style()
        try_request_statistics()
        refresh(controller.state)
        if (
            controller.scan_job is None
            and controller.statistics_job is None
            and controller.dem_style_job is None
        ):
            timer.deactivate()

    def coordinator_tick() -> None:
        try_request_statistics()
        refresh(controller.state)

    def bind_station_popups() -> None:
        stations_layer.run_method(
            ":eachLayer",
            "(layer) => { const p = (layer.feature || {}).properties || {}; "
            "const keys = ['cell','longitude','latitude','range','samples','created','updated']; "
            "const esc = (v) => String(v).replace(/[&<>\"']/g, "
            "(c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c])); "
            "const rows = keys.filter((k) => p[k] !== null && p[k] !== undefined)"
            ".map((k) => `<b>${esc(k)}</b>: ${esc(p[k])}`); "
            "if (rows.length) layer.bindPopup(rows.join('<br>')); }",
        )

    def bind_candidate_tooltips() -> None:
        candidates_by_id = {
            candidate.flat_grid_id: candidate for candidate in controller.state.candidates
        }
        for flat_grid_id, layer in candidate_layers.items():
            candidate = candidates_by_id.get(flat_grid_id)
            if candidate is None:
                continue
            layer.run_method(
                "bindTooltip",
                translator.text(
                    "candidates.tooltip",
                    grid_id=flat_grid_id,
                    count=candidate.point_count,
                ),
                {"sticky": True},
            )

    def handle_layer_removed(event: Any) -> None:
        nonlocal online_layer
        payload = getattr(event, "args", {})
        if (
            online_layer is None
            or not isinstance(payload, dict)
            or payload.get("id") != online_layer.id
        ):
            return
        if online_layer in map_element.layers:
            map_element.layers.remove(online_layer)
        online_layer = None
        controller.set_layer("online", False)
        online_switch.value = False
        ui.notify(translator.text("candidates.online_unavailable"), type="warning")

    map_element.on("map-click", handle_map_click, args=["latlng"])
    map_element.on("init", bind_station_popups)
    map_element.on("init", bind_candidate_tooltips)
    map_element.on("map-layerremove", handle_layer_removed)
    map_view_button.on("click", lambda: switch_view("map"))
    filmstrip_view_button.on("click", lambda: switch_view("filmstrip"))
    start_button.on("click", lambda: start_scan(force=False))
    cancel_button.on("click", cancel_scan)
    force_button.on("click", lambda: start_scan(force=True))
    previous_button.on("click", lambda: choose_relative(-1))
    next_button.on("click", lambda: choose_relative(1))
    choose_id_button.on("click", choose_direct)
    confirm_button.on("click", confirm_selection)
    dem_switch.on_value_change(
        lambda event: toggle_simple_layer("dem", dem_layer, bool(event.value))
    )
    boundary_switch.on_value_change(
        lambda event: toggle_simple_layer(
            "boundary", boundary_layer, bool(event.value)
        )
    )
    stations_switch.on_value_change(
        lambda event: toggle_simple_layer(
            "stations", stations_layer, bool(event.value)
        )
    )
    candidates_switch.on_value_change(
        lambda event: toggle_simple_layer(
            "candidates", None, bool(event.value)
        )
    )
    online_switch.on_value_change(lambda event: toggle_online(bool(event.value)))
    dem_opacity.on_value_change(lambda event: change_opacity(event.value))
    dem_style_select.on_value_change(lambda event: change_dem_style(event.value))

    timer = ui.timer(0.15, tick, active=True, immediate=True)
    coordinator_timer = ui.timer(
        0.5,
        coordinator_tick,
        active=True,
        immediate=True,
    )

    def cleanup() -> None:
        timer.deactivate()
        coordinator_timer.deactivate()
        controller.close()

    ui.context.client.on_delete(cleanup)
    switch_view(controller.state.view)
    if auto_start and session.scan_result is None:
        start_scan()
    else:
        refresh(controller.state)
    return CandidatePageView(
        controller=controller,
        map_element=map_element,
        timer=timer,
        coordinator_timer=coordinator_timer,
        map_container=map_container,
        filmstrip_container=filmstrip_container,
    )


__all__ = [
    "CandidateDisplayBounds",
    "CandidateExplorerController",
    "CandidateInspection",
    "CandidateMapBundle",
    "CandidatePageView",
    "CandidatePageState",
    "CandidateProgressEvent",
    "CandidateSession",
    "CandidateSessionRegistry",
    "CandidateStatisticsEvent",
    "build_candidate_map_bundle",
    "build_candidate_overlay",
    "candidate_display_bounds",
    "candidate_thumbnail_data_url",
    "default_online_tile_probe",
    "hit_test_candidate_indices",
    "online_tiles_available",
    "render_candidate_page",
    "render_candidate_unavailable",
    "reduce_progress",
    "reduce_statistics",
]
