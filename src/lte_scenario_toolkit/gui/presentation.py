"""Semantic, localizable presentation for GUI machine values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

StatusTone = Literal[
    "neutral",
    "info",
    "warning",
    "success",
    "danger",
    "active",
]


@dataclass(frozen=True, slots=True)
class PresentationSpec:
    """Translation and visual tone for one machine value."""

    label_key: str
    tone: StatusTone = "neutral"
    description_key: str | None = None


_UNKNOWN: Final = PresentationSpec("status.unknown")

_READINESS: Final = {
    "ready": PresentationSpec(
        "status.ready",
        "success",
        "readiness.ready.description",
    ),
    "boundary-ready": PresentationSpec(
        "status.boundary_ready",
        "warning",
        "readiness.boundary_ready.description",
    ),
    "dem-pending": PresentationSpec(
        "status.dem_pending",
        "warning",
        "readiness.dem_pending.description",
    ),
    "invalid": PresentationSpec(
        "status.invalid",
        "danger",
        "readiness.invalid.description",
    ),
}

_CACHE: Final = {
    "none": PresentationSpec("cache.none"),
    "hit": PresentationSpec("cache.hit", "success"),
    "miss": PresentationSpec("cache.miss", "info"),
}

_SCAN_MODES: Final = {
    "fast": PresentationSpec("scan.fast"),
    "complete": PresentationSpec("scan.complete"),
}

_ARTIFACT_LABELS: Final = {
    "csv": PresentationSpec(
        "generate.artifact.csv",
        description_key="generate.artifact.csv.description",
    ),
    "preview_png": PresentationSpec(
        "generate.artifact.preview_png",
        description_key="generate.artifact.preview_png.description",
    ),
    "terrain_png": PresentationSpec(
        "generate.artifact.terrain_png",
        description_key="generate.artifact.terrain_png.description",
    ),
    "terrain_eps": PresentationSpec(
        "generate.artifact.terrain_eps",
        description_key="generate.artifact.terrain_eps.description",
    ),
    "terrain_html": PresentationSpec(
        "generate.artifact.terrain_html",
        description_key="generate.artifact.terrain_html.description",
    ),
}

_ARTIFACT_STATES: Final = {
    "not-requested": PresentationSpec("status.not_requested"),
    "pending": PresentationSpec("status.pending", "active"),
    "published": PresentationSpec("status.published", "success"),
    "failed": PresentationSpec("status.failed", "danger"),
}

_RUN_STATES: Final = {
    "completed": PresentationSpec("status.completed", "success"),
    "partial": PresentationSpec("status.partial", "warning"),
}

_JOB_KINDS: Final = {
    "validation.full_checksum": PresentationSpec(
        "job.kind.full_checksum",
        "active",
    ),
    "generate": PresentationSpec("job.kind.generate", "active"),
    "selection.scan": PresentationSpec("job.kind.selection_scan", "active"),
    "candidate.dem_style": PresentationSpec("job.kind.dem_style", "active"),
    "candidate.statistics": PresentationSpec("job.kind.statistics", "active"),
    "figure-source": PresentationSpec("job.kind.figure_source", "active"),
    "figure-preview": PresentationSpec("job.kind.figure_preview", "active"),
    "figure-export": PresentationSpec("job.kind.figure_export", "active"),
}


def _present(
    value: object,
    mapping: dict[str, PresentationSpec],
) -> PresentationSpec:
    if type(value) is not str:
        return _UNKNOWN
    return mapping.get(value, _UNKNOWN)


def readiness_presentation(value: object) -> PresentationSpec:
    """Present a catalog readiness value without leaking its machine token."""

    return _present(value, _READINESS)


def cache_presentation(value: object) -> PresentationSpec:
    """Present candidate scan cache provenance."""

    return _present(value, _CACHE)


def scan_mode_presentation(value: object) -> PresentationSpec:
    """Present a candidate scan mode."""

    return _present(value, _SCAN_MODES)


def artifact_label_presentation(value: object) -> PresentationSpec:
    """Present a generated artifact kind."""

    return _present(value, _ARTIFACT_LABELS)


def artifact_state_presentation(value: object) -> PresentationSpec:
    """Present the publication state of one generated artifact."""

    return _present(value, _ARTIFACT_STATES)


def run_state_presentation(value: object) -> PresentationSpec:
    """Present a published run state."""

    return _present(value, _RUN_STATES)


def job_kind_presentation(value: object) -> PresentationSpec:
    """Present the operation owned by the process-local job coordinator."""

    return _present(value, _JOB_KINDS)


__all__ = [
    "PresentationSpec",
    "StatusTone",
    "artifact_label_presentation",
    "artifact_state_presentation",
    "cache_presentation",
    "job_kind_presentation",
    "readiness_presentation",
    "run_state_presentation",
    "scan_mode_presentation",
]
