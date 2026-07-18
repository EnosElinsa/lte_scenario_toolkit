"""Shared, package-local assets for every NiceGUI entry point."""

from __future__ import annotations

from importlib import resources
from typing import Any

from .leaflet_assets import register_station_dots_resource


def packaged_gui_css() -> str:
    """Return the packaged GUI stylesheet as UTF-8 text."""

    return (
        resources.files("lte_scenario_toolkit.gui")
        .joinpath("assets", "app.css")
        .read_text(encoding="utf-8")
    )


def install_gui_assets(app: Any, ui: Any) -> str:
    """Install shared CSS and return the local station-layer resource URL."""

    ui.add_css(packaged_gui_css(), shared=True)
    return register_station_dots_resource(app)


__all__ = ["install_gui_assets", "packaged_gui_css"]
