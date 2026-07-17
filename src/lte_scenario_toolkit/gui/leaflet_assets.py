from pathlib import Path
from typing import Any

STATION_DOTS_URL = "/_lte_gui/assets/station-dots.js"


def register_station_dots_resource(app: Any) -> str:
    """Expose the packaged Leaflet station layer at a stable local URL."""
    asset = Path(__file__).with_name("assets") / "station_dots.js"
    app.remove_route(STATION_DOTS_URL)
    return app.add_static_file(
        local_file=asset,
        url_path=STATION_DOTS_URL,
        strict=True,
        max_cache_age=3600,
    )
